"""Cross-subject DGCNN training on REFED, comparing normalization schemes.

The question being tested: does normalizing each recording session against
itself before feeding the model improve generalization to *unseen* sessions?

REFED records exactly one session per subject, so "cross-session" here is
literally "cross-subject" -- held-out subjects are never seen in any form
during training. Folds are subject-disjoint and every subject is tested once.

Three normalization conditions:

  global    mean/std computed once over the TRAIN-fold feature pool and reused
            everywhere. Realtime-legal. This is the current default in the
            quick20 pipeline.

  baseline  mean/std computed per subject from that subject's OWN resting
            baseline (the 5 s eyes-open segment before each clip). Realtime-
            legal: at inference time you record a baseline first, then predict.
            This is the condition under test.

  session   mean/std computed per subject over that subject's own *video*
            features. NOT realtime-legal -- it needs the whole session up
            front, i.e. it looks into the future. Included only as a reference
            upper bound; do not quote it as a deployable number.

Model selection uses a validation split carved out of the TRAIN subjects, so
the test fold is never used to pick an epoch.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" refed_train_dgcnn.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = Path(__file__).resolve().parent
EMOTION_MODEL_DIR = HERE.parent / "emotion_model"
sys.path.insert(0, str(EMOTION_MODEL_DIR))

from models.DGCNN import DGCNN  # noqa: E402
from utils.utils import setup_seed  # noqa: E402

IDX_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}
CONDITIONS = ["global", "baseline", "session"]
EPS = 1e-8


def per_subject_stats(X, subject, subjects):
    """{subject: (mean, std)} over flattened features."""
    stats = {}
    n_feat = X.shape[1] * X.shape[2]
    for s in subjects:
        flat = X[subject == s].reshape(-1, n_feat)
        stats[s] = (flat.mean(axis=0), flat.std(axis=0) + EPS)
    return stats


def apply_stats(X, subject, stats, fallback):
    """Normalize each subject's rows with that subject's own (mean, std)."""
    shape = X.shape
    flat = X.reshape(-1, shape[1] * shape[2]).copy()
    for s in np.unique(subject):
        mean, std = stats.get(s, fallback)
        m = subject == s
        flat[m] = (flat[m] - mean) / std
    return flat.reshape(shape)


def normalize(condition, Xtr, str_, Xte, ste, Xb, sb):
    """Return (Xtr_norm, Xte_norm) for the requested condition."""
    n_feat = Xtr.shape[1] * Xtr.shape[2]
    pooled = Xtr.reshape(-1, n_feat)
    global_stats = (pooled.mean(axis=0), pooled.std(axis=0) + EPS)

    if condition == "global":
        def f(X):
            shape = X.shape
            flat = X.reshape(-1, n_feat)
            return ((flat - global_stats[0]) / global_stats[1]).reshape(shape)
        return f(Xtr), f(Xte)

    if condition == "baseline":
        all_subjects = np.unique(np.concatenate([str_, ste]))
        stats = per_subject_stats(Xb, sb, [s for s in all_subjects if (sb == s).any()])
        return (apply_stats(Xtr, str_, stats, global_stats),
                apply_stats(Xte, ste, stats, global_stats))

    if condition == "session":
        stats = per_subject_stats(
            np.concatenate([Xtr, Xte]), np.concatenate([str_, ste]),
            np.unique(np.concatenate([str_, ste])))
        return (apply_stats(Xtr, str_, stats, global_stats),
                apply_stats(Xte, ste, stats, global_stats))

    raise ValueError(condition)


def evaluate(model, X, y, device, batch=4096):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            logits = model(torch.Tensor(X[i:i + batch]).to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    pred = np.concatenate(preds)
    acc = float((pred == y).mean())
    per_class = {}
    for cls in range(3):
        m = y == cls
        per_class[IDX_TO_LABEL[cls]] = float((pred[m] == y[m]).mean()) if m.any() else float("nan")
    macro = float(np.nanmean(list(per_class.values())))
    return acc, macro, per_class, pred


def run_fold(Xtr, ytr, Xva, yva, Xte, yte, device, epochs, lr, batch_size, seed):
    setup_seed(seed)
    model = DGCNN(num_electrodes=Xtr.shape[1], in_channels=Xtr.shape[2], num_classes=3,
                  k=2, relu_is=1, layers=[64], dropout_rate=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ds = torch.utils.data.TensorDataset(torch.Tensor(Xtr), torch.LongTensor(ytr))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    best_val, best_state, best_epoch = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        _, val_macro, _, _ = evaluate(model, Xva, yva, device)
        if val_macro > best_val:
            best_val = val_macro
            best_epoch = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    acc, macro, per_class, pred = evaluate(model, Xte, yte, device)
    return acc, macro, per_class, best_epoch, best_val, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(HERE / "refed_features.npz"))
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--val-subjects", type=int, default=4,
                    help="subjects held out of TRAIN for epoch selection")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--out", default=str(HERE / "refed_results.json"))
    args = ap.parse_args()

    z = np.load(args.features)
    X, y, subject = z["X"], z["y"], z["subject"]
    Xb, sb = z["X_baseline"], z["subject_baseline"]
    subjects = np.unique(subject)

    print(f"Features {X.shape}, {len(subjects)} subjects, margin=+/-{int(z['margin'])}")
    counts = np.bincount(y, minlength=3)
    print("Class balance:", {IDX_TO_LABEL[i]: f"{c} ({c/counts.sum()*100:.1f}%)"
                             for i, c in enumerate(counts)})
    print(f"Majority-class baseline: {counts.max()/counts.sum():.4f}\n")

    rng = np.random.RandomState(args.seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, args.folds)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {c: [] for c in args.conditions}
    per_subject = {c: {} for c in args.conditions}

    for f_i, test_subs in enumerate(folds):
        rest = np.array([s for s in subjects if s not in test_subs])
        fold_rng = np.random.RandomState(args.seed + f_i)
        rest_shuf = rest.copy()
        fold_rng.shuffle(rest_shuf)
        val_subs = rest_shuf[:args.val_subjects]
        train_subs = rest_shuf[args.val_subjects:]

        m_tr = np.isin(subject, train_subs)
        m_va = np.isin(subject, val_subs)
        m_te = np.isin(subject, test_subs)

        print(f"=== Fold {f_i + 1}/{args.folds} ===")
        print(f"  train {len(train_subs)} subs ({m_tr.sum()} samples) | "
              f"val {len(val_subs)} ({m_va.sum()}) | "
              f"test {len(test_subs)} ({m_te.sum()}) -> {sorted(test_subs.tolist())}")

        for cond in args.conditions:
            # Normalization stats must come from TRAIN ONLY for `global`; the
            # per-subject conditions use each subject's own data by definition.
            Xfit, Xeval = normalize(cond, X[m_tr], subject[m_tr],
                                    X[m_va | m_te], subject[m_va | m_te], Xb, sb)
            eval_subject = subject[m_va | m_te]
            va_mask = np.isin(eval_subject, val_subs)
            te_mask = np.isin(eval_subject, test_subs)
            yeval = y[m_va | m_te]

            acc, macro, per_class, best_ep, best_val, pred = run_fold(
                Xfit, y[m_tr], Xeval[va_mask], yeval[va_mask],
                Xeval[te_mask], yeval[te_mask],
                device, args.epochs, args.lr, args.batch_size, args.seed + f_i)

            # Per-subject accuracy: every subject is tested exactly once across
            # folds, so these pair up 1:1 between conditions for a paired test.
            te_subj = eval_subject[te_mask]
            yte_ = yeval[te_mask]
            for s in np.unique(te_subj):
                m = te_subj == s
                ys, ps = yte_[m], pred[m]
                recalls = [float((ps[ys == c] == c).mean())
                           for c in range(3) if (ys == c).any()]
                per_subject[cond][int(s)] = {
                    "acc": float((ps == ys).mean()),
                    "macro": float(np.mean(recalls)),
                }

            results[cond].append({
                "fold": f_i, "acc": acc, "macro": macro, "per_class": per_class,
                "best_epoch": best_ep, "val_macro": best_val,
                "test_subjects": sorted(int(s) for s in test_subs),
            })
            pc = " ".join(f"{k[:3]}={v:.3f}" for k, v in per_class.items())
            print(f"  {cond:9s} acc={acc:.4f} macro={macro:.4f}  [{pc}]  "
                  f"(ep{best_ep}, val={best_val:.3f})")
        print()

    print("=" * 72)
    print(f"{'condition':10s} {'fold acc':>18s} {'fold macro':>18s} {'per-subj acc':>18s}")
    print("=" * 72)
    for cond in args.conditions:
        accs = np.array([r["acc"] for r in results[cond]])
        macros = np.array([r["macro"] for r in results[cond]])
        subj_accs = np.array([per_subject[cond][s]["acc"] for s in sorted(per_subject[cond])])
        note = "" if cond != "session" else "   <- NOT realtime-legal"
        print(f"{cond:10s} {accs.mean():.4f} +/- {accs.std():.4f}  "
              f"{macros.mean():.4f} +/- {macros.std():.4f}  "
              f"{subj_accs.mean():.4f} +/- {subj_accs.std():.4f}{note}")

    # Paired comparison across all 32 subjects (each subject appears in exactly
    # one test fold, so the two conditions are paired sample-by-sample).
    # Accuracy is near-useless here -- the majority class is ~46% and a model
    # that collapses to it scores that without learning anything -- so macro
    # (mean per-class recall) is the metric to read.
    if "global" in args.conditions:
        from scipy.stats import wilcoxon
        common = sorted(per_subject["global"])
        for metric in ("acc", "macro"):
            base = np.array([per_subject["global"][s][metric] for s in common])
            print(f"\nPaired per-subject {metric} vs `global` (n={len(common)} subjects):")
            for cond in args.conditions:
                if cond == "global":
                    continue
                other = np.array([per_subject[cond][s][metric] for s in common])
                diff = other - base
                try:
                    _, p = wilcoxon(other, base)
                    ptxt = f"p={p:.4g}"
                except ValueError:
                    ptxt = "p=n/a"
                print(f"  {cond:9s} mean diff {diff.mean():+.4f}  "
                      f"(better on {int((diff > 0).sum())}/{len(diff)} subjects, {ptxt})")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results,
                   "per_subject": per_subject}, f, indent=2)
    print("\nSaved ->", args.out)


if __name__ == "__main__":
    main()
