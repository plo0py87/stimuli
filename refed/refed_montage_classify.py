"""3-class valence/arousal CLASSIFICATION on REFED with DGCNN, four montages.

Companion to refed_montage_regress.py. That script predicts the continuous
joystick trajectory and reports CCC; thresholding its output into 3 classes
gives a pessimistic accuracy, because an MSE-trained regressor shrinks its
predictions toward the mean and so rarely crosses the +/-40 margin. If you want
an honest "accuracy" number to quote, it has to come from a model actually
trained to classify -- that is what this script does.

Same montages, same by-video folds, same causal DE features, same two training
modes. The only differences are the target (3 classes per dimension via the
+/-40 joystick margin) and the loss (cross-entropy).

DGCNN emits 6 logits, reshaped to (batch, 2 dims, 3 classes), with one
cross-entropy per dimension summed -- valence and arousal share a trunk exactly
as they do in the regression version.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" refed_montage_classify.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "emotion_model"))

from models.DGCNN import DGCNN  # noqa: E402
from utils.utils import setup_seed  # noqa: E402
from refed_montage_regress import (  # noqa: E402
    MONTAGES, EXCLUDE_FROM_WHOLE, FOLDS, FEATURE_SETS, DIMS,
    to_class, normalize_per_subject, EPS,
)

CLASS_NAMES = ["negative", "neutral", "positive"]


def macro_recall(y, p):
    rec = [(p[y == c] == c).mean() for c in range(3) if (y == c).any()]
    return float(np.mean(rec)) if rec else float("nan")


def fit_predict(Xtr, Ctr, Xva, Cva, Xte, device, a, seed):
    setup_seed(seed)
    model = DGCNN(num_electrodes=Xtr.shape[1], in_channels=Xtr.shape[2],
                  num_classes=6, k=2, relu_is=1, layers=[64],
                  dropout_rate=a.dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    crit = nn.CrossEntropyLoss()

    def loss_of(logits, targets):
        lg = logits.view(-1, 2, 3)
        return crit(lg[:, 0], targets[:, 0]) + crit(lg[:, 1], targets[:, 1])

    ds = torch.utils.data.TensorDataset(torch.Tensor(Xtr), torch.LongTensor(Ctr))
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True)
    Xva_t = torch.Tensor(Xva).to(device)
    Cva_t = torch.LongTensor(Cva).to(device)

    best, best_state = float("inf"), None
    for _ in range(a.epochs):
        model.train()
        for xb, cb in loader:
            xb, cb = xb.to(device), cb.to(device)
            opt.zero_grad()
            loss_of(model(xb), cb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_of(model(Xva_t), Cva_t))
        if vl < best:
            best = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xte), 4096):
            lg = model(torch.Tensor(Xte[i:i + 4096]).to(device)).view(-1, 2, 3)
            out.append(lg.argmax(dim=2).cpu().numpy())
    return np.concatenate(out)


def run_pooled(Xm, C, subject, video, device, a):
    pred = np.zeros_like(C)
    for f_i, test_vids in enumerate(FOLDS):
        te = np.isin(video, test_vids)
        rest = [v for v in np.unique(video) if v not in test_vids]
        va, tr = np.isin(video, rest[:2]), np.isin(video, rest[2:])
        Xn = normalize_per_subject(Xm, subject, tr)
        pred[te] = fit_predict(Xn[tr], C[tr], Xn[va], C[va], Xn[te],
                               device, a, a.seed + f_i)
    return pred


def run_per_subject(Xm, C, subject, video, device, a):
    pred = np.zeros_like(C)
    for sub in np.unique(subject):
        ms = subject == sub
        Xs, Cs, vs = Xm[ms], C[ms], video[ms]
        ps = np.zeros_like(Cs)
        for f_i, test_vids in enumerate(FOLDS):
            te = np.isin(vs, test_vids)
            rest = [v for v in np.unique(vs) if v not in test_vids]
            if te.sum() == 0 or len(rest) <= 2:
                continue
            va, tr = np.isin(vs, rest[:2]), np.isin(vs, rest[2:])
            flat = Xs[tr].reshape(tr.sum(), -1)
            mu, sd = flat.mean(axis=0), flat.std(axis=0) + EPS

            def norm(x):
                sh = x.shape
                return ((x.reshape(len(x), -1) - mu) / sd).reshape(sh)

            ps[te] = fit_predict(norm(Xs[tr]), Cs[tr], norm(Xs[va]), Cs[va],
                                 norm(Xs[te]), device, a, a.seed + f_i)
        pred[ms] = ps
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(HERE / "refed_features_all64.npz"))
    ap.add_argument("--feature-set", choices=list(FEATURE_SETS), default="de")
    ap.add_argument("--modes", nargs="+", default=["pooled", "per_subject"])
    ap.add_argument("--montages", nargs="+", default=list(MONTAGES))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(HERE / "montage_results_cls.json"))
    ap.add_argument("--save-predictions",
                    default=str(HERE / "montage_predictions_cls.npz"))
    a = ap.parse_args()

    z = np.load(a.features, allow_pickle=True)
    X, Y = z["X"], z["Y"]
    subject, video, second = z["subject"], z["video"], z["second"]
    channels = [str(c) for c in z["channels"]]

    C = np.stack([to_class(Y[:, 0]), to_class(Y[:, 1])], axis=1)
    montages = dict(MONTAGES)
    montages["whole"] = [c for c in channels if c not in EXCLUDE_FROM_WHOLE]
    ch_idx = {n: [channels.index(c) for c in montages[n]] for n in a.montages}
    fsl = FEATURE_SETS[a.feature_set]

    for i, d in enumerate(DIMS):
        cnt = np.bincount(C[:, i], minlength=3)
        print(f"{d:8s} class balance: " +
              "  ".join(f"{n}={c} ({c/cnt.sum()*100:.1f}%)"
                        for n, c in zip(CLASS_NAMES, cnt)) +
              f"   majority baseline {cnt.max()/cnt.sum():.3f}")
    print()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    results, saved = {}, {}

    for mode in a.modes:
        results[mode] = {}
        runner = run_pooled if mode == "pooled" else run_per_subject
        for name, idx in ch_idx.items():
            pred = runner(X[:, idx, fsl], C, subject, video, device, a)
            saved[f"pred_{mode}_{name}"] = pred

            per_sub = {d: {"acc": [], "macro": []} for d in DIMS}
            for s in np.unique(subject):
                ms = subject == s
                for i, d in enumerate(DIMS):
                    per_sub[d]["acc"].append(float((pred[ms][:, i] == C[ms][:, i]).mean()))
                    per_sub[d]["macro"].append(macro_recall(C[ms][:, i], pred[ms][:, i]))
            confusion = {d: np.zeros((3, 3), int).tolist() for d in DIMS}
            for i, d in enumerate(DIMS):
                cm = np.zeros((3, 3), int)
                for t, p in zip(C[:, i], pred[:, i]):
                    cm[t, p] += 1
                confusion[d] = cm.tolist()

            results[mode][name] = {"n_channels": len(idx), "per_subject": per_sub,
                                   "confusion": confusion}
            print(f"[{mode:11s}] {name:8s} ({len(idx):2d}ch)  " + "  ".join(
                f"{d[0].upper()}: acc={np.mean(per_sub[d]['acc']):.3f} "
                f"macro={np.mean(per_sub[d]['macro']):.3f}" for d in DIMS), flush=True)

            with open(a.out, "w") as f:
                json.dump({"args": vars(a), "results": results,
                           "montage_channels": {k: montages[k] for k in ch_idx}},
                          f, indent=2)
        print()

    np.savez_compressed(a.save_predictions, C=C, subject=subject, video=video,
                        second=second, **saved)
    print("Saved ->", a.out, "and", a.save_predictions)


if __name__ == "__main__":
    main()
