"""Within-subject continuous valence/arousal regression on REFED with LibEER's
DGCNN, comparing four electrode montages.

Question: how much continuous-emotion information survives as the electrode set
shrinks from a full cap to what a pair of glasses could carry?

Montages: whole 62ch, quick20 19ch, headtop 21ch, glasses 11ch. Features are
extracted once for all 64 channels by refed_extract_all.py; each montage is a
slice of the channel axis, which is exactly equivalent to having extracted only
that subset (every feature op is per-channel independent).

Features: 5-band causal differential entropy, matching the DGCNN paper and the
SEED/LibEER benchmark convention. The extraction file also carries a 5-band
Welch PSD in dims 5-9; `--feature-set de_psd` uses both, as the quick20
pipeline does, but DE-only is the default here.

Protocol -- within-subject, split BY VIDEO, never by second. Adjacent seconds
are near-identical after Kalman smoothing, so a per-second split leaks badly
(the quick20 pipeline measured 83.1% vs 76.3% for exactly this mistake).
REFED's 15 clips are 5 emotion targets x 3 clips, so 3 folds each hold out one
clip per target -- every fold is emotion-balanced.

Two training modes, because they answer different questions:

  per_subject  One DGCNN per subject, the strict SEED subject-dependent
               convention. Only ~800 training seconds per model against a
               network whose fc layer alone is ~1M parameters, so it
               underperforms even ridge regression -- reported for
               completeness and to justify the pooled mode.

  pooled       One DGCNN per montage per fold, trained on every subject's
               training clips and tested on held-out clips. Each test subject
               is still represented in training, so this remains
               within-subject, but with ~32x the data.

Metrics: CCC (the standard continuous-affect metric), PCC, RMSE, plus a 3-class
accuracy from thresholding predictions and truth at the same +/-40 joystick
margin, so there is a comparable "accuracy" number.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" refed_montage_regress.py
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
sys.path.insert(0, str(HERE.parent / "emotion_model"))

from models.DGCNN import DGCNN  # noqa: E402
from utils.utils import setup_seed  # noqa: E402

MONTAGES = {
    "whole": None,  # all channels except the mastoid references -> SEED's 62
    "quick20": ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ',
                'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2'],
    "headtop": ['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',
                'C3', 'C1', 'CZ', 'C2', 'C4',
                'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
                'P3', 'P1', 'PZ', 'P2', 'P4'],
    "glasses": ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
                'F7', 'F8', 'FT7', 'FT8', 'T7', 'T8'],
    # How far back can the arms usefully reach? Each variant adds one pair.
    # TP7/TP8 sit just behind the ear, where a glasses arm tip naturally rests.
    "glasses_ear": ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
                    'F7', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8'],
    # M1/M2 are the mastoid bumps -- reachable by a longer arm tip, but also the
    # site most exposed to neck EMG, and REFED's nominal reference channels.
    "glasses_mastoid": ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
                        'F7', 'F8', 'FT7', 'FT8', 'T7', 'T8',
                        'TP7', 'TP8', 'M1', 'M2'],
    # P7/P8 are well onto the back of the head -- past what a glasses frame can
    # plausibly hold, included as an upper bound on what "going backward" buys.
    "glasses_wide": ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
                     'F7', 'F8', 'FT7', 'FT8', 'T7', 'T8',
                     'TP7', 'TP8', 'P7', 'P8'],
    # A VR headset is a full ring, not a frame: brow pad in front, arms over the
    # ears, rear strap across the occiput. That reaches occipital sites a
    # glasses frame never could.
    "vr_ring": ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4',          # brow pad
                'F7', 'F8', 'FT7', 'FT8', 'T7', 'T8',        # temples / over ears
                'TP7', 'TP8', 'P7', 'P8',                    # behind the ears
                'PO7', 'PO8', 'O1', 'OZ', 'O2'],             # rear strap
    # The strap often sits low enough to reach the cerebellar sites too.
    "vr_ring_low": ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
                    'F7', 'F8', 'FT7', 'FT8', 'T7', 'T8',
                    'TP7', 'TP8', 'P7', 'P8',
                    'PO7', 'PO8', 'O1', 'OZ', 'O2', 'CB1', 'CB2'],
}
EXCLUDE_FROM_WHOLE = ['M1', 'M2']

TARGET_GROUPS = {
    "MVMA": [1, 10, 12], "LVLA": [2, 6, 13], "LVHA": [3, 9, 11],
    "HVHA": [4, 7, 15], "HVLA": [5, 8, 14],
}
FOLDS = [[g[i] for g in TARGET_GROUPS.values()] for i in range(3)]

FEATURE_SETS = {"de": slice(0, 5), "de_psd": slice(0, 10)}
DIMS = ["valence", "arousal"]
CLASS_MARGIN = 40 / 127.0
EPS = 1e-8


def ccc(y_true, y_pred):
    """Concordance correlation coefficient: penalizes weak correlation AND
    shifted/rescaled predictions, unlike plain Pearson."""
    vt, vp = y_true.var(), y_pred.var()
    mt, mp = y_true.mean(), y_pred.mean()
    cov = ((y_true - mt) * (y_pred - mp)).mean()
    denom = vt + vp + (mt - mp) ** 2
    return float(2 * cov / denom) if denom > EPS else 0.0


def pcc(y_true, y_pred):
    if y_true.std() < EPS or y_pred.std() < EPS:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def to_class(v):
    out = np.ones_like(v, dtype=np.int64)
    out[v > CLASS_MARGIN] = 2
    out[v < -CLASS_MARGIN] = 0
    return out


def metrics(y_true, y_pred):
    return {d: {"ccc": ccc(y_true[:, i], y_pred[:, i]),
                "pcc": pcc(y_true[:, i], y_pred[:, i]),
                "rmse": float(np.sqrt(((y_true[:, i] - y_pred[:, i]) ** 2).mean())),
                "acc3": float((to_class(y_true[:, i]) == to_class(y_pred[:, i])).mean())}
            for i, d in enumerate(DIMS)}


def normalize_per_subject(Xm, subject, train_mask):
    """Z-score each subject against its OWN training clips (no leakage)."""
    Xn = np.empty_like(Xm)
    for s in np.unique(subject):
        ms = subject == s
        sel = ms & train_mask
        flat = Xm[sel].reshape(sel.sum(), -1)
        mu, sd = flat.mean(axis=0), flat.std(axis=0) + EPS
        sh = Xm[ms].shape
        Xn[ms] = ((Xm[ms].reshape(ms.sum(), -1) - mu) / sd).reshape(sh)
    return Xn


def fit_predict(Xtr, Ytr, Xva, Yva, Xte, device, epochs, lr, batch_size,
                dropout, weight_decay, seed):
    setup_seed(seed)
    model = DGCNN(num_electrodes=Xtr.shape[1], in_channels=Xtr.shape[2],
                  num_classes=2, k=2, relu_is=1, layers=[64],
                  dropout_rate=dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(torch.Tensor(Xtr), torch.Tensor(Ytr))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    Xva_t, Yva_t = torch.Tensor(Xva).to(device), torch.Tensor(Yva).to(device)

    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(crit(model(Xva_t), Yva_t))
        if vl < best:
            best = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xte), 4096):
            out.append(model(torch.Tensor(Xte[i:i + 4096]).to(device)).cpu().numpy())
    return np.concatenate(out)


def run_pooled(Xm, Y, subject, video, device, a):
    pred = np.zeros_like(Y)
    for f_i, test_vids in enumerate(FOLDS):
        te = np.isin(video, test_vids)
        rest = [v for v in np.unique(video) if v not in test_vids]
        va = np.isin(video, rest[:2])
        tr = np.isin(video, rest[2:])
        Xn = normalize_per_subject(Xm, subject, tr)
        pred[te] = fit_predict(Xn[tr], Y[tr], Xn[va], Y[va], Xn[te], device,
                               a.epochs, a.lr, a.batch_size, a.dropout,
                               a.weight_decay, a.seed + f_i)
    return pred


def run_per_subject(Xm, Y, subject, video, device, a):
    pred = np.zeros_like(Y)
    for sub in np.unique(subject):
        ms = subject == sub
        Xs, Ys, vs = Xm[ms], Y[ms], video[ms]
        ps = np.zeros_like(Ys)
        for f_i, test_vids in enumerate(FOLDS):
            te = np.isin(vs, test_vids)
            rest = [v for v in np.unique(vs) if v not in test_vids]
            if te.sum() == 0 or len(rest) <= 2:
                continue
            va = np.isin(vs, rest[:2])
            tr = np.isin(vs, rest[2:])
            flat = Xs[tr].reshape(tr.sum(), -1)
            mu, sd = flat.mean(axis=0), flat.std(axis=0) + EPS

            def norm(x):
                sh = x.shape
                return ((x.reshape(len(x), -1) - mu) / sd).reshape(sh)

            ps[te] = fit_predict(norm(Xs[tr]), Ys[tr], norm(Xs[va]), Ys[va],
                                 norm(Xs[te]), device, a.epochs, a.lr,
                                 a.batch_size, a.dropout, a.weight_decay,
                                 a.seed + f_i)
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
    ap.add_argument("--out", default=str(HERE / "montage_results.json"))
    ap.add_argument("--save-predictions", default=str(HERE / "montage_predictions.npz"))
    a = ap.parse_args()

    z = np.load(a.features, allow_pickle=True)
    X, Y = z["X"], z["Y"]
    subject, video, second = z["subject"], z["video"], z["second"]
    channels = [str(c) for c in z["channels"]]

    montages = dict(MONTAGES)
    montages["whole"] = [c for c in channels if c not in EXCLUDE_FROM_WHOLE]
    ch_idx = {n: [channels.index(c) for c in montages[n]] for n in a.montages}
    fsl = FEATURE_SETS[a.feature_set]

    print(f"Features {X.shape} -> feature set '{a.feature_set}' "
          f"({fsl.stop - fsl.start} dims/channel)")
    print(f"{len(np.unique(subject))} subjects, folds (held-out clips): {FOLDS}")
    for n, idx in ch_idx.items():
        print(f"  {n:8s} {len(idx):2d} ch")
    print()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    results, saved_preds = {}, {}

    for mode in a.modes:
        results[mode] = {}
        runner = run_pooled if mode == "pooled" else run_per_subject
        for name, idx in ch_idx.items():
            Xm = X[:, idx, fsl]
            pred = runner(Xm, Y, subject, video, device, a)
            saved_preds[f"pred_{mode}_{name}"] = pred

            per_sub = {d: {k: [] for k in ("ccc", "pcc", "rmse", "acc3")} for d in DIMS}
            for s in np.unique(subject):
                ms = subject == s
                m = metrics(Y[ms], pred[ms])
                for d in DIMS:
                    for k, v in m[d].items():
                        per_sub[d][k].append(v)
            pooled_m = metrics(Y, pred)

            results[mode][name] = {"n_channels": len(idx),
                                   "per_subject": per_sub, "pooled": pooled_m}
            print(f"[{mode:11s}] {name:8s} ({len(idx):2d}ch)  " + "  ".join(
                f"{d[0].upper()}: CCC={np.mean(per_sub[d]['ccc']):+.3f} "
                f"acc3={np.mean(per_sub[d]['acc3']):.3f}" for d in DIMS), flush=True)

            with open(a.out, "w") as f:
                json.dump({"args": vars(a), "results": results,
                           "montage_channels": {k: montages[k] for k in ch_idx}},
                          f, indent=2)
        print()

    print("=" * 84)
    print(f"{'mode':12s} {'montage':9s} {'ch':>3s} | "
          f"{'valence CCC':>13s} {'val acc3':>9s} | {'arousal CCC':>13s} {'aro acc3':>9s}")
    print("=" * 84)
    for mode in results:
        for name, r in results[mode].items():
            cells = []
            for d in DIMS:
                c = r["per_subject"][d]
                cells.append(f"{np.mean(c['ccc']):+.3f}+/-{np.std(c['ccc']):.3f} "
                             f"{np.mean(c['acc3']):9.3f}")
            print(f"{mode:12s} {name:9s} {r['n_channels']:3d} | " + " | ".join(cells))

    np.savez_compressed(a.save_predictions, Y=Y, subject=subject, video=video,
                        second=second, **saved_preds)
    print("\nSaved ->", a.out, "and", a.save_predictions)


if __name__ == "__main__":
    main()
