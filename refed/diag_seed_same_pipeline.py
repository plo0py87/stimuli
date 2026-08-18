"""Run the EXACT REFED pipeline on SEED, to separate method from data.

REFED per-subject barely beats its majority baseline while SEED subject-dependent
is published around 0.89. Two candidate explanations:

  method  our features are causal DE with no LDS smoothing, and our per-subject
          models see little data
  data    SEED's preprocessed recordings and 4-minute validated film clips simply
          carry a much stronger, cleaner emotion signal than REFED's raw 1-3
          minute clips with no artifact rejection

This runs SEED through the same code path as the REFED montage study: causal DE
(no LDS), per-subject-session DGCNN, folds by trial, same hyperparameters. If
SEED still scores high here, the gap is the data. If SEED collapses too, the
gap is our method.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" diag_seed_same_pipeline.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "emotion_model"))

from models.DGCNN import DGCNN  # noqa: E402
from utils.utils import setup_seed  # noqa: E402
from DGCNN_quick20_realtime_train import (  # noqa: E402
    causal_de_features, kalman_smooth_trial,
)
from refed_montage_regress import EPS  # noqa: E402

SEED_DIR = Path("D:/EEG dataset/SEED/SEED/SEED_EEG/Preprocessed_EEG")
SEED_SRATE = 200
# SEED ships one label sequence reused by all three sessions; shift to 0/1/2.
SEED_LABELS = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]) + 1


def fit_predict(Xtr, ytr, Xva, yva, Xte, device, epochs=40, seed=42):
    setup_seed(seed)
    model = DGCNN(num_electrodes=Xtr.shape[1], in_channels=Xtr.shape[2],
                  num_classes=3, k=2, relu_is=1, layers=[64],
                  dropout_rate=0.5).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.Tensor(Xtr), torch.LongTensor(ytr))
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)
    Xva_t, yva_t = torch.Tensor(Xva).to(device), torch.LongTensor(yva).to(device)

    best, state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(crit(model(Xva_t), yva_t))
        if vl < best:
            best, state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return model(torch.Tensor(Xte).to(device)).argmax(dim=1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=9, help="subject-session files to use")
    ap.add_argument("--kalman-q", type=float, default=0.01)
    ap.add_argument("--kalman-r", type=float, default=0.5)
    args = ap.parse_args()

    files = sorted(p for p in SEED_DIR.glob("*.mat") if p.name != "label.mat")[:args.files]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx62 = list(range(62))

    # Same 3 folds as REFED in spirit: hold out 5 trials at a time, balanced by
    # class (SEED has 5 trials per class).
    by_class = {c: [i for i in range(15) if SEED_LABELS[i] == c] for c in range(3)}
    folds = [[by_class[c][i] for c in range(3)] for i in range(5)]

    accs, majors = [], []
    for f in files:
        m = sio.loadmat(f)
        keys = [k for k in m if not k.startswith("__")]
        keys.sort(key=lambda k: int(k.split("_eeg")[-1]))

        feats, ys, tri = [], [], []
        for t_i, k in enumerate(keys):
            raw = m[k].astype(np.float64)[idx62]
            de = causal_de_features(raw, SEED_SRATE, idx62)
            de = kalman_smooth_trial(de, args.kalman_q, args.kalman_r)
            for v in de:
                feats.append(v)
                ys.append(SEED_LABELS[t_i])
                tri.append(t_i)
        Xs = np.stack(feats).astype(np.float32)
        ys = np.array(ys, dtype=np.int64)
        tri = np.array(tri)

        pred = np.zeros_like(ys)
        for f_i, test_trials in enumerate(folds):
            te = np.isin(tri, test_trials)
            rest = [t for t in range(15) if t not in test_trials]
            va, tr = np.isin(tri, rest[:2]), np.isin(tri, rest[2:])
            flat = Xs[tr].reshape(tr.sum(), -1)
            mu, sd = flat.mean(axis=0), flat.std(axis=0) + EPS

            def norm(x):
                sh = x.shape
                return ((x.reshape(len(x), -1) - mu) / sd).reshape(sh)

            pred[te] = fit_predict(norm(Xs[tr]), ys[tr], norm(Xs[va]), ys[va],
                                   norm(Xs[te]), device, seed=42 + f_i)
        acc = float((pred == ys).mean())
        maj = float(np.bincount(ys, minlength=3).max() / len(ys))
        accs.append(acc)
        majors.append(maj)
        print(f"  {f.name:22s} n={len(ys):5d}  acc={acc:.3f}  (majority {maj:.3f})",
              flush=True)

    print(f"\nSEED, REFED pipeline (causal DE, no LDS, per-subject-session, "
          f"folds by trial, 62ch)")
    print(f"  accuracy {np.mean(accs):.3f} +/- {np.std(accs):.3f} "
          f"over {len(accs)} subject-sessions   (majority {np.mean(majors):.3f})")
    print(f"\nREFED same pipeline, seed-like constant clip labels: 0.534 "
          f"(majority 0.518)")
    print(f"Published SEED subject-dependent with DE_LDS: ~0.89")


if __name__ == "__main__":
    main()
