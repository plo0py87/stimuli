"""Per-subject (strict SEED subject-dependent) version of
seed_montage_classify.py. That script's "per_session" dict is metrics
aggregated per subject AFTER a model trained POOLED across all 15 subjects --
the same pooled/per-subject naming confusion this project hit on the REFED
side. This script actually trains one DGCNN per subject, using only that
subject's own 15 trials, for a genuine apples-to-apples comparison against
REFED's per_subject numbers.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" seed_montage_classify_persubj.py
"""

import json
import re
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
from refed_extract_libeer import libeer_bandpass, libeer_de, PASS_BAND  # noqa: E402
from data_utils.preprocess import lds  # noqa: E402
from data_utils.constants.seed import SEED_CHANNEL_NAME  # noqa: E402
from models.DGCNN import DGCNN  # noqa: E402
from utils.utils import setup_seed  # noqa: E402
from refed_montage_regress import MONTAGES as REFED_MONTAGES, EPS  # noqa: E402

SEED_DIR = Path("D:/EEG dataset/SEED/SEED/SEED_EEG/Preprocessed_EEG")
SEED_SRATE = 200
SEED_LABELS = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]) + 1

MONTAGES = {
    "whole": SEED_CHANNEL_NAME,
    "vr_ring": REFED_MONTAGES["vr_ring"],
    "quick20": REFED_MONTAGES["quick20"],
    "headtop": REFED_MONTAGES["headtop"],
}


def macro_recall(y, p):
    rec = [(p[y == c] == c).mean() for c in range(3) if (y == c).any()]
    return float(np.mean(rec)) if rec else float("nan")


def seed_one_session_per_subject():
    files = sorted(p for p in SEED_DIR.glob("*.mat") if p.name != "label.mat")
    by_subject = {}
    for f in files:
        sub = re.match(r"(\d+)_", f.name).group(1)
        by_subject.setdefault(sub, []).append(f)
    for sub in sorted(by_subject, key=int):
        yield sub, sorted(by_subject[sub])[0]


def extract_subject(f):
    """DE_LDS per trial, all 62 channels, for one subject's one session."""
    m = sio.loadmat(f)
    keys = [k for k in m if not k.startswith("__")]
    keys.sort(key=lambda k: int(k.split("_eeg")[-1]))
    feats, ys, tri = [], [], []
    for t_i, k in enumerate(keys):
        raw = m[k].astype(np.float64)[:62]
        filt = libeer_bandpass(raw, SEED_SRATE, PASS_BAND)
        de = libeer_de(filt, SEED_SRATE)
        de_lds = lds(de)
        for v in de_lds:
            feats.append(v)
            ys.append(SEED_LABELS[t_i])
            tri.append(t_i)
    X = np.stack(feats).astype(np.float32)  # (N, 62, 5)
    return X, np.array(ys, np.int64), np.array(tri, np.int64)


def fit_predict(Xtr, ytr, Xva, yva, Xte, device, epochs=40, seed=42):
    setup_seed(seed)
    model = DGCNN(num_electrodes=Xtr.shape[1], in_channels=Xtr.shape[2],
                  num_classes=3, k=2, relu_is=1, layers=[64],
                  dropout_rate=0.5).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.Tensor(Xtr), torch.LongTensor(ytr))
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_class = {c: [i for i in range(15) if SEED_LABELS[i] == c] for c in range(3)}
    folds = [[by_class[c][i] for c in range(3)] for i in range(5)]
    print(f"5 folds (held-out trials): {folds}")

    results = {name: {"n_channels": len(chs), "per_subject": {"acc": [], "macro": []}}
              for name, chs in MONTAGES.items()}

    for s_i, (sub, f) in enumerate(seed_one_session_per_subject()):
        X, y, tri = extract_subject(f)
        for name, chs in MONTAGES.items():
            idx = [SEED_CHANNEL_NAME.index(c) for c in chs]
            Xm = X[:, idx, :]
            pred = np.zeros_like(y)
            for f_i, test_trials in enumerate(folds):
                te = np.isin(tri, test_trials)
                rest = [t for t in range(15) if t not in test_trials]
                va, tr = np.isin(tri, rest[:2]), np.isin(tri, rest[2:])
                flat = Xm[tr].reshape(tr.sum(), -1)
                mu, sd = flat.mean(axis=0), flat.std(axis=0) + EPS

                def norm(x):
                    return ((x.reshape(len(x), -1) - mu) / sd).reshape(x.shape)

                pred[te] = fit_predict(norm(Xm[tr]), y[tr], norm(Xm[va]), y[va],
                                       norm(Xm[te]), device, seed=42 + f_i)
            acc = float((pred == y).mean())
            macro = macro_recall(y, pred)
            results[name]["per_subject"]["acc"].append(acc)
            results[name]["per_subject"]["macro"].append(macro)
            print(f"  subject {sub} [{name:8s}] acc={acc:.3f} macro={macro:.3f}", flush=True)

    for name in MONTAGES:
        a = results[name]["per_subject"]["acc"]
        m = results[name]["per_subject"]["macro"]
        print(f"[{name:8s}] mean acc={np.mean(a):.3f} mean macro={np.mean(m):.3f}")

    with open(HERE / "seed_montage_cls_persubj.json", "w") as f:
        json.dump({"results": results, "montage_channels": MONTAGES}, f, indent=2)
    print("\nSaved -> seed_montage_cls_persubj.json")


if __name__ == "__main__":
    main()
