"""Why does SEED subject-dependent hit ~0.89 while REFED per-subject barely beats chance?

The suspicion is that it is mostly the LABEL STRUCTURE, not the sample count.
In SEED every second of a ~230 s trial carries the SAME label, so the model only
has to recognise a trial-level state and ~230 samples all reinforce one target.
In REFED the label moves second by second (valence sd within a clip = 0.125 on a
range of 2.0), so the model must track moment-to-moment fluctuation.

This holds everything else fixed -- same subjects, same causal DE features, same
DGCNN, same per-subject models, same by-clip folds -- and varies only the label:

  seed_like     one constant 3-class label per clip, taken from the clip's
                designed target emotion (HV -> positive, MV -> neutral,
                LV -> negative). This is exactly SEED's label structure.
  per_second    the continuous joystick trace thresholded per second, i.e. what
                the montage study used.

If seed_like jumps toward SEED-like accuracy, label structure is the answer.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" diag_seedlike.py
"""

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
    FOLDS, MONTAGES, EXCLUDE_FROM_WHOLE, FEATURE_SETS, to_class, EPS,
)

# Valence class implied by each clip's designed target emotion (Video_info.csv).
CLIP_VALENCE_CLASS = {
    1: 1, 10: 1, 12: 1,                      # MVMA neutral
    2: 0, 6: 0, 13: 0, 3: 0, 9: 0, 11: 0,    # LVLA sad + LVHA fear -> negative
    4: 2, 7: 2, 15: 2, 5: 2, 8: 2, 14: 2,    # HVHA happy + HVLA relax -> positive
}


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
    z = np.load(HERE / "refed_features_all64.npz", allow_pickle=True)
    X, Y, subject, video = z["X"], z["Y"], z["subject"], z["video"]
    channels = [str(c) for c in z["channels"]]
    idx = [channels.index(c) for c in channels if c not in EXCLUDE_FROM_WHOLE]
    Xm = X[:, idx, FEATURE_SETS["de"]]

    labels = {
        "seed_like": np.array([CLIP_VALENCE_CLASS[v] for v in video], dtype=np.int64),
        "per_second": to_class(Y[:, 0]),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("whole cap (62ch), causal DE, per-subject models, folds by clip.")
    print("SEED reference for the same model/protocol: ~0.89 (DE_LDS, 62ch)\n")

    for name, y in labels.items():
        accs, majors = [], []
        for sub in np.unique(subject):
            ms = subject == sub
            Xs, ys, vs = Xm[ms], y[ms], video[ms]
            pred = np.zeros_like(ys)
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

                pred[te] = fit_predict(norm(Xs[tr]), ys[tr], norm(Xs[va]), ys[va],
                                       norm(Xs[te]), device, seed=42 + f_i)
            accs.append(float((pred == ys).mean()))
            majors.append(float(np.bincount(ys, minlength=3).max() / len(ys)))
        print(f"{name:11s} accuracy {np.mean(accs):.3f} +/- {np.std(accs):.3f}"
              f"   (majority baseline {np.mean(majors):.3f})")


if __name__ == "__main__":
    main()
