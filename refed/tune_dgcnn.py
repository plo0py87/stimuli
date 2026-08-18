"""Probe DGCNN hyperparameters for the REFED continuous-regression task.

The first montage run scored below a ridge baseline on the same folds, which
points at the training setup rather than the task. This tries a small grid on
a few subjects, using the whole-cap and glasses montages (the two extremes of
electrode count), and reports leave-videos-out CCC so it is directly
comparable to diag_signal.py's ridge numbers.
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
from refed_montage_regress import FOLDS, MONTAGES, EXCLUDE_FROM_WHOLE, ccc  # noqa: E402

EPS = 1e-8

CONFIGS = [
    dict(name="base       ", dropout=0.5,  lr=1e-3, epochs=40,  wd=1e-4, nval=2),
    dict(name="drop.25    ", dropout=0.25, lr=1e-3, epochs=40,  wd=1e-4, nval=2),
    dict(name="drop.25 lo  ", dropout=0.25, lr=5e-4, epochs=80,  wd=1e-3, nval=3),
    dict(name="drop.1 lo   ", dropout=0.1,  lr=5e-4, epochs=80,  wd=1e-3, nval=3),
    dict(name="drop.25 wd2 ", dropout=0.25, lr=5e-4, epochs=120, wd=1e-2, nval=3),
]


def run(Xm, Ys, vs, cfg, device, seed=42):
    pred = np.zeros_like(Ys)
    for f_i, test_vids in enumerate(FOLDS):
        te = np.isin(vs, test_vids)
        rest = [v for v in np.unique(vs) if v not in test_vids]
        if te.sum() == 0 or len(rest) <= cfg["nval"]:
            continue
        va = np.isin(vs, rest[:cfg["nval"]])
        tr = np.isin(vs, rest[cfg["nval"]:])

        flat = Xm[tr].reshape(tr.sum(), -1)
        mu, sd = flat.mean(axis=0), flat.std(axis=0) + EPS

        def norm(a):
            sh = a.shape
            return ((a.reshape(len(a), -1) - mu) / sd).reshape(sh)

        setup_seed(seed + f_i)
        model = DGCNN(num_electrodes=Xm.shape[1], in_channels=Xm.shape[2],
                      num_classes=2, k=2, relu_is=1, layers=[64],
                      dropout_rate=cfg["dropout"]).to(device)
        opt = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        crit = nn.MSELoss()
        ds = torch.utils.data.TensorDataset(torch.Tensor(norm(Xm[tr])), torch.Tensor(Ys[tr]))
        loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)
        Xva = torch.Tensor(norm(Xm[va])).to(device)
        Yva = torch.Tensor(Ys[va]).to(device)

        best, best_state = float("inf"), None
        for _ in range(cfg["epochs"]):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = float(crit(model(Xva), Yva))
            if vl < best:
                best, best_state = vl, {k: v.detach().clone()
                                        for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred[te] = model(torch.Tensor(norm(Xm[te])).to(device)).cpu().numpy()
    return [ccc(Ys[:, i], pred[:, i]) for i in range(2)]


def main():
    z = np.load(HERE / "refed_features_all64.npz", allow_pickle=True)
    X, Y, subject, video = z["X"], z["Y"], z["subject"], z["video"]
    channels = [str(c) for c in z["channels"]]

    mont = {"whole": [channels.index(c) for c in channels if c not in EXCLUDE_FROM_WHOLE],
            "glasses": [channels.index(c) for c in MONTAGES["glasses"]]}
    subs = sorted(set(subject.tolist()))[:6]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("leave-videos-out CCC, mean over %d subjects "
          "(ridge reference: whole V=+0.238 A=+0.114)\n" % len(subs))
    for mname, idx in mont.items():
        print(f"--- {mname} ({len(idx)} ch) ---")
        for cfg in CONFIGS:
            vals = []
            for sub in subs:
                m = subject == sub
                vals.append(run(X[m][:, idx, :], Y[m], video[m], cfg, device))
            a = np.array(vals)
            print(f"  {cfg['name']} V={a[:,0].mean():+.3f}+/-{a[:,0].std():.3f}  "
                  f"A={a[:,1].mean():+.3f}+/-{a[:,1].std():.3f}")
        print()


if __name__ == "__main__":
    main()
