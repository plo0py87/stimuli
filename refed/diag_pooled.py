"""Does DGCNN become competitive when one model sees every subject?

Per-subject models get ~800 training seconds, against a DGCNN whose fc layer
alone is ~1M parameters -- hopeless. This checks the alternative reading of
"within-subject": a single model trained on ALL subjects' training clips and
tested on held-out clips, so each test subject is still represented in
training but the stimuli are unseen. That is ~25k training samples.

Reference numbers on the same folds (per-subject models, 6-8 subjects):
  ridge  whole V=+0.238 A=+0.114
  DGCNN  whole V=+0.104 A=+0.072
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


def main():
    z = np.load(HERE / "refed_features_all64.npz", allow_pickle=True)
    X, Y, subject, video = z["X"], z["Y"], z["subject"], z["video"]
    channels = [str(c) for c in z["channels"]]

    mont = {
        "whole": [channels.index(c) for c in channels if c not in EXCLUDE_FROM_WHOLE],
        "quick20": [channels.index(c) for c in MONTAGES["quick20"]],
        "glasses": [channels.index(c) for c in MONTAGES["glasses"]],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Feature dims 0-4 are the 5-band causal DE, dims 5-9 the 5-band Welch PSD
    # (that is the concatenation order in the quick20 pipeline). DGCNN's
    # original paper and the SEED/LibEER benchmarks use DE only, so test both.
    featsets = {"DE+PSD": slice(0, 10), "DE only": slice(0, 5)}
    fname = __import__("os").environ.get("FEATSET", "DE+PSD")
    fsl = featsets[fname]
    print(f"feature set: {fname} ({fsl.stop - fsl.start} dims/channel)")

    for mname, idx in mont.items():
        Xm = X[:, idx, fsl]
        pred = np.zeros_like(Y)

        for f_i, test_vids in enumerate(FOLDS):
            te = np.isin(video, test_vids)
            rest = [v for v in np.unique(video) if v not in test_vids]
            va = np.isin(video, rest[:2])
            tr = np.isin(video, rest[2:])

            # Per-subject normalization from that subject's TRAIN clips only.
            Xn = np.empty_like(Xm)
            for s in np.unique(subject):
                ms = subject == s
                flat = Xm[ms & tr].reshape((ms & tr).sum(), -1)
                mu, sd = flat.mean(axis=0), flat.std(axis=0) + EPS
                sh = Xm[ms].shape
                Xn[ms] = ((Xm[ms].reshape(ms.sum(), -1) - mu) / sd).reshape(sh)

            setup_seed(42 + f_i)
            model = DGCNN(num_electrodes=Xm.shape[1], in_channels=Xm.shape[2],
                          num_classes=2, k=2, relu_is=1, layers=[64],
                          dropout_rate=0.5).to(device)
            opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            crit = nn.MSELoss()
            ds = torch.utils.data.TensorDataset(torch.Tensor(Xn[tr]), torch.Tensor(Y[tr]))
            loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
            Xva = torch.Tensor(Xn[va]).to(device)
            Yva = torch.Tensor(Y[va]).to(device)

            best, best_state = float("inf"), None
            for _ in range(40):
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
                out = []
                Xte = Xn[te]
                for i in range(0, len(Xte), 4096):
                    out.append(model(torch.Tensor(Xte[i:i + 4096]).to(device)).cpu().numpy())
                pred[te] = np.concatenate(out)

        # Pooled CCC, and CCC computed per subject then averaged (comparable to
        # the per-subject-model numbers).
        per_sub = np.array([[ccc(Y[subject == s][:, i], pred[subject == s][:, i])
                             for i in range(2)] for s in np.unique(subject)])
        print(f"{mname:8s} ({len(idx):2d} ch)  "
              f"pooled V={ccc(Y[:,0], pred[:,0]):+.3f} A={ccc(Y[:,1], pred[:,1]):+.3f}   "
              f"per-subject mean V={per_sub[:,0].mean():+.3f}+/-{per_sub[:,0].std():.3f} "
              f"A={per_sub[:,1].mean():+.3f}+/-{per_sub[:,1].std():.3f}", flush=True)


if __name__ == "__main__":
    main()
