"""Gradient-based saliency map for the whole-cap (62ch) DGCNN classifier,
the same style of analysis as Zheng & Lu (2015) used on SEED (mean absolute
weight/gradient per channel, projected onto the scalp) -- but computed
directly from our own trained model instead of citing theirs.

Trains the same pooled 3-fold classifier as refed_montage_classify.py
(DE_LDS features, whole montage, valence), then for each fold's held-out
test samples, backprops the true-class logit to the input and takes the
mean absolute gradient per channel (averaged over the 5 DE bands and over
all test samples across all 3 folds).

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" saliency_whole.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "emotion_model"))
from models.DGCNN import DGCNN  # noqa: E402
from utils.utils import setup_seed  # noqa: E402
from refed_montage_regress import FOLDS, EPS, to_class, EXCLUDE_FROM_WHOLE  # noqa: E402
from refed_montage_figures import electrode_xy, draw_head, INK, BG, GRID  # noqa: E402


def fit(Xtr, Ctr, Xva, Cva, device, epochs=40, seed=42):
    setup_seed(seed)
    model = DGCNN(num_electrodes=Xtr.shape[1], in_channels=Xtr.shape[2],
                  num_classes=3, k=2, relu_is=1, layers=[64],
                  dropout_rate=0.5).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.Tensor(Xtr), torch.LongTensor(Ctr))
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    Xva_t, Cva_t = torch.Tensor(Xva).to(device), torch.LongTensor(Cva).to(device)

    best, state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for xb, cb in loader:
            xb, cb = xb.to(device), cb.to(device)
            opt.zero_grad()
            crit(model(xb), cb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(crit(model(Xva_t), Cva_t))
        if vl < best:
            best, state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model


def saliency_for_fold(model, Xte, Cte, device, batch=256):
    model.eval()
    grads = []
    for i in range(0, len(Xte), batch):
        xb = torch.Tensor(Xte[i:i + batch]).to(device).requires_grad_(True)
        cb = torch.LongTensor(Cte[i:i + batch]).to(device)
        out = model(xb)
        logit = out.gather(1, cb.unsqueeze(1)).sum()
        logit.backward()
        grads.append(xb.grad.detach().abs().cpu().numpy())
    return np.concatenate(grads, axis=0)  # (n, 62, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="refed_features_libeer_lds.npz")
    ap.add_argument("--tag", default="DE_LDS")
    ap.add_argument("--out-img", default="fig_saliency_whole.png")
    ap.add_argument("--out-npz", default="saliency_whole.npz")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(HERE / a.features, allow_pickle=True)
    X, Y = z["X"], z["Y"]
    subject, video = z["subject"], z["video"]
    channels = [str(c) for c in z["channels"]]
    whole = [c for c in channels if c not in EXCLUDE_FROM_WHOLE]
    idx = [channels.index(c) for c in whole]
    Xm = X[:, idx, 0:5]  # DE only
    C = to_class(Y[:, 0])  # valence, +/-40/127 margin

    all_grads = []
    for f_i, test_vids in enumerate(FOLDS):
        te = np.isin(video, test_vids)
        rest = [v for v in np.unique(video) if v not in test_vids]
        va, tr = np.isin(video, rest[:2]), np.isin(video, rest[2:])

        flat = Xm[tr].reshape(tr.sum(), -1)
        mu, sd = flat.mean(axis=0), flat.std(axis=0) + EPS

        def norm(x):
            return ((x.reshape(len(x), -1) - mu) / sd).reshape(x.shape)

        model = fit(norm(Xm[tr]), C[tr], norm(Xm[va]), C[va], device, seed=42 + f_i)
        g = saliency_for_fold(model, norm(Xm[te]), C[te], device)
        all_grads.append(g)
        print(f"fold {f_i} done, {g.shape[0]} test samples", flush=True)

    grads = np.concatenate(all_grads, axis=0)  # (N, 62, 5)
    importance = grads.mean(axis=(0, 2))  # (62,) mean |grad| over samples and bands
    importance_by_band = grads.mean(axis=0)  # (62, 5)

    order = np.argsort(-importance)
    print("\nTop 15 channels by mean |gradient|:")
    for i in order[:15]:
        print(f"  {whole[i]:5s} {importance[i]:.4f}")

    # --- figure: topographic saliency map ---
    plt.rcParams.update({
        "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": BG, "axes.facecolor": BG,
    })
    xy = electrode_xy(whole)
    imp_norm = (importance - importance.min()) / (importance.max() - importance.min() + EPS)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    r = max(np.hypot(x, y) for x, y in xy.values()) * 1.08
    ax.add_patch(plt.Circle((0, 0), r, fill=False, color=INK, lw=1.4))
    ax.plot([0, -r * 0.11, 0, r * 0.11, 0],
            [r, r * 1.13, r * 1.20, r * 1.13, r], color=INK, lw=1.4)
    for s in (-1, 1):
        ax.add_patch(plt.matplotlib.patches.Ellipse(
            (s * r, 0), r * 0.10, r * 0.32, fill=False, color=INK, lw=1.4))
    cmap = plt.cm.get_cmap("YlOrRd")
    for ch, (x, y) in xy.items():
        i = whole.index(ch)
        v = imp_norm[i]
        ax.scatter(x, y, s=90 + 260 * v, c=[cmap(0.15 + 0.85 * v)],
                  edgecolors=INK, linewidths=0.6, zorder=3)
    for i in order[:10]:
        x, y = xy[whole[i]]
        ax.text(x, y - r * 0.05, whole[i], ha="center", va="top", fontsize=7.5,
                color=INK, zorder=4, fontweight="bold")
    ax.set_xlim(-r * 1.3, r * 1.3)
    ax.set_ylim(-r * 1.3, r * 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Whole-cap (62ch) DGCNN saliency map\n"
                 f"mean |gradient of true-class logit| per channel, {a.tag}, "
                 "pooled across 3 test folds\ntop 10 channels labeled",
                 fontsize=11.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=importance.min(), vmax=importance.max()))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label("mean |gradient|", fontsize=9)
    fig.tight_layout()
    out = HERE / a.out_img
    fig.savefig(out, dpi=170, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved ->", out)

    np.savez(HERE / a.out_npz, importance=importance,
            importance_by_band=importance_by_band, channels=np.array(whole))


if __name__ == "__main__":
    main()
