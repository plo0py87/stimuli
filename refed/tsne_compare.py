"""t-SNE sanity check: are the 3 emotion classes visually/quantitatively
separable in REFED's feature space, compared to SEED (a dataset that's been
validated by dozens of papers)?

Both datasets go through the EXACT SAME feature recipe -- LibEER's SEED DE
(libeer_bandpass + libeer_de from refed_extract_libeer.py), 62 channels,
200Hz -- so any difference in cluster structure is about the DATA, not the
pipeline. This is the same apples-to-apples logic as diag_seed_same_pipeline.py,
just for visualization instead of classification accuracy.

Labels:
  SEED   static per-trial label (SEED_LABELS), the dataset's own convention.
  REFED  to_class() on valence with the +/-40/127 margin used throughout this
         project -- per-second, not per-trial, because REFED's label is
         genuinely dynamic. This labeling-philosophy mismatch is inherent to
         comparing the two datasets and is called out in the printed output,
         not hidden.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" tsne_compare.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_extract_libeer import libeer_bandpass, libeer_de, PASS_BAND  # noqa: E402
from refed_montage_regress import to_class, EPS  # noqa: E402
from refed_montage_figures import INK, BG, GRID  # noqa: E402

SEED_DIR = Path("D:/EEG dataset/SEED/SEED/SEED_EEG/Preprocessed_EEG")
SEED_SRATE = 200
SEED_LABELS = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]) + 1
CLASS_NAMES = ["negative", "neutral", "positive"]
CLASS_COLORS = {0: "#c1443f", 1: "#b8873a", 2: "#2f5f5c"}
N_SAMPLE = 3000
RNG = np.random.RandomState(42)

plt.rcParams.update({
    "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
})


def load_seed(n_files=9):
    files = sorted(p for p in SEED_DIR.glob("*.mat") if p.name != "label.mat")[:n_files]
    feats, ys = [], []
    for f in files:
        m = sio.loadmat(f)
        keys = [k for k in m if not k.startswith("__")]
        keys.sort(key=lambda k: int(k.split("_eeg")[-1]))
        for t_i, k in enumerate(keys):
            raw = m[k].astype(np.float64)[:62]
            filt = libeer_bandpass(raw, SEED_SRATE, PASS_BAND)
            de = libeer_de(filt, SEED_SRATE)  # (T, 62, 5)
            for v in de:
                feats.append(v.reshape(-1))
                ys.append(SEED_LABELS[t_i])
        print(f"  loaded {f.name}", flush=True)
    return np.stack(feats).astype(np.float32), np.array(ys, dtype=np.int64)


def load_refed():
    z = np.load(HERE / "refed_features_libeer.npz", allow_pickle=True)
    X, Y, channels = z["X"], z["Y"], [str(c) for c in z["channels"]]
    keep = [i for i, c in enumerate(channels) if c not in ("M1", "M2")]
    X = X[:, keep, :]  # -> 62 channels, matches SEED
    ys = to_class(Y[:, 0])  # valence, +/-40/127 margin
    return X.reshape(len(X), -1).astype(np.float32), ys


def subsample_balanced(X, y, n):
    idx = []
    for c in np.unique(y):
        ci = np.nonzero(y == c)[0]
        take = min(len(ci), n // 3)
        idx.append(RNG.choice(ci, take, replace=False))
    idx = np.concatenate(idx)
    RNG.shuffle(idx)
    return X[idx], y[idx]


def run_tsne(X, y, name):
    mu, sd = X.mean(axis=0), X.std(axis=0) + EPS
    Xn = (X - mu) / sd
    emb = TSNE(n_components=2, perplexity=30, random_state=42,
              init="pca", learning_rate="auto").fit_transform(Xn)
    sil_high = silhouette_score(Xn, y)
    sil_emb = silhouette_score(emb, y)
    print(f"{name}: n={len(y)}  silhouette(high-dim)={sil_high:+.3f}  "
          f"silhouette(t-SNE 2D)={sil_emb:+.3f}")
    return emb, sil_high, sil_emb


def main():
    print("Loading SEED...")
    Xs, ys = load_seed()
    print(f"SEED raw: {Xs.shape}, class balance {np.bincount(ys)}")
    Xs, ys = subsample_balanced(Xs, ys, N_SAMPLE)

    print("\nLoading REFED...")
    Xr, yr = load_refed()
    print(f"REFED raw: {Xr.shape}, class balance {np.bincount(yr)}")
    Xr, yr = subsample_balanced(Xr, yr, N_SAMPLE)

    print("\nRunning t-SNE...")
    emb_s, sh_s, se_s = run_tsne(Xs, ys, "SEED ")
    emb_r, sh_r, se_r = run_tsne(Xr, yr, "REFED")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, emb, y, title, sh, se in (
        (axes[0], emb_s, ys, "SEED (published, validated)", sh_s, se_s),
        (axes[1], emb_r, yr, "REFED (valence, +/-40/127)", sh_r, se_r),
    ):
        for c in range(3):
            m = y == c
            ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.6,
                      color=CLASS_COLORS[c], label=CLASS_NAMES[c])
        ax.set_title(f"{title}\nsilhouette: high-dim={sh:+.3f}, t-SNE={se:+.3f}",
                     fontsize=11)
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("t-SNE of 5-band DE features (LibEER's exact SEED recipe, 62ch), "
                 "3-class emotion labels\nsilhouette > 0 means classes separate "
                 "more than they mix; ~0 means no visible structure", fontsize=12, y=1.02)
    fig.tight_layout()
    out = HERE / "fig_tsne_seed_vs_refed.png"
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved ->", out)


if __name__ == "__main__":
    main()
