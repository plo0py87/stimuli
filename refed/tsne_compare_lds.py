"""Same t-SNE separability check as tsne_compare.py, but with DE_LDS instead
of plain DE on both sides -- LDS is literally the feature SEED's published
benchmarks are built on, so this is the more apples-to-apples version of
"is REFED's feature space as separable as SEED's."

SEED: same libeer_bandpass+libeer_de per trial as before, then LibEER's own
lds() applied per trial (each trial's (time,62,5) DE sequence is exactly the
shape lds() expects).
REFED: reuses the already-computed refed_features_libeer_lds.npz (add_lds.py),
just excludes M1/M2 to match SEED's 62 channels.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" tsne_compare_lds.py
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
sys.path.insert(0, str(HERE.parent / "emotion_model"))
from refed_extract_libeer import libeer_bandpass, libeer_de, PASS_BAND  # noqa: E402
from data_utils.preprocess import lds  # noqa: E402
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


def load_seed_lds(n_files=9):
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
            de_lds = lds(de)  # non-causal LDS, whole-trial mean init
            for v in de_lds:
                feats.append(v.reshape(-1))
                ys.append(SEED_LABELS[t_i])
        print(f"  loaded {f.name}", flush=True)
    return np.stack(feats).astype(np.float32), np.array(ys, dtype=np.int64)


def load_refed_lds():
    z = np.load(HERE / "refed_features_libeer_lds.npz", allow_pickle=True)
    X, Y, channels = z["X"], z["Y"], [str(c) for c in z["channels"]]
    keep = [i for i, c in enumerate(channels) if c not in ("M1", "M2")]
    X = X[:, keep, :]
    ys = to_class(Y[:, 0])
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
    print("Loading SEED (DE_LDS)...")
    Xs, ys = load_seed_lds()
    print(f"SEED raw: {Xs.shape}, class balance {np.bincount(ys)}")
    Xs, ys = subsample_balanced(Xs, ys, N_SAMPLE)

    print("\nLoading REFED (DE_LDS)...")
    Xr, yr = load_refed_lds()
    print(f"REFED raw: {Xr.shape}, class balance {np.bincount(yr)}")
    Xr, yr = subsample_balanced(Xr, yr, N_SAMPLE)

    print("\nRunning t-SNE...")
    emb_s, sh_s, se_s = run_tsne(Xs, ys, "SEED ")
    emb_r, sh_r, se_r = run_tsne(Xr, yr, "REFED")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, emb, y, title, sh, se in (
        (axes[0], emb_s, ys, "SEED, DE_LDS (published, validated)", sh_s, se_s),
        (axes[1], emb_r, yr, "REFED, DE_LDS (valence, +/-40/127)", sh_r, se_r),
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
    fig.suptitle("t-SNE of DE_LDS features (non-causal LDS smoothing), 62ch, "
                 "3-class emotion labels\nsame per-second sample granularity as "
                 "the plain-DE version, only the smoother changed", fontsize=12, y=1.02)
    fig.tight_layout()
    out = HERE / "fig_tsne_seed_vs_refed_lds.png"
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved ->", out)


if __name__ == "__main__":
    main()
