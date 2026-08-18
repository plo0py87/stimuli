"""Same as tsne_compare_insession.py but plain DE, no LDS -- direct visual
counterpart to fig_tsne_insession.png so the smoothing effect can be seen,
not just read off silhouette numbers.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" tsne_compare_insession_de.py
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

plt.rcParams.update({
    "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
})


def load_seed_one_session():
    f = sorted(p for p in SEED_DIR.glob("*.mat") if p.name != "label.mat")[0]
    m = sio.loadmat(f)
    keys = [k for k in m if not k.startswith("__")]
    keys.sort(key=lambda k: int(k.split("_eeg")[-1]))
    feats, ys = [], []
    for t_i, k in enumerate(keys):
        raw = m[k].astype(np.float64)[:62]
        filt = libeer_bandpass(raw, SEED_SRATE, PASS_BAND)
        de = libeer_de(filt, SEED_SRATE)
        for v in de:
            feats.append(v.reshape(-1))
            ys.append(SEED_LABELS[t_i])
    print(f"SEED session: {f.name}")
    return np.stack(feats).astype(np.float32), np.array(ys, dtype=np.int64)


def load_refed_one_subject(sub=9):
    z = np.load(HERE / "refed_features_libeer.npz", allow_pickle=True)
    X, Y, subject, channels = z["X"], z["Y"], z["subject"], [str(c) for c in z["channels"]]
    keep = [i for i, c in enumerate(channels) if c not in ("M1", "M2")]
    m = subject == sub
    X = X[m][:, keep, :]
    ys = to_class(Y[m, 0])
    print(f"REFED subject: {sub}")
    return X.reshape(len(X), -1).astype(np.float32), ys


def run_tsne(X, y, name):
    mu, sd = X.mean(axis=0), X.std(axis=0) + EPS
    Xn = (X - mu) / sd
    perp = min(30, max(5, len(y) // 10))
    emb = TSNE(n_components=2, perplexity=perp, random_state=42,
              init="pca", learning_rate="auto").fit_transform(Xn)
    sil_high = silhouette_score(Xn, y)
    sil_emb = silhouette_score(emb, y)
    print(f"{name}: n={len(y)}  class balance={np.bincount(y)}  "
          f"perplexity={perp}  silhouette(high-dim)={sil_high:+.3f}  "
          f"silhouette(t-SNE 2D)={sil_emb:+.3f}")
    return emb, sil_high, sil_emb


def main():
    print("Loading SEED (1 session, plain DE)...")
    Xs, ys = load_seed_one_session()
    print("\nLoading REFED (1 subject, plain DE)...")
    Xr, yr = load_refed_one_subject(sub=9)

    print("\nRunning t-SNE...")
    emb_s, sh_s, se_s = run_tsne(Xs, ys, "SEED  (1 session) ")
    emb_r, sh_r, se_r = run_tsne(Xr, yr, "REFED (1 subject)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, emb, y, title, sh, se in (
        (axes[0], emb_s, ys, "SEED -- ONE session, plain DE", sh_s, se_s),
        (axes[1], emb_r, yr, "REFED -- ONE subject, plain DE", sh_r, se_r),
    ):
        for c in range(3):
            m = y == c
            ax.scatter(emb[m, 0], emb[m, 1], s=14, alpha=0.7,
                      color=CLASS_COLORS[c], label=CLASS_NAMES[c])
        ax.set_title(f"{title}\nsilhouette: high-dim={sh:+.3f}, t-SNE={se:+.3f}",
                     fontsize=11)
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("t-SNE, plain DE (no smoothing), single session/subject only",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    out = HERE / "fig_tsne_insession_de.png"
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved ->", out)


if __name__ == "__main__":
    main()
