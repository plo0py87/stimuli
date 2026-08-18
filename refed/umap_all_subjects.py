"""Per-subject UMAP, DE and DE_LDS, for ALL 32 REFED subjects -- the exact
counterpart of tsne_all_subjects.py, so the dashboard can show UMAP next to
t-SNE for the same subject/feature and let the reader judge whether the
class structure (or lack of it) is an artifact of t-SNE specifically.

Same in-session z-score, same 3-class coloring, same silhouette-in-high-dim
and silhouette-in-embedding convention as the t-SNE script, so the two are
directly comparable.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" umap_all_subjects.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP
from sklearn.metrics import silhouette_score
from PIL import Image


def _quantize(path):
    """Scatter plots use only a handful of colors (background + 3 class
    dots) -- palette quantization cuts file size ~4x with no visible loss,
    same trick used for the dashboard's regression-curve images."""
    Image.open(path).convert("RGB").quantize(colors=32, method=Image.MEDIANCUT).save(
        path, optimize=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_regress import to_class, EPS  # noqa: E402
from refed_montage_figures import INK, BG  # noqa: E402

CLASS_COLORS = {0: "#c1443f", 1: "#b8873a", 2: "#2f5f5c"}
FEATURES = {
    "de": HERE / "refed_features_libeer.npz",
    "lds": HERE / "refed_features_libeer_lds.npz",
}

plt.rcParams.update({
    "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
})


def run_one(X, y, sub, tag):
    mu, sd = X.mean(axis=0), X.std(axis=0) + EPS
    Xn = (X - mu) / sd
    n_neighbors = min(15, max(2, len(y) - 1))
    emb = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
              random_state=42).fit_transform(Xn)
    sil_high = silhouette_score(Xn, y) if len(np.unique(y)) > 1 else float("nan")
    sil_emb = silhouette_score(emb, y) if len(np.unique(y)) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    for c in range(3):
        m = y == c
        if m.any():
            ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.65, color=CLASS_COLORS[c])
    ax.set_title(f"sub {sub} {tag}\nsil={sil_emb:+.3f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    out = HERE / f"dash_umap_sub{sub}_{tag}.png"
    fig.savefig(out, dpi=95, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    _quantize(out)
    return {"n": int(len(y)), "sil_high": float(sil_high), "sil_emb": float(sil_emb),
           "image": out.name}


N_POOLED_SAMPLE = 3000
RNG = np.random.RandomState(42)


def run_pooled(X, y, tag):
    """All 32 subjects combined, z-scored using POOLED stats (not per-subject) --
    same convention as tsne_all_subjects.py's run_pooled, subsampled for
    UMAP tractability (UMAP scales better than t-SNE but this keeps the
    pooled comparison apples-to-apples)."""
    idx = []
    for c in np.unique(y):
        ci = np.nonzero(y == c)[0]
        take = min(len(ci), N_POOLED_SAMPLE // 3)
        idx.append(RNG.choice(ci, take, replace=False))
    idx = np.concatenate(idx)
    RNG.shuffle(idx)
    Xs, ys = X[idx], y[idx]

    mu, sd = Xs.mean(axis=0), Xs.std(axis=0) + EPS
    Xn = (Xs - mu) / sd
    emb = UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
              random_state=42).fit_transform(Xn)
    sil_high = silhouette_score(Xn, ys)
    sil_emb = silhouette_score(emb, ys)

    fig, ax = plt.subplots(figsize=(5, 5))
    for c in range(3):
        m = ys == c
        if m.any():
            ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.55, color=CLASS_COLORS[c])
    ax.set_title(f"pooled (32 subjects) {tag}\nsil={sil_emb:+.3f}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    out = HERE / f"dash_umap_pooled_{tag}.png"
    fig.savefig(out, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    _quantize(out)
    print(f"pooled [{tag}] n={len(ys)} sil_high={sil_high:+.3f} sil_emb={sil_emb:+.3f}")
    return {"n": int(len(ys)), "sil_high": float(sil_high), "sil_emb": float(sil_emb),
           "image": out.name}


def main():
    results = {}
    pooled = {}
    for tag, path in FEATURES.items():
        z = np.load(path, allow_pickle=True)
        X, Y, subject, channels = z["X"], z["Y"], z["subject"], [str(c) for c in z["channels"]]
        keep = [i for i, c in enumerate(channels) if c not in ("M1", "M2")]
        Xfull = X[:, keep, :].reshape(len(X), -1).astype(np.float32)
        yfull = to_class(Y[:, 0])

        for sub in range(1, 33):
            m = subject == sub
            Xs = X[m][:, keep, :].reshape(m.sum(), -1).astype(np.float32)
            ys = to_class(Y[m, 0])
            r = run_one(Xs, ys, sub, tag)
            results.setdefault(str(sub), {})[tag] = r
            print(f"sub {sub:2d} [{tag}] n={r['n']} sil_high={r['sil_high']:+.3f} "
                  f"sil_emb={r['sil_emb']:+.3f}", flush=True)

        pooled[tag] = run_pooled(Xfull, yfull, tag)

    with open(HERE / "dash_umap_all.json", "w") as f:
        json.dump({"per_subject": results, "pooled": pooled}, f, indent=2)
    print("\nSaved -> dash_umap_all.json + 66 PNGs (64 per-subject + 2 pooled)")


if __name__ == "__main__":
    main()
