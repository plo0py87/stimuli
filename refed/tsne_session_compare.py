"""In-session t-SNE, multi-subject, dual coloring: emotion class vs subject
identity -- side by side, so a viewer can see whether apparent class
separation is actually a subject-identity artifact.

Unlike tsne_compare.py/tsne_compare_lds.py (which pool ALL subjects together
before z-scoring -- comparing across people, not within-session), every point
here is normalized using ONLY its own subject's mean/std (matching
normalize_per_subject() in refed_montage_regress.py, the same in-session
convention used everywhere else in this project). Multiple subjects are shown
in one embedding only for visual comparison; no subject's stats ever leak
into another's.

Two figures: plain DE and DE_LDS, each with two panels on the SAME t-SNE
embedding -- left colored by emotion class (as before), right colored by
subject ID.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" tsne_session_compare.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_regress import to_class, EPS  # noqa: E402
from refed_montage_figures import INK, BG, GRID  # noqa: E402

CLASS_NAMES = ["negative", "neutral", "positive"]
CLASS_COLORS = {0: "#c1443f", 1: "#b8873a", 2: "#2f5f5c"}
SUBJECTS = [1, 5, 9, 13, 17, 21, 25, 29]
SUBJ_CMAP = plt.colormaps["tab10"]
N_PER_SUBJECT = 350
RNG = np.random.RandomState(42)

plt.rcParams.update({
    "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
})


def load_in_session(feat_path, subjects, n_per_subject):
    """Each subject z-scored against ONLY its own data, then subsampled --
    in-session normalization, multi-subject only for the scatter plot."""
    z = np.load(feat_path, allow_pickle=True)
    X, Y, subject = z["X"], z["Y"], z["subject"]
    channels = [str(c) for c in z["channels"]]
    keep = [i for i, c in enumerate(channels) if c not in ("M1", "M2")]

    feats, ys, subs = [], [], []
    for sub in subjects:
        m = subject == sub
        Xs = X[m][:, keep, :].reshape(m.sum(), -1).astype(np.float32)
        ycls = to_class(Y[m, 0])
        mu, sd = Xs.mean(axis=0), Xs.std(axis=0) + EPS
        Xn = (Xs - mu) / sd  # in-session: this subject's own stats only

        idx = []
        for c in np.unique(ycls):
            ci = np.nonzero(ycls == c)[0]
            take = min(len(ci), n_per_subject // 3)
            idx.append(RNG.choice(ci, take, replace=False))
        idx = np.concatenate(idx)
        feats.append(Xn[idx])
        ys.append(ycls[idx])
        subs.append(np.full(len(idx), sub))

    return (np.concatenate(feats), np.concatenate(ys), np.concatenate(subs))


def run_and_plot(feat_path, out_path, title_tag):
    X, y, subj = load_in_session(feat_path, SUBJECTS, N_PER_SUBJECT)
    print(f"{title_tag}: n={len(y)}  subjects={SUBJECTS}  "
          f"class balance={np.bincount(y)}")

    emb = TSNE(n_components=2, perplexity=30, random_state=42,
              init="pca", learning_rate="auto").fit_transform(X)
    sil_class = silhouette_score(emb, y)
    sil_subj = silhouette_score(emb, subj)
    print(f"  silhouette by class={sil_class:+.3f}  by subject={sil_subj:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    for c in range(3):
        m = y == c
        ax.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.6,
                  color=CLASS_COLORS[c], label=CLASS_NAMES[c])
    ax.set_title(f"colored by emotion class\nsilhouette={sil_class:+.3f}", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    ax = axes[1]
    for i, sub in enumerate(SUBJECTS):
        m = subj == sub
        ax.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.6,
                  color=SUBJ_CMAP(i / len(SUBJECTS)), label=f"sub {sub}")
    ax.set_title(f"colored by subject identity\nsilhouette={sil_subj:+.3f}", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)

    for ax in axes:
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.suptitle(f"REFED, {title_tag}, {len(SUBJECTS)} subjects, in-session normalized "
                 "(each subject z-scored against its own data only)\n"
                 "same t-SNE embedding, two colorings -- if class separation only shows "
                 "up as subject clusters, it's identity, not emotion", fontsize=11.5, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("  Saved ->", out_path)
    return sil_class, sil_subj


def main():
    print("=== plain DE ===")
    run_and_plot(HERE / "refed_features_libeer.npz",
                HERE / "fig_tsne_session_de.png", "plain DE")
    print("\n=== DE_LDS ===")
    run_and_plot(HERE / "refed_features_libeer_lds.npz",
                HERE / "fig_tsne_session_lds.png", "DE_LDS")


if __name__ == "__main__":
    main()
