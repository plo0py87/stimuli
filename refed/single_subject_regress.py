"""Train DGCNN on ONE subject's own data only (no pooling across the other 31),
whole cap, LibEER+non-causal LDS features -- to check whether cross-subject
mixing is what's flattening the within-trial predictions, or whether it's
inherent to the task (per the within/between variance decomposition).

Same 3-fold-by-clip split as everywhere else, just restricted to one subject
so training happens on ~450 seconds across 12ish clips (train) instead of the
usual ~30x that. This reuses run_per_subject() from refed_montage_regress.py,
which already loops per subject -- we just filter the data to one subject
first.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" single_subject_regress.py --subject 9
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_regress import (  # noqa: E402
    run_per_subject, ccc, DIMS, EXCLUDE_FROM_WHOLE,
)
from refed_montage_figures import INK, BG, GRID  # noqa: E402


class Args:
    epochs = 40
    lr = 1e-3
    batch_size = 64  # smaller batch -- one subject has far fewer samples
    dropout = 0.5
    weight_decay = 1e-4
    seed = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=9)
    ap.add_argument("--features", default=str(HERE / "refed_features_libeer_lds.npz"))
    a = ap.parse_args()

    z = np.load(a.features, allow_pickle=True)
    X, Y = z["X"], z["Y"]
    subject, video, second = z["subject"], z["video"], z["second"]
    channels = [str(c) for c in z["channels"]]
    whole = [c for c in channels if c not in EXCLUDE_FROM_WHOLE]
    idx = [channels.index(c) for c in whole]

    m = subject == a.subject
    Xm, Ym, sub_m, vid_m = X[m][:, idx, 0:5], Y[m], subject[m], video[m]
    print(f"subject {a.subject}: {m.sum()} samples across {len(np.unique(vid_m))} clips")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = run_per_subject(Xm, Ym, sub_m, vid_m, device, Args())

    print("\nper-clip valence/arousal CCC (this subject only):")
    for v in np.unique(vid_m):
        mv = vid_m == v
        if mv.sum() < 10:
            continue
        cv = ccc(Ym[mv, 0], pred[mv, 0])
        ca = ccc(Ym[mv, 1], pred[mv, 1])
        print(f"  clip {v:2d}: valence CCC={cv:+.3f}  arousal CCC={ca:+.3f}  n={mv.sum()}")

    overall_v = ccc(Ym[:, 0], pred[:, 0])
    overall_a = ccc(Ym[:, 1], pred[:, 1])
    print(f"\noverall (all clips pooled): valence CCC={overall_v:+.3f}  arousal CCC={overall_a:+.3f}")

    # timeline plot, all clips for this subject
    plt.rcParams.update({
        "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": BG, "axes.facecolor": BG,
    })
    vids = sorted(np.unique(vid_m))
    fig, axes = plt.subplots(len(vids), 2, figsize=(11, 2.6 * len(vids)))
    for r, v in enumerate(vids):
        mv = vid_m == v
        order = np.argsort(second[m][mv])
        t = second[m][mv][order]
        for c_i, dim in enumerate(DIMS):
            ax = axes[r][c_i]
            ax.plot(t, Ym[mv][order][:, c_i], color=INK, lw=2.2, label="true")
            ax.plot(t, pred[mv][order][:, c_i], color="#2f5f5c", lw=1.4,
                    alpha=0.9, label="single-subject model")
            ax.axhline(0, color=GRID, lw=0.8, ls=":")
            ax.set_ylim(-1.05, 1.05)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(dim, fontsize=11)
            if c_i == 0:
                ax.set_ylabel(f"clip {v}", fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
                      bbox_to_anchor=(1.05, 1.5))
    fig.suptitle(f"Subject {a.subject} ONLY -- trained/tested on their own data, "
                 f"whole cap, LibEER+LDS\noverall CCC: valence {overall_v:+.3f}, "
                 f"arousal {overall_a:+.3f}", fontsize=12, y=1.0)
    fig.tight_layout()
    out = HERE / f"fig_single_subject_{a.subject}.png"
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("Saved ->", out)


if __name__ == "__main__":
    main()
