"""Whole-cap only: same trials as fig_timeline_lds.py, LibEER-plain-DE (no
smoothing) vs LibEER+non-causal-LDS predictions overlaid on the same axes,
plus a few extra trials for a broader look.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" fig_timeline_compare.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_regress import DIMS, TARGET_GROUPS, ccc  # noqa: E402
from refed_montage_figures import INK, BG, GRID  # noqa: E402

plt.rcParams.update({
    "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
})

COL_NOSMOOTH = "#c1443f"
COL_LDS = "#2f5f5c"


def pick_median_trial(pz, video):
    Y, subject, vid = pz["Y"], pz["subject"], pz["video"]
    pw = pz["pred_pooled_whole"]
    ccs, subs = [], []
    for s in np.unique(subject):
        m = (subject == s) & (vid == video)
        if m.sum() < 40:
            continue
        ccs.append(ccc(Y[m][:, 0], pw[m][:, 0]))
        subs.append(s)
    ccs, subs = np.array(ccs), np.array(subs)
    order = np.argsort(ccs)
    median_i = order[len(order) // 2]
    return int(subs[median_i]), float(ccs[median_i])


def main():
    pz_lds = np.load(HERE / "libeer_lds_all_pred.npz")
    pz_raw = np.load(HERE / "libeer_montage_pred.npz")  # plain LibEER DE, no smoothing

    # Same 5 trials as before (one per target, first clip in each group),
    # plus 3 more from the second clip in three of the groups, for variety.
    trials = []
    for target, vids in TARGET_GROUPS.items():
        sub, c = pick_median_trial(pz_lds, vids[0])
        trials.append((target, sub, vids[0]))
    for target in ["MVMA", "HVHA", "HVLA"]:
        vid = TARGET_GROUPS[target][1]
        sub, c = pick_median_trial(pz_lds, vid)
        trials.append((target + "*", sub, vid))

    n = len(trials)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.9 * n))
    for r, (target, sub, vid) in enumerate(trials):
        for c_i, dim in enumerate(DIMS):
            ax = axes[r][c_i]
            for pz, col, lab in ((pz_raw, COL_NOSMOOTH, "LibEER no-smooth"),
                                 (pz_lds, COL_LDS, "LibEER+non-causal LDS")):
                subject, video, second = pz["subject"], pz["video"], pz["second"]
                m = (subject == sub) & (video == vid)
                order = np.argsort(second[m])
                t = second[m][order]
                if col == COL_NOSMOOTH:
                    ax.plot(t, pz["Y"][m][order][:, c_i], color=INK, lw=2.4,
                            label="true (joystick)", zorder=5)
                p = pz[f"pred_pooled_whole"][m][order][:, c_i]
                ax.plot(t, p, color=col, lw=1.4, alpha=0.9, label=lab)
            ax.axhline(0, color=GRID, lw=0.8, ls=":")
            ax.set_ylim(-1.05, 1.05)
            ax.set_facecolor(BG)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(dim, fontsize=11)
            if c_i == 0:
                ax.set_ylabel(f"{target}\nsub {sub}, clip {vid}", fontsize=9)
            if r == n - 1:
                ax.set_xlabel("Time within clip (s)")
    axes[0][0].legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
                      bbox_to_anchor=(1.05, 1.5))
    fig.suptitle("Whole-cap (62ch) only: LibEER no-smoothing vs LibEER+non-causal "
                 "LDS, same trials as before + 3 extra", fontsize=12.5, y=1.0)
    fig.tight_layout()
    out = HERE / "fig_timeline_compare_wholecap.png"
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("Saved ->", out)


if __name__ == "__main__":
    main()
