"""Best-case trials (top single-trial valence CCC), whole-cap only, LibEER
no-smoothing vs LibEER+non-causal LDS overlaid -- the ceiling of what the
model can do, NOT representative. Across all 452 (subject, clip) trials the
mean single-trial valence CCC is only +0.007 (median 0.000); these are the
top of that distribution.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" fig_timeline_best.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_regress import DIMS, ccc  # noqa: E402
from refed_montage_figures import INK, BG, GRID  # noqa: E402

plt.rcParams.update({
    "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
})

COL_NOSMOOTH = "#c1443f"
COL_LDS = "#2f5f5c"
N_TOP = 6


def main():
    pz_lds = np.load(HERE / "libeer_lds_all_pred.npz")
    pz_raw = np.load(HERE / "libeer_montage_pred.npz")
    Y, subject, video = pz_lds["Y"], pz_lds["subject"], pz_lds["video"]
    pw = pz_lds["pred_pooled_whole"]

    rows = []
    for s in np.unique(subject):
        for v in np.unique(video):
            m = (subject == s) & (video == v)
            if m.sum() < 40:
                continue
            rows.append((ccc(Y[m][:, 0], pw[m][:, 0]), int(s), int(v)))
    rows.sort(key=lambda r: -r[0])
    all_cccs = [r[0] for r in rows]
    trials = rows[:N_TOP]

    n = len(trials)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.9 * n))
    for r, (cv, sub, vid) in enumerate(trials):
        for c_i, dim in enumerate(DIMS):
            ax = axes[r][c_i]
            for pz, col, lab in ((pz_raw, COL_NOSMOOTH, "LibEER no-smooth"),
                                 (pz_lds, COL_LDS, "LibEER+non-causal LDS")):
                sj, vd, sc = pz["subject"], pz["video"], pz["second"]
                m = (sj == sub) & (vd == vid)
                order = np.argsort(sc[m])
                t = sc[m][order]
                if col == COL_NOSMOOTH:
                    ax.plot(t, pz["Y"][m][order][:, c_i], color=INK, lw=2.4,
                            label="true (joystick)", zorder=5)
                p = pz["pred_pooled_whole"][m][order][:, c_i]
                ax.plot(t, p, color=col, lw=1.4, alpha=0.9, label=lab)
            ax.axhline(0, color=GRID, lw=0.8, ls=":")
            ax.set_ylim(-1.05, 1.05)
            ax.set_facecolor(BG)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(dim, fontsize=11)
            if c_i == 0:
                ax.set_ylabel(f"sub {sub}, clip {vid}\nvalence CCC={cv:+.3f}",
                              fontsize=9)
            if r == n - 1:
                ax.set_xlabel("Time within clip (s)")
    axes[0][0].legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
                      bbox_to_anchor=(1.05, 1.55))
    fig.suptitle(f"Best-case trials only (top {N_TOP} of 452 by single-trial "
                 f"valence CCC) -- whole cap, NOT representative\n"
                 f"mean single-trial valence CCC across all trials = "
                 f"{np.mean(all_cccs):+.3f}, median = {np.median(all_cccs):+.3f}",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    out = HERE / "fig_timeline_best_wholecap.png"
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("Saved ->", out)


if __name__ == "__main__":
    main()
