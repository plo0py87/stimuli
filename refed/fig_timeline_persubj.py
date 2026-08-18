"""Same as fig_timeline_lds.py, but using per-subject (in-session) trained
predictions -- pred_per_subject_* from libeer_lds_persubj_pred.npz -- instead
of the pooled-across-32-subjects predictions. This is the trajectory figure
that matches the report's corrected per-subject headline numbers.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" fig_timeline_persubj.py
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

MONT_ORDER = ["whole", "vr_ring", "quick20", "headtop"]
MONT_COLORS = {"whole": "#2f5f5c", "vr_ring": "#c1443f",
               "quick20": "#b8873a", "headtop": "#6a4c93"}
MONT_LABEL = {"whole": "whole cap", "vr_ring": "glasses", "quick20": "Quick-20",
              "headtop": "head-top"}

plt.rcParams.update({
    "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
})


def pick_median_trial(pz, video):
    Y, subject, vid = pz["Y"], pz["subject"], pz["video"]
    pw = pz["pred_per_subject_whole"]
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
    pz = np.load(HERE / "libeer_lds_persubj_pred.npz")
    Y, subject, video, second = pz["Y"], pz["subject"], pz["video"], pz["second"]

    trials = []
    for target, vids in TARGET_GROUPS.items():
        v = vids[0]
        sub, c = pick_median_trial(pz, v)
        trials.append((target, sub, v, c))
        print(f"{target}: clip {v}, subject {sub} (median whole-cap valence CCC={c:+.3f})")

    n = len(trials)
    fig, axes = plt.subplots(n, 2, figsize=(12.5, 2.9 * n), sharex=False)
    for r, (target, sub, vid, c) in enumerate(trials):
        m = (subject == sub) & (video == vid)
        order = np.argsort(second[m])
        t = second[m][order]
        for c_i, dim in enumerate(DIMS):
            ax = axes[r][c_i]
            ax.plot(t, Y[m][order][:, c_i], color=INK, lw=2.4,
                    label="true (joystick)", zorder=5)
            for name in MONT_ORDER:
                p = pz[f"pred_per_subject_{name}"][m][order][:, c_i]
                ax.plot(t, p, color=MONT_COLORS[name], lw=1.2, alpha=0.9,
                        label=MONT_LABEL[name])
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
    axes[0][0].legend(frameon=False, fontsize=8, ncol=5, loc="upper center",
                      bbox_to_anchor=(1.05, 1.45))
    fig.suptitle("Per-subject (in-session) training: predicted vs actual, one trial "
                 "per emotion target (median-performance subject, not cherry-picked)",
                 fontsize=12.5, y=1.0)
    fig.tight_layout()
    out = HERE / "fig_timeline_persubj_grid.png"
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("Saved ->", out)


if __name__ == "__main__":
    main()
