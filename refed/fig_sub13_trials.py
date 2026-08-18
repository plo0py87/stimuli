"""Subject 13 raw-data deep dive, one figure PER TRIAL (15 total, none
skipped) -- three panels each: valence vs time, arousal vs time, and the
same trial's path in the 2D valence-arousal plane. All raw joystick ground
truth, no model involved. Subject 13 was flagged in the AV-trajectory
overview as a likely data-quality outlier (many trials piling into the
valence=arousal=+1 corner); this lets you look at every trial individually
to judge that call.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" fig_sub13_trials.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_regress import TARGET_GROUPS  # noqa: E402
from refed_montage_figures import INK, BG, GRID  # noqa: E402
from fig_av_trajectory import plot_trajectory, VIDEO_COLORS  # noqa: E402

SUBJECT = 13
VIDEO_TARGET = {v: t for t, vids in TARGET_GROUPS.items() for v in vids}

plt.rcParams.update({
    "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
})


def main():
    z = np.load(HERE / "refed_features_libeer_lds.npz", allow_pickle=True)
    Y, subject, video, second = z["Y"], z["subject"], z["video"], z["second"]
    ms = subject == SUBJECT
    outputs = []

    for vid in sorted(np.unique(video[ms])):
        mv = ms & (video == vid)
        order = np.argsort(second[mv])
        t = second[mv][order]
        v, a = Y[mv][order, 0], Y[mv][order, 1]
        tgt = VIDEO_TARGET[int(vid)]
        col = VIDEO_COLORS[int(vid)]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                                 gridspec_kw={"width_ratios": [1.3, 1.3, 1]})

        ax = axes[0]
        ax.plot(t, v, color=col, lw=1.8)
        ax.axhline(0, color=GRID, lw=0.8, ls=":")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("valence")
        ax.set_title("valence vs time", fontsize=10.5)

        ax = axes[1]
        ax.plot(t, a, color=col, lw=1.8)
        ax.axhline(0, color=GRID, lw=0.8, ls=":")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("arousal")
        ax.set_title("arousal vs time", fontsize=10.5)

        ax = axes[2]
        plot_trajectory(ax, v, a, col, start_size=34, end_size=54, lw=1.8)
        ax.axhline(0, color=GRID, lw=0.9)
        ax.axvline(0, color=GRID, lw=0.9)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xlabel("valence")
        ax.set_ylabel("arousal")
        ax.set_title("AV plane (o=start, sq=end)", fontsize=10.5)

        for ax in axes[:2]:
            ax.set_facecolor(BG)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[2].set_facecolor(BG)
        for sp in axes[2].spines.values():
            sp.set_visible(False)

        fig.suptitle(f"Subject {SUBJECT}, clip {vid} (target: {tgt}), "
                     f"n={mv.sum()} seconds, duration {t.min():.0f}-{t.max():.0f}s",
                     fontsize=12, y=1.03)
        fig.tight_layout()
        out = HERE / f"fig_sub13_trial{int(vid):02d}.png"
        fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
        print("Saved ->", out)

    print(f"\n{len(outputs)} trial figures written.")


if __name__ == "__main__":
    main()
