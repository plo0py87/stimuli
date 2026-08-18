"""Individual (single-panel, small) AV trajectory plot per REFED subject, for
the interactive dashboard -- same per-trial coloring/rendering as
fig_av_trajectory.py's grouped grid, just one subject per image so the
dashboard can show exactly one person at a time without re-cropping a bigger
figure.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" dash_av_individual.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_figures import INK, BG, GRID  # noqa: E402
from fig_av_trajectory import plot_trajectory, VIDEO_COLORS, VIDEO_TARGET  # noqa: E402

plt.rcParams.update({
    "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
})


def main():
    z = np.load(HERE / "refed_features_libeer_lds.npz", allow_pickle=True)
    Y, subject, video, second = z["Y"], z["subject"], z["video"], z["second"]

    for sub in range(1, 33):
        ms = subject == sub
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        for vid in np.unique(video[ms]):
            mv = ms & (video == vid)
            order = np.argsort(second[mv])
            v, a = Y[mv][order, 0], Y[mv][order, 1]
            plot_trajectory(ax, v, a, VIDEO_COLORS[int(vid)], start_size=16,
                            end_size=26, lw=1.1)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.axvline(0, color=GRID, lw=0.8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        ax.set_xlabel("valence", fontsize=8)
        ax.set_ylabel("arousal", fontsize=8)
        fig.tight_layout()
        out = HERE / f"dash_av_sub{sub}.png"
        fig.savefig(out, dpi=110, facecolor=BG, bbox_inches="tight")
        plt.close(fig)
        print("Saved ->", out.name)

    print(f"\n32 individual AV trajectory images written.")


if __name__ == "__main__":
    main()
