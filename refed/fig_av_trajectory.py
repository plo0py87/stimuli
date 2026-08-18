"""Per-session ground-truth AV trajectories in the 2D valence-arousal plane
(the circumplex view), not V and A plotted separately over time.

For each subject (= one REFED session), draw all 15 clips' joystick paths as
lines in (valence, arousal) space, colored PER TRIAL (15 distinct colors) --
not by the 5-way emotion target, since 3 trials share each target and
grouping them by color hid trial-to-trial variation within the same target.
Each line is drawn as a light-to-dark gradient (time progression) with a
circle at trial start and a square at trial end -- NOT an arrowhead. An
arrowhead computed from the last two samples degenerates to nothing when
those two samples are numerically identical (common: several REFED trials
have the joystick frozen at the same value for the final 1-2 seconds), which
made some trials silently lose their direction marker. Start/end markers
never disappear, even when the last two samples coincide.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" fig_av_trajectory.py
"""
import colorsys
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_regress import TARGET_GROUPS  # noqa: E402
from refed_montage_figures import INK, BG, GRID  # noqa: E402

SUBJECTS = list(range(1, 33))
GROUP_SIZE = 8
VIDEO_TARGET = {v: t for t, vids in TARGET_GROUPS.items() for v in vids}
VIDEO_COLORS = {
    v: mcolors.to_hex(colorsys.hsv_to_rgb((i / 15), 0.50, 0.62))
    for i, v in enumerate(range(1, 16))
}

plt.rcParams.update({
    "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
})


def plot_trajectory(ax, v, a, base_color, start_size=26, end_size=42, lw=1.4):
    """Gradient line (light=start, full color=end) + explicit start (circle)
    and end (square) markers. Robust to the last two points coinciding.

    Two fixes over the first version: (1) the light end of the gradient
    blended almost to white, which nearly vanished against the cream page
    background right at trial start; now it only blends 45% toward white so
    the color stays legible throughout. (2) LineCollection segments were
    drawn with butt caps/joins, which leaves visible gaps at sharp turns in a
    noisy trajectory -- looks like the line breaks. Round caps/joins fix that,
    plus a constant-color low-alpha line underneath guarantees the path is
    never invisible even at the lightest point of the gradient."""
    if len(v) < 2:
        ax.scatter(v, a, color=base_color, s=start_size, zorder=4)
        return
    pts = np.array([v, a]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    ax.plot(v, a, color=base_color, lw=lw, alpha=0.30, zorder=2,
            solid_capstyle="round", solid_joinstyle="round")

    rgb = mcolors.to_rgb(base_color)
    light = tuple(0.55 * c + 0.45 for c in rgb)  # 45% toward white, not ~100%
    cmap = mcolors.LinearSegmentedColormap.from_list("grad", [light, base_color])
    t = np.linspace(0.0, 1.0, len(segs))
    lc = LineCollection(segs, colors=[cmap(ti) for ti in t], linewidths=lw, zorder=3,
                        capstyle="round", joinstyle="round")
    ax.add_collection(lc)
    ax.scatter(v[0], a[0], color=base_color, s=start_size, zorder=4, marker="o",
              edgecolors=INK, linewidths=0.5)
    ax.scatter(v[-1], a[-1], color=base_color, s=end_size, zorder=5, marker="s",
              edgecolors=INK, linewidths=0.6)


def plot_group(z, subs, group_idx, n_groups):
    Y, subject, video, second = z["Y"], z["subject"], z["video"], z["second"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for ax, sub in zip(axes.flat, subs):
        ms = subject == sub
        for vid in np.unique(video[ms]):
            mv = ms & (video == vid)
            order = np.argsort(second[mv])
            v, a = Y[mv][order, 0], Y[mv][order, 1]
            plot_trajectory(ax, v, a, VIDEO_COLORS[int(vid)])
        ax.axhline(0, color=GRID, lw=0.9)
        ax.axvline(0, color=GRID, lw=0.9)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(f"subject {sub}", fontsize=11)
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])

    for ax in axes[-1]:
        ax.set_xlabel("valence")
    for ax in axes[:, 0]:
        ax.set_ylabel("arousal")

    handles = [plt.Line2D([0], [0], color=VIDEO_COLORS[v], lw=2,
                         label=f"{v} ({VIDEO_TARGET[v]})") for v in range(1, 16)]
    handles += [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                  markeredgecolor=INK, markersize=7, label="trial start"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="none",
                  markeredgecolor=INK, markersize=7, label="trial end"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=9, frameon=False,
              fontsize=8.5, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"Ground-truth joystick AV trajectories, subjects {subs[0]}-{subs[-1]} "
                 f"(group {group_idx}/{n_groups} of all 32), all 15 clips each\n"
                 "line darkens over time (light=trial start), circle=start, "
                 "square=end, color=clip/trial number (legend: clip (target))",
                 fontsize=12, y=1.14)
    fig.tight_layout()
    out = HERE / f"fig_av_trajectory_g{group_idx}.png"
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("Saved ->", out)


def main():
    z = np.load(HERE / "refed_features_libeer_lds.npz", allow_pickle=True)
    groups = [SUBJECTS[i:i + GROUP_SIZE] for i in range(0, len(SUBJECTS), GROUP_SIZE)]
    for i, subs in enumerate(groups, start=1):
        plot_group(z, subs, i, len(groups))


if __name__ == "__main__":
    main()
