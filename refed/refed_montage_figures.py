"""Figures for the REFED four-montage DGCNN comparison.

Produces:
  fig_montages.png    where each montage's electrodes sit on the scalp
  fig_performance.png CCC and 3-class accuracy per montage, both training modes
  fig_timeline.png    true vs predicted valence/arousal over one clip, all montages
  fig_persubject.png  per-subject CCC spread, showing how much variance there is

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" refed_montage_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_regress import (  # noqa: E402
    MONTAGES, EXCLUDE_FROM_WHOLE, DIMS, ccc, to_class,
)

INK = "#1c2320"
BG = "#f6f2ea"
GRID = "#d8d0bf"
MONT_COLORS = {"whole": "#2f5f5c", "quick20": "#b8873a",
               "headtop": "#6a4c93", "glasses": "#c1443f"}
MONT_ORDER = ["whole", "quick20", "headtop", "glasses"]

plt.rcParams.update({
    "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
})


def electrode_xy(channels):
    """2D scalp positions for REFED's 64 channels.

    Uses an azimuthal-equidistant projection of MNE's standard_1005 3D
    coordinates (angle from vertex -> radius), which is what EEG topomaps use.
    Simply dropping z would bunch the rim electrodes toward the centre.
    """
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    up = {k.upper(): v for k, v in pos.items()}
    xy = {}
    for ch in channels:
        if ch not in up:
            continue
        x, y, z = up[ch]
        norm = np.linalg.norm([x, y, z])
        if norm < 1e-9:
            xy[ch] = (0.0, 0.0)
            continue
        theta = np.arccos(np.clip(z / norm, -1, 1))  # 0 at vertex
        phi = np.arctan2(y, x)
        r = theta / (np.pi / 2)
        xy[ch] = (r * np.cos(phi), r * np.sin(phi))
    # CB1/CB2 are cerebellar and absent from the standard montage; place them
    # just below and lateral to O1/O2.
    for cb, o, sign in (("CB1", "O1", -1), ("CB2", "O2", 1)):
        if cb in channels and o in xy:
            x, y = xy[o]
            xy[cb] = (x + sign * 0.10, y - 0.12)
    return xy


def draw_head(ax, xy, active, color, title, subtitle):
    r = max(np.hypot(x, y) for x, y in xy.values()) * 1.08
    ax.add_patch(plt.Circle((0, 0), r, fill=False, color=INK, lw=1.4))
    ax.plot([0, -r * 0.11, 0, r * 0.11, 0],
            [r, r * 1.13, r * 1.20, r * 1.13, r], color=INK, lw=1.4)  # nose
    for s in (-1, 1):
        ax.add_patch(plt.matplotlib.patches.Ellipse(
            (s * r, 0), r * 0.10, r * 0.32, fill=False, color=INK, lw=1.4))

    # 62 labels would collide into an unreadable mass; only label the sparse
    # montages, where knowing the exact electrode is the point.
    show_labels = len(active) <= 25
    for ch, (x, y) in xy.items():
        on = ch in active
        ax.scatter(x, y, s=52 if on else 14,
                   c=color if on else "none",
                   edgecolors=color if on else GRID,
                   linewidths=1.1, zorder=3 if on else 2)
        if on and show_labels:
            ax.text(x, y - r * 0.045, ch, ha="center", va="top", fontsize=6.0,
                    color=INK, zorder=4)
    ax.set_title(title, fontsize=11, color=color, pad=2)
    ax.text(0.5, -0.04, subtitle, transform=ax.transAxes, ha="center",
            fontsize=8.5, color="#55605a")
    ax.set_xlim(-r * 1.3, r * 1.3)
    ax.set_ylim(-r * 1.3, r * 1.35)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_montages(channels, montages, out):
    xy = electrode_xy(channels)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
    for ax, name in zip(axes, MONT_ORDER):
        chs = montages[name]
        draw_head(ax, xy, set(chs), MONT_COLORS[name], name,
                  f"{len(chs)} electrodes")
    fig.suptitle("Four electrode montages compared on REFED", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def _dim_legend(ax, label_v="valence", label_a="arousal"):
    """Bars are coloured by montage; the dimension is encoded by fill opacity,
    so the legend needs neutral swatches rather than the first two bar handles."""
    handles = [plt.matplotlib.patches.Patch(facecolor="#55605a", alpha=1.0,
                                            edgecolor=INK, label=label_v),
               plt.matplotlib.patches.Patch(facecolor="#55605a", alpha=0.45,
                                            edgecolor=INK, label=label_a)]
    ax.legend(handles=handles, frameon=False, fontsize=8.5)


def _grouped_bars(ax, res_for_mode, getter, order=MONT_ORDER, width=0.36):
    x = np.arange(len(order))
    for d_i, dim in enumerate(DIMS):
        vals = [getter(res_for_mode[m], dim)[0] for m in order]
        errs = [getter(res_for_mode[m], dim)[1] for m in order]
        ax.bar(x + (d_i - 0.5) * width, vals, width, yerr=errs, capsize=3,
               color=[MONT_COLORS[m] for m in order],
               alpha=1.0 if d_i == 0 else 0.45,
               edgecolor=INK, linewidth=0.6,
               error_kw=dict(lw=0.9, ecolor="#55605a"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{res_for_mode[m]['n_channels']}ch" for m in order],
                       fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_performance(res, out):
    modes = list(res)
    fig, axes = plt.subplots(1, len(modes), figsize=(6.4 * len(modes), 4.6),
                             squeeze=False)
    for c_i, mode in enumerate(modes):
        ax = axes[0][c_i]
        _grouped_bars(ax, res[mode], lambda r, d: (
            np.mean(r["per_subject"][d]["ccc"]), np.std(r["per_subject"][d]["ccc"])))
        ax.axhline(0, color=INK, lw=1.0)
        ax.set_ylabel("CCC   (0 = no better than predicting the mean)", fontsize=9)
        ax.set_title(f"{mode}", fontsize=11)
        _dim_legend(ax)
    fig.suptitle("Continuous-emotion agreement by electrode montage\n"
                 "error bars = sd across the 32 subjects", fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def fig_accuracy(res_cls, majority, out):
    """Accuracy from models actually trained to classify -- not the pessimistic
    numbers you get by thresholding a regressor's shrunken output."""
    modes = list(res_cls)
    fig, axes = plt.subplots(2, len(modes), figsize=(6.4 * len(modes), 8.0),
                             squeeze=False)
    for c_i, mode in enumerate(modes):
        for r_i, metric in enumerate(["acc", "macro"]):
            ax = axes[r_i][c_i]
            _grouped_bars(ax, res_cls[mode], lambda r, d: (
                np.mean(r["per_subject"][d][metric]),
                np.std(r["per_subject"][d][metric])))
            if metric == "acc":
                ax.set_ylabel("3-class accuracy", fontsize=9)
                ax.set_title(f"{mode} — accuracy", fontsize=11)
                for dim, style in zip(DIMS, ["--", ":"]):
                    ax.axhline(majority[dim], color="#c1443f", lw=1.1, ls=style)
                    ax.text(0.012, majority[dim], f"majority baseline ({dim})",
                            fontsize=7.2, color="#c1443f", va="bottom", ha="left",
                            transform=ax.get_yaxis_transform())
            else:
                ax.set_ylabel("macro recall (mean per-class)", fontsize=9)
                ax.set_title(f"{mode} — macro recall", fontsize=11)
                ax.axhline(1 / 3, color="#c1443f", lw=1.1, ls="--")
                ax.text(0.012, 1 / 3, "chance (1/3)", fontsize=7.2,
                        color="#c1443f", va="bottom", ha="left",
                        transform=ax.get_yaxis_transform())
            _dim_legend(ax)
    fig.suptitle("3-class accuracy by electrode montage\n"
                 "accuracy must be read against the majority baseline; "
                 "macro recall is the fairer number", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def fig_timeline(pz, mode, out, sub=None, vid=None):
    Y, subject, video, second = pz["Y"], pz["subject"], pz["video"], pz["second"]
    # Pick the subject/clip where the full-cap model tracks valence best, so the
    # figure shows what a good case actually looks like.
    if sub is None:
        best, key = -9, None
        pw = pz[f"pred_{mode}_whole"]
        for s in np.unique(subject):
            for v in np.unique(video):
                m = (subject == s) & (video == v)
                if m.sum() < 40:
                    continue
                c = ccc(Y[m][:, 0], pw[m][:, 0])
                if c > best:
                    best, key = c, (s, v)
        sub, vid = key
    m = (subject == sub) & (video == vid)
    order = np.argsort(second[m])

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    for ax, d_i, dim in zip(axes, range(2), DIMS):
        t = second[m][order]
        ax.plot(t, Y[m][order][:, d_i], color=INK, lw=2.4,
                label="true (joystick)", zorder=5)
        for name in MONT_ORDER:
            p = pz[f"pred_{mode}_{name}"][m][order][:, d_i]
            ax.plot(t, p, color=MONT_COLORS[name], lw=1.3, alpha=0.9,
                    label=f"{name}")
        ax.axhline(0, color=GRID, lw=0.8, ls=":")
        ax.set_ylabel(dim, fontsize=10)
        ax.set_facecolor(BG)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=5, loc="upper center",
                   bbox_to_anchor=(0.5, 1.28))
    axes[-1].set_xlabel("Time within clip (s)")
    fig.suptitle(f"Predicted vs actual emotion trajectory — subject {sub}, "
                 f"clip {vid} ({mode})", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return int(sub), int(vid)


def fig_persubject(res, out):
    modes = list(res)
    fig, axes = plt.subplots(len(modes), 2, figsize=(12, 4.2 * len(modes)),
                             squeeze=False)
    for r_i, mode in enumerate(modes):
        for c_i, dim in enumerate(DIMS):
            ax = axes[r_i][c_i]
            data = [res[mode][m]["per_subject"][dim]["ccc"] for m in MONT_ORDER]
            parts = ax.violinplot(data, showmeans=True, showextrema=False)
            for pc, m in zip(parts["bodies"], MONT_ORDER):
                pc.set_facecolor(MONT_COLORS[m])
                pc.set_alpha(0.45)
            parts["cmeans"].set_color(INK)
            for i, (vals, m) in enumerate(zip(data, MONT_ORDER)):
                jitter = np.random.RandomState(0).normal(0, 0.035, len(vals))
                ax.scatter(np.full(len(vals), i + 1) + jitter, vals, s=11,
                           color=MONT_COLORS[m], alpha=0.75, zorder=3)
            ax.axhline(0, color=INK, lw=1.0)
            ax.set_xticks(range(1, len(MONT_ORDER) + 1))
            ax.set_xticklabels(MONT_ORDER, fontsize=9)
            ax.set_ylabel("CCC per subject", fontsize=9)
            ax.set_title(f"{mode} — {dim}", fontsize=11)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.suptitle("Every subject, not just the average", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main():
    blob = json.loads((HERE / "montage_results.json").read_text())
    res = blob["results"]
    montages = blob["montage_channels"]
    pz = np.load(HERE / "montage_predictions.npz")
    channels = montages["whole"] + EXCLUDE_FROM_WHOLE

    fig_montages(channels, montages, HERE / "fig_montages.png")
    fig_performance(res, HERE / "fig_performance.png")
    fig_persubject(res, HERE / "fig_persubject.png")
    mode = "pooled" if "pooled" in res else list(res)[0]
    s, v = fig_timeline(pz, mode, HERE / "fig_timeline.png")
    print(f"figures written (timeline uses subject {s}, clip {v}, mode {mode})")

    cls_path = HERE / "montage_results_cls.json"
    if cls_path.exists():
        cls = json.loads(cls_path.read_text())
        # Class labels come from the same thresholding as the classifier uses,
        # so they can be derived from the regression targets -- the classifier's
        # own npz is only written once its whole run finishes.
        C = np.stack([to_class(pz["Y"][:, 0]), to_class(pz["Y"][:, 1])], axis=1)
        majority = {d: float(np.bincount(C[:, i], minlength=3).max() / len(C))
                    for i, d in enumerate(DIMS)}
        # Only plot modes that finished every montage.
        done = {m: r for m, r in cls["results"].items()
                if all(n in r for n in MONT_ORDER)}
        if done:
            fig_accuracy(done, majority, HERE / "fig_accuracy.png")
            print(f"fig_accuracy.png written for modes: {list(done)}")
        else:
            print("classification run incomplete, skipping fig_accuracy")


if __name__ == "__main__":
    main()
