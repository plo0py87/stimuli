"""Per-subject (= per-session, REFED has one session each) classification
accuracy, both as a full per-subject listing and as a spread-across-montages
summary. Uses the per_subject TRAINING MODE (one DGCNN per subject, no other
subject's clips in training) -- the strict SEED-comparable convention -- not
the pooled mode reported elsewhere.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" fig_persubject_accuracy.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_figures import INK, BG, GRID  # noqa: E402

MONTAGES = ["whole", "vr_ring", "quick20", "headtop"]
LABEL = {"whole": "whole cap", "vr_ring": "glasses", "quick20": "Quick-20",
         "headtop": "head-top band"}
DIM_COLORS = {"valence": "#2f5f5c", "arousal": "#c1443f"}
BASELINE = {"valence": 0.7037, "arousal": 0.6409}  # majority-class, 3-class scheme differs;
# recomputed below from the confusion matrix instead of hardcoding.

plt.rcParams.update({
    "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
})


def majority_baseline(confusion):
    cm = np.array(confusion)
    return cm.sum(axis=1).max() / cm.sum() if cm.sum() else float("nan")
    # NOTE: per-dim majority baseline from the pooled confusion matrix (same
    # class balance whichever mode trained the model).


def fig_bars(d, out):
    """Per-subject bars, whole-cap montage, sorted by subject ID."""
    entry = d["results"]["per_subject"]["whole"]
    subs = list(range(1, len(entry["per_subject"]["valence"]["acc"]) + 1))
    v = entry["per_subject"]["valence"]["acc"]
    a = entry["per_subject"]["arousal"]["acc"]
    base_v = majority_baseline(entry["confusion"]["valence"]) if "confusion" in entry else None

    y = np.arange(len(subs))
    fig, ax = plt.subplots(figsize=(9, 10))
    h = 0.38
    ax.barh(y + h / 2, v, h, color=DIM_COLORS["valence"], label="valence",
           edgecolor=INK, linewidth=0.4)
    ax.barh(y - h / 2, a, h, color=DIM_COLORS["arousal"], label="arousal",
           edgecolor=INK, linewidth=0.4)
    ax.axvline(np.mean(v), color=DIM_COLORS["valence"], lw=1.2, ls="--", alpha=0.8)
    ax.axvline(np.mean(a), color=DIM_COLORS["arousal"], lw=1.2, ls="--", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"sub {s}" for s in subs], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("3-class accuracy")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(f"Per-subject accuracy, whole-cap montage, per-subject (in-session) "
                f"training\nmean valence={np.mean(v):.3f}, "
                f"mean arousal={np.mean(a):.3f}", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("Saved ->", out)


def fig_spread(d, out):
    """Box + per-subject strip, across the 4 montages."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    RNG = np.random.RandomState(0)
    for ax, dim in zip(axes, ("valence", "arousal")):
        data = [d["results"]["per_subject"][m]["per_subject"][dim]["acc"] for m in MONTAGES]
        bp = ax.boxplot(data, positions=range(len(MONTAGES)), widths=0.5,
                        patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set(facecolor=DIM_COLORS[dim], alpha=0.35, edgecolor=INK)
        for med in bp["medians"]:
            med.set(color=INK, lw=1.4)
        for i, vals in enumerate(data):
            jitter = RNG.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=14,
                      color=DIM_COLORS[dim], alpha=0.55, zorder=3)
        ax.set_xticks(range(len(MONTAGES)))
        ax.set_xticklabels([LABEL[m] for m in MONTAGES], rotation=25, ha="right")
        ax.set_title(dim, fontsize=11.5)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("3-class accuracy (per subject)")
    fig.suptitle("Per-subject accuracy spread across montages, per-subject "
                 "(in-session) training\neach dot = one subject's own DGCNN, "
                 "trained only on that subject's clips", fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("Saved ->", out)


def main():
    d = json.loads((HERE / "libeer_lds_persubj_cls.json").read_text())
    fig_bars(d, HERE / "fig_persubject_bars.png")
    fig_spread(d, HERE / "fig_persubject_spread.png")


if __name__ == "__main__":
    main()
