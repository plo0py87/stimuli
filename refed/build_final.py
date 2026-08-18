"""Consolidated figures + report for the nine-montage REFED study.

Pulls together three preprocessing variants and nine electrode montages:

  causal        our realtime-legal pipeline: causal DE (lfilter) + forward-only
                Kalman, no global bandpass          -> montage_results / sweeps
  libeer        LibEER's exact SEED recipe: 0.3-50 Hz filtfilt bandpass, per-band
                filtfilt DE, log2/ddof=1, NO smoothing  -> libeer_montage.json
  libeer+kalman the same LibEER features with the causal Kalman smoother added
                back                                 -> libeer_kalman_all.json

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" build_final.py
"""

import base64
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_regress import MONTAGES, EXCLUDE_FROM_WHOLE, DIMS  # noqa: E402
from refed_montage_figures import electrode_xy, draw_head, INK, BG, GRID  # noqa: E402

ORDER = ["whole", "vr_ring", "quick20", "headtop"]
LABEL = {"whole": "whole cap", "vr_ring": "glasses", "quick20": "Quick-20",
         "headtop": "head-top band"}
COLORS = {"whole": "#2f5f5c", "vr_ring": "#c1443f",
          "quick20": "#b8873a", "headtop": "#6a4c93"}

plt.rcParams.update({
    "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG})


def load(path, key="results"):
    p = HERE / path
    return json.loads(p.read_text())[key] if p.exists() else {}


def collect_causal():
    """The causal arm is spread over three runs (all used the same seed)."""
    out = {}
    for f in ("montage_results.json", "glasses_sweep.json", "vr_sweep.json"):
        r = load(f)
        if "pooled" in r:
            out.update(r["pooled"])
    return out


def mean_ccc(entry, dim):
    return float(np.mean(entry["per_subject"][dim]["ccc"]))


def sd_ccc(entry, dim):
    return float(np.std(entry["per_subject"][dim]["ccc"]))


def fig_all_montages(montages, out):
    channels = montages["whole"] + EXCLUDE_FROM_WHOLE
    xy = electrode_xy(channels)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    for ax, name in zip(axes.flat, ORDER):
        chs = montages[name]
        draw_head(ax, xy, set(chs), COLORS[name], LABEL[name],
                  f"{len(chs)} electrodes")
    fig.suptitle("Four electrode montages", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def fig_ranking(causal, out):
    """Horizontal ranking, sorted by valence CCC, with the whole-cap reference."""
    names = sorted(causal, key=lambda m: -mean_ccc(causal[m], "valence"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    y = np.arange(len(names))
    ref = mean_ccc(causal["whole"], "valence")

    for ax, dim in zip(axes, DIMS):
        vals = [mean_ccc(causal[m], dim) for m in names]
        errs = [sd_ccc(causal[m], dim) for m in names]
        ax.barh(y, vals, xerr=errs, color=[COLORS[m] for m in names],
                edgecolor=INK, linewidth=0.6, height=0.68,
                error_kw=dict(lw=0.9, ecolor="#55605a"))
        for i, (m, v) in enumerate(zip(names, vals)):
            ax.text(v + errs[i] + 0.006, i, f"{v:+.3f}", va="center",
                    fontsize=8.5, color=INK)
        ax.axvline(0, color=INK, lw=1.0)
        ax.set_xlabel(f"{dim} CCC")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{LABEL[m]}  ({len(m)})" if False else
                             f"{LABEL[m]}" for m in names], fontsize=10)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.set_xlim(left=min(0, ax.get_xlim()[0]))
    fig.suptitle("Montage ranking — causal (realtime-legal) pipeline, pooled\n"
                 f"whole-cap valence reference = {ref:+.3f}", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def fig_preproc(causal, libeer, lib_kal, out):
    names = [m for m in ORDER if m in libeer and m in lib_kal and m in causal]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    x = np.arange(len(names))
    w = 0.27
    series = [("causal (realtime-legal)", causal, "#2f5f5c"),
              ("LibEER exact (no smoothing)", libeer, "#c1443f"),
              ("LibEER + Kalman", lib_kal, "#b8873a")]
    for ax, dim in zip(axes, DIMS):
        for i, (lab, data, col) in enumerate(series):
            vals = [mean_ccc(data[m], dim) for m in names]
            ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col,
                   edgecolor=INK, linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[m] for m in names], rotation=35,
                           ha="right", fontsize=8.5)
        ax.axhline(0, color=INK, lw=1.0)
        ax.set_ylabel(f"{dim} CCC")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Preprocessing matters more than the LibEER recipe: it is the "
                 "temporal smoother, not the bandpass\n"
                 "removing the Kalman smoother collapses every montage; adding "
                 "it back recovers almost all of it", fontsize=11.5, y=1.05)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main():
    causal = collect_causal()
    libeer = load("libeer_montage.json").get("pooled", {})
    lib_kal = load("libeer_kalman_all.json").get("pooled", {})

    blob = json.loads((HERE / "libeer_kalman_all.json").read_text())
    montages = blob["montage_channels"]

    fig_all_montages(montages, HERE / "fig_montages_all.png")
    fig_ranking(causal, HERE / "fig_ranking.png")
    fig_preproc(causal, libeer, lib_kal, HERE / "fig_preproc.png")
    print("figures:", [p.name for p in HERE.glob("fig_*.png")])

    # Console summary table
    print(f"\n{'montage':18s} {'ch':>3s} | {'causal V':>9s} {'causal A':>9s} | "
          f"{'libeer V':>9s} | {'lib+kal V':>10s}")
    print("-" * 72)
    for m in ORDER:
        if m not in causal:
            continue
        n = len(montages[m])
        lv = f"{mean_ccc(libeer[m], 'valence'):+.3f}" if m in libeer else "  n/a"
        kv = f"{mean_ccc(lib_kal[m], 'valence'):+.3f}" if m in lib_kal else "  n/a"
        print(f"{LABEL[m]:18s} {n:3d} | {mean_ccc(causal[m],'valence'):+9.3f} "
              f"{mean_ccc(causal[m],'arousal'):+9.3f} | {lv:>9s} | {kv:>10s}")


if __name__ == "__main__":
    main()
