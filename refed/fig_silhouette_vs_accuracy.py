"""Does higher per-subject silhouette (DE_LDS, in-session) actually predict
higher per-subject classification accuracy? Direct answer to: "if it looks
separable in t-SNE after LDS, why is accuracy still weak?"

Reads the per-subject silhouette numbers already computed by
silhouette_perperson_lds.py (parsed from its saved log) and the per-subject
(in-session) whole-cap valence accuracy from libeer_lds_persubj_cls.json.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" fig_silhouette_vs_accuracy.py
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_figures import INK, BG, GRID  # noqa: E402

plt.rcParams.update({
    "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
})


def main():
    log = (HERE / "silhouette_perperson_lds.log").read_text()
    refed_block = log.split("REFED, 32")[1]
    sil = {int(m.group(1)): float(m.group(2))
          for m in re.finditer(r"subject\s+(\d+): silhouette=([+-]\d+\.\d+)", refed_block)}

    cls = json.loads((HERE / "libeer_lds_persubj_cls.json").read_text())
    acc_list = cls["results"]["per_subject"]["whole"]["per_subject"]["valence"]["acc"]
    acc = {i + 1: acc_list[i] for i in range(len(acc_list))}

    subs = sorted(sil)
    s = np.array([sil[i] for i in subs])
    a = np.array([acc[i] for i in subs])
    r = np.corrcoef(s, a)[0, 1]
    b, m = np.polyfit(s, a, 1)[::-1]  # a = m*s + b -> polyfit returns [slope, intercept]
    slope, intercept = np.polyfit(s, a, 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(s, a, s=55, color="#2f5f5c", edgecolors=INK, linewidths=0.6, zorder=3)
    xs = np.linspace(s.min(), s.max(), 50)
    ax.plot(xs, slope * xs + intercept, color="#c1443f", lw=1.6, ls="--", zorder=2,
           label=f"linear fit (r={r:+.3f})")
    ax.axhline(0.473, color=GRID, lw=1.0, ls=":")
    ax.text(s.max(), 0.473, " valence majority baseline", va="center", ha="right",
           fontsize=8.5, color=INK)

    highlight = {9: "sub 9\n(t-SNE example figure)", 15: "sub 15\n(highest silhouette)",
                29: "sub 29", 26: "sub 26"}
    for i, lab in highlight.items():
        ax.annotate(lab, (sil[i], acc[i]), textcoords="offset points",
                   xytext=(8, 6), fontsize=8.5, color=INK)
        ax.scatter([sil[i]], [acc[i]], s=80, facecolors="none",
                  edgecolors="#c1443f", linewidths=1.4, zorder=4)

    ax.set_xlabel("per-subject silhouette, DE_LDS, in-session (high-dim feature space)")
    ax.set_ylabel("per-subject 3-class valence accuracy (per-subject trained DGCNN)")
    ax.set_title("Silhouette only weakly predicts classifier accuracy\n"
                 f"32 REFED subjects, Pearson r={r:+.3f} "
                 f"({r**2*100:.0f}% of accuracy variance explained)", fontsize=12.5)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    fig.tight_layout()
    out = HERE / "fig_silhouette_vs_accuracy.png"
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"r={r:+.3f}  n={len(subs)}")
    print("Saved ->", out)


if __name__ == "__main__":
    main()
