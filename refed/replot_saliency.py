"""Replot the whole-cap saliency map as a smooth interpolated topomap.

First attempt used griddata(cubic) + nearest-neighbor fallback for points
outside the 62-electrode convex hull (CB1/CB2 and other edge channels) --
nearest-neighbor fill produces blocky Voronoi-cell patches near the scalp
edge that look like the color is "leaking" past the head outline. Switched
to RBFInterpolator (thin-plate spline), which extrapolates smoothly
everywhere with no blocky artifacts, then clip to the head circle via
imshow + set_clip_path (much more robust than clipping contourf's per-level
polygons).

Reuses saliency_whole.npz so no retraining needed.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" replot_saliency.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_figures import electrode_xy, INK, BG  # noqa: E402

z = np.load(HERE / "saliency_whole.npz", allow_pickle=True)
importance = z["importance"]
channels = [str(c) for c in z["channels"]]

plt.rcParams.update({
    "font.family": "Georgia", "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
})

xy = electrode_xy(channels)
pts = np.array([xy[ch] for ch in channels])
r = max(np.hypot(x, y) for x, y in pts) * 1.08

rbf = RBFInterpolator(pts, importance, kernel="thin_plate_spline", smoothing=0.0)
n = 300
gx, gy = np.mgrid[-r * 1.02:r * 1.02:n * 1j, -r * 1.02:r * 1.02:n * 1j]
grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
grid = rbf(grid_pts).reshape(gx.shape)

order = np.argsort(-importance)
top10 = set(order[:10].tolist())

fig, ax = plt.subplots(figsize=(6.8, 6.8))
cmap = plt.colormaps["YlOrRd"]
im = ax.imshow(grid.T, origin="lower",
               extent=(-r * 1.02, r * 1.02, -r * 1.02, r * 1.02),
               cmap=cmap, interpolation="bilinear", zorder=1)
clip_circle = plt.Circle((0, 0), r, transform=ax.transData)
im.set_clip_path(clip_circle)
ax.contour(gx, gy, grid, levels=10, colors=INK, linewidths=0.3, alpha=0.3,
          zorder=2).set_clip_path(clip_circle)

# head outline, nose, ears
ax.add_patch(plt.Circle((0, 0), r, fill=False, color=INK, lw=1.6, zorder=5))
ax.plot([0, -r * 0.11, 0, r * 0.11, 0],
        [r, r * 1.13, r * 1.20, r * 1.13, r], color=INK, lw=1.6, zorder=5)
for s in (-1, 1):
    ax.add_patch(plt.matplotlib.patches.Ellipse(
        (s * r, 0), r * 0.10, r * 0.32, fill=False, color=INK, lw=1.6, zorder=5))

# uniform electrode dots, top 10 labeled
for i, ch in enumerate(channels):
    x, y = xy[ch]
    on = i in top10
    ax.scatter(x, y, s=26, c="none", edgecolors=INK, linewidths=0.9, zorder=6)
    if on:
        ax.text(x, y - r * 0.05, ch, ha="center", va="top", fontsize=7.5,
                color=INK, fontweight="bold", zorder=7)

ax.set_xlim(-r * 1.25, r * 1.25)
ax.set_ylim(-r * 1.25, r * 1.32)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Whole-cap (62ch) DGCNN saliency map\n"
             "mean |gradient of true-class logit| per channel, DE_LDS, "
             "pooled across 3 test folds\ntop 10 channels labeled",
             fontsize=11.5)
cbar = fig.colorbar(im, ax=ax, shrink=0.65, pad=0.03)
cbar.set_label("mean |gradient|", fontsize=9)
fig.tight_layout()
out = HERE / "fig_saliency_whole.png"
fig.savefig(out, dpi=170, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("Saved ->", out)
