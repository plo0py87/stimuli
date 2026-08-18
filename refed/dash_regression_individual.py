"""Individual test-set regression curves per REFED subject, for the
interactive dashboard. Uses the per-subject (in-session) DGCNN predictions
already saved in libeer_lds_persubj_pred.npz -- for each subject, ranks ALL
of their clips (typically 15) by whole-cap valence CCC and renders ONE image
per trial (not stacked into a single tall figure -- a 15-row montage stays
blurry even after wheel-zoom, since zoom just interpolates a low-resolution
source). The dashboard's trial browser swaps between these per-trial images
for a given subject. Also writes dash_reg_trials_{suffix}.json, a manifest of
{subject: [{rank, vid, ccc}, ...]} the dashboard JS uses to drive the
prev/next trial selector.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" dash_regression_individual.py
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" dash_regression_individual.py \
      --pred libeer_de_persubj_pred.npz --suffix de
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_regress import DIMS, ccc  # noqa: E402
from refed_montage_figures import INK, BG, GRID  # noqa: E402

# Antialiasing blends line edges into hundreds of intermediate colors, which
# is exactly what a small quantized palette then bands/loses. Turning it off
# means every pixel is drawn from the handful of colors actually used (bg,
# ink, gridline, the 4 montage colors), so aggressive palette quantization
# stays crisp instead of blurry/banded -- lets dpi go back up without the
# file-size cost antialiasing would otherwise add.
plt.rcParams.update({"lines.antialiased": False, "patch.antialiased": False,
                     "text.antialiased": False})

MONT_ORDER = ["whole", "vr_ring", "quick20", "headtop"]
MONT_COLORS = {"whole": "#2f5f5c", "vr_ring": "#c1443f",
               "quick20": "#b8873a", "headtop": "#6a4c93"}


def rank_trials(pz, sub):
    """Rank this subject's clips by whole-cap valence CCC, best first.
    Returns list of (ccc, video_id)."""
    Y, subject, video, second = pz["Y"], pz["subject"], pz["video"], pz["second"]
    ms = subject == sub
    scored = []
    for vid in np.unique(video[ms]):
        m = ms & (video == vid)
        if m.sum() < 10:
            continue
        order = np.argsort(second[m])
        c = ccc(Y[m][order][:, 0], pz["pred_per_subject_whole"][m][order][:, 0])
        scored.append((float(c), int(vid)))
    scored.sort(reverse=True)
    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="libeer_lds_persubj_pred.npz")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    tag = f"_{a.suffix}" if a.suffix else ""

    pz = np.load(HERE / a.pred)
    Y, subject, video, second = pz["Y"], pz["subject"], pz["video"], pz["second"]

    manifest = {}
    total_kb = 0.0
    for sub in range(1, 33):
        ranked = rank_trials(pz, sub)
        manifest[str(sub)] = [{"rank": r_i + 1, "vid": vid, "ccc": c_val}
                              for r_i, (c_val, vid) in enumerate(ranked)]

        for r_i, (c_val, vid) in enumerate(ranked):
            m = (subject == sub) & (video == vid)
            order = np.argsort(second[m])
            t = second[m][order]

            fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
            for c_i, dim in enumerate(DIMS):
                ax = axes[c_i]
                ax.plot(t, Y[m][order][:, c_i], color=INK, lw=1.6, label="true")
                for name in MONT_ORDER:
                    p = pz[f"pred_per_subject_{name}"][m][order][:, c_i]
                    ax.plot(t, p, color=MONT_COLORS[name], lw=0.9, alpha=0.85, label=name)
                ax.axhline(0, color=GRID, lw=0.6, ls=":")
                ax.set_ylim(-1.05, 1.05)
                ax.set_facecolor(BG)
                ax.tick_params(labelsize=7.5)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                ax.set_title(dim, fontsize=9.5)
            axes[0].legend(frameon=False, fontsize=6.5, ncol=1, loc="lower left")
            fig.suptitle(f"subject {sub} — #{r_i+1}/{len(ranked)} by CCC, clip {vid}, "
                         f"CCC={c_val:+.3f}", fontsize=9, y=1.03)
            fig.tight_layout()
            out = HERE / f"dash_reg_sub{sub}{tag}_t{r_i+1}.png"
            fig.savefig(out, dpi=100, facecolor=BG, bbox_inches="tight")
            plt.close(fig)
            # With antialiasing off there are only a handful of true colors
            # in the image, so this quantization is near-lossless (not the
            # blur/banding tradeoff it would be with antialiasing on) while
            # still keeping 960 of these (32 subjects x 15 trials x 2 feature
            # types) well under the dashboard's size budget.
            Image.open(out).convert("RGB").quantize(colors=32, method=Image.MEDIANCUT).save(
                out, optimize=True)
            total_kb += out.stat().st_size / 1024

        print(f"sub {sub:2d}: {len(ranked)} trials, best clip {ranked[0][1]} ccc={ranked[0][0]:+.3f} | "
              f"worst clip {ranked[-1][1]} ccc={ranked[-1][0]:+.3f}")

    manifest_path = HERE / f"dash_reg_trials{tag}.json"
    manifest_path.write_text(json.dumps(manifest))
    print(f"\nWrote {manifest_path.name}")
    print(f"Total regression-curve image size: {total_kb/1024:.2f} MB "
          f"({sum(len(v) for v in manifest.values())} images).")


if __name__ == "__main__":
    main()
