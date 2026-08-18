"""Recompute montage metrics with predictions clipped to the valid range.

The regression targets are (joystick - 128)/127 with joystick in [1, 255], so
every true value lies in [-1, 1] by construction. The network has no output
activation, so it happily predicts 2.3 -- values that are wrong by definition.
Clipping to [-1, 1] is free and principled, and needs no retraining: it is
applied to the saved predictions.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" refed_clip_metrics.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_regress import DIMS, MONTAGES, metrics  # noqa: E402

MONT_ORDER = ["whole", "quick20", "headtop", "glasses"]


def main():
    pz = np.load(HERE / "montage_predictions.npz")
    Y, subject = pz["Y"], pz["subject"]
    modes = sorted({k.split("_")[1] if k.startswith("pred_per_") is False else "per_subject"
                    for k in pz.files if k.startswith("pred_")})
    modes = []
    for k in pz.files:
        if not k.startswith("pred_"):
            continue
        rest = k[len("pred_"):]
        for m in ("per_subject", "pooled"):
            if rest.startswith(m) and m not in modes:
                modes.append(m)

    out = {}
    print(f"{'mode':12s} {'montage':9s} | {'valence CCC':>22s} | {'arousal CCC':>22s}")
    print(f"{'':12s} {'':9s} | {'raw':>10s} {'clipped':>11s} | {'raw':>10s} {'clipped':>11s}")
    print("=" * 74)
    for mode in modes:
        out[mode] = {}
        for name in MONT_ORDER:
            key = f"pred_{mode}_{name}"
            if key not in pz.files:
                continue
            raw = pz[key]
            clipped = np.clip(raw, -1.0, 1.0)

            def per_subject(pred):
                acc = {d: {k: [] for k in ("ccc", "pcc", "rmse", "acc3")} for d in DIMS}
                for s in np.unique(subject):
                    ms = subject == s
                    m = metrics(Y[ms], pred[ms])
                    for d in DIMS:
                        for k, v in m[d].items():
                            acc[d][k].append(v)
                return acc

            r_raw, r_clip = per_subject(raw), per_subject(clipped)
            out[mode][name] = {"raw": {d: {k: float(np.mean(v)) for k, v in r_raw[d].items()}
                                       for d in DIMS},
                               "clipped": {d: {k: float(np.mean(v)) for k, v in r_clip[d].items()}
                                           for d in DIMS},
                               "clipped_per_subject": {d: r_clip[d] for d in DIMS},
                               "frac_out_of_range": float((np.abs(raw) > 1).mean())}
            print(f"{mode:12s} {name:9s} | "
                  f"{np.mean(r_raw['valence']['ccc']):+10.3f} {np.mean(r_clip['valence']['ccc']):+11.3f} | "
                  f"{np.mean(r_raw['arousal']['ccc']):+10.3f} {np.mean(r_clip['arousal']['ccc']):+11.3f}"
                  f"    ({out[mode][name]['frac_out_of_range']*100:.1f}% out of range)")
        print()

    with open(HERE / "montage_results_clipped.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved -> montage_results_clipped.json")


if __name__ == "__main__":
    main()
