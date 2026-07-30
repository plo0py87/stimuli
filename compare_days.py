"""Compare EEG characteristics across different recording days (0709, 0713,
0720) for the same person (Shine), to see concretely what differs day-to-day:
per-band causal DE (mean + spread) and a simple motion/EMG artifact-rate
proxy (fraction of 1s windows with an abnormally large raw amplitude swing).

Usage:
  "C:/Users/USER/miniconda3/envs/EEG/python.exe" compare_days.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
SESSIONS_DIR = HERE / "sessions"
EMOTION_MODEL_DIR = HERE / "emotion_model"
sys.path.insert(0, str(EMOTION_MODEL_DIR))

from DGCNN_quick20_realtime_train import causal_de_features, BANDS  # noqa: E402
from train_dagcn_own_data import load_session_raw19, IDENTITY_INDICES  # noqa: E402

BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
ARTIFACT_UV_THRESHOLD = 150.0  # a 1s window counts as "artifact-y" if any channel's
                                # peak-to-peak amplitude in that window exceeds this


def sessions_for_day(prefix):
    return sorted(
        d for d in SESSIONS_DIR.glob(f"{prefix}*")
        if (d / "eeg_raw.npz").exists()
    )


def analyze_day(session_dirs, max_sessions=8):
    """Returns (band_means (5,), band_stds (5,), artifact_rate, total_seconds)."""
    all_de = []  # list of (T,19,5)
    artifact_flags = []  # list of (T,) bool
    total_seconds = 0

    for session_dir in session_dirs[:max_sessions]:
        try:
            raw19, srate = load_session_raw19(session_dir / "eeg_raw.npz")
        except ValueError as e:
            print(f"  SKIP {session_dir.name}: {e}")
            continue
        srate_int = int(round(srate))
        de = np.stack(causal_de_features(raw19, srate_int, IDENTITY_INDICES))  # (T,19,5)
        all_de.append(de)
        total_seconds += de.shape[0]

        n_secs = de.shape[0]
        raw19 = raw19[:, : n_secs * srate_int]
        windows = raw19.reshape(19, n_secs, srate_int)
        ptp = windows.max(axis=2) - windows.min(axis=2)  # (19, n_secs)
        is_artifact = (ptp > ARTIFACT_UV_THRESHOLD).any(axis=0)  # (n_secs,)
        artifact_flags.append(is_artifact)

    de_all = np.concatenate(all_de, axis=0)  # (sumT, 19, 5)
    artifact_all = np.concatenate(artifact_flags, axis=0)

    band_means = de_all.mean(axis=(0, 1))  # (5,)
    band_stds = de_all.std(axis=(0, 1))  # (5,)
    artifact_rate = artifact_all.mean()
    return band_means, band_stds, artifact_rate, total_seconds


def main():
    days = {
        "0709": sessions_for_day("Shine_20260709") + sessions_for_day("Shine_E_"),
        "0713": sessions_for_day("Shine_Chillseph") + sessions_for_day("Shine_Clashroyale") + sessions_for_day("Shine_50Hear"),
        "0720": sessions_for_day("Shine_20260720"),
    }

    results = {}
    for day, session_dirs in days.items():
        print(f"\n=== {day}: {len(session_dirs)} sessions found ===")
        for s in session_dirs[:8]:
            print("  ", s.name)
        band_means, band_stds, artifact_rate, total_s = analyze_day(session_dirs)
        results[day] = (band_means, band_stds, artifact_rate, total_s)
        print(f"  total seconds analyzed: {total_s}")
        print(f"  artifact rate (>{ARTIFACT_UV_THRESHOLD}uV p2p in >=1 channel): {artifact_rate*100:.1f}%")
        for name, m, sd in zip(BAND_NAMES, band_means, band_stds):
            print(f"  {name:6s}: mean DE={m:.3f}  std DE={sd:.3f}")

    # --- plot: grouped bars, mean DE per band per day ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(BAND_NAMES))
    width = 0.25
    colors = {"0709": "#7fd0ff", "0713": "#ffd93d", "0720": "#ff6b6b"}
    for i, (day, (band_means, band_stds, _artifact_rate, _t)) in enumerate(results.items()):
        axes[0].bar(x + (i - 1) * width, band_means, width, yerr=band_stds, capsize=3,
                    label=day, color=colors.get(day, None))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(BAND_NAMES)
    axes[0].set_ylabel("DE (mean +/- std across all channels & seconds)")
    axes[0].set_title("Per-band DE by recording day")
    axes[0].legend()

    days_list = list(results.keys())
    artifact_rates = [results[d][2] * 100 for d in days_list]
    bars = axes[1].bar(days_list, artifact_rates, color=[colors.get(d) for d in days_list])
    axes[1].set_ylabel(f"% of 1s windows with >{ARTIFACT_UV_THRESHOLD:.0f}uV p2p (any channel)")
    axes[1].set_title("Motion/EMG artifact-rate proxy by day")
    for bar, val in zip(bars, artifact_rates):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    out_path = HERE / "compare_days.png"
    plt.savefig(out_path, dpi=130)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
