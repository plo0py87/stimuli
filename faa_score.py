"""Frontal Alpha Asymmetry (FAA) score, per second, from three frontal pairs:

  F3/F4   - most common in the literature
  F7/F8   - anterior-temporal, stronger signal but more EMG contamination risk
  Fp1/Fp2 - closest to the eyes, highest artifact risk, but always available
            on the Quick-20

FAA_pair = ln(alpha_power_right) - ln(alpha_power_left)
(alpha power approximated as variance of the 8-13Hz bandpassed signal in each
non-overlapping 1s window -- same convention as the DE features used
elsewhere in this project). Positive score => relatively more LEFT frontal
activation (alpha power is inversely related to cortical activation) =>
classically associated with approach/positive affect; negative => withdrawal/
negative affect. This script only outputs the raw scores -- no
threshold/classification yet, by design.

Works on any npz with 'data'/'labels'/'srate' that includes Fp1, Fp2, F3, F4,
F7, F8 (case-insensitive) -- our own quick20 session recordings, or the SEED
trial npz's produced by process_seed_labeled_trials.py.

Usage:
  python faa_score.py --npz-path sessions/<session>/eeg_raw.npz
  python faa_score.py --npz-path SEED/trials/trial01_positive.npz
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

ALPHA_LOW, ALPHA_HIGH = 8.0, 13.0

PAIRS = [
    ("F3", "F4", "f3f4"),
    ("F7", "F8", "f7f8"),
    ("Fp1", "Fp2", "fp1fp2"),
]


def bandpass_filter(data, srate, low, high, order=4):
    nyq = 0.5 * srate
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, data, axis=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute per-second Frontal Alpha Asymmetry (FAA) scores.")
    parser.add_argument("--npz-path", required=True, help="Path to an npz with data/labels/srate")
    parser.add_argument("--out-csv", default=None, help="Defaults to <npz-dir>/<npz-stem>_faa.csv")
    return parser.parse_args()


def compute_faa(npz_path):
    with np.load(npz_path, allow_pickle=False) as z:
        data = z["data"].astype(np.float64)
        labels = [str(x) for x in z["labels"]]
        srate = float(z["srate"])

    label_to_idx = {lb.lower(): i for i, lb in enumerate(labels)}
    needed = {ch for pair in PAIRS for ch in pair[:2]}
    missing = [ch for ch in needed if ch.lower() not in label_to_idx]
    if missing:
        raise ValueError(f"npz is missing required channels for FAA: {missing} (has: {labels})")

    alpha = bandpass_filter(data, srate, ALPHA_LOW, ALPHA_HIGH)

    sec_len = max(1, int(round(srate)))
    n_secs = alpha.shape[1] // sec_len
    alpha = alpha[:, : n_secs * sec_len]

    log_power = {}
    for ch in needed:
        idx = label_to_idx[ch.lower()]
        windows = alpha[idx].reshape(n_secs, sec_len)
        power = windows.var(axis=1) + 1e-12
        log_power[ch] = np.log(power)

    pair_scores = {}
    for left, right, key in PAIRS:
        pair_scores[key] = log_power[right] - log_power[left]

    combined = np.mean(list(pair_scores.values()), axis=0)
    return pair_scores, combined, n_secs


def main():
    args = parse_args()
    npz_path = Path(args.npz_path)
    pair_scores, combined, n_secs = compute_faa(npz_path)

    out_csv = Path(args.out_csv) if args.out_csv else npz_path.with_name(npz_path.stem + "_faa.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "faa_f3f4", "faa_f7f8", "faa_fp1fp2", "faa_combined"])
        for t in range(n_secs):
            writer.writerow([
                t,
                f"{pair_scores['f3f4'][t]:.4f}",
                f"{pair_scores['f7f8'][t]:.4f}",
                f"{pair_scores['fp1fp2'][t]:.4f}",
                f"{combined[t]:.4f}",
            ])

    print(f"Loaded: {npz_path}")
    print(f"Computed FAA for {n_secs} seconds")
    for key, vals in pair_scores.items():
        print(f"  {key}: mean={vals.mean():.3f} std={vals.std():.3f} min={vals.min():.3f} max={vals.max():.3f}")
    print(f"  combined: mean={combined.mean():.3f} std={combined.std():.3f} min={combined.min():.3f} max={combined.max():.3f}")
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
