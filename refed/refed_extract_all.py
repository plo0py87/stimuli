"""Extract causal per-second features for ALL 64 REFED channels, plus the
continuous valence/arousal targets, for the electrode-montage comparison.

Features are computed per channel independently (bandpass -> variance -> DE,
and per-second Welch PSD), and the Kalman smoother runs per feature dimension
independently. That means subsetting channels AFTER extraction is exactly
equivalent to extracting only that subset -- so this runs once and every
montage is a slice of the channel axis.

Everything stays realtime-legal: forward-only Butterworth (lfilter), memoryless
per-second Welch PSD, forward-only scalar Kalman, causal decimation. No LDS,
no filtfilt, no look-ahead of any kind.

Targets are the raw 1 Hz joystick positions rescaled to [-1, 1] via
(v - 128) / 127, kept continuous for regression.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" refed_extract_all.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import decimate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "emotion_model"))

from DGCNN_quick20_realtime_train import (  # noqa: E402
    causal_de_features, kalman_smooth_trial,
)
from DAGCN_quick20_realtime_train import causal_psd_features  # noqa: E402

REFED_CHANNEL_NAME = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2',
    'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
    'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7',
    'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5',
    'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

REFED_SRATE = 1000
TARGET_SRATE = 200
DECIM = REFED_SRATE // TARGET_SRATE
N_VIDEOS = 15
N_SUBJECTS = 32

BASELINE_WARMUP_S = 3   # causal Butterworth transient (baselines are only 5 s)
TRIAL_WARMUP_S = 5      # never emit a sample from a clip's first seconds
JOYSTICK_CENTER = 128
JOYSTICK_SCALE = 127.0


def causal_decimate(sig, factor):
    if factor == 1:
        return sig
    return decimate(sig, factor, ftype='iir', axis=-1, zero_phase=False)


def features_for_segment(raw, srate, kalman_q, kalman_r):
    idx = list(range(raw.shape[0]))
    de = causal_de_features(raw, srate, idx)
    psd = causal_psd_features(raw, srate, idx)
    combined = [np.concatenate([d, p], axis=-1) for d, p in zip(de, psd)]
    return kalman_smooth_trial(combined, kalman_q, kalman_r)


def trial_start(valence, arousal):
    """First second at which BOTH joystick axes have been engaged.

    Before that the axis is parked at 128, which means "has not answered yet",
    not "feels neutral". Returns None if either axis never moved at all.
    """
    mv = np.nonzero(valence != JOYSTICK_CENTER)[0]
    ma = np.nonzero(arousal != JOYSTICK_CENTER)[0]
    if len(mv) == 0 or len(ma) == 0:
        return None
    return int(max(mv[0], ma[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refed-root", default="D:/EEG dataset/REFED")
    ap.add_argument("--out", default=str(HERE / "refed_features_all64.npz"))
    ap.add_argument("--kalman-q", type=float, default=0.01)
    ap.add_argument("--kalman-r", type=float, default=0.5)
    ap.add_argument("--subjects", type=int, default=N_SUBJECTS)
    args = ap.parse_args()

    root = Path(args.refed_root)
    n_ch = len(REFED_CHANNEL_NAME)
    print(f"REFED root: {root}")
    print(f"Extracting all {n_ch} channels, {REFED_SRATE} -> {TARGET_SRATE} Hz (causal)")

    feats, targets, subj, vids, secs = [], [], [], [], []
    base_feats, base_subj = [], []
    dropped = 0

    for sub in range(1, args.subjects + 1):
        sub_dir = root / "data" / str(sub)
        eeg_v = sio.loadmat(sub_dir / "EEG_videos.mat")
        eeg_b = sio.loadmat(sub_dir / "EEG_baselines.mat")
        ann = sio.loadmat(root / "annotations" / f"{sub}_label.mat")

        n_sub = 0
        for vid in range(1, N_VIDEOS + 1):
            key = f"video_{vid}"

            raw_b = causal_decimate(eeg_b[key].astype(np.float64), DECIM)
            for f in features_for_segment(raw_b, TARGET_SRATE,
                                          args.kalman_q, args.kalman_r)[BASELINE_WARMUP_S:]:
                base_feats.append(f)
                base_subj.append(sub)

            lab = ann[key]
            valence = lab[:, 0].astype(np.float64)
            arousal = lab[:, 1].astype(np.float64)
            start = trial_start(valence, arousal)
            if start is None:
                dropped += 1
                continue

            raw_v = causal_decimate(eeg_v[key].astype(np.float64), DECIM)
            fv = features_for_segment(raw_v, TARGET_SRATE, args.kalman_q, args.kalman_r)

            first = max(start, TRIAL_WARMUP_S)
            for t in range(first, min(len(fv), len(valence))):
                feats.append(fv[t])
                targets.append([(valence[t] - JOYSTICK_CENTER) / JOYSTICK_SCALE,
                                (arousal[t] - JOYSTICK_CENTER) / JOYSTICK_SCALE])
                subj.append(sub)
                vids.append(vid)
                secs.append(t)
                n_sub += 1
        print(f"sub {sub:2d}: {n_sub} labeled seconds")

    X = np.stack(feats).astype(np.float32)
    Y = np.array(targets, dtype=np.float32)
    print(f"\nFeatures {X.shape}  targets {Y.shape}  (dropped {dropped} trials: an axis never moved)")
    print(f"valence range [{Y[:,0].min():.2f}, {Y[:,0].max():.2f}] mean {Y[:,0].mean():+.3f}")
    print(f"arousal range [{Y[:,1].min():.2f}, {Y[:,1].max():.2f}] mean {Y[:,1].mean():+.3f}")

    np.savez_compressed(
        args.out, X=X, Y=Y,
        subject=np.array(subj, np.int64), video=np.array(vids, np.int64),
        second=np.array(secs, np.int64),
        X_baseline=np.stack(base_feats).astype(np.float32),
        subject_baseline=np.array(base_subj, np.int64),
        channels=np.array(REFED_CHANNEL_NAME), srate=TARGET_SRATE,
    )
    print("Saved ->", args.out)


if __name__ == "__main__":
    main()
