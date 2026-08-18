"""Extract realtime-legal (causal) DE+PSD features from the REFED dataset.

REFED is a 32-subject EEG+fNIRS dataset with continuous 1 Hz joystick
valence/arousal annotation. This script reduces it to exactly the same
per-second (19, 10) feature representation the quick20 pipeline uses, so the
same DGCNN training code can run on it.

Everything here is causal / realtime-legal:
  - forward-only Butterworth bandpass (lfilter) for DE, per SEED 5-band split
  - memoryless per-second Welch PSD
  - forward-only scalar Kalman smoothing (NOT LibEER's non-causal `_lds`)
  - causal decimation 1000 Hz -> 200 Hz (zero_phase=False)

Two feature sets are written per subject:
  - video features (one per second of each of the 15 clips) + labels
  - baseline features (from the 5 s resting segment preceding each clip),
    used for per-subject "session-baseline" normalization

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" refed_extract_features.py \
      --refed-root "D:/EEG dataset/REFED" --out refed_features.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import decimate

HERE = Path(__file__).resolve().parent
EMOTION_MODEL_DIR = HERE.parent / "emotion_model"
sys.path.insert(0, str(EMOTION_MODEL_DIR))

from DGCNN_quick20_realtime_train import (  # noqa: E402
    causal_de_features, kalman_smooth_trial,
)
from DAGCN_quick20_realtime_train import causal_psd_features  # noqa: E402

# The 19 channels the CGX Quick-20 records, in the order the model expects.
QUICK20_CHANNEL_NAME = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8',
    'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']

# REFED's own 64-channel order (EEG_channels.csv). All 19 above are present.
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

# The causal Butterworth transient needs ~3-5 s of context to settle (verified
# in the quick20 pipeline). REFED's resting baselines are only 5 s each, so the
# first BASELINE_WARMUP_S seconds of every baseline segment are discarded and
# only the settled remainder contributes to the per-subject normalization stats.
BASELINE_WARMUP_S = 3

# Same reasoning for video trials: never emit a labeled sample from the first
# TRIAL_WARMUP_S seconds of a clip.
TRIAL_WARMUP_S = 5

IDENTITY_INDICES = list(range(len(QUICK20_CHANNEL_NAME)))
LABEL_TO_IDX = {"negative": 0, "neutral": 1, "positive": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

JOYSTICK_CENTER = 128


def quick20_indices():
    return [REFED_CHANNEL_NAME.index(ch) for ch in QUICK20_CHANNEL_NAME]


def causal_decimate(sig, factor):
    """Downsample without any look-ahead (zero_phase=False keeps it causal)."""
    if factor == 1:
        return sig
    return decimate(sig, factor, ftype='iir', axis=-1, zero_phase=False)


def features_for_segment(raw19, srate, kalman_q, kalman_r):
    """(19, T) raw -> list of per-second (19, 10) causal DE+PSD features."""
    de = causal_de_features(raw19, srate, IDENTITY_INDICES)
    psd = causal_psd_features(raw19, srate, IDENTITY_INDICES)
    combined = [np.concatenate([d, p], axis=-1) for d, p in zip(de, psd)]
    return kalman_smooth_trial(combined, kalman_q, kalman_r)


def trial_labels(valence, margin):
    """Per-second 3-class valence labels from the raw joystick trace.

    The joystick starts parked at 128 and subjects take a median of ~13 s to
    first move it. That leading run of exact-128 means "has not answered yet",
    NOT "feels neutral", so it is dropped rather than labeled neutral.

    Returns (labels, start_index) or (None, None) if the subject never moved
    the joystick during this clip.
    """
    moved = np.nonzero(valence != JOYSTICK_CENTER)[0]
    if len(moved) == 0:
        return None, None
    start = int(moved[0])
    v = valence[start:]
    labels = np.full(len(v), LABEL_TO_IDX["neutral"], dtype=np.int64)
    labels[v > JOYSTICK_CENTER + margin] = LABEL_TO_IDX["positive"]
    labels[v < JOYSTICK_CENTER - margin] = LABEL_TO_IDX["negative"]
    return labels, start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refed-root", default="D:/EEG dataset/REFED")
    ap.add_argument("--out", default=str(HERE / "refed_features.npz"))
    ap.add_argument("--margin", type=int, default=40,
                    help="joystick deviation from 128 required for pos/neg (default 40)")
    ap.add_argument("--kalman-q", type=float, default=0.01)
    ap.add_argument("--kalman-r", type=float, default=0.5)
    ap.add_argument("--subjects", type=int, default=N_SUBJECTS)
    args = ap.parse_args()

    root = Path(args.refed_root)
    ch_idx = quick20_indices()
    print(f"REFED root: {root}")
    print(f"Using {len(ch_idx)} of 64 channels (Quick-20 subset), "
          f"{REFED_SRATE} Hz -> {TARGET_SRATE} Hz causal decimation")
    print(f"Label margin: +/-{args.margin} around {JOYSTICK_CENTER}")

    vid_feats, vid_labels, vid_subject, vid_video = [], [], [], []
    base_feats, base_subject = [], []

    for sub in range(1, args.subjects + 1):
        sub_dir = root / "data" / str(sub)
        eeg_v = sio.loadmat(sub_dir / "EEG_videos.mat")
        eeg_b = sio.loadmat(sub_dir / "EEG_baselines.mat")
        ann = sio.loadmat(root / "annotations" / f"{sub}_label.mat")

        n_sub_samples = 0
        for vid in range(1, N_VIDEOS + 1):
            key = f"video_{vid}"

            # ---- resting baseline (5 s) -> per-subject normalization stats ----
            raw_b = eeg_b[key][ch_idx, :].astype(np.float64)
            raw_b = causal_decimate(raw_b, DECIM)
            feats_b = features_for_segment(raw_b, TARGET_SRATE, args.kalman_q, args.kalman_r)
            for f in feats_b[BASELINE_WARMUP_S:]:
                base_feats.append(f)
                base_subject.append(sub)

            # ---- video trial ----
            valence = ann[key][:, 0].astype(np.float64)
            labels, start = trial_labels(valence, args.margin)
            if labels is None:
                print(f"  sub {sub} video {vid}: joystick never moved, skipped")
                continue

            raw_v = eeg_v[key][ch_idx, :].astype(np.float64)
            raw_v = causal_decimate(raw_v, DECIM)
            feats_v = features_for_segment(raw_v, TARGET_SRATE, args.kalman_q, args.kalman_r)

            # feats_v[t] is second t of the clip; labels[i] is second start+i.
            first_sec = max(start, TRIAL_WARMUP_S)
            for t in range(first_sec, min(len(feats_v), start + len(labels))):
                vid_feats.append(feats_v[t])
                vid_labels.append(labels[t - start])
                vid_subject.append(sub)
                vid_video.append(vid)
                n_sub_samples += 1

        print(f"sub {sub:2d}: {n_sub_samples} labeled seconds")

    X = np.stack(vid_feats).astype(np.float32)
    y = np.array(vid_labels, dtype=np.int64)
    subj = np.array(vid_subject, dtype=np.int64)
    vids = np.array(vid_video, dtype=np.int64)
    Xb = np.stack(base_feats).astype(np.float32)
    subj_b = np.array(base_subject, dtype=np.int64)

    print(f"\nVideo features:    {X.shape}")
    print(f"Baseline features: {Xb.shape} "
          f"({Xb.shape[0] // args.subjects} per subject)")
    counts = np.bincount(y, minlength=3)
    print("Class distribution:", {IDX_TO_LABEL[i]: int(c) for i, c in enumerate(counts)},
          "->", (counts / counts.sum()).round(3).tolist())

    np.savez_compressed(
        args.out, X=X, y=y, subject=subj, video=vids,
        X_baseline=Xb, subject_baseline=subj_b,
        margin=args.margin, kalman_q=args.kalman_q, kalman_r=args.kalman_r,
        srate=TARGET_SRATE,
    )
    print("Saved ->", args.out)


if __name__ == "__main__":
    main()
