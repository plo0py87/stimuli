"""Extract REFED features using LibEER's exact SEED preprocessing pipeline.

The montage study's first pass fed REFED's raw signal straight into feature
extraction, while SEED arrives already band-limited (SJTU ships Preprocessed_EEG
filtered to 0-75 Hz) and LibEER then filters it again. That asymmetry depressed
every REFED number. This script removes it by reproducing LibEER's pipeline
step for step.

Traced from data_utils/preprocess.py:
  preprocess()      -> bandpass_filter() -> eog_remove() -> feature_extraction()
  bandpass_filter() -> 5th-order Butterworth, filtfilt, default 0.3-50 Hz
                       (utils/args.py: -low_pass 0.3, -high_pass 50)
  de_extraction()   -> bands [[0.5,4],[4,8],[8,14],[14,30],[30,50]]
                       3rd-order Butterworth, filtfilt, per band
                       DE = 1/2 * log2(2*pi*e*var), ddof=1, 1 s window
  eog_remove()      -> builds a PCA object and returns data unchanged, i.e. it
                       is a no-op in LibEER, so there is nothing to reproduce

NOTE ON CAUSALITY: LibEER uses filtfilt (zero-phase, non-causal) throughout, so
features produced here look at future samples and are NOT valid for a realtime
claim. That is a deliberate choice to match the SEED benchmark exactly; use
refed_extract_all.py for the realtime-legal causal version.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" refed_extract_libeer.py
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import signal

HERE = Path(__file__).resolve().parent

REFED_CHANNEL_NAME = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2',
    'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
    'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7',
    'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5',
    'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

REFED_SRATE = 1000
TARGET_SRATE = 200          # SEED's rate, so band edges behave identically
N_VIDEOS = 15
N_SUBJECTS = 32

# LibEER defaults (utils/args.py:45-46, preprocess.py:245)
PASS_BAND = [0.3, 50]
EXTRACT_BANDS = [[0.5, 4], [4, 8], [8, 14], [14, 30], [30, 50]]
TIME_WINDOW = 1

JOYSTICK_CENTER = 128
JOYSTICK_SCALE = 127.0


def libeer_bandpass(data, srate, pass_band):
    """preprocess.py:87 -- 5th-order Butterworth, zero-phase filtfilt."""
    nyq = 0.5 * srate
    b, a = signal.butter(N=5, Wn=[pass_band[0] / nyq, pass_band[1] / nyq],
                         btype='bandpass')
    return signal.filtfilt(b, a, data)


def libeer_de(data, srate, extract_bands=EXTRACT_BANDS, time_window=TIME_WINDOW):
    """preprocess.py:233 -- per-band 3rd-order Butterworth + filtfilt, then
    DE = 1/2 * log2(2*pi*e*var) with ddof=1 over non-overlapping windows."""
    nyq = 0.5 * srate
    window = int(time_window * srate)
    n = data.shape[1] // window
    out = np.zeros((n, data.shape[0], len(extract_bands)))
    for b_i, band in enumerate(extract_bands):
        b, a = signal.butter(3, [band[0] / nyq, band[1] / nyq], 'bandpass')
        filt = signal.filtfilt(b, a, data)
        for i in range(n):
            seg = filt[:, i * window:(i + 1) * window]
            out[i, :, b_i] = 0.5 * np.log2(
                2 * np.pi * np.e * np.var(seg, axis=1, ddof=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refed-root", default="D:/EEG dataset/REFED")
    ap.add_argument("--out", default=str(HERE / "refed_features_libeer.npz"))
    ap.add_argument("--subjects", type=int, default=N_SUBJECTS)
    args = ap.parse_args()

    root = Path(args.refed_root)
    decim = REFED_SRATE // TARGET_SRATE
    print(f"LibEER-matched extraction: bandpass {PASS_BAND} (5th order, filtfilt), "
          f"bands {EXTRACT_BANDS} (3rd order, filtfilt), DE log2/ddof=1")
    print(f"{REFED_SRATE} -> {TARGET_SRATE} Hz\n")

    feats, targets, subj, vids, secs = [], [], [], [], []

    for sub in range(1, args.subjects + 1):
        eeg_v = sio.loadmat(root / "data" / str(sub) / "EEG_videos.mat")
        ann = sio.loadmat(root / "annotations" / f"{sub}_label.mat")
        n_sub = 0

        for vid in range(1, N_VIDEOS + 1):
            key = f"video_{vid}"
            lab = ann[key]
            valence, arousal = lab[:, 0].astype(float), lab[:, 1].astype(float)

            raw = eeg_v[key].astype(np.float64)
            # Zero-phase decimation is consistent with LibEER's filtfilt stance.
            raw = signal.decimate(raw, decim, ftype='iir', axis=-1, zero_phase=True)
            raw = libeer_bandpass(raw, TARGET_SRATE, PASS_BAND)
            de = libeer_de(raw, TARGET_SRATE)

            # Every trial from t=0, full length -- matches REFED-codes'
            # load_label()/reshape convention exactly, no warmup trim, no
            # requirement that the joystick move off (128,128) on either
            # axis. (128,128) is the paper's own trial-start reset position,
            # not a missing-data marker -- a trial that stays at 128 on one
            # axis the whole time is a real "stayed neutral" rating, not
            # unusable data, and dropping it was disproportionately deleting
            # MVMA (medium/neutral-target) trials.
            for t in range(min(len(de), len(valence))):
                feats.append(de[t])
                targets.append([(valence[t] - JOYSTICK_CENTER) / JOYSTICK_SCALE,
                                (arousal[t] - JOYSTICK_CENTER) / JOYSTICK_SCALE])
                subj.append(sub)
                vids.append(vid)
                secs.append(t)
                n_sub += 1
        print(f"sub {sub:2d}: {n_sub} labeled seconds", flush=True)

    X = np.stack(feats).astype(np.float32)
    Y = np.array(targets, dtype=np.float32)
    print(f"\nFeatures {X.shape}  targets {Y.shape}")

    np.savez_compressed(
        args.out, X=X, Y=Y,
        subject=np.array(subj, np.int64), video=np.array(vids, np.int64),
        second=np.array(secs, np.int64),
        X_baseline=np.zeros((1, X.shape[1], X.shape[2]), np.float32),
        subject_baseline=np.zeros(1, np.int64),
        channels=np.array(REFED_CHANNEL_NAME), srate=TARGET_SRATE,
    )
    print("Saved ->", args.out)


if __name__ == "__main__":
    main()
