"""Add causal Kalman smoothing back onto the LibEER-matched features.

Switching from the causal pipeline to LibEER's exact SEED recipe changed five
things at once (global bandpass, filtfilt vs lfilter, band edges, log2/ddof,
and dropping the Kalman smoother) and valence CCC fell from 0.205 to 0.049.
This isolates the last of those: same LibEER features, smoother added back.

Smoothing runs per (subject, clip) in chronological order, forward-only, so it
stays realtime-legal even though the underlying LibEER features are not.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" add_kalman.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "emotion_model"))
from DGCNN_quick20_realtime_train import kalman_smooth_trial  # noqa: E402


def main(q=0.01, r=0.5):
    src = HERE / "refed_features_libeer.npz"
    z = np.load(src, allow_pickle=True)
    X, subject, video, second = z["X"], z["subject"], z["video"], z["second"]

    out = np.empty_like(X)
    n_trials = 0
    for s in np.unique(subject):
        for v in np.unique(video):
            m = (subject == s) & (video == v)
            if not m.any():
                continue
            idx = np.nonzero(m)[0]
            idx = idx[np.argsort(second[idx])]  # chronological
            sm = kalman_smooth_trial([X[i] for i in idx], q, r)
            for i, val in zip(idx, sm):
                out[i] = val
            n_trials += 1

    print(f"smoothed {n_trials} trials  q={q} r={r}")
    print(f"std before {X.std():.3f} -> after {out.std():.3f}")

    dst = HERE / "refed_features_libeer_kalman.npz"
    np.savez_compressed(
        dst, X=out.astype(np.float32), Y=z["Y"], subject=subject, video=video,
        second=second, X_baseline=z["X_baseline"],
        subject_baseline=z["subject_baseline"], channels=z["channels"],
        srate=z["srate"])
    print("Saved ->", dst)


if __name__ == "__main__":
    main()
