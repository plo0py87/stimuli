"""Add LibEER's actual (non-causal) LDS smoothing onto the LibEER-matched
features, as a direct counterpart to add_kalman.py's causal Kalman pass.

LibEER's lds() (data_utils/preprocess.py:263) looks like a forward-only
Kalman recursion, but U[:,0] is initialized from `mean = data.mean(axis=0)`,
the mean over the WHOLE trial -- computed before the loop runs. So step 0
already depends on every later timestep. That single line is what makes it
non-causal / not realtime-legal, matching the project note that DE_LDS
"needs the whole trial mean". This script calls that exact function, per
(subject, video) trial in chronological order, on top of the same
refed_features_libeer.npz used by add_kalman.py, so the three preprocessing
arms (LibEER plain / LibEER+causal Kalman / LibEER+non-causal LDS) differ in
only the smoother.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" add_lds.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "emotion_model"))
from data_utils.preprocess import lds  # noqa: E402


def main():
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
            trial = X[idx]  # (time, channel, feature)
            sm = lds(trial)
            out[idx] = sm
            n_trials += 1

    print(f"lds-smoothed {n_trials} trials")
    print(f"std before {X.std():.3f} -> after {out.std():.3f}")

    dst = HERE / "refed_features_libeer_lds.npz"
    np.savez_compressed(
        dst, X=out.astype(np.float32), Y=z["Y"], subject=subject, video=video,
        second=second, X_baseline=z["X_baseline"],
        subject_baseline=z["subject_baseline"], channels=z["channels"],
        srate=z["srate"])
    print("Saved ->", dst)


if __name__ == "__main__":
    main()
