"""Same as silhouette_perperson.py but DE_LDS instead of plain DE, all 15
SEED subjects (one session each) and all 32 REFED subjects.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" silhouette_perperson_lds.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.metrics import silhouette_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "emotion_model"))
from refed_extract_libeer import libeer_bandpass, libeer_de, PASS_BAND  # noqa: E402
from data_utils.preprocess import lds  # noqa: E402
from refed_montage_regress import to_class, EPS  # noqa: E402

SEED_DIR = Path("D:/EEG dataset/SEED/SEED/SEED_EEG/Preprocessed_EEG")
SEED_SRATE = 200
SEED_LABELS = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]) + 1


def seed_one_session_per_subject():
    files = sorted(p for p in SEED_DIR.glob("*.mat") if p.name != "label.mat")
    by_subject = {}
    for f in files:
        sub = re.match(r"(\d+)_", f.name).group(1)
        by_subject.setdefault(sub, []).append(f)
    for sub in sorted(by_subject, key=int):
        yield sub, sorted(by_subject[sub])[0]


def sil_for_seed_file_lds(f):
    m = sio.loadmat(f)
    keys = [k for k in m if not k.startswith("__")]
    keys.sort(key=lambda k: int(k.split("_eeg")[-1]))
    feats, ys = [], []
    for t_i, k in enumerate(keys):
        raw = m[k].astype(np.float64)[:62]
        filt = libeer_bandpass(raw, SEED_SRATE, PASS_BAND)
        de = libeer_de(filt, SEED_SRATE)
        de_lds = lds(de)
        for v in de_lds:
            feats.append(v.reshape(-1))
            ys.append(SEED_LABELS[t_i])
    X = np.stack(feats).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    mu, sd = X.mean(axis=0), X.std(axis=0) + EPS
    return silhouette_score((X - mu) / sd, y)


def sil_for_refed_subject_lds(z, keep, sub):
    subject = z["subject"]
    m = subject == sub
    X = z["X"][m][:, keep, :].reshape(m.sum(), -1).astype(np.float32)
    y = to_class(z["Y"][m, 0])
    if len(np.unique(y)) < 2:
        return None
    mu, sd = X.mean(axis=0), X.std(axis=0) + EPS
    return silhouette_score((X - mu) / sd, y)


def main():
    print("=== SEED, one session per subject, DE_LDS ===")
    seed_sils = []
    for sub, f in seed_one_session_per_subject():
        s = sil_for_seed_file_lds(f)
        seed_sils.append(s)
        print(f"  subject {sub:>3s} ({f.name}): silhouette={s:+.3f}", flush=True)
    seed_sils = np.array(seed_sils)

    print("\n=== REFED, 32 subjects, DE_LDS ===")
    z = np.load(HERE / "refed_features_libeer_lds.npz", allow_pickle=True)
    channels = [str(c) for c in z["channels"]]
    keep = [i for i, c in enumerate(channels) if c not in ("M1", "M2")]
    refed_sils = []
    for sub in range(1, 33):
        s = sil_for_refed_subject_lds(z, keep, sub)
        if s is None:
            print(f"  subject {sub:2d}: skipped (only one class present)")
            continue
        refed_sils.append(s)
        print(f"  subject {sub:2d}: silhouette={s:+.3f}", flush=True)
    refed_sils = np.array(refed_sils)

    print("\n" + "=" * 60)
    print(f"SEED  (n={len(seed_sils)} subjects): mean={seed_sils.mean():+.3f} "
          f"+/-{seed_sils.std():.3f}  min={seed_sils.min():+.3f}  max={seed_sils.max():+.3f}")
    print(f"REFED (n={len(refed_sils)} subjects): mean={refed_sils.mean():+.3f} "
          f"+/-{refed_sils.std():.3f}  min={refed_sils.min():+.3f}  max={refed_sils.max():+.3f}")


if __name__ == "__main__":
    main()
