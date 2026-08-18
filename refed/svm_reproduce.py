"""Reproduce the REFED paper's SVM baseline (Table 6, EEG-only row) as closely
as a "simplest SVM" reading of Appendix E allows.

Paper protocol (Appendix E, p.30-31):
  - DE features, 64 EEG channels, 5 bands: delta 1-4, theta 4-8, alpha 8-14,
    beta 14-31, gamma 31-50 Hz -- flattened to a 320-dim vector per second.
  - Labels rescaled from the raw [1,255] joystick range to [0,1].
  - 3-class bins: low [0,0.4), medium [0.4,0.6], high (0.6,1.0] -- NOT
    symmetric thirds like the ±40/127 margin used elsewhere in this project.
  - 3-fold CV PER SUBJECT (not pooled): each participant's 15 clips split
    into 3 groups of 5, one clip per target emotion each -- identical to this
    project's TARGET_GROUPS/FOLDS.
  - SVM hyperparameter search over kernel (linear/rbf/poly) and C
    (0.001-100). This script uses a smaller grid (linear/rbf, C in
    [0.1, 1, 10]) as the "simplest" version -- noted as a deliberate
    simplification, not a hidden shortcut.
  - Reports accuracy and weighted F1, mean+/-std across the 32 subjects.

Target: Table 6 SVM/EEG row -- valence acc 0.524+/-0.135, F1 0.402+/-0.141;
arousal acc 0.621+/-0.150, F1 0.503+/-0.167.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" svm_reproduce.py
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import signal
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "emotion_model"))
from refed_montage_regress import TARGET_GROUPS  # noqa: E402
from data_utils.preprocess import lds  # noqa: E402

REFED_SRATE = 1000
N_VIDEOS = 15
N_SUBJECTS = 32
BANDS = [[1, 4], [4, 8], [8, 14], [14, 31], [31, 50]]  # paper's exact edges
FOLDS = [[g[i] for g in TARGET_GROUPS.values()] for i in range(3)]


def preprocess(raw, sample_rate=REFED_SRATE, car=False, notch=False):
    """Optional lightweight steps before band filtering -- no ICA, matching
    how the DGCNN pipeline has been run throughout this project."""
    if car:
        raw = raw - raw.mean(axis=0, keepdims=True)  # common average reference
    if notch:
        b, a = signal.iirnotch(50.0, Q=30.0, fs=sample_rate)  # mains hum
        raw = signal.filtfilt(b, a, raw)
    return raw


def paper_de(data, sample_rate=REFED_SRATE):
    """1-second-window DE, 5 bands, 3rd-order Butterworth + filtfilt.
    NOTE: this was a guess at "how the paper probably did it" based on the
    appendix text, but it is NOT what their released code actually does --
    see paper_de_exact() below, which is a direct port of DE_PSD.py from
    https://github.com/REFED-dataset/REFED-codes."""
    nyq = 0.5 * sample_rate
    window = sample_rate
    n = data.shape[1] // window
    out = np.zeros((n, data.shape[0], len(BANDS)))
    for b_i, band in enumerate(BANDS):
        b, a = signal.butter(3, [band[0] / nyq, band[1] / nyq], 'bandpass')
        filt = signal.filtfilt(b, a, data)
        for i in range(n):
            seg = filt[:, i * window:(i + 1) * window]
            out[i, :, b_i] = 0.5 * np.log2(2 * np.pi * np.e * np.var(seg, axis=1, ddof=1))
    return out


# ---- direct port of the official REFED-codes DE_PSD.py + label_to_3c ----
# (github.com/REFED-dataset/REFED-codes, DE_PSD.py and
# 7_supervised_learning/utils/utils_data.py) -- confirms paper_de()/to3()
# above were both wrong guesses: real feature extraction is STFT+Hanning+FFT
# magnitude-squared PSD (not Butterworth+variance), and the real 3-class
# thresholds are 0.3/0.7 on the raw label scale (not 0.4/0.6 on 0-1).
_STFT_PARA = dict(stftn=1000, fStart=[1, 4, 8, 14, 31], fEnd=[4, 8, 14, 31, 50],
                  fs=REFED_SRATE, window=1)


def _de_psd_one_window(data, stftn, fStart, fEnd, fs, window):
    """data: (n_channels, window*fs) -- one 1s window, all channels."""
    window_points = fs * window
    f_start_num = [int(f / fs * stftn) for f in fStart]
    f_end_num = [int(f / fs * stftn) for f in fEnd]
    n = data.shape[0]
    de = np.zeros((n, len(fStart)))
    hwin = np.array([0.5 - 0.5 * np.cos(2 * np.pi * i / (window_points + 1))
                     for i in range(1, window_points + 1)])
    for j in range(n):
        hdata = data[j] * hwin
        fft_data = np.fft.fft(hdata, stftn)
        mag = np.abs(fft_data[:stftn // 2])
        for p in range(len(fStart)):
            e = np.sum(mag[f_start_num[p] - 1:f_end_num[p]] ** 2)
            e = e / (f_end_num[p] - f_start_num[p] + 1)
            de[j, p] = np.log2(100 * e + 1e-8)
    return de


def paper_de_exact(data, sample_rate=REFED_SRATE):
    """Faithful port of DE_PSD.py, called per 1s window exactly as
    2_feature_EEG.ipynb does (STFTN=1000, bands identical to BANDS above,
    Hanning window, FFT magnitude-squared PSD, DE=log2(100*E+1e-8))."""
    para = dict(_STFT_PARA)
    window_points = para["fs"] * para["window"]
    n_ch, n_samples = data.shape
    n_windows = n_samples // window_points
    out = np.zeros((n_windows, n_ch, len(para["fStart"])))
    for w in range(n_windows):
        seg = data[:, w * window_points:(w + 1) * window_points]
        out[w] = _de_psd_one_window(seg, **para)
    return out


def to3(y01):
    """low [0,0.4) / medium [0.4,0.6] / high (0.6,1.0] -> 0/1/2.
    NOTE: guessed thresholds, not what the paper's code actually uses --
    see to3_exact() below."""
    out = np.ones_like(y01, dtype=np.int64)
    out[y01 < 0.4] = 0
    out[y01 > 0.6] = 2
    return out


def to3_exact(raw_label, thresholds=(0.3, 0.7), original_scale=256):
    """Direct port of label_to_3c() from utils_data.py -- operates on the
    RAW [1,255]-ish joystick scale, not a 0-1 rescaled value."""
    out = np.ones_like(raw_label, dtype=np.int64)
    out[raw_label < original_scale * thresholds[0]] = 0
    out[raw_label > original_scale * thresholds[1]] = 2
    return out


def to2(y01):
    """LibEER's DEAP convention: binary high/low split at the midpoint
    (label_process() with bounds=[0.5, 0.5]), not the paper's 3-class scheme."""
    return (y01 >= 0.5).astype(np.int64)


def extract_subject(root, sub, car=False, notch=False, use_lds=False, exact_de=False):
    eeg_v = sio.loadmat(root / "data" / str(sub) / "EEG_videos.mat")
    ann = sio.loadmat(root / "annotations" / f"{sub}_label.mat")
    feats, targets, targets_raw, vids = [], [], [], []
    for vid in range(1, N_VIDEOS + 1):
        key = f"video_{vid}"
        lab = ann[key].astype(float)
        valence01 = (lab[:, 0] - 1) / 254.0
        arousal01 = (lab[:, 1] - 1) / 254.0
        raw = eeg_v[key].astype(np.float64)
        raw = preprocess(raw, car=car, notch=notch)
        de = paper_de_exact(raw) if exact_de else paper_de(raw)  # (T, 64, 5)
        if use_lds:
            de = lds(de)  # LibEER's non-causal LDS smoothing, per trial
        n = min(len(de), len(valence01))
        for t in range(n):
            feats.append(de[t].reshape(-1))  # flatten 64x5 -> 320
            targets.append([valence01[t], arousal01[t]])
            targets_raw.append([lab[t, 0], lab[t, 1]])
            vids.append(vid)
    return (np.stack(feats).astype(np.float32), np.array(targets, np.float32),
            np.array(targets_raw, np.float32), np.array(vids))


def run_subject(X, Y, vids, grid, binary=False, global_norm=False, Y_raw=None, exact_thresh=False):
    """global_norm=True: z-score using the WHOLE session (all 15 clips,
    train+test) instead of the fold's training clips only -- a deliberate
    data-leakage experiment (test-fold stats included) to check whether the
    paper may have normalized this way, not something to use for a real
    reported number.
    exact_thresh=True: use to3_exact() (paper's real 0.3/0.7 raw-scale
    thresholds) on Y_raw instead of this script's originally-guessed
    to3()/to2()."""
    if exact_thresh:
        C_valence, C_arousal = to3_exact(Y_raw[:, 0]), to3_exact(Y_raw[:, 1])
    else:
        label_fn = to2 if binary else to3
        C_valence, C_arousal = label_fn(Y[:, 0]), label_fn(Y[:, 1])
    per_fold = {"valence": {"acc": [], "f1": []}, "arousal": {"acc": [], "f1": []}}

    if global_norm:
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
        X = (X - mu) / sd

    for test_vids in FOLDS:
        te = np.isin(vids, test_vids)
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        for dim, C in (("valence", C_valence), ("arousal", C_arousal)):
            if len(np.unique(C[tr])) < 2:
                continue
            if global_norm:
                clf = GridSearchCV(SVC(), grid, cv=3, n_jobs=1)
            else:
                clf = make_pipeline(StandardScaler(),
                                    GridSearchCV(SVC(), grid, cv=3, n_jobs=1))
            clf.fit(X[tr], C[tr])
            pred = clf.predict(X[te])
            per_fold[dim]["acc"].append(accuracy_score(C[te], pred))
            per_fold[dim]["f1"].append(f1_score(C[te], pred, average="weighted"))
    return per_fold


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=N_SUBJECTS)
    ap.add_argument("--grid", choices=["small", "full"], default="small")
    ap.add_argument("--car", action="store_true", help="common average reference")
    ap.add_argument("--notch", action="store_true", help="50Hz notch filter")
    ap.add_argument("--lds", action="store_true", help="LibEER non-causal LDS smoothing")
    ap.add_argument("--binary", action="store_true",
                    help="LibEER's DEAP-style 2-class high/low split at 0.5, "
                         "instead of the paper's 3-class low/medium/high")
    ap.add_argument("--global-norm", action="store_true",
                    help="z-score using the whole session incl. test clips "
                         "(data leakage, diagnostic only -- see run_subject)")
    ap.add_argument("--exact-de", action="store_true",
                    help="use paper_de_exact(), a direct port of the official "
                         "REFED-codes DE_PSD.py, instead of this script's "
                         "originally-guessed Butterworth+variance paper_de()")
    ap.add_argument("--exact-thresh", action="store_true",
                    help="use to3_exact(), the official REFED-codes "
                         "label_to_3c() thresholds (0.3/0.7 raw scale), "
                         "instead of this script's originally-guessed to3() "
                         "(0.4/0.6 on 0-1 scale)")
    ap.add_argument("--exact", action="store_true",
                    help="shorthand for --exact-de --exact-thresh together")
    a = ap.parse_args()
    exact_de = a.exact_de or a.exact
    exact_thresh = a.exact_thresh or a.exact

    root = Path("D:/EEG dataset/REFED")
    if a.grid == "full":
        grid = {"kernel": ["linear", "rbf", "poly"],
                "C": [0.001, 0.01, 0.1, 1, 10, 100]}
    else:
        grid = {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10]}
    print(f"grid={grid}  car={a.car}  notch={a.notch}  lds={a.lds}  binary={a.binary}  "
          f"global_norm={a.global_norm}  exact_de={exact_de}  exact_thresh={exact_thresh}")

    results = {"valence": {"acc": [], "f1": []}, "arousal": {"acc": [], "f1": []}}
    for sub in range(1, a.subjects + 1):
        X, Y, Y_raw, vids = extract_subject(root, sub, car=a.car, notch=a.notch,
                                            use_lds=a.lds, exact_de=exact_de)
        pf = run_subject(X, Y, vids, grid, binary=a.binary, global_norm=a.global_norm,
                         Y_raw=Y_raw, exact_thresh=exact_thresh)
        for dim in ("valence", "arousal"):
            if pf[dim]["acc"]:
                results[dim]["acc"].append(float(np.mean(pf[dim]["acc"])))
                results[dim]["f1"].append(float(np.mean(pf[dim]["f1"])))
        print(f"sub {sub:2d}: "
              f"V acc={np.mean(pf['valence']['acc']):.3f} f1={np.mean(pf['valence']['f1']):.3f} | "
              f"A acc={np.mean(pf['arousal']['acc']):.3f} f1={np.mean(pf['arousal']['f1']):.3f}",
              flush=True)

    print("\n" + "=" * 60)
    for dim in ("valence", "arousal"):
        acc = np.array(results[dim]["acc"])
        f1 = np.array(results[dim]["f1"])
        print(f"{dim:8s}: acc={acc.mean():.4f}+/-{acc.std():.4f}  "
              f"f1={f1.mean():.4f}+/-{f1.std():.4f}  (n={len(acc)} subjects)")
    print("\nPaper (Table 6, SVM/EEG): valence acc=0.5240+/-0.1354 f1=0.4023+/-0.1408 | "
          "arousal acc=0.6210+/-0.1503 f1=0.5029+/-0.1666")


if __name__ == "__main__":
    main()
