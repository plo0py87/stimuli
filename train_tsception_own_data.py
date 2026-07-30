"""Train a TSception model from scratch on your own quick20-recorded,
self-labeled data. TSception is a raw-waveform CNN (not a DE/PSD-feature
model like DAGCN/DGCNN), so the feature pipeline here differs from
train_dagcn_own_data.py in one respect: instead of per-second causal DE+PSD
features, each 1s window feeds the model as its raw (16-channel, causal
0.5-45Hz bandpassed) waveform. Everything else -- label collapse, per-second
windowing from ratings.csv, and the session-balanced-split search -- is
identical, so results are directly comparable.

TSception's architecture expects 62/32-electrode SEED/DEAP layouts collapsed
to a left/right-paired subset (see generate_TS_channel_order): for our
19-channel quick20 montage this drops the 3 midline channels (FZ/CZ/PZ,
no digit suffix to pair) and keeps 16 electrodes in L/R pairs.

Usage:
  "C:/Users/USER/miniconda3/envs/EEG/python.exe" train_tsception_own_data.py --split-by session
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import butter, lfilter

HERE = Path(__file__).parent
SESSIONS_DIR = HERE / "sessions"
EMOTION_MODEL_DIR = HERE / "emotion_model"
sys.path.insert(0, str(EMOTION_MODEL_DIR))

from utils.utils import setup_seed  # noqa: E402
from Trainer.training import train as libeer_train  # noqa: E402
from models.TSception import TSception, generate_TS_channel_order  # noqa: E402
from DGCNN_quick20_realtime_train import QUICK20_CHANNEL_NAME  # noqa: E402

DEVICE_TO_MODEL_LABEL = {
    "Fp1": "FP1", "Fp2": "FP2", "F7": "F7", "F3": "F3", "Fz": "FZ", "F4": "F4",
    "F8": "F8", "T3": "T7", "C3": "C3", "Cz": "CZ", "C4": "C4", "T4": "T8",
    "T5": "P7", "P3": "P3", "Pz": "PZ", "P4": "P4", "T6": "P8", "O1": "O1", "O2": "O2",
}
MODEL_LABEL_TO_DEVICE = {v: k for k, v in DEVICE_TO_MODEL_LABEL.items()}

TS_ORDER = generate_TS_channel_order(QUICK20_CHANNEL_NAME)  # (16,) indices into the 19-ch array
TS_CHANNEL_NAME = [QUICK20_CHANNEL_NAME[i] for i in TS_ORDER]

LABEL_COLLAPSE = {
    "負向": "negative", "偏負": "negative",
    "中性": "neutral",
    "偏正": "positive", "正向": "positive",
}
LABEL_TO_IDX = {"negative": 0, "neutral": 1, "positive": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


def load_session_raw19(npz_path):
    with np.load(npz_path, allow_pickle=False) as z:
        data = z["data"]
        labels = [str(x) for x in z["labels"]]
        srate = float(z["srate"])
    label_to_idx = {lb: i for i, lb in enumerate(labels)}
    missing = [dev for dev in DEVICE_TO_MODEL_LABEL if dev not in label_to_idx]
    if missing:
        raise ValueError(f"{npz_path}: missing channels {missing}")
    raw19 = np.stack(
        [data[label_to_idx[MODEL_LABEL_TO_DEVICE[model_lb]]] for model_lb in QUICK20_CHANNEL_NAME]
    ).astype(np.float64)
    return raw19, srate


def load_session_windows(session_dir):
    ratings_path = session_dir / "ratings.csv"
    if not ratings_path.exists():
        return []
    with open(ratings_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    windows = []
    prev_t = 0.0
    for r in rows:
        t = float(r["elapsed_s"])
        cat = r["category"].strip()
        if cat not in LABEL_COLLAPSE:
            print(f"  WARNING: unknown category '{cat}' in {ratings_path}, skipping window")
            prev_t = t
            continue
        windows.append((prev_t, t, LABEL_COLLAPSE[cat]))
        prev_t = t
    return windows


def causal_bandpass(raw19, srate_int, low=0.5, high=45.0):
    nyq = 0.5 * srate_int
    b, a = butter(N=4, Wn=[low / nyq, high / nyq], btype="bandpass")
    return lfilter(b, a, raw19, axis=1)


def build_dataset(min_window_s=3.0):
    """Returns list of (raw_window (16, srate), label_idx, session_name, window_idx)."""
    session_dirs = sorted(
        d for d in SESSIONS_DIR.glob("Shine_20260720_*")
        if (d / "eeg_raw.npz").exists() and (d / "ratings.csv").exists()
    )
    print(f"Found {len(session_dirs)} sessions with EEG + ratings")

    samples = []
    for session_dir in session_dirs:
        windows = load_session_windows(session_dir)
        if not windows:
            continue
        try:
            raw19, srate = load_session_raw19(session_dir / "eeg_raw.npz")
        except ValueError as e:
            print(f"  SKIP {session_dir.name}: {e}")
            continue

        srate_int = int(round(srate))
        filtered = causal_bandpass(raw19, srate_int)  # (19, T)
        ts_sig = filtered[TS_ORDER, :]  # (16, T)
        n_secs = ts_sig.shape[1] // srate_int
        ts_sig = ts_sig[:, : n_secs * srate_int]

        for w_i, (start_s, end_s, label) in enumerate(windows):
            if end_s - start_s < min_window_s:
                continue
            s0, s1 = int(round(start_s)), min(n_secs, int(round(end_s)))
            if s1 <= s0:
                continue
            for t in range(s0, s1):
                win = ts_sig[:, t * srate_int:(t + 1) * srate_int]
                samples.append((win.copy(), LABEL_TO_IDX[label], session_dir.name, w_i))

        print(f"  {session_dir.name}: {len(windows)} windows, {n_secs}s total")

    return samples, srate_int if session_dirs else 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--split-by", choices=["window", "session"], default="session")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else HERE / f"result_tsception_own_data_split_{args.split_by}"
    out_dir.mkdir(exist_ok=True)

    setup_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    print("Building dataset from Shine_20260720_* sessions...")
    samples, srate_int = build_dataset()
    print(f"\nTotal per-second samples: {len(samples)}  (srate={srate_int})")

    counts = {k: 0 for k in LABEL_TO_IDX}
    for _, lbl_idx, *_ in samples:
        counts[IDX_TO_LABEL[lbl_idx]] += 1
    print("Class distribution:", counts)

    if args.split_by == "session":
        session_names = sorted({s for _, _, s, _ in samples})
        session_class_counts = {s: Counter() for s in session_names}
        for _, lbl_idx, s, _ in samples:
            session_class_counts[s][IDX_TO_LABEL[lbl_idx]] += 1

        n_test_sessions = max(1, round(len(session_names) * args.test_frac))
        best = None
        search_rng = np.random.RandomState(args.seed)
        for _ in range(2000):
            order = session_names[:]
            search_rng.shuffle(order)
            candidate_test = set(order[:n_test_sessions])
            test_class_counts = Counter()
            for s in candidate_test:
                test_class_counts.update(session_class_counts[s])
            min_class_count = min(test_class_counts.get(c, 0) for c in LABEL_TO_IDX)
            total_test = sum(test_class_counts.values())
            frac_err = abs(total_test / len(samples) - args.test_frac)
            score = (min_class_count, -frac_err)
            if best is None or score > best[0]:
                best = (score, candidate_test, test_class_counts)

        _, test_sessions, test_class_counts = best
        test_keys = {(s, w) for _, _, s, w in samples if s in test_sessions}
        train_keys = {(s, w) for _, _, s, w in samples if s not in test_sessions}
        print(f"Sessions: {len(session_names)} total -> "
              f"{len(session_names) - len(test_sessions)} train / {len(test_sessions)} test")
        print("Test sessions:", sorted(test_sessions))
        print("Test per-class window counts (balanced search):", dict(test_class_counts))
    else:
        window_keys = sorted({(s, w) for _, _, s, w in samples})
        rng.shuffle(window_keys)
        n_test = max(1, int(len(window_keys) * args.test_frac))
        test_keys = set(window_keys[:n_test])
        train_keys = set(window_keys) - test_keys

    print(f"Windows: train={len(train_keys)} test={len(test_keys)}")

    train_X, train_y = [], []
    test_X, test_y = [], []
    for win, lbl_idx, s, w in samples:
        if (s, w) in test_keys:
            test_X.append(win)
            test_y.append(lbl_idx)
        else:
            train_X.append(win)
            train_y.append(lbl_idx)

    train_X, train_y = np.stack(train_X), np.array(train_y, dtype=np.int64)  # (N, 16, srate)
    test_X, test_y = np.stack(test_X), np.array(test_y, dtype=np.int64)
    print(f"Train samples: {len(train_X)}  Test samples: {len(test_X)}")

    # per-channel z-score (fit on train only), same spirit as the DE-feature scaler
    # in train_dagcn_own_data.py -- one mean/std per channel, broadcast over time.
    mean = train_X.mean(axis=(0, 2), keepdims=True)  # (1, 16, 1)
    std = train_X.std(axis=(0, 2), keepdims=True) + 1e-8

    train_X = (train_X - mean) / std
    test_X = (test_X - mean) / std

    device = torch.device(args.device)
    model = TSception(num_electrodes=16, num_datapoints=srate_int, num_classes=3)
    dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_X), torch.LongTensor(train_y))
    dataset_val = torch.utils.data.TensorDataset(torch.Tensor(train_X), torch.LongTensor(train_y))
    dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_X), torch.LongTensor(test_y))

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("\n=== Training TSception from scratch on your own labeled quick20 data ===")
    metric = libeer_train(
        model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test,
        device=device, output_dir=str(out_dir), metrics=["acc"], metric_choose="acc",
        optimizer=optimizer, batch_size=128, epochs=args.epochs, criterion=criterion,
    )
    print("Final LibEER-reported test metric:", metric)

    model.eval()
    with torch.no_grad():
        logits = model(torch.Tensor(test_X).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)
    acc = float((pred == test_y).mean())
    print(f"\n>>> Test accuracy: {acc:.4f}")

    per_class_correct = Counter()
    per_class_total = Counter()
    for p, y in zip(pred, test_y):
        per_class_total[IDX_TO_LABEL[y]] += 1
        per_class_correct[IDX_TO_LABEL[y]] += int(p == y)
    for cls in ["negative", "neutral", "positive"]:
        n = per_class_total[cls]
        c = per_class_correct[cls]
        print(f"  {cls}: {c}/{n} ({c/n*100:.1f}%)" if n else f"  {cls}: no test samples")

    torch.save({"model": model.state_dict(), "mean": mean, "std": std}, out_dir / "checkpoint-scratch")
    print("\nSaved checkpoint to", out_dir / "checkpoint-scratch")


if __name__ == "__main__":
    main()
