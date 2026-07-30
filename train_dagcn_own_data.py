"""Train a DAGCN model from scratch on your own quick20-recorded, self-labeled
data (the annotate_session.py sessions with 20s popup ratings), collapsing
the 5-point rating scale down to the model's 3 classes.

Label collapse (5 -> 3):
  負向 (negative)          -> negative
  偏負 (slightly negative)  -> negative
  中性 (neutral)            -> neutral
  偏正 (slightly positive)  -> positive
  正向 (positive)           -> positive

Each rating answered at time T labels the window [previous prompt's answer
time (or session start), T) -- matching how annotate_session.py's own
segments.csv is built. Causal DE+PSD features (+ Kalman smoothing) are
computed once per whole session (so smoothing has continuity across window
boundaries), then sliced into per-window chunks and labeled.

Split is by WHOLE WINDOW (not by individual second) into train/test, so
seconds from the same window never leak across the split.

Usage:
  "C:/Users/USER/miniconda3/envs/EEG/python.exe" train_dagcn_own_data.py
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

HERE = Path(__file__).parent
SESSIONS_DIR = HERE / "sessions"
EMOTION_MODEL_DIR = HERE / "emotion_model"
sys.path.insert(0, str(EMOTION_MODEL_DIR))

from utils.utils import setup_seed  # noqa: E402
from Trainer.training import train as libeer_train  # noqa: E402
from DAGCN_quick20_realtime_train import causal_psd_features, DAGCNQuick20Wrapper  # noqa: E402
from DGCNN_quick20_realtime_train import (  # noqa: E402
    QUICK20_CHANNEL_NAME, causal_de_features, kalman_smooth_trial,
)

DEVICE_TO_MODEL_LABEL = {
    "Fp1": "FP1", "Fp2": "FP2", "F7": "F7", "F3": "F3", "Fz": "FZ", "F4": "F4",
    "F8": "F8", "T3": "T7", "C3": "C3", "Cz": "CZ", "C4": "C4", "T4": "T8",
    "T5": "P7", "P3": "P3", "Pz": "PZ", "P4": "P4", "T6": "P8", "O1": "O1", "O2": "O2",
}
MODEL_LABEL_TO_DEVICE = {v: k for k, v in DEVICE_TO_MODEL_LABEL.items()}
IDENTITY_INDICES = list(range(19))  # our raw19 array is already in QUICK20_CHANNEL_NAME order

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
    """Returns [(start_s, end_s, collapsed_label), ...] from ratings.csv."""
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


def build_dataset(kalman_q, kalman_r, min_window_s=3.0):
    """Returns list of (feature_array (19,10), label_idx, session_name, window_idx)."""
    session_dirs = sorted(
        d for d in SESSIONS_DIR.glob("Shine_20260720_*")
        if (d / "eeg_raw.npz").exists() and (d / "ratings.csv").exists()
    )
    print(f"Found {len(session_dirs)} sessions with EEG + ratings")

    samples = []  # (feature, label_idx, session_name, window_idx)
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
        de = causal_de_features(raw19, srate_int, IDENTITY_INDICES)
        psd = causal_psd_features(raw19, srate_int, IDENTITY_INDICES)
        combined = [np.concatenate([d, p], axis=-1) for d, p in zip(de, psd)]
        combined = kalman_smooth_trial(combined, kalman_q, kalman_r)  # list of (19,10), one per second
        n_secs = len(combined)

        for w_i, (start_s, end_s, label) in enumerate(windows):
            if end_s - start_s < min_window_s:
                continue
            s0, s1 = int(round(start_s)), min(n_secs, int(round(end_s)))
            if s1 <= s0:
                continue
            for t in range(s0, s1):
                samples.append((combined[t], LABEL_TO_IDX[label], session_dir.name, w_i))

        print(f"  {session_dir.name}: {len(windows)} windows, {n_secs}s total")

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kalman-q", type=float, default=0.01)
    parser.add_argument("--kalman-r", type=float, default=0.5)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--split-by", choices=["window", "session"], default="window",
                         help="window: shuffle individual rating-windows into train/test (a session's "
                              "windows can end up on both sides). session: hold out whole sessions for "
                              "test, so test is a genuinely unseen recording -- a stricter, more honest "
                              "generalization estimate.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else HERE / f"result_dagcn_own_data_split_{args.split_by}"
    out_dir.mkdir(exist_ok=True)

    setup_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    print("Building dataset from Shine_20260720_* sessions...")
    samples = build_dataset(args.kalman_q, args.kalman_r)
    print(f"\nTotal per-second samples: {len(samples)}")

    counts = {k: 0 for k in LABEL_TO_IDX}
    for _, lbl_idx, *_ in samples:
        counts[IDX_TO_LABEL[lbl_idx]] += 1
    print("Class distribution:", counts)

    if args.split_by == "session":
        # Hold out whole sessions -- test is a recording the model never saw any
        # part of. A single random shuffle can easily leave one class with zero
        # test windows (sessions aren't class-balanced individually), so search
        # over many candidate shuffles and keep the one whose test set has the
        # most even per-class window counts, subject to total test fraction
        # staying near --test-frac.
        session_names = sorted({s for _, _, s, _ in samples})
        session_class_counts = {s: Counter() for s in session_names}
        for _, lbl_idx, s, _ in samples:
            session_class_counts[s][IDX_TO_LABEL[lbl_idx]] += 1
        total_by_class = Counter()
        for c in session_class_counts.values():
            total_by_class.update(c)

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
            score = (min_class_count, -frac_err)  # maximize min-per-class first, then closeness to target frac
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
        # split by (session, window) so seconds from the same window never leak across train/test,
        # but windows from the same session can still end up on both sides
        window_keys = sorted({(s, w) for _, _, s, w in samples})
        rng.shuffle(window_keys)
        n_test = max(1, int(len(window_keys) * args.test_frac))
        test_keys = set(window_keys[:n_test])
        train_keys = set(window_keys) - test_keys

    print(f"Windows: train={len(train_keys)} test={len(test_keys)}")

    train_X, train_y = [], []
    test_X, test_y = [], []
    for feat, lbl_idx, s, w in samples:
        if (s, w) in test_keys:
            test_X.append(feat)
            test_y.append(lbl_idx)
        else:
            train_X.append(feat)
            train_y.append(lbl_idx)

    train_X, train_y = np.stack(train_X), np.array(train_y, dtype=np.int64)
    test_X, test_y = np.stack(test_X), np.array(test_y, dtype=np.int64)
    print(f"Train samples: {len(train_X)}  Test samples: {len(test_X)}")

    shape = train_X.shape
    flat_train = train_X.reshape(-1, shape[1] * shape[2])
    mean, std = flat_train.mean(axis=0), flat_train.std(axis=0) + 1e-8

    def apply_scaler(x):
        flat = x.reshape(-1, shape[1] * shape[2])
        flat = (flat - mean) / std
        return flat.reshape(x.shape)

    train_X = apply_scaler(train_X)
    test_X = apply_scaler(test_X)

    device = torch.device("cuda")
    model = DAGCNQuick20Wrapper(19, 10, 3)
    dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_X), torch.LongTensor(train_y))
    dataset_val = torch.utils.data.TensorDataset(torch.Tensor(train_X), torch.LongTensor(train_y))
    dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_X), torch.LongTensor(test_y))

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("\n=== Training from scratch on your own labeled quick20 data ===")
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
