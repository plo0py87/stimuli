"""Same DGCNN checkpoint (train_dgcnn_own_data.py, trained with a GLOBAL
train-pool z-score), but instead of normalizing each trial's features with
that global train-pool mean/std, normalize with THIS SESSION's own resting
baseline (the Pre phase recorded before every 0728/0729 session: ~10s buffer
+ ~90s eyes-open rest, before video_phase_start) -- an in-session
normalization test of whether baseline-relative features generalize better
across days than the global-stats approach.

Model weights are unchanged; only the normalization stats change (global
train-pool mean/std -> this session's own baseline mean/std). This session's
Pre-phase baseline is entirely separate from all trial windows, so there's
no leakage.

Usage:
  "C:/Users/USER/miniconda3/envs/EEG/python.exe" predict_dgcnn_own_sessionnorm.py \
      --session-dir sessions/Shine_20260728_143129 \
      --checkpoint result_dgcnn_own_data_split_session/checkpoint-scratch
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
EMOTION_MODEL_DIR = HERE / "emotion_model"
sys.path.insert(0, str(EMOTION_MODEL_DIR))

from models.DGCNN import DGCNN  # noqa: E402
from DAGCN_quick20_realtime_train import causal_psd_features  # noqa: E402
from DGCNN_quick20_realtime_train import causal_de_features, kalman_smooth_trial  # noqa: E402
from train_dagcn_own_data import load_session_raw19, IDENTITY_INDICES, IDX_TO_LABEL  # noqa: E402


def get_baseline_window(markers_path):
    """Returns (start_s, end_s) for the Pre:open_eyes resting segment, falling
    back to the whole Pre phase (buffer+open_eyes) if open_eyes markers are
    missing."""
    with open(markers_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    events = {r["event"]: float(r["elapsed_s"]) for r in rows}
    if "phase_start:Pre:open_eyes" in events and "phase_end:Pre:open_eyes" in events:
        return events["phase_start:Pre:open_eyes"], events["phase_end:Pre:open_eyes"]
    if "phase_start:Pre:buffer" in events and "phase_sequence_end:Pre" in events:
        return events["phase_start:Pre:buffer"], events["phase_sequence_end:Pre"]
    raise ValueError(f"No Pre-phase baseline markers found in {markers_path}")


def compute_features(raw19, srate_int, kalman_q, kalman_r):
    de = causal_de_features(raw19, srate_int, IDENTITY_INDICES)
    psd = causal_psd_features(raw19, srate_int, IDENTITY_INDICES)
    combined = [np.concatenate([d, p], axis=-1) for d, p in zip(de, psd)]
    combined = kalman_smooth_trial(combined, kalman_q, kalman_r)
    return np.stack(combined)  # (T, 19, 10)


def compute_session_baseline_stats(session_dir, npz_name, markers_name, kalman_q, kalman_r):
    raw19, srate = load_session_raw19(session_dir / npz_name)
    srate_int = int(round(srate))
    start_s, end_s = get_baseline_window(session_dir / markers_name)
    i0, i1 = int(round(start_s * srate_int)), int(round(end_s * srate_int))
    baseline_raw19 = raw19[:, i0:i1]

    X = compute_features(baseline_raw19, srate_int, kalman_q, kalman_r)  # (T, 19, 10)
    flat = X.reshape(-1, X.shape[1] * X.shape[2])
    mean, std = flat.mean(axis=0), flat.std(axis=0) + 1e-8
    print(f"  Baseline window: [{start_s:.1f}s, {end_s:.1f}s] ({end_s - start_s:.1f}s, {X.shape[0]} seconds of features)")
    return mean, std


def predict_trial(trial_npz_path, model, mean, std, kalman_q, kalman_r, device):
    raw19, srate = load_session_raw19(trial_npz_path)
    srate_int = int(round(srate))
    X = compute_features(raw19, srate_int, kalman_q, kalman_r)

    shape = X.shape
    flat = X.reshape(-1, shape[1] * shape[2])
    flat = (flat - mean) / std
    X = flat.reshape(shape)

    with torch.no_grad():
        logits = model(torch.Tensor(X).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)
    return pred, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--npz-name", default="eeg_raw.npz")
    parser.add_argument("--markers-name", default="markers.csv")
    parser.add_argument("--trials-subdir", default="trials")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kalman-q", type=float, default=0.01)
    parser.add_argument("--kalman-r", type=float, default=0.5)
    parser.add_argument("--out-csv-name", default="dgcnn_own_predictions_sessionnorm.csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    device = torch.device(args.device)

    print(f"{session_dir.name}: computing in-session baseline stats...")
    mean, std = compute_session_baseline_stats(
        session_dir, args.npz_name, args.markers_name, args.kalman_q, args.kalman_r)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DGCNN(num_electrodes=19, in_channels=10, num_classes=3, k=2, relu_is=1,
                  layers=[64], dropout_rate=0.5)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    trial_dirs = sorted((session_dir / args.trials_subdir).glob("trial_*"))
    for trial_dir in trial_dirs:
        npz_path = trial_dir / args.npz_name
        if not npz_path.exists():
            continue
        pred, probs = predict_trial(npz_path, model, mean, std, args.kalman_q, args.kalman_r, device)

        out_csv = trial_dir / args.out_csv_name
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t_s", "pred_label", "prob_negative", "prob_neutral", "prob_positive"])
            for t, (c, p) in enumerate(zip(pred, probs)):
                writer.writerow([t, IDX_TO_LABEL[int(c)], f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])

        from collections import Counter
        counts = Counter(IDX_TO_LABEL[int(c)] for c in pred)
        total = len(pred)
        pct = {cls: counts[cls] / total * 100 for cls in ["negative", "neutral", "positive"]}
        print(f"  {trial_dir.name}: {total}s  neg={pct['negative']:.1f}%  "
              f"neu={pct['neutral']:.1f}%  pos={pct['positive']:.1f}%")


if __name__ == "__main__":
    main()
