"""Run a DAGCN checkpoint trained by train_dagcn_own_data.py against a
session's full continuous eeg_raw.npz timeline (not just the rated windows),
producing a per-second predictions CSV compatible with make_emotion_player.py.

Usage:
  "C:/Users/USER/miniconda3/envs/EEG/python.exe" predict_dagcn_own.py \
      --session-dir sessions/Shine_20260720_114710 \
      --checkpoint result_dagcn_own_data_split_session/checkpoint-scratch
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

from DAGCN_quick20_realtime_train import causal_psd_features, DAGCNQuick20Wrapper  # noqa: E402
from DGCNN_quick20_realtime_train import causal_de_features, kalman_smooth_trial  # noqa: E402
from train_dagcn_own_data import load_session_raw19, IDENTITY_INDICES, IDX_TO_LABEL  # noqa: E402


def predict_session(npz_path, checkpoint_path, kalman_q, kalman_r, device="cuda"):
    raw19, srate = load_session_raw19(npz_path)
    srate_int = int(round(srate))

    de = causal_de_features(raw19, srate_int, IDENTITY_INDICES)
    psd = causal_psd_features(raw19, srate_int, IDENTITY_INDICES)
    combined = [np.concatenate([d, p], axis=-1) for d, p in zip(de, psd)]
    combined = kalman_smooth_trial(combined, kalman_q, kalman_r)
    X = np.stack(combined)  # (T, 19, 10)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mean, std = ckpt["mean"], ckpt["std"]
    shape = X.shape
    flat = X.reshape(-1, shape[1] * shape[2])
    flat = (flat - mean) / std
    X = flat.reshape(shape)

    model = DAGCNQuick20Wrapper(19, 10, 3)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(torch.Tensor(X).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)
    return pred, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--npz-name", default="eeg_raw.npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kalman-q", type=float, default=0.01)
    parser.add_argument("--kalman-r", type=float, default=0.5)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    npz_path = session_dir / args.npz_name
    pred, probs = predict_session(npz_path, args.checkpoint, args.kalman_q, args.kalman_r, args.device)

    out_csv = Path(args.out_csv) if args.out_csv else session_dir / "dagcn_own_predictions.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "pred_label", "prob_negative", "prob_neutral", "prob_positive"])
        for t, (c, p) in enumerate(zip(pred, probs)):
            writer.writerow([t, IDX_TO_LABEL[int(c)], f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])

    from collections import Counter
    counts = Counter(IDX_TO_LABEL[int(c)] for c in pred)
    total = len(pred)
    print(f"{session_dir.name}: {total}s total")
    for cls in ["negative", "neutral", "positive"]:
        print(f"  {cls}: {counts[cls]} ({counts[cls]/total*100:.1f}%)")
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
