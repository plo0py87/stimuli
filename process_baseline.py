"""Extract each session's Pre-phase resting baseline (buffer + eyes-open,
~100s before video_phase_start) as its own npz+video, run the DGCNN own-data
checkpoint on it (global train-pool normalization -- the same checkpoint
used for all the trial players), and generate an emotion_player.html for it.

This is a sanity check: since nothing emotionally relevant happens during
the resting baseline, a well-behaved model should predict close to neutral
throughout. Session-baseline normalization is deliberately NOT applied here
-- normalizing the baseline by its own mean/std would trivially center it
near zero and isn't a meaningful test of anything.

Usage:
  "C:/Users/USER/miniconda3/envs/EEG/python.exe" process_baseline.py \
      --session-dir sessions/Shine_20260728_143129 \
      --checkpoint result_dgcnn_own_data_split_session/checkpoint-scratch
"""

import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "emotion_model"))

from predict_dgcnn_own_sessionnorm import get_baseline_window  # noqa: E402


def split_npz(npz_path, start_s, end_s, out_path):
    with np.load(npz_path, allow_pickle=False) as z:
        data, timestamps, srate, labels = z["data"], z["timestamps"], float(z["srate"]), z["labels"]
    i0 = int(round(start_s * srate))
    i1 = min(data.shape[1], int(round(end_s * srate)))
    np.savez(out_path, data=data[:, i0:i1], timestamps=timestamps[i0:i1], srate=srate, labels=labels)
    return i1 - i0


def split_video(video_path, start_s, end_s, out_path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    duration = end_s - start_s
    cmd = [
        ffmpeg, "-y", "-ss", f"{start_s:.3f}", "-i", str(video_path), "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--npz-name", default="eeg_raw.npz")
    parser.add_argument("--markers-name", default="markers.csv")
    parser.add_argument("--video-name", default="screen_recording.mp4")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    start_s, end_s = get_baseline_window(session_dir / args.markers_name)
    print(f"{session_dir.name}: baseline window [{start_s:.1f}s, {end_s:.1f}s] ({end_s - start_s:.1f}s)")

    baseline_dir = session_dir / "baseline"
    baseline_dir.mkdir(exist_ok=True)

    n_samples = split_npz(session_dir / args.npz_name, start_s, end_s, baseline_dir / "eeg_raw.npz")
    print(f"  eeg_raw.npz: {n_samples} samples")

    video_path = session_dir / args.video_name
    has_video = video_path.exists()
    if has_video:
        split_video(video_path, start_s, end_s, baseline_dir / "screen_recording_h264.mp4")
        print("  video trimmed")
    else:
        print("  no source video found, skipping video trim")

    # Run inference (import here, after sys.path is set up, to reuse the
    # existing global-norm predict function).
    import predict_dgcnn_own
    pred, probs = predict_dgcnn_own.predict_session(
        baseline_dir / "eeg_raw.npz", args.checkpoint, kalman_q=0.01, kalman_r=0.5, device="cuda")

    import csv
    from collections import Counter
    from train_dagcn_own_data import IDX_TO_LABEL
    out_csv = baseline_dir / "dgcnn_own_predictions.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "pred_label", "prob_negative", "prob_neutral", "prob_positive"])
        for t, (c, p) in enumerate(zip(pred, probs)):
            writer.writerow([t, IDX_TO_LABEL[int(c)], f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])
    counts = Counter(IDX_TO_LABEL[int(c)] for c in pred)
    total = len(pred)
    print(f"  predictions: neg={counts['negative']} ({counts['negative']/total*100:.1f}%)  "
          f"neu={counts['neutral']} ({counts['neutral']/total*100:.1f}%)  "
          f"pos={counts['positive']} ({counts['positive']/total*100:.1f}%)")

    import make_emotion_player
    make_emotion_player.make_player(
        baseline_dir, "dgcnn_own_predictions.csv",
        "screen_recording_h264.mp4" if has_video else None,
        "emotion_player.html", ratings_name="ratings.csv")


if __name__ == "__main__":
    main()
