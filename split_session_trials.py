"""Split a session's continuous recording into per-trial pieces based on
markers.csv's trial_start:N / trial_end:N events (the newer recording
format used for Shine_2026072[89]_* sessions, which have a Pre phase
[buffer + open_eyes] then multiple video trials in one continuous EEG +
screen recording).

For each trial segment, writes into session_dir/trials/trial_XX/:
  - eeg_raw.npz   (data/timestamps sliced to the segment, same keys/labels/srate)
  - ratings.csv   (only rows whose elapsed_s falls in the segment, elapsed_s
                   and wall_time/lsl_time kept absolute -- but a trial-relative
                   `t_s` column is added for convenience)
  - screen_recording_h264.mp4  (video trimmed + transcoded to H.264 via ffmpeg)

If the same trial number appears more than once (a restarted trial), the
segments are numbered trial_04a, trial_04b, ... in chronological order.

Usage:
  "C:/Users/USER/miniconda3/envs/EEG/python.exe" split_session_trials.py \
      --session-dir sessions/Shine_20260728_143129
"""

import argparse
import csv
import re
import string
import subprocess
from collections import defaultdict
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def parse_trial_segments(markers_path):
    """Returns [(trial_num:int, start_s:float, end_s:float), ...] in
    chronological order, matching each trial_start:N with the next
    trial_end:N for the same N."""
    with open(markers_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    open_starts = {}  # trial_num -> start_s
    segments = []
    for r in rows:
        event = r["event"]
        t = float(r["elapsed_s"])
        m = re.match(r"trial_start:(\d+)", event)
        if m:
            n = int(m.group(1))
            open_starts[n] = t
            continue
        m = re.match(r"trial_end:(\d+)", event)
        if m:
            n = int(m.group(1))
            if n in open_starts:
                segments.append((n, open_starts.pop(n), t))
    return segments


def label_segments(segments):
    """Assigns trial_04a/trial_04b style labels when a trial number repeats."""
    counts = defaultdict(int)
    for n, _, _ in segments:
        counts[n] += 1
    seen = defaultdict(int)
    labels = []
    for n, s, e in segments:
        if counts[n] > 1:
            suffix = string.ascii_lowercase[seen[n]]
            labels.append(f"trial_{n:02d}{suffix}")
        else:
            labels.append(f"trial_{n:02d}")
        seen[n] += 1
    return labels


def split_npz(npz_path, start_s, end_s, out_path):
    with np.load(npz_path, allow_pickle=False) as z:
        data, timestamps, srate, labels = z["data"], z["timestamps"], float(z["srate"]), z["labels"]
    i0 = int(round(start_s * srate))
    i1 = min(data.shape[1], int(round(end_s * srate)))
    np.savez(out_path, data=data[:, i0:i1], timestamps=timestamps[i0:i1], srate=srate, labels=labels)
    return i1 - i0


def split_ratings(ratings_path, start_s, end_s, out_path):
    with open(ratings_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fieldnames = rows[0].keys() if rows else ["elapsed_s", "wall_time", "lsl_time", "trial", "category"]
    kept = [r for r in rows if start_s <= float(r["elapsed_s"]) < end_s]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames) + ["t_s"])
        writer.writeheader()
        for r in kept:
            r = dict(r)
            r["t_s"] = f"{float(r['elapsed_s']) - start_s:.3f}"
            writer.writerow(r)
    return len(kept)


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
    parser.add_argument("--ratings-name", default="ratings.csv")
    parser.add_argument("--markers-name", default="markers.csv")
    parser.add_argument("--video-name", default="screen_recording.mp4")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    segments = parse_trial_segments(session_dir / args.markers_name)
    if not segments:
        print(f"No trial_start/trial_end pairs found in {session_dir / args.markers_name}")
        return
    labels = label_segments(segments)

    out_root = session_dir / "trials"
    out_root.mkdir(exist_ok=True)

    print(f"{session_dir.name}: {len(segments)} trial segments")
    for (n, start_s, end_s), label in zip(segments, labels):
        trial_dir = out_root / label
        trial_dir.mkdir(exist_ok=True)
        n_samples = split_npz(session_dir / args.npz_name, start_s, end_s, trial_dir / "eeg_raw.npz")
        n_ratings = split_ratings(session_dir / args.ratings_name, start_s, end_s, trial_dir / "ratings.csv")
        split_video(session_dir / args.video_name, start_s, end_s, trial_dir / "screen_recording_h264.mp4")
        print(f"  {label}: trial={n} [{start_s:.1f}s, {end_s:.1f}s] "
              f"dur={end_s - start_s:.1f}s  {n_samples} samples  {n_ratings} ratings")


if __name__ == "__main__":
    main()
