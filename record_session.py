"""Screen recording + EEG recording + emotion-marker session for one video per run.

Run it: it connects to the CGX EEG LSL stream, starts recording your screen,
and starts logging raw EEG. Provide subject and video name at startup.
Press SPACE any time you feel a strong emotion; it logs the exact elapsed
recording time (and LSL time) so you can go back later and cut out strong
windows around each mark. Press Q to stop and save.

Output (in sessions/<subject>_<timestamp>/):
  screen_recording.mp4  - the screen recording
  eeg_raw.npz           - raw EEG: data (n_channels, n_samples), timestamps (LSL clock), labels
  markers.csv           - elapsed_s, wall_time, lsl_time, trial, event
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import mss
import keyboard
from pylsl import StreamInfo, StreamOutlet, local_clock

from video_writer import FFmpegVideoWriter

HERE = Path(__file__).parent
SESSIONS_DIR = HERE / "sessions"
LSL_STREAM_DIR = HERE.parent / "lsl_stream"
sys.path.insert(0, str(LSL_STREAM_DIR))

FPS = 15


def make_marker_outlet():
    info = StreamInfo(name="EmoStimMarkers", type="Markers", channel_count=1,
                       nominal_srate=0, channel_format="string", source_id="emostim_marker_1")
    return StreamOutlet(info)


def safe_token(text):
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return token.strip("_") or "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=None, help="subject/session ID")
    parser.add_argument("--video-name", default=None, help="video name shown in markers/logs")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--monitor", type=int, default=1, help="mss monitor index (1 = primary)")
    parser.add_argument("--mark-key", default="space")
    parser.add_argument("--no-eeg", action="store_true",
                         help="skip connecting to the CGX EEG LSL stream (screen+markers only)")
    args = parser.parse_args()

    subject = (args.subject or input("Subject ID: ")).strip()
    if not subject:
        raise ValueError("Subject ID cannot be empty.")

    video_name = (args.video_name or input("Video name: ")).strip()
    if not video_name:
        raise ValueError("Video name cannot be empty.")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = SESSIONS_DIR / f"{safe_token(subject)}_{safe_token(video_name)}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "screen_recording.mp4"
    markers_path = out_dir / "markers.csv"
    eeg_path = out_dir / "eeg_raw.npz"

    rx = None
    if not args.no_eeg:
        from receiver import EEGReceiver  # from ../lsl_stream
        print("Connecting to CGX EEG LSL stream...")
        rx = EEGReceiver(buffer_seconds=5.0)
        eeg_chunks = []
        eeg_timestamps = []
    else:
        print("--no-eeg set: recording screen + markers only, no EEG.")

    outlet = None
    try:
        outlet = make_marker_outlet()
        print("LSL marker outlet created (EmoStimMarkers).")
    except Exception as e:
        print(f"Could not create LSL outlet ({e}); continuing with local log only.")

    marker_rows = []
    trial_state = {"name": video_name}

    def log_event(event):
        wall = time.time()
        lsl_t = local_clock()
        marker_rows.append({"elapsed_s": round(wall - t_start, 3), "wall_time": wall,
                             "lsl_time": lsl_t, "trial": trial_state["name"], "event": event})
        if outlet is not None:
            outlet.push_sample([f"{event}|trial={trial_state['name']}"], lsl_t)

    mark_count = {"n": 0}

    def on_mark(_event):
        mark_count["n"] += 1
        log_event(f"emotion_mark_{mark_count['n']}")
        trial_label = trial_state["name"] or "(no trial set)"
        print(f"  [MARK #{mark_count['n']}] at t={time.time() - t_start:.1f}s  trial={trial_label}")

    stop_flag = {"stop": False}
    video_dead_warned = {"n": False}

    def on_quit(_event):
        stop_flag["stop"] = True

    with mss.mss() as sct:
        monitor = sct.monitors[args.monitor]
        width, height = monitor["width"], monitor["height"]

        writer = FFmpegVideoWriter(video_path, width, height, args.fps)
        if not writer.isOpened():
            raise RuntimeError("Could not start ffmpeg for video writing.")

        keyboard.on_press_key(args.mark_key, on_mark, suppress=False)
        keyboard.on_press_key("q", on_quit, suppress=False)

        print(f"Recording screen ({width}x{height} @ {args.fps}fps) to {video_path}")
        print(f"Subject: {subject}")
        print(f"Video: {video_name}")
        print(f"Press {args.mark_key.upper()} to mark strong emotion.")
        print("Press Q to stop.\n")

        t_start = time.time()
        log_event("recording_start")
        log_event(f"trial_start:{trial_state['name']}")
        frame_interval = 1.0 / args.fps
        next_frame_t = t_start

        try:
            while not stop_flag["stop"]:
                now = time.time()
                if now < next_frame_t:
                    time.sleep(max(0, next_frame_t - now))
                frame_bgra = np.array(sct.grab(monitor))
                frame_bgr = np.ascontiguousarray(frame_bgra[:, :, :3])  # drop alpha channel
                if not writer.write(frame_bgr) and not video_dead_warned["n"]:
                    video_dead_warned["n"] = True
                    print(f"WARNING: video encoder died at t={time.time() - t_start:.1f}s -- "
                          f"screen recording has stopped, but EEG keeps recording normally. "
                          f"See {writer.log_path} for why.")
                    log_event("video_writer_died")
                next_frame_t += frame_interval

                if rx is not None:
                    _n, chunk, ts = rx.pull_new_ts(timeout=0.0)
                    if chunk is not None:
                        eeg_chunks.append(chunk)  # (n_samples, n_channels)
                        eeg_timestamps.append(ts)  # (n_samples,) LSL time
        except KeyboardInterrupt:
            pass
        finally:
            log_event(f"trial_end:{trial_state['name']}")
            log_event("recording_end")
            writer.release()
            keyboard.unhook_all()

    with open(markers_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["elapsed_s", "wall_time", "lsl_time", "trial", "event"])
        w.writeheader()
        w.writerows(marker_rows)

    duration = marker_rows[-1]["elapsed_s"] if marker_rows else 0
    print(f"\nSaved recording ({duration:.1f}s) to {video_path}")
    print(f"Saved {mark_count['n']} marker(s) to {markers_path}")

    if rx is not None:
        if eeg_chunks:
            data = np.concatenate(eeg_chunks, axis=0).T  # (n_channels, n_samples)
            timestamps = np.concatenate(eeg_timestamps, axis=0)  # (n_samples,) LSL time
            np.savez(eeg_path, data=data, timestamps=timestamps, srate=rx.srate,
                     labels=np.array(rx.labels))
            print(f"Saved EEG ({data.shape[1]} samples, {data.shape[0]} channels) to {eeg_path}")
        else:
            print("WARNING: no EEG samples were captured during the session.")


if __name__ == "__main__":
    main()
