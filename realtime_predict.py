"""Real-time emotion prediction window: connects to the CGX Quick-20 LSL
stream, first records a short resting baseline (10s settle + 90s eyes-open,
matching the Pre phase in annotate_session.py) to compute THIS session's own
normalization mean/std, then computes causal DE+PSD features once per second
from a short rolling lookback window (so the causal bandpass filters have
settled by the time we read the newest second), keeps a persistent
(not-reset-per-tick) Kalman smoother state across ticks -- matching the
causal, online pipeline used everywhere else in this project -- and runs the
result through the DGCNN model trained on your own labeled data
(train_dgcnn_own_data.py).

Baseline normalization instead of the checkpoint's saved global train-pool
mean/std: see predict_dgcnn_own_sessionnorm.py / eval_own_data_accuracy.py
for the offline comparison that motivated trying this live.

Also records a full session just like record_session.py/annotate_session.py:
screen recording + raw EEG + markers + the live per-second prediction
timeline, saved to sessions/<subject>_realtime_<timestamp>/ when you stop
(button or window close). The saved dgcnn_own_predictions.csv is in the same
format predict_dgcnn_own.py produces, so you can run make_emotion_player.py
on the session afterward exactly like any other recording.

Usage:
  "C:/Users/USER/miniconda3/envs/EEG/python.exe" realtime_predict.py \
      --subject Shine \
      --checkpoint result_dgcnn_own_data_split_session/checkpoint-scratch
"""

import argparse
import csv
import sys
import time
import tkinter as tk
from pathlib import Path

import mss
import numpy as np
import torch
from pylsl import local_clock

HERE = Path(__file__).parent
SESSIONS_DIR = HERE / "sessions"
EMOTION_MODEL_DIR = HERE / "emotion_model"
LSL_STREAM_DIR = HERE.parent / "lsl_stream"

# lsl_stream/config.py and emotion_model/config/ (a package) both use the
# bare name `config`. Whichever gets imported first is cached in
# sys.modules and reused by the other side regardless of sys.path order, so
# import the LSL receiver in isolation first, then clear the `config` cache
# before importing anything from emotion_model.
sys.path.insert(0, str(LSL_STREAM_DIR))
from receiver import EEGReceiver  # noqa: E402
sys.path.remove(str(LSL_STREAM_DIR))
for _mod_name in list(sys.modules):
    if _mod_name == "config" or _mod_name.startswith("config."):
        del sys.modules[_mod_name]

from video_writer import FFmpegVideoWriter  # noqa: E402

sys.path.insert(0, str(EMOTION_MODEL_DIR))
from models.DGCNN import DGCNN  # noqa: E402
from DAGCN_quick20_realtime_train import causal_psd_features  # noqa: E402
from DGCNN_quick20_realtime_train import causal_de_features, kalman_smooth_trial  # noqa: E402
from train_dagcn_own_data import DEVICE_TO_MODEL_LABEL, MODEL_LABEL_TO_DEVICE, QUICK20_CHANNEL_NAME  # noqa: E402

IDENTITY_INDICES = list(range(19))
LABEL_NAMES = ["negative", "neutral", "positive"]
COLORS = {"negative": "#ff6b6b", "neutral": "#ffd93d", "positive": "#6bcB77"}

LOOKBACK_S = 10.0  # context fed to the causal filters so they settle before the newest second
TICK_MS = 1000
VIDEO_FPS = 15

# Baseline phase, matching annotate_session.py's Pre phase: buffer lets the
# causal filters settle (discarded), eyes-open resting is what actually
# gets used to compute this session's own normalization mean/std.
BASELINE_BUFFER_S = 10.0
BASELINE_REST_S = 90.0
BASELINE_TOTAL_S = BASELINE_BUFFER_S + BASELINE_REST_S


class RealtimeApp:
    def __init__(self, subject, checkpoint_path, kalman_q, kalman_r, device, fps, monitor, blind=False):
        self.blind = blind
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.out_dir = SESSIONS_DIR / f"{subject}_realtime_{ts}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.out_dir / "screen_recording.mp4"
        self.eeg_path = self.out_dir / "eeg_raw.npz"
        self.markers_path = self.out_dir / "markers.csv"
        self.pred_csv_path = self.out_dir / "dgcnn_own_predictions.csv"

        print("Connecting to CGX EEG LSL stream...")
        self.rx = EEGReceiver(buffer_seconds=BASELINE_TOTAL_S + LOOKBACK_S + 10.0)
        self.srate = self.rx.srate
        self.device_labels = {lb: i for i, lb in enumerate(self.rx.labels)}
        missing = [d for d in DEVICE_TO_MODEL_LABEL if d not in self.device_labels]
        if missing:
            raise RuntimeError(f"LSL stream is missing expected channels: {missing}")

        self.kalman_q = kalman_q
        self.kalman_r = kalman_r
        self.x_est = None
        self.p_est = None
        self.n_samples_seen = 0
        self.history = []  # list of (neg, neu, pos)

        self.baseline_done = False
        self.mean = None
        self.std = None

        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = DGCNN(num_electrodes=19, in_channels=10, num_classes=3, k=2, relu_is=1,
                            layers=[64], dropout_rate=0.5)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()

        # Recording state -- same shape as record_session.py/annotate_session.py:
        # accumulate every EEG chunk independently of the small rolling ring
        # buffer (which evicts old data), so eeg_raw.npz has the FULL session.
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor]
        width, height = self.monitor["width"], self.monitor["height"]
        self.writer = FFmpegVideoWriter(self.video_path, width, height, fps)
        if not self.writer.isOpened():
            raise RuntimeError("Could not start ffmpeg for video writing.")
        self.video_tick_interval = int(1000 / fps)
        self.video_dead_warned = False

        self.eeg_chunks = []
        self.eeg_timestamps = []
        self.marker_rows = []
        self.pred_rows = []  # dicts: t_s, pred_label, prob_negative, prob_neutral, prob_positive
        self.stopped = False
        self.t_start = time.time()

        self._build_gui()
        self.log_marker("recording_start")
        self.video_after_id = self.root.after(self.video_tick_interval, self._video_tick)
        self.tick_after_id = self.root.after(200, self.tick)

    def log_marker(self, event):
        wall = time.time()
        lsl_t = local_clock()
        self.marker_rows.append({"elapsed_s": round(wall - self.t_start, 3), "wall_time": wall,
                                  "lsl_time": lsl_t, "event": event})

    def _build_gui(self):
        self.root = tk.Tk()
        title = "Real-time emotion recording (blind mode -- no live feedback)" if self.blind \
            else "Real-time emotion prediction (baseline-normalized, recording)"
        self.root.title(title)
        self.root.configure(bg="#111111")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

        self.status_var = tk.StringVar(value="Warming up...")
        tk.Label(self.root, textvariable=self.status_var, bg="#111111", fg="#999999",
                 font=("Segoe UI", 10)).pack(pady=(10, 4))

        if self.blind:
            # No prediction label, no score bars, no history graph -- nothing
            # that could leak the model's live output and bias how you feel.
            # Just a neutral recording indicator + elapsed time (via status_var).
            tk.Label(self.root, text="RECORDING", bg="#111111", fg="#666666",
                     font=("Segoe UI", 20, "bold")).pack(pady=(0, 10))
            tk.Frame(self.root, bg="#111111", width=320, height=100).pack(padx=16, pady=(10, 16))
        else:
            self.pred_var = tk.StringVar(value="-")
            tk.Label(self.root, textvariable=self.pred_var, bg="#111111", fg="#ffffff",
                     font=("Segoe UI", 20, "bold")).pack(pady=(0, 10))

            self.bar_canvases = {}
            self.pct_vars = {}
            for name in LABEL_NAMES:
                row = tk.Frame(self.root, bg="#111111")
                row.pack(fill="x", padx=16, pady=4)
                tk.Label(row, text=name.capitalize(), width=10, anchor="w",
                         bg="#111111", fg=COLORS[name], font=("Segoe UI", 11)).pack(side="left")
                pct_var = tk.StringVar(value="0%")
                self.pct_vars[name] = pct_var
                tk.Label(row, textvariable=pct_var, width=6, anchor="e",
                         bg="#111111", fg="#ffffff", font=("Segoe UI", 11)).pack(side="right")
                track = tk.Canvas(row, height=14, bg="#333333", highlightthickness=0)
                track.pack(side="left", fill="x", expand=True, padx=8)
                self.bar_canvases[name] = track

            self.history_canvas = tk.Canvas(self.root, height=100, bg="#1c1c1c", highlightthickness=0)
            self.history_canvas.pack(fill="x", padx=16, pady=(10, 16))

        tk.Button(self.root, text="Stop & Save", font=("Segoe UI", 11), bg="#a33", fg="white",
                  command=self.stop).pack(fill="x", padx=16, pady=(0, 16))

    def compute_causal_features(self, raw19):
        srate_int = int(round(self.srate))
        de = causal_de_features(raw19, srate_int, IDENTITY_INDICES)
        psd = causal_psd_features(raw19, srate_int, IDENTITY_INDICES)
        return [np.concatenate([d, p], axis=-1) for d, p in zip(de, psd)]

    def _video_tick(self):
        if self.stopped:
            return
        frame_bgra = np.array(self.sct.grab(self.monitor))
        frame_bgr = np.ascontiguousarray(frame_bgra[:, :, :3])
        if not self.writer.write(frame_bgr) and not self.video_dead_warned:
            self.video_dead_warned = True
            print(f"WARNING: video encoder died at t={time.time() - self.t_start:.1f}s -- "
                  f"screen recording has stopped, but EEG/predictions keep recording normally. "
                  f"See {self.writer.log_path} for why.")
            self.log_marker("video_writer_died")

        n, chunk, chunk_ts = self.rx.pull_new_ts(timeout=0.0)
        if chunk is not None:
            self.eeg_chunks.append(chunk)
            self.eeg_timestamps.append(chunk_ts)

        self.video_after_id = self.root.after(self.video_tick_interval, self._video_tick)

    def _log_pred_row(self, pred_label, probs):
        self.pred_rows.append({
            "t_s": len(self.pred_rows),
            "pred_label": pred_label,
            "prob_negative": float(probs[0]),
            "prob_neutral": float(probs[1]),
            "prob_positive": float(probs[2]),
        })

    def tick(self):
        if self.stopped:
            return
        _data, ts = self.rx.buffer.get_ordered()
        prev_n = getattr(self, "n_samples_seen", 0)
        self.n_samples_seen = len(ts)

        # Diagnostic: the ring buffer should only ever grow (or plateau once
        # full) -- if it ever drops, that's a real disconnect/reconnect on
        # the LSL/hardware side, not something this script's logic can cause.
        print(f"[tick] buffer_len={self.n_samples_seen} "
              f"write_idx={self.rx.buffer.write_idx} filled={self.rx.buffer.filled}"
              + (" <-- DROPPED (buffer shrank!)" if self.n_samples_seen < prev_n else ""))

        if not self.baseline_done:
            self._baseline_tick()
            return

        need_samples = int(LOOKBACK_S * self.srate)
        if self.n_samples_seen < need_samples:
            self.status_var.set(
                f"Warming up... {self.n_samples_seen / self.srate:.1f}s / {LOOKBACK_S:.0f}s buffered")
            self._log_pred_row("warming_up", (0.0, 1.0, 0.0))
            self.tick_after_id = self.root.after(TICK_MS, self.tick)
            return

        win = self.rx.latest_window(need_samples)  # (n_device_channels, need_samples)
        raw19 = np.stack(
            [win[self.device_labels[MODEL_LABEL_TO_DEVICE[model_lb]]] for model_lb in QUICK20_CHANNEL_NAME]
        ).astype(np.float64)

        combined = self.compute_causal_features(raw19)
        raw_new = combined[-1]  # newest full second, after the filters have had LOOKBACK_S-1s to settle

        p_pred = self.p_est + self.kalman_q
        k = p_pred / (p_pred + self.kalman_r)
        self.x_est = self.x_est + k * (raw_new - self.x_est)
        self.p_est = (1 - k) * p_pred

        shape = self.x_est.shape
        flat = self.x_est.reshape(1, shape[0] * shape[1])
        flat = (flat - self.mean) / self.std
        x = flat.reshape(1, *shape)

        with torch.no_grad():
            logits = self.model(torch.Tensor(x).to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        self.history.append(tuple(probs))
        self.history = self.history[-120:]
        self.status_var.set(f"t={self.n_samples_seen / self.srate:.0f}s  |  live @ {1000/TICK_MS:.0f}Hz  "
                             f"(baseline-normalized, recording{', blind' if self.blind else ''})")
        if not self.blind:
            self._update_display(probs)
        self._log_pred_row(LABEL_NAMES[int(np.argmax(probs))], probs)

        self.tick_after_id = self.root.after(TICK_MS, self.tick)

    def _baseline_tick(self):
        need_samples = int(BASELINE_TOTAL_S * self.srate)
        if self.n_samples_seen < need_samples:
            elapsed = self.n_samples_seen / self.srate
            if elapsed < BASELINE_BUFFER_S:
                self.status_var.set(
                    f"Baseline: settling... {elapsed:.0f}s / {BASELINE_BUFFER_S:.0f}s")
            else:
                self.status_var.set(
                    f"Baseline: please rest, eyes open... "
                    f"{elapsed - BASELINE_BUFFER_S:.0f}s / {BASELINE_REST_S:.0f}s")
            if not self.blind:
                self.pred_var.set("BASELINE")
            self._log_pred_row("baseline", (0.0, 1.0, 0.0))
            self.tick_after_id = self.root.after(TICK_MS, self.tick)
            return

        # Enough data collected -- compute this session's own baseline mean/std,
        # same procedure as predict_dgcnn_own_sessionnorm.py's
        # compute_session_baseline_stats(): slice OFF the buffer portion from
        # the raw EEG first, then run the causal filters + Kalman smoother
        # fresh starting exactly at the eyes-open onset (not at the buffer
        # onset). This matters because the filters/Kalman recursion aren't
        # memoryless -- starting them at a different point in time produces
        # measurably different mean/std (verified: median ~5-15% relative
        # difference vs. discarding computed features post-hoc instead of
        # slicing the raw input first).
        win = self.rx.latest_window(need_samples)
        raw19 = np.stack(
            [win[self.device_labels[MODEL_LABEL_TO_DEVICE[model_lb]]] for model_lb in QUICK20_CHANNEL_NAME]
        ).astype(np.float64)
        buffer_samples = int(BASELINE_BUFFER_S * self.srate)
        open_eyes_raw19 = raw19[:, buffer_samples:]
        combined = self.compute_causal_features(open_eyes_raw19)
        combined = kalman_smooth_trial(combined, self.kalman_q, self.kalman_r)

        rest_features = np.stack(combined)  # (BASELINE_REST_S, 19, 10)
        flat = rest_features.reshape(-1, rest_features.shape[1] * rest_features.shape[2])
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0) + 1e-8

        # Carry the Kalman state forward from the end of the baseline window
        # into live tracking, instead of resetting it, so there's no jump at
        # the baseline->live transition.
        shape = combined[-1].shape
        self.x_est = combined[-1].copy()
        self.p_est = np.full(shape, self.kalman_q)  # p_est after many steps converges near q under steady r

        self.baseline_done = True
        self.log_marker("baseline_end")
        print(f"Baseline complete: mean/std computed from {rest_features.shape[0]}s of eyes-open resting data.")
        self.tick_after_id = self.root.after(TICK_MS, self.tick)

    def _update_display(self, probs):
        pred_idx = int(np.argmax(probs))
        pred_name = LABEL_NAMES[pred_idx]
        self.pred_var.set(pred_name.upper())

        for i, name in enumerate(LABEL_NAMES):
            pct = probs[i] * 100
            self.pct_vars[name].set(f"{pct:.0f}%")
            canvas = self.bar_canvases[name]
            canvas.delete("all")
            w = canvas.winfo_width() or 200
            h = canvas.winfo_height()
            canvas.create_rectangle(0, 0, w * probs[i], h, fill=COLORS[name], width=0)

        self._draw_history()

    def _draw_history(self):
        c = self.history_canvas
        c.delete("all")
        w = c.winfo_width() or 400
        h = c.winfo_height()
        n = len(self.history)
        if n < 2:
            return
        for series_i, name in enumerate(LABEL_NAMES):
            points = []
            for i, probs in enumerate(self.history):
                x = (i / (n - 1)) * w
                y = h - probs[series_i] * h
                points.extend([x, y])
            c.create_line(*points, fill=COLORS[name], width=2)

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        for attr in ("tick_after_id", "video_after_id"):
            after_id = getattr(self, attr, None)
            if after_id is not None:
                try:
                    self.root.after_cancel(after_id)
                except Exception:
                    pass
        self.log_marker("recording_end")
        self.writer.release()
        self.sct.close()
        self._save()
        self.root.destroy()

    def _save(self):
        with open(self.markers_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["elapsed_s", "wall_time", "lsl_time", "event"])
            w.writeheader()
            w.writerows(self.marker_rows)

        with open(self.pred_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "pred_label", "prob_negative", "prob_neutral", "prob_positive"])
            for row in self.pred_rows:
                w.writerow([row["t_s"], row["pred_label"], f"{row['prob_negative']:.4f}",
                            f"{row['prob_neutral']:.4f}", f"{row['prob_positive']:.4f}"])

        if self.eeg_chunks:
            data = np.concatenate(self.eeg_chunks, axis=0).T  # (n_channels, n_samples)
            timestamps = np.concatenate(self.eeg_timestamps, axis=0)
            np.savez(self.eeg_path, data=data, timestamps=timestamps, srate=self.rx.srate,
                     labels=np.array(self.rx.labels))
            print(f"Saved EEG ({data.shape[1]} samples, {data.shape[0]} channels) to {self.eeg_path}")
        else:
            print("WARNING: no EEG samples were captured during the session.")

        print(f"Saved video to {self.video_path}")
        print(f"Saved {len(self.marker_rows)} marker(s) to {self.markers_path}")
        print(f"Saved {len(self.pred_rows)} prediction(s) to {self.pred_csv_path}")
        print(f"\nTo visualize: \"C:/Users/USER/miniconda3/python.exe\" make_emotion_player.py "
              f"--session-dir {self.out_dir} --csv-name dgcnn_own_predictions.csv")

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=None, help="subject/session ID")
    parser.add_argument("--checkpoint", default="result_dgcnn_own_data_split_session/checkpoint-scratch")
    parser.add_argument("--kalman-q", type=float, default=0.01)
    parser.add_argument("--kalman-r", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--monitor", type=int, default=1, help="mss monitor index (1 = primary)")
    parser.add_argument("--blind", action="store_true",
                         help="hide all live prediction feedback (label, score bars, history graph) so "
                              "watching the model's output can't bias how you feel during recording -- "
                              "everything is still recorded and saved for review afterward.")
    args = parser.parse_args()

    subject = (args.subject or input("Subject ID: ")).strip()
    if not subject:
        raise ValueError("Subject ID cannot be empty.")

    app = RealtimeApp(subject, args.checkpoint, args.kalman_q, args.kalman_r, args.device, args.fps, args.monitor,
                       blind=args.blind)
    print(f"Recording to {app.out_dir}")
    print(f"Window open. First {BASELINE_TOTAL_S:.0f}s ({BASELINE_BUFFER_S:.0f}s settle + "
          f"{BASELINE_REST_S:.0f}s eyes-open rest) records your session baseline -- sit still, "
          f"eyes open, no stimulus. Live prediction starts automatically after that.")
    if args.blind:
        print("Blind mode: no live prediction feedback will be shown.")
    print("Click 'Stop & Save', close the window, or Ctrl+C to stop and save the recording.")
    app.run()


if __name__ == "__main__":
    main()
