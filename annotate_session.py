"""Full recorded-session protocol: Pre-rest -> video task (periodic-prompt
emotion annotation) -> Post-rest.

Pre:  buffer 10s -> eyes-open fixation-cross 90s
      (runs automatically as soon as the app starts)
(video task): every N seconds a small popup asks how you feel right now —
      one click picks a category (正向/偏正/中性/偏負/負向), then a second
      click rates how confident you are in that answer (1-5, asked BEFORE
      you ever see any model prediction, so it's usable later to filter out
      low-confidence labels without post-hoc bias). Popup closes, recording
      continues uninterrupted. You watch/play video yourself.
Post: buffer 10s -> eyes-open fixation-cross 45s
      (triggered by the "Finish video -> Post rest" button once you're done
      watching; auto-saves and stops when it completes)

Runs alongside a full screen recording and raw EEG capture, continuously
from the very start of Pre through the end of Post. Everything (screen
capture, EEG pulling, phase timers, the periodic popup) is driven by
Tkinter's own event loop via .after(), so there's no separate thread
touching the GUI — avoids Tkinter's not-thread-safe pitfalls entirely.

Output (in sessions/<subject>_<timestamp>/):
  screen_recording.mp4   - the screen recording (covers Pre + video task + Post)
  eeg_raw.npz            - raw EEG: data (n_channels, n_samples), timestamps (LSL clock), labels
  ratings.csv            - elapsed_s, wall_time, lsl_time, trial, category, confidence   (one row per prompt answered)
  markers.csv            - elapsed_s, wall_time, lsl_time, trial, event     (phase_start/end, prompt_*, trial_*, ...)
"""

import argparse
import csv
import sys
import time
import tkinter as tk
from pathlib import Path

import numpy as np
import mss
from pylsl import StreamInfo, StreamOutlet, local_clock

from video_writer import FFmpegVideoWriter

HERE = Path(__file__).parent
SESSIONS_DIR = HERE / "sessions"
LSL_STREAM_DIR = HERE.parent / "lsl_stream"
sys.path.insert(0, str(LSL_STREAM_DIR))

FPS = 15
PROMPT_INTERVAL_S = 20

CATEGORIES = [
    ("正向", "#2a8f4a"),
    ("偏正", "#6ab04c"),
    ("中性", "#666"),
    ("偏負", "#c67a4e"),
    ("負向", "#a33"),
]

# (key, display label, duration in seconds). "open_eyes" phases show a
# fixation cross instead of the label text.
PRE_PHASES = [
    ("buffer", "請稍候", 10),
    ("open_eyes", "睜眼注視十字", 90),
]
POST_PHASES = [
    ("buffer", "請稍候", 10),
    ("open_eyes", "睜眼注視十字", 45),
]


def make_outlets():
    rating_info = StreamInfo(name="EmotionCategory", type="Ratings", channel_count=1,
                              nominal_srate=0, channel_format="string", source_id="emotion_category_1")
    marker_info = StreamInfo(name="EmoStimMarkers", type="Markers", channel_count=1,
                              nominal_srate=0, channel_format="string", source_id="emostim_marker_1")
    return StreamOutlet(rating_info), StreamOutlet(marker_info)


class AnnotatorApp:
    def __init__(self, args):
        self.args = args

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.out_dir = SESSIONS_DIR / f"{args.subject}_{ts}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.out_dir / "screen_recording.mp4"
        self.ratings_path = self.out_dir / "ratings.csv"
        self.markers_path = self.out_dir / "markers.csv"
        self.eeg_path = self.out_dir / "eeg_raw.npz"

        self.rx = None
        self.eeg_chunks = []
        self.eeg_timestamps = []
        if not args.no_eeg:
            from receiver import EEGReceiver  # from ../lsl_stream
            print("Connecting to CGX EEG LSL stream...")
            self.rx = EEGReceiver(buffer_seconds=5.0)
        else:
            print("--no-eeg set: recording screen + ratings only, no EEG.")

        try:
            self.rating_outlet, self.marker_outlet = make_outlets()
            print("LSL outlets created: EmotionCategory, EmoStimMarkers.")
        except Exception as e:
            print(f"Could not create LSL outlets ({e}); continuing with local logs only.")
            self.rating_outlet, self.marker_outlet = None, None

        self.sct = mss.mss()
        self.monitor = self.sct.monitors[args.monitor]
        width, height = self.monitor["width"], self.monitor["height"]
        self.writer = FFmpegVideoWriter(self.video_path, width, height, args.fps)
        if not self.writer.isOpened():
            raise RuntimeError("Could not start ffmpeg for video writing.")

        self.rating_rows = []
        self.marker_rows = []
        self.trial_name = ""
        self.popup = None
        self.pending_category = None
        self.phase_win = None
        self.prompt_after_id = None
        self.tick_after_id = None
        self.phase_after_id = None
        self.video_phase_active = False
        self.stopped = False
        self.video_dead_warned = False
        self.t_start = time.time()

        self._build_ui()
        self.log_marker("recording_start")

        self.tick_interval = int(1000 / args.fps)
        self.tick_after_id = self.root.after(self.tick_interval, self.tick)

        # recording (screen+EEG) covers the whole thing; the annotated
        # video task only starts once Pre-rest finishes.
        self._run_phase_sequence(PRE_PHASES, "Pre", on_complete=self._start_video_phase)

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Emotion Annotation")
        self.root.attributes("-topmost", True)

        self.status_var = tk.StringVar(value="Pre-rest starting...")
        tk.Label(self.root, text="Recording", font=("Segoe UI", 12, "bold")).pack(pady=(10, 0), padx=16)
        tk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", 9), fg="gray",
                 wraplength=160, justify="center").pack()

        tk.Label(self.root, text="Trial name:", font=("Segoe UI", 9)).pack(pady=(12, 0))
        self.trial_entry = tk.Entry(self.root, width=16)
        self.trial_entry.pack()
        self.trial_entry.bind("<Return>", lambda _e: self.set_trial())
        self.set_trial_button = tk.Button(self.root, text="Set trial", command=self.set_trial)
        self.set_trial_button.pack(pady=2)
        self.end_trial_button = tk.Button(self.root, text="End trial", command=self.end_trial,
                                           state="disabled")
        self.end_trial_button.pack(pady=2)

        self.trial_label_var = tk.StringVar(value="(no trial set)")
        tk.Label(self.root, textvariable=self.trial_label_var, font=("Segoe UI", 8), fg="gray").pack()

        self.finish_video_button = tk.Button(
            self.root, text="Finish video ->\nStart Post-rest", command=self._start_post_phase,
            bg="#a60", fg="white", state="disabled")
        self.finish_video_button.pack(pady=(12, 4))

        tk.Button(self.root, text="Stop && Save", command=self.stop,
                  bg="#c44", fg="white").pack(pady=(4, 4))

        tk.Frame(self.root, height=10).pack()  # bottom margin

        # size the window to fit its actual content, then place it in the
        # bottom-right corner with extra clearance for the Windows taskbar.
        self.root.update_idletasks()
        win_w, win_h = self.root.winfo_reqwidth(), self.root.winfo_reqheight()
        margin_x, margin_y = 40, 90
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = screen_w - win_w - margin_x
        y = screen_h - win_h - margin_y
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")  # bottom-right corner, out of the way

        self.root.protocol("WM_DELETE_WINDOW", self.stop)

    # ---- Pre/Post phase sequencing ------------------------------------

    def _run_phase_sequence(self, phases, seq_name, on_complete):
        self._phase_queue = list(phases)
        self._phase_seq_name = seq_name
        self._phase_on_complete = on_complete
        self._next_phase()

    def _next_phase(self):
        if not self._phase_queue:
            if self.phase_win is not None:
                self.phase_win.destroy()
                self.phase_win = None
            self.log_marker(f"phase_sequence_end:{self._phase_seq_name}")
            self._phase_on_complete()
            return

        key, label, duration = self._phase_queue.pop(0)
        self.log_marker(f"phase_start:{self._phase_seq_name}:{key}")
        self.status_var.set(f"{self._phase_seq_name}: {label}")
        self._show_phase_screen(key, label)
        self._phase_key = key
        self._phase_end_time = time.time() + duration
        self._update_phase_countdown()

    def _show_phase_screen(self, key, label):
        if self.phase_win is None:
            self.phase_win = tk.Toplevel(self.root)
            self.phase_win.attributes("-fullscreen", True)
            self.phase_win.attributes("-topmost", True)
            self.phase_win.configure(bg="black")
            self.phase_text_var = tk.StringVar()
            self.phase_label_widget = tk.Label(
                self.phase_win, textvariable=self.phase_text_var, bg="black", fg="white")
            self.phase_label_widget.place(relx=0.5, rely=0.45, anchor="center")
            self.phase_countdown_var = tk.StringVar()
            tk.Label(self.phase_win, textvariable=self.phase_countdown_var, bg="black", fg="#888",
                     font=("Segoe UI", 14)).place(relx=0.5, rely=0.65, anchor="center")

        if key == "open_eyes":
            self.phase_label_widget.configure(font=("Segoe UI", 120, "bold"))
            self.phase_text_var.set("+")
        else:
            self.phase_label_widget.configure(font=("Segoe UI", 36, "bold"))
            self.phase_text_var.set(label)

    def _update_phase_countdown(self):
        if self.stopped:
            return
        remaining = self._phase_end_time - time.time()
        if remaining <= 0:
            self.log_marker(f"phase_end:{self._phase_seq_name}:{self._phase_key}")
            self._next_phase()
            return
        self.phase_countdown_var.set(f"{remaining:.0f}s")
        self.phase_after_id = self.root.after(200, self._update_phase_countdown)

    def _start_video_phase(self):
        self.video_phase_active = True
        self.status_var.set("Video task running")
        self.finish_video_button.configure(state="normal")
        self.log_marker("video_phase_start")
        self.prompt_after_id = self.root.after(self.args.prompt_interval * 1000, self.show_prompt)

    def _start_post_phase(self):
        if not self.video_phase_active:
            return
        self.video_phase_active = False
        self.finish_video_button.configure(state="disabled")

        if self.prompt_after_id is not None:
            self.root.after_cancel(self.prompt_after_id)
            self.prompt_after_id = None
        if self.popup is not None:
            self.popup.destroy()
            self.popup = None

        self.log_marker("video_phase_end")
        self._run_phase_sequence(POST_PHASES, "Post", on_complete=self.stop)

    # ---- trial / rating / marker logging -------------------------------

    def set_trial(self):
        if self.trial_name:
            # a trial is already active — must End trial before starting another.
            return
        new_name = self.trial_entry.get().strip()
        if not new_name:
            return
        self.trial_name = new_name
        self.log_marker(f"trial_start:{new_name}")
        self.trial_label_var.set(new_name)
        self.trial_entry.configure(state="disabled")
        self.set_trial_button.configure(state="disabled")
        self.end_trial_button.configure(state="normal")

    def end_trial(self):
        if not self.trial_name:
            return
        self.log_marker(f"trial_end:{self.trial_name}")
        self.trial_name = ""
        self.trial_label_var.set("(no trial set)")
        self.trial_entry.configure(state="normal")
        self.trial_entry.delete(0, "end")
        self.set_trial_button.configure(state="normal")
        self.end_trial_button.configure(state="disabled")

    def log_marker(self, event):
        wall = time.time()
        lsl_t = local_clock()
        self.marker_rows.append({"elapsed_s": round(wall - self.t_start, 3), "wall_time": wall,
                                  "lsl_time": lsl_t, "trial": self.trial_name, "event": event})
        if self.marker_outlet is not None:
            self.marker_outlet.push_sample([f"{event}|trial={self.trial_name}"], lsl_t)

    def log_rating(self, category, confidence):
        wall = time.time()
        lsl_t = local_clock()
        self.rating_rows.append({"elapsed_s": round(wall - self.t_start, 3), "wall_time": wall,
                                  "lsl_time": lsl_t, "trial": self.trial_name, "category": category,
                                  "confidence": confidence})
        if self.rating_outlet is not None:
            self.rating_outlet.push_sample([f"{category}|confidence={confidence}"], lsl_t)

    def show_prompt(self):
        if self.stopped:
            return
        if self.popup is not None:
            # previous prompt never got answered — skip this cycle, try again later.
            self.prompt_after_id = self.root.after(self.args.prompt_interval * 1000, self.show_prompt)
            return

        self.log_marker("prompt_shown")
        self.pending_category = None

        popup = tk.Toplevel(self.root)
        self.popup = popup
        popup.title("How do you feel right now?")
        popup.attributes("-topmost", True)
        self._build_category_step(popup)
        popup.protocol("WM_DELETE_WINDOW", lambda: self._on_answer("(dismissed)", ""))

    def _center_popup(self, popup):
        popup.update_idletasks()
        pw, ph = popup.winfo_reqwidth(), popup.winfo_reqheight()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        popup.geometry(f"{pw}x{ph}+{(screen_w - pw)//2}+{(screen_h - ph)//2}")

    def _build_category_step(self, popup):
        tk.Label(popup, text="How do you feel right now?",
                 font=("Segoe UI", 11, "bold")).pack(pady=(12, 8), padx=16)

        for label, color in CATEGORIES:
            tk.Button(popup, text=label, font=("Segoe UI", 11), bg=color, fg="white",
                      width=16, command=lambda c=label: self._on_category(c)).pack(pady=3, padx=16)

        tk.Frame(popup, height=12).pack()  # bottom margin
        self._center_popup(popup)

    def _on_category(self, category):
        self.pending_category = category
        self.log_marker(f"category_answered:{category}")
        if self.popup is None:
            return
        for child in list(self.popup.winfo_children()):
            child.destroy()
        self._build_confidence_step(self.popup)

    def _build_confidence_step(self, popup):
        tk.Label(popup, text="How confident are you in that answer?\n(1 = just guessing, 5 = very confident)",
                 font=("Segoe UI", 11, "bold"), justify="center").pack(pady=(12, 8), padx=16)

        row = tk.Frame(popup)
        row.pack(pady=3, padx=16)
        for value in range(1, 6):
            tk.Button(row, text=str(value), font=("Segoe UI", 11), bg="#444", fg="white",
                      width=3, command=lambda v=value: self._on_answer(self.pending_category, v)).pack(
                side="left", padx=3)

        tk.Frame(popup, height=12).pack()  # bottom margin
        self._center_popup(popup)

    def _on_answer(self, category, confidence):
        self.log_rating(category, confidence)
        self.log_marker(f"prompt_answered:{category}|confidence={confidence}")
        if self.popup is not None:
            self.popup.destroy()
            self.popup = None
        if self.video_phase_active:
            self.prompt_after_id = self.root.after(self.args.prompt_interval * 1000, self.show_prompt)

    # ---- recording loop --------------------------------------------------

    def tick(self):
        if self.stopped:
            return
        frame_bgra = np.array(self.sct.grab(self.monitor))
        frame_bgr = np.ascontiguousarray(frame_bgra[:, :, :3])  # drop alpha channel
        if not self.writer.write(frame_bgr) and not self.video_dead_warned:
            self.video_dead_warned = True
            print(f"WARNING: video encoder died at t={time.time() - self.t_start:.1f}s -- "
                  f"screen recording has stopped, but EEG/ratings/markers keep recording normally. "
                  f"See {self.writer.log_path} for why.")
            self.log_marker("video_writer_died")

        if self.rx is not None:
            _n, chunk, ts = self.rx.pull_new_ts(timeout=0.0)
            if chunk is not None:
                self.eeg_chunks.append(chunk)
                self.eeg_timestamps.append(ts)

        self.tick_after_id = self.root.after(self.tick_interval, self.tick)

    def _cancel_pending_after_calls(self):
        for attr in ("tick_after_id", "phase_after_id", "prompt_after_id"):
            after_id = getattr(self, attr, None)
            if after_id is not None:
                try:
                    self.root.after_cancel(after_id)
                except Exception:
                    pass
                setattr(self, attr, None)

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        self._cancel_pending_after_calls()
        if self.trial_name:
            self.log_marker(f"trial_end:{self.trial_name}")
        self.log_marker("recording_end")
        self.writer.release()
        self.sct.close()
        self.root.destroy()
        self._save()

    def _save(self):
        with open(self.ratings_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["elapsed_s", "wall_time", "lsl_time", "trial", "category", "confidence"])
            w.writeheader()
            w.writerows(self.rating_rows)

        with open(self.markers_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["elapsed_s", "wall_time", "lsl_time", "trial", "event"])
            w.writeheader()
            w.writerows(self.marker_rows)

        duration = self.marker_rows[-1]["elapsed_s"] if self.marker_rows else 0
        print(f"\nSaved recording ({duration:.1f}s) to {self.video_path}")
        print(f"Saved {len(self.rating_rows)} answered prompt(s) to {self.ratings_path}")
        print(f"Saved {len(self.marker_rows)} marker(s) to {self.markers_path}")

        if self.rx is not None:
            if self.eeg_chunks:
                data = np.concatenate(self.eeg_chunks, axis=0).T
                timestamps = np.concatenate(self.eeg_timestamps, axis=0)
                np.savez(self.eeg_path, data=data, timestamps=timestamps, srate=self.rx.srate,
                         labels=np.array(self.rx.labels))
                print(f"Saved EEG ({data.shape[1]} samples, {data.shape[0]} channels) to {self.eeg_path}")
            else:
                print("WARNING: no EEG samples were captured during the session.")

    @staticmethod
    def _describe_phases(phases):
        return " -> ".join(f"{label} {duration}s" for _key, label, duration in phases)

    def run(self):
        eeg_note = "+ EEG" if self.rx is not None else "(NO EEG)"
        print(f"Recording screen {eeg_note}.")
        print(f"Pre-rest runs automatically: {self._describe_phases(PRE_PHASES)}.")
        print(f"Then the video task starts: a prompt pops up every {self.args.prompt_interval}s.")
        print("When you're done watching, click 'Finish video -> Start Post-rest' to run "
              f"Post-rest ({self._describe_phases(POST_PHASES)}), which auto-saves when it completes.")
        print("Set a trial name (optional) via the box + button, or press Enter after typing.")
        print("Click 'Stop & Save' any time to end early.\n")
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="session")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--prompt-interval", type=int, default=PROMPT_INTERVAL_S,
                         help="seconds between popups during the video task (default 20)")
    # EEG is on by default -- pass --no-eeg to skip connecting (e.g. for a
    # UI-only dry run without the CGX headset). An entire batch of sessions
    # was silently recorded without EEG because this used to default the
    # other way and the earlier "no EEG" console message went unnoticed.
    parser.add_argument("--no-eeg", dest="no_eeg", action="store_true", default=False,
                         help="skip connecting to the CGX EEG LSL stream (screen+ratings only)")
    args = parser.parse_args()

    app = AnnotatorApp(args)
    app.run()


if __name__ == "__main__":
    main()
