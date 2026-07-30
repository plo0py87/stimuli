"""Interactive EEG paged viewer.

Usage example:
  python eeg_paged_viewer.py --session-dir sessions/Shine_E_SchindlersList_clip_2_20260709_152657

For a side-by-side comparison of two recordings, use eeg_dual_viewer.py instead
(file selection happens inside the window via the "Open NPZ..." button).

Controls:
  Right / Left      next / previous page
  Shift+Right/Left  jump forward / backward 10 pages
  + / -             increase / decrease display gain
  [ / ]             decrease / increase page seconds
  M                 toggle marker lines
  C                 toggle per-channel centering
  B / Shift+B       next / previous frequency band
  P                 toggle waveform / PSD view
  D                 toggle waveform / per-second Differential Entropy (DE) view
  Home / End        first / last page
  Click channel label   mute / unmute that channel
  Q                 quit
"""

import argparse
import csv
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from scipy.signal import butter, filtfilt, welch

NON_EEG_LABELS = {"ACC_X", "ACC_Y", "ACC_Z", "PacketCounter", "TRIGGER"}

BANDS = [
    ("Raw (unfiltered)", None),
    ("Broadband 0.5-45Hz", (0.5, 45.0)),
    ("Delta 0.5-4Hz", (0.5, 4.0)),
    ("Theta 4-8Hz", (4.0, 8.0)),
    ("Alpha 8-13Hz", (8.0, 13.0)),
    ("Beta 13-30Hz", (13.0, 30.0)),
    ("Gamma 30-45Hz", (30.0, 45.0)),
]


def bandpass_filter(data, srate, low, high, order=4):
    nyq = 0.5 * srate
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, data, axis=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive paged viewer for EEG NPZ sessions.")
    parser.add_argument("--session-dir", default=None, help="Session folder containing eeg_raw.npz")
    parser.add_argument("--npz-name", default="eeg_raw.npz", help="NPZ filename in the session dir")
    parser.add_argument("--markers-name", default="markers.csv", help="Markers CSV filename in the session dir")
    parser.add_argument("--channels", type=int, default=20, help="Number of channels to display")
    parser.add_argument("--page-seconds", type=float, default=2.0, help="Seconds per page")
    parser.add_argument("--gain", type=float, default=3.0, help="Display gain multiplier")
    parser.add_argument("--spacing", type=float, default=360.0, help="Vertical spacing between channels")
    parser.add_argument("--line-width", type=float, default=0.9, help="Waveform line width")
    parser.add_argument("--filter-low", type=float, default=None, help="Bandpass low cutoff Hz (e.g. 0.5)")
    parser.add_argument("--filter-high", type=float, default=None, help="Bandpass high cutoff Hz (e.g. 45)")
    return parser.parse_args()


def load_markers(markers_path):
    markers = []
    if markers_path is None or not markers_path.exists():
        return markers

    with open(markers_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row.get("elapsed_s", "nan"))
            except Exception:
                continue
            if np.isfinite(t):
                markers.append((t, row.get("event", "")))
    return markers


def shorten_event(event_name):
    if event_name.startswith("emotion_mark"):
        return "mark"
    if event_name.startswith("trial_start"):
        return "start"
    if event_name.startswith("trial_end"):
        return "end"
    if event_name in ("recording_start", "recording_end"):
        return event_name.replace("recording_", "")
    return event_name[:12]


def read_npz(npz_path, channels=20):
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        data = z["data"]
        labels = [str(x) for x in z["labels"]] if "labels" in z.files else [f"ch{i}" for i in range(data.shape[0])]
        srate = float(z["srate"]) if "srate" in z.files else 500.0

    keep_idx = [i for i, lb in enumerate(labels) if lb not in NON_EEG_LABELS]
    if not keep_idx:
        keep_idx = list(range(data.shape[0]))
    keep_idx = keep_idx[: max(1, channels)]

    display_data = data[keep_idx].astype(np.float64)
    display_labels = [labels[i] for i in keep_idx]
    return display_data, display_labels, srate


def pick_npz_file(title="Open NPZ file", initialdir=None):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        initialdir=initialdir or str(Path(__file__).parent),
        filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")],
    )
    root.destroy()
    return path or None


class EegPanel:
    """One channel-stack view (waveform or PSD) living inside a matplotlib Axes.

    Can be used standalone (its own figure, Q quits) or embedded alongside
    other panels in a shared figure (see eeg_dual_viewer.py), in which case
    the host is responsible for key/pick event routing and for closing the
    figure.
    """

    def __init__(
        self,
        ax,
        title_prefix="EEG",
        page_seconds=2.0,
        gain=3.0,
        spacing=360.0,
        line_width=0.9,
        channels=20,
        standalone=False,
        open_button_ax=None,
    ):
        self.ax = ax
        self.fig = ax.figure
        self.title_prefix = title_prefix
        self.channels = channels
        self.standalone = standalone

        self.raw_data = None
        self.labels = []
        self.srate = 500.0
        self.markers = []
        self.loaded_name = None

        self.band_idx = 0
        self._band_cache = {}

        self.page_seconds = max(0.2, float(page_seconds))
        self.gain = max(0.1, float(gain))
        self.spacing = float(spacing)
        self.line_width = float(line_width)

        self.page_idx = 0
        self.show_markers = True
        self.center_channels = True
        self.muted = set()
        self.view_mode = "wave"  # or "psd"

        self.open_button = None
        if open_button_ax is not None:
            self.open_button = Button(open_button_ax, f"Open NPZ... ({title_prefix})")
            self.open_button.on_clicked(self.on_open_clicked)

        if standalone:
            self.fig.canvas.mpl_connect("key_press_event", self.on_key)
            self.fig.canvas.mpl_connect("pick_event", self.on_pick)

    # -- data loading -----------------------------------------------------

    def load(self, npz_path, markers_path=None):
        data, labels, srate = read_npz(npz_path, channels=self.channels)
        self.raw_data = data
        self.labels = labels
        self.srate = srate
        self.markers = load_markers(Path(markers_path)) if markers_path else []
        self.loaded_name = Path(npz_path).name
        self.band_idx = 0
        self._band_cache = {}
        self.page_idx = 0
        self.muted = set()
        self.redraw()

    def on_open_clicked(self, event):
        path = pick_npz_file(title=f"Open NPZ for {self.title_prefix}")
        if not path:
            return
        markers_guess = Path(path).parent / "markers.csv"
        self.load(path, markers_path=markers_guess if markers_guess.exists() else None)

    # -- derived data -------------------------------------------------------

    @property
    def data(self):
        if self.band_idx not in self._band_cache:
            band = BANDS[self.band_idx][1]
            if band is None:
                self._band_cache[self.band_idx] = self.raw_data
            else:
                low, high = band
                self._band_cache[self.band_idx] = bandpass_filter(self.raw_data, self.srate, low, high)
        return self._band_cache[self.band_idx]

    @property
    def band_label(self):
        return BANDS[self.band_idx][0]

    @property
    def samples_per_page(self):
        return max(1, int(self.page_seconds * self.srate))

    @property
    def page_count(self):
        return max(1, int(np.ceil(self.data.shape[1] / self.samples_per_page)))

    def clamp_page_idx(self):
        self.page_idx = min(max(0, self.page_idx), self.page_count - 1)

    # -- drawing -----------------------------------------------------------

    def redraw(self):
        if self.raw_data is None:
            self.ax.clear()
            self.ax.text(
                0.5,
                0.5,
                f"{self.title_prefix}\n\nClick 'Open NPZ...' to load a file",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                fontsize=11,
                color="gray",
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.fig.canvas.draw_idle()
            return

        if self.view_mode == "psd":
            self._redraw_psd()
        elif self.view_mode == "de":
            self._redraw_de()
        else:
            self._redraw_wave()

    def _redraw_wave(self):
        self.clamp_page_idx()
        i0 = self.page_idx * self.samples_per_page
        i1 = min((self.page_idx + 1) * self.samples_per_page, self.data.shape[1])
        t = np.arange(i0, i1) / self.srate
        seg = self.data[:, i0:i1]

        if self.center_channels:
            seg = seg - np.median(seg, axis=1, keepdims=True)
        seg = seg * self.gain

        offs = np.arange(len(self.labels))[::-1] * self.spacing
        rail = min(0.45 * self.spacing, 300.0)

        self.ax.clear()
        for i in range(len(self.labels)):
            if i in self.muted:
                self.ax.hlines(offs[i], t[0], t[-1], colors="lightgray", linestyles="--", linewidth=0.7)
            else:
                self.ax.plot(t, seg[i] + offs[i], lw=self.line_width)
            self.ax.hlines(
                [offs[i] - rail, offs[i] + rail],
                t[0],
                t[-1],
                colors="k",
                linestyles=":",
                alpha=0.10,
                linewidth=0.6,
            )

        if self.show_markers and self.markers:
            in_win = [(mt, ev) for mt, ev in self.markers if t[0] <= mt <= t[-1]]
            top_y = offs[0] + rail + 70
            for mt, ev in in_win:
                self.ax.axvline(mt, color="crimson", alpha=0.35, linewidth=1.0)
                self.ax.text(
                    mt,
                    top_y,
                    shorten_event(ev),
                    rotation=90,
                    fontsize=8,
                    color="crimson",
                    va="bottom",
                    ha="center",
                )

        self.ax.set_yticks(offs)
        self.ax.set_yticklabels([])
        self.ax.tick_params(axis="y", length=0)
        self._draw_channel_labels(offs)
        self.ax.set_ylim(offs[-1] - rail - 120, offs[0] + rail + 140)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Channel")
        self.ax.grid(axis="x", alpha=0.25)
        self.ax.set_title(
            f"{self.title_prefix} [{self.loaded_name}] | "
            f"page {self.page_idx + 1}/{self.page_count} | "
            f"{t[0]:.1f}-{t[-1]:.1f}s | "
            f"gain x{self.gain:.2f} | page_sec {self.page_seconds:.2f} | "
            f"markers {'on' if self.show_markers else 'off'} | "
            f"center {'on' if self.center_channels else 'off'} | "
            f"band [{self.band_label}] (B/Shift+B) | P=PSD | click label to mute",
            fontsize=9,
        )

        self.fig.canvas.draw_idle()

    def _redraw_psd(self):
        self.clamp_page_idx()
        i0 = self.page_idx * self.samples_per_page
        i1 = min((self.page_idx + 1) * self.samples_per_page, self.data.shape[1])
        seg = self.data[:, i0:i1]

        # 1-second Welch segments (1 Hz frequency resolution), averaged
        # across however many 1s windows fit in the current page.
        nperseg = int(min(seg.shape[1], max(16, self.srate)))
        noverlap = nperseg // 2
        row_step = 6.0

        self.ax.clear()
        offs = np.arange(len(self.labels))[::-1] * row_step
        fmax = min(45.0, self.srate / 2.0)
        for i in range(len(self.labels)):
            if i in self.muted:
                continue
            freqs, pxx = welch(seg[i], fs=self.srate, nperseg=nperseg, noverlap=noverlap)
            mask = freqs <= fmax
            log_p = np.log10(pxx[mask] + 1e-12)
            self.ax.plot(freqs[mask], log_p + offs[i], lw=1.0)

        self.ax.set_yticks(offs)
        self.ax.set_yticklabels([])
        self.ax.tick_params(axis="y", length=0)
        self._draw_channel_labels(offs)
        self.ax.set_xlim(0, fmax)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Channel (log power, stacked)")
        self.ax.grid(alpha=0.25)
        self.ax.set_title(
            f"{self.title_prefix} [{self.loaded_name}] | PSD (Welch, 1s window) | "
            f"page {self.page_idx + 1}/{self.page_count} | {i0 / self.srate:.1f}-{i1 / self.srate:.1f}s | "
            "P=waveform | click label to mute",
            fontsize=9,
        )

        self.fig.canvas.draw_idle()

    def _redraw_de(self):
        """Differential Entropy per channel, computed in 1-second bins.

        DE of a Gaussian-distributed signal is 0.5*log(2*pi*e*variance), the
        standard per-band feature used by SEED-style emotion-classification
        pipelines. This is computed on whatever band is currently selected
        (B / Shift+B) so it always matches what the waveform view shows.
        """
        self.clamp_page_idx()
        i0 = self.page_idx * self.samples_per_page
        i1 = min((self.page_idx + 1) * self.samples_per_page, self.data.shape[1])
        seg = self.data[:, i0:i1]

        sec_len = max(1, int(round(self.srate)))
        n_secs = max(1, int(np.ceil(seg.shape[1] / sec_len)))
        de = np.full((len(self.labels), n_secs), np.nan)
        for s in range(n_secs):
            a = s * sec_len
            b = min(a + sec_len, seg.shape[1])
            if b - a < 2:
                continue
            var = seg[:, a:b].var(axis=1) + 1e-12
            de[:, s] = 0.5 * np.log(2 * np.pi * np.e * var)
        de[list(self.muted), :] = np.nan

        self.ax.clear()
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("white")
        im = self.ax.imshow(
            de,
            aspect="auto",
            origin="upper",
            cmap=cmap,
            extent=[0, n_secs, len(self.labels), 0],
            interpolation="nearest",
        )
        self.fig.colorbar(im, ax=self.ax, label="DE (nats)", fraction=0.03, pad=0.02)

        offs = np.arange(len(self.labels)) + 0.5
        self.ax.set_yticks(offs)
        self.ax.set_yticklabels([])
        self.ax.tick_params(axis="y", length=0)
        self._draw_channel_labels(offs)
        self.ax.set_xticks(np.arange(n_secs) + 0.5)
        self.ax.set_xticklabels([f"{int(i0 / self.srate) + s}" for s in range(n_secs)])
        self.ax.set_xlabel("Time (s, absolute)")
        self.ax.set_ylabel("Channel")
        self.ax.set_title(
            f"{self.title_prefix} [{self.loaded_name}] | Differential Entropy, 1s bins | "
            f"band [{self.band_label}] (B/Shift+B) | "
            f"page {self.page_idx + 1}/{self.page_count} | {i0 / self.srate:.1f}-{i1 / self.srate:.1f}s | "
            "D=waveform | click label to mute",
            fontsize=9,
        )

        self.fig.canvas.draw_idle()

    def _draw_channel_labels(self, offs):
        """Draw clickable channel-name labels just left of the axes.

        Native tick-label Text objects are unreliable pick targets across
        matplotlib versions/backends, so channel labels are drawn as our own
        Text artists with an explicit boolean picker instead.
        """
        trans = self.ax.get_yaxis_transform()  # x: axes fraction, y: data coords
        self._label_artists = {}
        for i, lb in enumerate(self.labels):
            muted = i in self.muted
            txt = self.ax.text(
                -0.01,
                offs[i],
                lb,
                transform=trans,
                ha="right",
                va="center",
                fontsize=10,
                color="lightgray" if muted else "black",
                fontstyle="italic" if muted else "normal",
                picker=True,
                clip_on=False,
            )
            self._label_artists[txt] = i

    # -- events --------------------------------------------------------------

    def on_key(self, event):
        if self.raw_data is None:
            return
        key = event.key or ""

        if key == "right":
            self.page_idx += 1
        elif key == "left":
            self.page_idx -= 1
        elif key == "shift+right":
            self.page_idx += 10
        elif key == "shift+left":
            self.page_idx -= 10
        elif key in ("+", "="):
            self.gain *= 1.25
        elif key in ("-", "_"):
            self.gain /= 1.25
        elif key == "]":
            self.page_seconds *= 1.25
        elif key == "[":
            self.page_seconds /= 1.25
        elif key in ("m", "M"):
            self.show_markers = not self.show_markers
        elif key in ("c", "C"):
            self.center_channels = not self.center_channels
        elif key == "b":
            self.band_idx = (self.band_idx + 1) % len(BANDS)
        elif key in ("B", "shift+b"):
            self.band_idx = (self.band_idx - 1) % len(BANDS)
        elif key in ("p", "P"):
            self.view_mode = "wave" if self.view_mode == "psd" else "psd"
        elif key in ("d", "D"):
            self.view_mode = "wave" if self.view_mode == "de" else "de"
        elif key == "home":
            self.page_idx = 0
        elif key == "end":
            self.page_idx = self.page_count - 1
        elif key in ("q", "Q") and self.standalone:
            plt.close(self.fig)
            return
        else:
            return

        self.redraw()

    def on_pick(self, event):
        if self.raw_data is None:
            return
        artist = event.artist
        text = artist.get_text() if hasattr(artist, "get_text") else None
        if text not in self.labels:
            return
        i = self.labels.index(text)
        if i in self.muted:
            self.muted.discard(i)
        else:
            self.muted.add(i)
        self.redraw()

    def show(self):
        self.redraw()
        plt.show()


def main():
    args = parse_args()

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_axes([0.06, 0.08, 0.90, 0.82])
    open_ax = fig.add_axes([0.80, 0.94, 0.16, 0.04])

    panel = EegPanel(
        ax,
        title_prefix="EEG",
        page_seconds=args.page_seconds,
        gain=args.gain,
        spacing=args.spacing,
        line_width=args.line_width,
        channels=args.channels,
        standalone=True,
        open_button_ax=open_ax,
    )

    if args.session_dir is not None:
        session_dir = Path(args.session_dir)
        npz_path = session_dir / args.npz_name
        markers_path = session_dir / args.markers_name
        if not npz_path.exists():
            raise FileNotFoundError(f"EEG file not found: {npz_path}")
        panel.load(npz_path, markers_path=markers_path)

        if args.filter_low is not None and args.filter_high is not None:
            for i, (_, band) in enumerate(BANDS):
                if band is not None and abs(band[0] - args.filter_low) < 1e-6 and abs(band[1] - args.filter_high) < 1e-6:
                    panel.band_idx = i
                    break
            else:
                BANDS.append((f"{args.filter_low:g}-{args.filter_high:g}Hz", (args.filter_low, args.filter_high)))
                panel.band_idx = len(BANDS) - 1

        print("Loaded:", npz_path)
        print("Displayed channels:", ", ".join(panel.labels))
        print("Sampling rate:", panel.srate)
        print("Markers loaded:", len(panel.markers))
        print("Starting band:", BANDS[panel.band_idx][0])
    else:
        print("No --session-dir given. Click 'Open NPZ...' in the window to pick a file.")

    print("Controls: Right/Left, Shift+Right/Left, +/- , [/] , M, C, B/Shift+B, P, Home/End, Q")

    panel.show()


if __name__ == "__main__":
    main()
