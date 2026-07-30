"""Side-by-side EEG viewer: compare two recordings in one window.

Usage:
  python eeg_dual_viewer.py

No file paths are required on the command line — click "Open NPZ..." above
each panel to pick a file. The clicked panel becomes "active" (highlighted
red border) and receives keyboard shortcuts.

Controls (apply to the active/highlighted panel — click inside a panel to
make it active):
  Right / Left      next / previous page
  Shift+Right/Left  jump forward / backward 10 pages
  + / -             increase / decrease display gain
  [ / ]             decrease / increase page seconds
  M                 toggle marker lines
  C                 toggle per-channel centering
  B / Shift+B       next / previous frequency band
  P                 toggle waveform / PSD view
  Home / End        first / last page
  Click channel label   mute / unmute that channel
  Q                 quit (closes the whole window)

Optional CLI flags let you preload both sides instead of clicking Open:
  python eeg_dual_viewer.py --left SEED/1_1_quick20.npz --right SEED/3_1_quick20.npz
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from eeg_paged_viewer import EegPanel


def parse_args():
    parser = argparse.ArgumentParser(description="Side-by-side EEG comparison viewer.")
    parser.add_argument("--left", default=None, help="Optional NPZ path to preload in the left panel")
    parser.add_argument("--right", default=None, help="Optional NPZ path to preload in the right panel")
    parser.add_argument("--channels", type=int, default=20, help="Number of channels to display")
    parser.add_argument("--page-seconds", type=float, default=2.0, help="Seconds per page")
    parser.add_argument("--gain", type=float, default=3.0, help="Display gain multiplier")
    parser.add_argument("--spacing", type=float, default=360.0, help="Vertical spacing between channels")
    return parser.parse_args()


class DualApp:
    def __init__(self, channels, page_seconds, gain, spacing):
        self.fig = plt.figure(figsize=(20, 9))

        open_ax_l = self.fig.add_axes([0.06, 0.94, 0.18, 0.04])
        open_ax_r = self.fig.add_axes([0.56, 0.94, 0.18, 0.04])
        ax_l = self.fig.add_axes([0.04, 0.06, 0.45, 0.84])
        ax_r = self.fig.add_axes([0.53, 0.06, 0.45, 0.84])

        common = dict(
            page_seconds=page_seconds,
            gain=gain,
            spacing=spacing,
            channels=channels,
            standalone=False,
        )
        self.panels = [
            EegPanel(ax_l, title_prefix="LEFT", open_button_ax=open_ax_l, **common),
            EegPanel(ax_r, title_prefix="RIGHT", open_button_ax=open_ax_r, **common),
        ]
        self.active = 0

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)

    def on_click(self, event):
        for i, panel in enumerate(self.panels):
            if event.inaxes == panel.ax:
                self.active = i
                self.highlight_active()
                break

    def on_key(self, event):
        if event.key in ("q", "Q"):
            plt.close(self.fig)
            return
        self.panels[self.active].on_key(event)

    def on_pick(self, event):
        artist_ax = getattr(event.artist, "axes", None)
        for panel in self.panels:
            if artist_ax == panel.ax:
                panel.on_pick(event)
                break

    def highlight_active(self):
        for i, panel in enumerate(self.panels):
            is_active = i == self.active
            for spine in panel.ax.spines.values():
                spine.set_edgecolor("crimson" if is_active else "black")
                spine.set_linewidth(2.2 if is_active else 0.8)
        self.fig.canvas.draw_idle()

    def show(self):
        for panel in self.panels:
            panel.redraw()
        self.highlight_active()
        plt.show()


def main():
    args = parse_args()

    app = DualApp(
        channels=args.channels,
        page_seconds=args.page_seconds,
        gain=args.gain,
        spacing=args.spacing,
    )

    if args.left:
        left_path = Path(args.left)
        markers = left_path.parent / "markers.csv"
        app.panels[0].load(left_path, markers_path=markers if markers.exists() else None)
    if args.right:
        right_path = Path(args.right)
        markers = right_path.parent / "markers.csv"
        app.panels[1].load(right_path, markers_path=markers if markers.exists() else None)

    print("Click inside a panel to make it active (red border), then use keyboard shortcuts.")
    print("Click 'Open NPZ...' above a panel to load/replace its file.")
    print("Controls: Right/Left, Shift+Right/Left, +/- , [/] , M, C, B/Shift+B, P, Home/End, Q(quits window)")

    app.show()


if __name__ == "__main__":
    main()
