"""Browser-compatible (H.264) screen-recording writer, replacing
cv2.VideoWriter's default 'mp4v' codec — Chrome/Edge don't support mp4v at
all in <video>, so recordings made with it can only be opened in desktop
players like VLC, not viewed in the annotated-timeline HTML report.

Pipes raw BGR frames into a bundled ffmpeg binary (via imageio-ffmpeg, no
system install required) encoding libx264 with +faststart (moov atom at
the front, so the browser can start playback before the full file is
downloaded/read).

IMPORTANT (fixed after a real incident): the previous version told ffmpeg
a fixed `-r fps` input rate and assumed every write() call represented
exactly 1/fps seconds of real time. Over long sessions, whenever actual
frame delivery fell behind that nominal rate (encoder backpressure, GUI
tick lag, etc.), the encoded video's internal clock kept advancing at the
nominal rate anyway -- so the video's timeline ran faster than real time.
The gap compounded over the session until the video simply ran out of
frames while EEG (recorded on real timestamps) kept going, silently
truncating recordings to as little as ~45% of their real duration with no
error anywhere (stderr was also discarded to DEVNULL, hiding it further).

Fix: each frame is timestamped with the real wall-clock time it was
captured (`-use_wallclock_as_timestamps 1` on the rawvideo input, `-fps_mode
vfr` on output) instead of assuming a fixed nominal rate, so the video's
duration always matches real elapsed time even if the effective capture
rate drops below `fps`. ffmpeg's stderr is also captured to a log file
next to the output (`<output>.ffmpeg.log`) instead of being discarded, so
encoder failures are visible instead of silent.
"""

import subprocess
from pathlib import Path

import imageio_ffmpeg


class FFmpegVideoWriter:
    def __init__(self, output_path, width, height, fps, crf=23, preset="veryfast"):
        output_path = Path(output_path)
        self.log_path = output_path.with_suffix(output_path.suffix + ".ffmpeg.log")
        self._log_file = open(self.log_path, "wb")

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-use_wallclock_as_timestamps", "1",
            "-i", "-",
            "-fps_mode", "vfr",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output_path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL, stderr=self._log_file)
        self.frames_written = 0
        self.dead = False

    def isOpened(self):
        return self.proc.poll() is None

    def write(self, frame_bgr):
        """Returns True if the frame was written, False if the encoder
        process has died (caller should stop calling write() and surface a
        loud warning -- check self.log_path for why)."""
        if self.dead:
            return False
        try:
            self.proc.stdin.write(frame_bgr.tobytes())
        except (BrokenPipeError, OSError):
            self.dead = True
            return False
        self.frames_written += 1
        return True

    def release(self):
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        self.proc.wait()
        self._log_file.close()
        if self.proc.returncode != 0:
            print(f"WARNING: ffmpeg exited with code {self.proc.returncode}; "
                  f"see {self.log_path} for details.")
