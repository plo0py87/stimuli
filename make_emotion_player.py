"""Generate a self-contained HTML video player that shows, second-by-second,
the DGCNN emotion model's prediction (label + neg/neutral/positive scores)
in sync with the session's screen_recording.mp4.

Usage:
  python make_emotion_player.py --session-dir sessions/Shine_E_LoveActually_clip_1_20260709_145626

Requires emotion_predictions.csv to already exist in the session dir (run
run_emotion_inference.py first).
"""

import argparse
import base64
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

NON_EEG_LABELS = {"ACC_X", "ACC_Y", "ACC_Z", "PacketCounter", "TRIGGER"}
WAVE_DISPLAY_HZ = 100.0  # decimated rate for the in-browser scrolling waveform
WAVE_WINDOW_S = 4.0  # total width of the scrolling window (seconds)
WAVE_FILTER_LOW = 0.5
WAVE_FILTER_HIGH = 45.0

# 5-class self-report -> 3-class collapse, same as train_dagcn_own_data.py
LABEL_COLLAPSE = {
    "負向": "negative", "偏負": "negative",
    "中性": "neutral",
    "偏正": "positive", "正向": "positive",
}


def bandpass_filter(data, srate, low, high, order=4):
    nyq = 0.5 * srate
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, data, axis=1)

TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Emotion playback - {session_name}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }}
  h2 {{ font-weight: 500; font-size: 15px; color: #aaa; margin: 0 0 12px; }}
  video {{ width: 100%; max-height: 60vh; background: #000; display: block; border-radius: 6px; }}
  .panel {{ max-width: 960px; margin: 0 auto; }}
  .scores {{ display: flex; gap: 12px; margin: 14px 0; }}
  .score {{ flex: 1; background: #1c1c1c; border-radius: 6px; padding: 10px 12px; }}
  .score .name {{ font-size: 12px; color: #999; text-transform: uppercase; letter-spacing: .04em; }}
  .score .pct {{ font-size: 26px; font-weight: 600; margin-top: 2px; }}
  .bar-track {{ height: 6px; background: #333; border-radius: 3px; margin-top: 8px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 3px; transition: width 80ms linear; }}
  .neg .pct, .neg .bar-fill {{ color: #ff6b6b; background: #ff6b6b; }}
  .neu .pct, .neu .bar-fill {{ color: #ffd93d; background: #ffd93d; }}
  .pos .pct, .pos .bar-fill {{ color: #6bcB77; background: #6bcB77; }}
  .label-line {{ font-size: 14px; color: #ccc; margin: 4px 0 0; }}
  .label-line b {{ color: #fff; text-transform: uppercase; }}
  canvas {{ width: 100%; height: 140px; display: block; background: #1c1c1c; border-radius: 6px; }}
  #gtstrip {{ height: 28px; margin-top: 6px; }}
  #rawwave {{ height: 420px; margin-top: 16px; }}
  .legend {{ font-size: 11px; color: #999; margin-top: 4px; }}
  .legend span {{ margin-right: 14px; }}
  .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }}
  .section-title {{ font-size: 12px; color: #999; text-transform: uppercase; letter-spacing: .04em; margin: 18px 0 4px; }}
  .scrubber {{ background: #1c1c1c; border-radius: 6px; padding: 14px; }}
  .scrubber button {{ background: #333; color: #eee; border: none; border-radius: 4px; padding: 8px 16px; font-size: 14px; cursor: pointer; margin-right: 10px; }}
  .scrubber button:hover {{ background: #444; }}
  .scrubber input[type=range] {{ width: 100%; margin-top: 10px; }}
</style>
</head>
<body>
<div class="panel">
  <h2>{session_name}</h2>
  {video_html}

  <p class="label-line">t=<span id="tsec">0</span>s &nbsp; prediction: <b id="predlabel">-</b> &nbsp; your label: <b id="truelabel">-</b></p>

  <div class="scores">
    <div class="score neg">
      <div class="name">Negative</div>
      <div class="pct" id="pct-neg">0%</div>
      <div class="bar-track"><div class="bar-fill" id="bar-neg" style="width:0%"></div></div>
    </div>
    <div class="score neu">
      <div class="name">Neutral</div>
      <div class="pct" id="pct-neu">0%</div>
      <div class="bar-track"><div class="bar-fill" id="bar-neu" style="width:0%"></div></div>
    </div>
    <div class="score pos">
      <div class="name">Positive</div>
      <div class="pct" id="pct-pos">0%</div>
      <div class="bar-track"><div class="bar-fill" id="bar-pos" style="width:0%"></div></div>
    </div>
  </div>

  <canvas id="timeline"></canvas>
  <div class="legend">
    <span><span class="dot" style="background:#ff6b6b"></span>negative</span>
    <span><span class="dot" style="background:#ffd93d"></span>neutral</span>
    <span><span class="dot" style="background:#6bcB77"></span>positive</span>
    <span>(vertical line = current playback position)</span>
    <span id="scale-note"></span>
  </div>

  <div class="section-title">Your label (self-report, collapsed to 3-class)</div>
  <canvas id="gtstrip"></canvas>
  <div class="legend">
    <span><span class="dot" style="background:#ff6b6b"></span>negative</span>
    <span><span class="dot" style="background:#ffd93d"></span>neutral</span>
    <span><span class="dot" style="background:#6bcB77"></span>positive</span>
    <span><span class="dot" style="background:#444"></span>no label (outside rated window)</span>
  </div>

  <div class="section-title">Raw EEG ({wave_filter_low:g}-{wave_filter_high:g}Hz, scrolling, centered white line = now) &nbsp;
    <span id="gain-note" style="text-transform:none; letter-spacing:0;">gain x4.0 (press +/- to adjust)</span>
  </div>
  <canvas id="rawwave" tabindex="0"></canvas>
</div>

<script>
const DATA = {data_json};  // [[neg, neu, pos], ...] one row per second
const LABELS = {labels_json};
const TRUE_LABELS = {true_labels_json};  // your self-reported label per second, or null if unrated
// True wall-clock recording duration (from markers.csv), vs the video
// file's own duration (frames/fps). The screen-capture loop can't always
// sustain its target fps, so the encoded video can be noticeably shorter
// than the real recording -- video.currentTime then needs rescaling to
// land on the right second of EEG-derived predictions.
const TRUE_DURATION_S = {true_duration_json};
let timeScale = 1.0;

// Raw EEG waveform, decimated to {wave_display_hz}Hz for browser display.
// WAVE_B64 decodes to a flat Float32Array laid out as (channel, sample),
// i.e. all of channel 0's samples, then all of channel 1's, etc.
const WAVE_LABELS = {wave_labels_json};
const WAVE_SRATE = {wave_srate_json};
const WAVE_SCALES = {wave_scales_json};  // per-channel robust scale, for display gain
const WAVE_N_SAMPLES = {wave_n_samples_json};
const WAVE_WINDOW_S = {wave_window_s_json};
let waveData = null;  // Float32Array, filled in from base64 below
{{
  const b64 = "{wave_b64}";
  if (b64) {{
    const bin = atob(b64);
    const buf = new ArrayBuffer(bin.length);
    const view = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
    waveData = new Float32Array(buf);
  }}
}}

{video_js_init}
const canvas = document.getElementById('timeline');
const ctx = canvas.getContext('2d');
const gtCanvas = document.getElementById('gtstrip');
const gtCtx = gtCanvas.getContext('2d');
const waveCanvas = document.getElementById('rawwave');
const waveCtx = waveCanvas.getContext('2d');
let waveGain = 4.0;  // default higher than 1x since the 95th-pct auto-scale alone reads flat

function resizeCanvas() {{
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * devicePixelRatio;
  canvas.height = rect.height * devicePixelRatio;
  const gtRect = gtCanvas.getBoundingClientRect();
  gtCanvas.width = gtRect.width * devicePixelRatio;
  gtCanvas.height = gtRect.height * devicePixelRatio;
  const waveRect = waveCanvas.getBoundingClientRect();
  waveCanvas.width = waveRect.width * devicePixelRatio;
  waveCanvas.height = waveRect.height * devicePixelRatio;
}}

const LABEL_COLORS = {{negative: '#ff6b6b', neutral: '#ffd93d', positive: '#6bcB77'}};

function drawGroundTruth(currentSec) {{
  const w = gtCanvas.width, h = gtCanvas.height;
  gtCtx.clearRect(0, 0, w, h);
  const n = TRUE_LABELS.length;
  if (n < 2) return;

  for (let i = 0; i < n; i++) {{
    const label = TRUE_LABELS[i];
    const x0 = (i / n) * w;
    const x1 = ((i + 1) / n) * w;
    gtCtx.fillStyle = LABEL_COLORS[label] || '#444';
    gtCtx.fillRect(x0, 0, x1 - x0 + 1, h);
  }}

  const px = (currentSec / (n - 1)) * w;
  gtCtx.beginPath();
  gtCtx.strokeStyle = '#fff';
  gtCtx.lineWidth = 2 * devicePixelRatio;
  gtCtx.moveTo(px, 0);
  gtCtx.lineTo(px, h);
  gtCtx.stroke();
}}

function drawWave(realElapsed) {{
  const w = waveCanvas.width, h = waveCanvas.height;
  waveCtx.clearRect(0, 0, w, h);
  if (!waveData || !WAVE_LABELS.length) return;

  const nCh = WAVE_LABELS.length;
  const rowH = h / nCh;
  const halfWin = WAVE_WINDOW_S / 2;
  const tStart = realElapsed - halfWin;
  const tEnd = realElapsed + halfWin;
  const sStart = Math.floor(tStart * WAVE_SRATE);
  const sEnd = Math.ceil(tEnd * WAVE_SRATE);

  const winLo = Math.max(0, sStart);
  const winHi = Math.min(WAVE_N_SAMPLES - 1, sEnd);

  for (let ch = 0; ch < nCh; ch++) {{
    const rowY = (ch + 0.5) * rowH;
    const scale = rowH * 0.45 * waveGain / (WAVE_SCALES[ch] || 1);
    const base = ch * WAVE_N_SAMPLES;

    // Data is already {wave_filter_low}-{wave_filter_high}Hz bandpass filtered
    // (see load_wave_for_display in make_emotion_player.py), which removes
    // slow drift at the source -- no need for a per-frame re-centering hack.
    waveCtx.beginPath();
    waveCtx.strokeStyle = '#7fd0ff';
    waveCtx.lineWidth = 1 * devicePixelRatio;
    let started = false;
    for (let s = winLo; s <= winHi; s++) {{
      const t = s / WAVE_SRATE;
      const x = ((t - tStart) / WAVE_WINDOW_S) * w;
      const y = rowY - waveData[base + s] * scale;
      if (!started) {{ waveCtx.moveTo(x, y); started = true; }} else {{ waveCtx.lineTo(x, y); }}
    }}
    waveCtx.stroke();

    waveCtx.font = `${{11 * devicePixelRatio}}px system-ui, sans-serif`;
    waveCtx.fillStyle = '#999';
    waveCtx.textBaseline = 'middle';
    waveCtx.fillText(WAVE_LABELS[ch], 4 * devicePixelRatio, rowY);
  }}

  const nowX = (halfWin / WAVE_WINDOW_S) * w;
  waveCtx.beginPath();
  waveCtx.strokeStyle = '#fff';
  waveCtx.lineWidth = 2 * devicePixelRatio;
  waveCtx.moveTo(nowX, 0);
  waveCtx.lineTo(nowX, h);
  waveCtx.stroke();
}}

function drawTimeline(currentSec) {{
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const n = DATA.length;
  if (n < 2) return;

  const colors = ['#ff6b6b', '#ffd93d', '#6bcB77'];
  for (let series = 0; series < 3; series++) {{
    ctx.beginPath();
    ctx.strokeStyle = colors[series];
    ctx.lineWidth = 1.5 * devicePixelRatio;
    for (let i = 0; i < n; i++) {{
      const x = (i / (n - 1)) * w;
      const y = h - DATA[i][series] * h;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }}

  const px = (currentSec / (n - 1)) * w;
  ctx.beginPath();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2 * devicePixelRatio;
  ctx.moveTo(px, 0);
  ctx.lineTo(px, h);
  ctx.stroke();
}}

function update() {{
  const realElapsed = video.currentTime * timeScale;
  const idx = Math.max(0, Math.min(DATA.length - 1, Math.round(realElapsed)));
  const [neg, neu, pos] = DATA[idx];

  document.getElementById('tsec').textContent = idx;
  document.getElementById('predlabel').textContent = LABELS[idx];
  const trueLabel = TRUE_LABELS[idx];
  const trueLabelEl = document.getElementById('truelabel');
  trueLabelEl.textContent = trueLabel || '(unrated)';
  trueLabelEl.style.color = trueLabel ? LABEL_COLORS[trueLabel] : '#666';

  document.getElementById('pct-neg').textContent = (neg * 100).toFixed(1) + '%';
  document.getElementById('pct-neu').textContent = (neu * 100).toFixed(1) + '%';
  document.getElementById('pct-pos').textContent = (pos * 100).toFixed(1) + '%';
  document.getElementById('bar-neg').style.width = (neg * 100) + '%';
  document.getElementById('bar-neu').style.width = (neu * 100) + '%';
  document.getElementById('bar-pos').style.width = (pos * 100) + '%';

  drawTimeline(idx);
  drawGroundTruth(idx);
  drawWave(realElapsed);
}}

// `timeupdate` only fires a handful of times per second in most browsers,
// which makes the raw-wave "now" line visibly hitch/jump instead of sitting
// still in the center. Redraw every animation frame while playing instead,
// so it stays rock-steady in the middle.
let rafId = null;
function rafLoop() {{
  update();
  rafId = requestAnimationFrame(rafLoop);
}}
video.addEventListener('play', () => {{ if (rafId === null) rafId = requestAnimationFrame(rafLoop); }});
video.addEventListener('pause', () => {{ if (rafId !== null) {{ cancelAnimationFrame(rafId); rafId = null; }} update(); }});
video.addEventListener('ended', () => {{ if (rafId !== null) {{ cancelAnimationFrame(rafId); rafId = null; }} update(); }});

window.addEventListener('keydown', (e) => {{
  if (e.key === '+' || e.key === '=') {{ waveGain *= 1.3; }}
  else if (e.key === '-' || e.key === '_') {{ waveGain /= 1.3; }}
  else {{ return; }}
  document.getElementById('gain-note').textContent = `gain x${{waveGain.toFixed(1)}} (press +/- to adjust)`;
  update();
}});

video.addEventListener('timeupdate', update);
video.addEventListener('seeked', update);
window.addEventListener('resize', () => {{ resizeCanvas(); update(); }});
video.addEventListener('loadedmetadata', () => {{
  if (TRUE_DURATION_S && video.duration && isFinite(video.duration) && video.duration > 0) {{
    timeScale = TRUE_DURATION_S / video.duration;
    if (Math.abs(timeScale - 1) > 0.01) {{
      document.getElementById('scale-note').textContent =
        ` | video is ${{video.duration.toFixed(1)}}s but recording was ${{TRUE_DURATION_S.toFixed(1)}}s ` +
        `(capture dropped frames) -- timeline rescaled x${{timeScale.toFixed(3)}}`;
    }}
  }}
  resizeCanvas();
  update();
}});
resizeCanvas();
update();
</script>
</body>
</html>
"""


def load_wave_for_display(session_dir, npz_name, display_hz=WAVE_DISPLAY_HZ):
    """Load eeg_raw.npz, keep only EEG channels, decimate to display_hz for
    a lightweight in-browser scrolling waveform. Returns None if the npz is
    missing (player still works, just without the raw-wave panel)."""
    npz_path = Path(session_dir) / npz_name
    if not npz_path.exists():
        return None

    with np.load(npz_path, allow_pickle=False) as z:
        data = z["data"]
        labels = [str(x) for x in z["labels"]]
        srate = float(z["srate"])

    keep_idx = [i for i, lb in enumerate(labels) if lb not in NON_EEG_LABELS]
    if not keep_idx:
        keep_idx = list(range(data.shape[0]))
    eeg = data[keep_idx].astype(np.float64)
    eeg_labels = [labels[i] for i in keep_idx]

    # Bandpass at full resolution before decimating, so drift/DC offset and
    # high-frequency noise above Nyquist-after-decimation are both handled
    # properly (filtering after decimating would alias).
    eeg = bandpass_filter(eeg, srate, WAVE_FILTER_LOW, WAVE_FILTER_HIGH)

    factor = max(1, int(round(srate / display_hz)))
    eeg = eeg[:, ::factor]
    actual_display_hz = srate / factor

    eeg = eeg - np.median(eeg, axis=1, keepdims=True)
    # Robust per-channel scale (95th percentile of abs value) so occasional
    # huge motion/EMG artifacts don't visually flatten the rest of the trace.
    scales = np.percentile(np.abs(eeg), 95, axis=1)
    scales = np.where(scales < 1e-6, 1.0, scales)

    eeg = eeg.astype(np.float32)
    wave_b64 = base64.b64encode(eeg.tobytes(order="C")).decode("ascii")

    return {
        "labels": eeg_labels,
        "srate": actual_display_hz,
        "scales": scales.tolist(),
        "n_samples": eeg.shape[1],
        "b64": wave_b64,
    }


def load_true_labels(session_dir, ratings_name, n_seconds):
    """Returns a list of length n_seconds: your self-reported collapsed
    3-class label ('negative'/'neutral'/'positive') for every second, or
    None where no rating window covers that second. Each rating answered
    at time T labels [previous prompt's time (or 0), T) -- same convention
    as train_dagcn_own_data.py's load_session_windows. Uses the trial-local
    `t_s` column if present (from split_session_trials.py), else falls back
    to `elapsed_s`. Returns None (not a list) if ratings.csv doesn't exist."""
    ratings_path = Path(session_dir) / ratings_name
    if not ratings_path.exists():
        return None
    with open(ratings_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [None] * n_seconds

    time_key = "t_s" if "t_s" in rows[0] else "elapsed_s"
    true_labels = [None] * n_seconds
    prev_t = 0.0
    for r in rows:
        t = float(r[time_key])
        cat = r["category"].strip()
        if cat not in LABEL_COLLAPSE:
            prev_t = t
            continue
        label = LABEL_COLLAPSE[cat]
        s0, s1 = int(round(prev_t)), min(n_seconds, int(round(t)))
        for s in range(max(0, s0), s1):
            true_labels[s] = label
        prev_t = t
    return true_labels


def get_true_duration_s(session_dir, markers_name="markers.csv"):
    """Real wall-clock recording duration, from markers.csv's max elapsed_s
    (recording_end/trial_end). Falls back to None if markers are missing."""
    markers_path = Path(session_dir) / markers_name
    if not markers_path.exists():
        return None
    with open(markers_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    elapsed = [float(r["elapsed_s"]) for r in rows if r.get("elapsed_s")]
    return max(elapsed) if elapsed else None


VIDEO_JS = "const video = document.getElementById('vid');"

NOVIDEO_HTML = """<div class="scrubber">
    <button id="btn-play">Play</button>
    <button id="btn-pause">Pause</button>
    <input type="range" id="scrub-range" min="0" max="1" step="1" value="0">
    <div style="font-size:12px; color:#999; margin-top:6px;">No screen recording for this file (e.g. SEED source data) — scrub or press Play to step through the per-second predictions/waveform below.</div>
  </div>"""

NOVIDEO_JS = """// No <video> element for this session (e.g. a bare SEED .npz with no
// screen recording). Emulate the same interface (currentTime/duration/
// addEventListener/play/pause) with a small shim driven by a slider + timer,
// so the rest of the script (drawWave/drawTimeline/update) needs no changes.
class FakeVideo {
  constructor(duration) {
    this.currentTime = 0;
    this.duration = duration;
    this.paused = true;
    this._listeners = {};
    this._raf = null;
  }
  addEventListener(type, fn) {
    (this._listeners[type] = this._listeners[type] || []).push(fn);
  }
  _emit(type) {
    (this._listeners[type] || []).forEach((fn) => fn());
  }
  play() {
    this.paused = false;
    this._emit('play');
    const step = () => {
      if (this.paused) return;
      this.currentTime += 1 / 20;
      if (this.currentTime >= this.duration) {
        this.currentTime = this.duration;
        this.pause();
        this._emit('ended');
        return;
      }
      document.getElementById('scrub-range').value = Math.round(this.currentTime);
      this._emit('timeupdate');
      this._raf = requestAnimationFrame(step);
    };
    this._raf = requestAnimationFrame(step);
  }
  pause() {
    this.paused = true;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._emit('pause');
  }
  seekTo(t) {
    this.currentTime = Math.max(0, Math.min(this.duration, t));
    this._emit('seeked');
  }
}
const video = new FakeVideo(DATA.length - 1);
document.getElementById('scrub-range').max = DATA.length - 1;
document.getElementById('btn-play').addEventListener('click', () => video.play());
document.getElementById('btn-pause').addEventListener('click', () => video.pause());
document.getElementById('scrub-range').addEventListener('input', (e) => video.seekTo(Number(e.target.value)));
video._emit('loadedmetadata');"""


def make_player(session_dir, csv_name, video_name, out_name, markers_name="markers.csv", npz_name="eeg_raw.npz",
                 ratings_name="ratings.csv"):
    session_dir = Path(session_dir)
    csv_path = session_dir / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {csv_path} (run run_emotion_inference.py first)")

    has_video = video_name is not None
    if has_video:
        video_path = session_dir / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    data = [[float(r["prob_negative"]), float(r["prob_neutral"]), float(r["prob_positive"])] for r in rows]
    labels = [r["pred_label"] for r in rows]
    true_duration = get_true_duration_s(session_dir, markers_name) if has_video else None

    true_labels = load_true_labels(session_dir, ratings_name, len(data))
    if true_labels is None:
        true_labels = [None] * len(data)
        print(f"NOTE: {ratings_name} not found in {session_dir} — player will skip the ground-truth label strip.")

    wave = load_wave_for_display(session_dir, npz_name)
    if wave is None:
        wave = {"labels": [], "srate": WAVE_DISPLAY_HZ, "scales": [], "n_samples": 0, "b64": ""}
        print(f"NOTE: {npz_name} not found in {session_dir} — player will skip the raw-waveform panel.")

    if has_video:
        video_html = f'<video id="vid" controls src="{video_name}"></video>'
        video_js_init = VIDEO_JS
    else:
        video_html = NOVIDEO_HTML
        video_js_init = NOVIDEO_JS

    html = TEMPLATE.format(
        session_name=session_dir.name,
        video_html=video_html,
        video_js_init=video_js_init,
        data_json=json.dumps(data),
        labels_json=json.dumps(labels),
        true_labels_json=json.dumps(true_labels),
        true_duration_json=json.dumps(true_duration),
        wave_display_hz=WAVE_DISPLAY_HZ,
        wave_labels_json=json.dumps(wave["labels"]),
        wave_srate_json=json.dumps(wave["srate"]),
        wave_scales_json=json.dumps(wave["scales"]),
        wave_n_samples_json=json.dumps(wave["n_samples"]),
        wave_window_s_json=json.dumps(WAVE_WINDOW_S),
        wave_filter_low=WAVE_FILTER_LOW,
        wave_filter_high=WAVE_FILTER_HIGH,
        wave_b64=wave["b64"],
    )

    out_path = session_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("Saved:", out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--csv-name", default="emotion_predictions.csv")
    parser.add_argument("--video-name", default="screen_recording.mp4")
    parser.add_argument("--no-video", action="store_true", help="No screen recording (e.g. bare SEED .npz) — scrubber UI instead of a <video>")
    parser.add_argument("--out-name", default="emotion_player.html")
    parser.add_argument("--npz-name", default="eeg_raw.npz")
    parser.add_argument("--ratings-name", default="ratings.csv")
    args = parser.parse_args()
    video_name = None if args.no_video else args.video_name
    make_player(args.session_dir, args.csv_name, video_name, args.out_name, npz_name=args.npz_name,
                ratings_name=args.ratings_name)


if __name__ == "__main__":
    main()
