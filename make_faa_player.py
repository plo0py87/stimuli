"""Generate a self-contained HTML player for Frontal Alpha Asymmetry (FAA)
scores (see faa_score.py), synced with the raw EEG waveform panel. No
video required -- reuses the no-video scrubber UI from make_emotion_player.py.

This shows the raw continuous FAA scores only -- no threshold/classification
applied (that's a deliberate next step, not done here).

Usage:
  python make_faa_player.py --npz-path SEED/trials/trial01_positive.npz
  python make_faa_player.py --npz-path sessions/<session>/eeg_raw.npz --video-name screen_recording_h264.mp4
"""

import argparse
import csv
import json
from pathlib import Path

from make_emotion_player import (
    NOVIDEO_HTML, NOVIDEO_JS, VIDEO_JS, WAVE_DISPLAY_HZ, WAVE_FILTER_HIGH,
    WAVE_FILTER_LOW, WAVE_WINDOW_S, get_true_duration_s, load_wave_for_display,
)

TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>FAA playback - {session_name}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }}
  h2 {{ font-weight: 500; font-size: 15px; color: #aaa; margin: 0 0 12px; }}
  video {{ width: 100%; max-height: 60vh; background: #000; display: block; border-radius: 6px; }}
  .panel {{ max-width: 960px; margin: 0 auto; }}
  .scores {{ display: flex; gap: 12px; margin: 14px 0; flex-wrap: wrap; }}
  .score {{ flex: 1; min-width: 140px; background: #1c1c1c; border-radius: 6px; padding: 10px 12px; }}
  .score .name {{ font-size: 12px; color: #999; text-transform: uppercase; letter-spacing: .04em; }}
  .score .val {{ font-size: 24px; font-weight: 600; margin-top: 2px; font-family: ui-monospace, monospace; }}
  .label-line {{ font-size: 14px; color: #ccc; margin: 4px 0 0; }}
  .label-line b {{ color: #fff; }}
  canvas {{ width: 100%; height: 140px; display: block; background: #1c1c1c; border-radius: 6px; }}
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
  <h2>{session_name} &mdash; Frontal Alpha Asymmetry (no threshold applied, raw scores only)</h2>
  {video_html}

  <p class="label-line">t=<span id="tsec">0</span>s</p>

  <div class="scores">
    <div class="score f3f4"><div class="name">F3/F4</div><div class="val" id="val-f3f4">0.00</div></div>
    <div class="score f7f8"><div class="name">F7/F8</div><div class="val" id="val-f7f8">0.00</div></div>
    <div class="score fp1fp2"><div class="name">Fp1/Fp2</div><div class="val" id="val-fp1fp2">0.00</div></div>
    <div class="score combined"><div class="name">Combined</div><div class="val" id="val-combined">0.00</div></div>
  </div>

  <canvas id="timeline"></canvas>
  <div class="legend">
    <span><span class="dot" style="background:#ff6b6b"></span>F3/F4</span>
    <span><span class="dot" style="background:#ffd93d"></span>F7/F8</span>
    <span><span class="dot" style="background:#6bcB77"></span>Fp1/Fp2</span>
    <span><span class="dot" style="background:#fff"></span>Combined</span>
    <span>(positive = relatively more left frontal activation -&gt; classically approach/positive; negative = withdrawal/negative)</span>
    <span id="scale-note"></span>
  </div>

  <div class="section-title">Raw EEG ({wave_filter_low:g}-{wave_filter_high:g}Hz, scrolling, centered white line = now) &nbsp;
    <span id="gain-note" style="text-transform:none; letter-spacing:0;">gain x4.0 (press +/- to adjust)</span>
  </div>
  <canvas id="rawwave" tabindex="0"></canvas>
</div>

<script>
const DATA = {data_json};  // [[f3f4, f7f8, fp1fp2, combined], ...] one row per second
const TRUE_DURATION_S = {true_duration_json};
let timeScale = 1.0;

const WAVE_LABELS = {wave_labels_json};
const WAVE_SRATE = {wave_srate_json};
const WAVE_SCALES = {wave_scales_json};
const WAVE_N_SAMPLES = {wave_n_samples_json};
const WAVE_WINDOW_S_ = {wave_window_s_json};
let waveData = null;
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
const waveCanvas = document.getElementById('rawwave');
const waveCtx = waveCanvas.getContext('2d');
let waveGain = 4.0;

function resizeCanvas() {{
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * devicePixelRatio;
  canvas.height = rect.height * devicePixelRatio;
  const waveRect = waveCanvas.getBoundingClientRect();
  waveCanvas.width = waveRect.width * devicePixelRatio;
  waveCanvas.height = waveRect.height * devicePixelRatio;
}}

function drawWave(realElapsed) {{
  const w = waveCanvas.width, h = waveCanvas.height;
  waveCtx.clearRect(0, 0, w, h);
  if (!waveData || !WAVE_LABELS.length) return;

  const nCh = WAVE_LABELS.length;
  const rowH = h / nCh;
  const halfWin = WAVE_WINDOW_S_ / 2;
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

    waveCtx.beginPath();
    waveCtx.strokeStyle = '#7fd0ff';
    waveCtx.lineWidth = 1 * devicePixelRatio;
    let started = false;
    for (let s = winLo; s <= winHi; s++) {{
      const t = s / WAVE_SRATE;
      const x = ((t - tStart) / WAVE_WINDOW_S_) * w;
      const y = rowY - waveData[base + s] * scale;
      if (!started) {{ waveCtx.moveTo(x, y); started = true; }} else {{ waveCtx.lineTo(x, y); }}
    }}
    waveCtx.stroke();

    waveCtx.font = `${{11 * devicePixelRatio}}px system-ui, sans-serif`;
    waveCtx.fillStyle = '#999';
    waveCtx.textBaseline = 'middle';
    waveCtx.fillText(WAVE_LABELS[ch], 4 * devicePixelRatio, rowY);
  }}

  const nowX = (halfWin / WAVE_WINDOW_S_) * w;
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

  let lo = Infinity, hi = -Infinity;
  for (const row of DATA) for (const v of row) {{ if (v < lo) lo = v; if (v > hi) hi = v; }}
  if (lo === hi) {{ lo -= 1; hi += 1; }}
  const pad = (hi - lo) * 0.08;
  lo -= pad; hi += pad;
  const yOf = (v) => h - ((v - lo) / (hi - lo)) * h;

  // zero line
  ctx.beginPath();
  ctx.strokeStyle = '#555';
  ctx.setLineDash([4 * devicePixelRatio, 4 * devicePixelRatio]);
  ctx.moveTo(0, yOf(0));
  ctx.lineTo(w, yOf(0));
  ctx.stroke();
  ctx.setLineDash([]);

  const colors = ['#ff6b6b', '#ffd93d', '#6bcB77', '#ffffff'];
  const widths = [1.2, 1.2, 1.2, 2.0];
  for (let series = 0; series < 4; series++) {{
    ctx.beginPath();
    ctx.strokeStyle = colors[series];
    ctx.lineWidth = widths[series] * devicePixelRatio;
    for (let i = 0; i < n; i++) {{
      const x = (i / (n - 1)) * w;
      const y = yOf(DATA[i][series]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }}

  const px = (currentSec / (n - 1)) * w;
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(255,255,255,0.5)';
  ctx.lineWidth = 2 * devicePixelRatio;
  ctx.moveTo(px, 0);
  ctx.lineTo(px, h);
  ctx.stroke();
}}

function update() {{
  const realElapsed = video.currentTime * timeScale;
  const idx = Math.max(0, Math.min(DATA.length - 1, Math.round(realElapsed)));
  const [f3f4, f7f8, fp1fp2, combined] = DATA[idx];

  document.getElementById('tsec').textContent = idx;
  document.getElementById('val-f3f4').textContent = f3f4.toFixed(3);
  document.getElementById('val-f7f8').textContent = f7f8.toFixed(3);
  document.getElementById('val-fp1fp2').textContent = fp1fp2.toFixed(3);
  document.getElementById('val-combined').textContent = combined.toFixed(3);

  drawTimeline(idx);
  drawWave(realElapsed);
}}

let rafId = null;
function rafLoop() {{ update(); rafId = requestAnimationFrame(rafLoop); }}
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
        ` | video is ${{video.duration.toFixed(1)}}s but recording was ${{TRUE_DURATION_S.toFixed(1)}}s -- rescaled x${{timeScale.toFixed(3)}}`;
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


def make_faa_player(npz_path, faa_csv_path, out_path, video_name=None, session_dir=None):
    npz_path = Path(npz_path)
    faa_csv_path = Path(faa_csv_path)
    session_dir = Path(session_dir) if session_dir else npz_path.parent

    with open(faa_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    data = [[float(r["faa_f3f4"]), float(r["faa_f7f8"]), float(r["faa_fp1fp2"]), float(r["faa_combined"])] for r in rows]

    has_video = video_name is not None
    if has_video:
        video_path = session_dir / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        video_html = f'<video id="vid" controls src="{video_name}"></video>'
        video_js_init = VIDEO_JS
        true_duration = get_true_duration_s(session_dir)
    else:
        video_html = NOVIDEO_HTML
        video_js_init = NOVIDEO_JS
        true_duration = None

    wave = load_wave_for_display(npz_path.parent, npz_path.name)
    if wave is None:
        wave = {"labels": [], "srate": WAVE_DISPLAY_HZ, "scales": [], "n_samples": 0, "b64": ""}

    html = TEMPLATE.format(
        session_name=npz_path.stem,
        video_html=video_html,
        video_js_init=video_js_init,
        data_json=json.dumps(data),
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

    out_path = Path(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("Saved:", out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--faa-csv", default=None, help="Defaults to <npz-stem>_faa.csv next to the npz")
    parser.add_argument("--out-name", default=None, help="Defaults to <npz-stem>_faa_player.html next to the npz")
    parser.add_argument("--video-name", default=None)
    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    faa_csv = Path(args.faa_csv) if args.faa_csv else npz_path.with_name(npz_path.stem + "_faa.csv")
    out_name = Path(args.out_name) if args.out_name else npz_path.with_name(npz_path.stem + "_faa_player.html")
    make_faa_player(npz_path, faa_csv, out_name, video_name=args.video_name)


if __name__ == "__main__":
    main()
