"""Build the HTML report for the nine-montage REFED study."""

import base64
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_final import (  # noqa: E402
    ORDER, LABEL, COLORS, collect_causal, load, mean_ccc, sd_ccc,
)
from refed_montage_regress import DIMS, to_class  # noqa: E402


def fig(name, alt, cap):
    p = HERE / name
    if not p.exists():
        return ""
    b = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b}" alt="{alt}">'
            f'<figcaption>{cap}</figcaption></figure>')


def main():
    causal = collect_causal()
    libeer = load("libeer_montage.json").get("pooled", {})
    lib_kal = load("libeer_kalman_all.json").get("pooled", {})
    montages = json.loads((HERE / "libeer_kalman_all.json").read_text())["montage_channels"]

    pz = np.load(HERE / "montage_predictions.npz")
    C = np.stack([to_class(pz["Y"][:, 0]), to_class(pz["Y"][:, 1])], axis=1)
    maj = {d: float(np.bincount(C[:, i], minlength=3).max() / len(C))
           for i, d in enumerate(DIMS)}
    n_sub = len(np.unique(pz["subject"]))
    ref = mean_ccc(causal["whole"], "valence")

    rows = ""
    for m in ORDER:
        if m not in causal:
            continue
        pct = mean_ccc(causal[m], "valence") / ref * 100
        lv = f"{mean_ccc(libeer[m],'valence'):+.3f}" if m in libeer else "—"
        kv = f"{mean_ccc(lib_kal[m],'valence'):+.3f}" if m in lib_kal else "—"
        rows += (
            f"<tr><td><span class='dot' style='background:{COLORS[m]}'></span>"
            f"{LABEL[m]}</td><td class='num'>{len(montages[m])}</td>"
            f"<td class='num'><strong>{mean_ccc(causal[m],'valence'):+.3f}</strong>"
            f" <span class='sd'>±{sd_ccc(causal[m],'valence'):.3f}</span></td>"
            f"<td class='num'>{mean_ccc(causal[m],'arousal'):+.3f}"
            f" <span class='sd'>±{sd_ccc(causal[m],'arousal'):.3f}</span></td>"
            f"<td class='num'>{pct:.0f}%</td>"
            f"<td class='num sub'>{lv}</td><td class='num sub'>{kv}</td></tr>")

    chan_rows = "".join(
        f"<tr><td><span class='dot' style='background:{COLORS[m]}'></span>{LABEL[m]}</td>"
        f"<td class='num'>{len(montages[m])}</td><td class='chan'>"
        + (" ".join(montages[m]) if m != "whole"
           else "all 64 except M1, M2 (mastoid references) — the SEED 62 set")
        + "</td></tr>" for m in ORDER)

    html = f"""<title>REFED × DGCNN — Electrode Montages for a Head-Worn Device</title>
<style>
:root{{--bg:#f6f2ea;--ink:#1c2320;--sub:#55605a;--card:#fff;--line:#e2dccb;
--accent:#2f5f5c;--warn:#c1443f;--gold:#b8873a}}
:root[data-theme="dark"]{{--bg:#14181a;--ink:#eef0ec;--sub:#a7b0aa;--card:#1d2325;
--line:#333c3a;--accent:#6fb3a8;--warn:#e08a86;--gold:#d9ab5f}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
--bg:#14181a;--ink:#eef0ec;--sub:#a7b0aa;--card:#1d2325;--line:#333c3a;
--accent:#6fb3a8;--warn:#e08a86;--gold:#d9ab5f}}}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--ink);font-family:Georgia,serif;
line-height:1.55;padding:2.5rem 1.25rem 5rem}}
.wrap{{max-width:980px;margin:0 auto}}
.eyebrow{{font-family:"Segoe UI",system-ui,sans-serif;text-transform:uppercase;
letter-spacing:.14em;font-size:.72rem;color:var(--sub);margin:0 0 .4rem}}
h1{{font-size:2rem;margin:0 0 .35rem;text-wrap:balance;font-weight:600}}
.dek{{color:var(--sub);font-family:"Segoe UI",system-ui,sans-serif;font-size:1rem;
max-width:65ch;margin:0 0 2rem}}
h2{{font-size:1.2rem;margin:2.6rem 0 .9rem;padding-bottom:.4rem;
border-bottom:1px solid var(--line);font-weight:600}}
p{{margin:.55rem 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.9rem;margin:1rem 0}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:.85rem 1rem}}
.stat .n{{font-family:"Segoe UI",system-ui,sans-serif;font-variant-numeric:tabular-nums;
font-size:1.4rem;font-weight:700;color:var(--accent)}}
.stat .l{{font-family:"Segoe UI",system-ui,sans-serif;font-size:.76rem;color:var(--sub);margin-top:.15rem}}
.tablewrap{{overflow-x:auto;border:1px solid var(--line);border-radius:6px;margin:1rem 0}}
table{{width:100%;border-collapse:collapse;font-family:"Segoe UI",system-ui,sans-serif;font-size:.85rem}}
th,td{{text-align:left;padding:.5rem .7rem;border-bottom:1px solid var(--line);white-space:nowrap}}
th{{color:var(--sub);font-weight:600;text-transform:uppercase;font-size:.67rem;letter-spacing:.05em}}
td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
td.sub{{color:var(--sub)}}
td.chan{{font-family:ui-monospace,Consolas,monospace;font-size:.76rem;white-space:normal;color:var(--sub)}}
tr:last-child td{{border-bottom:none}}
.sd{{color:var(--sub);font-size:.85em}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:.45rem;vertical-align:middle}}
figure{{margin:1.2rem 0 1.8rem}}
figure img{{width:100%;display:block;border-radius:6px;border:1px solid var(--line)}}
figcaption{{font-family:"Segoe UI",system-ui,sans-serif;font-size:.8rem;color:var(--sub);margin-top:.5rem}}
.callout{{border-left:3px solid var(--gold);background:color-mix(in srgb,var(--gold) 8%,var(--card));
padding:.85rem 1rem;border-radius:0 6px 6px 0;font-family:"Segoe UI",system-ui,sans-serif;
font-size:.88rem;margin:1.1rem 0}}
.callout.warn{{border-left-color:var(--warn);background:color-mix(in srgb,var(--warn) 8%,var(--card))}}
code{{font-family:ui-monospace,Consolas,monospace;font-size:.9em}}
.foot{{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);
font-family:"Segoe UI",system-ui,sans-serif;font-size:.78rem;color:var(--sub)}}
</style>
<div class="wrap">
<p class="eyebrow">REFED · LibEER DGCNN · continuous valence &amp; arousal</p>
<h1>Where should the electrodes go on a head-worn device?</h1>
<p class="dek">Nine electrode layouts — from a full 62-channel cap down to a
glasses frame — trained with the same DGCNN on the same data, predicting the
continuous joystick valence/arousal trace of REFED's {n_sub} subjects. The
question is not how well emotion can be decoded in the abstract, but how much
of it survives the layouts a wearable can actually carry.</p>

<div class="grid">
<div class="stat"><div class="n">{n_sub}</div><div class="l">subjects</div></div>
<div class="stat"><div class="n">9</div><div class="l">electrode montages</div></div>
<div class="stat"><div class="n">3</div><div class="l">preprocessing variants</div></div>
<div class="stat"><div class="n">86%</div><div class="l">of full-cap signal from a 20-electrode VR ring</div></div>
</div>

<h2>Headline</h2>
<div class="callout"><strong>Placement beats count.</strong> Three montages with
almost the same number of electrodes span a 1.7× range: the VR ring (20 ch)
reaches +0.177 valence CCC, Quick-20 (19 ch) +0.152, and the head-top band
(21 ch) only +0.106. What separates them is how far apart the electrodes are —
the ring spans forehead to occiput, the head-top band crowds the vertex.</div>

<div class="tablewrap"><table>
<thead><tr>
<th>montage</th><th class="num">ch</th>
<th class="num">valence CCC</th><th class="num">arousal CCC</th>
<th class="num">vs whole</th>
<th class="num">LibEER</th><th class="num">LibEER+K</th>
</tr></thead>
<tbody>{rows}</tbody>
</table></div>
<p style="font-size:.88rem;color:var(--sub)">Bold columns are the causal,
realtime-legal pipeline (the headline). The last two columns show the same
montages under LibEER's exact SEED recipe, without and with the temporal
smoother, to demonstrate that the ranking is not an artifact of our
preprocessing. ± is spread across subjects, not a confidence interval.</p>

{fig("fig_ranking.png", "Montage ranking by CCC",
     "Sorted by valence CCC. Error bars are the subject-to-subject spread, which "
     "is wide enough that neighbouring montages are not individually separable — "
     "the trend across the range is the finding, not any single pairwise gap.")}

<h2>The nine layouts</h2>
{fig("fig_montages_all.png", "Nine electrode montages drawn on a head outline",
     "Filled circles are each montage's electrodes; faint circles are REFED's "
     "remaining channels.")}

<div class="tablewrap"><table>
<thead><tr><th>montage</th><th class="num">ch</th><th>channels</th></tr></thead>
<tbody>{chan_rows}</tbody>
</table></div>

<h2>How far back should the strap reach?</h2>
<p>Extending backward from the glasses frame helps — but only up to a point,
and the point is the occipital line.</p>
<div class="callout"><strong>Useful:</strong> adding P7/P8 lifts valence from
+0.109 to +0.141, and completing the occipital ring (PO7/PO8, O1/OZ/O2) reaches
+0.177. <strong>Counterproductive:</strong> pushing lower to the mastoids
(M1/M2, +0.100) or the cerebellar sites (CB1/CB2, +0.166 vs the ring's +0.177)
makes things worse — those positions pick up more neck EMG than brain signal.
For hardware: the rear strap should sit on the occiput, not below it.</div>

<h2>Preprocessing: it is the smoother, not the bandpass</h2>
<p>Our pipeline originally had no global bandpass, while SEED arrives already
band-limited — a real asymmetry worth closing. Reproducing LibEER's SEED recipe
exactly (0.3–50 Hz filtfilt, per-band filtfilt DE, log2/ddof=1) closed it, and
made every result dramatically <em>worse</em>: whole-cap valence fell from
+0.205 to +0.049. Adding the forward-only Kalman smoother back — the one thing
LibEER's plain DE omits — recovered it to +0.189.</p>
{fig("fig_preproc.png", "Three preprocessing variants across montages",
     "The bandpass was never what mattered. Temporal smoothing was carrying the "
     "result the whole time.")}
<div class="callout"><strong>Why.</strong> REFED's joystick label moves slowly
and in steps; unsmoothed per-second DE jitters far faster. Smoothing aligns the
feature timescale to the label's. LibEER's plain DE is tuned for SEED, where each
trial carries one fixed label and every second is classified independently — a
task that needs no cross-second smoothing. Same library, different task,
different right answer.</div>
<div class="callout"><strong>Realtime costs nothing here.</strong> The causal
pipeline (+0.205) slightly outperforms LibEER's non-causal recipe with smoothing
(+0.189). That is the opposite of what the same comparison shows on SEED, where
non-causal LDS is worth roughly ten accuracy points. The difference is that
SEED's label is static, so smoothing across the whole trial is nearly free
information; REFED's label moves, and a forward-only smoother is already
sufficient.</div>

<h2>What a prediction looks like</h2>
{fig("fig_timeline.png", "Predicted versus true trajectory",
     "One subject, one clip. The truth (black) is step-shaped because subjects "
     "nudge the joystick and hold it. Predictions track the direction of change "
     "but overshoot its magnitude.")}

<h2>The variance behind the averages</h2>
{fig("fig_persubject.png", "Per-subject CCC distributions",
     "Each dot is a subject. Every montage has subjects below zero, where the "
     "model does worse than predicting that person's mean.")}

<h2>Caveats before anyone quotes this</h2>
<div class="callout warn"><strong>Front electrodes carry eye movement.</strong>
FP1/FPZ/FP2/AF3/AF4 sit directly above the eyes and F7/F8/FT7/FT8 near the
temples, so they record EOG from blinks and saccades. Eye behaviour correlates
with video content, so part of every frontal montage's score may be ocular
rather than neural. Nothing here separates them.</div>
<div class="callout warn"><strong>Occipital electrodes carry visual response.</strong>
The VR ring's advantage comes substantially from O1/OZ/O2, which sit over visual
cortex. While subjects watch video, those channels track luminance, motion and
scene cuts — which correlate with emotional content without being emotion. The
ring's lead over Quick-20 needs a visual-confound control before it supports a
product claim.</div>
<div class="callout warn"><strong>Absolute performance is weak.</strong> A
whole-cap valence CCC of {ref:.2f} is real signal but far from usable, and 3-class
accuracy sits barely above the majority baseline
({maj['valence']*100:.1f}% valence, {maj['arousal']*100:.1f}% arousal). These runs
establish a <em>relative</em> ordering among montages; they do not establish that
any of them is deployable.</div>

<h2>Method</h2>
<p><strong>Features.</strong> 5-band differential entropy (δ 1–4, θ 4–8, α 8–14,
β 14–31, γ 31–50 Hz), forward-only Butterworth via <code>lfilter</code>, variance
over 1 s windows, then a forward-only scalar Kalman smoother (q=0.01, r=0.5).
Causal decimation 1000→200 Hz. No LDS, no <code>filtfilt</code>, no look-ahead:
realtime-legal end to end. The LibEER-matched arm replaces all of this with
<code>data_utils/preprocess.py</code>'s exact steps and is <em>not</em>
realtime-legal.</p>
<p><strong>Splits.</strong> Within-subject, by video and never by second —
adjacent seconds are near-identical after smoothing. REFED's 15 clips are 5
emotion targets × 3 clips, so each of 3 folds holds out one clip per target.
Two training clips are reserved for early stopping; normalization uses each
subject's own training clips. One model is trained per montage per fold across
all subjects (a per-subject variant was also run: with only ~800 training
seconds against DGCNN's ~1M-parameter fc layer it underfits so badly that the
smallest montage wins, which is an overfitting artifact, not a ranking).</p>
<p><strong>Targets.</strong> 1 Hz joystick position rescaled to [-1, 1]. The
leading stretch where the stick still sits parked at 128 is dropped rather than
labeled neutral — subjects take a median of 13 s to first move it. Trials where
either axis never moved are dropped (28 of 480).</p>

<div class="foot">Scripts in <code>C:/Dev/BCI/stimuli/refed/</code>:
<code>refed_extract_all.py</code> (causal features),
<code>refed_extract_libeer.py</code> (LibEER-matched), <code>add_kalman.py</code>,
<code>refed_montage_regress.py</code>, <code>refed_montage_classify.py</code>,
<code>build_final.py</code>, plus <code>diag_*.py</code> — the ridge, pooling,
SEED-label and same-pipeline-on-SEED diagnostics that shaped this design.
Data: REFED, CC BY-NC-SA 4.0.</div>
</div>"""

    (HERE / "report.html").write_text(html, encoding="utf-8")
    print(f"report.html written ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
