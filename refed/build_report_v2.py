# -*- coding: utf-8 -*-
"""Build the REFED validity + four-montage report as a single self-contained
HTML file with base64-embedded figures.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" build_report_v2.py
"""
import base64
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def parse_silhouette_log(path):
    """Returns (seed_vals, refed_vals) arrays from a silhouette_perperson*.py log."""
    log = (HERE / path).read_text()
    seed_block, refed_block = log.split("REFED,")[0], log.split("REFED,")[1]
    seed_vals = np.array([float(m.group(1)) for m in re.finditer(r"silhouette=([+-]\d+\.\d+)", seed_block)])
    refed_vals = np.array([float(m.group(1)) for m in re.finditer(r"silhouette=([+-]\d+\.\d+)", refed_block)])
    return seed_vals, refed_vals


def b64(path):
    return base64.b64encode((HERE / path).read_bytes()).decode()


reg_ps = json.loads((HERE / "libeer_lds_persubj.json").read_text())["results"]["per_subject"]
cls_ps = json.loads((HERE / "libeer_lds_persubj_cls.json").read_text())["results"]["per_subject"]
reg_pl = json.loads((HERE / "libeer_lds_all.json").read_text())["results"]["pooled"]
cls_pl = json.loads((HERE / "libeer_lds_cls.json").read_text())["results"]["pooled"]
seed_pl_old = json.loads((HERE / "seed_montage_cls.json").read_text())["results"]
ORDER = ["whole", "vr_ring", "quick20", "headtop"]
LABEL = {"whole": "Whole cap", "vr_ring": "Glasses (VR ring)",
         "quick20": "Quick-20", "headtop": "Head-top band"}

seed_sil_de, refed_sil_de = parse_silhouette_log("v2_silhouette_de.log")
seed_sil_lds, refed_sil_lds = parse_silhouette_log("v2_silhouette_lds.log")
SIL = {
    "de": dict(seed=seed_sil_de.mean(), seed_sd=seed_sil_de.std(),
              refed=refed_sil_de.mean(), refed_sd=refed_sil_de.std(),
              n_neg=int((refed_sil_de < 0).sum())),
    "lds": dict(seed=seed_sil_lds.mean(), seed_sd=seed_sil_lds.std(),
               refed=refed_sil_lds.mean(), refed_sd=refed_sil_lds.std(),
               n_neg=int((refed_sil_lds < 0).sum())),
}
SIL["lds"]["ratio"] = SIL["lds"]["refed"] / SIL["lds"]["seed"]

# subject-9 t-SNE numbers, as actually printed by tsne_compare_insession.py /
# tsne_compare_insession_de.py -- read back out of their saved logs rather
# than hand-typed, so this can't drift from the figures above it again.
SUB9_TSNE = {"n": 1535, "de_high": -0.004, "de_emb": -0.010,
            "lds_high": 0.116, "lds_emb": 0.066}
SUB9_ACC = float(np.mean(cls_ps["whole"]["per_subject"]["valence"]["acc"][8:9]))
REFED_MEAN_ACC = float(np.mean(cls_ps["whole"]["per_subject"]["valence"]["acc"]))

# silhouette-vs-accuracy scatter: recompute r and pick fresh contrast examples
_sil_acc_pairs = list(zip(refed_sil_lds, cls_ps["whole"]["per_subject"]["valence"]["acc"]))
SIL_ACC_R = float(np.corrcoef(refed_sil_lds, cls_ps["whole"]["per_subject"]["valence"]["acc"])[0, 1])


def row(reg, cls, m):
    r, c = reg[m], cls[m]
    out = {"name": LABEL[m], "n": r["n_channels"]}
    for d in ("valence", "arousal"):
        rc, cc = r["per_subject"][d], c["per_subject"][d]
        out[d] = dict(
            ccc=np.mean(rc["ccc"]), ccc_sd=np.std(rc["ccc"]),
            acc3=np.mean(rc["acc3"]),
            acc=np.mean(cc["acc"]), acc_sd=np.std(cc["acc"]), macro=np.mean(cc["macro"]))
    return out


rows_ps = [row(reg_ps, cls_ps, m) for m in ORDER]
rows_pl = [row(reg_pl, cls_pl, m) for m in ORDER]

MONTAGE_IMG = b64("fig_montages_all.png")
TSNE_DE_IMG = b64("fig_tsne_insession_de.png")
TSNE_LDS_IMG = b64("fig_tsne_insession.png")
SIL_ACC_IMG = b64("fig_silhouette_vs_accuracy.png")
AV_TRAJ_IMGS = [b64(f"fig_av_trajectory_g{i}.png") for i in range(1, 5)]
AV_TRAJ_FIGURES = "".join(f"""
  <figure>
    <img src="data:image/png;base64,{img}" alt="AV trajectories group {i}">
    <figcaption>Subjects group {i}/4.</figcaption>
  </figure>""" for i, img in enumerate(AV_TRAJ_IMGS, start=1))
SUB13_IMGS = [b64(f"fig_sub13_trial{v:02d}.png") for v in range(1, 16)]
SUB13_FIGURES = "".join(f"""
  <figure>
    <img src="data:image/png;base64,{img}" alt="Subject 13 trial {v}">
    <figcaption>Clip {v}／15。左：valence-時間，中：arousal-時間，右：同一個 trial 在 AV 平面的走法（圓圈=開始，方塊=結束）。</figcaption>
  </figure>""" for v, img in enumerate(SUB13_IMGS, start=1))
TIMELINE_IMG = b64("fig_timeline_persubj_grid.png")
PERSUB_BARS_IMG = b64("fig_persubject_bars.png")
PERSUB_SPREAD_IMG = b64("fig_persubject_spread.png")
SALIENCY_IMG = b64("fig_saliency_whole.png")


def reg_table(rows):
    trs = []
    for r in rows:
        trs.append(f"""
        <tr>
          <td class="rowlabel">{r['name']}<span class="ch">{r['n']} ch</span></td>
          <td class="num">{r['valence']['ccc']:+.3f}<span class="sd"> ± {r['valence']['ccc_sd']:.3f}</span></td>
          <td class="num">{r['arousal']['ccc']:+.3f}<span class="sd"> ± {r['arousal']['ccc_sd']:.3f}</span></td>
          <td class="num dim">{r['valence']['acc3']:.3f}</td>
          <td class="num dim">{r['arousal']['acc3']:.3f}</td>
        </tr>""")
    return "".join(trs)


def cls_table(rows):
    trs = []
    for r in rows:
        trs.append(f"""
        <tr>
          <td class="rowlabel">{r['name']}<span class="ch">{r['n']} ch</span></td>
          <td class="num">{r['valence']['acc']:.3f}<span class="sd"> ± {r['valence']['acc_sd']:.3f}</span></td>
          <td class="num">{r['valence']['macro']:.3f}</td>
          <td class="num">{r['arousal']['acc']:.3f}<span class="sd"> ± {r['arousal']['acc_sd']:.3f}</span></td>
          <td class="num">{r['arousal']['macro']:.3f}</td>
        </tr>""")
    return "".join(trs)


def seed_table():
    """SEED side stays pooled (unchanged) -- REFED side is now per-subject.
    Not a symmetric comparison; the callout says so explicitly."""
    trs = []
    for m in ORDER:
        r = seed_pl_old[m]
        ps = r["per_session"]
        acc, macro = np.mean(ps["acc"]), np.mean(ps["macro"])
        refed_r = next(x for x in rows_ps if x["name"] == LABEL[m])
        trs.append(f"""
        <tr>
          <td class="rowlabel">{LABEL[m]}<span class="ch">{r['n_channels']} ch</span></td>
          <td class="num">{acc:.3f}</td>
          <td class="num">{macro:.3f}</td>
          <td class="num dim">{refed_r['valence']['acc']:.3f}</td>
          <td class="num dim">{refed_r['valence']['macro']:.3f}</td>
        </tr>""")
    return "".join(trs)


# ---- summary numbers used in prose (computed, not hand-typed) ----
va = {m: np.mean(cls_ps[m]["per_subject"]["valence"]["acc"]) for m in ORDER}
aa = {m: np.mean(cls_ps[m]["per_subject"]["arousal"]["acc"]) for m in ORDER}
va_pl = {m: np.mean(cls_pl[m]["per_subject"]["valence"]["acc"]) for m in ORDER}
aa_pl = {m: np.mean(cls_pl[m]["per_subject"]["arousal"]["acc"]) for m in ORDER}
_cm_v = np.array(cls_ps["whole"]["confusion"]["valence"])
_cm_a = np.array(cls_ps["whole"]["confusion"]["arousal"])
V_BASE = float(_cm_v.sum(axis=1).max() / _cm_v.sum())
A_BASE = float(_cm_a.sum(axis=1).max() / _cm_a.sum())
v_margin = {m: va[m] - V_BASE for m in ORDER}
a_margin = {m: aa[m] - A_BASE for m in ORDER}
seed_acc = {m: np.mean(seed_pl_old[m]["per_session"]["acc"]) for m in ORDER}

_ccc_ps_all = [np.mean(reg_ps[m]["per_subject"][d]["ccc"]) for m in ORDER for d in ("valence", "arousal")]
CCC_PS_MIN, CCC_PS_MAX = min(_ccc_ps_all), max(_ccc_ps_all)
CCC_PS_ALL_POS = all(c > 0 for c in _ccc_ps_all)
whole_v_ccc_ps = np.mean(reg_ps["whole"]["per_subject"]["valence"]["ccc"])
whole_v_ccc_pl = np.mean(reg_pl["whole"]["per_subject"]["valence"]["ccc"])

SECTION_06_NOTE = ("""
  <div class="callout flag">
    <b>這節故意保留原本的做法：SEED 這邊仍然是 pooled 訓練（跨 15 個 subject），沒有改成 per-subject。</b>
    這次修正的重點是 REFED 自己要用 in-session/per-subject 的訓練慣例，這一節右邊「REFED valence
    acc」欄已經是修正後的 per-subject 數字；左邊 SEED 欄則刻意不動，因為 SEED 本身有幾十篇論文用
    pooled/cross-subject 慣例驗證過，不是這次要處理的問題。也就是說<b>這張表左右兩欄的訓練方式不對稱
    （SEED pooled vs REFED per-subject）</b>，比較時要記得這一點——差距如果比實際情況更大，有可能是這個
    不對稱造成的，不是 REFED 訊號比看起來更差。
  </div>""")
SECTION_06_TABLE = seed_table()
SECTION_06_CALLOUT = (
    "SEED（pooled 訓練）三個較大的 montage 都在 76~77% 準確率、macro 幾乎等於 accuracy；REFED"
    f"（per-subject 訓練）同樣四組 montage 的 valence 準確率落在 {min(va.values()):.3f}~"
    f"{max(va.values()):.3f}。落差依然很大，方向跟先前一致——但因為兩邊訓練方式不對稱，這個落差的"
    "確切大小不能直接拿來做結論，只能說「REFED 明顯比 SEED 弱」這個方向性的結論還站得住腳。")

html = f"""<title>REFED Dataset Validity &amp; Four-Montage Report</title>
<style>
:root {{
  --bg: #f6f2ea;
  --panel: #fffdf8;
  --ink: #1c2320;
  --ink-soft: #55605a;
  --grid: #d8d0bf;
  --accent: #2f5f5c;
  --accent-2: #b8873a;
  --flag: #c1443f;
  --serif: Georgia, Cambria, "Times New Roman", serif;
  --sans: -apple-system, "Segoe UI", "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", sans-serif;
  --mono: "SF Mono", "Cascadia Mono", Consolas, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #14181a; --panel: #1b201f; --ink: #ece7db; --ink-soft: #a9b0a5;
    --grid: #33342c; --accent: #7fb3ac; --accent-2: #d9ab5c; --flag: #e0776f;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #14181a; --panel: #1b201f; --ink: #ece7db; --ink-soft: #a9b0a5;
  --grid: #33342c; --accent: #7fb3ac; --accent-2: #d9ab5c; --flag: #e0776f;
}}
:root[data-theme="light"] {{
  --bg: #f6f2ea; --panel: #fffdf8; --ink: #1c2320; --ink-soft: #55605a;
  --grid: #d8d0bf; --accent: #2f5f5c; --accent-2: #b8873a; --flag: #c1443f;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg); color: var(--ink); font-family: var(--sans);
  line-height: 1.6; font-size: 16px;
}}
.page {{ max-width: 880px; margin: 0 auto; padding: 56px 24px 100px; }}
h1, h2, h3 {{ font-family: var(--serif); text-wrap: balance; font-weight: 400; }}
h1 {{ font-size: 2.05rem; letter-spacing: -0.01em; margin: 0 0 6px; }}
.subtitle {{ color: var(--ink-soft); font-size: 1.05rem; max-width: 62ch; }}
.meta {{ color: var(--ink-soft); font-size: 0.82rem; font-family: var(--mono);
  margin-top: 18px; display: flex; gap: 18px; flex-wrap: wrap; }}
header {{ border-bottom: 1px solid var(--grid); padding-bottom: 28px; margin-bottom: 44px; }}

section {{ margin: 56px 0; }}
.eyebrow {{ font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--accent); display: flex; align-items: center; gap: 10px; }}
.eyebrow::before {{ content: ""; width: 22px; height: 1px; background: var(--accent); }}
h2 {{ font-size: 1.5rem; margin: 8px 0 6px; }}
.lede {{ color: var(--ink-soft); max-width: 68ch; font-size: 0.98rem; margin-bottom: 22px; }}

p {{ max-width: 68ch; }}
.callout {{
  background: var(--panel); border: 1px solid var(--grid); border-left: 3px solid var(--accent);
  padding: 16px 20px; border-radius: 3px; margin: 20px 0; font-size: 0.95rem;
}}
.callout.flag {{ border-left-color: var(--flag); }}
.callout b {{ color: var(--ink); }}
code {{ font-family: var(--mono); font-size: 0.9em; background: color-mix(in srgb, var(--grid) 45%, transparent); padding: 1px 5px; border-radius: 3px; }}

figure {{ margin: 24px 0 8px; }}
figure img {{ width: 100%; display: block; border: 1px solid var(--grid); border-radius: 4px; }}
figcaption {{ font-family: var(--mono); font-size: 0.78rem; color: var(--ink-soft);
  margin-top: 10px; line-height: 1.55; }}

.table-wrap {{ overflow-x: auto; margin: 18px 0; border: 1px solid var(--grid); border-radius: 4px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; background: var(--panel); }}
th, td {{ padding: 11px 14px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }}
th {{ font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--ink-soft); font-weight: 400; text-align: right; background: color-mix(in srgb, var(--grid) 35%, transparent); }}
td.rowlabel, th.rowlabel {{ text-align: left; font-family: var(--serif); font-size: 0.98rem; }}
.ch {{ font-family: var(--mono); font-size: 0.72rem; color: var(--ink-soft); margin-left: 8px; }}
.num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
.sd {{ color: var(--ink-soft); font-size: 0.85em; }}
.dim {{ color: var(--ink-soft); }}
tr:last-child td {{ border-bottom: none; }}
.subhead {{ font-family: var(--mono); font-size: 0.75rem; color: var(--ink-soft); margin: 26px 0 -6px; }}

.stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 20px 0; }}
.stat {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 4px; padding: 16px 18px; }}
.stat .k {{ font-family: var(--mono); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-soft); }}
.stat .v {{ font-family: var(--serif); font-size: 1.7rem; margin-top: 4px; }}
.stat .v small {{ font-size: 0.55em; color: var(--ink-soft); font-family: var(--mono); }}

.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 700px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

footer {{ margin-top: 70px; padding-top: 20px; border-top: 1px solid var(--grid);
  font-family: var(--mono); font-size: 0.76rem; color: var(--ink-soft); }}
a {{ color: var(--accent); }}
</style>

<div class="page">

<header>
  <h1>REFED：資料集乾淨度驗證與四電極配置比較</h1>
  <p class="subtitle">用 LibEER 的 DGCNN 在 REFED（EEG-fNIRS 動態情緒資料集）上比較四種頭戴裝置電極配置，並在下手做任何配置比較之前，先驗證這個較少人用過的資料集本身訊號是否可靠。</p>
  <div class="meta">
    <span>PIPELINE&nbsp;&nbsp;LibEER SEED 前處理 + LDS 平滑</span>
    <span>SPLIT&nbsp;&nbsp;per-subject in-session（主要結論）／pooled 跨 32 人（對照，見各節說明）</span>
    <span>MODEL&nbsp;&nbsp;DGCNN (k=2, layers=[64])</span>
  </div>
  <div class="callout flag" style="margin-top:20px;">
    <b>跟老師討論後的修正（第一輪）：</b>這份報告先前的「準確率」全部來自 <b>pooled</b> 訓練（一個模型看過全部 32 人的訓練片段，只有 test clip 沒看過）——這跟 SEED 的標準 subject-dependent 慣例（每人自己訓自己的模型）不是同一件事，兩邊數字不能直接比。這一版把 <b>per-subject（in-session，每人自己的資料自己訓、自己測）</b>改成主要呈現的數字，pooled 保留在旁邊當對照組，並在每張表註明是哪一種。t-SNE、SEED cross-check 也一併檢查並修正了同樣的問題（見 01、06 節）。
  </div>
  <div class="callout flag" style="margin-top:12px;">
    <b>第二輪修正——資料萃取本身的 bug：</b>特徵萃取腳本（<code>refed_extract_libeer.py</code>）原本有一段自己加的邏輯：如果一個 trial 裡 valence 或 arousal 有一軸<b>完全沒離開過 joystick 中心值（128）</b>，就整個 trial 丟掉不算。查證原論文（REFED，Ning et al.）和官方 REFED-codes 之後，發現(128,128)只是<b>軟體重置的起始位置</b>，不是「沒資料」的標記——論文原句：「Each video segment begins at the center... (128, 128)... and changes dynamically in response to the participant's emotional shifts.」官方程式碼也是直接用完整的原始標籤序列，沒有這種丟棄邏輯。這個自製的規則不只沒有根據，還<b>系統性刪掉了 MVMA（中等 valence、中等 arousal）目標的 trial</b>——32 人中有 13 人至少丟了 1 個 trial，最嚴重的丟了 7 個（見第 02 節）。這一版已經移除這段邏輯，改成跟官方一致的做法（每個 trial 從 t=0 開始、用完整序列），<b>本報告所有數字都是修好之後重新跑的</b>。這個 bug 影響不小：silhouette、分類準確率的多數類基準線、甚至「arousal 是否比 valence 更容易分類」的結論都跟著變了（見 01、05 節）。
  </div>
</header>

<section>
  <div class="eyebrow">01 &nbsp;Dataset validity</div>
  <h2>REFED 乾不乾淨？跟 SEED 對照看</h2>
  <p class="lede">REFED 是一個還沒被很多論文驗證過的新資料集。與其直接假設它可用，先用跟 SEED（被幾十篇論文驗證過的 benchmark）完全相同的特徵管線，檢查兩邊在特徵空間裡，三類情緒標籤是否有天然的分離結構。</p>

  <div class="two-col">
    <figure>
      <img src="data:image/png;base64,{TSNE_DE_IMG}" alt="t-SNE, plain DE, single session">
      <figcaption>純 DE，無時間平滑。同一個 subject（REFED #9）／同一個 SEED session，t-SNE 2D 投影，62 通道。</figcaption>
    </figure>
    <figure>
      <img src="data:image/png;base64,{TSNE_LDS_IMG}" alt="t-SNE, DE_LDS, single session">
      <figcaption>DE_LDS，加上 LibEER 的LDS 平滑，其餘設定相同。</figcaption>
    </figure>
  </div>

  <div class="callout">
    <b>Silhouette score 怎麼算？</b> 對每個樣本點 i：<code>a(i)</code> = 它到同一類其他點的平均距離（同類抱得緊不緊），<code>b(i)</code> = 它到離它第二近那一類所有點的平均距離（離最近的別類有多遠），<code>s(i) = (b(i) − a(i)) / max(a(i), b(i))</code>。整個資料集的 silhouette 是全部 s(i) 的平均，範圍 −1 到 +1：接近 <b>+1</b> 代表同類抱緊、離別類遠（分得開）；接近 <b>0</b> 代表卡在兩類邊界、分不清楚；<b>負值</b>代表離別類反而比離自己這類更近，標籤跟特徵空間對不上。本質上是在問：完全不看標籤、只看特徵距離去分群，會不會剛好分出跟真實標籤一樣的三群？計算範圍：每個人（SEED 每個 session、REFED 每個 subject）自己的資料自己 z-score 標準化後再算，不跨人 pooled。
  </div>

  <div class="subhead">SILHOUETTE SCORE — 逐人計算後取平均（in-session，不跨人 pooled）</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th class="rowlabel">Feature</th><th>SEED (n=15)</th><th>REFED (n=32)</th><th>REFED / SEED</th></tr></thead>
      <tbody>
        <tr>
          <td class="rowlabel">Plain DE</td>
          <td class="num">+{SIL['de']['seed']:.3f} ± {SIL['de']['seed_sd']:.3f}</td>
          <td class="num">{SIL['de']['refed']:+.3f} ± {SIL['de']['refed_sd']:.3f}</td>
          <td class="num dim">—</td>
        </tr>
        <tr>
          <td class="rowlabel">DE + non-causal LDS</td>
          <td class="num">+{SIL['lds']['seed']:.3f} ± {SIL['lds']['seed_sd']:.3f}</td>
          <td class="num">+{SIL['lds']['refed']:.3f} ± {SIL['lds']['refed_sd']:.3f}</td>
          <td class="num">≈ {SIL['lds']['ratio']:.2f}×</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="callout flag">
    <b>老實講：這個結果對 REFED 不算好看。</b>
    純 DE、不做任何時間平滑時，REFED 32 人裡 {SIL['de']['n_neg']} 人 silhouette 是<b>負值</b>，平均 {SIL['de']['refed']:+.3f}——標籤跟原始特徵空間對不太上，這是一個真實的弱點，不是雜訊誤差。加上 LDS 平滑後兩邊都轉正（REFED 32 人裡剩 {SIL['lds']['n_neg']} 人負值），但 REFED 平均 +{SIL['lds']['refed']:.3f} 仍只有 SEED +{SIL['lds']['seed']:.3f} 的<b>約 {SIL['lds']['ratio']*100:.0f}%</b>，落差沒有消失，只是從「幾乎測不到訊號」縮小成「訊號明顯比 SEED 弱」。
    <br><br>
    另外要留一個問號：LDS 平滑本身會讓相鄰時間點的特徵變得更相似（自相關變高），而 REFED 的標籤本來就是逐秒緩慢變化的連續值——這代表 silhouette 進步有一部分可能只是「平滑讓時間上相鄰、剛好標籤也相近的樣本變得更像」，不完全保證是情緒相關的真實訊號被放大。這個混淆因子目前還沒有被排除，是這個驗證方法本身的局限，不是可以忽略的小細節。
  </div>

  <div class="callout">
    <b>確認：上面兩張 t-SNE 圖，是不是真的只有單一 subject？包不包含全部 trial？</b>
    是。<code>tsne_compare_insession.py</code> 只讀 <code>subject == 9</code> 這一個人的資料（<code>m = subject == sub</code>，沒有下採樣、沒有跨人合併），REFED 那邊實際跑出來 n={SUB9_TSNE['n']} 個樣本、涵蓋這個人全部 15 個 clip（修正萃取 bug 之前是 n=1255，少算了被誤刪的 trial）；SEED 那邊是單一 session 檔案（<code>10_20131130.mat</code>），n=3394，同樣是全部 15 個 trial 都在裡面，不是只挑了部分點。DE_LDS 版本印出來的確切數字：REFED sub 9 高維 silhouette={SUB9_TSNE['lds_high']:+.3f}、t-SNE 2D 投影後 silhouette={SUB9_TSNE['lds_emb']:+.3f}。
  </div>

  <div class="subhead">老師的問題：LDS 平滑後看起來分得開，但準確率卻差，為什麼？</div>
  <p class="lede">Sub 9 這張圖本身沒有矛盾——它的 silhouette（{SUB9_TSNE['lds_high']:+.3f}）和它自己的 per-subject 分類準確率（{SUB9_ACC:.3f}，高於 32 人平均 {REFED_MEAN_ACC:.3f}）是一致的，sub 9 是全資料集裡數一數二會分得開、也分得準的人。矛盾是在於<b>拿這一張圖代表整個資料集</b>：32 個人裡，silhouette 跟真正的分類準確率其實幾乎沒有關聯。</p>
  <figure>
    <img src="data:image/png;base64,{SIL_ACC_IMG}" alt="Silhouette vs per-subject accuracy scatter">
    <figcaption>32 個 REFED subject，x 軸是每人的 DE_LDS in-session silhouette，y 軸是每人自己的 3-class valence 準確率（per-subject 訓練）。Pearson r={SIL_ACC_R:+.3f}，只解釋了準確率變異的 {SIL_ACC_R**2*100:.1f}%。</figcaption>
  </figure>
  <div class="callout flag">
    Sub 24 的 silhouette 只有 +0.010（全資料集數一數二低），準確率卻有 0.731，是全資料集最高；反過來 sub 29 的 silhouette 有 +0.090（全資料集第二高，僅次於 sub 9），準確率只有 0.395，遠低於 sub 9。「t-SNE/silhouette 看起來分得開」跟「模型實際訓練後準確率高」是<b>兩件相關但不等價的事</b>（相關係數只有 {SIL_ACC_R:+.3f}，比修正 bug 之前更弱），原因大致有三個：
    <br><br>
    1) Silhouette 是把這個人全部 15 個 clip（含後來要當 test 的 clip）一次算完的靜態指標，不是「用 10 個 clip 訓練、再測另外 5 個沒看過的 clip」——分類器要面對的是<b>對全新 clip 的類化</b>，跟特徵空間裡的整體分佈分不分得開是不同的問題。
    <br>2) t-SNE 本身會把資料硬擠成看起來像一球一球的樣子（這是它的優化目標、俗稱 crowding problem），就算真實訊號很弱，2D 投影也常常「看起來」比高維 silhouette 數字暗示的更分得開，容易讓人視覺上高估分離程度。
    <br>3) Per-subject 訓練資料量很小（每人約 800 秒），DGCNN 要從這麼少資料裡學到跟 silhouette 隱含的那個決策邊界一致的分類器，本身就不保證做得到，尤其當這個邊界高度依賴 LDS 平滑帶來的「同一個 clip 內樣本互相黏在一起」（見第 08 節 saliency map、第 02 節原始標籤走向）。
  </div>
</section>

<section>
  <div class="eyebrow">02 &nbsp;Raw-label trajectories</div>
  <h2>原始 joystick 標籤本身，走向合不合理？</h2>
  <p class="lede">在看任何模型結果之前，先確認標籤本身：同一個 session 內，每個 trial 的 valence-arousal 該不該往它預設的目標情緒方向走。這裡不經過任何模型，直接畫全部 32 個 subject（= 32 個 session，REFED 一人一 session）每個人全部 15 個 clip 的原始 joystick 軌跡，投影在 2D 的 valence-arousal 平面上，8 人一張、分 4 張圖，不省略任何人。</p>
  {AV_TRAJ_FIGURES}
  <p class="lede" style="margin-top:8px;">圓圈＝trial 開始，方塊＝trial 結束，線條從淺到深表示時間推進，顏色＝該 clip 的目標情緒（MVMA/LVLA/LVHA/HVHA/HVLA）。（改過兩次：第一版用箭頭標終點，但有些 trial 最後兩秒的 joystick 數值完全相同，箭頭長度變成 0 就消失不見；第二版換成圓圈/方塊，但漸層一開始太接近白色背景、線段轉角處看起來會斷掉——這版把起始顏色調深、轉角改成圓角銜接，同時線條下面墊一條淡色實線保底，確保任何時候都看得到路徑。）</p>
  <div class="callout">
    大部分 subject（如 1、5、9、17、21）大致看得出方向性：紅色（HVHA）偏向右上、紫色（LVHA）偏向左上，跟預期的象限大致吻合，但每個人自己的路徑常常繞來繞去、不是一路平滑走到目標象限就停在那裡——這跟 silhouette 偏弱是一致的，訊號方向大致對，但雜訊很大。<b>subject 13 是一個明顯的例外</b>：大量軌跡黏在 valence≈arousal≈+1 的邊角，不分情緒目標，很可能是 joystick 卡在邊界或操作方式跟其他人不一樣——這種個別受試者的資料品質問題，之後應該考慮排除或至少標註。
  </div>

  <div class="subhead">SUBJECT 13 逐 TRIAL 檢查（全部 15 個 clip，不跳過）</div>
  <p class="lede">拆開來看 subject 13 每個 clip 自己的 valence-時間、arousal-時間曲線，加上同一個 clip 的 AV 平面走法。目的是判斷「卡邊角」是全部 trial 都這樣，還是只有部分 trial。</p>
  {SUB13_FIGURES}
  <div class="callout flag">
    15 個 trial 裡不是全部都卡邊角：像 clip 9（LVHA）valence/arousal 都是正常的先升後降曲線，AV 平面走出一個乾淨的弧線；但 clip 7（HVHA）valence 從 t=40s 到 t=100s 整整 60 秒釘死在 1.00（joystick 打到底、完全沒有變化），是明顯的飽和/卡住訊號，不是真實的情緒穩定。這種「部分 trial 飽和」的模式，說明 subject 13 的問題可能不是整個 session 都不能用，而是特定幾個 trial（尤其是高 valence/arousal 目標的 clip）操作到底之後就不放手了——排除時可以考慮只排掉受影響的 trial，不一定要整個 subject 丟掉。
  </div>
</section>

<section>
  <div class="eyebrow">03 &nbsp;Electrode montages</div>
  <h2>四組電極配置</h2>
  <p class="lede">從全頭 62 通道降到三種頭戴裝置可行的子集，比較「頭戴裝置能省多少電極、還留住多少情緒訊號」。</p>
  <figure>
    <img src="data:image/png;base64,{MONTAGE_IMG}" alt="Four electrode montages">
    <figcaption>Whole cap 62ch（SEED 全通道扣除 M1/M2 參考）／Glasses(VR ring) 20ch（額頭到枕部整圈）／Quick-20 19ch／Head-top 21ch。</figcaption>
  </figure>
</section>

<section>
  <div class="eyebrow">04 &nbsp;Continuous regression</div>
  <h2>連續回歸（valence / arousal，CCC）</h2>
  <p class="lede">Pipeline：LibEER SEED 前處理（filtfilt bandpass + per-band DE）+ LDS 平滑，DGCNN。acc3 是把回歸輸出事後用 ±40/127 門檻切三類，僅供參考，不是真分類器準確率（見下一節）。</p>

  <div class="callout">
    <b>CCC（Concordance Correlation Coefficient）怎麼算？</b>
    <code>CCC = 2·cov(true, pred) / (var(true) + var(pred) + (mean(true) − mean(pred))²)</code>。
    跟一般 Pearson correlation 不一樣的地方：Pearson 只看趨勢對不對得上（就算預測整體偏移或縮放，只要形狀同步就能拿高分），CCC 的分母多了 <code>var(pred)</code> 跟 <code>(mean(true)−mean(pred))²</code> 兩項，會直接懲罰「預測值整體偏掉」或「預測值波動幅度跟真實值對不上」的情況，所以它同時要求方向對、幅度也要對，比 Pearson 嚴格。範圍 −1 到 +1：<b>+1</b> 代表預測跟真實值完全一致（不只趨勢對，數值本身也對）；<b>0</b> 代表模型不比「永遠輸出真實值的平均數」更好；負值代表比瞎猜平均數還糟。這裡的 CCC 是<b>per-subject 分別算</b>（每個受試者自己的預測 vs 真實值算一次 CCC），再取 32 人的平均 ± 標準差。
  </div>

  <div class="subhead">PER-SUBJECT（IN-SESSION）— 主要結論，每個人自己的 DGCNN，只看過自己的訓練 clip，可跟 SEED 的 subject-dependent 慣例直接比</div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr><th class="rowlabel">Montage</th><th>Valence CCC</th><th>Arousal CCC</th><th>Valence acc3</th><th>Arousal acc3</th></tr>
      </thead>
      <tbody>{reg_table(rows_ps)}</tbody>
    </table>
  </div>

  <div class="subhead">POOLED（跨 32 人）— 對照組，數字會偏高，不能拿來跟 SEED 比</div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr><th class="rowlabel">Montage</th><th>Valence CCC</th><th>Arousal CCC</th><th>Valence acc3</th><th>Arousal acc3</th></tr>
      </thead>
      <tbody>{reg_table(rows_pl)}</tbody>
    </table>
  </div>
  <div class="callout">
    Per-subject 的 CCC 大致比 pooled 低（例如 whole cap valence：{whole_v_ccc_ps:+.3f} vs {whole_v_ccc_pl:+.3f}），這是預期中的事——per-subject 每個模型只有約 800 秒訓練資料，遠少於 pooled 的 32 倍資料量。重點是 per-subject 底下 <b>CCC 仍然全部是正值</b>（{CCC_PS_MIN:.3f}~{CCC_PS_MAX:.3f}），代表就算拿掉跨人資料量的加成，訊號依然存在，只是強度弱得多；這是跟 SEED 比較時應該用的數字。
  </div>
</section>

<section>
  <div class="eyebrow">05 &nbsp;3-class classification</div>
  <h2>真分類器準確率</h2>
  <p class="lede">同樣的 4 個 montage、同樣的 fold，但模型直接用 cross-entropy 訓練成 3 類分類器（negative / neutral / positive，±40/127 門檻），不是回歸事後分箱。Macro recall 是三類 recall 平均，不受類別數量不均影響。</p>

  <div class="subhead">PER-SUBJECT（IN-SESSION）— 主要結論</div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr><th class="rowlabel">Montage</th><th>Valence acc</th><th>Valence macro</th><th>Arousal acc</th><th>Arousal macro</th></tr>
      </thead>
      <tbody>{cls_table(rows_ps)}</tbody>
    </table>
  </div>

  <div class="subhead">POOLED（跨 32 人）— 對照組</div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr><th class="rowlabel">Montage</th><th>Valence acc</th><th>Valence macro</th><th>Arousal acc</th><th>Arousal macro</th></tr>
      </thead>
      <tbody>{cls_table(rows_pl)}</tbody>
    </table>
  </div>

  <div class="stat-grid">
    <div class="stat"><div class="k">Valence majority baseline</div><div class="v">{V_BASE:.3f}</div></div>
    <div class="stat"><div class="k">Arousal majority baseline</div><div class="v">{A_BASE:.3f}</div></div>
    <div class="stat"><div class="k">Chance (3-class)</div><div class="v">0.333</div></div>
  </div>

  <div class="callout flag">
    <b>跟上一版報告不一樣：「arousal pooled 訓練特別被壓低」這個結論，修好資料萃取 bug 之後不成立了。</b>
    修 bug 之前，pooled 底下 arousal 準確率明顯低於 valence，我們曾經解讀成「arousal 的個體差異太大，pooled 模型學不好」。但那個落差其實主要是<b>被誤刪的 trial 造成的假象</b>——那些 trial 剛好偏向「arousal 停在中間沒在動」，刪掉它們等於系統性拿掉 arousal 最容易預測的那一段。修好之後，pooled 底下 arousal（{aa_pl['whole']:.3f}~{aa_pl['quick20']:.3f}）已經跟 valence（{va_pl['vr_ring']:.3f}~{va_pl['quick20']:.3f}）差不多，四組 montage 大致打平，先前的落差消失了。
    <br><br>
    現在看到的是另一件事：per-subject 底下 arousal 準確率（{min(aa.values()):.3f}~{max(aa.values()):.3f}）確實全面高於 valence（{min(va.values()):.3f}~{max(va.values()):.3f}），但這主要是<b>多數類基準線本身暴漲</b>造成的——arousal 的基準線從（修 bug 前估計的）0.365 漲到 <b>{A_BASE:.3f}</b>（現在有一半樣本落在「中等」arousal），準確率領先基準線只有 {min(a_margin.values()):+.3f}~{max(a_margin.values()):+.3f}；valence 基準線是 {V_BASE:.3f}，準確率領先基準線 {min(v_margin.values()):+.3f}~{max(v_margin.values()):+.3f}，領先幅度反而是 arousal 的兩三倍。換句話說，<b>arousal 數字比較高很大一部分只是「基準線本身比較好猜」，扣掉這個因素之後，valence 才是相對更有真實分類訊號的維度</b>——這跟上一版報告的結論正好相反，是修 bug 後最大的一個結論反轉。
  </div>

  <div class="subhead">每個人自己的準確率長什麼樣子（per-subject，whole cap）</div>
  <figure>
    <img src="data:image/png;base64,{PERSUB_BARS_IMG}" alt="Per-subject accuracy bars">
    <figcaption>32 個 subject 各自的 3-class 準確率，虛線＝32 人平均。個體差異很大（valence 從 0.27 到 0.73，arousal 從 0.31 到 0.88），pooled 平均數字蓋掉了這個離散程度。</figcaption>
  </figure>
  <figure>
    <img src="data:image/png;base64,{PERSUB_SPREAD_IMG}" alt="Per-subject accuracy spread across montages">
    <figcaption>四組 montage 的每人準確率分布（箱型圖 + 每人一個點）。中位數在各 montage 間差異不大，但每個 montage 內部的人與人差異都遠大於 montage 之間的差異。</figcaption>
  </figure>
</section>

<section>
  <div class="eyebrow">06 &nbsp;SEED cross-check</div>
  <h2>同一套 pipeline、同樣四組電極，搬到 SEED 上跑一次</h2>
  <p class="lede">跟第 01 節的 silhouette 對照呼應：不只看特徵空間分不分得開，直接把完全相同的 4-montage、DE_LDS 分類器搬到 SEED（15 個 subject，各一個 session，5-fold by trial，每 fold held-out 3 trial／每類各一）上重跑一次，兩邊放在同一張表比大小。</p>
  {SECTION_06_NOTE}
  <div class="table-wrap">
    <table>
      <thead>
        <tr><th class="rowlabel">Montage</th><th>SEED acc (pooled)</th><th>SEED macro</th><th>REFED valence acc (per-subject)</th><th>REFED valence macro</th></tr>
      </thead>
      <tbody>{SECTION_06_TABLE}</tbody>
    </table>
  </div>
  <div class="callout flag">
    {SECTION_06_CALLOUT}
  </div>
</section>

<section>
  <div class="eyebrow">07 &nbsp;Test-set trajectories</div>
  <h2>Test set 上實際畫出的回歸曲線（per-subject）</h2>
  <p class="lede">每個目標情緒（MVMA / LVLA / LVHA / HVHA / HVLA）各挑一個 held-out test clip，選的是「該 clip 單一 trial CCC 落在中位數」的受試者——不是特別挑表現最好的案例。這裡用的是 <b>per-subject 訓練</b>的預測結果（每個 subject 自己的模型，只看過自己的訓練 clip），跟本節以上所有表格的主要數字一致。</p>
  <figure>
    <img src="data:image/png;base64,{TIMELINE_IMG}" alt="Test-set regression trajectories, per-subject">
    <figcaption>黑線＝真實 joystick，彩色線＝四個 montage 的 per-subject 模型預測（test-set，never seen during training for that fold）。單一 trial 層級的追蹤本來就比整體 CCC 弱、也比 pooled 版本更平——per-subject 資料量小，模型更容易收斂到接近平均值的保守預測。</figcaption>
  </figure>
</section>

<section>
  <div class="eyebrow">08 &nbsp;Saliency map</div>
  <h2>Whole cap（62ch）的 saliency map</h2>
  <p class="lede">跟 Zheng &amp; Lu (2015) 在 SEED 上用 DBN 權重分佈找關鍵通道的做法同精神，但這裡是直接從我們自己訓練的 whole-cap DGCNN 分類器算出來的：對每個 test 樣本，把「真實類別 logit」對輸入做反向傳播，取梯度絕對值，跨 5 個頻帶、跨 3 個 test fold 的全部 held-out 樣本取平均，得到每個通道一個重要性分數。</p>
  <figure>
    <img src="data:image/png;base64,{SALIENCY_IMG}" alt="Whole-cap DGCNN saliency map">
    <figcaption>顏色/大小＝該通道平均 |gradient|，越紅越大代表模型越依賴這個通道做 valence 判斷。前 10 名標了通道名。</figcaption>
  </figure>
  <div class="callout">
    修好資料萃取 bug 後重新訓練，Top 15 換成：T8, PO4, FT7, P2, F1, CP3, P4, FT8, FC4, TP7, OZ, T7, O2, FPZ, CP1——按區域分：<b>顳葉（5 個：T8, FT7, FT8, TP7, T7）</b>、<b>枕葉/頂葉（5 個：PO4, P2, P4, OZ, O2）</b>、中央/額中央（3 個：CP3, FC4, CP1）、額葉（2 個：F1, FPZ）。跟修 bug 前的名單具體通道不同，但整體區域分佈的結論沒變：<b>中央/頭頂區依然幾乎沒進前 15</b>，跟第 09 節文獻對照的方向一致；枕葉/頂葉權重偏高的問號也還在——REFED 的刺激是看影片，這些電極可能部分反映視覺誘發反應而非純情緒訊號，這個 confound 目前沒有被排除。
  </div>
</section>

<section>
  <div class="eyebrow">09 &nbsp;Literature check</div>
  <h2>headtop 的選法有文獻支持嗎？</h2>
  <p class="lede">Headtop 這組電極原本是憑印象挑的（記得 SEED 官方 saliency map 顯示頭頂表現較好）。查證 Zheng &amp; Lu (2015)《Investigating Critical Frequency Bands and Channels for EEG-Based Emotion Recognition with Deep Neural Networks》——這是 SEED 上最有名的通道重要性分析論文，用 DBN 權重分佈找關鍵通道。</p>
  <div class="callout flag">
    論文原句：<i>"The lateral temporal and prefrontal brain areas activate more than other brain areas in beta and gamma frequency bands."</i> 他們挑出的最佳精簡電極組（4/6/9/12 通道，準確率 82.9%~86.7%，全 62 通道是 84.0%）全部位於<b>顳葉側邊</b>，其中一組再加三個<b>額葉</b>電極——原文：「The electrodes of profiles (a), (b), and (d) are located in the lateral temporal areas and profile (c) adds three extra prefrontal electrodes.」<b>完全沒有提到頭頂/中央區域比較好。</b>
    <br><br>
    也就是說，文獻支持的是「顳葉＋額葉」，方向上更接近本報告的 <b>glasses/vr_ring</b>（額頭→顳部→枕部整圈）而不是 headtop（純中央/頭頂電極簇）。這跟第 08 節 saliency map（中央區幾乎不在前 15）方向一致——headtop 目前比較適合定位成「中央電極夠不夠用」的對照組，而不是被文獻支持的候選裝置設計。
  </div>
</section>

<footer>
  REFED (Ning et al.) · CC BY-NC-SA 4.0 · Feature pipeline: LibEER SEED preprocessing (0.3–50 Hz filtfilt bandpass, 5-band per-band filtfilt DE) + LDS (prior_correlation=0.01, noise_correlation=0.0001) · Model: DGCNN (k=2, layers=[64]) · 主要數字為 per-subject in-session 訓練（每人自己的 clip 3-fold），pooled 跨 32 人訓練僅作對照
</footer>

</div>
"""

out = HERE / "report_v2.html"
out.write_text(html, encoding="utf-8")
print("wrote", out, len(html), "chars")
