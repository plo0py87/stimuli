# REFED EEG emotion recognition pipeline

Investigates the [REFED dataset](https://huggingface.co/datasets/REFED2025/REFED-dataset)
(Ning et al., NeurIPS 2025 Datasets & Benchmarks Track) as a candidate for
continuous, real-time emotion labeling — compared against
[SEED](http://bcmi.sjtu.edu.cn/~seed/) (Zheng & Lu), the established discrete-label
benchmark. This README is written so a fresh agent (or you, months from now)
can pick this up without re-deriving the whole investigation from scratch.

**Live dashboard**: [claude.ai artifact](https://claude.ai/code/artifact/60874cb9-afcb-4067-85dd-62442d351672)
(also self-contained locally as `dashboard.html`, ~11MB, open directly in a browser).

## Environment

Everything runs under the vendored LibEER venv:

```
"C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" <script>.py
```

Needs `torch`+CUDA, `scikit-learn`, `scipy`, `umap-learn`, `matplotlib`,
`Pillow`, `scipy.io` (for `.mat` loading). Every script's docstring states the
exact command to run it with. `emotion_model/`'s `DGCNN` model and
`data_utils.preprocess.lds`/`libeer_bandpass` come from the same vendored
LibEER copy the parent project uses — see `sys.path.insert(0, ...)` at the
top of each script for the exact import path.

## Data (not in this repo)

- **REFED**: `D:/EEG dataset/REFED` — 32 subjects, 15 video clips each, 64ch
  EEG @ 1000Hz, continuous per-second joystick valence/arousal labels
  (`data/<subject>/EEG_videos.mat`, `annotations/<subject>_label.mat`).
  32GB+, download separately from HuggingFace.
- **SEED**: `D:/EEG dataset/SEED/SEED/SEED_EEG/Preprocessed_EEG` — 15
  subjects, one session each, 62ch @ 200Hz, static per-trial 3-class label.

Both are gitignored (`refed/*.npz` too — see below). A fresh clone has none
of this; every script that touches raw data takes `--refed-root`/hardcodes
the `D:/EEG dataset/...` path — update it for a new machine.

## Pipeline

**1. Feature extraction** — two parallel feature types, both from the same
LibEER-matched bandpass+DE pipeline (`refed_extract_libeer.py`, mirrors
LibEER's own SEED preprocessing step for step so REFED and SEED are
comparable):

```
"...python.exe" refed_extract_libeer.py         # -> refed_features_libeer.npz (plain DE)
"...python.exe" add_lds.py                      # -> refed_features_libeer_lds.npz (+ non-causal LDS smoothing)
```

`add_lds.py`'s smoothing is **non-causal** (its `lds()` call uses the whole
trial's mean before the recursion even starts) — not realtime-legal, kept
only to match the SEED-benchmark convention. DE_LDS consistently beats plain
DE by 5-12 accuracy points throughout this project; if a realtime system is
ever built from this, expect to lose that gap.

**2. Per-subject training** (the only mode used — see "pooled mode" pitfall
below) — DGCNN regression + classification, SVM regression + classification,
each run twice (once per feature type):

```
"...python.exe" refed_montage_regress.py --modes per_subject   # DGCNN, continuous CCC
"...python.exe" refed_montage_classify.py                      # DGCNN, 3-class accuracy
"...python.exe" svm_persubj_montage.py --features refed_features_libeer_lds.npz --out svm_persubj_montage.json
"...python.exe" svm_persubj_montage.py --features refed_features_libeer.npz     --out svm_persubj_montage_de.json
```

4 electrode montages tested every time (`whole`/`vr_ring`/`quick20`/`headtop`
— defined in `refed_montage_regress.py`'s `MONTAGES`, simulating a full cap
vs. lighter wearables). Folds are **emotion-balanced by clip**, not random:
15 clips group into 5 target emotions (`TARGET_GROUPS`), each of 3 folds
takes one clip per group. `svm_persubj_montage.py` caps `SVC`/`SVR` at
`max_iter=20000` — see pitfall #2 below for why.

**3. SEED comparison, truncated for a fair fight**:

```
"...python.exe" seed_montage_classify_persubj_trunc.py                 # DE_LDS
"...python.exe" seed_montage_classify_persubj_trunc.py --feat de       # plain DE
```

Truncates every SEED trial to REFED's mean length (~102s vs SEED's native
~226s) before training, so any REFED-vs-SEED accuracy gap can't just be "SEED
has more data per trial."

**4. Visualization + dashboard**:

```
"...python.exe" tsne_all_subjects.py            # per-subject + pooled t-SNE, both features
"...python.exe" umap_all_subjects.py            # same, UMAP
"...python.exe" dash_regression_individual.py --pred libeer_lds_persubj_pred.npz --suffix lds
"...python.exe" dash_regression_individual.py --pred libeer_de_persubj_pred.npz  --suffix de
"...python.exe" build_dashboard.py              # -> dashboard.html (self-contained, ~11MB)
```

`build_dashboard.py` reads every JSON/npz result file above and inlines every
image as base64 into one HTML file — no server, no external files needed to
view it. Rebuilding after regenerating any upstream result just means
rerunning this last step.

## Key findings so far

- **Electrode count barely matters.** Whole cap (62ch) vs. the three lighter
  montages: all within ~2 accuracy points of each other. Good news for a
  wearable-device pitch — you don't pay much for fewer electrodes.
- **DE_LDS beats plain DE substantially** (see above) — but it's not
  realtime-legal as implemented.
- **DGCNN ≈ SVM.** With only ~8 training clips per subject per fold, the
  extra model complexity doesn't pay for itself; SVM is a legitimate
  lightweight alternative.
- **SEED still clearly outperforms REFED even after the length-truncation
  fix** (e.g. whole-cap valence: SEED ~0.71 vs REFED ~0.50 at DE_LDS). This
  is *not* explained by trial length. Two confounded differences remain
  between the datasets — REFED's labels are continuous-in-time (per-second)
  **and** continuous-in-value (dimensional AV self-report) where SEED's are
  static-per-trial **and** discrete (curator-assigned category) — and no
  public dataset exists that varies only one of those two axes, so this
  project could not cleanly attribute the gap to either one. See the
  dashboard's SEED-comparison tab and this repo's Notion report ("REFED 資料集驗證")
  section 01 for the full writeup.
- **Regression (continuous CCC) is weak and highly subject-dependent**
  (mean valence CCC ≈0.19, range −0.26 to +0.54 across 32 subjects) — the
  3-class accuracy numbers are the more usable result of this dataset.

## Known pitfalls (read before repeating them)

1. **"Pooled" training mode leaks the target subject's own data.**
   `refed_montage_regress.py`'s `run_pooled()` trains on every subject's
   clips *including the target subject's own other clips* — it answers "does
   more data help this person's held-out clip," not "does this generalize to
   a new person." It looked like a genuine subject-independent number until
   traced through the fold logic. Dropped from the dashboard entirely; if you
   resurrect it, label it very clearly as within-subject, not
   cross-subject.
2. **Unbounded SVM `max_iter` can hang for hours.** Plain DE (noisier, no
   smoothing) made some (kernel, C) grid combination in
   `svm_persubj_montage.py` fail to converge — one run sat for 9+ hours of
   real CPU time on a single fold before being caught. Fixed by capping
   `max_iter=20000`; if you see a training run stall with no new log output,
   check `wmic process where "ProcessId=<pid>" get KernelModeTime` for
   accumulating CPU time before assuming it's just slow.
3. **The venv's `python.exe` is a stub that spawns the real interpreter as a
   child process.** `tasklist`/`wmic` will show two `python.exe` entries
   (one from `.venv39/Scripts/python.exe`, one from the base install) for
   what is actually a single run — don't kill both thinking it's a
   duplicate launch; check `wmic process ... get CommandLine` first.
4. **The REFED paper's own text and its own released code disagree.**
   Section 4.2 (p.8) and Appendix E's Eq. 1 describe a 3-class threshold of
   [0.4, 0.6] and a closed-form Gaussian-entropy DE formula — reproducing
   *those* gets nowhere near the paper's reported Table 6 SVM numbers
   (valence 0.47 vs. paper's 0.524, arousal 0.43 vs. 0.621). The actual
   released code ([github.com/REFED-dataset/REFED-codes](https://github.com/REFED-dataset/REFED-codes),
   `DE_PSD.py` + `label_to_3c()` in `utils_data.py`) uses an STFT+Hanning-window
   PSD-based DE formula and **[0.3, 0.7]** thresholds instead. Porting the
   actual code (`svm_reproduce.py --exact`) reproduces the paper almost
   exactly (valence 0.523 vs 0.524, arousal 0.609 vs 0.621). An ablation
   (`--exact-de` / `--exact-thresh` in isolation) showed the threshold width
   accounts for nearly all of the gap — the DE formula choice barely matters
   for this particular accuracy number. **Moral: when trying to match a
   paper's numbers, check their released code before trusting the prose.**
   `svm_reproduce.py` keeps both the "paper text" and "paper code" versions
   of each function side by side, with the mismatch documented inline.
5. **PNG antialiasing fights palette quantization.** The dashboard embeds
   ~1100 small chart images; naively cutting `dpi` to hit a size budget just
   makes them blurry. What actually works: turn `lines.antialiased`/
   `patch.antialiased` off in matplotlib (so each image only ever contains a
   handful of true colors — background, ink, gridline, the few series
   colors) *then* quantize the palette — near-lossless at 4-8x smaller,
   because there's no antialiasing gradient left to lose. See
   `dash_regression_individual.py` and `umap_all_subjects.py`'s `_quantize()`.
6. **The claude.ai Artifact size cap is 16MB**, and base64 inflates raw
   bytes by ~4/3. Budget raw image bytes accordingly before generating a
   large batch (960+ images at moderate quality blew past this twice during
   this project before the antialiasing fix above).

## What's excluded from this repo (gitignored)

`refed/*.npz` (~250MB of extracted-feature and prediction caches —
regenerate via the extraction/training commands above), `refed/*.log` (run
output, not needed to reproduce results), `refed/dash_*.png` (~1100 files,
already embedded as base64 in `dashboard.html`). The curated `fig_*.png`
report figures and the HTML deliverables (`dashboard.html`, `report.html`,
`report_v2.html`) are kept as-is.
