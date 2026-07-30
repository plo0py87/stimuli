# CGX Quick-20 EEG emotion recognition pipeline

Real-time and offline emotion classification (negative/neutral/positive) from
a CGX Quick-20 EEG headset, trained on your own labeled recordings rather
than a generic benchmark. This README is written so a fresh agent (or you,
months from now) can pick this up on a new machine with a new dataset
without re-deriving the whole pipeline from scratch.

## Environments

Two conda environments are used, split by what needs GPU/heavy deps:

- **base miniconda** (`python.exe`) — lightweight scripts with no torch
  dependency: `split_session_trials.py`, `eval_own_data_accuracy.py`,
  `compare_days.py` (analysis only), plain data-wrangling.
- **`EEG` conda env** (`python.exe` inside the env) — everything that needs
  `torch` (+ CUDA), `mne`, `scipy`, `pandas`, `mss`, `pylsl`,
  `imageio-ffmpeg`, `scikit-learn`. This is basically everything that trains
  or runs a model, plus anything that captures screen/EEG live
  (`annotate_session.py`, `record_session.py`, `realtime_predict.py`).

When in doubt, check a script's own docstring — every script's header
comment states the exact env/command to run it with.

`emotion_model/` is a vendored working copy of the [LibEER](../LibEER)
package (models, Trainer, data_utils, utils, config). Every script that
touches a model does `sys.path.insert(0, ".../emotion_model")` before
importing from it. If LibEER upstream changes, `emotion_model/` needs to be
manually re-synced (it is not a symlink or submodule).

## Directory layout

```
stimuli/
├── sessions/              # recorded data (gitignored -- local only, can be GBs)
│   └── <subject>_<timestamp>/
│       ├── eeg_raw.npz        # data (n_channels, n_samples), timestamps, srate, labels
│       ├── markers.csv        # elapsed_s, wall_time, lsl_time, event
│       ├── ratings.csv        # elapsed_s, wall_time, lsl_time, trial, category, confidence
│       ├── screen_recording.mp4  (or screen_recording_h264.mp4)
│       └── trials/            # only if split_session_trials.py has been run
│           └── trial_XX/      # same file shapes as above, sliced to one trial
├── result_dgcnn_own_data_split_session/   # BEST model checkpoint (see below)
├── result_dagcn_own_data_split_session/   # comparison checkpoint
├── result_tsception_own_data_split_session/  # comparison checkpoint
├── emotion_model/          # vendored LibEER copy (models/, Trainer/, utils/, config/)
└── *.py                    # pipeline scripts, see below
```

`sessions/` and `SEED/` are in `.gitignore` -- a fresh clone of this repo
will NOT have any recorded data. You need your own `sessions/` folder with
your own recordings before anything in the pipeline is useful.

## Recording a new session

Two recorder scripts exist, pick based on protocol:

- **`annotate_session.py`** (recommended, richer protocol): Pre-rest phase
  (10s buffer + 90s eyes-open baseline) → video task with a popup every
  `--prompt-interval` seconds (default 20s) asking you to rate how you feel
  right now (5-class: 正向/偏正/中性/偏負/負向), **then** a second popup asks
  you to rate your confidence in that answer (1-5, asked *before* you ever
  see any model prediction -- this exists specifically so you can filter
  low-confidence labels later without post-hoc bias/data-snooping, see
  "Known pitfalls" below) → optional Post-rest. Can hold multiple
  `trial_start:N`/`trial_end:N` segments in one continuous recording (e.g.
  multiple video clips back to back) -- use `split_session_trials.py`
  afterward to cut these into separate per-trial folders.

  ```
  & "EEG_python.exe" annotate_session.py --subject Shine
  ```

- **`record_session.py`** (simpler, one trial per run): screen + EEG
  recording for a single video, press SPACE to mark strong-emotion moments,
  Q to stop. No rating popups -- label the whole recording after the fact
  or use it for unlabeled exploration.

Both write into `sessions/<subject>_<timestamp>/` and use
**`video_writer.py`** (`FFmpegVideoWriter`) for screen capture -- pipes raw
BGR frames into ffmpeg (bundled via `imageio-ffmpeg`), encoding H.264
directly (browser-playable, no separate transcode step needed). Frames are
timestamped by real wall-clock capture time (`-use_wallclock_as_timestamps
1` + `-fps_mode vfr`), NOT by assuming a fixed nominal fps -- this matters:
an earlier version assumed constant fps and silently lost up to ~55% of a
long recording's video when actual capture rate fell behind (EEG was
unaffected, only video). If you ever see a session where the video is much
shorter than the EEG/markers duration again, check
`<video_path>.ffmpeg.log` (ffmpeg's stderr is captured there, not
discarded) for what went wrong.

If a session has multiple `trial_start:N`/`trial_end:N` segments (the
`annotate_session.py` multi-trial format), split them first:

```
python split_session_trials.py --session-dir sessions/<subject>_<ts>
```

This writes `sessions/<subject>_<ts>/trials/trial_01/`, `trial_02/`, ...
(handling restarted trials as `trial_04a`/`trial_04b` etc.), each with its
own sliced `eeg_raw.npz`, trial-relative `ratings.csv` (adds a `t_s`
column, trial-local elapsed seconds), and a trimmed+transcoded
`screen_recording_h264.mp4`.

## Channel mapping and label convention

The CGX Quick-20 device channel names differ from the model's expected
order. `DEVICE_TO_MODEL_LABEL` (defined in `train_dagcn_own_data.py`, reused
everywhere) maps device names (`Fp1`, `T3`, `T4`, `T5`, `T6`, ...) to model
names (`FP1`, `T7`, `T8`, `P7`, `P8`, ...) -- the device's 20th channel
(`A2`, reference) plus `ACC_X/Y/Z`, `PacketCounter`, `TRIGGER` are not fed
to the model. Model channel order (19 channels):

```
FP1 FP2 F7 F3 FZ F4 F8 T7 C3 CZ C4 T8 P7 P3 PZ P4 P8 O1 O2
```

Labels: the self-report categories are 5-class Chinese
(負向/偏負/中性/偏正/正向), collapsed to 3-class for the model via
`LABEL_COLLAPSE` (負向+偏負→negative, 中性→neutral, 偏正+正向→positive).
If your new dataset has genuinely continuous (not discrete-prompt)
annotation, you'll need a different windowing function than
`load_session_windows()` -- that function currently assumes each rating
answered at time T labels the window `[previous prompt's time, T)`. A
continuous annotation stream would instead give you a label (or a
continuously-varying score) at every timestamp directly; adapt
`build_dataset()` in `train_dgcnn_own_data.py` accordingly rather than
reusing the windowing logic as-is.

## Feature extraction (causal, online-safe)

Every script uses the same two feature extractors from
`emotion_model/DGCNN_quick20_realtime_train.py` /
`DAGCN_quick20_realtime_train.py`:

- `causal_de_features(raw19, srate, channel_indices)` -- 5-band
  (delta/theta/alpha/beta/gamma, `[(1,4),(4,8),(8,14),(14,31),(31,50)]`)
  differential entropy via causal (forward-only, `lfilter`) Butterworth
  bandpass + variance over non-overlapping 1s windows. **Verified**: the
  filter transient settles within ~3-5s of context (tested against a
  continuous full-trial filter pass across 3 independent trials); a 10s
  lookback window (used everywhere) is already comfortably sufficient --
  don't bother extending it further, it doesn't meaningfully help.
- `causal_psd_features(raw19, srate, channel_indices)` -- same 5 bands via
  per-second Welch periodogram, `10*log10(mean psd in band)`. This one is
  **memoryless** by construction (each second's value only depends on that
  second's raw samples) -- no settle-time concern at all.

Both are concatenated to a `(19, 10)` per-second feature vector, optionally
smoothed with a forward-only scalar Kalman filter
(`kalman_smooth_trial(trial_list, q, r)`, defaults `q=0.01, r=0.5` --
verified identical across training and every inference script). This is
NOT the same as LibEER's own official `_lds`-suffixed features, which look
ahead into future data and are not safe for real-time use.

## Normalization: two options, neither is a clear winner yet

- **Global (train-pool) normalization**: mean/std computed once over the
  whole training pool, saved in the checkpoint, reused at inference. This
  is what the checkpoints' `mean`/`std` keys contain.
- **Session-baseline normalization**: mean/std computed from *that
  session's own* resting baseline (eyes-open, 90s) instead. Implemented in
  `predict_dgcnn_own_sessionnorm.py` (offline) and built into
  `realtime_predict.py` (live).

**Honest status** (see `eval_own_data_accuracy.py` results across 24
trials/4 sessions of real labeled data): session-baseline normalization
scored higher on pooled accuracy (47.9% vs 39.8%) but macro-averaged across
classes the gap nearly closes (47.5% vs 45.7%) -- it mostly trades "can't
detect neutral at all" for "can detect negative," not a clean win. On a
subset of sessions it looked much better (56-60% vs ~28-32%), but chasing
that number further by excluding "bad" trials turned out to be classic
post-hoc data-snooping (the exclusions were decided *after* seeing which
trials the model got wrong) -- don't trust that inflated number. If you
want to test exclusion-based cleanup on the new dataset, decide which
trials/sessions to trust **before** running any inference, using the
`confidence` column in `ratings.csv` (rated at recording time, blind to any
model output) as the filter criterion, not vibes after the fact.

## Training

```
& "EEG_python.exe" train_dgcnn_own_data.py --split-by session
& "EEG_python.exe" train_dagcn_own_data.py --split-by session
& "EEG_python.exe" train_tsception_own_data.py --split-by session
```

All three read every `sessions/<subject>_<date>_*` folder with both
`eeg_raw.npz` and `ratings.csv`, build per-second labeled samples, and
split by **whole session** (not by individual window) so held-out test data
is never seen in any form during training -- `--split-by window` also
exists but leaks (same-session windows can land on both sides) and gives an
optimistic, dishonest accuracy number. The session split does a balanced
search over 2000 random shuffles to keep per-class test-window counts even
(sessions aren't individually class-balanced).

**Results so far** (4 sessions / 24 trials, `Shine_2026072[89]_*`, session
split): DGCNN 83.8% > DAGCN 76.3% > TSception 64.4% overall. DGCNN is also
the most class-balanced (86.3/74.5/90.7% per class vs DAGCN's
85.7/52.2/91.3%). **DGCNN is the current best model** --
`result_dgcnn_own_data_split_session/checkpoint-scratch` is the one to use
by default.

A 4th architecture, "STGCLSTM," was requested once but doesn't exist
anywhere in LibEER (checked both the local `models/` and upstream) -- if
asked for it again, it's not a typo for `DGCNN_LSTM.py` unless confirmed;
ask before assuming.

## Inference + visualization

```
& "EEG_python.exe" predict_dgcnn_own.py --session-dir sessions/<s>/trials/trial_01 \
    --checkpoint result_dgcnn_own_data_split_session/checkpoint-scratch
& "EEG_python.exe" predict_dgcnn_own_sessionnorm.py --session-dir sessions/<s> \
    --checkpoint result_dgcnn_own_data_split_session/checkpoint-scratch
python make_emotion_player.py --session-dir sessions/<s>/trials/trial_01 \
    --csv-name dgcnn_own_predictions.csv --video-name screen_recording_h264.mp4
```

`make_emotion_player.py` builds a self-contained HTML player: video +
scrolling raw-EEG waveform (0.5-45Hz bandpassed, decimated to 100Hz) +
prediction timeline + a **ground-truth strip** showing your actual
self-reported label per second (red/yellow/green, gray = unrated) with the
same playback-position marker, so predicted vs actual can be compared
visually. Works without a video too (`--no-video`, scrubber UI instead).

`process_baseline.py` runs the same model against just the resting-baseline
segment of a session (sanity check -- should predict close to neutral
throughout, since nothing emotionally relevant happens there).

`eval_own_data_accuracy.py` computes per-second accuracy against
`ratings.csv` ground truth for a set of trials, comparing global vs
session-baseline normalization.

Other visualization/analysis tools kept from earlier investigation:
`eeg_paged_viewer.py` / `eeg_dual_viewer.py` (interactive raw-waveform
viewer with band filtering and PSD view), `saliency_map.py` (per-class
mean |gradient| topomap for a checkpoint), `compare_days.py` (per-band DE +
artifact-rate comparison across recording days), `faa_score.py` /
`make_faa_player.py` (an alternative non-ML frontal-alpha-asymmetry score,
explored but not adopted as the primary approach).

## Real-time prediction

```
& "EEG_python.exe" realtime_predict.py --subject Shine [--blind]
```

Connects to the CGX LSL stream, records a 100s baseline (10s settle + 90s
eyes-open, same phase structure as `annotate_session.py`'s Pre phase) to
compute session-baseline normalization stats live, then predicts once per
second using a 10s rolling lookback window + the persistent (carried across
ticks, not reset) Kalman smoother, displaying live scores. **Also records
a full session** (screen + EEG + markers + a `dgcnn_own_predictions.csv` in
the same format the offline scripts produce) into
`sessions/<subject>_realtime_<timestamp>/`, so you can run
`make_emotion_player.py` on it afterward exactly like a normal recording.

`--blind` hides all live prediction feedback (label, score bars, history
graph) -- everything still gets recorded, you just can't see it during
capture. Exists because watching the live score is a plausible source of
bias (you unconsciously react to what the model displays).

## Known pitfalls (read before repeating them)

1. **Data snooping when excluding "unreliable" trials.** Deciding a trial's
   label is untrustworthy *after* seeing that the model disagrees with it
   is not a valid exclusion criterion -- it mechanically inflates apparent
   accuracy in whatever direction you're already hoping to see. Use the
   `confidence` rating (collected blind, at recording time) instead.
2. **Window-split vs session-split.** Always use `--split-by session` for
   any accuracy number you plan to trust. Window-split leaks and gives
   inflated numbers (verified: 83.1% window-split vs 76.3% session-split
   for the same DAGCN checkpoint).
3. **Rolling-window feature computation is fine at 10s lookback --
   verified**, don't assume it needs to be longer without re-checking; an
   earlier ~15% discrepancy claim against continuous full-trial filtering
   turned out to be an alignment bug in the verification script itself, not
   a real gap.
4. **Baseline-normalization computation must match between live and
   offline** -- the causal filters + Kalman filter are not memoryless, so
   *where* in time you start them matters. `predict_dgcnn_own_sessionnorm.py`
   starts fresh exactly at the eyes-open baseline onset (excluding the 10s
   settle/buffer phase from the filter input entirely); `realtime_predict.py`
   was fixed to match this exactly (previously it ran filters continuously
   from the buffer onset instead, producing systematically different
   mean/std, ~5-15% median relative error).
5. **Video duration can silently be shorter than the real recording** if
   using an old/unfixed `video_writer.py` -- always sanity-check
   `ffprobe`-reported video duration against the session's
   `markers.csv`-derived true duration before trusting a video for
   anything. The current `video_writer.py` fixes this at the source
   (real per-frame timestamps), so newly recorded sessions shouldn't hit
   this, but anything recorded before the fix might.
6. **`annotate_session.py --no-eeg` used to default to True** (a past
   incident wiped an entire day's recordings of actual EEG data, video+
   ratings only) -- it now defaults to recording EEG; don't flip that
   default back without a very good reason.
