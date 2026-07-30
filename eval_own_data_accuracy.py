"""Compute per-second accuracy of the DGCNN own-data checkpoint's predictions
against your actual self-reported labels (ratings.csv), for the 24 newly
split trials (0728/0729 sessions) -- comparing:
  1. global normalization  (dgcnn_own_predictions.csv, train-pool mean/std)
  2. in-session baseline normalization (dgcnn_own_predictions_sessionnorm.csv,
     this session's own Pre-phase resting mean/std)

Each rated window [prev_prompt_t_s, this_prompt_t_s) is labeled with the
collapsed 3-class category from the prompt answered at its end (same
convention as train_dagcn_own_data.py's load_session_windows), and every
per-second prediction inside that window is checked against it.

Usage:
  "C:/Users/USER/miniconda3/envs/EEG/python.exe" eval_own_data_accuracy.py
"""

import csv
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
SESSIONS_DIR = HERE / "sessions"

LABEL_COLLAPSE = {
    "負向": "negative", "偏負": "negative",
    "中性": "neutral",
    "偏正": "positive", "正向": "positive",
}

SESSIONS = [
    "Shine_20260728_143129",
    "Shine_20260729_142432",
    "Shine_20260729_150939",
    "Shine_20260728_141303",
]


def load_windows(ratings_path):
    with open(ratings_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    windows = []
    prev_t = 0.0
    for r in rows:
        t = float(r["t_s"])
        cat = r["category"].strip()
        if cat not in LABEL_COLLAPSE:
            prev_t = t
            continue
        windows.append((prev_t, t, LABEL_COLLAPSE[cat]))
        prev_t = t
    return windows


def load_predictions(csv_path):
    if not csv_path.exists():
        return None
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {int(float(r["t_s"])): r["pred_label"] for r in rows}


def eval_trial(windows, preds):
    correct = Counter()
    total = Counter()
    for start_s, end_s, label in windows:
        s0, s1 = int(round(start_s)), int(round(end_s))
        for t in range(s0, s1):
            if t not in preds:
                continue
            total[label] += 1
            if preds[t] == label:
                correct[label] += 1
    return correct, total


def main():
    grand_correct = {"global": Counter(), "sessionnorm": Counter()}
    grand_total = {"global": Counter(), "sessionnorm": Counter()}

    for session_name in SESSIONS:
        trials_dir = SESSIONS_DIR / session_name / "trials"
        if not trials_dir.exists():
            continue
        print(f"\n=== {session_name} ===")
        for trial_dir in sorted(trials_dir.glob("trial_*")):
            ratings_path = trial_dir / "ratings.csv"
            if not ratings_path.exists():
                continue
            windows = load_windows(ratings_path)
            if not windows:
                continue

            global_preds = load_predictions(trial_dir / "dgcnn_own_predictions.csv")
            session_preds = load_predictions(trial_dir / "dgcnn_own_predictions_sessionnorm.csv")

            line = f"  {trial_dir.name}:"
            for name, preds in [("global", global_preds), ("sessionnorm", session_preds)]:
                if preds is None:
                    line += f"  {name}=N/A"
                    continue
                correct, total = eval_trial(windows, preds)
                grand_correct[name].update(correct)
                grand_total[name].update(total)
                n_correct, n_total = sum(correct.values()), sum(total.values())
                acc = n_correct / n_total * 100 if n_total else float("nan")
                line += f"  {name}={acc:.1f}% ({n_correct}/{n_total})"
            print(line)

    print("\n=== OVERALL (all 24 trials pooled) ===")
    for name in ["global", "sessionnorm"]:
        n_correct, n_total = sum(grand_correct[name].values()), sum(grand_total[name].values())
        acc = n_correct / n_total * 100 if n_total else float("nan")
        print(f"{name}: {acc:.1f}% ({n_correct}/{n_total})")
        for cls in ["negative", "neutral", "positive"]:
            c, t = grand_correct[name][cls], grand_total[name][cls]
            print(f"  {cls}: {c}/{t} ({c/t*100:.1f}%)" if t else f"  {cls}: no samples")


if __name__ == "__main__":
    main()
