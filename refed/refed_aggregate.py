"""Aggregate REFED normalization-comparison runs across seeds.

Each run is subject-disjoint 4-fold, so every subject is tested exactly once
per seed. Averaging a subject's score over seeds gives one number per subject
per condition, which then pair up for a Wilcoxon signed-rank test across the
32 subjects -- far more statistical power than comparing 4 fold means.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" refed_aggregate.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
CONDITIONS = ["global", "baseline", "session"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="refed_results_seed*.json")
    args = ap.parse_args()

    paths = sorted(HERE.glob(args.glob))
    if not paths:
        raise SystemExit(f"no result files matching {args.glob}")
    runs = [json.loads(p.read_text()) for p in paths]
    print(f"Aggregating {len(runs)} runs: {[p.name for p in paths]}\n")

    conds = [c for c in CONDITIONS if c in runs[0]["results"]]

    # ---- fold-level metrics pooled over seeds ----
    print("=" * 74)
    print(f"{'condition':10s} {'accuracy':>22s} {'macro (mean recall)':>24s}")
    print("=" * 74)
    for cond in conds:
        accs = np.array([r["acc"] for run in runs for r in run["results"][cond]])
        macros = np.array([r["macro"] for run in runs for r in run["results"][cond]])
        note = "" if cond != "session" else "  <- NOT realtime-legal"
        print(f"{cond:10s} {accs.mean():.4f} +/- {accs.std():.4f} (n={len(accs):2d})   "
              f"{macros.mean():.4f} +/- {macros.std():.4f} (n={len(macros):2d}){note}")

    # ---- per-subject, averaged over seeds, paired across conditions ----
    subjects = sorted(int(s) for s in runs[0]["per_subject"][conds[0]])
    per_subj = {c: {m: np.array([
        np.mean([run["per_subject"][c][str(s)][m] for run in runs]) for s in subjects
    ]) for m in ("acc", "macro")} for c in conds}

    for metric in ("acc", "macro"):
        print(f"\nPaired per-subject {metric} vs `global` "
              f"(n={len(subjects)} subjects, averaged over {len(runs)} seeds):")
        base = per_subj["global"][metric]
        for cond in conds:
            if cond == "global":
                continue
            other = per_subj[cond][metric]
            diff = other - base
            _, p = wilcoxon(other, base)
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {cond:9s} {base.mean():.4f} -> {other.mean():.4f}  "
                  f"diff {diff.mean():+.4f}  "
                  f"better on {int((diff > 0).sum())}/{len(diff)}  p={p:.4g} {stars}")

    # ---- per-class recall breakdown, pooled ----
    print("\nPer-class recall (pooled over folds and seeds):")
    print(f"  {'condition':10s} {'negative':>10s} {'neutral':>10s} {'positive':>10s}")
    for cond in conds:
        rows = [r["per_class"] for run in runs for r in run["results"][cond]]
        vals = {c: np.mean([row[c] for row in rows])
                for c in ("negative", "neutral", "positive")}
        print(f"  {cond:10s} {vals['negative']:10.3f} {vals['neutral']:10.3f} "
              f"{vals['positive']:10.3f}")

    # ---- how early the best epoch lands (overfitting sanity check) ----
    print("\nBest epoch chosen on validation (low = overfits to train subjects fast):")
    for cond in conds:
        eps = [r["best_epoch"] for run in runs for r in run["results"][cond]]
        print(f"  {cond:10s} median={int(np.median(eps)):3d}  "
              f"min={min(eps):3d} max={max(eps):3d}")


if __name__ == "__main__":
    main()
