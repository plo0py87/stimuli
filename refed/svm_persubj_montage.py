"""Per-subject (in-session) SVM classification AND regression, 4 montages,
DE_LDS -- the SVM counterpart to refed_montage_regress.py /
refed_montage_classify.py's per_subject DGCNN runs, so SVM and DGCNN can be
compared on identical data/folds/thresholds for the per-subject dashboard.

Unlike svm_reproduce.py (which replicates the ORIGINAL paper's setup: 64ch
only, paper's own band edges/asymmetric 3-class thresholds), this uses the
SAME feature file, montages, folds, and +/-40/127 threshold as the DGCNN
in-session runs, so SVM vs DGCNN is apples-to-apples.

SVC handles both dims' classes jointly (fit separately per dim, since
sklearn's SVC is single-output); SVR is likewise fit separately per dim for
regression (CCC).

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" svm_persubj_montage.py
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" svm_persubj_montage.py \
      --features refed_features_libeer.npz --out svm_persubj_montage_de.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_regress import (  # noqa: E402
    MONTAGES, EXCLUDE_FROM_WHOLE, FOLDS, DIMS, to_class, ccc, pcc, EPS,
)

ORDER = ["whole", "vr_ring", "quick20", "headtop"]
GRID = {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10]}


def macro_recall(y, p):
    rec = [(p[y == c] == c).mean() for c in range(3) if (y == c).any()]
    return float(np.mean(rec)) if rec else float("nan")


def norm(Xtr, X):
    mu, sd = Xtr.reshape(len(Xtr), -1).mean(axis=0), Xtr.reshape(len(Xtr), -1).std(axis=0) + EPS
    return (X.reshape(len(X), -1) - mu) / sd


MAX_ITER = 20000  # sklearn's SVC/SVR default to unlimited (-1) iterations;
# on noisy/poorly-separable data (plain DE has no temporal smoothing, so it's
# much noisier than DE_LDS) some (kernel, C) combos in the grid can fail to
# converge and spin indefinitely -- observed one run stall for 9+ hours of
# CPU time on a single fold. Capping bounds worst-case fit time; a fold that
# hits the cap just returns its best-effort (possibly non-converged) fit
# instead of hanging forever, at the cost of occasionally-suboptimal SVM fits.


def fit_predict_cls(Xtr, ytr, Xte):
    clf = GridSearchCV(SVC(max_iter=MAX_ITER), GRID, cv=3, n_jobs=1)
    clf.fit(Xtr, ytr)
    return clf.predict(Xte)


def fit_predict_reg(Xtr, ytr, Xte):
    reg = GridSearchCV(SVR(max_iter=MAX_ITER), GRID, cv=3, n_jobs=1)
    reg.fit(Xtr, ytr)
    return reg.predict(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="refed_features_libeer_lds.npz")
    ap.add_argument("--out", default="svm_persubj_montage.json")
    a = ap.parse_args()

    z = np.load(HERE / a.features, allow_pickle=True)
    X, Y, subject, video, channels = (z["X"], z["Y"], z["subject"], z["video"],
                                      [str(c) for c in z["channels"]])
    montages = dict(MONTAGES)
    montages["whole"] = [c for c in channels if c not in EXCLUDE_FROM_WHOLE]
    ch_idx = {n: [channels.index(c) for c in montages[n]] for n in ORDER}

    results = {m: {"cls": {d: {"acc": [], "macro": []} for d in DIMS},
                   "reg": {d: {"ccc": [], "acc3": []} for d in DIMS}}
              for m in ORDER}

    for sub in range(1, 33):
        ms = subject == sub
        for name in ORDER:
            Xm = X[ms][:, ch_idx[name], :5]
            Ym = Y[ms]
            vid = video[ms]
            C = np.stack([to_class(Ym[:, 0]), to_class(Ym[:, 1])], axis=1)

            fold_cls = {d: {"acc": [], "macro": []} for d in DIMS}
            fold_reg = {d: {"ccc": [], "acc3": []} for d in DIMS}
            for test_vids in FOLDS:
                te = np.isin(vid, test_vids)
                tr = ~te
                if te.sum() == 0 or tr.sum() == 0:
                    continue
                Xtr_n = norm(Xm[tr], Xm[tr])
                Xte_n = norm(Xm[tr], Xm[te])
                for i, d in enumerate(DIMS):
                    if len(np.unique(C[tr, i])) >= 2:
                        pred_c = fit_predict_cls(Xtr_n, C[tr, i], Xte_n)
                        fold_cls[d]["acc"].append(accuracy_score(C[te, i], pred_c))
                        fold_cls[d]["macro"].append(macro_recall(C[te, i], pred_c))
                    pred_r = fit_predict_reg(Xtr_n, Ym[tr, i], Xte_n)
                    fold_reg[d]["ccc"].append(ccc(Ym[te, i], pred_r))
                    fold_reg[d]["acc3"].append(
                        float((to_class(Ym[te, i]) == to_class(pred_r)).mean()))

            for d in DIMS:
                if fold_cls[d]["acc"]:
                    results[name]["cls"][d]["acc"].append(float(np.mean(fold_cls[d]["acc"])))
                    results[name]["cls"][d]["macro"].append(float(np.mean(fold_cls[d]["macro"])))
                results[name]["reg"][d]["ccc"].append(float(np.mean(fold_reg[d]["ccc"])))
                results[name]["reg"][d]["acc3"].append(float(np.mean(fold_reg[d]["acc3"])))

        print(f"sub {sub:2d} done: " + "  ".join(
            f"{name} Vacc={np.mean(results[name]['cls']['valence']['acc'][-1:]):.3f}"
            for name in ORDER), flush=True)

    with open(HERE / a.out, "w") as f:
        json.dump({"results": results, "montage_channels": {k: montages[k] for k in ORDER}},
                  f, indent=2)
    print("\nSaved ->", a.out)

    print("\n" + "=" * 70)
    for name in ORDER:
        v_acc = np.mean(results[name]["cls"]["valence"]["acc"])
        a_acc = np.mean(results[name]["cls"]["arousal"]["acc"])
        v_ccc = np.mean(results[name]["reg"]["valence"]["ccc"])
        a_ccc = np.mean(results[name]["reg"]["arousal"]["ccc"])
        print(f"{name:8s}  V acc={v_acc:.3f} ccc={v_ccc:+.3f} | "
              f"A acc={a_acc:.3f} ccc={a_ccc:+.3f}")


if __name__ == "__main__":
    main()
