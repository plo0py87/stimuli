"""Diagnostic: is there ANY learnable continuous-valence signal in REFED,
and does the protocol or the model explain the near-zero CCC?

Compares, per subject, on the whole-cap montage:
  - trivial      predict the training mean (CCC == 0 by construction)
  - ridge/LOVO   ridge regression, leave-videos-out (same protocol as DGCNN)
  - ridge/within train on first 60% of each clip, test on last 40%
                 (easier: the model has seen this clip's content)

If ridge/LOVO ~ 0 but ridge/within is clearly positive, the signal exists but
does not generalize to unseen stimuli -- a protocol/task property, not a bug.
If both are ~0, the features carry almost no continuous-valence information.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from refed_montage_regress import FOLDS, ccc, pcc  # noqa: E402

EPS = 1e-8


def zscore_fit(a):
    flat = a.reshape(len(a), -1)
    return flat.mean(axis=0), flat.std(axis=0) + EPS


def main():
    z = np.load(HERE / "refed_features_all64.npz", allow_pickle=True)
    X, Y, subject, video = z["X"], z["Y"], z["subject"], z["video"]
    second = z["second"]

    subs = sorted(set(subject.tolist()))[:8]
    rows = {"ridge_lovo": [], "ridge_within": [], "trivial": []}

    for sub in subs:
        m = subject == sub
        Xs = X[m].reshape((m.sum(), -1))
        Ys, vs, ss = Y[m], video[m], second[m]

        # --- leave-videos-out (same folds as the DGCNN run) ---
        pred = np.zeros_like(Ys)
        for test_vids in FOLDS:
            te = np.isin(vs, test_vids)
            tr = ~te
            if te.sum() == 0 or tr.sum() == 0:
                continue
            mu, sd = Xs[tr].mean(axis=0), Xs[tr].std(axis=0) + EPS
            r = Ridge(alpha=100.0).fit((Xs[tr] - mu) / sd, Ys[tr])
            pred[te] = r.predict((Xs[te] - mu) / sd)
        rows["ridge_lovo"].append([ccc(Ys[:, i], pred[:, i]) for i in range(2)])

        # --- trivial: predict the train-fold mean ---
        triv = np.zeros_like(Ys)
        for test_vids in FOLDS:
            te = np.isin(vs, test_vids)
            tr = ~te
            if te.sum() == 0 or tr.sum() == 0:
                continue
            triv[te] = Ys[tr].mean(axis=0)
        rows["trivial"].append([ccc(Ys[:, i], triv[:, i]) for i in range(2)])

        # --- within-clip temporal split: first 60% train, last 40% test ---
        tr_mask = np.zeros(len(Ys), bool)
        for v in np.unique(vs):
            idx = np.nonzero(vs == v)[0]
            order = idx[np.argsort(ss[idx])]
            tr_mask[order[:int(len(order) * 0.6)]] = True
        mu, sd = Xs[tr_mask].mean(axis=0), Xs[tr_mask].std(axis=0) + EPS
        r = Ridge(alpha=100.0).fit((Xs[tr_mask] - mu) / sd, Ys[tr_mask])
        p_in = r.predict((Xs[~tr_mask] - mu) / sd)
        rows["ridge_within"].append(
            [ccc(Ys[~tr_mask][:, i], p_in[:, i]) for i in range(2)])

        print(f"sub {sub:2d}  "
              f"LOVO V={rows['ridge_lovo'][-1][0]:+.3f} A={rows['ridge_lovo'][-1][1]:+.3f}   "
              f"within V={rows['ridge_within'][-1][0]:+.3f} A={rows['ridge_within'][-1][1]:+.3f}")

    print("\n" + "=" * 60)
    for name, vals in rows.items():
        a = np.array(vals)
        print(f"{name:14s} valence CCC {a[:,0].mean():+.3f} +/- {a[:,0].std():.3f}   "
              f"arousal CCC {a[:,1].mean():+.3f} +/- {a[:,1].std():.3f}")


if __name__ == "__main__":
    main()
