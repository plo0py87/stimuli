# -*- coding: utf-8 -*-
"""Assemble the interactive per-subject dashboard: one self-contained HTML
file with all 32 subjects' t-SNE/UMAP/AV-trajectory/regression-curve images
and SVM-vs-DGCNN accuracy/CCC numbers embedded (both plain DE and DE_LDS),
plus a vanilla-JS app for per-subject browsing and cross-subject metric
comparison.

Usage:
  "C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" build_dashboard.py
"""
import base64
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MONTAGES = ["whole", "vr_ring", "quick20", "headtop"]
MONT_LABEL = {"whole": "Whole cap", "vr_ring": "Glasses", "quick20": "Quick-20",
             "headtop": "Head-top"}
DIMS = ["valence", "arousal"]
FEATS = ["lds", "de"]
# NOTE: no "pooled" DGCNN results are loaded here. REFED's pooled mode trains
# on every subject's clips including the target subject's OWN other clips
# (refed_montage_regress.py:25-34) -- it is not a subject-independent metric,
# so it was dropped from the dashboard entirely rather than shown alongside
# genuine per-subject numbers.
FEAT_FILES = {
    "lds": dict(reg_ps="libeer_lds_persubj.json", cls_ps="libeer_lds_persubj_cls.json",
               svm="svm_persubj_montage.json", reg_pred="libeer_lds_persubj_pred.npz"),
    "de": dict(reg_ps="libeer_de_persubj.json", cls_ps="libeer_de_persubj_cls.json",
              svm="svm_persubj_montage_de.json", reg_pred="libeer_de_persubj_pred.npz"),
}


def b64(path):
    return base64.b64encode((HERE / path).read_bytes()).decode()


def load(name):
    return json.loads((HERE / name).read_text())


# ---- load all data sources, both feature types ----
D = {}
REG_TRIALS = {}
for feat, files in FEAT_FILES.items():
    D[feat] = {
        "reg_ps": load(files["reg_ps"])["results"]["per_subject"],
        "cls_ps": load(files["cls_ps"])["results"]["per_subject"],
        "svm": load(files["svm"])["results"],
    }
    for m in MONTAGES:
        for d in DIMS:
            n = len(D[feat]["svm"][m]["cls"][d]["acc"])
            assert n == 32, f"[{feat}] svm cls {m}/{d} has {n} entries, expected 32"
    REG_TRIALS[feat] = load(f"dash_reg_trials_{feat}.json")

tsne = load("dash_tsne_all.json")
umap_d = load("dash_umap_all.json")
# only the genuine per-subject SEED comparison is used -- see FEAT_FILES note
# above re: why pooled numbers were dropped project-wide.
seed_trunc_persubj = {
    "lds": load("seed_montage_cls_persubj_trunc.json")["results"],
    "de": load("seed_montage_cls_persubj_trunc_de.json")["results"],
}

# ---- images: collect once, referenced by key from JS ----
IMAGES = {}
for sub in range(1, 33):
    IMAGES[f"av_{sub}"] = b64(f"dash_av_sub{sub}.png")
    IMAGES[f"tsne_{sub}_de"] = b64(tsne["per_subject"][str(sub)]["de"]["image"])
    IMAGES[f"tsne_{sub}_lds"] = b64(tsne["per_subject"][str(sub)]["lds"]["image"])
    IMAGES[f"umap_{sub}_de"] = b64(umap_d["per_subject"][str(sub)]["de"]["image"])
    IMAGES[f"umap_{sub}_lds"] = b64(umap_d["per_subject"][str(sub)]["lds"]["image"])
    for feat in FEATS:
        for trial in REG_TRIALS[feat][str(sub)]:
            r = trial["rank"]
            IMAGES[f"reg_{sub}_{feat}_t{r}"] = b64(f"dash_reg_sub{sub}_{feat}_t{r}.png")
IMAGES["tsne_pooled_de"] = b64(tsne["pooled"]["de"]["image"])
IMAGES["tsne_pooled_lds"] = b64(tsne["pooled"]["lds"]["image"])
IMAGES["umap_pooled_de"] = b64(umap_d["pooled"]["de"]["image"])
IMAGES["umap_pooled_lds"] = b64(umap_d["pooled"]["lds"]["image"])


def per_subject_block():
    out = {}
    for i, sub in enumerate(range(1, 33)):
        rec = {
            "tsne": {
                "de": {"sil_high": tsne["per_subject"][str(sub)]["de"]["sil_high"],
                      "sil_emb": tsne["per_subject"][str(sub)]["de"]["sil_emb"],
                      "n": tsne["per_subject"][str(sub)]["de"]["n"]},
                "lds": {"sil_high": tsne["per_subject"][str(sub)]["lds"]["sil_high"],
                       "sil_emb": tsne["per_subject"][str(sub)]["lds"]["sil_emb"],
                       "n": tsne["per_subject"][str(sub)]["lds"]["n"]},
            },
            "umap": {
                "de": {"sil_high": umap_d["per_subject"][str(sub)]["de"]["sil_high"],
                      "sil_emb": umap_d["per_subject"][str(sub)]["de"]["sil_emb"],
                      "n": umap_d["per_subject"][str(sub)]["de"]["n"]},
                "lds": {"sil_high": umap_d["per_subject"][str(sub)]["lds"]["sil_high"],
                       "sil_emb": umap_d["per_subject"][str(sub)]["lds"]["sil_emb"],
                       "n": umap_d["per_subject"][str(sub)]["lds"]["n"]},
            },
        }
        for feat in FEATS:
            d_ = D[feat]
            rec[f"dgcnn_{feat}"] = {"cls": {}, "reg": {}}
            rec[f"svm_{feat}"] = {"cls": {}, "reg": {}}
            rec[f"regTrials_{feat}"] = REG_TRIALS[feat][str(sub)]
            for m in MONTAGES:
                rec[f"dgcnn_{feat}"]["cls"][m] = {
                    dim: {"acc": d_["cls_ps"][m]["per_subject"][dim]["acc"][i],
                         "macro": d_["cls_ps"][m]["per_subject"][dim]["macro"][i]}
                    for dim in DIMS}
                rec[f"dgcnn_{feat}"]["reg"][m] = {
                    dim: {"ccc": d_["reg_ps"][m]["per_subject"][dim]["ccc"][i],
                         "acc3": d_["reg_ps"][m]["per_subject"][dim]["acc3"][i]}
                    for dim in DIMS}
                rec[f"svm_{feat}"]["cls"][m] = {
                    dim: {"acc": d_["svm"][m]["cls"][dim]["acc"][i],
                         "macro": d_["svm"][m]["cls"][dim]["macro"][i]}
                    for dim in DIMS}
                rec[f"svm_{feat}"]["reg"][m] = {
                    dim: {"ccc": d_["svm"][m]["reg"][dim]["ccc"][i],
                         "acc3": d_["svm"][m]["reg"][dim]["acc3"][i]}
                    for dim in DIMS}
        out[str(sub)] = rec
    return out


def seed_fair_block():
    """SEED, truncated to REFED's mean trial length, genuine per-subject
    training (one DGCNN per subject, own trials only) -- for both feats."""
    out = {}
    for feat in FEATS:
        out[feat] = {}
        for m in MONTAGES:
            r = seed_trunc_persubj[feat][m]["per_subject"]
            out[feat][m] = {"acc": float(np.mean(r["acc"])), "macro": float(np.mean(r["macro"]))}
    return out


_z = np.load(HERE / "refed_features_libeer_lds.npz", allow_pickle=True)
_durs = []
for _s in np.unique(_z["subject"]):
    for _v in np.unique(_z["video"]):
        _n = int(((_z["subject"] == _s) & (_z["video"] == _v)).sum())
        if _n > 0:
            _durs.append(_n)
REFED_TRIAL_SEC = round(float(np.mean(_durs)))

DASH = {
    "subjects": list(range(1, 33)),
    "montages": MONTAGES,
    "montageLabel": MONT_LABEL,
    "dims": DIMS,
    "feats": FEATS,
    "featLabel": {"lds": "DE_LDS (smoothed)", "de": "plain DE (no smoothing)"},
    "flaggedSubjects": [13],
    "perSubject": per_subject_block(),
    "seedFair": seed_fair_block(),
    "refedTrialSec": REFED_TRIAL_SEC,
    "seedTrialSec": 226,
}

DASH_JSON = json.dumps(DASH)
IMAGES_JSON = json.dumps(IMAGES)

print(f"DASH payload: {len(DASH_JSON)/1e6:.2f} MB, IMAGES payload: {len(IMAGES_JSON)/1e6:.2f} MB")

html = (HERE / "dashboard_template.html").read_text(encoding="utf-8")
html = html.replace("__DASH_JSON__", DASH_JSON).replace("__IMAGES_JSON__", IMAGES_JSON)

out = HERE / "dashboard.html"
out.write_text(html, encoding="utf-8")
print("wrote", out, f"{out.stat().st_size/1e6:.2f} MB")
