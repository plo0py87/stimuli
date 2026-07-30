"""Saliency map for the DAGCN checkpoint trained in train_dagcn_own_data.py:
gradient of each predicted class's logit w.r.t. the (19,10) input feature
tensor, averaged over the 10 feature dims (5 causal DE bands + 5 causal PSD
bands) to get one importance score per channel, then rendered as a scalp
topomap per class using MNE's standard_1020 montage.

Usage:
  "C:/Users/USER/miniconda3/envs/EEG/python.exe" saliency_map.py \
      --checkpoint result_dagcn_own_data_split_session/checkpoint-scratch
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import torch

HERE = Path(__file__).parent
EMOTION_MODEL_DIR = HERE / "emotion_model"
sys.path.insert(0, str(EMOTION_MODEL_DIR))

from DAGCN_quick20_realtime_train import causal_psd_features, DAGCNQuick20Wrapper  # noqa: E402
from DGCNN_quick20_realtime_train import causal_de_features, kalman_smooth_trial  # noqa: E402
from train_dagcn_own_data import (  # noqa: E402
    QUICK20_CHANNEL_NAME, IDENTITY_INDICES, LABEL_TO_IDX, IDX_TO_LABEL,
    load_session_raw19, load_session_windows, LABEL_COLLAPSE, SESSIONS_DIR,
)

# MNE's standard_1020 montage uses mixed-case names (Fp1, Fz, Cz, Pz...)
MNE_CH_NAME = {
    "FP1": "Fp1", "FP2": "Fp2", "FZ": "Fz", "CZ": "Cz", "PZ": "Pz",
}


def mne_name(ch):
    return MNE_CH_NAME.get(ch, ch)


def build_eval_samples(kalman_q, kalman_r, max_sessions=None):
    """Reuse the same feature pipeline as training, returns (X, y) over all
    Shine_20260720_* sessions (no train/test split needed for saliency)."""
    session_dirs = sorted(
        d for d in SESSIONS_DIR.glob("Shine_20260720_*")
        if (d / "eeg_raw.npz").exists() and (d / "ratings.csv").exists()
    )
    if max_sessions:
        session_dirs = session_dirs[:max_sessions]

    samples_X, samples_y = [], []
    for session_dir in session_dirs:
        windows = load_session_windows(session_dir)
        if not windows:
            continue
        raw19, srate = load_session_raw19(session_dir / "eeg_raw.npz")
        srate_int = int(round(srate))
        de = causal_de_features(raw19, srate_int, IDENTITY_INDICES)
        psd = causal_psd_features(raw19, srate_int, IDENTITY_INDICES)
        combined = [np.concatenate([d, p], axis=-1) for d, p in zip(de, psd)]
        combined = kalman_smooth_trial(combined, kalman_q, kalman_r)
        n_secs = len(combined)

        for start_s, end_s, label in windows:
            s0, s1 = int(round(start_s)), min(n_secs, int(round(end_s)))
            if s1 <= s0:
                continue
            for t in range(s0, s1):
                samples_X.append(combined[t])
                samples_y.append(LABEL_TO_IDX[label])

    return np.stack(samples_X), np.array(samples_y, dtype=np.int64)


def compute_saliency(model, X, y, mean, std, device):
    """Returns dict class_idx -> (19,) per-channel mean |gradient| of that
    class's logit w.r.t. the normalized input, averaged over all samples
    whose TRUE label is that class (so each map reflects 'what the model
    looks at when it's shown a real example of this emotion')."""
    shape = X.shape
    flat = X.reshape(-1, shape[1] * shape[2])
    flat = (flat - mean) / std
    Xn = flat.reshape(shape)

    saliency_sum = {c: np.zeros(19) for c in range(3)}
    saliency_count = {c: 0 for c in range(3)}

    batch = 64
    for i in range(0, len(Xn), batch):
        xb = torch.tensor(Xn[i:i + batch], dtype=torch.float32, device=device, requires_grad=True)
        yb = y[i:i + batch]
        logits = model(xb)
        for c in range(3):
            mask = (yb == c)
            if not mask.any():
                continue
            model.zero_grad()
            if xb.grad is not None:
                xb.grad.zero_()
            target = logits[mask, c].sum()
            target.backward(retain_graph=True)
            grad = xb.grad[mask].detach().cpu().numpy()  # (n_masked, 19, 10)
            per_channel = np.abs(grad).mean(axis=(0, 2))  # (19,)
            saliency_sum[c] += per_channel * mask.sum()
            saliency_count[c] += int(mask.sum())

    return {IDX_TO_LABEL[c]: (saliency_sum[c] / max(1, saliency_count[c])) for c in range(3)}


def plot_saliency(saliency, out_path):
    montage = mne.channels.make_standard_montage("standard_1020")
    mne_names = [mne_name(ch) for ch in QUICK20_CHANNEL_NAME]
    info = mne.create_info(mne_names, sfreq=1.0, ch_types="eeg")
    info.set_montage(montage)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    vmax = max(v.max() for v in saliency.values())
    for ax, (label, values) in zip(axes, saliency.items()):
        im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="Reds", vlim=(0, vmax))
        ax.set_title(f"{label}\n(n saliency)", fontsize=12)
    fig.suptitle("DAGCN saliency (mean |d logit / d input| per channel), own-data checkpoint", fontsize=13)
    cbar = fig.colorbar(im, ax=axes, shrink=0.7, label="saliency (a.u.)")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="result_dagcn_own_data_split_session/checkpoint-scratch")
    parser.add_argument("--kalman-q", type=float, default=0.01)
    parser.add_argument("--kalman-r", type=float, default=0.5)
    parser.add_argument("--out", default="saliency_map.png")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mean, std = ckpt["mean"], ckpt["std"]
    model = DAGCNQuick20Wrapper(19, 10, 3)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    print("Building evaluation samples from Shine_20260720_* sessions...")
    X, y = build_eval_samples(args.kalman_q, args.kalman_r)
    print(f"Total samples: {len(X)}")

    saliency = compute_saliency(model, X, y, mean, std, device)
    for label, values in saliency.items():
        top = sorted(zip(QUICK20_CHANNEL_NAME, values), key=lambda kv: -kv[1])[:5]
        print(f"{label}: top channels -> " + ", ".join(f"{ch}({v:.3f})" for ch, v in top))

    plot_saliency(saliency, args.out)


if __name__ == "__main__":
    main()
