"""Visualize map refinement and frame-to-frame DOA jitter."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import acousticTrackingDataset as at_dataset
import acousticTrackingLearners as at_learners
from utils import cart2sph

from maba_doa.train_maba_doa import build_datasets, build_model, load_config, set_seed


def frame_jitter_deg(doa_cart):
    """Mean frame-to-frame angular change in degrees."""
    doa_sph = cart2sph(doa_cart)  # (..., 2)
    doa_np = doa_sph.detach().cpu().numpy()
    the = doa_np[..., 0]
    phi = doa_np[..., 1]
    d_the = np.diff(the, axis=-2)
    d_phi = np.diff(phi, axis=-2)
    d = np.sqrt(d_the ** 2 + d_phi ** 2)
    return float(np.mean(d) * 180.0 / np.pi)


def main():
    parser = argparse.ArgumentParser(description="Visualize MABA map refinement.")
    parser.add_argument("--config", type=str, default="maba_doa/configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=str, default="maba_doa/outputs/map_refinement.png")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"]["seed"]))

    model = build_model(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    if not args.cpu and torch.cuda.is_available():
        model = model.cuda()

    dataset_train, dataset_test, _ = build_datasets(cfg)
    del dataset_train
    mic_sig_batch, acoustic_scene_batch = dataset_test.get_batch(0, 1)

    preprocessor = at_learners.TrackingFromIcoMapsPreprocessor(
        N=at_dataset.benchmark2_array_setup.mic_pos.shape[0],
        K=int(cfg["data"]["window"]["K"]),
        r=int(cfg["data"]["r"]),
        rn=at_dataset.benchmark2_array_setup.mic_pos,
        fs=int(cfg["data"]["fs"]),
        apply_vad=bool(cfg["data"]["use_vad"]),
    )
    if not args.cpu and torch.cuda.is_available():
        preprocessor.cuda()

    x_batch, _ = preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
    model.eval()
    with torch.no_grad():
        if hasattr(model, "maba"):
            doa, maps = model(x_batch, return_maps=True)
            maps_before = maps["maps_before"][0].detach().cpu().numpy()
            maps_after = maps["maps_after"][0].detach().cpu().numpy()
        else:
            doa = model(x_batch)
            maps_before = model.apply_cnn(x_batch)[0].detach().cpu().numpy()
            maps_after = maps_before.copy()

    jitter = frame_jitter_deg(doa)

    frame_idx = int(np.clip(args.frame, 0, maps_before.shape[0] - 1))
    before_frame = maps_before[frame_idx].reshape(5, -1)
    after_frame = maps_after[frame_idx].reshape(5, -1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(before_frame, aspect="auto", cmap="viridis")
    axes[0].set_title("Before MABA (frame {})".format(frame_idx))
    axes[0].set_xlabel("Flattened HW")
    axes[0].set_ylabel("Chart")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(after_frame, aspect="auto", cmap="viridis")
    axes[1].set_title("After MABA (frame {})".format(frame_idx))
    axes[1].set_xlabel("Flattened HW")
    axes[1].set_ylabel("Chart")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Map refinement; frame-to-frame jitter = {:.3f} deg".format(jitter))
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print("Saved visualization:", args.output)
    print("Mean frame jitter (deg):", jitter)


if __name__ == "__main__":
    main()
