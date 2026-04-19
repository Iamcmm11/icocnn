from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import acousticTrackingDataset as at_dataset
from acousticTrackingDataset import Parameter
from ifan_edge.bridges import icoCNN
from ifan_edge.eval.stage3 import STAGE3_SCENARIOS, build_librispeech_dataset, build_random_trajectory_dataset
from ifan_edge.features import DualFeatureIcoPreprocessor
from ifan_edge.features.phat import ensure_mic_tensor
from ifan_edge.training import IFANTrainingConfig, IFANTrainingPipeline
from utils import sph2cart


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose LMS filter peak vs tau_idx mismatch on fixed validation samples.")
    parser.add_argument("--config", default="IFAN_Edge/configs/stage3_default.toml")
    parser.add_argument("--size", type=int, default=4, help="Number of trajectories to inspect.")
    parser.add_argument("--scenario", choices=[row["name"] for row in STAGE3_SCENARIOS], default="scene_3")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--output-suffix", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = IFANTrainingConfig.from_toml(args.config)
    if args.device is not None:
        config.device = args.device

    device = IFANTrainingPipeline(config).resolve_device()
    source_dataset, source_path = build_librispeech_dataset(config.librispeech_path, config.test_split, config.trajectory_seconds)
    scenario = next(row for row in STAGE3_SCENARIOS if row["name"] == args.scenario)
    dataset = build_random_trajectory_dataset(
        source_dataset=source_dataset,
        k=config.k,
        step=config.step,
        size=args.size,
        t60=Parameter(float(scenario["t60_s"])),
        snr=Parameter(float(scenario["snr_db"])),
        nb_points=config.nb_points,
    )

    preprocessor = DualFeatureIcoPreprocessor(
        N=at_dataset.benchmark2_array_setup.mic_pos.shape[0],
        K=config.k,
        r=config.r,
        rn=at_dataset.benchmark2_array_setup.mic_pos,
        fs=config.fs,
        apply_vad=config.apply_vad,
        lms_order=config.lms_order,
        lms_step_size=config.lms_step_size,
        lms_map_normalize=config.lms_map_normalize,
        lms_map_mode=config.lms_map_mode,
        lms_peak_sigma=config.lms_peak_sigma,
        lms_update_mode=config.lms_update_mode,
        lms_normalized=config.lms_normalized,
        lms_include_self_pairs=config.lms_include_self_pairs,
        lms_backend=config.lms_backend,
        lms_block_size=config.lms_block_size,
        lms_fft_size=config.lms_fft_size,
    )
    IFANTrainingPipeline.move_ifan_preprocessor(preprocessor, device)

    mic_sig_batch, acoustic_scene_batch = dataset.get_batch(0, len(dataset))
    with torch.no_grad():
        _, doa_batch = preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)

    mic_sig_batch = ensure_mic_tensor(mic_sig_batch).float()
    if device.type == "cuda":
        mic_sig_batch = mic_sig_batch.to(device)
    lms = preprocessor.lms

    batch_frames = mic_sig_batch.reshape(-1, lms.N, lms.K)
    sources = batch_frames[:, lms.pair_i, :].permute(1, 0, 2).contiguous()
    targets = batch_frames[:, lms.pair_j, :].permute(1, 0, 2).contiguous()
    with torch.no_grad():
        pair_filters = lms._estimate_pair_filters(sources, targets)

    peak_idx = pair_filters.abs().argmax(dim=-1)

    grid = torch.from_numpy(icoCNN.icosahedral_grid_coordinates(config.r)).float()
    grid_flat = grid.reshape(-1, 3).to(doa_batch.device)
    doa_cart = sph2cart(doa_batch).reshape(-1, 3)
    nearest = torch.matmul(doa_cart, grid_flat.T).argmax(dim=-1)
    nearest_tau = lms.tau_idx.to(peak_idx.device)[:, nearest]
    bias = peak_idx - nearest_tau
    abs_bias = bias.abs()

    summary = {
        "source_path": str(source_path),
        "scenario": dict(scenario),
        "size": int(args.size),
        "device": str(device),
        "lms_options": {
            "map_normalize": bool(config.lms_map_normalize),
            "normalized_lms": bool(config.lms_normalized),
            "include_self_pairs": bool(config.lms_include_self_pairs),
            "lms_order": int(config.lms_order),
            "lms_step_size": float(config.lms_step_size),
            "lms_map_mode": str(config.lms_map_mode),
            "lms_peak_sigma": float(config.lms_peak_sigma),
            "lms_update_mode": str(config.lms_update_mode),
            "lms_backend": str(config.lms_backend),
            "lms_block_size": int(config.lms_block_size),
            "lms_fft_size": None if config.lms_fft_size is None else int(config.lms_fft_size),
        },
        "pair_count": int(peak_idx.shape[0]),
        "frame_count": int(peak_idx.shape[1]),
        "bias_stats": {
            "mean_abs_bias": float(abs_bias.float().mean().item()),
            "median_abs_bias": float(abs_bias.float().median().item()),
            "max_abs_bias": int(abs_bias.max().item()),
            "pct_exact": float((abs_bias == 0).float().mean().item()),
            "pct_within_1": float((abs_bias <= 1).float().mean().item()),
            "pct_within_2": float((abs_bias <= 2).float().mean().item()),
        },
        "sample_frames": [
            {
                "frame_index": int(frame_idx),
                "nearest_grid_index": int(nearest[frame_idx].item()),
                "pair_samples": [
                    {
                        "pair_index": int(pair_idx),
                        "peak_idx": int(peak_idx[pair_idx, frame_idx].item()),
                        "tau_idx": int(nearest_tau[pair_idx, frame_idx].item()),
                        "bias": int(bias[pair_idx, frame_idx].item()),
                    }
                    for pair_idx in range(min(5, peak_idx.shape[0]))
                ],
            }
            for frame_idx in range(min(5, peak_idx.shape[1]))
        ],
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    output_path = Path(config.output_root) / f"lms_peak_diagnostic_{args.scenario}{suffix}_{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), **summary["bias_stats"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
