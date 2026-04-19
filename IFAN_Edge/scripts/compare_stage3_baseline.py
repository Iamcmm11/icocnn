from __future__ import annotations

import argparse
import json
import sys
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
import acousticTrackingLearners as at_learners

from ifan_edge.eval.stage3 import build_librispeech_dataset, build_scenario_caches
from ifan_edge.features import DualFeatureIcoPreprocessor
from ifan_edge.models import IFANModel
from ifan_edge.training import IFANTrainingConfig, IFANTrainingPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a trained IFAN stage-3 checkpoint against the icoCNN r=2 baseline.")
    parser.add_argument("--checkpoint", required=True, help="Path to an IFAN stage-3 checkpoint (*.pt).")
    parser.add_argument("--config", default=None, help="Optional stage-3 config TOML. Uses checkpoint metadata when omitted.")
    parser.add_argument("--scenario-eval-size", type=int, default=None)
    parser.add_argument("--trajectory-seconds", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if args.config is not None:
        config = IFANTrainingConfig.from_toml(args.config)
    else:
        config = IFANTrainingConfig(**checkpoint["training_config"])

    if args.scenario_eval_size is not None:
        config.scenario_eval_size = int(args.scenario_eval_size)
    if args.trajectory_seconds is not None:
        config.trajectory_seconds = int(args.trajectory_seconds)
    if args.device is not None:
        config.device = args.device

    pipeline = IFANTrainingPipeline(config)
    device = pipeline.resolve_device()

    val_source, _ = build_librispeech_dataset(
        config.librispeech_path,
        config.test_split,
        config.trajectory_seconds,
    )

    ifan_preprocessor = DualFeatureIcoPreprocessor(
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
    baseline_preprocessor = at_learners.TrackingFromIcoMapsPreprocessor(
        N=at_dataset.benchmark2_array_setup.mic_pos.shape[0],
        K=config.k,
        r=config.r,
        rn=at_dataset.benchmark2_array_setup.mic_pos,
        fs=config.fs,
        apply_vad=config.apply_vad,
    )
    pipeline.move_ifan_preprocessor(ifan_preprocessor, device)
    pipeline.move_baseline_preprocessor(baseline_preprocessor, device)

    scenario_caches = build_scenario_caches(
        source_dataset=val_source,
        ifan_preprocessor=ifan_preprocessor,
        baseline_preprocessor=baseline_preprocessor,
        model_config=pipeline.model_config,
        k=config.k,
        step=config.step,
        batch_size=config.scenario_eval_batch_size,
        scenario_size=config.scenario_eval_size,
        seed=config.seed,
        nb_points=config.nb_points,
    )

    model = IFANModel(pipeline.model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    report = pipeline.compare_against_baseline(
        model=model,
        scenario_caches=scenario_caches,
        device=device,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
