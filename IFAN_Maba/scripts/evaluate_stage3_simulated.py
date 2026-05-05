from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ifan_maba.eval import (
    build_baseline_preprocessor_from_config,
    build_ifan_preprocessor_from_config,
    build_librispeech_dataset,
    build_scenario_caches,
    evaluate_model_on_cache,
)
from ifan_maba.models import IFANModel
from ifan_maba.training import IFANTrainingConfig, IFANTrainingPipeline


def quiet_progress(*_args, **_kwargs) -> None:
    return None


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def aggregate_simulated_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    if not runs:
        return {
            "validation": {"loss": summarize_values([]), "rmsae_deg": summarize_values([])},
            "mean_rmsae_deg": {
                "ifan": summarize_values([]),
                "baseline": summarize_values([]),
                "delta": summarize_values([]),
            },
            "hard_scenarios_mean_rmsae_deg": {
                "ifan": summarize_values([]),
                "baseline": summarize_values([]),
                "delta": summarize_values([]),
            },
            "scenarios": [],
        }

    validation_loss = [float(run["validation"]["loss"]) for run in runs]
    validation_rmsae = [float(run["validation"]["rmsae_deg"]) for run in runs]
    mean_ifan = [float(run["baseline_compare"]["mean_rmsae_deg"]["ifan"]) for run in runs]
    mean_baseline = [float(run["baseline_compare"]["mean_rmsae_deg"]["baseline"]) for run in runs]
    mean_delta = [float(run["baseline_compare"]["mean_rmsae_deg"]["delta"]) for run in runs]
    hard_ifan = [float(run["baseline_compare"]["hard_scenarios_mean_rmsae_deg"]["ifan"]) for run in runs]
    hard_baseline = [float(run["baseline_compare"]["hard_scenarios_mean_rmsae_deg"]["baseline"]) for run in runs]
    hard_delta = [float(run["baseline_compare"]["hard_scenarios_mean_rmsae_deg"]["delta"]) for run in runs]

    scenario_names = [row["name"] for row in runs[0]["baseline_compare"]["scenarios"]]
    scenarios: list[dict[str, object]] = []
    for scenario_name in scenario_names:
        first = next(row for row in runs[0]["baseline_compare"]["scenarios"] if row["name"] == scenario_name)
        rows = [
            next(row for row in run["baseline_compare"]["scenarios"] if row["name"] == scenario_name)
            for run in runs
        ]
        scenarios.append(
            {
                "name": scenario_name,
                "snr_db": float(first["snr_db"]),
                "t60_s": float(first["t60_s"]),
                "ifan": {
                    "loss": summarize_values([float(row["ifan"]["loss"]) for row in rows]),
                    "rmsae_deg": summarize_values([float(row["ifan"]["rmsae_deg"]) for row in rows]),
                },
                "baseline": {
                    "loss": summarize_values([float(row["baseline"]["loss"]) for row in rows]),
                    "rmsae_deg": summarize_values([float(row["baseline"]["rmsae_deg"]) for row in rows]),
                },
                "rmsae_delta_deg": summarize_values([float(row["rmsae_delta_deg"]) for row in rows]),
            }
        )

    return {
        "validation": {
            "loss": summarize_values(validation_loss),
            "rmsae_deg": summarize_values(validation_rmsae),
        },
        "mean_rmsae_deg": {
            "ifan": summarize_values(mean_ifan),
            "baseline": summarize_values(mean_baseline),
            "delta": summarize_values(mean_delta),
        },
        "hard_scenarios_mean_rmsae_deg": {
            "ifan": summarize_values(hard_ifan),
            "baseline": summarize_values(hard_baseline),
            "delta": summarize_values(hard_delta),
        },
        "scenarios": scenarios,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run larger-cache, multi-seed simulated IFAN-vs-baseline evaluation.")
    parser.add_argument("--checkpoint", action="append", required=True, help="Path to an IFAN stage-3 checkpoint (*.pt). Repeat to compare multiple checkpoints.")
    parser.add_argument("--label", action="append", default=None, help="Optional label matching each --checkpoint in order.")
    parser.add_argument("--config", default=None, help="Optional stage-3 config TOML. Uses checkpoint metadata when omitted.")
    parser.add_argument("--validation-size", type=int, default=128, help="Validation cache size per seed.")
    parser.add_argument("--scenario-eval-size", type=int, default=64, help="Per-scenario cache size per seed.")
    parser.add_argument("--trajectory-seconds", type=int, default=None, help="Override simulated trajectory duration.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Evaluation seeds.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--output", default=None, help="Optional path to write the JSON report.")
    return parser


def load_config(checkpoint_payload: dict[str, object], config_path: str | None) -> IFANTrainingConfig:
    if config_path is not None:
        return IFANTrainingConfig.from_toml(config_path)
    return IFANTrainingConfig(**checkpoint_payload["training_config"])


def cache_signature(config: IFANTrainingConfig, device: torch.device) -> str:
    payload: dict[str, Any] = {
        "device": str(device),
        "librispeech_path": config.librispeech_path,
        "test_split": config.test_split,
        "trajectory_seconds": config.trajectory_seconds,
        "k": config.k,
        "step": config.step,
        "r": config.r,
        "fs": config.fs,
        "smooth_vertices": config.smooth_vertices,
        "apply_vad": config.apply_vad,
        "lms_order": config.lms_order,
        "lms_step_size": config.lms_step_size,
        "lms_map_normalize": config.lms_map_normalize,
        "lms_map_mode": config.lms_map_mode,
        "lms_peak_sigma": config.lms_peak_sigma,
        "lms_update_mode": config.lms_update_mode,
        "lms_normalized": config.lms_normalized,
        "lms_include_self_pairs": config.lms_include_self_pairs,
        "lms_backend": config.lms_backend,
        "lms_block_size": config.lms_block_size,
        "lms_fft_size": config.lms_fft_size,
        "input_ablation_mode": config.input_ablation_mode,
        "nb_points": config.nb_points,
        "validation_dataset_size": config.validation_dataset_size,
        "validation_batch_size": config.validation_batch_size,
        "validation_snr_min": config.validation_snr_min,
        "validation_snr_max": config.validation_snr_max,
        "scenario_eval_size": config.scenario_eval_size,
        "scenario_eval_batch_size": config.scenario_eval_batch_size,
        "seed": config.seed,
        "baseline_checkpoint_path": config.baseline_checkpoint_path,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def apply_eval_overrides(config: IFANTrainingConfig, args: argparse.Namespace, seed: int) -> IFANTrainingConfig:
    config.validation_dataset_size = int(args.validation_size)
    config.scenario_eval_size = int(args.scenario_eval_size)
    config.seed = int(seed)
    if args.trajectory_seconds is not None:
        config.trajectory_seconds = int(args.trajectory_seconds)
    if args.device is not None:
        config.device = str(args.device)
    return config


def build_eval_resources(
    *,
    config: IFANTrainingConfig,
    pipeline: IFANTrainingPipeline,
    device: torch.device,
) -> dict[str, object]:
    val_source, val_source_path = build_librispeech_dataset(
        config.librispeech_path,
        config.test_split,
        config.trajectory_seconds,
    )
    ifan_preprocessor = build_ifan_preprocessor_from_config(config, device)
    baseline_preprocessor = build_baseline_preprocessor_from_config(config, device)
    validation_cache = pipeline.build_validation_cache(
        source_dataset=val_source,
        preprocessor=ifan_preprocessor,
        progress_callback=quiet_progress,
    )
    scenario_caches = build_scenario_caches(
        source_dataset=val_source,
        ifan_preprocessor=ifan_preprocessor,
        baseline_preprocessor=baseline_preprocessor,
        model_config=pipeline.model_config,
        input_ablation_mode=config.input_ablation_mode,
        k=config.k,
        step=config.step,
        batch_size=config.scenario_eval_batch_size,
        scenario_size=config.scenario_eval_size,
        seed=config.seed,
        nb_points=config.nb_points,
    )
    return {
        "validation_source_path": str(val_source_path),
        "validation_cache": validation_cache,
        "scenario_caches": scenario_caches,
    }


def evaluate_checkpoint_once(
    *,
    checkpoint: dict[str, object],
    config: IFANTrainingConfig,
    pipeline: IFANTrainingPipeline,
    device: torch.device,
    resources: dict[str, object],
) -> dict[str, object]:
    model = IFANModel(pipeline.model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    validation = evaluate_model_on_cache(model, resources["validation_cache"])
    baseline_compare = pipeline.compare_against_baseline(
        model=model,
        scenario_caches=resources["scenario_caches"],
        device=device,
    )
    return {
        "seed": int(config.seed),
        "device": str(device),
        "validation_source_path": str(resources["validation_source_path"]),
        "validation": validation,
        "baseline_compare": baseline_compare,
    }


def checkpoint_label(path: str | Path, index: int, labels: list[str] | None) -> str:
    if labels is not None and index < len(labels) and labels[index]:
        return str(labels[index])
    return Path(path).stem


def main() -> None:
    args = build_parser().parse_args()
    if args.label is not None and len(args.label) > len(args.checkpoint):
        raise ValueError("Received more --label values than --checkpoint values.")

    checkpoint_payloads = [torch.load(path, map_location="cpu") for path in args.checkpoint]
    report = {
        "kind": "stage3_simulated_analysis",
        "validation_size": int(args.validation_size),
        "scenario_eval_size": int(args.scenario_eval_size),
        "trajectory_seconds": None if args.trajectory_seconds is None else int(args.trajectory_seconds),
        "seeds": [int(seed) for seed in args.seeds],
        "checkpoints": [],
    }

    resource_cache: dict[str, dict[str, object]] = {}
    for index, checkpoint_path in enumerate(args.checkpoint):
        checkpoint_payload = checkpoint_payloads[index]
        runs = []
        for seed in args.seeds:
            config = apply_eval_overrides(load_config(checkpoint_payload, args.config), args, seed)
            pipeline = IFANTrainingPipeline(config)
            device = pipeline.resolve_device()
            signature = cache_signature(config, device)
            if signature not in resource_cache:
                resource_cache[signature] = build_eval_resources(
                    config=config,
                    pipeline=pipeline,
                    device=device,
                )
            runs.append(
                evaluate_checkpoint_once(
                    checkpoint=checkpoint_payload,
                    config=config,
                    pipeline=pipeline,
                    device=device,
                    resources=resource_cache[signature],
                )
            )
        report["checkpoints"].append(
            {
                "label": checkpoint_label(checkpoint_path, index, args.label),
                "checkpoint_path": str(Path(checkpoint_path)),
                "runs": runs,
                "aggregate": aggregate_simulated_runs(runs),
            }
        )

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
