from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ifan_edge.eval import (
    STAGE3_SCENARIOS,
    build_baseline_preprocessor_from_config,
    build_ifan_preprocessor_from_config,
    build_librispeech_dataset,
    build_single_scenario_cache,
    evaluate_detailed_model_on_cache,
    load_baseline_model_from_config,
)
from ifan_edge.models import IFANModel
from ifan_edge.training import IFANTrainingConfig, IFANTrainingPipeline


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


def flatten_trajectory_rmsae(report: dict[str, object]) -> list[float]:
    values: list[float] = []
    for batch in report["batch_reports"]:
        values.extend(np.asarray(batch["trajectory_rmsae_deg"], dtype=np.float64).reshape(-1).tolist())
    return values


def stack_frame_errors(report: dict[str, object]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for batch in report["batch_reports"]:
        frame_errors = np.asarray(batch["frame_errors_deg"], dtype=np.float64)
        if frame_errors.size == 0:
            continue
        rows.extend(frame_errors.reshape(-1, frame_errors.shape[-1]))
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.stack(rows, axis=0)


def summarize_detailed_report(report: dict[str, object]) -> dict[str, object]:
    trajectory_rmsae = flatten_trajectory_rmsae(report)
    frame_errors = stack_frame_errors(report)
    frame_mean = frame_errors.mean(axis=0).tolist() if frame_errors.size else []
    frame_std = frame_errors.std(axis=0, ddof=0).tolist() if frame_errors.size else []
    frame_p90 = np.percentile(frame_errors, 90, axis=0).tolist() if frame_errors.size else []
    return {
        "loss": float(report["loss"]),
        "rmsae_deg": float(report["rmsae_deg"]),
        "trajectory_rmsae_deg": trajectory_rmsae,
        "trajectory_stats": summarize_values(trajectory_rmsae),
        "frame_mean_deg": [float(value) for value in frame_mean],
        "frame_std_deg": [float(value) for value in frame_std],
        "frame_p90_deg": [float(value) for value in frame_p90],
        "trajectory_count": len(trajectory_rmsae),
        "frame_count": int(frame_errors.shape[1]) if frame_errors.ndim == 2 else 0,
    }


def plot_trajectory_rmsae(
    *,
    ifan_values: list[float],
    baseline_values: list[float],
    scenario_name: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)
    x = np.arange(max(len(ifan_values), len(baseline_values)))
    if ifan_values:
        ax.plot(x[: len(ifan_values)], ifan_values, marker="o", linewidth=1.5, label="IFAN")
    if baseline_values:
        ax.plot(x[: len(baseline_values)], baseline_values, marker="o", linewidth=1.5, label="Baseline")
    ax.set_title(f"{scenario_name} trajectory RMSAE")
    ax.set_xlabel("Trajectory index")
    ax.set_ylabel("RMSAE [deg]")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_frame_error_bands(
    *,
    ifan_mean: list[float],
    ifan_std: list[float],
    baseline_mean: list[float],
    baseline_std: list[float],
    scenario_name: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)
    if ifan_mean:
        x_ifan = np.arange(len(ifan_mean))
        ifan_mean_arr = np.asarray(ifan_mean, dtype=np.float64)
        ifan_std_arr = np.asarray(ifan_std, dtype=np.float64)
        ax.plot(x_ifan, ifan_mean_arr, linewidth=1.5, label="IFAN mean")
        ax.fill_between(x_ifan, ifan_mean_arr - ifan_std_arr, ifan_mean_arr + ifan_std_arr, alpha=0.2)
    if baseline_mean:
        x_baseline = np.arange(len(baseline_mean))
        baseline_mean_arr = np.asarray(baseline_mean, dtype=np.float64)
        baseline_std_arr = np.asarray(baseline_std, dtype=np.float64)
        ax.plot(x_baseline, baseline_mean_arr, linewidth=1.5, label="Baseline mean")
        ax.fill_between(
            x_baseline,
            baseline_mean_arr - baseline_std_arr,
            baseline_mean_arr + baseline_std_arr,
            alpha=0.2,
        )
    ax.set_title(f"{scenario_name} framewise angular error")
    ax.set_xlabel("Frame index after 5-frame exclusion")
    ax.set_ylabel("Angular error [deg]")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect one stage-3 scenario with larger caches, per-trajectory stats, and plots.")
    parser.add_argument("--checkpoint", required=True, help="Path to an IFAN stage-3 checkpoint (*.pt).")
    parser.add_argument("--config", default=None, help="Optional stage-3 config TOML. Uses checkpoint metadata when omitted.")
    parser.add_argument("--scenario", choices=[row["name"] for row in STAGE3_SCENARIOS], default="scene_2")
    parser.add_argument("--size", type=int, default=64, help="Number of trajectories to evaluate in the selected scenario.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trajectory-seconds", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--output-dir", default=None, help="Optional directory for JSON and figure outputs.")
    return parser


def load_config(checkpoint_payload: dict[str, object], config_path: str | None) -> IFANTrainingConfig:
    if config_path is not None:
        return IFANTrainingConfig.from_toml(config_path)
    return IFANTrainingConfig(**checkpoint_payload["training_config"])


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = load_config(checkpoint, args.config)
    if args.seed is not None:
        config.seed = int(args.seed)
    if args.trajectory_seconds is not None:
        config.trajectory_seconds = int(args.trajectory_seconds)
    if args.device is not None:
        config.device = str(args.device)

    pipeline = IFANTrainingPipeline(config)
    device = pipeline.resolve_device()
    source_dataset, source_path = build_librispeech_dataset(
        config.librispeech_path,
        config.test_split,
        config.trajectory_seconds,
    )
    ifan_preprocessor = build_ifan_preprocessor_from_config(config, device)
    baseline_preprocessor = build_baseline_preprocessor_from_config(config, device)
    scenario_cache = build_single_scenario_cache(
        source_dataset=source_dataset,
        scenario=args.scenario,
        ifan_preprocessor=ifan_preprocessor,
        baseline_preprocessor=baseline_preprocessor,
        model_config=pipeline.model_config,
        input_ablation_mode=config.input_ablation_mode,
        k=config.k,
        step=config.step,
        batch_size=config.scenario_eval_batch_size,
        scenario_size=int(args.size),
        seed=config.seed,
        nb_points=config.nb_points,
    )

    ifan_model = IFANModel(pipeline.model_config)
    ifan_model.load_state_dict(checkpoint["model_state_dict"])
    ifan_model.to(device)
    baseline_model = load_baseline_model_from_config(config, device)

    ifan_report = evaluate_detailed_model_on_cache(ifan_model, scenario_cache["ifan_batches"])
    baseline_report = evaluate_detailed_model_on_cache(baseline_model, scenario_cache["baseline_batches"])
    ifan_summary = summarize_detailed_report(ifan_report)
    baseline_summary = summarize_detailed_report(baseline_report)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.output_root) / "analysis" / f"{Path(args.checkpoint).stem}_{args.scenario}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_plot_path = output_dir / f"{args.scenario}_trajectory_rmsae.png"
    frame_plot_path = output_dir / f"{args.scenario}_frame_error_band.png"
    plot_trajectory_rmsae(
        ifan_values=ifan_summary["trajectory_rmsae_deg"],
        baseline_values=baseline_summary["trajectory_rmsae_deg"],
        scenario_name=str(args.scenario),
        output_path=trajectory_plot_path,
    )
    plot_frame_error_bands(
        ifan_mean=ifan_summary["frame_mean_deg"],
        ifan_std=ifan_summary["frame_std_deg"],
        baseline_mean=baseline_summary["frame_mean_deg"],
        baseline_std=baseline_summary["frame_std_deg"],
        scenario_name=str(args.scenario),
        output_path=frame_plot_path,
    )

    summary = {
        "kind": "stage3_scene_analysis",
        "checkpoint_path": str(Path(args.checkpoint)),
        "scenario": scenario_cache["scenario"],
        "source_path": str(source_path),
        "seed": int(config.seed),
        "size": int(args.size),
        "device": str(device),
        "trajectory_plot_path": str(trajectory_plot_path),
        "frame_plot_path": str(frame_plot_path),
        "ifan": ifan_summary,
        "baseline": baseline_summary,
        "delta": {
            "rmsae_deg": float(ifan_summary["rmsae_deg"] - baseline_summary["rmsae_deg"]),
            "trajectory_rmsae_deg": summarize_values(
                [
                    float(ifan - base)
                    for ifan, base in zip(ifan_summary["trajectory_rmsae_deg"], baseline_summary["trajectory_rmsae_deg"])
                ]
            ),
        },
    }
    summary_path = output_dir / f"{args.scenario}_analysis.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "delta_rmsae_deg": summary["delta"]["rmsae_deg"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
