from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
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

import acousticTrackingDataset as at_dataset
from acousticTrackingDataset import Parameter
from ifan_edge.bridges import icoCNN
from ifan_edge.eval.stage3 import STAGE3_SCENARIOS, build_librispeech_dataset, build_random_trajectory_dataset
from ifan_edge.features import SRPLMSIcoMap
from ifan_edge.features.phat import ensure_mic_tensor
from ifan_edge.training import IFANTrainingConfig, IFANTrainingPipeline
from utils import sph2cart


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare stage-3 LMS frontend outputs between time_reference and frequency_block backends.")
    parser.add_argument("--config", default="IFAN_Edge/configs/stage3_default.toml")
    parser.add_argument("--mode", choices=("scenario", "validation", "scenario_suite"), default="scenario")
    parser.add_argument("--scenario", choices=[row["name"] for row in STAGE3_SCENARIOS], default="scene_3")
    parser.add_argument("--size", type=int, default=2, help="Number of trajectories to compare.")
    parser.add_argument("--batch-size", type=int, default=1, help="Trajectory batch size during frontend comparison.")
    parser.add_argument("--trajectory-seconds", type=int, default=None, help="Optional override for faster smoke comparisons.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--lms-block-size", type=int, default=None)
    parser.add_argument("--lms-fft-size", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser


def build_lms_frontend(config: IFANTrainingConfig, *, backend: str, device: torch.device) -> SRPLMSIcoMap:
    frontend = SRPLMSIcoMap(
        N=at_dataset.benchmark2_array_setup.mic_pos.shape[0],
        K=config.k,
        r=config.r,
        rn=at_dataset.benchmark2_array_setup.mic_pos,
        fs=config.fs,
        lms_order=config.lms_order,
        step_size=config.lms_step_size,
        normalize=config.lms_map_normalize,
        map_mode=config.lms_map_mode,
        peak_sigma=config.lms_peak_sigma,
        update_mode=config.lms_update_mode,
        normalized_lms=config.lms_normalized,
        include_self_pairs=config.lms_include_self_pairs,
        lms_backend=backend,
        lms_block_size=config.lms_block_size,
        lms_fft_size=config.lms_fft_size,
    )
    if device.type == "cuda":
        frontend.cuda()
    else:
        frontend.cpu()
    return frontend


def build_dataset(config: IFANTrainingConfig, *, mode: str, scenario: str, size: int):
    source_dataset, source_path = build_librispeech_dataset(
        config.librispeech_path,
        config.test_split,
        config.trajectory_seconds,
    )
    if mode == "scenario":
        scenario_row = next(row for row in STAGE3_SCENARIOS if row["name"] == scenario)
        dataset = build_random_trajectory_dataset(
            source_dataset=source_dataset,
            k=config.k,
            step=config.step,
            size=size,
            t60=Parameter(float(scenario_row["t60_s"])),
            snr=Parameter(float(scenario_row["snr_db"])),
            nb_points=config.nb_points,
        )
        return dataset, source_path, {
            "mode": mode,
            "scenario": dict(scenario_row),
        }

    dataset = build_random_trajectory_dataset(
        source_dataset=source_dataset,
        k=config.k,
        step=config.step,
        size=size,
        t60=Parameter(config.train_t60_min, config.train_t60_max),
        snr=Parameter(config.validation_snr_min, config.validation_snr_max),
        nb_points=config.nb_points,
    )
    return dataset, source_path, {
        "mode": mode,
        "validation_snr_range_db": [float(config.validation_snr_min), float(config.validation_snr_max)],
        "validation_t60_range_s": [float(config.train_t60_min), float(config.train_t60_max)],
    }


def build_doa_tensor(acoustic_scene_batch) -> torch.Tensor:
    return torch.tensor(
        np.stack(
            [
                np.stack(
                    [
                        acoustic_scene_batch[i].DOAw[n].astype(np.float32)
                        for n in range(len(acoustic_scene_batch[i].DOAw))
                    ]
                )
                for i in range(len(acoustic_scene_batch))
            ]
        )
    )


def build_vad_mask(acoustic_scene_batch) -> torch.Tensor:
    vad_batch = np.array([acoustic_scene_batch[i].vad for i in range(len(acoustic_scene_batch))])
    return torch.from_numpy((vad_batch.mean(axis=-1) > 2 / 3).astype(np.bool_))


def time_frontend(frontend: SRPLMSIcoMap, mic_sig_batch: torch.Tensor) -> tuple[torch.Tensor, float]:
    start = time.perf_counter()
    with torch.no_grad():
        maps = frontend(mic_sig_batch)
    if maps.is_cuda:
        torch.cuda.synchronize(maps.device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return maps, elapsed_ms


def masked_flatten(maps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat = maps.reshape(maps.shape[0], maps.shape[2], -1)
    return flat[mask].float()


def grid_tensor(r: int, device: torch.device) -> torch.Tensor:
    grid = torch.from_numpy(icoCNN.icosahedral_grid_coordinates(r)).float()
    grid = grid.reshape(-1, 3)
    grid = grid / grid.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return grid.to(device)


def angular_distance_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    cosine = torch.sum(a * b, dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def summarize_tensor(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "max": float(values.max().item()),
    }


def compare_batch(
    *,
    ref_maps: torch.Tensor,
    freq_maps: torch.Tensor,
    doa_batch: torch.Tensor,
    vad_mask: torch.Tensor,
    grid_flat: torch.Tensor,
) -> dict[str, Any]:
    ref_flat = masked_flatten(ref_maps, vad_mask)
    freq_flat = masked_flatten(freq_maps, vad_mask)
    if doa_batch.ndim == 4 and doa_batch.shape[1] == 1:
        doa_batch = doa_batch[:, 0, ...]
    doa_active = doa_batch[vad_mask].to(ref_flat.device)

    if ref_flat.numel() == 0:
        return {
            "active_frames": 0,
            "map_delta": {"mae": 0.0, "rmse": 0.0, "cosine_mean": 0.0, "cosine_min": 0.0},
            "peak_agreement": {"exact_ratio": 0.0, "peak_angle_gap_deg": {"mean": 0.0, "median": 0.0, "max": 0.0}},
            "frontend_accuracy": {
                "time_reference_peak_error_deg": {"mean": 0.0, "median": 0.0, "max": 0.0},
                "frequency_block_peak_error_deg": {"mean": 0.0, "median": 0.0, "max": 0.0},
                "frequency_minus_reference_mean_deg": 0.0,
            },
        }

    diff = freq_flat - ref_flat
    cosine = torch.nn.functional.cosine_similarity(ref_flat, freq_flat, dim=-1)
    ref_peak = ref_flat.argmax(dim=-1)
    freq_peak = freq_flat.argmax(dim=-1)
    ref_peak_cart = grid_flat[ref_peak]
    freq_peak_cart = grid_flat[freq_peak]
    doa_cart = sph2cart(doa_active)

    ref_peak_error = angular_distance_deg(ref_peak_cart, doa_cart)
    freq_peak_error = angular_distance_deg(freq_peak_cart, doa_cart)
    peak_gap = angular_distance_deg(ref_peak_cart, freq_peak_cart)

    return {
        "active_frames": int(ref_flat.shape[0]),
        "map_delta": {
            "mae": float(diff.abs().mean().item()),
            "rmse": float(torch.sqrt(diff.square().mean()).item()),
            "cosine_mean": float(cosine.mean().item()),
            "cosine_min": float(cosine.min().item()),
        },
        "peak_agreement": {
            "exact_ratio": float((ref_peak == freq_peak).float().mean().item()),
            "peak_angle_gap_deg": summarize_tensor(peak_gap),
        },
        "frontend_accuracy": {
            "time_reference_peak_error_deg": summarize_tensor(ref_peak_error),
            "frequency_block_peak_error_deg": summarize_tensor(freq_peak_error),
            "frequency_minus_reference_mean_deg": float(freq_peak_error.mean().item() - ref_peak_error.mean().item()),
        },
    }


def aggregate_batch_reports(batch_reports: list[dict[str, Any]], ref_times_ms: list[float], freq_times_ms: list[float]) -> dict[str, Any]:
    active_frames = sum(int(row["active_frames"]) for row in batch_reports)
    if active_frames == 0:
        return {
            "active_frames": 0,
            "timing_ms": {
                "time_reference_mean": float(np.mean(ref_times_ms)) if ref_times_ms else 0.0,
                "frequency_block_mean": float(np.mean(freq_times_ms)) if freq_times_ms else 0.0,
                "speedup_vs_reference": 0.0,
            },
            "map_delta": {"mae": 0.0, "rmse": 0.0, "cosine_mean": 0.0, "cosine_min": 0.0},
            "peak_agreement": {"exact_ratio": 0.0, "peak_angle_gap_deg": {"mean": 0.0, "median": 0.0, "max": 0.0}},
            "frontend_accuracy": {
                "time_reference_peak_error_deg": {"mean": 0.0, "median": 0.0, "max": 0.0},
                "frequency_block_peak_error_deg": {"mean": 0.0, "median": 0.0, "max": 0.0},
                "frequency_minus_reference_mean_deg": 0.0,
            },
        }

    def weighted_mean(key_path: tuple[str, ...]) -> float:
        total = 0.0
        for row in batch_reports:
            value = row
            for key in key_path:
                value = value[key]
            total += float(value) * int(row["active_frames"])
        return total / float(active_frames)

    cosine_min = min(float(row["map_delta"]["cosine_min"]) for row in batch_reports)
    peak_gap_max = max(float(row["peak_agreement"]["peak_angle_gap_deg"]["max"]) for row in batch_reports)
    ref_peak_max = max(float(row["frontend_accuracy"]["time_reference_peak_error_deg"]["max"]) for row in batch_reports)
    freq_peak_max = max(float(row["frontend_accuracy"]["frequency_block_peak_error_deg"]["max"]) for row in batch_reports)

    ref_mean_time = float(np.mean(ref_times_ms)) if ref_times_ms else 0.0
    freq_mean_time = float(np.mean(freq_times_ms)) if freq_times_ms else 0.0
    speedup = ref_mean_time / freq_mean_time if ref_mean_time > 0.0 and freq_mean_time > 0.0 else 0.0

    return {
        "active_frames": int(active_frames),
        "timing_ms": {
            "time_reference_mean": ref_mean_time,
            "frequency_block_mean": freq_mean_time,
            "speedup_vs_reference": speedup,
        },
        "map_delta": {
            "mae": weighted_mean(("map_delta", "mae")),
            "rmse": weighted_mean(("map_delta", "rmse")),
            "cosine_mean": weighted_mean(("map_delta", "cosine_mean")),
            "cosine_min": cosine_min,
        },
        "peak_agreement": {
            "exact_ratio": weighted_mean(("peak_agreement", "exact_ratio")),
            "peak_angle_gap_deg": {
                "mean": weighted_mean(("peak_agreement", "peak_angle_gap_deg", "mean")),
                "median": weighted_mean(("peak_agreement", "peak_angle_gap_deg", "median")),
                "max": peak_gap_max,
            },
        },
        "frontend_accuracy": {
            "time_reference_peak_error_deg": {
                "mean": weighted_mean(("frontend_accuracy", "time_reference_peak_error_deg", "mean")),
                "median": weighted_mean(("frontend_accuracy", "time_reference_peak_error_deg", "median")),
                "max": ref_peak_max,
            },
            "frequency_block_peak_error_deg": {
                "mean": weighted_mean(("frontend_accuracy", "frequency_block_peak_error_deg", "mean")),
                "median": weighted_mean(("frontend_accuracy", "frequency_block_peak_error_deg", "median")),
                "max": freq_peak_max,
            },
            "frequency_minus_reference_mean_deg": weighted_mean(
                ("frontend_accuracy", "frequency_minus_reference_mean_deg")
            ),
        },
    }


def default_output_path(config: IFANTrainingConfig, mode: str, scenario: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = scenario if scenario is not None else mode
    return Path(config.output_root) / "analysis" / f"lms_backend_compare_{tag}_{stamp}.json"


def markdown_summary(result: dict[str, Any]) -> str:
    config = result["config"]
    lines = [
        "# LMS Backend Frontend Comparison",
        "",
        f"- config: `{config['config_path']}`",
        f"- device: `{config['device']}`",
        f"- trajectory_seconds: `{config['trajectory_seconds']}`",
        f"- lms_map_mode: `{config['lms_map_mode']}`",
        f"- lms_update_mode: `{config['lms_update_mode']}`",
        "",
    ]

    if result["dataset"]["mode"] == "scenario_suite":
        lines.extend(
            [
                "## Scenario Suite",
                "",
                "| Scenario | Active Frames | Cosine Mean | Exact Peak Ratio | Freq-Ref Error Delta (deg) | Speedup |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["scenario_reports"]:
            summary = row["summary"]
            lines.append(
                f"| {row['dataset']['scenario']['name']} | {summary['active_frames']} | "
                f"{summary['map_delta']['cosine_mean']:.6f} | {summary['peak_agreement']['exact_ratio']:.4f} | "
                f"{summary['frontend_accuracy']['frequency_minus_reference_mean_deg']:+.6f} | "
                f"{summary['timing_ms']['speedup_vs_reference']:.2f}x |"
            )
        lines.extend(
            [
                "",
                "## Aggregate",
                "",
            ]
        )
        aggregate = result["aggregate_summary"]
    else:
        aggregate = result["summary"]
        lines.extend(
            [
                f"## {result['dataset']['mode'].replace('_', ' ').title()} Summary",
                "",
            ]
        )

    lines.extend(
        [
            f"- active_frames: `{aggregate['active_frames']}`",
            f"- map cosine mean: `{aggregate['map_delta']['cosine_mean']:.6f}`",
            f"- map cosine min: `{aggregate['map_delta']['cosine_min']:.6f}`",
            f"- exact peak ratio: `{aggregate['peak_agreement']['exact_ratio']:.6f}`",
            f"- time_reference mean frontend time: `{aggregate['timing_ms']['time_reference_mean']:.3f} ms`",
            f"- frequency_block mean frontend time: `{aggregate['timing_ms']['frequency_block_mean']:.3f} ms`",
            f"- speedup vs reference: `{aggregate['timing_ms']['speedup_vs_reference']:.2f}x`",
            f"- frequency minus reference mean peak error: `{aggregate['frontend_accuracy']['frequency_minus_reference_mean_deg']:+.6f} deg`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_single_compare(
    *,
    config: IFANTrainingConfig,
    device: torch.device,
    mode: str,
    scenario: str,
    size: int,
    batch_size: int,
) -> dict[str, Any]:
    dataset, source_path, dataset_meta = build_dataset(
        config,
        mode=mode,
        scenario=scenario,
        size=size,
    )

    reference = build_lms_frontend(config, backend="time_reference", device=device)
    frequency = build_lms_frontend(config, backend="frequency_block", device=device)
    grid_flat = grid_tensor(config.r, device)

    batch_reports: list[dict[str, Any]] = []
    ref_times_ms: list[float] = []
    freq_times_ms: list[float] = []

    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        mic_sig_batch, acoustic_scene_batch = dataset.get_batch(start, stop)
        mic_sig_batch = ensure_mic_tensor(mic_sig_batch).float()
        if device.type == "cuda":
            mic_sig_batch = mic_sig_batch.to(device)
        doa_batch = build_doa_tensor(acoustic_scene_batch).to(device)
        vad_mask = build_vad_mask(acoustic_scene_batch).to(device)

        ref_maps, ref_ms = time_frontend(reference, mic_sig_batch)
        freq_maps, freq_ms = time_frontend(frequency, mic_sig_batch)
        ref_times_ms.append(ref_ms)
        freq_times_ms.append(freq_ms)

        report = compare_batch(
            ref_maps=ref_maps,
            freq_maps=freq_maps,
            doa_batch=doa_batch,
            vad_mask=vad_mask,
            grid_flat=grid_flat,
        )
        report["batch_range"] = [int(start), int(stop)]
        batch_reports.append(report)

    summary = aggregate_batch_reports(batch_reports, ref_times_ms, freq_times_ms)
    return {
        "dataset": {
            "source_path": str(source_path),
            "size": int(size),
            "batch_size": batch_size,
            **dataset_meta,
        },
        "summary": summary,
        "batch_reports": batch_reports,
    }


def aggregate_suite_reports(scenario_reports: list[dict[str, Any]]) -> dict[str, Any]:
    total_active_frames = sum(int(row["summary"]["active_frames"]) for row in scenario_reports)
    if total_active_frames <= 0:
        return {
            "active_frames": 0,
            "timing_ms": {
                "time_reference_mean": 0.0,
                "frequency_block_mean": 0.0,
                "speedup_vs_reference": 0.0,
            },
            "map_delta": {"mae": 0.0, "rmse": 0.0, "cosine_mean": 0.0, "cosine_min": 0.0},
            "peak_agreement": {"exact_ratio": 0.0, "peak_angle_gap_deg": {"mean": 0.0, "median": 0.0, "max": 0.0}},
            "frontend_accuracy": {
                "time_reference_peak_error_deg": {"mean": 0.0, "median": 0.0, "max": 0.0},
                "frequency_block_peak_error_deg": {"mean": 0.0, "median": 0.0, "max": 0.0},
                "frequency_minus_reference_mean_deg": 0.0,
            },
        }

    def weighted(path: tuple[str, ...]) -> float:
        total = 0.0
        for row in scenario_reports:
            value = row["summary"]
            for key in path:
                value = value[key]
            total += float(value) * int(row["summary"]["active_frames"])
        return total / float(total_active_frames)

    time_ref_mean = float(np.mean([row["summary"]["timing_ms"]["time_reference_mean"] for row in scenario_reports]))
    freq_mean = float(np.mean([row["summary"]["timing_ms"]["frequency_block_mean"] for row in scenario_reports]))
    speedup = time_ref_mean / freq_mean if time_ref_mean > 0.0 and freq_mean > 0.0 else 0.0

    return {
        "active_frames": int(total_active_frames),
        "timing_ms": {
            "time_reference_mean": time_ref_mean,
            "frequency_block_mean": freq_mean,
            "speedup_vs_reference": speedup,
        },
        "map_delta": {
            "mae": weighted(("map_delta", "mae")),
            "rmse": weighted(("map_delta", "rmse")),
            "cosine_mean": weighted(("map_delta", "cosine_mean")),
            "cosine_min": min(float(row["summary"]["map_delta"]["cosine_min"]) for row in scenario_reports),
        },
        "peak_agreement": {
            "exact_ratio": weighted(("peak_agreement", "exact_ratio")),
            "peak_angle_gap_deg": {
                "mean": weighted(("peak_agreement", "peak_angle_gap_deg", "mean")),
                "median": weighted(("peak_agreement", "peak_angle_gap_deg", "median")),
                "max": max(float(row["summary"]["peak_agreement"]["peak_angle_gap_deg"]["max"]) for row in scenario_reports),
            },
        },
        "frontend_accuracy": {
            "time_reference_peak_error_deg": {
                "mean": weighted(("frontend_accuracy", "time_reference_peak_error_deg", "mean")),
                "median": weighted(("frontend_accuracy", "time_reference_peak_error_deg", "median")),
                "max": max(float(row["summary"]["frontend_accuracy"]["time_reference_peak_error_deg"]["max"]) for row in scenario_reports),
            },
            "frequency_block_peak_error_deg": {
                "mean": weighted(("frontend_accuracy", "frequency_block_peak_error_deg", "mean")),
                "median": weighted(("frontend_accuracy", "frequency_block_peak_error_deg", "median")),
                "max": max(float(row["summary"]["frontend_accuracy"]["frequency_block_peak_error_deg"]["max"]) for row in scenario_reports),
            },
            "frequency_minus_reference_mean_deg": weighted(("frontend_accuracy", "frequency_minus_reference_mean_deg")),
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    config = IFANTrainingConfig.from_toml(args.config)
    if args.device is not None:
        config.device = args.device
    if args.trajectory_seconds is not None:
        config.trajectory_seconds = int(args.trajectory_seconds)
    if args.lms_block_size is not None:
        config.lms_block_size = int(args.lms_block_size)
    if args.lms_fft_size is not None:
        config.lms_fft_size = int(args.lms_fft_size)

    pipeline = IFANTrainingPipeline(config)
    device = pipeline.resolve_device()
    batch_size = max(int(args.batch_size), 1)
    config_payload = {
        "config_path": str(Path(args.config).resolve()),
        "device": str(device),
        "trajectory_seconds": int(config.trajectory_seconds),
        "k": int(config.k),
        "step": int(config.step),
        "apply_vad": bool(config.apply_vad),
        "lms_order": int(config.lms_order),
        "lms_step_size": float(config.lms_step_size),
        "lms_map_mode": str(config.lms_map_mode),
        "lms_update_mode": str(config.lms_update_mode),
        "lms_normalized": bool(config.lms_normalized),
        "lms_map_normalize": bool(config.lms_map_normalize),
        "lms_include_self_pairs": bool(config.lms_include_self_pairs),
        "lms_block_size": int(config.lms_block_size),
        "lms_fft_size": None if config.lms_fft_size is None else int(config.lms_fft_size),
    }

    if args.mode == "scenario_suite":
        scenario_reports = [
            {
                "scenario_name": row["name"],
                **run_single_compare(
                    config=config,
                    device=device,
                    mode="scenario",
                    scenario=row["name"],
                    size=args.size,
                    batch_size=batch_size,
                ),
            }
            for row in STAGE3_SCENARIOS
        ]
        result = {
            "kind": "stage3_lms_backend_frontend_compare_suite",
            "config": config_payload,
            "dataset": {
                "mode": "scenario_suite",
                "size": int(args.size),
                "batch_size": batch_size,
                "scenario_count": len(scenario_reports),
            },
            "aggregate_summary": aggregate_suite_reports(scenario_reports),
            "scenario_reports": scenario_reports,
        }
    else:
        single = run_single_compare(
            config=config,
            device=device,
            mode=args.mode,
            scenario=args.scenario,
            size=args.size,
            batch_size=batch_size,
        )
        result = {
            "kind": "stage3_lms_backend_frontend_compare",
            "config": config_payload,
            "dataset": single["dataset"],
            "summary": single["summary"],
            "batch_reports": single["batch_reports"],
        }

    output_path = Path(args.output) if args.output is not None else default_output_path(
        config,
        mode=args.mode,
        scenario=args.scenario if args.mode == "scenario" else None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(markdown_summary(result), encoding="utf-8")

    summary = result["aggregate_summary"] if args.mode == "scenario_suite" else result["summary"]
    print(json.dumps({"output_path": str(output_path), "markdown_path": str(markdown_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
