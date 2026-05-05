from __future__ import annotations

import argparse
import json
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
from ifan_edge.features import DualFeatureIcoPreprocessor, SRPPHATIcoMapAdapter
from ifan_edge.features.phat import ensure_mic_tensor
from ifan_edge.training import IFANTrainingConfig, IFANTrainingPipeline
from utils import sph2cart


DEFAULT_VARIANTS = ("paper_original", "lc_reference", "lc_edge")
DEFAULT_COMPARISONS = (("paper_original", "lc_reference"), ("lc_reference", "lc_edge"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare stage-3 PHAT frontend variants and export HLS-friendly cache metadata.")
    parser.add_argument("--config", default="IFAN_Edge/configs/stage3_default.toml")
    parser.add_argument("--mode", choices=("scenario", "validation", "scenario_suite"), default="scenario")
    parser.add_argument("--scenario", choices=[row["name"] for row in STAGE3_SCENARIOS], default="scene_3")
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3, help="How many PHAT-only timing repeats to run per batch.")
    parser.add_argument("--trajectory-seconds", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--lms-backend", choices=("time_reference", "frequency_block"), default=None, help="Optional fallback backend for the dual-preprocessor smoke path.")
    parser.add_argument("--phat-sinc-half-width", type=int, default=None, help="Override the sinc half-width used by lc_reference/lc_edge.")
    parser.add_argument("--variant", action="append", choices=DEFAULT_VARIANTS, default=None, help="Limit the comparison to specific PHAT variants.")
    parser.add_argument("--output", default=None)
    return parser


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
        return dataset, source_path, {"mode": mode, "scenario": dict(scenario_row)}

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


def build_phat_frontend(config: IFANTrainingConfig, *, variant: str, device: torch.device) -> SRPPHATIcoMapAdapter:
    frontend = SRPPHATIcoMapAdapter(
        N=at_dataset.benchmark2_array_setup.mic_pos.shape[0],
        K=config.k,
        r=config.r,
        rn=at_dataset.benchmark2_array_setup.mic_pos,
        fs=config.fs,
        srp_variant=variant,
        sinc_half_width=config.phat_sinc_half_width,
    )
    if device.type == "cuda":
        frontend.cuda()
    else:
        frontend.cpu()
    return frontend


def build_dual_preprocessor(config: IFANTrainingConfig, *, variant: str, device: torch.device) -> DualFeatureIcoPreprocessor:
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
        srp_variant=variant,
        phat_sinc_half_width=config.phat_sinc_half_width,
    )
    IFANTrainingPipeline.move_ifan_preprocessor(preprocessor, device)
    return preprocessor


def time_phat_frontend(frontend: SRPPHATIcoMapAdapter, mic_sig_batch: torch.Tensor, repeats: int) -> tuple[torch.Tensor, float]:
    maps = None
    timings_ms: list[float] = []
    for _ in range(max(int(repeats), 1)):
        start = time.perf_counter()
        with torch.no_grad():
            maps = frontend(mic_sig_batch)
        if maps.is_cuda:
            torch.cuda.synchronize(maps.device)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    assert maps is not None
    return maps, float(np.mean(timings_ms))


def time_dual_preprocessor(
    preprocessor: DualFeatureIcoPreprocessor,
    mic_sig_batch: torch.Tensor,
    acoustic_scene_batch,
) -> tuple[torch.Tensor, float]:
    start = time.perf_counter()
    with torch.no_grad():
        maps, _ = preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
    if maps.is_cuda:
        torch.cuda.synchronize(maps.device)
    return maps, (time.perf_counter() - start) * 1000.0


def masked_flatten(maps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat = maps.reshape(maps.shape[0], maps.shape[2], -1)
    return flat[mask].float()


def grid_tensor(r: int, device: torch.device) -> torch.Tensor:
    grid = torch.from_numpy(icoCNN.icosahedral_grid_coordinates(r)).float().reshape(-1, 3)
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


def compare_variant_maps(
    *,
    reference_maps: torch.Tensor,
    candidate_maps: torch.Tensor,
    doa_batch: torch.Tensor,
    vad_mask: torch.Tensor,
    grid_flat: torch.Tensor,
) -> dict[str, Any]:
    ref_flat = masked_flatten(reference_maps, vad_mask)
    cand_flat = masked_flatten(candidate_maps, vad_mask)
    if doa_batch.ndim == 4 and doa_batch.shape[1] == 1:
        doa_batch = doa_batch[:, 0, ...]
    doa_active = doa_batch[vad_mask].to(ref_flat.device)

    if ref_flat.numel() == 0:
        return {
            "active_frames": 0,
            "map_delta": {"mae": 0.0, "rmse": 0.0, "cosine_mean": 0.0, "cosine_min": 0.0},
            "peak_agreement": {"exact_ratio": 0.0, "within_one_ratio": 0.0, "peak_angle_gap_deg": {"mean": 0.0, "median": 0.0, "max": 0.0}},
            "frontend_accuracy": {
                "reference_peak_error_deg": {"mean": 0.0, "median": 0.0, "max": 0.0},
                "candidate_peak_error_deg": {"mean": 0.0, "median": 0.0, "max": 0.0},
                "candidate_minus_reference_mean_deg": 0.0,
            },
        }

    diff = cand_flat - ref_flat
    cosine = torch.nn.functional.cosine_similarity(ref_flat, cand_flat, dim=-1)
    ref_peak = ref_flat.argmax(dim=-1)
    cand_peak = cand_flat.argmax(dim=-1)
    ref_peak_cart = grid_flat[ref_peak]
    cand_peak_cart = grid_flat[cand_peak]
    doa_cart = sph2cart(doa_active)

    ref_peak_error = angular_distance_deg(ref_peak_cart, doa_cart)
    cand_peak_error = angular_distance_deg(cand_peak_cart, doa_cart)
    peak_gap = angular_distance_deg(ref_peak_cart, cand_peak_cart)

    return {
        "active_frames": int(ref_flat.shape[0]),
        "map_delta": {
            "mae": float(diff.abs().mean().item()),
            "rmse": float(torch.sqrt(diff.square().mean()).item()),
            "cosine_mean": float(cosine.mean().item()),
            "cosine_min": float(cosine.min().item()),
        },
        "peak_agreement": {
            "exact_ratio": float((ref_peak == cand_peak).float().mean().item()),
            "within_one_ratio": float(((ref_peak - cand_peak).abs() <= 1).float().mean().item()),
            "peak_angle_gap_deg": summarize_tensor(peak_gap),
        },
        "frontend_accuracy": {
            "reference_peak_error_deg": summarize_tensor(ref_peak_error),
            "candidate_peak_error_deg": summarize_tensor(cand_peak_error),
            "candidate_minus_reference_mean_deg": float(cand_peak_error.mean().item() - ref_peak_error.mean().item()),
        },
    }


def aggregate_reports(batch_reports: list[dict[str, Any]]) -> dict[str, Any]:
    active_frames = sum(int(row["active_frames"]) for row in batch_reports)
    if active_frames <= 0:
        return {
            "active_frames": 0,
            "map_delta": {"mae": 0.0, "rmse": 0.0, "cosine_mean": 0.0, "cosine_min": 0.0},
            "peak_agreement": {"exact_ratio": 0.0, "within_one_ratio": 0.0, "peak_angle_gap_deg": {"mean": 0.0, "median": 0.0, "max": 0.0}},
            "frontend_accuracy": {
                "reference_peak_error_deg": {"mean": 0.0, "median": 0.0, "max": 0.0},
                "candidate_peak_error_deg": {"mean": 0.0, "median": 0.0, "max": 0.0},
                "candidate_minus_reference_mean_deg": 0.0,
            },
        }

    def weighted(path: tuple[str, ...]) -> float:
        total = 0.0
        for row in batch_reports:
            value = row
            for key in path:
                value = value[key]
            total += float(value) * int(row["active_frames"])
        return total / float(active_frames)

    return {
        "active_frames": int(active_frames),
        "map_delta": {
            "mae": weighted(("map_delta", "mae")),
            "rmse": weighted(("map_delta", "rmse")),
            "cosine_mean": weighted(("map_delta", "cosine_mean")),
            "cosine_min": min(float(row["map_delta"]["cosine_min"]) for row in batch_reports),
        },
        "peak_agreement": {
            "exact_ratio": weighted(("peak_agreement", "exact_ratio")),
            "within_one_ratio": weighted(("peak_agreement", "within_one_ratio")),
            "peak_angle_gap_deg": {
                "mean": weighted(("peak_agreement", "peak_angle_gap_deg", "mean")),
                "median": weighted(("peak_agreement", "peak_angle_gap_deg", "median")),
                "max": max(float(row["peak_agreement"]["peak_angle_gap_deg"]["max"]) for row in batch_reports),
            },
        },
        "frontend_accuracy": {
            "reference_peak_error_deg": {
                "mean": weighted(("frontend_accuracy", "reference_peak_error_deg", "mean")),
                "median": weighted(("frontend_accuracy", "reference_peak_error_deg", "median")),
                "max": max(float(row["frontend_accuracy"]["reference_peak_error_deg"]["max"]) for row in batch_reports),
            },
            "candidate_peak_error_deg": {
                "mean": weighted(("frontend_accuracy", "candidate_peak_error_deg", "mean")),
                "median": weighted(("frontend_accuracy", "candidate_peak_error_deg", "median")),
                "max": max(float(row["frontend_accuracy"]["candidate_peak_error_deg"]["max"]) for row in batch_reports),
            },
            "candidate_minus_reference_mean_deg": weighted(("frontend_accuracy", "candidate_minus_reference_mean_deg")),
        },
    }


def build_complexity_comparison(variant_profiles: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    for left, right in DEFAULT_COMPARISONS:
        if left not in variant_profiles or right not in variant_profiles:
            continue
        left_proxy = variant_profiles[left]["complexity_proxy"]
        right_proxy = variant_profiles[right]["complexity_proxy"]
        rows: dict[str, Any] = {}
        for key in (
            "sample_reads_per_frame",
            "weight_multiplies_per_frame",
            "pair_reduction_adds_per_frame",
            "grid_accumulation_adds_per_frame",
            "symmetry_multiplies_per_frame",
            "total_arithmetic_ops_per_frame",
        ):
            left_value = int(left_proxy[key])
            right_value = int(right_proxy[key])
            rows[key] = {
                left: left_value,
                right: right_value,
                "delta": right_value - left_value,
                "ratio": (float(right_value) / float(left_value)) if left_value > 0 else None,
                "reduction_percent": (100.0 * (float(left_value - right_value) / float(left_value))) if left_value > 0 else None,
            }
        comparisons[f"{left}_vs_{right}"] = rows
    return comparisons


def default_output_path(config: IFANTrainingConfig, mode: str, scenario: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = scenario if scenario is not None else mode
    return Path(config.output_root) / "analysis" / f"phat_variant_compare_{tag}_{stamp}.json"


def markdown_summary(result: dict[str, Any]) -> str:
    config = result["config"]
    lines = [
        "# PHAT Variant Frontend Comparison",
        "",
        f"- config: `{config['config_path']}`",
        f"- device: `{config['device']}`",
        f"- trajectory_seconds: `{config['trajectory_seconds']}`",
        f"- srp_variants: `{', '.join(result['variants'])}`",
        f"- lms_backend context: `{config['lms_backend']}`",
        "",
    ]

    if result["kind"] == "stage3_phat_variant_compare_suite":
        lines.extend(
            [
                "## Scenario Suite",
                "",
                "| Scenario | Comparison | Cosine Mean | Within-One Ratio | Candidate-Ref Mean Error (deg) |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for scenario_report in result["scenario_reports"]:
            scenario_name = scenario_report["scenario_name"]
            for name, summary in scenario_report["comparisons"].items():
                lines.append(
                    f"| {scenario_name} | {name} | {summary['map_delta']['cosine_mean']:.6f} | "
                    f"{summary['peak_agreement']['within_one_ratio']:.6f} | "
                    f"{summary['frontend_accuracy']['candidate_minus_reference_mean_deg']:+.6f} |"
                )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Variant Profiles",
            "",
            "| Variant | Pairs | Full Pairs | Taps | Cache Bytes | Sample Reads / Frame | Arithmetic Ops / Frame | PHAT Mean (ms) | Dual Mean (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in result["variants"]:
        profile = result["variant_profiles"][variant]
        complexity = profile["complexity_proxy"]
        timing = result["variant_timing"][variant]
        lines.append(
            f"| {variant} | {profile['pair_count']} | {profile['full_pair_count']} | {profile['interpolation_taps']} | "
            f"{profile['cache_table_bytes']} | {complexity['sample_reads_per_frame']} | {complexity['total_arithmetic_ops_per_frame']} | "
            f"{timing['phat_mean_ms']:.3f} | {timing['dual_mean_ms']:.3f} |"
        )

    lines.extend(["", "## Pairwise Summary", ""])
    for name, summary in result["comparisons"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- active_frames: `{summary['active_frames']}`",
                f"- cosine_mean: `{summary['map_delta']['cosine_mean']:.6f}`",
                f"- cosine_min: `{summary['map_delta']['cosine_min']:.6f}`",
                f"- exact_peak_ratio: `{summary['peak_agreement']['exact_ratio']:.6f}`",
                f"- within_one_ratio: `{summary['peak_agreement']['within_one_ratio']:.6f}`",
                f"- candidate_minus_reference_mean_deg: `{summary['frontend_accuracy']['candidate_minus_reference_mean_deg']:+.6f}`",
                "",
            ]
        )
    if result.get("complexity_comparisons"):
        lines.extend(["## Complexity Proxy", ""])
        for name, rows in result["complexity_comparisons"].items():
            total = rows["total_arithmetic_ops_per_frame"]
            reads = rows["sample_reads_per_frame"]
            lines.extend(
                [
                    f"### {name}",
                    "",
                    f"- sample_reads_per_frame: `{reads}`",
                    f"  ratio(candidate/reference): `{reads['ratio'] if reads['ratio'] is not None else 'n/a'}`",
                    f"  reduction_percent: `{reads['reduction_percent'] if reads['reduction_percent'] is not None else 'n/a'}`",
                    f"- total_arithmetic_ops_per_frame: `{total}`",
                    f"  ratio(candidate/reference): `{total['ratio'] if total['ratio'] is not None else 'n/a'}`",
                    f"  reduction_percent: `{total['reduction_percent'] if total['reduction_percent'] is not None else 'n/a'}`",
                    "",
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
    repeats: int,
    variants: tuple[str, ...],
) -> dict[str, Any]:
    dataset, source_path, dataset_meta = build_dataset(config, mode=mode, scenario=scenario, size=size)
    frontends = {variant: build_phat_frontend(config, variant=variant, device=device) for variant in variants}
    preprocessors = {variant: build_dual_preprocessor(config, variant=variant, device=device) for variant in variants}
    grid_flat = grid_tensor(config.r, device)

    timing_rows = {variant: {"phat_ms": [], "dual_ms": []} for variant in variants}
    batch_reports = {f"{left}_vs_{right}": [] for left, right in DEFAULT_COMPARISONS if left in variants and right in variants}

    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        mic_sig_batch, acoustic_scene_batch = dataset.get_batch(start, stop)
        mic_sig_batch = ensure_mic_tensor(mic_sig_batch).float()
        if device.type == "cuda":
            mic_sig_batch = mic_sig_batch.to(device)
        doa_batch = build_doa_tensor(acoustic_scene_batch).to(device)
        vad_mask = build_vad_mask(acoustic_scene_batch).to(device)

        phat_maps: dict[str, torch.Tensor] = {}
        for variant in variants:
            phat_maps[variant], phat_ms = time_phat_frontend(frontends[variant], mic_sig_batch, repeats)
            _, dual_ms = time_dual_preprocessor(preprocessors[variant], mic_sig_batch, acoustic_scene_batch)
            timing_rows[variant]["phat_ms"].append(phat_ms)
            timing_rows[variant]["dual_ms"].append(dual_ms)

        for left, right in DEFAULT_COMPARISONS:
            if left not in variants or right not in variants:
                continue
            name = f"{left}_vs_{right}"
            report = compare_variant_maps(
                reference_maps=phat_maps[left],
                candidate_maps=phat_maps[right],
                doa_batch=doa_batch,
                vad_mask=vad_mask,
                grid_flat=grid_flat,
            )
            report["batch_range"] = [int(start), int(stop)]
            batch_reports[name].append(report)

    variant_profiles = {variant: frontends[variant].frontend_profile() for variant in variants}
    variant_timing = {
        variant: {
            "phat_mean_ms": float(np.mean(rows["phat_ms"])) if rows["phat_ms"] else 0.0,
            "dual_mean_ms": float(np.mean(rows["dual_ms"])) if rows["dual_ms"] else 0.0,
        }
        for variant, rows in timing_rows.items()
    }
    comparisons = {name: aggregate_reports(rows) for name, rows in batch_reports.items()}
    complexity_comparisons = build_complexity_comparison(variant_profiles)

    return {
        "dataset": {"source_path": str(source_path), "size": int(size), "batch_size": int(batch_size), **dataset_meta},
        "variants": list(variants),
        "variant_profiles": variant_profiles,
        "variant_timing": variant_timing,
        "comparisons": comparisons,
        "complexity_comparisons": complexity_comparisons,
        "batch_reports": batch_reports,
    }


def main() -> None:
    args = build_parser().parse_args()
    config = IFANTrainingConfig.from_toml(args.config)
    if args.device is not None:
        config.device = args.device
    if args.trajectory_seconds is not None:
        config.trajectory_seconds = int(args.trajectory_seconds)
    if args.lms_backend is not None:
        config.lms_backend = args.lms_backend
    if args.phat_sinc_half_width is not None:
        config.phat_sinc_half_width = int(args.phat_sinc_half_width)

    pipeline = IFANTrainingPipeline(config)
    device = pipeline.resolve_device()
    variants = tuple(args.variant) if args.variant else DEFAULT_VARIANTS
    batch_size = max(int(args.batch_size), 1)

    config_payload = {
        "config_path": str(Path(args.config).resolve()),
        "device": str(device),
        "trajectory_seconds": int(config.trajectory_seconds),
        "k": int(config.k),
        "step": int(config.step),
        "r": int(config.r),
        "lms_backend": str(config.lms_backend),
        "phat_sinc_half_width": int(config.phat_sinc_half_width),
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
                    repeats=args.repeats,
                    variants=variants,
                ),
            }
            for row in STAGE3_SCENARIOS
        ]
        result = {
            "kind": "stage3_phat_variant_compare_suite",
            "config": config_payload,
            "dataset": {"mode": "scenario_suite", "size": int(args.size), "batch_size": batch_size, "scenario_count": len(scenario_reports)},
            "variants": list(variants),
            "scenario_reports": scenario_reports,
        }
    else:
        result = {
            "kind": "stage3_phat_variant_compare",
            "config": config_payload,
            **run_single_compare(
                config=config,
                device=device,
                mode=args.mode,
                scenario=args.scenario,
                size=args.size,
                batch_size=batch_size,
                repeats=args.repeats,
                variants=variants,
            ),
        }

    output_path = Path(args.output) if args.output is not None else default_output_path(
        config,
        args.mode,
        args.scenario if args.mode == "scenario" else None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(markdown_summary(result), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "markdown_path": str(markdown_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
