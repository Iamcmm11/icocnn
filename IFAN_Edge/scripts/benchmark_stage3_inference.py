from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_stage3_locata import (  # noqa: E402
    build_locata_dataset,
    extract_metadata_from_directory,
    load_config_from_checkpoint,
    remap_legacy_map_refiner_keys,
)
from ifan_edge.eval import (  # noqa: E402
    build_baseline_preprocessor_from_config,
    build_ifan_preprocessor_from_config,
    load_baseline_model_from_config,
    select_model_inputs,
)
from ifan_edge.models import IFANModel  # noqa: E402
from ifan_edge.training import IFANTrainingConfig, IFANTrainingPipeline  # noqa: E402
from profile_stage2_model import baseline_summary  # noqa: E402


DEFAULT_CHECKPOINT = (
    "IFAN_Edge/outputs/stage3/"
    "ifan_stage3_full20_freqblock_paper_original_20260419_005314/"
    "checkpoints/best_rmsae.pt"
)
DEFAULT_CONFIG = "IFAN_Edge/configs/stage3_default.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark stage-3 IFAN / baseline inference on CPU or GPU."
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to an IFAN stage-3 checkpoint (*.pt). Required for targets that include `ifan`.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional stage-3 config TOML. Uses checkpoint metadata when omitted for IFAN; "
        "uses stage3_default.toml for baseline-only runs when omitted.",
    )
    parser.add_argument(
        "--target",
        choices=("ifan", "baseline", "both"),
        default="both",
        help="Which model(s) to benchmark.",
    )
    parser.add_argument(
        "--mode",
        choices=("model_only", "end_to_end"),
        default="model_only",
        help="`model_only` benchmarks forward on precomputed feature tensors; "
        "`end_to_end` benchmarks preprocessor + model forward.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--locata-root", default="datasets/LOCATA/LOCATA")
    parser.add_argument("--subset", choices=("dev", "eval"), default="eval")
    parser.add_argument("--array", choices=("dummy", "eigenmike", "benchmark2", "dicit"), default="benchmark2")
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="LOCATA tasks to benchmark. Prefer single-source tasks 1, 3, and 5.",
    )
    parser.add_argument(
        "--recording",
        nargs="+",
        default=None,
        help="Optional specific recording names, e.g. recording1 recording5.",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=3,
        help="Maximum number of LOCATA recordings to benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations before timing.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Timed benchmark iterations.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. A Markdown summary with the same stem will also be written.",
    )
    return parser


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_seconds(values: list[float]) -> dict[str, float]:
    arr = sorted(float(value) for value in values)
    if not arr:
        return {
            "mean_s": 0.0,
            "median_s": 0.0,
            "std_s": 0.0,
            "min_s": 0.0,
            "max_s": 0.0,
            "p90_s": 0.0,
        }
    if len(arr) == 1:
        return {
            "mean_s": arr[0],
            "median_s": arr[0],
            "std_s": 0.0,
            "min_s": arr[0],
            "max_s": arr[0],
            "p90_s": arr[0],
        }
    p90_index = min(len(arr) - 1, int(0.9 * (len(arr) - 1)))
    return {
        "mean_s": float(statistics.mean(arr)),
        "median_s": float(statistics.median(arr)),
        "std_s": float(statistics.pstdev(arr)),
        "min_s": float(arr[0]),
        "max_s": float(arr[-1]),
        "p90_s": float(arr[p90_index]),
    }


def format_recording_list(dataset, indices: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in indices:
        metadata = extract_metadata_from_directory(dataset.directories[idx])
        rows.append(
            {
                "dataset_index": int(idx),
                "task": int(metadata["task"]),
                "recording": str(metadata["recording"]),
                "array": str(metadata["array"]),
                "directory": str(metadata["directory"]),
            }
        )
    return rows


def resolve_config_and_checkpoint(args: argparse.Namespace) -> tuple[dict[str, Any] | None, IFANTrainingConfig]:
    if args.target in ("ifan", "both"):
        checkpoint, config = load_config_from_checkpoint(args.checkpoint, args.config)
        return checkpoint, config

    config_path = args.config or DEFAULT_CONFIG
    config = IFANTrainingConfig.from_toml(config_path)
    return None, config


def select_indices(dataset, recording_filter: set[str] | None, max_recordings: int) -> list[int]:
    indices: list[int] = []
    for idx, directory in enumerate(dataset.directories):
        metadata = extract_metadata_from_directory(directory)
        if recording_filter is not None and metadata["recording"] not in recording_filter:
            continue
        indices.append(idx)
        if len(indices) >= max_recordings:
            break
    if not indices:
        raise RuntimeError("No LOCATA recordings matched the requested filter.")
    return indices


def preload_ifan_model_inputs(
    *,
    dataset,
    indices: list[int],
    preprocessor,
    model_config,
    input_ablation_mode: str,
) -> list[torch.Tensor]:
    inputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for idx in indices:
            mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx, idx + 1)
            maps, _ = preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
            tensor = select_model_inputs(maps, model_config, input_ablation_mode)
            inputs.append(tensor)
    return inputs


def preload_baseline_model_inputs(
    *,
    dataset,
    indices: list[int],
    preprocessor,
) -> list[torch.Tensor]:
    inputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for idx in indices:
            mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx, idx + 1)
            maps, _ = preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
            inputs.append(maps)
    return inputs


def benchmark_callable(
    *,
    runner: Callable[[], None],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            runner()
            sync_if_needed(device)

        elapsed: list[float] = []
        for _ in range(repeats):
            sync_if_needed(device)
            start = time.perf_counter()
            runner()
            sync_if_needed(device)
            elapsed.append(time.perf_counter() - start)
    return elapsed


def per_sample_metrics(
    total_seconds_summary: dict[str, float],
    *,
    sample_count: int,
    mac_proxy_total: int | None,
) -> dict[str, Any]:
    sample_count = max(1, int(sample_count))
    mean_s = float(total_seconds_summary["mean_s"]) / float(sample_count)
    median_s = float(total_seconds_summary["median_s"]) / float(sample_count)
    fps = 0.0 if mean_s <= 0.0 else 1.0 / mean_s
    payload: dict[str, Any] = {
        "latency_ms_per_sample": {
            "mean": mean_s * 1000.0,
            "median": median_s * 1000.0,
            "p90": float(total_seconds_summary["p90_s"]) / float(sample_count) * 1000.0,
            "min": float(total_seconds_summary["min_s"]) / float(sample_count) * 1000.0,
            "max": float(total_seconds_summary["max_s"]) / float(sample_count) * 1000.0,
            "std": float(total_seconds_summary["std_s"]) / float(sample_count) * 1000.0,
        },
        "fps": fps,
    }
    if mac_proxy_total is not None and mean_s > 0.0:
        gmac_s = float(mac_proxy_total) / mean_s / 1e9
        payload["throughput"] = {
            "mac_per_sample": int(mac_proxy_total),
            "gmac_s": gmac_s,
            "gops_assuming_1mac_eq_2ops": 2.0 * gmac_s,
        }
    return payload


def run_ifan_benchmark(
    *,
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    config,
    pipeline: IFANTrainingPipeline,
    device: torch.device,
    dataset,
    indices: list[int],
) -> dict[str, Any]:
    model = IFANModel(pipeline.model_config)
    state_dict = remap_legacy_map_refiner_keys(
        checkpoint["model_state_dict"],
        model_config=pipeline.model_config,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    preprocessor = build_ifan_preprocessor_from_config(config, device)
    model_profile = pipeline.build_model_profile(model)
    recording_rows = format_recording_list(dataset, indices)

    if args.mode == "model_only":
        cached_inputs = preload_ifan_model_inputs(
            dataset=dataset,
            indices=indices,
            preprocessor=preprocessor,
            model_config=pipeline.model_config,
            input_ablation_mode=config.input_ablation_mode,
        )

        def runner() -> None:
            for tensor in cached_inputs:
                _ = model(tensor)

    else:
        raw_batches = [dataset.get_batch(idx, idx + 1) for idx in indices]

        def runner() -> None:
            for mic_sig_batch, acoustic_scene_batch in raw_batches:
                maps, _ = preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
                inputs = select_model_inputs(maps, pipeline.model_config, config.input_ablation_mode)
                _ = model(inputs)

    elapsed = benchmark_callable(
        runner=runner,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    summary = summarize_seconds(elapsed)
    return {
        "target": "ifan",
        "mode": args.mode,
        "recordings": recording_rows,
        "timing_total_per_iteration_s": summary,
        "metrics": per_sample_metrics(
            summary,
            sample_count=len(indices),
            mac_proxy_total=int(model_profile["mac_proxy_total"]),
        ),
        "model_profile": model_profile,
    }


def run_baseline_benchmark(
    *,
    args: argparse.Namespace,
    config,
    pipeline: IFANTrainingPipeline,
    device: torch.device,
    dataset,
    indices: list[int],
) -> dict[str, Any]:
    model = load_baseline_model_from_config(config, device)
    model.eval()
    preprocessor = build_baseline_preprocessor_from_config(config, device)
    baseline_profile = baseline_summary()
    recording_rows = format_recording_list(dataset, indices)

    if args.mode == "model_only":
        cached_inputs = preload_baseline_model_inputs(
            dataset=dataset,
            indices=indices,
            preprocessor=preprocessor,
        )

        def runner() -> None:
            for tensor in cached_inputs:
                _ = model(tensor)

    else:
        raw_batches = [dataset.get_batch(idx, idx + 1) for idx in indices]

        def runner() -> None:
            for mic_sig_batch, acoustic_scene_batch in raw_batches:
                maps, _ = preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
                _ = model(maps)

    elapsed = benchmark_callable(
        runner=runner,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    summary = summarize_seconds(elapsed)
    return {
        "target": "baseline",
        "mode": args.mode,
        "recordings": recording_rows,
        "timing_total_per_iteration_s": summary,
        "metrics": per_sample_metrics(
            summary,
            sample_count=len(indices),
            mac_proxy_total=int(baseline_profile["paper_style_complexity"]["mac_proxy_total"]),
        ),
        "model_profile": baseline_profile,
    }


def markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Stage-3 Inference Benchmark",
        "",
        f"- Device requested: `{report['device_requested']}`",
        f"- Device resolved: `{report['device_resolved']}`",
        f"- Mode: `{report['mode']}`",
        f"- LOCATA subset/array/tasks: `{report['subset']}` / `{report['array']}` / `{', '.join(f'task{task}' for task in report['tasks'])}`",
        f"- Recording count: `{report['recording_count']}`",
        f"- Warmup / repeats: `{report['warmup']}` / `{report['repeats']}`",
        "",
    ]
    for target_name, payload in report["targets"].items():
        metrics = payload["metrics"]
        latency = metrics["latency_ms_per_sample"]
        lines.extend(
            [
                f"## {target_name}",
                "",
                f"- Mean latency per sample: `{latency['mean']:.3f} ms`",
                f"- Median latency per sample: `{latency['median']:.3f} ms`",
                f"- P90 latency per sample: `{latency['p90']:.3f} ms`",
                f"- FPS: `{metrics['fps']:.3f}`",
            ]
        )
        if "throughput" in metrics:
            throughput = metrics["throughput"]
            lines.extend(
                [
                    f"- MAC per sample: `{throughput['mac_per_sample']}`",
                    f"- Throughput: `{throughput['gmac_s']:.3f} GMAC/s`",
                    f"- Throughput (1 MAC = 2 ops): `{throughput['gops_assuming_1mac_eq_2ops']:.3f} GOPS`",
                ]
            )
        lines.extend(
            [
                "",
                "Recordings:",
            ]
        )
        for row in payload["recordings"]:
            lines.append(
                f"- task{row['task']} / {row['recording']} / {row['array']} (dataset idx={row['dataset_index']})"
            )
        lines.append("")
    lines.extend(
        [
            "## Note",
            "",
            "- `model_only` measures forward latency on precomputed feature tensors.",
            "- `end_to_end` measures preprocessor + model forward latency on the selected LOCATA recordings.",
            "- The throughput estimate is a software-side proxy derived from model MAC counts and measured latency. It is useful for CPU/GPU/FPGA discussion, but it is not a hardware-verified full-system throughput number.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    checkpoint, config = resolve_config_and_checkpoint(args)
    config.device = str(args.device)

    pipeline = IFANTrainingPipeline(config)
    device = pipeline.resolve_device()
    recording_filter = None if not args.recording else {str(value) for value in args.recording}
    dataset, subset_root = build_locata_dataset(
        root=args.locata_root,
        subset=args.subset,
        array=args.array,
        fs=config.fs,
        k=config.k,
        step=config.step,
        tasks=tuple(int(task) for task in args.tasks),
        recording_filter=recording_filter,
    )
    indices = select_indices(dataset, recording_filter, int(args.max_recordings))

    report: dict[str, Any] = {
        "kind": "stage3_inference_benchmark",
        "checkpoint": None if checkpoint is None else str(Path(args.checkpoint).resolve()),
        "config_path": None if args.config is None else str(Path(args.config).resolve()),
        "device_requested": str(args.device),
        "device_resolved": str(device),
        "mode": str(args.mode),
        "subset": str(args.subset),
        "subset_path": str(subset_root),
        "array": str(args.array),
        "tasks": [int(task) for task in args.tasks],
        "recording_filter": None if recording_filter is None else sorted(recording_filter),
        "recording_count": len(indices),
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "targets": {},
    }

    if args.target in ("ifan", "both"):
        if checkpoint is None:
            raise ValueError("IFAN benchmark requires a valid checkpoint.")
        report["targets"]["ifan"] = run_ifan_benchmark(
            args=args,
            checkpoint=checkpoint,
            config=config,
            pipeline=pipeline,
            device=device,
            dataset=dataset,
            indices=indices,
        )

    if args.target in ("baseline", "both"):
        report["targets"]["baseline"] = run_baseline_benchmark(
            args=args,
            config=config,
            pipeline=pipeline,
            device=device,
            dataset=dataset,
            indices=indices,
        )

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        output_path.with_suffix(".md").write_text(markdown_summary(report), encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
