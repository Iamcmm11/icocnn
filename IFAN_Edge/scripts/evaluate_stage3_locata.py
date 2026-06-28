from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import gpuRIR
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acousticTrackingDataset import LocataDataset, Windowing
from utils import cart2sph

from ifan_edge.eval import (
    build_baseline_preprocessor_from_config,
    build_ifan_preprocessor_from_config,
    load_baseline_model_from_config,
    select_model_inputs,
)
from ifan_edge.models import IFANModel
from ifan_edge.training import IFANTrainingConfig, IFANTrainingPipeline


def summarize_scalar(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def subtract_summary(lhs: dict[str, float], rhs: dict[str, float]) -> dict[str, float]:
    keys = ("mean", "median", "std", "min", "max")
    return {key: float(lhs.get(key, 0.0) - rhs.get(key, 0.0)) for key in keys}


def recording_level_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"best": 0.0, "mean": 0.0, "std": 0.0, "median": 0.0, "average": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "best": float(arr.min()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "average": float(arr.mean()),
    }


def subtract_recording_level_summary(lhs: dict[str, float], rhs: dict[str, float]) -> dict[str, float]:
    return {key: float(lhs.get(key, 0.0) - rhs.get(key, 0.0)) for key in ("best", "mean", "std", "median", "average")}


def build_recording_level_summary(
    per_recording: list[dict[str, Any]],
    *,
    model_key: str,
    baseline_key: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"task_counts": {}, "tasks": {}, "recordings": {}}
    for task in sorted({int(row["task"]) for row in per_recording}):
        task_rows = [row for row in per_recording if int(row["task"]) == task]
        summary["task_counts"][f"task{task}"] = len(task_rows)
        model_ws = [float(row[model_key]["with_silences_rmsae_deg"]) for row in task_rows]
        model_ns = [float(row[model_key]["without_silences_rmsae_deg"]) for row in task_rows]
        baseline_ws = [float(row[baseline_key]["with_silences_rmsae_deg"]) for row in task_rows]
        baseline_ns = [float(row[baseline_key]["without_silences_rmsae_deg"]) for row in task_rows]
        model_task = {
            "with_silences_rmsae_deg": recording_level_summary(model_ws),
            "without_silences_rmsae_deg": recording_level_summary(model_ns),
        }
        baseline_task = {
            "with_silences_rmsae_deg": recording_level_summary(baseline_ws),
            "without_silences_rmsae_deg": recording_level_summary(baseline_ns),
        }
        summary["tasks"][f"task{task}"] = {
            model_key: model_task,
            baseline_key: baseline_task,
            "delta_vs_baseline": {
                "with_silences_rmsae_deg": subtract_recording_level_summary(
                    model_task["with_silences_rmsae_deg"],
                    baseline_task["with_silences_rmsae_deg"],
                ),
                "without_silences_rmsae_deg": subtract_recording_level_summary(
                    model_task["without_silences_rmsae_deg"],
                    baseline_task["without_silences_rmsae_deg"],
                ),
            },
        }

    model_all_ws = [float(row[model_key]["with_silences_rmsae_deg"]) for row in per_recording]
    model_all_ns = [float(row[model_key]["without_silences_rmsae_deg"]) for row in per_recording]
    baseline_all_ws = [float(row[baseline_key]["with_silences_rmsae_deg"]) for row in per_recording]
    baseline_all_ns = [float(row[baseline_key]["without_silences_rmsae_deg"]) for row in per_recording]
    summary["recordings"] = {
        "count": len(per_recording),
        model_key: {
            "with_silences_rmsae_deg": recording_level_summary(model_all_ws),
            "without_silences_rmsae_deg": recording_level_summary(model_all_ns),
        },
        baseline_key: {
            "with_silences_rmsae_deg": recording_level_summary(baseline_all_ws),
            "without_silences_rmsae_deg": recording_level_summary(baseline_all_ns),
        },
    }
    summary["recordings"]["delta_vs_baseline"] = {
        "with_silences_rmsae_deg": subtract_recording_level_summary(
            summary["recordings"][model_key]["with_silences_rmsae_deg"],
            summary["recordings"][baseline_key]["with_silences_rmsae_deg"],
        ),
        "without_silences_rmsae_deg": subtract_recording_level_summary(
            summary["recordings"][model_key]["without_silences_rmsae_deg"],
            summary["recordings"][baseline_key]["without_silences_rmsae_deg"],
        ),
    }
    return summary


def load_config_from_checkpoint(checkpoint_path: str | Path, config_path: str | None) -> tuple[dict[str, Any], IFANTrainingConfig]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if config_path is not None:
        config = IFANTrainingConfig.from_toml(config_path)
    else:
        config = IFANTrainingConfig(**checkpoint["training_config"])
    return checkpoint, config


def remap_legacy_map_refiner_keys(
    state_dict: dict[str, torch.Tensor],
    *,
    model_config,
) -> dict[str, torch.Tensor]:
    """Map older pre_readout checkpoints onto the current feature_refiner naming."""
    if getattr(model_config, "map_refiner_position", "pre_softargmax") != "pre_readout":
        return state_dict
    has_feature_keys = any(key.startswith("feature_refiner.") for key in state_dict.keys())
    has_map_keys = any(key.startswith("map_refiner.") for key in state_dict.keys())
    if has_feature_keys or not has_map_keys:
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("map_refiner."):
            remapped[key.replace("map_refiner.", "feature_refiner.", 1)] = value
        else:
            remapped[key] = value
    return remapped


def normalize_tasks(values: list[int]) -> tuple[int, ...]:
    tasks = tuple(int(value) for value in values)
    for task in tasks:
        if task not in (1, 3, 5):
            raise ValueError(f"Only single-source LOCATA tasks 1, 3, and 5 are supported, got task={task}")
    return tasks


def parse_recording_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {str(value) for value in values}


def build_locata_dataset(
    *,
    root: str | Path,
    subset: str,
    array: str,
    fs: int,
    k: int,
    step: int,
    tasks: tuple[int, ...],
    recording_filter: set[str] | None,
) -> tuple[LocataDataset, Path]:
    subset_root = Path(root).expanduser().resolve() / subset
    if not subset_root.is_dir():
        raise FileNotFoundError(f"LOCATA subset directory does not exist: {subset_root}")
    dataset = LocataDataset(
        str(subset_root),
        array,
        fs,
        tasks=tasks,
        recording=None,
        dev=True,
        transforms=[Windowing(k, step, window=np.hanning)],
    )
    if recording_filter is not None:
        dataset.directories = [path for path in dataset.directories if Path(path).parent.name in recording_filter]
        dataset.directories.sort()
    return dataset, subset_root


def extract_metadata_from_directory(directory: str | Path) -> dict[str, Any]:
    path = Path(directory)
    return {
        "task": int(path.parents[1].name.replace("task", "")),
        "recording": path.parent.name,
        "array": path.name,
        "directory": str(path),
    }


def predict_ifan_scene(
    *,
    model: IFANModel,
    preprocessor,
    model_config,
    input_ablation_mode: str,
    mic_sig_batch,
    acoustic_scene_batch,
) -> np.ndarray:
    with torch.no_grad():
        maps, _ = preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
        inputs = select_model_inputs(maps, model_config, input_ablation_mode)
        doa_pred = cart2sph(model(inputs).contiguous()).cpu().detach()
    return doa_pred[0, ...].unsqueeze(0).numpy()


def predict_baseline_scene(
    *,
    model,
    preprocessor,
    mic_sig_batch,
    acoustic_scene_batch,
) -> np.ndarray:
    with torch.no_grad():
        maps, _ = preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
        doa_pred = cart2sph(model(maps).contiguous()).cpu().detach()
    return doa_pred[0, ...].unsqueeze(0).numpy()


def evaluate_locata_dataset(
    *,
    dataset: LocataDataset,
    ifan_model: IFANModel,
    ifan_preprocessor,
    model_config,
    input_ablation_mode: str,
    baseline_model,
    baseline_preprocessor,
) -> dict[str, Any]:
    per_recording: list[dict[str, Any]] = []
    ifan_with: list[float] = []
    ifan_without: list[float] = []
    baseline_with: list[float] = []
    baseline_without: list[float] = []

    for idx in range(len(dataset)):
        mic_sig_batch, acoustic_scene_batch = dataset.get_batch(idx, idx + 1)
        metadata = extract_metadata_from_directory(dataset.directories[idx])
        scene_ifan = acoustic_scene_batch[0]
        scene_baseline = acoustic_scene_batch[0]

        scene_ifan.DOAw_pred = predict_ifan_scene(
            model=ifan_model,
            preprocessor=ifan_preprocessor,
            model_config=model_config,
            input_ablation_mode=input_ablation_mode,
            mic_sig_batch=mic_sig_batch,
            acoustic_scene_batch=acoustic_scene_batch,
        )
        ifan_with_silences = float(scene_ifan.get_rmsae(exclude_silences=False))
        ifan_without_silences = float(scene_ifan.get_rmsae(exclude_silences=True))

        mic_sig_batch_base, acoustic_scene_batch_base = dataset.get_batch(idx, idx + 1)
        scene_baseline = acoustic_scene_batch_base[0]
        scene_baseline.DOAw_pred = predict_baseline_scene(
            model=baseline_model,
            preprocessor=baseline_preprocessor,
            mic_sig_batch=mic_sig_batch_base,
            acoustic_scene_batch=acoustic_scene_batch_base,
        )
        baseline_with_silences = float(scene_baseline.get_rmsae(exclude_silences=False))
        baseline_without_silences = float(scene_baseline.get_rmsae(exclude_silences=True))

        ifan_with.append(ifan_with_silences)
        ifan_without.append(ifan_without_silences)
        baseline_with.append(baseline_with_silences)
        baseline_without.append(baseline_without_silences)

        per_recording.append(
            {
                **metadata,
                "ifan": {
                    "with_silences_rmsae_deg": ifan_with_silences,
                    "without_silences_rmsae_deg": ifan_without_silences,
                },
                "baseline": {
                    "with_silences_rmsae_deg": baseline_with_silences,
                    "without_silences_rmsae_deg": baseline_without_silences,
                },
            }
        )

    per_task: dict[str, Any] = {}
    for task in sorted({row["task"] for row in per_recording}):
        task_rows = [row for row in per_recording if row["task"] == task]
        per_task[f"task{task}"] = {
            "count": len(task_rows),
            "ifan": {
                "with_silences_rmsae_deg": summarize_scalar([row["ifan"]["with_silences_rmsae_deg"] for row in task_rows]),
                "without_silences_rmsae_deg": summarize_scalar([row["ifan"]["without_silences_rmsae_deg"] for row in task_rows]),
            },
            "baseline": {
                "with_silences_rmsae_deg": summarize_scalar([row["baseline"]["with_silences_rmsae_deg"] for row in task_rows]),
                "without_silences_rmsae_deg": summarize_scalar([row["baseline"]["without_silences_rmsae_deg"] for row in task_rows]),
            },
        }
        per_task[f"task{task}"]["delta_vs_baseline"] = {
            "with_silences_rmsae_deg": subtract_summary(
                per_task[f"task{task}"]["ifan"]["with_silences_rmsae_deg"],
                per_task[f"task{task}"]["baseline"]["with_silences_rmsae_deg"],
            ),
            "without_silences_rmsae_deg": subtract_summary(
                per_task[f"task{task}"]["ifan"]["without_silences_rmsae_deg"],
                per_task[f"task{task}"]["baseline"]["without_silences_rmsae_deg"],
            ),
        }

    overall = {
        "count": len(per_recording),
        "ifan": {
            "with_silences_rmsae_deg": summarize_scalar(ifan_with),
            "without_silences_rmsae_deg": summarize_scalar(ifan_without),
        },
        "baseline": {
            "with_silences_rmsae_deg": summarize_scalar(baseline_with),
            "without_silences_rmsae_deg": summarize_scalar(baseline_without),
        },
    }
    overall["delta_vs_baseline"] = {
        "with_silences_rmsae_deg": subtract_summary(
            overall["ifan"]["with_silences_rmsae_deg"],
            overall["baseline"]["with_silences_rmsae_deg"],
        ),
        "without_silences_rmsae_deg": subtract_summary(
            overall["ifan"]["without_silences_rmsae_deg"],
            overall["baseline"]["without_silences_rmsae_deg"],
        ),
    }

    return {
        "per_recording": per_recording,
        "per_task": per_task,
        "overall": overall,
    }


def paper_reference_payload(tasks: tuple[int, ...], subset: str, array: str) -> dict[str, Any]:
    per_task = {f"task{task}": None for task in tasks}
    return {
        "paper": "IFAN: An Icosahedral Feature Attention Network for Sound Source Localization (IEEE TIM 2024)",
        "subset": subset,
        "array": array,
        "tasks": list(tasks),
        "tables": {
            "with_silences": {
                "table": "Table III",
                "reported_ifan_rmsae_deg": per_task,
                "note": "Fill manually from the paper table; values are intentionally not hard-coded.",
            },
            "without_silences": {
                "table": "Table IV",
                "reported_ifan_rmsae_deg": per_task,
                "note": "Fill manually from the paper table; values are intentionally not hard-coded.",
            },
        },
    }


def markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# LOCATA Validation Summary",
        "",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- subset: `{report['subset']}`",
        f"- array: `{report['array']}`",
        f"- tasks: `{', '.join(f'task{task}' for task in report['tasks'])}`",
        "",
        "## Overall",
        "",
        f"- IFAN with silences mean RMSAE: `{report['overall']['ifan']['with_silences_rmsae_deg']['mean']:.4f} deg`",
        f"- IFAN without silences mean RMSAE: `{report['overall']['ifan']['without_silences_rmsae_deg']['mean']:.4f} deg`",
        f"- Baseline with silences mean RMSAE: `{report['overall']['baseline']['with_silences_rmsae_deg']['mean']:.4f} deg`",
        f"- Baseline without silences mean RMSAE: `{report['overall']['baseline']['without_silences_rmsae_deg']['mean']:.4f} deg`",
        f"- Delta vs baseline with silences: `{report['overall']['delta_vs_baseline']['with_silences_rmsae_deg']['mean']:+.4f} deg`",
        f"- Delta vs baseline without silences: `{report['overall']['delta_vs_baseline']['without_silences_rmsae_deg']['mean']:+.4f} deg`",
        "",
        "## Per Task",
        "",
    ]
    for task_key, payload in report["per_task"].items():
        lines.extend(
            [
                f"### {task_key}",
                "",
                f"- count: `{payload['count']}`",
                f"- IFAN with silences mean RMSAE: `{payload['ifan']['with_silences_rmsae_deg']['mean']:.4f} deg`",
                f"- IFAN without silences mean RMSAE: `{payload['ifan']['without_silences_rmsae_deg']['mean']:.4f} deg`",
                f"- Baseline with silences mean RMSAE: `{payload['baseline']['with_silences_rmsae_deg']['mean']:.4f} deg`",
                f"- Baseline without silences mean RMSAE: `{payload['baseline']['without_silences_rmsae_deg']['mean']:.4f} deg`",
                f"- Delta vs baseline with silences: `{payload['delta_vs_baseline']['with_silences_rmsae_deg']['mean']:+.4f} deg`",
                f"- Delta vs baseline without silences: `{payload['delta_vs_baseline']['without_silences_rmsae_deg']['mean']:+.4f} deg`",
                "",
            ]
        )
    lines.extend(
        [
            "## Paper Reference",
            "",
            "- LOCATA is a paper evaluation dataset, not a training dataset.",
            "- Training dataset in the paper: `LibriSpeech train-clean-100`.",
            "- Simulated test dataset in the paper: `LibriSpeech test-clean`.",
            "- Manual table transcription target:",
            f"  - with silences: `{report['paper_reference']['tables']['with_silences']['table']}`",
            f"  - without silences: `{report['paper_reference']['tables']['without_silences']['table']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def default_output_path(checkpoint_path: str | Path, subset: str, array: str) -> Path:
    stem = Path(checkpoint_path).stem
    return Path("IFAN_Edge/outputs/stage3/analysis") / f"locata_{subset}_{array}_{stem}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the current IFAN stage-3 checkpoint on LOCATA Task 1/3/5.")
    parser.add_argument(
        "--checkpoint",
        default="IFAN_Edge/outputs/stage3/ifan_stage3_full20_freqblock_paper_original_20260419_005314/checkpoints/best_rmsae.pt",
        help="Path to an IFAN stage-3 checkpoint (*.pt).",
    )
    parser.add_argument("--config", default=None, help="Optional stage-3 config TOML. Uses checkpoint metadata when omitted.")
    parser.add_argument("--locata-root", default="datasets/LOCATA/LOCATA", help="Path to the LOCATA root directory containing eval/dev.")
    parser.add_argument("--subset", choices=("dev", "eval"), default="eval")
    parser.add_argument("--array", choices=("dummy", "eigenmike", "benchmark2", "dicit"), default="benchmark2")
    parser.add_argument("--tasks", nargs="+", type=int, default=[1, 3, 5], help="LOCATA tasks to evaluate. Only single-source tasks 1, 3, and 5 are supported.")
    parser.add_argument("--recording", action="append", default=None, help="Optional recording name filter for smoke runs. Can be repeated.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output path. A Markdown summary will be written alongside it.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint, config = load_config_from_checkpoint(args.checkpoint, args.config)
    if args.device is not None:
        config.device = args.device

    tasks = normalize_tasks(args.tasks)
    recording_filter = parse_recording_filter(args.recording)
    pipeline = IFANTrainingPipeline(config)
    device = pipeline.resolve_device()

    dataset, subset_root = build_locata_dataset(
        root=args.locata_root,
        subset=args.subset,
        array=args.array,
        fs=config.fs,
        k=config.k,
        step=config.step,
        tasks=tasks,
        recording_filter=recording_filter,
    )

    model_state_dict = remap_legacy_map_refiner_keys(
        checkpoint["model_state_dict"],
        model_config=pipeline.model_config,
    )
    legacy_norm_keys = [
        "phat_branch.residual.norm.weight",
        "phat_branch.residual.norm.bias",
        "aux_branch.residual.norm.weight",
        "aux_branch.residual.norm.bias",
    ]
    if any(key in model_state_dict for key in legacy_norm_keys):
        pipeline.model_config.legacy_frontend_residual = True
    ifan_model = IFANModel(pipeline.model_config)
    ifan_model.load_state_dict(model_state_dict)
    ifan_model.to(device)
    ifan_model.eval()
    baseline_model = load_baseline_model_from_config(config, device)
    baseline_model.eval()

    ifan_preprocessor = build_ifan_preprocessor_from_config(config, device)
    baseline_preprocessor = build_baseline_preprocessor_from_config(config, device)

    results = evaluate_locata_dataset(
        dataset=dataset,
        ifan_model=ifan_model,
        ifan_preprocessor=ifan_preprocessor,
        model_config=pipeline.model_config,
        input_ablation_mode=config.input_ablation_mode,
        baseline_model=baseline_model,
        baseline_preprocessor=baseline_preprocessor,
    )

    report = {
        "kind": "stage3_locata_evaluation",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "subset": str(args.subset),
        "subset_path": str(subset_root),
        "array": str(args.array),
        "tasks": list(tasks),
        "recording_filter": None if recording_filter is None else sorted(recording_filter),
        "device": str(device),
        "config": {
            "fs": int(config.fs),
            "k": int(config.k),
            "step": int(config.step),
            "apply_vad": bool(config.apply_vad),
            "input_ablation_mode": str(config.input_ablation_mode),
            "lms_backend": str(config.lms_backend),
            "lms_map_mode": str(config.lms_map_mode),
            "lms_update_mode": str(config.lms_update_mode),
            "pre_fusion_pooling": bool(config.pre_fusion_pooling),
            "final_head_pooling": bool(config.final_head_pooling),
        },
        "experiment_contract": config.experiment_contract(),
        "model_profile": pipeline.build_model_profile(ifan_model),
        **results,
        "recording_level_summary": build_recording_level_summary(
            results["per_recording"],
            model_key="ifan",
            baseline_key="baseline",
        ),
        "paper_reference": paper_reference_payload(tasks, args.subset, args.array),
    }

    output_path = Path(args.output) if args.output is not None else default_output_path(args.checkpoint, args.subset, args.array)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(markdown_summary(report), encoding="utf-8")
    print(json.dumps({"json_path": str(output_path), "markdown_path": str(markdown_path), "recording_count": report["overall"]["count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
