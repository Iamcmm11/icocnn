"""Evaluate baseline and MABA checkpoints on LOCATA dev task 1/3/5."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import acousticTrackingDataset as at_dataset
import acousticTrackingLearners as at_learners

from maba_doa.train_maba_doa import build_model, load_config, set_seed


def normalize_tasks(values: Iterable[int]) -> Tuple[int, ...]:
    tasks = tuple(int(v) for v in values)
    for task in tasks:
        if task not in (1, 3, 5):
            raise ValueError(
                "Only single-source LOCATA tasks 1, 3, and 5 are supported, got task={}".format(task)
            )
    return tasks


def summarize_scalar(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _resolve_summary_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser().resolve()
    if path.is_dir():
        path = path / "summary.json"
    return path


def _resolve_checkpoint_from_summary(summary_path: Path) -> Tuple[dict[str, Any], Path]:
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    model_path = Path(summary["model_path"])
    if not model_path.is_absolute():
        repo_candidate = (REPO_ROOT / model_path).resolve()
        summary_candidate = (summary_path.parent / model_path).resolve()
        if repo_candidate.exists():
            model_path = repo_candidate
        else:
            model_path = summary_candidate
    return summary, model_path


def resolve_model_spec(
    *,
    role: str,
    summary_arg: str | None,
    checkpoint_arg: str | None,
) -> Dict[str, Any]:
    summary_path = _resolve_summary_path(summary_arg)
    if summary_path is not None:
        summary, checkpoint_path = _resolve_checkpoint_from_summary(summary_path)
        variant = summary.get("variant", role)
        return {
            "role": role,
            "variant": variant,
            "summary_path": str(summary_path),
            "checkpoint_path": str(checkpoint_path),
            "summary": summary,
        }

    if checkpoint_arg is None:
        raise ValueError("Either --{}-summary or --{}-checkpoint is required.".format(role, role))

    checkpoint_path = Path(checkpoint_arg).expanduser().resolve()
    return {
        "role": role,
        "variant": role,
        "summary_path": None,
        "checkpoint_path": str(checkpoint_path),
        "summary": {},
    }


def extract_metadata_from_directory(directory: str | Path) -> Dict[str, Any]:
    path = Path(directory)
    return {
        "task": int(path.parents[1].name.replace("task", "")),
        "recording": path.parent.name,
        "array": path.name,
        "directory": str(path),
    }


def build_locata_dataset(
    *,
    locata_root: str | Path,
    subset: str,
    array: str,
    fs: int,
    ksize: int,
    hop_ratio: float,
    tasks: Tuple[int, ...],
) -> Tuple[at_dataset.LocataDataset, Path]:
    step = int(ksize * hop_ratio)
    windowing = at_dataset.Windowing(ksize, step, window=np.hanning)
    root = Path(locata_root).expanduser().resolve()
    subset_root = root / subset
    if not subset_root.is_dir():
        raise FileNotFoundError("LOCATA subset root does not exist: {}".format(subset_root))
    dataset = at_dataset.LocataDataset(
        str(subset_root),
        array,
        fs,
        tasks=tasks,
        recording=None,
        dev=True,
        transforms=[windowing],
    )
    return dataset, subset_root


def create_learner(cfg: Dict[str, Any], model_variant: str, array_setup):
    cfg_model = json.loads(json.dumps(cfg))
    cfg_model["model"]["variant"] = model_variant
    model = build_model(cfg_model)
    fs = int(cfg_model["data"]["fs"])
    ksize = int(cfg_model["data"]["window"]["K"])
    preprocessor = at_learners.TrackingFromIcoMapsPreprocessor(
        N=array_setup.mic_pos.shape[0],
        K=ksize,
        r=int(cfg_model["data"]["r"]),
        rn=array_setup.mic_pos,
        fs=fs,
        apply_vad=bool(cfg_model["data"]["use_vad"]),
    )
    learner = at_learners.OneSourceTrackingLearner(model, preprocessor)
    requested_device = cfg_model.get("device", "cpu")
    if requested_device == "cuda" and torch.cuda.is_available():
        learner.cuda()
        device = "cuda"
    else:
        device = "cpu"
    return learner, device


def load_checkpoint_into_learner(learner, checkpoint_path: str) -> None:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    learner.model.load_state_dict(state_dict)


def evaluate_model_on_locata(
    *,
    learner,
    dataset: at_dataset.LocataDataset,
) -> List[Dict[str, Any]]:
    acoustic_scenes = learner.predict_dataset(dataset, 1, save_x=False)
    rows = []
    for idx, scene in enumerate(acoustic_scenes):
        metadata = extract_metadata_from_directory(dataset.directories[idx])
        rows.append(
            {
                **metadata,
                "with_silences_rmsae_deg": float(scene.get_rmsae(exclude_silences=False)),
                "without_silences_rmsae_deg": float(scene.get_rmsae(exclude_silences=True)),
            }
        )
    return rows


def _metric_delta(baseline_stats: Dict[str, float], maba_stats: Dict[str, float]) -> Dict[str, float]:
    return {
        "mean": float(maba_stats["mean"] - baseline_stats["mean"]),
        "median": float(maba_stats["median"] - baseline_stats["median"]),
        "std": float(maba_stats["std"] - baseline_stats["std"]),
        "min": float(maba_stats["min"] - baseline_stats["min"]),
        "max": float(maba_stats["max"] - baseline_stats["max"]),
    }


def recording_level_summary(values: List[float]) -> Dict[str, float]:
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


def recording_level_delta(baseline_stats: Dict[str, float], model_stats: Dict[str, float]) -> Dict[str, float]:
    return {
        key: float(model_stats[key] - baseline_stats[key])
        for key in ("best", "mean", "std", "median", "average")
    }


def build_report(
    *,
    baseline_spec: Dict[str, Any],
    maba_spec: Dict[str, Any],
    baseline_rows: List[Dict[str, Any]],
    maba_rows: List[Dict[str, Any]],
    config_path: str,
    locata_root: str,
    array: str,
    tasks: Tuple[int, ...],
    subset: str,
    subset_root: Path,
) -> Dict[str, Any]:
    baseline_by_key = {
        (row["task"], row["recording"], row["array"]): row for row in baseline_rows
    }
    maba_by_key = {(row["task"], row["recording"], row["array"]): row for row in maba_rows}
    keys = sorted(set(baseline_by_key) & set(maba_by_key))

    recordings = []
    baseline_ws_all: List[float] = []
    baseline_ns_all: List[float] = []
    maba_ws_all: List[float] = []
    maba_ns_all: List[float] = []

    for key in keys:
        base_row = baseline_by_key[key]
        maba_row = maba_by_key[key]
        baseline_ws_all.append(base_row["with_silences_rmsae_deg"])
        baseline_ns_all.append(base_row["without_silences_rmsae_deg"])
        maba_ws_all.append(maba_row["with_silences_rmsae_deg"])
        maba_ns_all.append(maba_row["without_silences_rmsae_deg"])
        recordings.append(
            {
                "task": base_row["task"],
                "recording": base_row["recording"],
                "array": base_row["array"],
                "directory": base_row["directory"],
                "baseline": {
                    "with_silences_rmsae_deg": base_row["with_silences_rmsae_deg"],
                    "without_silences_rmsae_deg": base_row["without_silences_rmsae_deg"],
                },
                "maba": {
                    "with_silences_rmsae_deg": maba_row["with_silences_rmsae_deg"],
                    "without_silences_rmsae_deg": maba_row["without_silences_rmsae_deg"],
                },
                "delta_maba_minus_baseline": {
                    "with_silences_rmsae_deg": float(
                        maba_row["with_silences_rmsae_deg"] - base_row["with_silences_rmsae_deg"]
                    ),
                    "without_silences_rmsae_deg": float(
                        maba_row["without_silences_rmsae_deg"] - base_row["without_silences_rmsae_deg"]
                    ),
                },
            }
        )

    task_summary: Dict[str, Any] = {}
    for task in tasks:
        task_rows = [row for row in recordings if row["task"] == task]
        baseline_ws = [row["baseline"]["with_silences_rmsae_deg"] for row in task_rows]
        baseline_ns = [row["baseline"]["without_silences_rmsae_deg"] for row in task_rows]
        maba_ws = [row["maba"]["with_silences_rmsae_deg"] for row in task_rows]
        maba_ns = [row["maba"]["without_silences_rmsae_deg"] for row in task_rows]
        baseline_ws_stats = summarize_scalar(baseline_ws)
        baseline_ns_stats = summarize_scalar(baseline_ns)
        maba_ws_stats = summarize_scalar(maba_ws)
        maba_ns_stats = summarize_scalar(maba_ns)
        task_summary["task{}".format(task)] = {
            "count": len(task_rows),
            "baseline": {
                "with_silences_rmsae_deg": baseline_ws_stats,
                "without_silences_rmsae_deg": baseline_ns_stats,
            },
            "maba": {
                "with_silences_rmsae_deg": maba_ws_stats,
                "without_silences_rmsae_deg": maba_ns_stats,
            },
            "delta_maba_minus_baseline": {
                "with_silences_rmsae_deg": _metric_delta(baseline_ws_stats, maba_ws_stats),
                "without_silences_rmsae_deg": _metric_delta(baseline_ns_stats, maba_ns_stats),
            },
        }

    baseline_ws_stats = summarize_scalar(baseline_ws_all)
    baseline_ns_stats = summarize_scalar(baseline_ns_all)
    maba_ws_stats = summarize_scalar(maba_ws_all)
    maba_ns_stats = summarize_scalar(maba_ns_all)

    overall_summary = {
        "count": len(recordings),
        "baseline": {
            "with_silences_rmsae_deg": baseline_ws_stats,
            "without_silences_rmsae_deg": baseline_ns_stats,
        },
        "maba": {
            "with_silences_rmsae_deg": maba_ws_stats,
            "without_silences_rmsae_deg": maba_ns_stats,
        },
        "delta_maba_minus_baseline": {
            "with_silences_rmsae_deg": _metric_delta(baseline_ws_stats, maba_ws_stats),
            "without_silences_rmsae_deg": _metric_delta(baseline_ns_stats, maba_ns_stats),
        },
    }

    comparison = {
        "better_model": {
            "with_silences": "maba"
            if maba_ws_stats["mean"] < baseline_ws_stats["mean"]
            else "baseline",
            "without_silences": "maba"
            if maba_ns_stats["mean"] < baseline_ns_stats["mean"]
            else "baseline",
        },
        "task_winners": {
            key: {
                "with_silences": "maba"
                if value["maba"]["with_silences_rmsae_deg"]["mean"]
                < value["baseline"]["with_silences_rmsae_deg"]["mean"]
                else "baseline",
                "without_silences": "maba"
                if value["maba"]["without_silences_rmsae_deg"]["mean"]
                < value["baseline"]["without_silences_rmsae_deg"]["mean"]
                else "baseline",
            }
            for key, value in task_summary.items()
        },
    }

    recording_level_summary_payload: Dict[str, Any] = {"task_counts": {}, "tasks": {}, "recordings": {}}
    for task in tasks:
        task_key = "task{}".format(task)
        task_rows = [row for row in recordings if row["task"] == task]
        baseline_ws = [row["baseline"]["with_silences_rmsae_deg"] for row in task_rows]
        baseline_ns = [row["baseline"]["without_silences_rmsae_deg"] for row in task_rows]
        maba_ws = [row["maba"]["with_silences_rmsae_deg"] for row in task_rows]
        maba_ns = [row["maba"]["without_silences_rmsae_deg"] for row in task_rows]
        recording_level_summary_payload["task_counts"][task_key] = len(task_rows)
        base_task = {
            "with_silences_rmsae_deg": recording_level_summary(baseline_ws),
            "without_silences_rmsae_deg": recording_level_summary(baseline_ns),
        }
        model_task = {
            "with_silences_rmsae_deg": recording_level_summary(maba_ws),
            "without_silences_rmsae_deg": recording_level_summary(maba_ns),
        }
        recording_level_summary_payload["tasks"][task_key] = {
            "baseline": base_task,
            "maba": model_task,
            "delta_maba_minus_baseline": {
                "with_silences_rmsae_deg": recording_level_delta(
                    base_task["with_silences_rmsae_deg"],
                    model_task["with_silences_rmsae_deg"],
                ),
                "without_silences_rmsae_deg": recording_level_delta(
                    base_task["without_silences_rmsae_deg"],
                    model_task["without_silences_rmsae_deg"],
                ),
            },
        }

    base_all = {
        "with_silences_rmsae_deg": recording_level_summary(baseline_ws_all),
        "without_silences_rmsae_deg": recording_level_summary(baseline_ns_all),
    }
    model_all = {
        "with_silences_rmsae_deg": recording_level_summary(maba_ws_all),
        "without_silences_rmsae_deg": recording_level_summary(maba_ns_all),
    }
    recording_level_summary_payload["recordings"] = {
        "count": len(recordings),
        "baseline": base_all,
        "maba": model_all,
        "delta_maba_minus_baseline": {
            "with_silences_rmsae_deg": recording_level_delta(
                base_all["with_silences_rmsae_deg"],
                model_all["with_silences_rmsae_deg"],
            ),
            "without_silences_rmsae_deg": recording_level_delta(
                base_all["without_silences_rmsae_deg"],
                model_all["without_silences_rmsae_deg"],
            ),
        },
    }

    return {
        "kind": "maba_doa_locata_evaluation",
        "config_path": config_path,
        "subset": subset,
        "locata_root": str(subset_root),
        "array": array,
        "tasks": list(tasks),
        "baseline_model": {
            "variant": baseline_spec["variant"],
            "summary_path": baseline_spec["summary_path"],
            "checkpoint_path": baseline_spec["checkpoint_path"],
            "param_count": baseline_spec["summary"].get("param_count"),
        },
        "maba_model": {
            "variant": maba_spec["variant"],
            "summary_path": maba_spec["summary_path"],
            "checkpoint_path": maba_spec["checkpoint_path"],
            "param_count": maba_spec["summary"].get("param_count"),
        },
        "recordings": recordings,
        "task_summary": task_summary,
        "overall_summary": overall_summary,
        "recording_level_summary": recording_level_summary_payload,
        "comparison": comparison,
    }


def _fmt(value: float) -> str:
    return "{:.4f}".format(float(value))


def markdown_summary(report: Dict[str, Any]) -> str:
    lines = [
        "# MABA-DOA LOCATA Baseline vs MABA",
        "",
        "- subset: `{}`".format(report["subset"]),
        "- array: `{}`".format(report["array"]),
        "- tasks: `{}`".format(", ".join(str(t) for t in report["tasks"])),
        "- baseline checkpoint: `{}`".format(report["baseline_model"]["checkpoint_path"]),
        "- maba checkpoint: `{}`".format(report["maba_model"]["checkpoint_path"]),
        "- LOCATA is an evaluation dataset, not a training dataset.",
        "",
        "## Overall Summary",
        "",
        "| Metric | Baseline | MABA | Delta (MABA-Baseline) | Better |",
        "| --- | ---: | ---: | ---: | --- |",
        "| With silences mean RMSAE (deg) | {} | {} | {} | {} |".format(
            _fmt(report["overall_summary"]["baseline"]["with_silences_rmsae_deg"]["mean"]),
            _fmt(report["overall_summary"]["maba"]["with_silences_rmsae_deg"]["mean"]),
            _fmt(report["overall_summary"]["delta_maba_minus_baseline"]["with_silences_rmsae_deg"]["mean"]),
            report["comparison"]["better_model"]["with_silences"],
        ),
        "| Without silences mean RMSAE (deg) | {} | {} | {} | {} |".format(
            _fmt(report["overall_summary"]["baseline"]["without_silences_rmsae_deg"]["mean"]),
            _fmt(report["overall_summary"]["maba"]["without_silences_rmsae_deg"]["mean"]),
            _fmt(report["overall_summary"]["delta_maba_minus_baseline"]["without_silences_rmsae_deg"]["mean"]),
            report["comparison"]["better_model"]["without_silences"],
        ),
        "",
        "## Task Summary",
        "",
        "| Task | Count | Baseline WS mean | MABA WS mean | Delta WS | Baseline NS mean | MABA NS mean | Delta NS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task_key, stats in report["task_summary"].items():
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                task_key,
                stats["count"],
                _fmt(stats["baseline"]["with_silences_rmsae_deg"]["mean"]),
                _fmt(stats["maba"]["with_silences_rmsae_deg"]["mean"]),
                _fmt(stats["delta_maba_minus_baseline"]["with_silences_rmsae_deg"]["mean"]),
                _fmt(stats["baseline"]["without_silences_rmsae_deg"]["mean"]),
                _fmt(stats["maba"]["without_silences_rmsae_deg"]["mean"]),
                _fmt(stats["delta_maba_minus_baseline"]["without_silences_rmsae_deg"]["mean"]),
            )
        )

    lines.extend(
        [
            "",
            "## Recording-level Results",
            "",
            "| Task | Recording | Baseline WS | MABA WS | Delta WS | Baseline NS | MABA NS | Delta NS |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["recordings"]:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["task"],
                row["recording"],
                _fmt(row["baseline"]["with_silences_rmsae_deg"]),
                _fmt(row["maba"]["with_silences_rmsae_deg"]),
                _fmt(row["delta_maba_minus_baseline"]["with_silences_rmsae_deg"]),
                _fmt(row["baseline"]["without_silences_rmsae_deg"]),
                _fmt(row["maba"]["without_silences_rmsae_deg"]),
                _fmt(row["delta_maba_minus_baseline"]["without_silences_rmsae_deg"]),
            )
        )

    lines.extend(
        [
            "",
            "## Short Conclusion",
            "",
            "- With silences better model: `{}`".format(report["comparison"]["better_model"]["with_silences"]),
            "- Without silences better model: `{}`".format(report["comparison"]["better_model"]["without_silences"]),
        ]
    )
    for task_key, winners in report["comparison"]["task_winners"].items():
        lines.append(
            "- {}: with silences=`{}`, without silences=`{}`".format(
                task_key, winners["with_silences"], winners["without_silences"]
            )
        )
    return "\n".join(lines) + "\n"


def default_output_json(array: str) -> Path:
    return Path("maba_doa/outputs") / "locata_eval_{}_baseline_vs_maba.json".format(array)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained baseline and MABA checkpoints on LOCATA dev task 1/3/5."
    )
    parser.add_argument("--config", default="maba_doa/configs/local_librispeech.yaml")
    parser.add_argument(
        "--baseline-summary",
        default="maba_doa/outputs/maba_doa_r2_baseline_20260406_220454/summary.json",
        help="Baseline summary.json path or run directory containing summary.json.",
    )
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument(
        "--maba-summary",
        default="maba_doa/outputs/maba_doa_r2_maba_20260407_005546/summary.json",
        help="MABA summary.json path or run directory containing summary.json.",
    )
    parser.add_argument("--maba-checkpoint", default=None)
    parser.add_argument("--locata-root", default="datasets/LOCATA/LOCATA")
    parser.add_argument("--subset", choices=("dev", "eval"), default="eval")
    parser.add_argument("--array", default="benchmark2")
    parser.add_argument("--tasks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"]["seed"]))
    tasks = normalize_tasks(args.tasks)

    baseline_spec = resolve_model_spec(
        role="baseline",
        summary_arg=args.baseline_summary,
        checkpoint_arg=args.baseline_checkpoint,
    )
    maba_spec = resolve_model_spec(
        role="maba",
        summary_arg=args.maba_summary,
        checkpoint_arg=args.maba_checkpoint,
    )

    dataset, subset_root = build_locata_dataset(
        locata_root=args.locata_root,
        subset=args.subset,
        array=args.array,
        fs=int(cfg["data"]["fs"]),
        ksize=int(cfg["data"]["window"]["K"]),
        hop_ratio=float(cfg["data"]["window"]["hop_ratio"]),
        tasks=tasks,
    )
    array_setup = dataset.array_setup

    baseline_learner, baseline_device = create_learner(cfg, baseline_spec["variant"], array_setup)
    maba_learner, maba_device = create_learner(cfg, maba_spec["variant"], array_setup)
    load_checkpoint_into_learner(baseline_learner, baseline_spec["checkpoint_path"])
    load_checkpoint_into_learner(maba_learner, maba_spec["checkpoint_path"])

    baseline_rows = evaluate_model_on_locata(learner=baseline_learner, dataset=dataset)
    maba_rows = evaluate_model_on_locata(learner=maba_learner, dataset=dataset)
    report = build_report(
        baseline_spec={**baseline_spec, "device": baseline_device},
        maba_spec={**maba_spec, "device": maba_device},
        baseline_rows=baseline_rows,
        maba_rows=maba_rows,
        config_path=str(Path(args.config).expanduser().resolve()),
        locata_root=str(Path(args.locata_root).expanduser().resolve()),
        array=args.array,
        tasks=tasks,
        subset=str(args.subset),
        subset_root=subset_root,
    )

    output_json = Path(args.output_json) if args.output_json else default_output_json(args.array)
    output_md = Path(args.output_md) if args.output_md else output_json.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_text = markdown_summary(report)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
