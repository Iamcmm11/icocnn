from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SPECS = {
    "baseline": {
        "kind": "maba_pair",
        "path": "maba_doa/outputs/locata_eval_benchmark2_baseline_vs_replace1d.json",
        "model_key": "baseline",
        "name": "baseline",
        "param_count_path": "baseline_model.param_count",
    },
    "replace_1d_with_maba": {
        "kind": "maba_pair",
        "path": "maba_doa/outputs/locata_eval_benchmark2_baseline_vs_replace1d.json",
        "model_key": "maba",
        "name": "replace_1d_with_maba",
        "param_count_path": "maba_model.param_count",
    },
    "ablation_no_gate": {
        "kind": "maba_pair",
        "path": "maba_doa/outputs/locata_eval_benchmark2_baseline_vs_ablation_no_gate.json",
        "model_key": "maba",
        "name": "ablation_no_gate",
        "param_count_path": "maba_model.param_count",
    },
    "IFAN": {
        "kind": "ifan_pair",
        "path": "IFAN_Edge/outputs/stage3/analysis/locata_eval_benchmark2_best_rerun.json",
        "model_key": "ifan",
        "name": "IFAN",
        "param_count_path": "model_profile.trainable_params",
    },
    "IFAN_Maba": {
        "kind": "ifan_pair",
        "path": "IFAN_Maba/outputs/stage3/analysis/locata_eval_benchmark2_best.json",
        "model_key": "ifan",
        "name": "IFAN_Maba",
        "param_count_path": "model_profile.trainable_params",
    },
    "IFAN_80": {
        "kind": "ifan_pair",
        "path": "IFAN_Edge/outputs/stage3/analysis/locata_eval_benchmark2_ifan80_best.json",
        "model_key": "ifan",
        "name": "IFAN_80",
        "param_count_path": "model_profile.trainable_params",
    },
    "IFAN_LC": {
        "kind": "ifan_pair",
        "path": "IFAN_Edge/outputs/stage3/analysis/locata_eval_benchmark2_ifan_lc_best.json",
        "model_key": "ifan",
        "name": "IFAN_LC",
        "param_count_path": "model_profile.trainable_params",
    },
}

SUMMARY_FIELDS = (
    ("task1_best", "task1", "best"),
    ("task1_mean", "task1", "mean"),
    ("task3_best", "task3", "best"),
    ("task3_mean", "task3", "mean"),
    ("task5_best", "task5", "best"),
    ("task5_mean", "task5", "mean"),
    ("standard_deviation", "recordings", "std"),
    ("median", "recordings", "median"),
    ("average", "recordings", "average"),
)


def get_path_value(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recording_level_from_ifan_report(report: dict[str, Any], model_key: str) -> dict[str, Any]:
    if "recording_level_summary" in report:
        return report["recording_level_summary"]

    per_recording = report["per_recording"]
    summary: dict[str, Any] = {"task_counts": {}, "tasks": {}, "recordings": {}}
    for task in sorted({int(row["task"]) for row in per_recording}):
        rows = [row for row in per_recording if int(row["task"]) == task]
        model_ws = [float(row[model_key]["with_silences_rmsae_deg"]) for row in rows]
        model_ns = [float(row[model_key]["without_silences_rmsae_deg"]) for row in rows]
        summary["task_counts"][f"task{task}"] = len(rows)
        summary["tasks"][f"task{task}"] = {
            model_key: {
                "with_silences_rmsae_deg": summarize_recording_values(model_ws),
                "without_silences_rmsae_deg": summarize_recording_values(model_ns),
            }
        }
    model_all_ws = [float(row[model_key]["with_silences_rmsae_deg"]) for row in per_recording]
    model_all_ns = [float(row[model_key]["without_silences_rmsae_deg"]) for row in per_recording]
    summary["recordings"] = {
        "count": len(per_recording),
        model_key: {
            "with_silences_rmsae_deg": summarize_recording_values(model_all_ws),
            "without_silences_rmsae_deg": summarize_recording_values(model_all_ns),
        },
    }
    return summary


def summarize_recording_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"best": 0.0, "mean": 0.0, "std": 0.0, "median": 0.0, "average": 0.0}
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    return {
        "best": float(arr.min()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "average": float(arr.mean()),
    }


def recording_level_from_maba_pair_report(report: dict[str, Any], model_key: str) -> dict[str, Any]:
    if "recording_level_summary" in report:
        return report["recording_level_summary"]

    per_recording = report["recordings"]
    summary: dict[str, Any] = {"task_counts": {}, "tasks": {}, "recordings": {}}
    for task in sorted({int(row["task"]) for row in per_recording}):
        rows = [row for row in per_recording if int(row["task"]) == task]
        model_ws = [float(row[model_key]["with_silences_rmsae_deg"]) for row in rows]
        model_ns = [float(row[model_key]["without_silences_rmsae_deg"]) for row in rows]
        summary["task_counts"][f"task{task}"] = len(rows)
        summary["tasks"][f"task{task}"] = {
            model_key: {
                "with_silences_rmsae_deg": summarize_recording_values(model_ws),
                "without_silences_rmsae_deg": summarize_recording_values(model_ns),
            }
        }
    model_all_ws = [float(row[model_key]["with_silences_rmsae_deg"]) for row in per_recording]
    model_all_ns = [float(row[model_key]["without_silences_rmsae_deg"]) for row in per_recording]
    summary["recordings"] = {
        "count": len(per_recording),
        model_key: {
            "with_silences_rmsae_deg": summarize_recording_values(model_all_ws),
            "without_silences_rmsae_deg": summarize_recording_values(model_all_ns),
        },
    }
    return summary


def extract_model_metrics(spec: dict[str, Any]) -> dict[str, Any]:
    report = load_json(REPO_ROOT / spec["path"])
    if spec["kind"] == "maba_pair":
        recording_summary = recording_level_from_maba_pair_report(report, spec["model_key"])
        subset = report["subset"]
        array = report["array"]
        tasks = tuple(report["tasks"])
        total_count = int(len(report["recordings"]))
    else:
        recording_summary = recording_level_from_ifan_report(report, spec["model_key"])
        subset = report["subset"]
        array = report["array"]
        tasks = tuple(report["tasks"])
        total_count = int(len(report["per_recording"]))

    param_count = get_path_value(report, spec["param_count_path"])
    if param_count is None:
        raise KeyError(f"Could not resolve param count for {spec['name']} from {spec['path']}")

    return {
        "name": spec["name"],
        "source": str((REPO_ROOT / spec["path"]).resolve()),
        "subset": subset,
        "array": array,
        "tasks": list(tasks),
        "param_count": int(param_count),
        "recording_level_summary": recording_summary,
        "total_count": total_count,
    }


def build_model_row(model: dict[str, Any], baseline: dict[str, Any], metric_key: str) -> dict[str, Any]:
    model_key = "baseline" if model["name"] == "baseline" else ("maba" if model["name"] in {"replace_1d_with_maba", "ablation_no_gate"} else "ifan")
    baseline_key = "baseline"
    summary = model["recording_level_summary"]
    baseline_summary = baseline["recording_level_summary"]

    row = {
        "model": model["name"],
        "params": int(model["param_count"]),
    }
    for column_name, scope, field_name in SUMMARY_FIELDS:
        if scope == "recordings":
            model_values = summary["recordings"][model_key][metric_key]
            baseline_values = baseline_summary["recordings"][baseline_key][metric_key]
        else:
            model_values = summary["tasks"][scope][model_key][metric_key]
            baseline_values = baseline_summary["tasks"][scope][baseline_key][metric_key]
        row[column_name] = float(model_values[field_name])
        row[f"{column_name}_delta"] = float(model_values[field_name] - baseline_values[field_name])
    return row


def build_table_rows(models: dict[str, dict[str, Any]], metric_key: str) -> list[dict[str, Any]]:
    baseline = models["baseline"]
    ordered = ["baseline", "replace_1d_with_maba", "ablation_no_gate", "IFAN", "IFAN_Maba", "IFAN_80", "IFAN_LC"]
    return [build_model_row(models[name], baseline, metric_key) for name in ordered]


def markdown_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    headers = [
        "Model",
        "Params",
        "Task1 Best",
        "Delta",
        "Task1 Mean",
        "Delta",
        "Task3 Best",
        "Delta",
        "Task3 Mean",
        "Delta",
        "Task5 Best",
        "Delta",
        "Task5 Mean",
        "Delta",
        "Std",
        "Delta",
        "Median",
        "Delta",
        "Average",
        "Delta",
    ]
    keys = [
        "model",
        "params",
        "task1_best",
        "task1_best_delta",
        "task1_mean",
        "task1_mean_delta",
        "task3_best",
        "task3_best_delta",
        "task3_mean",
        "task3_mean_delta",
        "task5_best",
        "task5_best_delta",
        "task5_mean",
        "task5_mean_delta",
        "standard_deviation",
        "standard_deviation_delta",
        "median",
        "median_delta",
        "average",
        "average_delta",
    ]
    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        rendered = []
        for key in keys:
            value = row[key]
            if key in {"model"}:
                rendered.append(str(value))
            elif key in {"params"}:
                rendered.append(str(int(value)))
            else:
                rendered.append(f"{float(value):+.4f}" if key.endswith("_delta") else f"{float(value):.4f}")
        lines.append("| " + " | ".join(rendered) + " |")
    lines.append("")
    return lines


def default_output_json() -> Path:
    return REPO_ROOT / "IFAN_Maba/outputs/stage3/analysis/locata_four_model_compare.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the multi-model LOCATA comparison tables from per-model evaluation JSON files.")
    parser.add_argument("--baseline-report", default=DEFAULT_SPECS["baseline"]["path"])
    parser.add_argument("--replace1d-report", default=DEFAULT_SPECS["replace_1d_with_maba"]["path"])
    parser.add_argument("--ablation-report", default=DEFAULT_SPECS["ablation_no_gate"]["path"])
    parser.add_argument("--ifan-report", default=DEFAULT_SPECS["IFAN"]["path"])
    parser.add_argument("--ifan-maba-report", default=DEFAULT_SPECS["IFAN_Maba"]["path"])
    parser.add_argument("--ifan80-report", default=DEFAULT_SPECS["IFAN_80"]["path"])
    parser.add_argument("--ifan-lc-report", default=DEFAULT_SPECS["IFAN_LC"]["path"])
    parser.add_argument("--output-json", default=str(default_output_json()))
    parser.add_argument("--output-md", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    specs = {
        "baseline": {**DEFAULT_SPECS["baseline"], "path": args.baseline_report},
        "replace_1d_with_maba": {**DEFAULT_SPECS["replace_1d_with_maba"], "path": args.replace1d_report},
        "ablation_no_gate": {**DEFAULT_SPECS["ablation_no_gate"], "path": args.ablation_report},
        "IFAN": {**DEFAULT_SPECS["IFAN"], "path": args.ifan_report},
        "IFAN_Maba": {**DEFAULT_SPECS["IFAN_Maba"], "path": args.ifan_maba_report},
        "IFAN_80": {**DEFAULT_SPECS["IFAN_80"], "path": args.ifan80_report},
        "IFAN_LC": {**DEFAULT_SPECS["IFAN_LC"], "path": args.ifan_lc_report},
    }

    models = {name: extract_model_metrics(spec) for name, spec in specs.items()}

    subset_values = {model["subset"] for model in models.values()}
    array_values = {model["array"] for model in models.values()}
    task_values = {tuple(model["tasks"]) for model in models.values()}
    if len(subset_values) != 1 or len(array_values) != 1 or len(task_values) != 1:
        raise ValueError("All model reports must share the same subset, array, and task definition.")

    task_counts = models["baseline"]["recording_level_summary"]["task_counts"]
    total_count = models["baseline"]["recording_level_summary"]["recordings"]["count"]
    ws_rows = build_table_rows(models, "with_silences_rmsae_deg")
    ns_rows = build_table_rows(models, "without_silences_rmsae_deg")

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "subset": next(iter(subset_values)),
        "array": next(iter(array_values)),
        "tasks": list(next(iter(task_values))),
        "available_recording_counts": {
            **task_counts,
            "total": int(total_count),
        },
        "table_rows": {
            "with_silences": ws_rows,
            "without_silences": ns_rows,
        },
        "models": models,
        "notes": {
            "average_definition": "recording-weighted mean over all available recordings",
            "best_definition": "minimum recording-level RMSAE within each task",
            "standard_deviation_definition": "population standard deviation across all available recordings",
            "median_definition": "recording-level median across all available recordings",
        },
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md) if args.output_md is not None else output_json.with_suffix(".md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# LOCATA Model Compare",
        "",
        f"- subset: `{result['subset']}`",
        f"- array: `{result['array']}`",
        f"- tasks: `{', '.join('task{}'.format(task) for task in result['tasks'])}`",
        f"- available recordings: `task1={task_counts.get('task1', 0)}, task3={task_counts.get('task3', 0)}, task5={task_counts.get('task5', 0)}, total={total_count}`",
        "",
    ]
    lines.extend(markdown_table("With Silences", ws_rows))
    lines.extend(markdown_table("Without Silences", ns_rows))
    output_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"json_path": str(output_json), "markdown_path": str(output_md), "counts": result["available_recording_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
