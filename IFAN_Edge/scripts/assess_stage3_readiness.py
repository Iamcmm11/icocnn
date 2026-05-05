from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_json_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "summary.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"Could not resolve JSON input from {path}")
    return candidate.resolve()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve_json_path(path).read_text(encoding="utf-8"))


def extract_summary_metrics(summary: dict[str, Any]) -> dict[str, float]:
    baseline_compare = summary["baseline_compare"]
    return {
        "sim_overall_delta_deg": float(baseline_compare["mean_rmsae_deg"]["delta"]),
        "sim_hard_delta_deg": float(baseline_compare["hard_scenarios_mean_rmsae_deg"]["delta"]),
        "best_val_rmsae_deg": float(summary["best_val_rmsae_deg"]),
    }


def mean_delta(container: dict[str, Any], ifan_key: str, baseline_key: str) -> float:
    if "delta_vs_baseline" in container:
        return float(container["delta_vs_baseline"][ifan_key]["mean"])
    return float(container["ifan"][ifan_key]["mean"] - container["baseline"][baseline_key]["mean"])


def extract_locata_metrics(report: dict[str, Any]) -> dict[str, float]:
    per_task = report["per_task"]
    return {
        "overall_ws_ifan_deg": float(report["overall"]["ifan"]["with_silences_rmsae_deg"]["mean"]),
        "overall_ws_baseline_deg": float(report["overall"]["baseline"]["with_silences_rmsae_deg"]["mean"]),
        "overall_ws_delta_deg": mean_delta(
            report["overall"],
            "with_silences_rmsae_deg",
            "with_silences_rmsae_deg",
        ),
        "overall_ns_ifan_deg": float(report["overall"]["ifan"]["without_silences_rmsae_deg"]["mean"]),
        "overall_ns_baseline_deg": float(report["overall"]["baseline"]["without_silences_rmsae_deg"]["mean"]),
        "overall_ns_delta_deg": mean_delta(
            report["overall"],
            "without_silences_rmsae_deg",
            "without_silences_rmsae_deg",
        ),
        "task3_ws_delta_deg": mean_delta(
            per_task["task3"],
            "with_silences_rmsae_deg",
            "with_silences_rmsae_deg",
        ),
        "task3_ns_delta_deg": mean_delta(
            per_task["task3"],
            "without_silences_rmsae_deg",
            "without_silences_rmsae_deg",
        ),
        "task5_ws_delta_deg": mean_delta(
            per_task["task5"],
            "with_silences_rmsae_deg",
            "with_silences_rmsae_deg",
        ),
        "task5_ns_delta_deg": mean_delta(
            per_task["task5"],
            "without_silences_rmsae_deg",
            "without_silences_rmsae_deg",
        ),
    }


def compute_iteration_improvement(
    current_summary: dict[str, Any],
    current_locata: dict[str, Any],
    previous_summary: dict[str, Any] | None,
    previous_locata: dict[str, Any] | None,
) -> dict[str, float] | None:
    if previous_summary is None or previous_locata is None:
        return None
    current_sim = extract_summary_metrics(current_summary)
    previous_sim = extract_summary_metrics(previous_summary)
    current_loc = extract_locata_metrics(current_locata)
    previous_loc = extract_locata_metrics(previous_locata)
    improvements = {
        "sim_overall_delta_deg": previous_sim["sim_overall_delta_deg"] - current_sim["sim_overall_delta_deg"],
        "sim_hard_delta_deg": previous_sim["sim_hard_delta_deg"] - current_sim["sim_hard_delta_deg"],
        "locata_ws_ifan_deg": previous_loc["overall_ws_ifan_deg"] - current_loc["overall_ws_ifan_deg"],
        "locata_ns_ifan_deg": previous_loc["overall_ns_ifan_deg"] - current_loc["overall_ns_ifan_deg"],
    }
    improvements["mean_improvement_deg"] = float(statistics.mean(improvements.values()))
    return improvements


def assess_readiness(
    current_summary: dict[str, Any],
    current_locata: dict[str, Any],
    *,
    previous_summary: dict[str, Any] | None,
    previous_locata: dict[str, Any] | None,
    improvement_threshold_deg: float,
    task_regression_tolerance_deg: float,
) -> dict[str, Any]:
    current_sim = extract_summary_metrics(current_summary)
    current_loc = extract_locata_metrics(current_locata)
    iteration_improvement = compute_iteration_improvement(
        current_summary,
        current_locata,
        previous_summary,
        previous_locata,
    )

    overall_locata_win = (
        current_loc["overall_ws_delta_deg"] <= 0.0 and current_loc["overall_ns_delta_deg"] <= 0.0
    )
    task3_task5_stable = all(
        current_loc[key] <= task_regression_tolerance_deg
        for key in ("task3_ws_delta_deg", "task3_ns_delta_deg", "task5_ws_delta_deg", "task5_ns_delta_deg")
    )

    if iteration_improvement is not None:
        diminishing_returns = iteration_improvement["mean_improvement_deg"] < improvement_threshold_deg
    else:
        diminishing_returns = False

    if overall_locata_win and task3_task5_stable and (iteration_improvement is None or diminishing_returns):
        verdict = "ready_for_lightweighting"
    elif overall_locata_win and task3_task5_stable:
        verdict = "keep_reproduction_until_improvement_slows"
    else:
        verdict = "continue_reproduction"

    reasons = {
        "overall_locata_win": overall_locata_win,
        "task3_task5_stable": task3_task5_stable,
        "has_previous_iteration": iteration_improvement is not None,
        "diminishing_returns": diminishing_returns,
    }
    return {
        "verdict": verdict,
        "thresholds": {
            "improvement_threshold_deg": improvement_threshold_deg,
            "task_regression_tolerance_deg": task_regression_tolerance_deg,
        },
        "current": {
            "simulated": current_sim,
            "locata": current_loc,
        },
        "previous_iteration_improvement": iteration_improvement,
        "reasons": reasons,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# IFAN Stage-3 Readiness Assessment",
        "",
        f"- verdict: `{report['assessment']['verdict']}`",
        f"- current_summary: `{report['current_summary_path']}`",
        f"- current_locata: `{report['current_locata_path']}`",
        "",
        "## Current Metrics",
        "",
        f"- simulated overall delta vs baseline: `{report['assessment']['current']['simulated']['sim_overall_delta_deg']:+.4f} deg`",
        f"- simulated hard-scene delta vs baseline: `{report['assessment']['current']['simulated']['sim_hard_delta_deg']:+.4f} deg`",
        f"- LOCATA overall with-silence delta vs baseline: `{report['assessment']['current']['locata']['overall_ws_delta_deg']:+.4f} deg`",
        f"- LOCATA overall without-silence delta vs baseline: `{report['assessment']['current']['locata']['overall_ns_delta_deg']:+.4f} deg`",
        f"- LOCATA Task3 without-silence delta: `{report['assessment']['current']['locata']['task3_ns_delta_deg']:+.4f} deg`",
        f"- LOCATA Task5 without-silence delta: `{report['assessment']['current']['locata']['task5_ns_delta_deg']:+.4f} deg`",
        "",
        "## Gate Reasons",
        "",
        f"- overall_locata_win: `{report['assessment']['reasons']['overall_locata_win']}`",
        f"- task3_task5_stable: `{report['assessment']['reasons']['task3_task5_stable']}`",
        f"- diminishing_returns: `{report['assessment']['reasons']['diminishing_returns']}`",
    ]
    if report["assessment"]["previous_iteration_improvement"] is not None:
        lines.extend(
            [
                "",
                "## Previous Iteration Improvement",
                "",
                f"- mean improvement: `{report['assessment']['previous_iteration_improvement']['mean_improvement_deg']:.4f} deg`",
                f"- simulated overall delta improvement: `{report['assessment']['previous_iteration_improvement']['sim_overall_delta_deg']:.4f} deg`",
                f"- simulated hard-scene delta improvement: `{report['assessment']['previous_iteration_improvement']['sim_hard_delta_deg']:.4f} deg`",
                f"- LOCATA with-silence improvement: `{report['assessment']['previous_iteration_improvement']['locata_ws_ifan_deg']:.4f} deg`",
                f"- LOCATA without-silence improvement: `{report['assessment']['previous_iteration_improvement']['locata_ns_ifan_deg']:.4f} deg`",
            ]
        )
    return "\n".join(lines) + "\n"


def default_output_path(summary_path: str | Path) -> Path:
    stem = Path(summary_path).stem
    return Path("IFAN_Edge/outputs/stage3/analysis") / f"readiness_{stem}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assess whether the current IFAN line is ready to transition into lightweighting.")
    parser.add_argument("--current-summary", required=True, help="Current stage-3 summary.json or run directory.")
    parser.add_argument("--current-locata", required=True, help="Current LOCATA report JSON.")
    parser.add_argument("--previous-summary", default=None, help="Previous stage-3 summary.json or run directory.")
    parser.add_argument("--previous-locata", default=None, help="Previous LOCATA report JSON.")
    parser.add_argument("--improvement-threshold-deg", type=float, default=0.3)
    parser.add_argument("--task-regression-tolerance-deg", type=float, default=0.3)
    parser.add_argument("--output", default=None, help="Optional JSON output path. A Markdown report will be written alongside it.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    current_summary = load_json(args.current_summary)
    current_locata = load_json(args.current_locata)
    previous_summary = None if args.previous_summary is None else load_json(args.previous_summary)
    previous_locata = None if args.previous_locata is None else load_json(args.previous_locata)
    assessment = assess_readiness(
        current_summary,
        current_locata,
        previous_summary=previous_summary,
        previous_locata=previous_locata,
        improvement_threshold_deg=float(args.improvement_threshold_deg),
        task_regression_tolerance_deg=float(args.task_regression_tolerance_deg),
    )
    report = {
        "kind": "stage3_readiness_assessment",
        "current_summary_path": str(resolve_json_path(args.current_summary)),
        "current_locata_path": str(resolve_json_path(args.current_locata)),
        "previous_summary_path": None if args.previous_summary is None else str(resolve_json_path(args.previous_summary)),
        "previous_locata_path": None if args.previous_locata is None else str(resolve_json_path(args.previous_locata)),
        "assessment": assessment,
    }
    output_path = Path(args.output) if args.output is not None else default_output_path(args.current_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json_path": str(output_path), "markdown_path": str(markdown_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
