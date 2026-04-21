from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_summary_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "summary.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"Could not find summary.json from {path}")
    return candidate.resolve()


def load_summary(path: str | Path) -> dict[str, object]:
    summary_path = resolve_summary_path(path)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def scenario_map(summary: dict[str, object]) -> dict[str, dict[str, object]]:
    scenarios = summary["baseline_compare"]["scenarios"]
    return {row["name"]: row for row in scenarios}


def summarize_scene_group(summaries: dict[str, dict[str, object]], names: list[str]) -> dict[str, float]:
    if not names:
        return {"mean_delta": 0.0, "max_delta": 0.0, "min_delta": 0.0}
    deltas = np.asarray([float(summaries[name]["rmsae_delta_deg"]) for name in names], dtype=np.float64)
    return {
        "mean_delta": float(deltas.mean()),
        "max_delta": float(deltas.max()),
        "min_delta": float(deltas.min()),
    }


def classify_transition(
    before_summary: dict[str, object],
    after_summary: dict[str, object],
    *,
    easy_scene_names: tuple[str, ...] = ("scene_1", "scene_2"),
    hard_scene_names: tuple[str, ...] = ("scene_3", "scene_4"),
) -> dict[str, object]:
    before_scenarios = scenario_map(before_summary)
    after_scenarios = scenario_map(after_summary)
    shared_names = [name for name in before_scenarios.keys() if name in after_scenarios]

    scenario_transitions = []
    for name in shared_names:
        before = before_scenarios[name]
        after = after_scenarios[name]
        scenario_transitions.append(
            {
                "name": name,
                "snr_db": float(after["snr_db"]),
                "t60_s": float(after["t60_s"]),
                "before_delta_deg": float(before["rmsae_delta_deg"]),
                "after_delta_deg": float(after["rmsae_delta_deg"]),
                "delta_change_deg": float(after["rmsae_delta_deg"] - before["rmsae_delta_deg"]),
            }
        )

    overall_before = float(before_summary["baseline_compare"]["mean_rmsae_deg"]["delta"])
    overall_after = float(after_summary["baseline_compare"]["mean_rmsae_deg"]["delta"])
    hard_before = float(before_summary["baseline_compare"]["hard_scenarios_mean_rmsae_deg"]["delta"])
    hard_after = float(after_summary["baseline_compare"]["hard_scenarios_mean_rmsae_deg"]["delta"])

    before_easy_stats = summarize_scene_group(before_scenarios, [name for name in easy_scene_names if name in before_scenarios])
    after_easy_stats = summarize_scene_group(after_scenarios, [name for name in easy_scene_names if name in after_scenarios])
    before_hard_stats = summarize_scene_group(before_scenarios, [name for name in hard_scene_names if name in before_scenarios])
    after_hard_stats = summarize_scene_group(after_scenarios, [name for name in hard_scene_names if name in after_scenarios])
    easy_transition = summarize_scene_group(
        {row["name"]: {"rmsae_delta_deg": row["delta_change_deg"]} for row in scenario_transitions},
        [name for name in easy_scene_names if name in {row["name"] for row in scenario_transitions}],
    )
    hard_transition = summarize_scene_group(
        {row["name"]: {"rmsae_delta_deg": row["delta_change_deg"]} for row in scenario_transitions},
        [name for name in hard_scene_names if name in {row["name"] for row in scenario_transitions}],
    )

    improves_hard = hard_after < hard_before
    harms_easy = easy_transition["mean_delta"] > 0.0
    net_better = overall_after < overall_before
    no_material_change = np.isclose(overall_after, overall_before) and np.isclose(hard_after, hard_before)

    if no_material_change:
        verdict = "no_material_change"
    elif improves_hard and net_better:
        verdict = "improves_hard_scenes_without_hurting_overall"
    elif improves_hard and harms_easy:
        verdict = "hard_scene_gain_with_easy_scene_cost"
    elif not improves_hard and not net_better:
        verdict = "overall_regression"
    else:
        verdict = "mixed_tradeoff"

    return {
        "before": {
            "overall_mean_delta_deg": overall_before,
            "hard_scenes_mean_delta_deg": hard_before,
            "easy_scenes_mean_delta_deg": before_easy_stats["mean_delta"],
        },
        "after": {
            "overall_mean_delta_deg": overall_after,
            "hard_scenes_mean_delta_deg": hard_after,
            "easy_scenes_mean_delta_deg": after_easy_stats["mean_delta"],
        },
        "changes": {
            "overall_mean_delta_deg": float(overall_after - overall_before),
            "hard_scenes_mean_delta_deg": float(hard_after - hard_before),
            "easy_scenes_mean_delta_deg": float(easy_transition["mean_delta"]),
            "hard_scenes_scene_delta_change_deg": float(hard_transition["mean_delta"]),
        },
        "classification": {
            "improves_hard_scenes": bool(improves_hard),
            "harms_easy_scenes": bool(harms_easy),
            "net_improves_overall": bool(net_better),
            "verdict": verdict,
        },
        "scenarios": scenario_transitions,
        "group_stats": {
            "before_easy_scenes": before_easy_stats,
            "after_easy_scenes": after_easy_stats,
            "before_hard_scenes": before_hard_stats,
            "after_hard_scenes": after_hard_stats,
            "easy_scene_transition": easy_transition,
            "hard_scene_transition": hard_transition,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two stage-3 run summaries and produce a compact gate report.")
    parser.add_argument("--before", required=True, help="Run directory or summary.json for the baseline side of the comparison.")
    parser.add_argument("--after", required=True, help="Run directory or summary.json for the changed side of the comparison.")
    parser.add_argument("--before-label", default="before")
    parser.add_argument("--after-label", default="after")
    parser.add_argument("--output", default=None, help="Optional path to write the JSON report.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    before_summary = load_summary(args.before)
    after_summary = load_summary(args.after)

    report = {
        "kind": "stage3_run_comparison",
        "before_label": str(args.before_label),
        "after_label": str(args.after_label),
        "before_summary_path": str(resolve_summary_path(args.before)),
        "after_summary_path": str(resolve_summary_path(args.after)),
        "comparison": classify_transition(before_summary, after_summary),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
