from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EXPERIMENT_ROLE = "mainline_baseline"
DEFAULT_SRP_VARIANT = "paper_original"
DEFAULT_TEMPORAL_CONV_VARIANT = "standard_1d"
DEFAULT_TEMPORAL_MODULE = "conv"
DEFAULT_BENCHMARK_SUITE = "simulated_4scene+hard_scenes+locata_eval_benchmark2_task1_3_5"
DEFAULT_MAINLINE_ANCHOR_RUN = (
    "IFAN_Edge/outputs/stage3/ifan_stage3_full20_freqblock_paper_original_20260419_005314"
)
DEFAULT_MAINLINE_LOCATA_REPORT = "IFAN_Edge/outputs/stage3/analysis/locata_eval_benchmark2_best.json"


def build_experiment_contract(config: Any) -> dict[str, Any]:
    return {
        "stage": getattr(config, "stage_name", "stage_03"),
        "experiment_role": getattr(config, "experiment_role", DEFAULT_EXPERIMENT_ROLE),
        "model_topology": "paper_dual_mainline",
        "feature_pair": "phat+lms",
        "pre_fusion_pooling": getattr(config, "pre_fusion_pooling", True),
        "srp_variant": getattr(config, "srp_variant", DEFAULT_SRP_VARIANT),
        "lms_backend": getattr(config, "lms_backend", "frequency_block"),
        "temporal_conv_variant": getattr(config, "temporal_conv_variant", DEFAULT_TEMPORAL_CONV_VARIANT),
        "temporal_module": getattr(config, "temporal_module", DEFAULT_TEMPORAL_MODULE),
        "primary_benchmark_suite": getattr(config, "primary_benchmark_suite", DEFAULT_BENCHMARK_SUITE),
        "baseline_anchor": {
            "run_dir": getattr(config, "mainline_anchor_run", DEFAULT_MAINLINE_ANCHOR_RUN),
            "locata_report": getattr(config, "mainline_anchor_locata_report", DEFAULT_MAINLINE_LOCATA_REPORT),
            "baseline_checkpoint_path": getattr(
                config,
                "baseline_checkpoint_path",
                "models/1sourceTracking_icoCNN_robot_K4096_r2_model.bin",
            ),
        },
    }


def load_config_from_inputs(checkpoint_path: str | None, config_path: str | None) -> Any:
    if config_path is not None:
        with Path(config_path).open("rb") as handle:
            raw = tomllib.load(handle)
        data = raw.get("data", {})
        model = raw.get("model", {})
        training = raw.get("training", {})
        evaluation = raw.get("evaluation", {})
        paths = raw.get("paths", {})
        contract = raw.get("contract", {})
        return SimpleNamespace(
            stage_name=str(raw.get("project", {}).get("stage_name", "stage_03")),
            train_split=str(paths.get("train_split", "train-clean-100")),
            test_split=str(paths.get("test_split", "test-clean")),
            fs=int(data.get("fs", 16000)),
            k=int(data.get("k", 4096)),
            step=int(data.get("step", 3072)),
            trajectory_seconds=int(data.get("trajectory_seconds", 20)),
            epochs=int(training.get("epochs", 40)),
            phase1_epochs=int(training.get("phase1_epochs", 20)),
            train_snr_min_phase1=float(training.get("train_snr_min_phase1", 30.0)),
            train_snr_max_phase1=float(training.get("train_snr_max_phase1", 30.0)),
            train_snr_min_phase2=float(training.get("train_snr_min_phase2", 5.0)),
            train_snr_max_phase2=float(training.get("train_snr_max_phase2", 30.0)),
            train_t60_min=float(training.get("train_t60_min", 0.2)),
            train_t60_max=float(training.get("train_t60_max", 1.3)),
            pre_fusion_pooling=bool(model.get("pre_fusion_pooling", True)),
            final_head_pooling=bool(model.get("final_head_pooling", False)),
            lms_backend=str(data.get("lms_backend", "frequency_block")),
            baseline_checkpoint_path=str(
                paths.get("baseline_checkpoint_path", "models/1sourceTracking_icoCNN_robot_K4096_r2_model.bin")
            ),
            experiment_role=str(contract.get("experiment_role", DEFAULT_EXPERIMENT_ROLE)),
            srp_variant=str(contract.get("srp_variant", DEFAULT_SRP_VARIANT)),
            temporal_conv_variant=str(contract.get("temporal_conv_variant", DEFAULT_TEMPORAL_CONV_VARIANT)),
            temporal_module=str(contract.get("temporal_module", DEFAULT_TEMPORAL_MODULE)),
            primary_benchmark_suite=str(contract.get("primary_benchmark_suite", DEFAULT_BENCHMARK_SUITE)),
            mainline_anchor_run=str(contract.get("mainline_anchor_run", DEFAULT_MAINLINE_ANCHOR_RUN)),
            mainline_anchor_locata_report=str(
                contract.get("mainline_anchor_locata_report", DEFAULT_MAINLINE_LOCATA_REPORT)
            ),
        )
    if checkpoint_path is None:
        raise ValueError("Either --config or --checkpoint must be provided.")
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return SimpleNamespace(**checkpoint["training_config"])


def load_optional_json(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protocol_status(paper_value: Any, local_value: Any) -> str:
    if paper_value == local_value:
        return "match"
    return "gap"


def build_protocol_rows(config: Any, locata_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    locata_tasks = None
    silence_reporting = "missing"
    if locata_report is not None:
        locata_tasks = tuple(int(task) for task in locata_report.get("tasks", []))
        if "overall" in locata_report and "with_silences_rmsae_deg" in locata_report["overall"].get("ifan", {}):
            silence_reporting = "with+without"

    rows = [
        {
            "item": "Training dataset",
            "paper": "LibriSpeech train-clean-100",
            "local": config.train_split,
            "status": protocol_status("train-clean-100", config.train_split),
            "impact": "high",
            "note": "Paper and local training split should match exactly.",
        },
        {
            "item": "Simulated test dataset",
            "paper": "LibriSpeech test-clean",
            "local": config.test_split,
            "status": protocol_status("test-clean", config.test_split),
            "impact": "high",
            "note": "Paper simulated evaluation uses test-clean.",
        },
        {
            "item": "Sampling rate",
            "paper": 16000,
            "local": config.fs,
            "status": protocol_status(16000, config.fs),
            "impact": "high",
            "note": "Paper values are reported at 16 kHz.",
        },
        {
            "item": "Frame length K",
            "paper": 4096,
            "local": config.k,
            "status": protocol_status(4096, config.k),
            "impact": "high",
            "note": "Paper values are reported at K=4096.",
        },
        {
            "item": "Frame step",
            "paper": 3072,
            "local": config.step,
            "status": protocol_status(3072, config.step),
            "impact": "high",
            "note": "Paper values are reported at step=3072.",
        },
        {
            "item": "Trajectory duration",
            "paper": "20 s",
            "local": f"{config.trajectory_seconds} s",
            "status": protocol_status("20 s", f"{config.trajectory_seconds} s"),
            "impact": "medium",
            "note": "Current mainline should stay on the paper 20-second trajectory setting.",
        },
        {
            "item": "Training schedule",
            "paper": "80 epochs (20 + 60)",
            "local": f"{config.epochs} epochs ({config.phase1_epochs} + {config.epochs - config.phase1_epochs})",
            "status": "match" if config.epochs == 80 and config.phase1_epochs == 20 else "gap",
            "impact": "high",
            "note": "40-epoch runs are still useful, but they are below the paper training budget.",
        },
        {
            "item": "Phase-1 SNR",
            "paper": "30 dB",
            "local": f"{config.train_snr_min_phase1:.1f} to {config.train_snr_max_phase1:.1f} dB",
            "status": "match"
            if config.train_snr_min_phase1 == 30.0 and config.train_snr_max_phase1 == 30.0
            else "gap",
            "impact": "medium",
            "note": "Phase 1 should remain fixed at 30 dB.",
        },
        {
            "item": "Phase-2 SNR range",
            "paper": "5 to 30 dB",
            "local": f"{config.train_snr_min_phase2:.1f} to {config.train_snr_max_phase2:.1f} dB",
            "status": "match"
            if config.train_snr_min_phase2 == 5.0 and config.train_snr_max_phase2 == 30.0
            else "gap",
            "impact": "medium",
            "note": "Phase 2 should span 5 to 30 dB.",
        },
        {
            "item": "Training T60 range",
            "paper": "0.2 to 1.3 s",
            "local": f"{config.train_t60_min:.1f} to {config.train_t60_max:.1f} s",
            "status": "match"
            if config.train_t60_min == 0.2 and config.train_t60_max == 1.3
            else "gap",
            "impact": "medium",
            "note": "The paper training distribution caps T60 at 1.3 seconds.",
        },
        {
            "item": "Simulated metric",
            "paper": "RMSAE, ignore first 5 frames",
            "local": "RMSAE, ignore first 5 frames",
            "status": "match",
            "impact": "high",
            "note": "The local stage-3 evaluation path trims the first 5 frames before RMSAE.",
        },
        {
            "item": "Simulated evaluation suite",
            "paper": "paper simulation benchmark",
            "local": "four-scene matched baseline compare + hard-scene summary",
            "status": "gap",
            "impact": "high",
            "note": "Current local simulated acceptance is a compact engineering suite, not the paper's full reported benchmark.",
        },
        {
            "item": "LOCATA task scope",
            "paper": "(1, 3, 5)",
            "local": None if locata_tasks is None else str(locata_tasks),
            "status": "missing" if locata_tasks is None else ("match" if locata_tasks == (1, 3, 5) else "gap"),
            "impact": "high",
            "note": "Paper LOCATA evaluation is limited to the single-source tasks.",
        },
        {
            "item": "LOCATA silence reporting",
            "paper": "with + without silences",
            "local": silence_reporting,
            "status": "missing" if locata_report is None else ("match" if silence_reporting == "with+without" else "gap"),
            "impact": "medium",
            "note": "Paper tables separate the two silence protocols.",
        },
        {
            "item": "LMS backend implementation",
            "paper": "reference/original behavior",
            "local": config.lms_backend,
            "status": "context" if config.lms_backend == "frequency_block" else "match",
            "impact": "high",
            "note": "frequency_block is the current training mainline, but it still needs equivalence checks against the reference backend.",
        },
    ]
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    high_priority_gaps = [row for row in rows if row["impact"] == "high" and row["status"] in {"gap", "context", "missing"}]
    return {
        "counts": counts,
        "high_priority_gaps": high_priority_gaps,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# IFAN Stage-3 Protocol Audit",
        "",
        f"- config_source: `{report['config_source']}`",
        f"- locata_report: `{report['locata_report']}`",
        "",
        "| Item | Paper | Local | Status | Impact | Note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['item']} | `{row['paper']}` | `{row['local']}` | `{row['status']}` | `{row['impact']}` | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- counts: `{json.dumps(report['summary']['counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- high_priority_gap_count: `{len(report['summary']['high_priority_gaps'])}`",
        ]
    )
    return "\n".join(lines) + "\n"


def default_output_path(config_source: str | None) -> Path:
    stem = Path(config_source or "stage3_default").stem
    return Path("IFAN_Edge/outputs/stage3/analysis") / f"protocol_audit_{stem}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the local IFAN stage-3 protocol against the paper protocol.")
    parser.add_argument("--checkpoint", default=None, help="Optional stage-3 checkpoint for loading embedded config.")
    parser.add_argument("--config", default="IFAN_Edge/configs/stage3_default.toml", help="Optional stage-3 config TOML.")
    parser.add_argument("--locata-report", default=None, help="Optional LOCATA evaluation JSON for task/silence protocol checks.")
    parser.add_argument("--output", default=None, help="Optional JSON output path. A Markdown report will be written alongside it.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config_from_inputs(args.checkpoint, args.config)
    locata_report = load_optional_json(args.locata_report)
    rows = build_protocol_rows(config, locata_report)
    report = {
        "kind": "stage3_protocol_audit",
        "config_source": str(args.config if args.config is not None else args.checkpoint),
        "locata_report": None if args.locata_report is None else str(Path(args.locata_report).resolve()),
        "experiment_contract": build_experiment_contract(config),
        "rows": rows,
        "summary": summarize_rows(rows),
    }
    output_path = Path(args.output) if args.output is not None else default_output_path(args.config or args.checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json_path": str(output_path), "markdown_path": str(markdown_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
