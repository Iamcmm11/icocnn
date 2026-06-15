from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

# DCASE evaluation reads real audio and never runs simulation.  acousticTracking
# imports gpuRIR eagerly, so stub it out on hosts where the simulator is not
# usable.
if "gpuRIR" not in sys.modules:
    sys.modules["gpuRIR"] = types.SimpleNamespace()

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ifan_edge.dcase_stereo import (
    build_model_from_checkpoint,
    build_report,
    checkpoint_model_kind,
    create_preprocessor,
    evaluate_model_on_prepared,
    load_manifest,
    prepare_rows,
    stereo_proxy_positions,
    write_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an IFAN checkpoint on the filtered DCASE2025 stereo subset.")
    parser.add_argument(
        "--checkpoint",
        default="IFAN_Edge/outputs/stage3/ifan_stage3_long80_c8_r2_maba_pre_readout_20260518_215408/checkpoints/best_rmsae.pt",
    )
    parser.add_argument("--dataset-root", default="datasets/DCASE2025_Task3")
    parser.add_argument("--manifest", default="datasets/DCASE2025_Task3/locata_like_devtest_strict/manifest_all.csv")
    parser.add_argument("--output", default="IFAN_Edge/outputs/stage3/analysis/dcase2025_locata_like_devtest_strict_ifan_c8_r2_maba_pre_readout_best.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--cpu", action="store_true", help="Shortcut for --device cpu.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stereo-baseline-m", type=float, default=0.08)
    parser.add_argument("--exclude-initial-windows", type=int, default=5)
    parser.add_argument("--no-vad-mask", action="store_true", help="Disable the all-ones VAD mask passed through the IFAN frontend.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    requested_device = "cpu" if args.cpu else args.device
    device = torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model, model_config, load_notes = build_model_from_checkpoint(checkpoint, device=device)

    training_config = dict(checkpoint["training_config"])
    rn = stereo_proxy_positions(args.stereo_baseline_m)
    preprocessor = create_preprocessor(
        training_config,
        rn=rn,
        device=device,
        apply_vad=not args.no_vad_mask and bool(training_config.get("apply_vad", True)),
    )

    rows = load_manifest(manifest_path, limit=args.limit)
    prepared = prepare_rows(
        rows,
        dataset_root,
        fs=int(training_config["fs"]),
        k=int(training_config["k"]),
        step=int(training_config["step"]),
        progress_label="eval_prepare",
    )
    reports = evaluate_model_on_prepared(
        model=model,
        preprocessor=preprocessor,
        prepared=prepared,
        device=device,
        batch_size=int(args.batch_size),
        exclude_initial_windows=int(args.exclude_initial_windows),
        progress_callback=lambda done, total: print(
            json.dumps(
                {
                    "event": "dcase_eval_progress",
                    "processed": int(done),
                    "total": int(total),
                    "reports": int(done),
                },
                ensure_ascii=False,
            ),
            flush=True,
        ),
    )

    report = build_report(
        reports=reports,
        checkpoint=str(checkpoint_path),
        manifest=str(manifest_path),
        dataset_root=str(dataset_root),
        device=str(device),
        limit=args.limit,
        exclude_initial_windows=int(args.exclude_initial_windows),
        stereo_baseline_m=float(args.stereo_baseline_m),
        rn=rn,
        config={
            "fs": int(training_config["fs"]),
            "k": int(training_config["k"]),
            "step": int(training_config["step"]),
            "r": int(training_config["r"]),
            "apply_vad": not args.no_vad_mask and bool(training_config.get("apply_vad", True)),
            "lms_backend": str(training_config.get("lms_backend", "")),
            "lms_map_mode": str(training_config.get("lms_map_mode", "")),
            "lms_update_mode": str(training_config.get("lms_update_mode", "")),
            "srp_variant": str(training_config.get("srp_variant", "")),
        },
        model_payload={
            "epoch": int(checkpoint.get("epoch", -1)),
            "model_kind": checkpoint_model_kind(checkpoint),
            "model_config": checkpoint.get("model_config", checkpoint.get("backbone_model_config", {})),
            "backbone_model_config": checkpoint.get("backbone_model_config"),
            "azimuth_head_config": checkpoint.get("azimuth_head_config"),
            "metrics": checkpoint.get("metrics", {}),
            "load_notes": load_notes,
        },
    )
    json_path, markdown_path = write_report(output_path, report)
    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "markdown_path": str(markdown_path),
                "clips": report["overall"]["count"],
                "doa_error_deg": report["overall"]["doa_error_deg"]["mean"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
