"""Run fixed baseline + MABA + ablation experiments."""

import argparse
import copy
import json
import os

from maba_doa.train_maba_doa import load_config, run_training


def main():
    parser = argparse.ArgumentParser(description="Run baseline and MABA ablation suite.")
    parser.add_argument("--config", type=str, default="maba_doa/configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    if args.epochs is not None:
        base_cfg["training"]["epochs"] = int(args.epochs)
    if args.cpu:
        base_cfg["device"] = "cpu"

    variants = ["baseline", "maba", "ablation_no_gate", "ablation_no_state"]
    summaries = []
    for variant in variants:
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["variant"] = variant
        summary = run_training(cfg, output_suffix=variant)
        summaries.append(summary)

    # Save one combined report under the most recent run directory root.
    output_root = base_cfg["experiment"]["output_root"]
    os.makedirs(output_root, exist_ok=True)
    report_path = os.path.join(output_root, "ablation_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"runs": summaries}, f, indent=2, ensure_ascii=False)

    print(json.dumps({"report_path": report_path, "runs": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
