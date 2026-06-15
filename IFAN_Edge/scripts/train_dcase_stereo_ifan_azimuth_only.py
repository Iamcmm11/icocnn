from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

# DCASE stereo training reads real audio and should not depend on gpuRIR
# simulation.  acousticTracking imports gpuRIR eagerly, so provide a stub.
if "gpuRIR" not in sys.modules:
    sys.modules["gpuRIR"] = types.SimpleNamespace()

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ifan_edge.training.dcase_stereo import (
    DcaseStereoAzimuthOnlyTrainer,
    DcaseStereoAzimuthOnlyTrainingConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DCASE stereo azimuth-only IFAN branch.")
    parser.add_argument("--config", default="IFAN_Edge/configs/dcase_stereo_azimuth_only_c8_r2_maba_pre_readout.toml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--output-suffix", default=None)
    parser.add_argument("--cpu", action="store_true", help="Shortcut for --device cpu.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = DcaseStereoAzimuthOnlyTrainingConfig.from_toml(args.config)
    if args.cpu:
        config.device = "cpu"
    elif args.device is not None:
        config.device = args.device
    if args.epochs is not None:
        config.epochs = int(args.epochs)
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    if args.micro_batch_size is not None:
        config.micro_batch_size = int(args.micro_batch_size)
    if args.eval_batch_size is not None:
        config.eval_batch_size = int(args.eval_batch_size)
    if args.lr is not None:
        config.lr = float(args.lr)
    if args.output_suffix is not None:
        config.output_suffix = str(args.output_suffix)

    trainer = DcaseStereoAzimuthOnlyTrainer(
        config,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
    )
    summary = trainer.run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
