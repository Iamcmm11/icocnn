from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ifan_edge.eval import run_engineering_check
from ifan_edge.models import IFANModel, IFANModelConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PHAT+LMS stage-2 forward/backward engineering checks.")
    parser.add_argument(
        "--librispeech-path",
        default="datasets/LibriSpeech",
        help="Path to a LibriSpeech root or split directory. Falls back to a synthetic source when missing.",
    )
    parser.add_argument("--signal-length", type=int, default=2, help="Source signal duration in seconds.")
    parser.add_argument("--k", type=int, default=4096, help="Window length.")
    parser.add_argument("--step", type=int, default=3072, help="Window step.")
    parser.add_argument("--fs", type=int, default=16000, help="Sampling rate.")
    parser.add_argument("--r", type=int, default=2, help="Icosahedral map resolution.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_engineering_check(
        IFANModelConfig(
            r=args.r,
            phat_in_channels=1,
            aux_in_channels=1,
            final_head_pooling=False,
        ),
        librispeech_path=args.librispeech_path,
        signal_length=args.signal_length,
        k=args.k,
        step=args.step,
        fs=args.fs,
    )
    target_gap_percent = 100.0 * (summary["trainable_params"] - 70_000) / 70_000.0

    print("Source dataset:", summary["source_dataset"])
    print("Windowed microphone batch shape:", summary["windowed_microphone_batch_shape"])
    print("Input shape:", summary["input_shape"])
    print("PHAT shape:", summary["phat_shape"])
    print("LMS shape:", summary["lms_shape"])
    print("DOA tensor shape:", summary["doa_tensor_shape"])
    print("Output shape:", summary["output_shape"])
    print("PHAT attention shape:", summary["phat_attention_shape"])
    print("Aux attention shape:", summary["lms_attention_shape"])
    print("Trainable params:", summary["trainable_params"])
    print("Param target gap (%):", float(target_gap_percent))
    print("Loss:", summary["loss"])
    print("Finite input windows:", summary["finite_input_windows"])
    print("Finite output:", summary["finite_output"])
    print("Finite gradients:", summary["finite_gradients"])
    print("Nonzero gradient params:", summary["nonzero_gradient_params"])

    assert abs(summary["trainable_params"] - 125_440) <= 64

    print("PHAT + LMS stage-2 engineering check passed.")


if __name__ == "__main__":
    main()
