from __future__ import annotations

import argparse
import atexit
import faulthandler
import json
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gpuRIR

from ifan_edge.training import IFANTrainingConfig, IFANTrainingPipeline


def emit_process_event(event: str, **fields) -> None:
    payload = {
        "event": event,
        "pid": os.getpid(),
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def install_process_diagnostics() -> None:
    def on_exit() -> None:
        emit_process_event("process_exit")

    def on_signal(signum, _frame) -> None:
        signal_name = signal.Signals(signum).name
        emit_process_event("signal_received", signal=signal_name)
        raise SystemExit(128 + int(signum))

    atexit.register(on_exit)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        current_handler = signal.getsignal(sig)
        # Preserve handlers that were already explicitly ignored, such as SIGHUP under nohup.
        if current_handler is signal.SIG_IGN:
            continue
        signal.signal(sig, on_signal)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run IFAN stage-3 training, validation, and baseline comparison.")
    parser.add_argument("--config", default="IFAN_Edge/configs/stage3_default.toml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--phase1-epochs", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=None, help="Override stage-3 train dataset size per epoch.")
    parser.add_argument("--val-size", type=int, default=None, help="Override fixed validation dataset size.")
    parser.add_argument("--scenario-eval-size", type=int, default=None, help="Override four-scenario evaluation size.")
    parser.add_argument("--trajectory-seconds", type=int, default=None, help="Override trajectory duration for smoke runs.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--cpu", action="store_true", help="Shortcut for --device cpu.")
    parser.add_argument("--output-suffix", default=None)
    parser.add_argument("--resume-checkpoint", default=None, help="Resume stage-3 training from a saved checkpoint.")
    parser.add_argument("--resume-output-dir", default=None, help="Reuse an existing stage-3 output directory when resuming.")
    parser.add_argument("--resume-log", default=None, help="Existing training log used to recover epoch history for resumed runs.")
    parser.add_argument("--ifan-init-checkpoint", default=None, help="Initialize IFAN weights from a stage-3 checkpoint without resuming optimizer/epoch state.")
    parser.add_argument("--saf-lite", action="store_true", help="Enable fixed SAF-lite structured pruning for C=16 ConvIco layers.")
    parser.add_argument("--saf-lite-keep-per-8", type=int, default=None, help="Retained input channels per 8-channel block for SAF-lite.")
    parser.add_argument("--experiment-role", default=None, help="Override the experiment contract role recorded in summary metadata.")
    parser.add_argument(
        "--srp-variant",
        choices=("paper_original", "lc_reference", "lc_edge"),
        default=None,
        help="Choose the PHAT SRP frontend variant.",
    )
    parser.add_argument("--phat-sinc-half-width", type=int, default=None, help="Half width of the sinc interpolation support used by lc_* PHAT variants.")
    parser.add_argument(
        "--temporal-conv-variant",
        choices=("standard_1d", "depthwise_separable_1d"),
        default=None,
        help="Override the recorded temporal convolution variant tag for experiment comparison.",
    )
    parser.add_argument(
        "--temporal-module",
        choices=("conv", "maba"),
        default=None,
        help="Override the recorded temporal module tag for experiment comparison.",
    )
    parser.add_argument("--input-ablation-mode", choices=("none", "phat_only", "lms_only"), default=None)
    parser.add_argument("--branch-channels", type=int, default=None, help="Override IFAN branch width for lightweight experiments.")
    parser.add_argument("--final-head-pooling", action="store_true", help="Apply the optional final pooling stage before SoftArgMax.")
    parser.add_argument("--map-refiner", choices=("none", "maba"), default=None, help="Optional map-level temporal refiner inserted before SoftArgMax.")
    parser.add_argument("--map-refiner-position", choices=("pre_softargmax", "pre_readout"), default=None, help="Insertion point for the optional MABA refiner.")
    parser.add_argument("--map-maba-d-model", type=int, default=None, help="Hidden size for the optional map-level MABA refiner.")
    parser.add_argument("--map-maba-state-dim", type=int, default=None, help="State size for the optional map-level MABA refiner.")
    parser.add_argument("--map-maba-conv-kernel", type=int, default=None, help="Depthwise temporal kernel for the optional map-level MABA refiner.")
    parser.add_argument("--map-maba-dropout", type=float, default=None, help="Dropout for the optional map-level MABA refiner.")
    parser.add_argument("--map-maba-no-residual", action="store_true", help="Disable residual add in the optional map-level MABA refiner.")
    parser.add_argument("--map-maba-no-gate", action="store_true", help="Disable dynamic gating in the optional map-level MABA refiner.")
    parser.add_argument("--map-maba-no-state", action="store_true", help="Disable recurrent state updates in the optional map-level MABA refiner.")
    parser.add_argument("--weak-map-refiner", choices=("none", "maba"), default=None, help="Optional second lightweight refiner applied before SoftArgMax.")
    parser.add_argument("--weak-map-maba-d-model", type=int, default=None, help="Hidden size for the weak second refiner.")
    parser.add_argument("--weak-map-maba-state-dim", type=int, default=None, help="State size for the weak second refiner.")
    parser.add_argument("--weak-map-maba-conv-kernel", type=int, default=None, help="Depthwise temporal kernel for the weak second refiner.")
    parser.add_argument("--weak-map-maba-dropout", type=float, default=None, help="Dropout for the weak second refiner.")
    parser.add_argument("--weak-map-maba-no-residual", action="store_true", help="Disable residual add in the weak second refiner.")
    parser.add_argument("--weak-map-maba-no-gate", action="store_true", help="Disable dynamic gating in the weak second refiner.")
    parser.add_argument("--weak-map-maba-no-state", action="store_true", help="Disable recurrent state updates in the weak second refiner.")
    parser.add_argument("--lms-plain", action="store_true", help="Disable NLMS-style normalization in the LMS branch.")
    parser.add_argument("--lms-no-self-pairs", action="store_true", help="Exclude diagonal microphone pairs from the LMS branch.")
    parser.add_argument("--lms-no-map-normalize", action="store_true", help="Disable per-map normalization in the LMS branch output.")
    parser.add_argument("--lms-map-mode", choices=("tau_sample", "peak_proximity"), default=None, help="Choose how LMS filters are converted into icosahedral maps.")
    parser.add_argument("--lms-peak-sigma", type=float, default=None, help="Kernel width for peak_proximity LMS map mode.")
    parser.add_argument("--lms-update-mode", choices=("frame_reset", "trajectory_tracking"), default=None, help="Choose whether LMS weights reset every frame or track across the whole trajectory.")
    parser.add_argument("--lms-backend", choices=("time_reference", "frequency_block"), default=None, help="Choose the LMS estimation backend.")
    parser.add_argument("--lms-block-size", type=int, default=None, help="Block size for the frequency_block LMS backend.")
    parser.add_argument("--lms-fft-size", type=int, default=None, help="FFT size for the frequency_block LMS backend.")
    parser.add_argument("--lms-paper-original", action="store_true", help="Use the paper-style LMS preset: plain LMS, trajectory tracking, tau-sample readout, and full pair set.")
    parser.add_argument("--batch-size-phase1", type=int, default=None)
    parser.add_argument("--batch-size-phase2", type=int, default=None)
    parser.add_argument("--micro-batch-size-phase1", type=int, default=None)
    parser.add_argument("--micro-batch-size-phase2", type=int, default=None)
    parser.add_argument("--lr-phase1", type=float, default=None)
    parser.add_argument("--lr-phase2", type=float, default=None)
    parser.add_argument("--train-t60-min", type=float, default=None)
    parser.add_argument("--train-t60-max", type=float, default=None)
    parser.add_argument("--train-snr-min-phase1", type=float, default=None)
    parser.add_argument("--train-snr-max-phase1", type=float, default=None)
    parser.add_argument("--train-snr-min-phase2", type=float, default=None)
    parser.add_argument("--train-snr-max-phase2", type=float, default=None)
    parser.add_argument("--validation-snr-min", type=float, default=None)
    parser.add_argument("--validation-snr-max", type=float, default=None)
    return parser


def main() -> None:
    faulthandler.enable(all_threads=True)
    install_process_diagnostics()
    args = build_parser().parse_args()
    config = IFANTrainingConfig.from_toml(args.config)

    if args.seed is not None:
        config.seed = int(args.seed)
    if args.epochs is not None:
        config.epochs = int(args.epochs)
    if args.phase1_epochs is not None:
        config.phase1_epochs = int(args.phase1_epochs)
    if args.train_size is not None:
        config.train_dataset_size = int(args.train_size)
    if args.val_size is not None:
        config.validation_dataset_size = int(args.val_size)
    if args.scenario_eval_size is not None:
        config.scenario_eval_size = int(args.scenario_eval_size)
    if args.trajectory_seconds is not None:
        config.trajectory_seconds = int(args.trajectory_seconds)
    if args.output_suffix is not None:
        config.output_suffix = str(args.output_suffix)
    if args.ifan_init_checkpoint is not None:
        config.ifan_init_checkpoint_path = str(args.ifan_init_checkpoint)
    if args.saf_lite:
        config.saf_lite_enabled = True
    if args.saf_lite_keep_per_8 is not None:
        config.saf_lite_keep_per_8 = int(args.saf_lite_keep_per_8)
    if args.experiment_role is not None:
        config.experiment_role = str(args.experiment_role)
    if args.srp_variant is not None:
        config.srp_variant = str(args.srp_variant)
    if args.phat_sinc_half_width is not None:
        config.phat_sinc_half_width = int(args.phat_sinc_half_width)
    if args.temporal_conv_variant is not None:
        config.temporal_conv_variant = str(args.temporal_conv_variant)
    if args.temporal_module is not None:
        config.temporal_module = str(args.temporal_module)
    if args.input_ablation_mode is not None:
        config.input_ablation_mode = str(args.input_ablation_mode)
    if args.branch_channels is not None:
        config.branch_channels = int(args.branch_channels)
    if args.final_head_pooling:
        config.final_head_pooling = True
    if args.map_refiner is not None:
        config.map_refiner = str(args.map_refiner)
    if args.map_refiner_position is not None:
        config.map_refiner_position = str(args.map_refiner_position)
    if args.map_maba_d_model is not None:
        config.map_maba.d_model = int(args.map_maba_d_model)
    if args.map_maba_state_dim is not None:
        config.map_maba.state_dim = int(args.map_maba_state_dim)
    if args.map_maba_conv_kernel is not None:
        config.map_maba.conv_kernel = int(args.map_maba_conv_kernel)
    if args.map_maba_dropout is not None:
        config.map_maba.dropout = float(args.map_maba_dropout)
    if args.map_maba_no_residual:
        config.map_maba.use_residual = False
    if args.map_maba_no_gate:
        config.map_maba.use_gate = False
    if args.map_maba_no_state:
        config.map_maba.use_state = False
    if args.weak_map_refiner is not None:
        config.weak_map_refiner = str(args.weak_map_refiner)
    if args.weak_map_maba_d_model is not None:
        config.weak_map_maba.d_model = int(args.weak_map_maba_d_model)
    if args.weak_map_maba_state_dim is not None:
        config.weak_map_maba.state_dim = int(args.weak_map_maba_state_dim)
    if args.weak_map_maba_conv_kernel is not None:
        config.weak_map_maba.conv_kernel = int(args.weak_map_maba_conv_kernel)
    if args.weak_map_maba_dropout is not None:
        config.weak_map_maba.dropout = float(args.weak_map_maba_dropout)
    if args.weak_map_maba_no_residual:
        config.weak_map_maba.use_residual = False
    if args.weak_map_maba_no_gate:
        config.weak_map_maba.use_gate = False
    if args.weak_map_maba_no_state:
        config.weak_map_maba.use_state = False
    if args.lms_plain:
        config.lms_normalized = False
    if args.lms_no_self_pairs:
        config.lms_include_self_pairs = False
    if args.lms_no_map_normalize:
        config.lms_map_normalize = False
    if args.lms_map_mode is not None:
        config.lms_map_mode = args.lms_map_mode
    if args.lms_peak_sigma is not None:
        config.lms_peak_sigma = float(args.lms_peak_sigma)
    if args.lms_update_mode is not None:
        config.lms_update_mode = args.lms_update_mode
    if args.lms_backend is not None:
        config.lms_backend = args.lms_backend
    if args.lms_block_size is not None:
        config.lms_block_size = int(args.lms_block_size)
    if args.lms_fft_size is not None:
        config.lms_fft_size = int(args.lms_fft_size)
    if args.batch_size_phase1 is not None:
        config.batch_size_phase1 = int(args.batch_size_phase1)
    if args.batch_size_phase2 is not None:
        config.batch_size_phase2 = int(args.batch_size_phase2)
    if args.micro_batch_size_phase1 is not None:
        config.micro_batch_size_phase1 = int(args.micro_batch_size_phase1)
    if args.micro_batch_size_phase2 is not None:
        config.micro_batch_size_phase2 = int(args.micro_batch_size_phase2)
    if args.lr_phase1 is not None:
        config.lr_phase1 = float(args.lr_phase1)
    if args.lr_phase2 is not None:
        config.lr_phase2 = float(args.lr_phase2)
    if args.train_t60_min is not None:
        config.train_t60_min = float(args.train_t60_min)
    if args.train_t60_max is not None:
        config.train_t60_max = float(args.train_t60_max)
    if args.train_snr_min_phase1 is not None:
        config.train_snr_min_phase1 = float(args.train_snr_min_phase1)
    if args.train_snr_max_phase1 is not None:
        config.train_snr_max_phase1 = float(args.train_snr_max_phase1)
    if args.train_snr_min_phase2 is not None:
        config.train_snr_min_phase2 = float(args.train_snr_min_phase2)
    if args.train_snr_max_phase2 is not None:
        config.train_snr_max_phase2 = float(args.train_snr_max_phase2)
    if args.validation_snr_min is not None:
        config.validation_snr_min = float(args.validation_snr_min)
    if args.validation_snr_max is not None:
        config.validation_snr_max = float(args.validation_snr_max)
    if args.lms_paper_original:
        config.lms_normalized = False
        config.lms_include_self_pairs = True
        config.lms_map_mode = "tau_sample"
        config.lms_update_mode = "trajectory_tracking"
    if args.cpu:
        config.device = "cpu"
    elif args.device is not None:
        config.device = args.device

    config.__post_init__()
    summary = IFANTrainingPipeline(
        config,
        resume_checkpoint_path=args.resume_checkpoint,
        resume_output_dir=args.resume_output_dir,
        resume_log_path=args.resume_log,
    ).run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
