from __future__ import annotations

import argparse
import copy
import json
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from math import pi
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# Real LOCATA audio does not use room simulation, but legacy modules import
# gpuRIR at module import time. Keep this path usable when gpuRIR is absent.
sys.modules.setdefault("gpuRIR", types.SimpleNamespace())


class EnergyVad:
    def set_mode(self, _mode: int) -> None:
        return None

    def is_speech(self, frame_bytes: bytes, _fs: int) -> bool:
        frame = np.frombuffer(frame_bytes, dtype=np.int16)
        if frame.size == 0:
            return False
        return float(np.mean(np.abs(frame))) > 80.0


sys.modules.setdefault("webrtcvad", types.SimpleNamespace(Vad=EnergyVad))

from evaluate_stage3_locata import (  # noqa: E402
    build_locata_dataset,
    load_config_from_checkpoint,
    normalize_tasks,
    predict_baseline_scene,
    predict_ifan_scene,
    remap_legacy_map_refiner_keys,
)
from ifan_edge.eval import (  # noqa: E402
    build_baseline_preprocessor_from_config,
    build_ifan_preprocessor_from_config,
    load_baseline_model_from_config,
)
from ifan_edge.models import IFANModel  # noqa: E402
from ifan_edge.training import IFANTrainingConfig, IFANTrainingPipeline  # noqa: E402


DEFAULT_IFAN80_CHECKPOINT = (
    "IFAN_Edge/outputs/stage3/ifan_stage3_long80_freqblock_paper_original_20260426_155330/"
    "checkpoints/best_rmsae.pt"
)
DEFAULT_EDGE_CHECKPOINT = (
    "IFAN_Edge/outputs/stage3/ifan_stage3_long80_c8_paper_original_20260505_222115/"
    "checkpoints/best_rmsae.pt"
)
DEFAULT_EDGE_MABA_CHECKPOINT = (
    "IFAN_Edge/outputs/stage3/ifan_stage3_long80_c8_r2_maba_pre_readout_20260518_215408/"
    "checkpoints/best_rmsae.pt"
)


@dataclass(frozen=True)
class IFANCheckpointSpec:
    label: str
    checkpoint: Path
    config: Path | None = None


@dataclass
class PredictionSeries:
    label: str
    doa_rad: np.ndarray
    rmsae_with_silences_deg: float
    rmsae_without_silences_deg: float | None
    frame_error_deg: np.ndarray


def angular_error_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    theta_pred = pred[..., 0]
    phi_pred = pred[..., 1]
    theta_true = target[..., 0]
    phi_true = target[..., 1]
    aux = np.cos(theta_true) * np.cos(theta_pred) + np.sin(theta_true) * np.sin(theta_pred) * np.cos(phi_true - phi_pred)
    return np.arccos(np.clip(aux, -1.0, 1.0)) * 180.0 / pi


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def silence_mask_for_scene(scene, frame_count: int) -> np.ndarray | None:
    if not hasattr(scene, "vad"):
        return None
    vad = np.asarray(scene.vad)
    if vad.size == 0:
        return None
    if vad.ndim == 1:
        silence = vad[:frame_count] < (2.0 / 3.0)
    else:
        silence = vad[:frame_count].mean(axis=1) < (2.0 / 3.0)
    if silence.shape[0] < frame_count:
        padded = np.zeros(frame_count, dtype=bool)
        padded[: silence.shape[0]] = silence.astype(bool)
        silence = padded
    return silence[:frame_count].astype(bool)


def time_axis_for_scene(scene, frame_count: int, *, fs: int, step: int) -> np.ndarray:
    if hasattr(scene, "tw"):
        time = np.asarray(scene.tw, dtype=np.float64).reshape(-1)
        if time.size >= frame_count:
            return time[:frame_count]
    return np.arange(frame_count, dtype=np.float64) * float(step) / float(fs)


def contiguous_true_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = idx
        if start is not None and (not value or idx == len(mask) - 1):
            end = idx if value and idx == len(mask) - 1 else idx - 1
            ranges.append((start, end))
            start = None
    return ranges


def unwrap_angle_deg(angle_rad: np.ndarray) -> np.ndarray:
    return np.unwrap(np.asarray(angle_rad, dtype=np.float64)) * 180.0 / pi


def slice_series_for_plot(
    time_s: np.ndarray,
    target: np.ndarray,
    predictions: list[PredictionSeries],
    silence_mask: np.ndarray | None,
    *,
    start_sec: float | None,
    seconds: float | None,
) -> tuple[np.ndarray, np.ndarray, list[PredictionSeries], np.ndarray | None]:
    frame_count = target.shape[0]
    selection = np.ones(frame_count, dtype=bool)
    if start_sec is not None:
        selection &= time_s >= float(start_sec)
    if seconds is not None:
        start = float(start_sec) if start_sec is not None else float(time_s[0])
        selection &= time_s <= start + float(seconds)
    if not selection.any():
        raise ValueError("The requested plot time window does not overlap the recording.")
    indexes = np.flatnonzero(selection)
    first, last = int(indexes[0]), int(indexes[-1]) + 1
    sliced_predictions = [
        PredictionSeries(
            label=series.label,
            doa_rad=series.doa_rad[first:last],
            rmsae_with_silences_deg=series.rmsae_with_silences_deg,
            rmsae_without_silences_deg=series.rmsae_without_silences_deg,
            frame_error_deg=series.frame_error_deg[first:last],
        )
        for series in predictions
    ]
    sliced_silence = None if silence_mask is None else silence_mask[first:last]
    return time_s[first:last], target[first:last], sliced_predictions, sliced_silence


def copy_scene_for_rms(scene, prediction: np.ndarray):
    scene_copy = copy.copy(scene)
    scene_copy.DOAw_pred = [np.asarray(prediction, dtype=np.float32)]
    return scene_copy


def prediction_series(
    *,
    label: str,
    scene,
    target: np.ndarray,
    prediction: np.ndarray,
    silence_mask: np.ndarray | None,
) -> PredictionSeries:
    frame_count = min(target.shape[0], prediction.shape[0])
    target = target[:frame_count]
    prediction = prediction[:frame_count]
    errors = angular_error_deg(prediction, target)
    with_silences = rms(errors)
    without_silences = None
    if silence_mask is not None:
        active = ~silence_mask[:frame_count]
        without_silences = rms(errors[active])

    # Keep the project-native RMSAE for exact consistency with existing LOCATA reports.
    try:
        scene_with = copy_scene_for_rms(scene, prediction)
        with_silences = float(scene_with.get_rmsae(exclude_silences=False))
        if silence_mask is not None:
            scene_without = copy_scene_for_rms(scene, prediction)
            without_silences = float(scene_without.get_rmsae(exclude_silences=True))
    except Exception:
        pass

    return PredictionSeries(
        label=label,
        doa_rad=prediction,
        rmsae_with_silences_deg=with_silences,
        rmsae_without_silences_deg=without_silences,
        frame_error_deg=errors,
    )


def load_ifan_model(
    *,
    spec: IFANCheckpointSpec,
    device: torch.device,
    device_override: str | None,
) -> tuple[IFANModel, IFANTrainingConfig, Any, str]:
    checkpoint, config = load_config_from_checkpoint(spec.checkpoint, None if spec.config is None else str(spec.config))
    if device_override is not None:
        config.device = device_override
    pipeline = IFANTrainingPipeline(config)
    model = IFANModel(pipeline.model_config)
    state_dict = remap_legacy_map_refiner_keys(
        checkpoint["model_state_dict"],
        model_config=pipeline.model_config,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config, pipeline.model_config, config.input_ablation_mode


def validate_dataset_contract(reference: IFANTrainingConfig, others: list[IFANTrainingConfig]) -> None:
    fields = ("fs", "k", "step")
    for config in others:
        for field in fields:
            if getattr(config, field) != getattr(reference, field):
                raise ValueError(
                    "All checkpoints must share the same LOCATA windowing contract. "
                    f"Mismatch for {field}: {getattr(reference, field)} vs {getattr(config, field)}"
                )


def parse_ifan_specs(values: list[str] | None) -> list[IFANCheckpointSpec]:
    if not values:
        return [
            IFANCheckpointSpec("DFA-IcoNet", Path(DEFAULT_IFAN80_CHECKPOINT)),
            IFANCheckpointSpec("DFA-IcoNet-Edge", Path(DEFAULT_EDGE_CHECKPOINT)),
            IFANCheckpointSpec("DFA-IcoNet-Edge-MABA", Path(DEFAULT_EDGE_MABA_CHECKPOINT)),
        ]

    specs: list[IFANCheckpointSpec] = []
    for value in values:
        parts = value.split("=", 1)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(
                "--ifan-checkpoint expects LABEL=PATH. "
                "Example: --ifan-checkpoint DFA-IcoNet=IFAN_Edge/outputs/.../best_rmsae.pt"
            )
        specs.append(IFANCheckpointSpec(parts[0].strip(), Path(parts[1].strip())))
    return specs


def require_existing_paths(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required model checkpoint(s) are missing:\n{missing_text}")


def plot_tracking(
    *,
    output_path: Path,
    title: str,
    time_s: np.ndarray,
    target: np.ndarray,
    predictions: list[PredictionSeries],
    silence_mask: np.ndarray | None,
    angle: str,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    colors = {
        "GT": "#111827",
        "icoCNN baseline": "#7c3aed",
        "DFA-IcoNet": "#2563eb",
        "DFA-IcoNet-Edge": "#f97316",
        "DFA-IcoNet-Edge-MABA": "#059669",
    }
    linestyles = {
        "GT": "-",
        "icoCNN baseline": (0, (4, 2)),
        "DFA-IcoNet": "-",
        "DFA-IcoNet-Edge": "-.",
        "DFA-IcoNet-Edge-MABA": "-",
    }

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )
    track_ax, error_ax = axes
    dim = 1 if angle == "azimuth" else 0
    ylabel = "Azimuth [deg]" if angle == "azimuth" else "Polar angle [deg]"

    target_angle = unwrap_angle_deg(target[:, dim]) if angle == "azimuth" else target[:, dim] * 180.0 / pi
    track_ax.plot(time_s, target_angle, color=colors["GT"], linewidth=2.4, label="GT", linestyle=linestyles["GT"])
    for series in predictions:
        pred_angle = unwrap_angle_deg(series.doa_rad[:, dim]) if angle == "azimuth" else series.doa_rad[:, dim] * 180.0 / pi
        track_ax.plot(
            time_s,
            pred_angle,
            color=colors.get(series.label, None),
            linewidth=1.8,
            alpha=0.95,
            label=f"{series.label} ({series.rmsae_without_silences_deg or series.rmsae_with_silences_deg:.2f} deg)",
            linestyle=linestyles.get(series.label, "-"),
        )

    for series in predictions:
        error_ax.plot(
            time_s,
            series.frame_error_deg,
            color=colors.get(series.label, None),
            linewidth=1.4,
            alpha=0.92,
            label=series.label,
            linestyle=linestyles.get(series.label, "-"),
        )

    if silence_mask is not None and silence_mask.any():
        dt = float(np.median(np.diff(time_s))) if time_s.size > 1 else 0.0
        for start, end in contiguous_true_ranges(silence_mask):
            x0 = time_s[start] - 0.5 * dt
            x1 = time_s[end] + 0.5 * dt
            track_ax.axvspan(x0, x1, color="#9ca3af", alpha=0.18, linewidth=0)
            error_ax.axvspan(x0, x1, color="#9ca3af", alpha=0.18, linewidth=0)

    track_ax.set_title(title)
    track_ax.set_ylabel(ylabel)
    track_ax.grid(alpha=0.22, linewidth=0.8)
    track_ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.04), frameon=False)

    error_ax.set_xlabel("Time [s]")
    error_ax.set_ylabel("Angular error [deg]")
    error_ax.grid(alpha=0.22, linewidth=0.8)
    error_ax.set_ylim(bottom=0.0)
    error_ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.26), frameon=False)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a PPT-ready LOCATA tracking plot: GT vs icoCNN baseline vs IFAN variants."
    )
    parser.add_argument("--locata-root", default="datasets/LOCATA/LOCATA", help="LOCATA root containing eval/dev.")
    parser.add_argument("--subset", choices=("dev", "eval"), default="eval")
    parser.add_argument("--array", choices=("dummy", "eigenmike", "benchmark2", "dicit"), default="benchmark2")
    parser.add_argument("--task", type=int, default=5, choices=(1, 3, 5))
    parser.add_argument("--recording", default="recording1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--ifan-checkpoint",
        action="append",
        default=None,
        help="IFAN line as LABEL=PATH. Can be repeated. Defaults to DFA-IcoNet, Edge, and Edge-MABA.",
    )
    parser.add_argument("--baseline-label", default="icoCNN baseline")
    parser.add_argument("--plot-start-sec", type=float, default=None, help="Optional display crop start time.")
    parser.add_argument("--plot-seconds", type=float, default=None, help="Optional display crop duration.")
    parser.add_argument("--angle", choices=("azimuth", "polar"), default="azimuth")
    parser.add_argument("--output-dir", default="IFAN_Edge/outputs/stage3/analysis/locata_tracking")
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--no-pdf", action="store_true", help="Skip the PDF companion export.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    specs = parse_ifan_specs(args.ifan_checkpoint)
    require_existing_paths([spec.checkpoint for spec in specs])

    _, reference_config = load_config_from_checkpoint(specs[0].checkpoint, None)
    if args.device is not None:
        reference_config.device = args.device
    pipeline = IFANTrainingPipeline(reference_config)
    device = pipeline.resolve_device()

    dataset, subset_root = build_locata_dataset(
        root=args.locata_root,
        subset=args.subset,
        array=args.array,
        fs=reference_config.fs,
        k=reference_config.k,
        step=reference_config.step,
        tasks=normalize_tasks([args.task]),
        recording_filter={args.recording},
    )
    if len(dataset) != 1:
        raise RuntimeError(
            f"Expected exactly one LOCATA recording for task{args.task}/{args.recording}, got {len(dataset)}. "
            f"Check --locata-root, --subset, --array, --task, and --recording."
        )

    mic_sig_batch, acoustic_scene_batch = dataset.get_batch(0, 1)
    scene = acoustic_scene_batch[0]
    target = np.asarray(scene.DOAw[0], dtype=np.float64)
    frame_count = target.shape[0]
    silence_mask = silence_mask_for_scene(scene, frame_count)
    time_s = time_axis_for_scene(scene, frame_count, fs=reference_config.fs, step=reference_config.step)

    baseline_preprocessor = build_baseline_preprocessor_from_config(reference_config, device)
    baseline_model = load_baseline_model_from_config(reference_config, device)
    baseline_model.eval()
    baseline_pred = predict_baseline_scene(
        model=baseline_model,
        preprocessor=baseline_preprocessor,
        mic_sig_batch=mic_sig_batch,
        acoustic_scene_batch=acoustic_scene_batch,
    )[0]
    predictions = [
        prediction_series(
            label=args.baseline_label,
            scene=scene,
            target=target,
            prediction=baseline_pred,
            silence_mask=silence_mask,
        )
    ]

    for spec in specs:
        model, config, model_config, input_ablation_mode = load_ifan_model(
            spec=spec,
            device=device,
            device_override=args.device,
        )
        validate_dataset_contract(reference_config, [config])
        preprocessor = build_ifan_preprocessor_from_config(config, device)
        pred = predict_ifan_scene(
            model=model,
            preprocessor=preprocessor,
            model_config=model_config,
            input_ablation_mode=input_ablation_mode,
            mic_sig_batch=mic_sig_batch,
            acoustic_scene_batch=acoustic_scene_batch,
        )[0]
        predictions.append(
            prediction_series(
                label=spec.label,
                scene=scene,
                target=target,
                prediction=pred,
                silence_mask=silence_mask,
            )
        )

    min_frames = min([target.shape[0], time_s.shape[0], *(series.doa_rad.shape[0] for series in predictions)])
    target = target[:min_frames]
    time_s = time_s[:min_frames]
    if silence_mask is not None:
        silence_mask = silence_mask[:min_frames]
    predictions = [
        PredictionSeries(
            label=series.label,
            doa_rad=series.doa_rad[:min_frames],
            rmsae_with_silences_deg=series.rmsae_with_silences_deg,
            rmsae_without_silences_deg=series.rmsae_without_silences_deg,
            frame_error_deg=series.frame_error_deg[:min_frames],
        )
        for series in predictions
    ]

    plot_time_s, plot_target, plot_predictions, plot_silence = slice_series_for_plot(
        time_s,
        target,
        predictions,
        silence_mask,
        start_sec=args.plot_start_sec,
        seconds=args.plot_seconds,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    if stem is None:
        stem = f"task{args.task}_{args.recording}_{args.array}_{args.angle}_tracking"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    title = f"LOCATA {args.subset}/task{args.task}/{args.recording}/{args.array} tracking"
    plot_tracking(
        output_path=png_path,
        title=title,
        time_s=plot_time_s,
        target=plot_target,
        predictions=plot_predictions,
        silence_mask=plot_silence,
        angle=args.angle,
    )
    if not args.no_pdf:
        plot_tracking(
            output_path=pdf_path,
            title=title,
            time_s=plot_time_s,
            target=plot_target,
            predictions=plot_predictions,
            silence_mask=plot_silence,
            angle=args.angle,
        )

    summary = {
        "kind": "stage3_locata_tracking",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "subset": args.subset,
        "subset_path": str(subset_root),
        "array": args.array,
        "task": int(args.task),
        "recording": args.recording,
        "device": str(device),
        "angle": args.angle,
        "frame_count": int(min_frames),
        "plot_frame_count": int(plot_time_s.shape[0]),
        "plot_start_sec": None if args.plot_start_sec is None else float(args.plot_start_sec),
        "plot_seconds": None if args.plot_seconds is None else float(args.plot_seconds),
        "png_path": str(png_path),
        "pdf_path": None if args.no_pdf else str(pdf_path),
        "checkpoints": [
            {"label": spec.label, "checkpoint": str(spec.checkpoint)}
            for spec in specs
        ],
        "metrics": [
            {
                "label": series.label,
                "with_silences_rmsae_deg": float(series.rmsae_with_silences_deg),
                "without_silences_rmsae_deg": None
                if series.rmsae_without_silences_deg is None
                else float(series.rmsae_without_silences_deg),
            }
            for series in predictions
        ],
    }
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"png_path": str(png_path), "pdf_path": summary["pdf_path"], "json_path": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
