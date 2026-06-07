from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Import order matters on this host: torch before gpuRIR makes gpuRIR cuRAND
# initialization fail with status 203.
import gpuRIR  # noqa: F401

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw
from scipy import signal

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import acousticTrackingDataset as at_dataset
from ifan_edge.eval.stage1 import (
    save_dual_feature_figure,
    save_dual_feature_projection_figure,
    save_single_projection_figure,
)
from ifan_edge.features import DualFeatureIcoPreprocessor


SCENARIOS = (
    {"name": "scene_1", "label": "high_snr_low_reverb", "snr_db": 30.0, "t60_s": 0.2},
    {"name": "scene_2", "label": "high_snr_high_reverb", "snr_db": 30.0, "t60_s": 1.4},
    {"name": "scene_3", "label": "low_snr_low_reverb", "snr_db": 5.0, "t60_s": 0.2},
    {"name": "scene_4", "label": "low_snr_high_reverb", "snr_db": 5.0, "t60_s": 1.4},
)


class FixedSourceDataset:
    def __init__(self, source_path: Path, *, fs: int, max_seconds: float | None):
        source, source_fs = sf.read(source_path, always_2d=True)
        source = source[:, 0].astype(np.float32, copy=False)
        if source_fs != fs:
            sample_count = int(round(source.shape[0] * fs / source_fs))
            source = signal.resample_poly(source, fs, source_fs).astype(np.float32, copy=False)
            source = source[:sample_count]
        if max_seconds is not None:
            source = source[: int(round(max_seconds * fs))]
        source = source - float(source.mean())
        peak = float(np.max(np.abs(source)))
        if peak > 0:
            source = source / peak
        self.source = source.astype(np.float32, copy=False)
        self.vad = np.ones_like(self.source, dtype=bool)
        self.fs = int(fs)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        if idx != 0:
            raise IndexError(idx)
        return self.source.copy(), self.vad.copy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standard gpuRIR four scenes from LOCATA task3 speech and export strong stage-1 frames.")
    parser.add_argument(
        "--source-audio",
        default="datasets/LOCATA/LOCATA/eval/task3/recording4/benchmark2/audio_source_talker4.wav",
        help="LOCATA task3 source audio to drive the standard moving-source simulation.",
    )
    parser.add_argument(
        "--output-dir",
        default="IFAN_Edge/outputs/stage1_features/locata_task3_source_standard_four_scenes_strong_frames",
        help="Directory for generated four-scene strong-frame outputs.",
    )
    parser.add_argument("--top-per-scene", type=int, default=2, help="Strong frames exported per scene.")
    parser.add_argument("--min-frame-gap", type=int, default=8, help="Minimum frame gap among selected frames within each scene.")
    parser.add_argument("--signal-seconds", type=float, default=None, help="Optional source duration cap.")
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--step", type=int, default=3072)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260607, help="Seed reset for every scene to keep trajectory/room fixed.")
    parser.add_argument("--nb-points", type=int, default=156)
    parser.add_argument("--projection-theta", type=int, default=181)
    parser.add_argument("--projection-phi", type=int, default=360)
    return parser


def build_dataset(source_dataset: FixedSourceDataset, scenario: dict[str, float | str], args: argparse.Namespace):
    return at_dataset.RandomTrajectoryDataset(
        sourceDataset=source_dataset,
        room_sz=at_dataset.Parameter([6.0, 5.0, 3.0]),
        T60=at_dataset.Parameter(float(scenario["t60_s"])),
        abs_weights=at_dataset.Parameter([0.8] * 6),
        array_setup=at_dataset.benchmark2_array_setup,
        array_pos=at_dataset.Parameter([0.5, 0.5, 0.5]),
        SNR=at_dataset.Parameter(float(scenario["snr_db"])),
        nb_points=args.nb_points,
        transforms=[at_dataset.Windowing(args.k, args.step, window=np.hanning)],
    )


def score_frames(scenario: dict[str, float | str], dual_maps: np.ndarray, mic_windows: np.ndarray) -> list[dict[str, float | int | str]]:
    phat = dual_maps[0, 0]
    lms = dual_maps[0, 1]
    energy = np.sqrt(np.mean(mic_windows[0] * mic_windows[0], axis=(1, 2)))
    energy_norm = energy / max(float(energy.max()), 1e-12)
    rows = []
    for frame_idx in range(phat.shape[0]):
        phat_std = float(phat[frame_idx].std())
        lms_std = float(lms[frame_idx].std())
        phat_ptp = float(np.ptp(phat[frame_idx]))
        lms_ptp = float(np.ptp(lms[frame_idx]))
        score = 2.0 * lms_std + 0.65 * lms_ptp + 0.8 * phat_std + 0.25 * phat_ptp + 0.03 * float(energy_norm[frame_idx])
        rows.append(
            {
                "scene": str(scenario["name"]),
                "label": str(scenario["label"]),
                "snr_db": float(scenario["snr_db"]),
                "t60_s": float(scenario["t60_s"]),
                "frame_idx": int(frame_idx),
                "score": float(score),
                "energy": float(energy[frame_idx]),
                "phat_std": phat_std,
                "lms_std": lms_std,
                "phat_peak_to_peak": phat_ptp,
                "lms_peak_to_peak": lms_ptp,
            }
        )
    return rows


def select_rows(rows: list[dict[str, float | int | str]], *, top_k: int, min_frame_gap: int) -> list[dict[str, float | int | str]]:
    selected: list[dict[str, float | int | str]] = []
    for row in sorted(rows, key=lambda item: float(item["score"]), reverse=True):
        frame_idx = int(row["frame_idx"])
        if all(abs(frame_idx - int(prev["frame_idx"])) >= min_frame_gap for prev in selected):
            selected.append(row)
        if len(selected) >= top_k:
            break
    return selected


def export_figures(
    *,
    row: dict[str, float | int | str],
    dual_maps: np.ndarray,
    scene_dir: Path,
    args: argparse.Namespace,
) -> Path:
    frame_idx = int(row["frame_idx"])
    rank = int(row["rank"])
    export_dir = scene_dir / f"rank_{rank:02d}_frame_{frame_idx:03d}"
    export_dir.mkdir(parents=True, exist_ok=True)
    phat_maps = dual_maps[:, 0:1]
    lms_maps = dual_maps[:, 1:2]
    title = (
        f"{row['scene']}: {row['label']}, SNR={row['snr_db']}dB, T60={row['t60_s']}s, "
        f"frame={frame_idx}, LMS std={row['lms_std']:.4f}"
    )
    save_dual_feature_figure(phat_maps, lms_maps, export_dir / "feature_maps_charts.png", frame_idx=frame_idx, title=title)
    save_dual_feature_projection_figure(phat_maps, lms_maps, export_dir / "feature_maps_projection.png", frame_idx=frame_idx, title=title)
    save_dual_feature_projection_figure(
        phat_maps,
        lms_maps,
        export_dir / "feature_maps_projection_contrast.png",
        frame_idx=frame_idx,
        title=title,
        adaptive_contrast=True,
    )
    save_single_projection_figure(
        phat_maps,
        export_dir / "phat_projection_contrast.png",
        frame_idx=frame_idx,
        title=f"{row['scene']} frame {frame_idx} SRP-PHAT",
        adaptive_contrast=True,
    )
    save_single_projection_figure(
        lms_maps,
        export_dir / "lms_projection_contrast.png",
        frame_idx=frame_idx,
        title=f"{row['scene']} frame {frame_idx} SRP-LMS",
        adaptive_contrast=True,
    )
    return export_dir


def make_overview(output_dir: Path, selected: list[dict[str, float | int | str]], export_dirs: list[Path]) -> None:
    images = [(row, Image.open(path / "feature_maps_projection_contrast.png").convert("RGB")) for row, path in zip(selected, export_dirs)]
    width = max(image.width for _, image in images)
    height = max(image.height for _, image in images)
    label_h = 48
    cols = 2
    rows_count = int(np.ceil(len(images) / cols))
    canvas = Image.new("RGB", (width * cols, (height + label_h) * rows_count), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (row, image) in enumerate(images):
        x = (idx % cols) * width
        y = (idx // cols) * (height + label_h)
        label = (
            f"{row['scene']} rank {row['rank']} frame {row['frame_idx']} "
            f"SNR={row['snr_db']} T60={row['t60_s']} LMS std={row['lms_std']:.4f}"
        )
        draw.text((x + 12, y + 12), label, fill=(0, 0, 0))
        canvas.paste(image, (x, y + label_h))
    canvas.save(output_dir / "four_scenes_strong_frames_projection_contrast_overview.png")


def write_manifest(output_dir: Path, *, source_audio: Path, args: argparse.Namespace, selected: list[dict[str, float | int | str]]) -> None:
    payload = {
        "kind": "locata_task3_source_standard_four_scenes_strong_frames",
        "source_audio": str(source_audio),
        "target_fs": args.fs,
        "k": args.k,
        "step": args.step,
        "r": args.r,
        "seed": args.seed,
        "top_per_scene": args.top_per_scene,
        "selection_score": "2*lms_std + 0.65*lms_ptp + 0.8*phat_std + 0.25*phat_ptp + 0.03*normalized_energy",
        "selected_frames": selected,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# LOCATA Task3 Source Standard Four-Scene Strong Frames",
        "",
        "The LOCATA task3 source audio drives the standard `RandomTrajectoryDataset + gpuRIR` moving-source simulation. The RNG seed is reset for each scene so trajectory and room geometry stay fixed; SNR/T60 change by scene.",
        "",
        f"- source audio: `{source_audio}`",
        f"- target sampling rate: `{args.fs}` Hz",
        f"- window: `K={args.k}`, `step={args.step}`",
        f"- top frames per scene: `{args.top_per_scene}`",
        "",
        "| Scene | Rank | Frame | SNR | T60 | Score | PHAT std | LMS std | PHAT peak-to-peak | LMS peak-to-peak |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected:
        lines.append(
            f"| `{row['scene']}` | {row['rank']} | {row['frame_idx']} | {row['snr_db']:.1f} dB | {row['t60_s']:.1f} s | "
            f"{row['score']:.6f} | {row['phat_std']:.6f} | {row['lms_std']:.6f} | "
            f"{row['phat_peak_to_peak']:.6f} | {row['lms_peak_to_peak']:.6f} |"
        )
    lines.extend(["", "Overview:", "", "- `four_scenes_strong_frames_projection_contrast_overview.png`"])
    (output_dir / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    source_audio = Path(args.source_audio).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dataset = FixedSourceDataset(source_audio, fs=args.fs, max_seconds=args.signal_seconds)
    preprocessor = DualFeatureIcoPreprocessor(
        N=at_dataset.benchmark2_array_setup.mic_pos.shape[0],
        K=args.k,
        r=args.r,
        rn=at_dataset.benchmark2_array_setup.mic_pos,
        fs=args.fs,
        apply_vad=False,
    )

    selected_all: list[dict[str, float | int | str]] = []
    export_dirs: list[Path] = []
    for scenario in SCENARIOS:
        np.random.seed(args.seed)
        dataset = build_dataset(source_dataset, scenario, args)
        mic_windows, acoustic_scene_batch = dataset.get_batch(0, 1)
        dual_maps, _ = preprocessor.data_transformation(mic_windows, acoustic_scene_batch)
        dual_np = dual_maps.detach().cpu().numpy()

        scene_dir = output_dir / str(scenario["name"])
        scene_dir.mkdir(parents=True, exist_ok=True)
        np.save(scene_dir / "dual_maps.npy", dual_np)
        np.save(scene_dir / "mic_windows.npy", mic_windows)
        np.save(scene_dir / "doaw.npy", acoustic_scene_batch[0].DOAw[0])

        selected = select_rows(score_frames(scenario, dual_np, mic_windows), top_k=args.top_per_scene, min_frame_gap=args.min_frame_gap)
        for rank, row in enumerate(selected, start=1):
            row["rank"] = rank
            selected_all.append(row)
            export_dirs.append(export_figures(row=row, dual_maps=dual_np, scene_dir=scene_dir, args=args))
        print(f"Exported {scenario['name']}: maps={dual_np.shape}, selected={[int(row['frame_idx']) for row in selected]}")

    make_overview(output_dir, selected_all, export_dirs)
    write_manifest(output_dir, source_audio=source_audio, args=args, selected=selected_all)
    print(f"Exported task3 source standard four-scene strong frames to {output_dir}")
    for row in selected_all:
        print(
            f"{row['scene']} rank {row['rank']} frame {row['frame_idx']}: "
            f"SNR={row['snr_db']} T60={row['t60_s']} PHAT std={row['phat_std']:.6f} LMS std={row['lms_std']:.6f}"
        )


if __name__ == "__main__":
    main()
