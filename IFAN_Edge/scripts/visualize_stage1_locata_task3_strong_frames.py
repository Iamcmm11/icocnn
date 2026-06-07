from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

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

# Real LOCATA audio does not need gpuRIR. Avoid importing the CUDA extension after torch.
sys.modules.setdefault("gpuRIR", types.SimpleNamespace())

from ifan_edge.features import DualFeatureIcoPreprocessor

STAGE1_SPEC = importlib.util.spec_from_file_location("ifan_edge.eval.stage1", PACKAGE_ROOT / "ifan_edge" / "eval" / "stage1.py")
if STAGE1_SPEC is None or STAGE1_SPEC.loader is None:
    raise ImportError("Could not load ifan_edge.eval.stage1")
stage1_eval = importlib.util.module_from_spec(STAGE1_SPEC)
STAGE1_SPEC.loader.exec_module(stage1_eval)
save_dual_feature_figure = stage1_eval.save_dual_feature_figure
save_dual_feature_projection_figure = stage1_eval.save_dual_feature_projection_figure
save_single_projection_figure = stage1_eval.save_single_projection_figure


BENCHMARK2_MIC_POS = np.array(
    (
        (-0.028, 0.030, -0.040),
        (0.006, 0.057, 0.000),
        (0.022, 0.022, -0.046),
        (-0.055, -0.024, -0.025),
        (-0.031, 0.023, 0.042),
        (-0.032, 0.011, 0.046),
        (-0.025, -0.003, 0.051),
        (-0.036, -0.027, 0.038),
        (-0.035, -0.043, 0.025),
        (0.029, -0.048, -0.012),
        (0.034, -0.030, 0.037),
        (0.035, 0.025, 0.039),
    ),
    dtype=np.float32,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan LOCATA task3 moving-source recordings and export strong stage-1 feature frames.")
    parser.add_argument("--task3-root", default="datasets/LOCATA/LOCATA/eval/task3", help="LOCATA eval/task3 root.")
    parser.add_argument(
        "--output-dir",
        default="IFAN_Edge/outputs/stage1_features/locata_task3_benchmark2_strong_frames",
        help="Directory for task3 strong-frame exports.",
    )
    parser.add_argument("--top-k", type=int, default=6, help="Number of strong frames to export.")
    parser.add_argument("--min-frame-gap", type=int, default=8, help="Minimum frame gap within the same recording.")
    parser.add_argument("--k", type=int, default=4096, help="Window length at target sampling rate.")
    parser.add_argument("--step", type=int, default=3072, help="Window step at target sampling rate.")
    parser.add_argument("--fs", type=int, default=16000, help="Target sampling rate.")
    parser.add_argument("--r", type=int, default=2, help="Icosahedral map resolution.")
    parser.add_argument("--projection-theta", type=int, default=181, help="Polar-angle resolution for projected figures.")
    parser.add_argument("--projection-phi", type=int, default=360, help="Azimuth resolution for projected figures.")
    return parser


def load_windows(recording_dir: Path, *, fs: int, k: int, step: int) -> tuple[np.ndarray, int]:
    audio, source_fs = sf.read(recording_dir / "audio_array_benchmark2.wav", always_2d=True)
    audio = audio.astype(np.float32, copy=False)
    if source_fs != fs:
        sample_count = int(round(audio.shape[0] * fs / source_fs))
        audio = signal.resample_poly(audio, fs, source_fs, axis=0).astype(np.float32, copy=False)
        audio = audio[:sample_count]

    frame_count = 1 + max(0, (audio.shape[0] - k) // step)
    if frame_count <= 0:
        raise ValueError(f"Recording is too short for k={k}: {audio.shape[0]} samples")

    window = np.hanning(k).astype(np.float32)
    frames = np.empty((frame_count, audio.shape[1], k), dtype=np.float32)
    for frame_idx in range(frame_count):
        start = frame_idx * step
        frames[frame_idx] = (audio[start : start + k] * window[:, None]).T
    return frames[np.newaxis, ...], source_fs


def score_frames(recording_name: str, dual_maps: np.ndarray, mic_windows: np.ndarray) -> list[dict[str, float | int | str]]:
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
                "recording": recording_name,
                "frame_idx": frame_idx,
                "score": float(score),
                "energy": float(energy[frame_idx]),
                "phat_std": phat_std,
                "lms_std": lms_std,
                "phat_peak_to_peak": phat_ptp,
                "lms_peak_to_peak": lms_ptp,
            }
        )
    return rows


def select_frames(rows: list[dict[str, float | int | str]], *, top_k: int, min_frame_gap: int) -> list[dict[str, float | int | str]]:
    selected: list[dict[str, float | int | str]] = []
    for row in sorted(rows, key=lambda item: float(item["score"]), reverse=True):
        recording = str(row["recording"])
        frame_idx = int(row["frame_idx"])
        if all(recording != str(prev["recording"]) or abs(frame_idx - int(prev["frame_idx"])) >= min_frame_gap for prev in selected):
            selected.append(row)
        if len(selected) >= top_k:
            break
    return selected


def export_frame_figures(
    *,
    row: dict[str, float | int | str],
    recording_dir: Path,
    dual_maps: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    frame_idx = int(row["frame_idx"])
    export_dir = output_dir / f"rank_{int(row['rank']):02d}_{row['recording']}_frame_{frame_idx:03d}"
    export_dir.mkdir(parents=True, exist_ok=True)

    phat_maps = dual_maps[:, 0:1]
    lms_maps = dual_maps[:, 1:2]
    title = (
        f"LOCATA task3 {row['recording']} benchmark2, frame={frame_idx}, "
        f"LMS std={row['lms_std']:.4f}, PHAT std={row['phat_std']:.4f}"
    )
    save_dual_feature_figure(phat_maps, lms_maps, export_dir / "feature_maps_charts.png", frame_idx=frame_idx, title=title)
    save_dual_feature_projection_figure(
        phat_maps,
        lms_maps,
        export_dir / "feature_maps_projection.png",
        frame_idx=frame_idx,
        title=title,
        res_theta=args.projection_theta,
        res_phi=args.projection_phi,
    )
    save_dual_feature_projection_figure(
        phat_maps,
        lms_maps,
        export_dir / "feature_maps_projection_contrast.png",
        frame_idx=frame_idx,
        title=title,
        res_theta=args.projection_theta,
        res_phi=args.projection_phi,
        adaptive_contrast=True,
    )
    save_single_projection_figure(
        phat_maps,
        export_dir / "phat_projection_contrast.png",
        frame_idx=frame_idx,
        title=f"task3 {row['recording']} frame {frame_idx} SRP-PHAT",
        res_theta=args.projection_theta,
        res_phi=args.projection_phi,
        adaptive_contrast=True,
    )
    save_single_projection_figure(
        lms_maps,
        export_dir / "lms_projection_contrast.png",
        frame_idx=frame_idx,
        title=f"task3 {row['recording']} frame {frame_idx} SRP-LMS",
        res_theta=args.projection_theta,
        res_phi=args.projection_phi,
        adaptive_contrast=True,
    )
    (export_dir / "source_recording.txt").write_text(str(recording_dir) + "\n", encoding="utf-8")
    return export_dir


def make_overview(output_dir: Path, selected: list[dict[str, float | int | str]], export_dirs: list[Path]) -> None:
    images = []
    for row, export_dir in zip(selected, export_dirs):
        images.append((row, Image.open(export_dir / "feature_maps_projection_contrast.png").convert("RGB")))
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
            f"rank {row['rank']} {row['recording']} frame {row['frame_idx']}  "
            f"LMS std {row['lms_std']:.4f}  score {row['score']:.4f}"
        )
        draw.text((x + 12, y + 12), label, fill=(0, 0, 0))
        canvas.paste(image, (x, y + label_h))
    canvas.save(output_dir / "task3_strong_frames_projection_contrast_overview.png")


def write_manifest(output_dir: Path, *, task3_root: Path, args: argparse.Namespace, selected: list[dict[str, float | int | str]]) -> None:
    payload = {
        "kind": "locata_task3_benchmark2_strong_stage1_frames",
        "task3_root": str(task3_root),
        "array": "benchmark2",
        "target_fs": args.fs,
        "k": args.k,
        "step": args.step,
        "r": args.r,
        "selection": {
            "top_k": args.top_k,
            "min_frame_gap": args.min_frame_gap,
            "score": "2*lms_std + 0.65*lms_ptp + 0.8*phat_std + 0.25*phat_ptp + 0.03*normalized_energy",
        },
        "selected_frames": selected,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# LOCATA Task3 Benchmark2 Strong Stage1 Frames",
        "",
        "Real LOCATA task3 moving-source recordings are scanned with the stage1 PHAT/LMS frontend. Frames are ranked with an LMS-weighted contrast score so LMS-visible examples are favored.",
        "",
        f"- task3 root: `{task3_root}`",
        f"- target sampling rate: `{args.fs}` Hz",
        f"- window: `K={args.k}`, `step={args.step}`",
        f"- top-k: `{args.top_k}`",
        "",
        "| Rank | Recording | Frame | Score | Energy | PHAT std | LMS std | PHAT peak-to-peak | LMS peak-to-peak |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected:
        lines.append(
            f"| {row['rank']} | `{row['recording']}` | {row['frame_idx']} | {row['score']:.6f} | {row['energy']:.6f} | "
            f"{row['phat_std']:.6f} | {row['lms_std']:.6f} | {row['phat_peak_to_peak']:.6f} | {row['lms_peak_to_peak']:.6f} |"
        )
    lines.extend(["", "Overview:", "", "- `task3_strong_frames_projection_contrast_overview.png`"])
    (output_dir / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    task3_root = Path(args.task3_root).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = DualFeatureIcoPreprocessor(
        N=BENCHMARK2_MIC_POS.shape[0],
        K=args.k,
        r=args.r,
        rn=BENCHMARK2_MIC_POS,
        fs=args.fs,
        apply_vad=False,
    )

    recording_maps: dict[str, np.ndarray] = {}
    recording_dirs: dict[str, Path] = {}
    all_rows: list[dict[str, float | int | str]] = []
    for recording_dir in sorted(task3_root.glob("recording*/benchmark2")):
        recording_name = recording_dir.parent.name
        mic_windows, source_fs = load_windows(recording_dir, fs=args.fs, k=args.k, step=args.step)
        dual_maps = preprocessor.data_transformation(mic_windows).detach().cpu().numpy()
        recording_maps[recording_name] = dual_maps
        recording_dirs[recording_name] = recording_dir
        recording_out = output_dir / "full_maps" / recording_name
        recording_out.mkdir(parents=True, exist_ok=True)
        np.save(recording_out / "dual_maps.npy", dual_maps)
        np.save(recording_out / "mic_windows.npy", mic_windows)
        all_rows.extend(score_frames(recording_name, dual_maps, mic_windows))
        print(f"Scanned {recording_name}: source_fs={source_fs}, frames={dual_maps.shape[2]}, shape={dual_maps.shape}")

    selected = select_frames(all_rows, top_k=args.top_k, min_frame_gap=args.min_frame_gap)
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank

    export_dirs = [
        export_frame_figures(
            row=row,
            recording_dir=recording_dirs[str(row["recording"])],
            dual_maps=recording_maps[str(row["recording"])],
            output_dir=output_dir,
            args=args,
        )
        for row in selected
    ]
    make_overview(output_dir, selected, export_dirs)
    write_manifest(output_dir, task3_root=task3_root, args=args, selected=selected)
    print(f"Exported task3 strong frames to {output_dir}")
    for row in selected:
        print(
            f"rank {row['rank']}: {row['recording']} frame {row['frame_idx']} "
            f"score={row['score']:.6f} LMS std={row['lms_std']:.6f} PHAT std={row['phat_std']:.6f}"
        )


if __name__ == "__main__":
    main()
