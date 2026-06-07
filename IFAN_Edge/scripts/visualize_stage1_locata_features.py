from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The legacy acousticTracking modules import gpuRIR at module import time even when
# only real LOCATA audio is used. A stub is enough for this visualization path.
sys.modules.setdefault("gpuRIR", types.SimpleNamespace())

from ifan_edge.features import DualFeatureIcoPreprocessor

eval_pkg = types.ModuleType("ifan_edge.eval")
eval_pkg.__path__ = [str(PACKAGE_ROOT / "ifan_edge" / "eval")]
sys.modules.setdefault("ifan_edge.eval", eval_pkg)
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
    parser = argparse.ArgumentParser(description="Export IFAN stage-1 feature visualizations for one LOCATA recording.")
    parser.add_argument(
        "--recording-dir",
        default="datasets/LOCATA/LOCATA/eval/task1/recording5/benchmark2",
        help="LOCATA array directory, e.g. datasets/LOCATA/LOCATA/eval/task1/recording5/benchmark2.",
    )
    parser.add_argument("--array", default="benchmark2", choices=("benchmark2",), help="LOCATA array name.")
    parser.add_argument(
        "--output-dir",
        default="IFAN_Edge/outputs/stage1_features/locata_task1_recording5_benchmark2",
        help="Directory to store exported figures and numpy dumps.",
    )
    parser.add_argument("--frame-idx", type=int, default=0, help="Window/frame index to visualize.")
    parser.add_argument("--max-frames", type=int, default=16, help="Number of frames to compute from the recording.")
    parser.add_argument("--k", type=int, default=4096, help="Window length at target sampling rate.")
    parser.add_argument("--step", type=int, default=3072, help="Window step at target sampling rate.")
    parser.add_argument("--fs", type=int, default=16000, help="Target sampling rate.")
    parser.add_argument("--r", type=int, default=2, help="Icosahedral map resolution.")
    parser.add_argument("--projection-theta", type=int, default=181, help="Polar-angle resolution for projected figures.")
    parser.add_argument("--projection-phi", type=int, default=360, help="Azimuth resolution for projected figures.")
    return parser


def load_locata_windows(recording_dir: Path, *, array: str, fs: int, k: int, step: int, max_frames: int) -> tuple[np.ndarray, int]:
    audio_path = recording_dir / f"audio_array_{array}.wav"
    audio, source_fs = sf.read(audio_path, always_2d=True)
    audio = audio.astype(np.float32, copy=False)
    if source_fs != fs:
        sample_count = int(round(audio.shape[0] * fs / source_fs))
        audio = signal.resample_poly(audio, fs, source_fs, axis=0).astype(np.float32, copy=False)
        audio = audio[:sample_count]

    frame_count = 1 + max(0, (audio.shape[0] - k) // step)
    frame_count = min(frame_count, max_frames)
    if frame_count <= 0:
        raise ValueError(f"Recording is too short for k={k}: {audio.shape[0]} samples at {fs} Hz")

    window = np.hanning(k).astype(np.float32)
    frames = np.empty((frame_count, audio.shape[1], k), dtype=np.float32)
    for frame_idx in range(frame_count):
        start = frame_idx * step
        frames[frame_idx] = (audio[start : start + k] * window[:, None]).T
    return frames[np.newaxis, ...], source_fs


def main() -> None:
    args = build_parser().parse_args()
    recording_dir = Path(args.recording_dir).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mic_sig_batch, source_fs = load_locata_windows(
        recording_dir,
        array=args.array,
        fs=args.fs,
        k=args.k,
        step=args.step,
        max_frames=args.max_frames,
    )
    preprocessor = DualFeatureIcoPreprocessor(
        N=BENCHMARK2_MIC_POS.shape[0],
        K=args.k,
        r=args.r,
        rn=BENCHMARK2_MIC_POS,
        fs=args.fs,
        apply_vad=False,
    )
    dual_maps = preprocessor.data_transformation(mic_sig_batch)
    phat_maps, lms_maps = preprocessor.split_features(dual_maps)

    np.save(output_dir / "mic_windows.npy", mic_sig_batch)
    np.save(output_dir / "dual_maps.npy", dual_maps.detach().cpu().numpy())
    np.save(output_dir / "phat_maps.npy", phat_maps.detach().cpu().numpy())
    np.save(output_dir / "lms_maps.npy", lms_maps.detach().cpu().numpy())

    title = f"LOCATA eval/task1/recording5 benchmark2, frame={args.frame_idx}"
    save_dual_feature_figure(
        phat_maps,
        lms_maps,
        output_dir / "feature_maps_charts.png",
        frame_idx=args.frame_idx,
        title=title,
    )
    save_dual_feature_projection_figure(
        phat_maps,
        lms_maps,
        output_dir / "feature_maps_projection.png",
        frame_idx=args.frame_idx,
        title=title,
        res_theta=args.projection_theta,
        res_phi=args.projection_phi,
    )
    save_dual_feature_projection_figure(
        phat_maps,
        lms_maps,
        output_dir / "feature_maps_projection_contrast.png",
        frame_idx=args.frame_idx,
        title=title,
        res_theta=args.projection_theta,
        res_phi=args.projection_phi,
        adaptive_contrast=True,
    )
    save_single_projection_figure(
        phat_maps,
        output_dir / "phat_projection_contrast.png",
        frame_idx=args.frame_idx,
        title="LOCATA recording5 SRP-PHAT",
        res_theta=args.projection_theta,
        res_phi=args.projection_phi,
        adaptive_contrast=True,
    )
    save_single_projection_figure(
        lms_maps,
        output_dir / "lms_projection_contrast.png",
        frame_idx=args.frame_idx,
        title="LOCATA recording5 SRP-LMS",
        res_theta=args.projection_theta,
        res_phi=args.projection_phi,
        adaptive_contrast=True,
    )
    print(
        f"Exported LOCATA stage1 features to {output_dir} "
        f"(source_fs={source_fs}, target_fs={args.fs}, windows={mic_sig_batch.shape[1]}, shape={tuple(dual_maps.shape)})"
    )


if __name__ == "__main__":
    main()
