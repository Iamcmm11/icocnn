from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ifan_edge.bridges import at_dataset
from ifan_edge.eval import STAGE1_SCENARIOS, save_dual_feature_figure
from ifan_edge.features import DualFeatureIcoPreprocessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export IFAN stage-1 PHAT/LMS feature visualizations.")
    parser.add_argument("--librispeech-path", required=True, help="Path to a LibriSpeech split on the training server.")
    parser.add_argument("--output-dir", required=True, help="Directory to store exported figures and numpy dumps.")
    parser.add_argument("--signal-length", type=int, default=20, help="Source signal duration in seconds.")
    parser.add_argument("--frame-idx", type=int, default=0, help="Window/frame index to visualize.")
    parser.add_argument("--k", type=int, default=4096, help="Window length.")
    parser.add_argument("--step", type=int, default=3072, help="Window step.")
    parser.add_argument("--fs", type=int, default=16000, help="Sampling rate.")
    parser.add_argument("--r", type=int, default=2, help="Icosahedral map resolution.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windowing = at_dataset.Windowing(args.k, args.step, window=np.hanning)
    source_dataset = at_dataset.LibriSpeechDataset(args.librispeech_path, args.signal_length, return_vad=True)
    preprocessor = DualFeatureIcoPreprocessor(
        N=12,
        K=args.k,
        r=args.r,
        rn=at_dataset.benchmark2_array_setup.mic_pos,
        fs=args.fs,
        apply_vad=False,
    )

    for scenario in STAGE1_SCENARIOS:
        dataset = at_dataset.RandomTrajectoryDataset(
            sourceDataset=source_dataset,
            room_sz=at_dataset.Parameter([3, 3, 2.5], [10, 8, 6]),
            T60=at_dataset.Parameter(scenario["t60_s"]),
            abs_weights=at_dataset.Parameter([0.5] * 6, [1.0] * 6),
            array_setup=at_dataset.benchmark2_array_setup,
            array_pos=at_dataset.Parameter([0.1, 0.1, 0.1], [0.9, 0.9, 0.5]),
            SNR=at_dataset.Parameter(scenario["snr_db"]),
            nb_points=156,
            transforms=[windowing],
        )

        mic_sig_batch, acoustic_scene_batch = dataset.get_batch(0, 1)
        dual_maps, _ = preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
        phat_maps, lms_maps = preprocessor.split_features(dual_maps)

        scenario_dir = output_dir / scenario["name"]
        scenario_dir.mkdir(parents=True, exist_ok=True)

        np.save(scenario_dir / "dual_maps.npy", dual_maps.detach().cpu().numpy())
        np.save(scenario_dir / "phat_maps.npy", phat_maps.detach().cpu().numpy())
        np.save(scenario_dir / "lms_maps.npy", lms_maps.detach().cpu().numpy())

        title = f'{scenario["name"]}: SNR={scenario["snr_db"]}dB, T60={scenario["t60_s"]}s'
        save_dual_feature_figure(
            phat_maps,
            lms_maps,
            scenario_dir / "feature_maps.png",
            frame_idx=args.frame_idx,
            title=title,
        )
        print(f"Exported {scenario['name']} to {scenario_dir}")


if __name__ == "__main__":
    main()
