from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ifan_edge.bridges import at_dataset
from ifan_edge.features import DualFeatureIcoPreprocessor
from ifan_edge.models import IFANModel, IFANModelConfig
from visualize_stage1_features import resolve_librispeech_split


class SyntheticWindowSourceDataset:
    """Small deterministic fallback source dataset for stage-2 engineering checks."""

    def __init__(self, signal_length_s: int, fs: int, size: int = 8):
        self.signal_length_s = int(signal_length_s)
        self.fs = int(fs)
        self.size = int(size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        num_samples = self.signal_length_s * self.fs
        t = np.arange(num_samples, dtype=np.float32) / float(self.fs)
        freq_1 = 180.0 + 17.0 * (idx % 5)
        freq_2 = 360.0 + 29.0 * (idx % 7)
        envelope = 0.6 + 0.4 * np.sin(2.0 * np.pi * 1.5 * t)
        carrier = 0.65 * np.sin(2.0 * np.pi * freq_1 * t) + 0.35 * np.sin(2.0 * np.pi * freq_2 * t + 0.3)
        silence_mask = ((t % 0.5) < 0.42).astype(np.float32)
        signal = (envelope * carrier * silence_mask).astype(np.float32)
        vad = (silence_mask > 0).astype(np.float32)
        return signal, vad


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


def build_source_dataset(librispeech_path: str | Path, signal_length: int, fs: int):
    try:
        split_path = resolve_librispeech_split(librispeech_path)
    except FileNotFoundError:
        split_path = None

    if split_path is not None:
        return (
            at_dataset.LibriSpeechDataset(str(split_path), signal_length, return_vad=True),
            f"LibriSpeech split: {split_path}",
        )

    return (
        SyntheticWindowSourceDataset(signal_length_s=signal_length, fs=fs),
        "synthetic source fallback",
    )


def main() -> None:
    args = build_parser().parse_args()
    n_mics = 12
    height = 2**args.r
    width = 2 ** (args.r + 1)

    source_dataset, source_label = build_source_dataset(args.librispeech_path, args.signal_length, args.fs)
    windowing = at_dataset.Windowing(args.k, args.step, window=np.hanning)
    dataset = at_dataset.RandomTrajectoryDataset(
        sourceDataset=source_dataset,
        room_sz=at_dataset.Parameter([3, 3, 2.5], [10, 8, 6]),
        T60=at_dataset.Parameter(0.2),
        abs_weights=at_dataset.Parameter([0.5] * 6, [1.0] * 6),
        array_setup=at_dataset.benchmark2_array_setup,
        array_pos=at_dataset.Parameter([0.1, 0.1, 0.1], [0.9, 0.9, 0.5]),
        SNR=at_dataset.Parameter(30.0),
        nb_points=156,
        transforms=[windowing],
    )

    mic_sig_batch, acoustic_scene_batch = dataset.get_batch(0, 1)
    mic_sig_batch = mic_sig_batch[:, :3, :, :]
    if mic_sig_batch.shape[1] != 3:
        raise AssertionError(f"Expected three windowed frames for the engineering check, got {mic_sig_batch.shape[1]}")

    preprocessor = DualFeatureIcoPreprocessor(
        N=n_mics,
        K=args.k,
        r=args.r,
        rn=at_dataset.benchmark2_array_setup.mic_pos,
        fs=args.fs,
        apply_vad=False,
    )
    maps, doa = preprocessor.data_transformation(mic_sig_batch=mic_sig_batch, acoustic_scene_batch=acoustic_scene_batch)
    phat_maps, lms_maps = preprocessor.split_features(maps)

    model = IFANModel(
        IFANModelConfig(
            r=args.r,
            phat_in_channels=1,
            aux_in_channels=1,
            branch_channels=16,
            fused_channels=16,
            use_residual_block=False,
        )
    )
    model.train()
    coords, attention = model(maps, return_attention=True)
    target = torch.linspace(-0.25, 0.25, steps=coords.numel(), dtype=coords.dtype, device=coords.device).reshape_as(coords)
    loss = torch.nn.functional.mse_loss(coords, target)
    loss.backward()

    input_finite = bool(torch.isfinite(torch.from_numpy(mic_sig_batch)).all().item())
    grad_tensors = [param.grad for param in model.parameters() if param.grad is not None]
    finite_grads = all(torch.isfinite(grad).all().item() for grad in grad_tensors)
    nonzero_grad_params = sum(int(grad.abs().sum().item() > 0) for grad in grad_tensors)

    print("Source dataset:", source_label)
    print("Windowed microphone batch shape:", tuple(mic_sig_batch.shape))
    print("Input shape:", tuple(maps.shape))
    print("PHAT shape:", tuple(phat_maps.shape))
    print("LMS shape:", tuple(lms_maps.shape))
    print("DOA tensor shape:", tuple(doa.shape))
    print("Output shape:", tuple(coords.shape))
    print("PHAT attention shape:", tuple(attention["phat"].shape))
    print("Aux attention shape:", tuple(attention["aux"].shape))
    print("Loss:", float(loss.item()))
    print("Finite input windows:", input_finite)
    print("Finite output:", bool(torch.isfinite(coords).all().item()))
    print("Finite gradients:", bool(finite_grads))
    print("Nonzero gradient params:", nonzero_grad_params)

    assert tuple(mic_sig_batch.shape) == (1, 3, n_mics, args.k)
    assert tuple(maps.shape) == (1, 2, 3, 5, height, width)
    assert tuple(phat_maps.shape) == (1, 1, 3, 5, height, width)
    assert tuple(lms_maps.shape) == (1, 1, 3, 5, height, width)
    assert tuple(coords.shape) == (1, 3, 3)
    assert attention["phat"].shape == attention["aux"].shape
    assert attention["phat"].shape[2] == 16
    assert input_finite
    assert torch.isfinite(maps).all()
    assert torch.isfinite(coords).all()
    assert torch.isfinite(loss).all()
    assert torch.isfinite(attention["phat"]).all()
    assert torch.isfinite(attention["aux"]).all()
    assert grad_tensors
    assert finite_grads
    assert nonzero_grad_params > 0

    print("PHAT + LMS stage-2 engineering check passed.")


if __name__ == "__main__":
    main()
