from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..bridges import at_dataset
from ..features import DualFeatureIcoPreprocessor
from ..models import IFANModel, IFANModelConfig


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


def resolve_librispeech_split(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"LibriSpeech path does not exist: {path}")

    if (path / "train-clean-100").is_dir():
        return path / "train-clean-100"
    if (path / "LibriSpeech" / "train-clean-100").is_dir():
        return path / "LibriSpeech" / "train-clean-100"
    return path


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


def run_engineering_check(
    config: IFANModelConfig,
    librispeech_path: str | Path = "datasets/LibriSpeech",
    signal_length: int = 2,
    k: int = 4096,
    step: int = 3072,
    fs: int = 16000,
) -> dict[str, object]:
    n_mics = 12
    height = 2**config.r
    width = 2 ** (config.r + 1)

    source_dataset, source_label = build_source_dataset(librispeech_path, signal_length, fs)
    windowing = at_dataset.Windowing(k, step, window=np.hanning)
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
        K=k,
        r=config.r,
        rn=at_dataset.benchmark2_array_setup.mic_pos,
        fs=fs,
        apply_vad=False,
    )
    maps, doa = preprocessor.data_transformation(mic_sig_batch=mic_sig_batch, acoustic_scene_batch=acoustic_scene_batch)
    phat_maps, lms_maps = preprocessor.split_features(maps)

    model = IFANModel(config)
    model.train()
    coords, attention = model(maps, return_attention=True)
    target = torch.linspace(-0.25, 0.25, steps=coords.numel(), dtype=coords.dtype, device=coords.device).reshape_as(coords)
    loss = torch.nn.functional.mse_loss(coords, target)
    loss.backward()

    input_finite = bool(torch.isfinite(torch.from_numpy(mic_sig_batch)).all().item())
    grad_tensors = [param.grad for param in model.parameters() if param.grad is not None]
    finite_grads = all(torch.isfinite(grad).all().item() for grad in grad_tensors)
    nonzero_grad_params = sum(int(grad.abs().sum().item() > 0) for grad in grad_tensors)

    summary = {
        "source_dataset": source_label,
        "windowed_microphone_batch_shape": tuple(mic_sig_batch.shape),
        "input_shape": tuple(maps.shape),
        "phat_shape": tuple(phat_maps.shape),
        "lms_shape": tuple(lms_maps.shape),
        "doa_tensor_shape": tuple(doa.shape),
        "output_shape": tuple(coords.shape),
        "phat_attention_shape": tuple(attention["phat"].shape),
        "lms_attention_shape": tuple(attention["lms"].shape),
        "loss": float(loss.item()),
        "trainable_params": model.count_parameters(trainable_only=True),
        "finite_input_windows": input_finite,
        "finite_output": bool(torch.isfinite(coords).all().item()),
        "finite_gradients": bool(finite_grads),
        "nonzero_gradient_params": nonzero_grad_params,
    }

    assert summary["windowed_microphone_batch_shape"] == (1, 3, n_mics, k)
    assert summary["input_shape"] == (1, 2, 3, 5, height, width)
    assert summary["phat_shape"] == (1, 1, 3, 5, height, width)
    assert summary["lms_shape"] == (1, 1, 3, 5, height, width)
    assert summary["output_shape"] == (1, 3, 3)
    assert attention["phat"].shape == attention["aux"].shape
    assert attention["phat"].shape[2] == config.branch_channels
    assert input_finite
    assert torch.isfinite(maps).all()
    assert torch.isfinite(coords).all()
    assert torch.isfinite(loss).all()
    assert torch.isfinite(attention["phat"]).all()
    assert torch.isfinite(attention["aux"]).all()
    assert grad_tensors
    assert finite_grads
    assert nonzero_grad_params > 0

    return summary
