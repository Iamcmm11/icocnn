from __future__ import annotations

import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from acousticTrackingDataset import Parameter
from utils import cart2sph, rms_angular_error_deg, sph2cart

from ..bridges import at_dataset
from ..features import DualFeatureIcoPreprocessor
from ..models import IFANModelConfig


STAGE3_SCENARIOS = (
    {"name": "scene_1", "snr_db": 30.0, "t60_s": 0.2},
    {"name": "scene_2", "snr_db": 30.0, "t60_s": 0.8},
    {"name": "scene_3", "snr_db": 5.0, "t60_s": 0.8},
    {"name": "scene_4", "snr_db": 5.0, "t60_s": 1.4},
)


def should_sync_cuda() -> bool:
    value = os.getenv("IFAN_SYNC_CUDA", "")
    return value.lower() not in ("", "0", "false", "no")


def resolve_librispeech_split(path: str | Path, split_name: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"LibriSpeech path does not exist: {path}")

    if path.name == split_name and path.is_dir():
        return path
    if (path / split_name).is_dir():
        return path / split_name
    if (path / "LibriSpeech" / split_name).is_dir():
        return path / "LibriSpeech" / split_name
    raise FileNotFoundError(f"Could not resolve LibriSpeech split '{split_name}' from {path}")


def build_librispeech_dataset(path: str | Path, split_name: str, signal_length: int):
    split_path = resolve_librispeech_split(path, split_name)
    return at_dataset.LibriSpeechDataset(str(split_path), signal_length, return_vad=True), split_path


def select_model_inputs(
    maps: torch.Tensor,
    model_config: IFANModelConfig,
    input_ablation_mode: str = "none",
) -> torch.Tensor:
    if input_ablation_mode == "none":
        return maps
    phat_maps, lms_maps = DualFeatureIcoPreprocessor.split_features(maps)
    if input_ablation_mode == "phat_only":
        return torch.cat((phat_maps, torch.zeros_like(lms_maps)), dim=1)
    if input_ablation_mode == "lms_only":
        return torch.cat((torch.zeros_like(phat_maps), lms_maps), dim=1)
    raise ValueError(f"Unsupported input_ablation_mode={input_ablation_mode!r}")


def frames_to_exclude(doa_batch: torch.Tensor) -> int:
    return 5 if doa_batch.shape[-2] > 5 else 0


def compute_metrics(model: torch.nn.Module, inputs: torch.Tensor, doa_batch: torch.Tensor) -> dict[str, float]:
    coords = model(inputs).contiguous()
    doa_cart = sph2cart(doa_batch.contiguous())
    loss = torch.nn.functional.mse_loss(coords.reshape(-1, 3), doa_cart.reshape(-1, 3))
    pred_sph = cart2sph(coords)
    offset = frames_to_exclude(doa_batch)
    rmsae = rms_angular_error_deg(
        doa_batch[..., offset:, :].reshape(-1, 2),
        pred_sph[..., offset:, :].reshape(-1, 2),
    )
    return {
        "loss": float(loss.item()),
        "rmsae_deg": float(rmsae.item() if hasattr(rmsae, "item") else rmsae),
    }


def evaluate_model_on_cache(model: torch.nn.Module, batches: Iterable[dict[str, torch.Tensor]]) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_rmsae = 0.0
    count = 0
    with torch.inference_mode():
        for batch in batches:
            metrics = compute_metrics(model, batch["inputs"], batch["doa"])
            total_loss += metrics["loss"]
            total_rmsae += metrics["rmsae_deg"]
            count += 1
    if count == 0:
        return {"loss": 0.0, "rmsae_deg": 0.0}
    return {
        "loss": total_loss / float(count),
        "rmsae_deg": total_rmsae / float(count),
    }


@contextmanager
def temporary_seed(seed: int):
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def build_random_trajectory_dataset(
    *,
    source_dataset,
    k: int,
    step: int,
    size: int,
    t60,
    snr,
    nb_points: int = 156,
):
    windowing = at_dataset.Windowing(k, step, window=np.hanning)
    return at_dataset.RandomTrajectoriesDataset(
        sourceDataset=source_dataset,
        room_sz=Parameter([3, 3, 2.5], [10, 8, 6]),
        T60=t60,
        abs_weights=Parameter([0.5] * 6, [1.0] * 6),
        array_setup=at_dataset.benchmark2_array_setup,
        array_pos=Parameter([0.1, 0.1, 0.1], [0.9, 0.9, 0.5]),
        SNR=snr,
        nb_points=nb_points,
        size=size,
        transforms=[windowing],
    )


def cache_ifan_batches(
    *,
    dataset,
    preprocessor: DualFeatureIcoPreprocessor,
    model_config: IFANModelConfig,
    input_ablation_mode: str = "none",
    batch_size: int,
    progress_callback=None,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    total_batches = max((len(dataset) + batch_size - 1) // batch_size, 1)
    for batch_index, start in enumerate(range(0, len(dataset), batch_size), start=1):
        stop = min(start + batch_size, len(dataset))
        if progress_callback is not None:
            progress_callback("before_get_batch", batch_index, total_batches, start, stop)
        mic_sig_batch, acoustic_scene_batch = dataset.get_batch(start, stop)
        if progress_callback is not None:
            progress_callback("after_get_batch", batch_index, total_batches, start, stop)
        with torch.no_grad():
            maps, doa_batch = preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
        if maps.is_cuda and should_sync_cuda():
            torch.cuda.synchronize(maps.device)
        if progress_callback is not None:
            progress_callback("after_preprocess", batch_index, total_batches, start, stop)
        batches.append(
            {
                "inputs": select_model_inputs(maps, model_config, input_ablation_mode).detach(),
                "doa": doa_batch.detach(),
            }
        )
        if progress_callback is not None:
            progress_callback("after_cache_append", batch_index, total_batches, start, stop)
    return batches


def cache_baseline_batches(
    *,
    dataset,
    preprocessor,
    batch_size: int,
    progress_callback=None,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    total_batches = max((len(dataset) + batch_size - 1) // batch_size, 1)
    for batch_index, start in enumerate(range(0, len(dataset), batch_size), start=1):
        stop = min(start + batch_size, len(dataset))
        if progress_callback is not None:
            progress_callback("before_get_batch", batch_index, total_batches, start, stop)
        mic_sig_batch, acoustic_scene_batch = dataset.get_batch(start, stop)
        if progress_callback is not None:
            progress_callback("after_get_batch", batch_index, total_batches, start, stop)
        with torch.no_grad():
            maps, doa_batch = preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
        if maps.is_cuda and should_sync_cuda():
            torch.cuda.synchronize(maps.device)
        if progress_callback is not None:
            progress_callback("after_preprocess", batch_index, total_batches, start, stop)
        batches.append({"inputs": maps.detach(), "doa": doa_batch.detach()})
        if progress_callback is not None:
            progress_callback("after_cache_append", batch_index, total_batches, start, stop)
    return batches


def build_scenario_caches(
    *,
    source_dataset,
    ifan_preprocessor: DualFeatureIcoPreprocessor,
    baseline_preprocessor,
    model_config: IFANModelConfig,
    input_ablation_mode: str = "none",
    k: int,
    step: int,
    batch_size: int,
    scenario_size: int,
    seed: int,
    nb_points: int = 156,
    progress_callback=None,
) -> dict[str, dict[str, object]]:
    scenario_caches: dict[str, dict[str, object]] = {}
    total_scenarios = len(STAGE3_SCENARIOS)
    for index, scenario in enumerate(STAGE3_SCENARIOS):
        with temporary_seed(seed + 500 + index):
            dataset = build_random_trajectory_dataset(
                source_dataset=source_dataset,
                k=k,
                step=step,
                size=scenario_size,
                t60=Parameter(float(scenario["t60_s"])),
                snr=Parameter(float(scenario["snr_db"])),
                nb_points=nb_points,
            )
            ifan_batches: list[dict[str, torch.Tensor]] = []
            baseline_batches: list[dict[str, torch.Tensor]] = []
            total_batches = max((len(dataset) + batch_size - 1) // batch_size, 1)
            for batch_index, start in enumerate(range(0, len(dataset), batch_size), start=1):
                stop = min(start + batch_size, len(dataset))
                if progress_callback is not None:
                    progress_callback(
                        scenario["name"],
                        index + 1,
                        total_scenarios,
                        batch_index,
                        total_batches,
                        start,
                        stop,
                    )
                mic_sig_batch, acoustic_scene_batch = dataset.get_batch(start, stop)
                with torch.no_grad():
                    ifan_maps, ifan_doa = ifan_preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
                    baseline_maps, baseline_doa = baseline_preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
                ifan_batches.append(
                    {
                        "inputs": select_model_inputs(ifan_maps, model_config, input_ablation_mode).detach(),
                        "doa": ifan_doa.detach(),
                    }
                )
                baseline_batches.append(
                    {
                        "inputs": baseline_maps.detach(),
                        "doa": baseline_doa.detach(),
                    }
                )
            scenario_caches[scenario["name"]] = {
                "scenario": dict(scenario),
                "ifan_batches": ifan_batches,
                "baseline_batches": baseline_batches,
            }
    return scenario_caches
