from __future__ import annotations

import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from acousticTrackingDataset import Parameter
from utils import angular_error, cart2sph, rms_angular_error_deg, sph2cart

from ..bridges import at_dataset, at_learners, at_models
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


def build_ifan_preprocessor_from_config(config, device: torch.device | None = None) -> DualFeatureIcoPreprocessor:
    preprocessor = DualFeatureIcoPreprocessor(
        N=at_dataset.benchmark2_array_setup.mic_pos.shape[0],
        K=config.k,
        r=config.r,
        rn=at_dataset.benchmark2_array_setup.mic_pos,
        fs=config.fs,
        apply_vad=config.apply_vad,
        lms_order=config.lms_order,
        lms_step_size=config.lms_step_size,
        lms_map_normalize=config.lms_map_normalize,
        lms_map_mode=config.lms_map_mode,
        lms_peak_sigma=config.lms_peak_sigma,
        lms_update_mode=config.lms_update_mode,
        lms_normalized=config.lms_normalized,
        lms_include_self_pairs=config.lms_include_self_pairs,
        lms_backend=config.lms_backend,
        lms_block_size=config.lms_block_size,
        lms_fft_size=config.lms_fft_size,
    )
    if device is not None:
        if device.type == "cuda":
            preprocessor.cuda_activated = True
            preprocessor.phat.cuda()
            preprocessor.lms.cuda()
        else:
            preprocessor.cuda_activated = False
            preprocessor.phat.cpu()
            preprocessor.lms.cpu()
    return preprocessor


def build_baseline_preprocessor_from_config(config, device: torch.device | None = None):
    preprocessor = at_learners.TrackingFromIcoMapsPreprocessor(
        N=at_dataset.benchmark2_array_setup.mic_pos.shape[0],
        K=config.k,
        r=config.r,
        rn=at_dataset.benchmark2_array_setup.mic_pos,
        fs=config.fs,
        apply_vad=config.apply_vad,
    )
    if device is not None:
        if device.type == "cuda":
            preprocessor.cuda_activated = True
            preprocessor.gcc.cuda()
            preprocessor.srp.cuda()
        else:
            preprocessor.cuda_activated = False
            preprocessor.gcc.cpu()
            preprocessor.srp.cpu()
    return preprocessor


def load_baseline_model_from_config(config, device: torch.device):
    model = at_models.IcoTempCNN(config.r, 32, Cin=1, smooth_vertices=config.smooth_vertices)
    checkpoint_path = Path(config.baseline_checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint does not exist: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    non_buffer_missing = [key for key in incompatible.missing_keys if not key.endswith(".mask")]
    if non_buffer_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Baseline checkpoint is incompatible with the current icoCNN model.\n"
            f"Unexpected keys: {list(incompatible.unexpected_keys)}\n"
            f"Missing non-buffer keys: {non_buffer_missing}"
        )
    model.to(device)
    return model


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
    details = compute_prediction_details(model, inputs, doa_batch)
    return {
        "loss": details["loss"],
        "rmsae_deg": details["rmsae_deg"],
    }


def compute_prediction_details(model: torch.nn.Module, inputs: torch.Tensor, doa_batch: torch.Tensor) -> dict[str, object]:
    coords = model(inputs).contiguous()
    doa_cart = sph2cart(doa_batch.contiguous())
    loss = torch.nn.functional.mse_loss(coords.reshape(-1, 3), doa_cart.reshape(-1, 3))
    pred_sph = cart2sph(coords)
    offset = frames_to_exclude(doa_batch)
    target_trim = doa_batch[..., offset:, :]
    pred_trim = pred_sph[..., offset:, :]
    frame_errors_deg = angular_error(
        pred_trim[..., 0],
        pred_trim[..., 1],
        target_trim[..., 0],
        target_trim[..., 1],
    ) * (180.0 / np.pi)
    trajectory_rmsae_deg = torch.sqrt(torch.mean(torch.square(frame_errors_deg), dim=-1))
    rmsae = rms_angular_error_deg(
        target_trim.reshape(-1, 2),
        pred_trim.reshape(-1, 2),
    )
    return {
        "loss": float(loss.item()),
        "rmsae_deg": float(rmsae.item() if hasattr(rmsae, "item") else rmsae),
        "pred_sph": pred_sph.detach(),
        "frame_errors_deg": frame_errors_deg.detach(),
        "trajectory_rmsae_deg": trajectory_rmsae_deg.detach(),
        "offset_frames": int(offset),
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


def evaluate_detailed_model_on_cache(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
) -> dict[str, object]:
    model.eval()
    batch_reports: list[dict[str, object]] = []
    total_loss = 0.0
    total_rmsae = 0.0
    count = 0
    with torch.inference_mode():
        for batch in batches:
            details = compute_prediction_details(model, batch["inputs"], batch["doa"])
            total_loss += float(details["loss"])
            total_rmsae += float(details["rmsae_deg"])
            count += 1
            batch_reports.append(
                {
                    "loss": float(details["loss"]),
                    "rmsae_deg": float(details["rmsae_deg"]),
                    "offset_frames": int(details["offset_frames"]),
                    "trajectory_rmsae_deg": details["trajectory_rmsae_deg"].detach().cpu().tolist(),
                    "frame_errors_deg": details["frame_errors_deg"].detach().cpu().tolist(),
                }
            )
    aggregate = {"loss": 0.0, "rmsae_deg": 0.0}
    if count > 0:
        aggregate = {
            "loss": total_loss / float(count),
            "rmsae_deg": total_rmsae / float(count),
        }
    return {
        **aggregate,
        "batch_reports": batch_reports,
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


def resolve_stage3_scenario(name: str) -> dict[str, float | str]:
    for scenario in STAGE3_SCENARIOS:
        if scenario["name"] == name:
            return dict(scenario)
    raise KeyError(f"Unknown stage-3 scenario: {name}")


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
        scenario_caches[scenario["name"]] = build_single_scenario_cache(
            source_dataset=source_dataset,
            scenario=scenario,
            scenario_index=index,
            total_scenarios=total_scenarios,
            ifan_preprocessor=ifan_preprocessor,
            baseline_preprocessor=baseline_preprocessor,
            model_config=model_config,
            input_ablation_mode=input_ablation_mode,
            k=k,
            step=step,
            batch_size=batch_size,
            scenario_size=scenario_size,
            seed=seed,
            nb_points=nb_points,
            progress_callback=progress_callback,
        )
    return scenario_caches


def build_single_scenario_cache(
    *,
    source_dataset,
    scenario: dict[str, object] | str,
    scenario_index: int | None = None,
    total_scenarios: int = 1,
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
) -> dict[str, object]:
    scenario_dict = resolve_stage3_scenario(scenario) if isinstance(scenario, str) else dict(scenario)
    if "name" not in scenario_dict:
        raise KeyError("Scenario must include a 'name' field.")
    if scenario_index is None:
        index = next(
            idx for idx, candidate in enumerate(STAGE3_SCENARIOS) if candidate["name"] == scenario_dict["name"]
        )
    else:
        index = int(scenario_index)
    with temporary_seed(seed + 500 + index):
        dataset = build_random_trajectory_dataset(
            source_dataset=source_dataset,
            k=k,
            step=step,
            size=scenario_size,
            t60=Parameter(float(scenario_dict["t60_s"])),
            snr=Parameter(float(scenario_dict["snr_db"])),
            nb_points=nb_points,
        )
        ifan_batches: list[dict[str, torch.Tensor]] = []
        baseline_batches: list[dict[str, torch.Tensor]] = []
        total_batches = max((len(dataset) + batch_size - 1) // batch_size, 1)
        for batch_index, start in enumerate(range(0, len(dataset), batch_size), start=1):
            stop = min(start + batch_size, len(dataset))
            if progress_callback is not None:
                progress_callback(
                    str(scenario_dict["name"]),
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
    return {
        "scenario": scenario_dict,
        "ifan_batches": ifan_batches,
        "baseline_batches": baseline_batches,
    }
