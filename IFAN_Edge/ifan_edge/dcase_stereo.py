from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from .features import DualFeatureIcoPreprocessor
from .models import DcaseAzimuthHeadConfig, DcaseAzimuthOnlyIFANModel, IFANModel, IFANModelConfig
from .models.map_maba import MapMABATemporalConfig
from utils import cart2sph, sph2cart


@dataclass
class PreparedDcaseClip:
    row: dict[str, str]
    windows: np.ndarray
    scene: Any
    target_azimuth_deg: np.ndarray
    target_distance_cm: np.ndarray
    active_mask: np.ndarray


class DcaseScene:
    def __init__(self, doa: np.ndarray, vad: np.ndarray):
        self.DOAw = [doa.astype(np.float32, copy=False)]
        self.vad = vad.astype(np.float32, copy=False)


def summarize_scalar(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def circular_diff_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return ((pred - target + 180.0) % 360.0) - 180.0


def fold_front_deg(angle: np.ndarray) -> np.ndarray:
    wrapped = ((angle + 180.0) % 360.0) - 180.0
    return np.where(wrapped > 90.0, 180.0 - wrapped, np.where(wrapped < -90.0, -180.0 - wrapped, wrapped))


def dcase_azimuth_to_phi_rad(azimuth_deg: np.ndarray) -> np.ndarray:
    return np.deg2rad(90.0 - azimuth_deg)


def phi_rad_to_dcase_azimuth_deg(phi_rad: np.ndarray) -> np.ndarray:
    return ((90.0 - np.rad2deg(phi_rad) + 180.0) % 360.0) - 180.0


def angular_error_deg(
    theta_pred: np.ndarray,
    phi_pred: np.ndarray,
    theta_true: np.ndarray,
    phi_true: np.ndarray,
) -> np.ndarray:
    aux = (
        np.cos(theta_true) * np.cos(theta_pred)
        + np.sin(theta_true) * np.sin(theta_pred) * np.cos(phi_true - phi_pred)
    )
    return np.rad2deg(np.arccos(np.clip(aux, -0.99999, 0.99999)))


def load_manifest(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]
    return rows


def select_rows_by_split(rows: list[dict[str, str]], allowed_splits: set[str]) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] in allowed_splits]


def split_manifest_rows(
    rows: list[dict[str, str]],
    *,
    validation_fraction: float,
    validation_min_per_bucket: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["split"], row["subset"])
        grouped.setdefault(key, []).append(row)

    train_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []
    rng = random.Random(seed)
    for key in sorted(grouped):
        bucket = list(grouped[key])
        rng.shuffle(bucket)
        if len(bucket) == 1:
            validation_count = 0
        else:
            validation_count = max(validation_min_per_bucket, int(round(len(bucket) * validation_fraction)))
            validation_count = min(validation_count, len(bucket) - 1)
        validation_rows.extend(bucket[:validation_count])
        train_rows.extend(bucket[validation_count:])
    return train_rows, validation_rows


def read_metadata_targets(metadata_path: Path, window_centers_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_frame: dict[int, list[tuple[float, float]]] = {}
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            by_frame.setdefault(frame, []).append((float(row["azimuth"]), float(row["distance"])))
    if not by_frame:
        raise ValueError(f"Metadata has no active frames: {metadata_path}")

    frames = sorted(by_frame)
    azimuths: list[float] = []
    distances: list[float] = []
    for frame in frames:
        positions = sorted(set(by_frame[frame]))
        if len(positions) != 1:
            raise ValueError(f"Expected one position in {metadata_path} frame {frame}, got {positions}")
        azimuth, distance = positions[0]
        azimuths.append(azimuth)
        distances.append(distance)

    frame_times = np.asarray(frames, dtype=np.float64) * 0.1 + 0.05
    azimuth_interp = np.interp(window_centers_s, frame_times, np.asarray(azimuths, dtype=np.float64))
    distance_interp = np.interp(window_centers_s, frame_times, np.asarray(distances, dtype=np.float64))
    center_frames = np.floor(window_centers_s / 0.1).astype(int)
    active_mask = np.asarray([frame in by_frame for frame in center_frames], dtype=bool)
    return azimuth_interp.astype(np.float32), distance_interp.astype(np.float32), active_mask


def resample_audio(audio: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return audio.astype(np.float32, copy=False)
    gcd = math.gcd(int(fs_in), int(fs_out))
    up = int(fs_out // gcd)
    down = int(fs_in // gcd)
    return signal.resample_poly(audio, up, down, axis=0).astype(np.float32, copy=False)


def window_audio(audio: np.ndarray, *, k: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f"Expected stereo audio with shape [samples, 2], got {audio.shape}")
    length = int(audio.shape[0])
    if length < k:
        raise ValueError(f"Audio shorter than window: length={length}, k={k}")
    n_windows = int(np.floor(length / step - k / step + 1))
    if n_windows <= 0:
        raise ValueError(f"No windows for length={length}, k={k}, step={step}")
    shape = (n_windows, k, audio.shape[1])
    strides = (step * audio.shape[1] * audio.itemsize, audio.shape[1] * audio.itemsize, audio.itemsize)
    windows = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
    windows = np.ascontiguousarray(windows.transpose(0, 2, 1))
    windows = windows * np.hanning(k).astype(np.float32)
    center_samples = np.arange(n_windows, dtype=np.float64) * step + 0.5 * k
    return windows.astype(np.float32, copy=False), center_samples


def build_model_config(raw: dict[str, Any]) -> IFANModelConfig:
    payload = dict(raw)
    payload["map_maba"] = MapMABATemporalConfig.from_mapping(payload.get("map_maba", {}))
    payload["weak_map_maba"] = MapMABATemporalConfig.from_mapping(payload.get("weak_map_maba", {}))
    return IFANModelConfig(**payload)


def build_azimuth_head_config(raw: dict[str, Any] | None) -> DcaseAzimuthHeadConfig:
    return DcaseAzimuthHeadConfig.from_mapping(raw)


def remap_legacy_map_refiner_keys(
    state_dict: dict[str, torch.Tensor],
    *,
    model_config: IFANModelConfig,
) -> dict[str, torch.Tensor]:
    state_dict = dict(state_dict)
    if getattr(model_config, "map_refiner_position", "pre_softargmax") != "pre_readout":
        return state_dict
    has_feature_keys = any(key.startswith("feature_refiner.") for key in state_dict)
    has_map_keys = any(key.startswith("map_refiner.") for key in state_dict)
    if has_feature_keys or not has_map_keys:
        return state_dict
    return {
        key.replace("map_refiner.", "feature_refiner.", 1) if key.startswith("map_refiner.") else key: value
        for key, value in state_dict.items()
    }


def load_ifan_checkpoint(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    model_config: IFANModelConfig,
    *,
    state_key_prefix: str = "",
) -> list[str]:
    state_dict = remap_legacy_map_refiner_keys(checkpoint["model_state_dict"], model_config=model_config)
    model_keys = set(model.state_dict().keys())
    load_notes: list[str] = []
    stale_norm_keys = [
        f"{state_key_prefix}phat_branch.residual.norm.weight",
        f"{state_key_prefix}phat_branch.residual.norm.bias",
        f"{state_key_prefix}aux_branch.residual.norm.weight",
        f"{state_key_prefix}aux_branch.residual.norm.bias",
    ]
    dropped = [key for key in stale_norm_keys if key in state_dict and key not in model_keys]
    if dropped:
        for key in dropped:
            state_dict.pop(key, None)
        load_notes.append(
            "Dropped stale frontend residual norm checkpoint keys no longer present in IFANModel: "
            + ", ".join(dropped)
        )
    model.load_state_dict(state_dict, strict=True)
    return load_notes


def checkpoint_model_kind(checkpoint: dict[str, Any]) -> str:
    return str(checkpoint.get("model_kind", "ifan_coords"))


def load_ifan_backbone_into_azimuth_model(
    model: DcaseAzimuthOnlyIFANModel,
    checkpoint: dict[str, Any],
    model_config: IFANModelConfig,
) -> list[str]:
    state_dict = remap_legacy_map_refiner_keys(checkpoint["model_state_dict"], model_config=model_config)
    remapped = {f"backbone.{key}": value for key, value in state_dict.items()}
    model_keys = set(model.state_dict().keys())
    load_notes: list[str] = []
    stale_norm_keys = [
        "backbone.phat_branch.residual.norm.weight",
        "backbone.phat_branch.residual.norm.bias",
        "backbone.aux_branch.residual.norm.weight",
        "backbone.aux_branch.residual.norm.bias",
    ]
    dropped = [key for key in stale_norm_keys if key in remapped and key not in model_keys]
    if dropped:
        for key in dropped:
            remapped.pop(key, None)
        load_notes.append(
            "Dropped stale frontend residual norm checkpoint keys no longer present in IFANModel: "
            + ", ".join(dropped)
        )
    incompatible = model.load_state_dict(remapped, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("azimuth_head.")]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "IFAN backbone init checkpoint is incompatible with the DCASE azimuth-only model.\n"
            f"Unexpected keys: {unexpected}\n"
            f"Missing keys: {missing}"
        )
    load_notes.append("Initialized azimuth-only model backbone from Stage-1 IFAN checkpoint; azimuth_head kept random.")
    return load_notes


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, IFANModelConfig, list[str]]:
    model_kind = checkpoint_model_kind(checkpoint)
    if model_kind == "dcase_azimuth_only_ifan":
        raw_backbone_config = checkpoint.get("backbone_model_config")
        if raw_backbone_config is None:
            raw_backbone_config = checkpoint["model_config"]
        backbone_config = build_model_config(raw_backbone_config)
        head_config = build_azimuth_head_config(checkpoint.get("azimuth_head_config"))
        model = DcaseAzimuthOnlyIFANModel(backbone_config, head_config)
        load_notes = load_ifan_checkpoint(
            model,
            checkpoint,
            backbone_config,
            state_key_prefix="backbone.",
        )
        model.to(device)
        model.eval()
        return model, backbone_config, load_notes

    model_config = build_model_config(checkpoint["model_config"])
    model = IFANModel(model_config)
    load_notes = load_ifan_checkpoint(model, checkpoint, model_config)
    model.to(device)
    model.eval()
    return model, model_config, load_notes


def stereo_proxy_positions(baseline_m: float) -> np.ndarray:
    half = float(baseline_m) / 2.0
    return np.asarray([[-half, 0.0, 0.0], [half, 0.0, 0.0]], dtype=np.float32)


def create_preprocessor(
    config: dict[str, Any],
    *,
    rn: np.ndarray,
    device: torch.device,
    apply_vad: bool,
) -> DualFeatureIcoPreprocessor:
    preprocessor = DualFeatureIcoPreprocessor(
        N=2,
        K=int(config["k"]),
        r=int(config["r"]),
        rn=rn,
        fs=int(config["fs"]),
        apply_vad=apply_vad,
        lms_order=int(config.get("lms_order", 64)),
        lms_step_size=float(config.get("lms_step_size", 0.01)),
        lms_map_normalize=bool(config.get("lms_map_normalize", True)),
        lms_map_mode=str(config.get("lms_map_mode", "tau_sample")),
        lms_peak_sigma=float(config.get("lms_peak_sigma", 2.0)),
        lms_update_mode=str(config.get("lms_update_mode", "trajectory_tracking")),
        lms_normalized=bool(config.get("lms_normalized", False)),
        lms_include_self_pairs=bool(config.get("lms_include_self_pairs", True)),
        lms_backend=str(config.get("lms_backend", "frequency_block")),
        lms_block_size=int(config.get("lms_block_size", 256)),
        lms_fft_size=config.get("lms_fft_size", None),
        srp_variant=str(config.get("srp_variant", "paper_original")),
        phat_sinc_half_width=int(config.get("phat_sinc_half_width", 0)),
    )
    if device.type == "cuda":
        preprocessor.cuda_activated = True
        preprocessor.phat.cuda()
        preprocessor.lms.cuda()
    else:
        preprocessor.cuda_activated = False
        preprocessor.phat.cpu()
        preprocessor.lms.cpu()
    return preprocessor


def prepare_row(row: dict[str, str], dataset_root: Path, *, fs: int, k: int, step: int) -> PreparedDcaseClip:
    audio_path = dataset_root / row["audio_relpath"]
    metadata_path = dataset_root / row["metadata_relpath"]
    audio, fs_in = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio = resample_audio(audio, int(fs_in), int(fs))
    windows, center_samples = window_audio(audio, k=k, step=step)
    centers_s = center_samples / float(fs)
    azimuth_deg, distance_cm, active_mask = read_metadata_targets(metadata_path, centers_s)
    theta = np.full_like(azimuth_deg, np.pi / 2.0, dtype=np.float32)
    phi = dcase_azimuth_to_phi_rad(azimuth_deg).astype(np.float32)
    doa = np.stack((theta, phi), axis=-1)
    vad = np.ones((windows.shape[0], windows.shape[-1]), dtype=np.float32)
    return PreparedDcaseClip(
        row=row,
        windows=windows,
        scene=DcaseScene(doa=doa, vad=vad),
        target_azimuth_deg=azimuth_deg,
        target_distance_cm=distance_cm,
        active_mask=active_mask,
    )


def prepare_rows(
    rows: list[dict[str, str]],
    dataset_root: Path,
    *,
    fs: int,
    k: int,
    step: int,
    progress_label: str | None = None,
) -> list[PreparedDcaseClip]:
    prepared: list[PreparedDcaseClip] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        prepared.append(prepare_row(row, dataset_root, fs=fs, k=k, step=step))
        if progress_label is not None and (index == total or index % 50 == 0):
            print(
                json.dumps(
                    {
                        "event": "dcase_prepare_progress",
                        "label": progress_label,
                        "processed": index,
                        "total": total,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return prepared


def build_active_mask_tensor(
    clips: list[PreparedDcaseClip],
    *,
    exclude_initial_windows: int,
    device: torch.device,
) -> torch.Tensor:
    mask = np.stack([clip.active_mask.copy() for clip in clips], axis=0)
    if exclude_initial_windows > 0:
        mask[:, :exclude_initial_windows] = False
    return torch.from_numpy(mask.astype(np.bool_)).to(device=device)


def build_folded_azimuth_sincos_tensor(
    clips: list[PreparedDcaseClip],
    *,
    device: torch.device,
) -> torch.Tensor:
    folded_azimuth_deg = np.stack([fold_front_deg(clip.target_azimuth_deg) for clip in clips], axis=0)
    phi = dcase_azimuth_to_phi_rad(folded_azimuth_deg).astype(np.float32)
    target = np.stack((np.sin(phi), np.cos(phi)), axis=-1).astype(np.float32)
    return torch.from_numpy(target).to(device=device)


def evaluate_prepared_batch(
    *,
    model: torch.nn.Module,
    preprocessor: DualFeatureIcoPreprocessor,
    prepared: list[PreparedDcaseClip],
    device: torch.device,
    exclude_initial_windows: int,
) -> list[dict[str, Any]]:
    mic_batch = np.stack([clip.windows for clip in prepared], axis=0)
    scenes = [clip.scene for clip in prepared]
    with torch.inference_mode():
        maps, _ = preprocessor.data_transformation(mic_batch, scenes)
        coords = model(maps.to(device)).contiguous()
        pred_sph = cart2sph(coords).detach().cpu().numpy()

    reports = []
    for clip, pred in zip(prepared, pred_sph):
        row = clip.row
        target_az = clip.target_azimuth_deg
        active_mask = clip.active_mask.copy()
        if exclude_initial_windows > 0:
            active_mask[:exclude_initial_windows] = False
        if not np.any(active_mask):
            continue
        pred_theta = pred[:, 0]
        pred_phi = pred[:, 1]
        pred_az_raw = phi_rad_to_dcase_azimuth_deg(pred_phi)
        pred_az_folded = fold_front_deg(pred_az_raw)
        target_az_folded = fold_front_deg(target_az)
        raw_diff = circular_diff_deg(pred_az_raw[active_mask], target_az[active_mask])
        folded_diff = circular_diff_deg(pred_az_folded[active_mask], target_az_folded[active_mask])
        spherical_error = angular_error_deg(
            pred_theta[active_mask],
            pred_phi[active_mask],
            np.full(np.count_nonzero(active_mask), np.pi / 2.0, dtype=np.float32),
            dcase_azimuth_to_phi_rad(target_az[active_mask]),
        )
        reports.append(
            {
                "clip_id": row["clip_id"],
                "subset": row["subset"],
                "split": row["split"],
                "audio_relpath": row["audio_relpath"],
                "metadata_relpath": row["metadata_relpath"],
                "evaluated_windows": int(np.count_nonzero(active_mask)),
                "target_azimuth_mean_deg": float(np.mean(target_az[active_mask])),
                "pred_azimuth_raw_mean_deg": float(np.mean(pred_az_raw[active_mask])),
                "pred_azimuth_folded_mean_deg": float(np.mean(pred_az_folded[active_mask])),
                "doa_error_deg": float(np.mean(np.abs(folded_diff))),
                "folded_azimuth_mae_deg": float(np.mean(np.abs(folded_diff))),
                "folded_azimuth_rmse_deg": float(np.sqrt(np.mean(np.square(folded_diff)))),
                "raw_azimuth_mae_deg": float(np.mean(np.abs(raw_diff))),
                "raw_azimuth_rmse_deg": float(np.sqrt(np.mean(np.square(raw_diff)))),
                "horizontal_assumption_rmsae_deg": float(np.sqrt(np.mean(np.square(spherical_error)))),
            }
        )
    return reports


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "doa_error_deg",
        "folded_azimuth_mae_deg",
        "folded_azimuth_rmse_deg",
        "raw_azimuth_mae_deg",
        "raw_azimuth_rmse_deg",
        "horizontal_assumption_rmsae_deg",
    ]
    payload = {"count": len(reports), "evaluated_windows": int(sum(row["evaluated_windows"] for row in reports))}
    for field in fields:
        payload[field] = summarize_scalar([float(row[field]) for row in reports])
    return payload


def build_report(
    *,
    reports: list[dict[str, Any]],
    checkpoint: str,
    manifest: str,
    dataset_root: str,
    device: str,
    limit: int | None,
    exclude_initial_windows: int,
    stereo_baseline_m: float,
    rn: np.ndarray,
    config: dict[str, Any],
    model_payload: dict[str, Any],
) -> dict[str, Any]:
    by_subset = {
        subset: aggregate_reports([row for row in reports if row["subset"] == subset])
        for subset in sorted({row["subset"] for row in reports})
    }
    by_split = {
        split: aggregate_reports([row for row in reports if row["split"] == split])
        for split in sorted({row["split"] for row in reports})
    }
    return {
        "kind": "stage3_dcase2025_stereo_transfer_evaluation",
        "checkpoint": checkpoint,
        "manifest": manifest,
        "dataset_root": dataset_root,
        "device": device,
        "limit": limit,
        "exclude_initial_windows": int(exclude_initial_windows),
        "stereo_proxy": {
            "baseline_m": float(stereo_baseline_m),
            "mic_positions_m": rn.tolist(),
            "target_mapping": "DCASE azimuth 0 deg -> +y/front; right positive -> +x",
        },
        "config": config,
        "model": model_payload,
        "overall": aggregate_reports(reports),
        "by_subset": by_subset,
        "by_split": by_split,
        "per_clip": reports,
    }


def build_markdown(report: dict[str, Any]) -> str:
    def row(label: str, payload: dict[str, Any]) -> str:
        return (
            f"| {label} | {payload['count']} | {payload['evaluated_windows']} | "
            f"{payload['doa_error_deg']['mean']:.4f} | "
            f"{payload['folded_azimuth_rmse_deg']['mean']:.4f} | "
            f"{payload['raw_azimuth_mae_deg']['mean']:.4f} | "
            f"{payload['raw_azimuth_rmse_deg']['mean']:.4f} | "
            f"{payload['horizontal_assumption_rmsae_deg']['mean']:.4f} |"
        )

    lines = [
        "# DCASE2025 立体声迁移测试",
        "",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- manifest: `{report['manifest']}`",
        f"- 立体声代理基线间距: `{report['stereo_proxy']['baseline_m']:.3f} m`",
        f"- 前部排除窗口数: `{report['exclude_initial_windows']}`",
        "",
        "| Subset | Clips | Windows | DOA error (deg) | Folded Az RMSE | Raw Az MAE | Raw Az RMSE | Horizontal Assumption RMSAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        row("overall", report["overall"]),
    ]
    for subset in sorted(report["by_subset"]):
        lines.append(row(subset, report["by_subset"][subset]))
    lines.extend(
        [
            "",
            "说明:",
            "- 官方 DCASE Task 3 的 `DOA error (deg)` 定义是在匹配成功的 true-positive 事件上计算的。",
            "- 在我们这个筛过的单声源子集上，由于每一帧只有一个活跃声源、也没有单独的检测分支，这里的 `DOA error (deg)` 近似为 active frame 上的 folded azimuth 平均绝对误差。",
            "- DCASE 标签的方位角被折叠到前方视野，因此这里以 folded azimuth 作为主要定位观察量。",
            "- 由于 DCASE2025 Task3 不提供 elevation 标注，`Horizontal Assumption RMSAE` 将目标 elevation 固定为 `0 deg`。",
            "- 这是一项基于 2 通道 M/S stereo 简单双耳间距代理前端的迁移测试，不等价于 LOCATA `benchmark2` 阵列评测。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(output_path: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(build_markdown(report), encoding="utf-8")
    return output_path, markdown_path


def evaluate_model_on_prepared(
    *,
    model: torch.nn.Module,
    preprocessor: DualFeatureIcoPreprocessor,
    prepared: list[PreparedDcaseClip],
    device: torch.device,
    batch_size: int,
    exclude_initial_windows: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    total = len(prepared)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        reports.extend(
            evaluate_prepared_batch(
                model=model,
                preprocessor=preprocessor,
                prepared=prepared[start:stop],
                device=device,
                exclude_initial_windows=exclude_initial_windows,
            )
        )
        if progress_callback is not None:
            progress_callback(stop, total)
    return reports


def masked_coordinate_loss(
    *,
    coords: torch.Tensor,
    doa_batch: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    target_cart = sph2cart(doa_batch[:, 0, ...].contiguous())
    valid_pred = coords[active_mask]
    valid_target = target_cart[active_mask]
    if valid_pred.numel() == 0:
        return coords.sum() * 0.0
    return torch.nn.functional.mse_loss(valid_pred, valid_target)


def masked_azimuth_sincos_loss(
    *,
    pred_sincos: torch.Tensor,
    doa_batch: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    phi = doa_batch[:, 0, :, 1].contiguous()
    target = torch.stack((torch.sin(phi), torch.cos(phi)), dim=-1)
    valid_pred = pred_sincos[active_mask]
    valid_target = target[active_mask]
    if valid_pred.numel() == 0:
        return pred_sincos.sum() * 0.0
    return torch.nn.functional.mse_loss(valid_pred, valid_target)


def masked_sincos_loss(
    *,
    pred_sincos: torch.Tensor,
    target_sincos: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    valid_pred = pred_sincos[active_mask]
    valid_target = target_sincos[active_mask]
    if valid_pred.numel() == 0:
        return pred_sincos.sum() * 0.0
    return torch.nn.functional.mse_loss(valid_pred, valid_target)
