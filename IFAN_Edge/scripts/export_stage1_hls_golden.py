from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import platform
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_CHECKPOINT = Path(
    "IFAN_Edge/outputs/stage3/"
    "ifan_stage3_long80_c8_r2_maba_pre_readout_20260518_215408/"
    "checkpoints/best_rmsae.pt"
)
DEFAULT_INPUT = Path("IFAN_Edge/outputs/stage1_features/scene_1/dual_maps.npy")
DEFAULT_OUTPUT_DIR = Path("hls_testdata/stage1_ifan_c8_r2/scene_1_t6")
DEFAULT_LAYER2_5_OUTPUT_DIR = Path("hls_testdata/layer2-5_c8_t6")

STAGE1_INPUT_SHAPE = (2, 6, 5, 4, 8)
STAGE1_GOLDEN_SHAPE = (6, 8, 6, 5, 2, 4)
FEATURE_MABA_POSITION_SHAPE = (240, 6, 8)
FEATURE_MABA_LATENT_SHAPE = (240, 6, 16)
FEATURE_MABA_STATE_SHAPE = (240, 6, 8)
LAYER2_5_DATASET_NAMES = ("fusion0", "fusion1", "fusion2", "fusion3", "final")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_ifan_classes(root: Path):
    """Load IFAN model code without importing IFAN_Edge.__init__.

    The package __init__ imports the feature frontend, which can initialize gpuRIR.
    For checkpoint export we only need the model classes and the lightweight
    bridge objects used by placeholders.py.
    """

    for path in (root / "icoCNN-master", root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    import acousticTrackingModules as at_modules
    import icoCNN

    pkg = types.ModuleType("ifan_edge")
    pkg.__path__ = [str(root / "IFAN_Edge" / "ifan_edge")]
    sys.modules["ifan_edge"] = pkg

    models_pkg = types.ModuleType("ifan_edge.models")
    models_pkg.__path__ = [str(root / "IFAN_Edge" / "ifan_edge" / "models")]
    sys.modules["ifan_edge.models"] = models_pkg

    bridges = types.ModuleType("ifan_edge.bridges")
    bridges.at_modules = at_modules
    bridges.icoCNN = icoCNN
    sys.modules["ifan_edge.bridges"] = bridges

    models_dir = root / "IFAN_Edge" / "ifan_edge" / "models"
    map_maba = import_module_from_path("ifan_edge.models.map_maba", models_dir / "map_maba.py")
    placeholders = import_module_from_path("ifan_edge.models.placeholders", models_dir / "placeholders.py")
    return placeholders.IFANModel, placeholders.IFANModelConfig, map_maba.MapMABATemporalConfig


def dataclass_from_mapping(cls, payload: dict[str, Any]):
    field_names = {field.name for field in dataclasses.fields(cls)}
    filtered = {key: value for key, value in payload.items() if key in field_names}
    return cls(**filtered)


def load_model(checkpoint_path: Path, root: Path):
    IFANModel, IFANModelConfig, _ = load_ifan_classes(root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config_payload = checkpoint.get("model_config") or checkpoint.get("training_config")
    if not isinstance(config_payload, dict):
        raise RuntimeError("Checkpoint does not contain model_config or training_config.")

    config = dataclass_from_mapping(IFANModelConfig, config_payload)
    if config.r != 2 or config.branch_channels != 8:
        raise RuntimeError(f"Expected IFAN C8 R2 config, got r={config.r}, branch_channels={config.branch_channels}")
    if config.temporal_conv_variant != "standard_1d":
        raise RuntimeError(f"HLS export expects standard_1d temporal conv, got {config.temporal_conv_variant!r}")

    model = IFANModel(config)
    state = checkpoint.get("model_state_dict")
    if state is None:
        raise RuntimeError("Checkpoint does not contain model_state_dict.")
    state = dict(state)
    model_state_keys = set(model.state_dict().keys())
    load_notes: list[str] = []
    if config.map_refiner == "maba" and config.map_refiner_position == "pre_readout":
        legacy_prefix = "map_refiner."
        current_prefix = "feature_refiner."
        legacy_keys = [key for key in state if key.startswith(legacy_prefix)]
        current_keys = [key for key in state if key.startswith(current_prefix)]
        if legacy_keys and not current_keys:
            for key in legacy_keys:
                state[current_prefix + key[len(legacy_prefix) :]] = state.pop(key)
            load_notes.append(
                "Remapped legacy map_refiner.* checkpoint keys to feature_refiner.* for pre_readout MABA."
            )
    legacy_branch_norm_keys = [
        "phat_branch.residual.norm.weight",
        "phat_branch.residual.norm.bias",
        "aux_branch.residual.norm.weight",
        "aux_branch.residual.norm.bias",
    ]
    stale_branch_norm_keys = [key for key in legacy_branch_norm_keys if key in state and key not in model_state_keys]
    if stale_branch_norm_keys:
        for key in stale_branch_norm_keys:
            state.pop(key, None)
        load_notes.append(
            "Dropped stale frontend residual norm checkpoint keys no longer present in IFANModel: "
            + ", ".join(stale_branch_norm_keys)
        )
    model.load_state_dict(state, strict=True)
    model.eval()
    checkpoint["_stage1_export_load_notes"] = load_notes
    return model, config, checkpoint


def tensor_to_numpy(tensor: torch.Tensor, *, drop_batch: bool = False) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if drop_batch and array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    if array.dtype.kind == "f":
        return np.ascontiguousarray(array.astype(np.float32))
    return np.ascontiguousarray(array)


def as_float_array(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float32))


def save_txt(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%d" if np.issubdtype(array.dtype, np.integer) else "%.9e"
    np.savetxt(path, array.reshape(-1), fmt=fmt, header=f"Shape: {tuple(array.shape)}")


def save_npy_txt(base_path: Path, array: np.ndarray, *, write_text: bool = True) -> dict[str, str]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    npy_path = base_path.with_suffix(".npy")
    np.save(npy_path, array)
    files = {"npy": npy_path.name if npy_path.parent == base_path.parent else str(npy_path)}
    if write_text:
        txt_path = base_path.with_suffix(".txt")
        save_txt(txt_path, array)
        files["txt"] = txt_path.name if txt_path.parent == base_path.parent else str(txt_path)
    return files


def array_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    finite = np.isfinite(arr) if arr.dtype.kind in "fc" else np.ones(arr.shape, dtype=bool)
    payload: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "finite": bool(finite.all()),
    }
    if arr.size:
        if arr.dtype.kind in "fciu":
            numeric = arr.astype(np.float64, copy=False)
            payload.update(
                {
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                    "mean": float(numeric.mean()),
                    "std": float(numeric.std()),
                }
            )
    return payload


def collect_debug_arrays(debug: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if torch.is_tensor(value):
            arrays[prefix] = tensor_to_numpy(value, drop_batch=True)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(f"{prefix}_{index}", item)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(f"{prefix}_{key}", item)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            scalars[prefix] = value
        else:
            scalars[prefix] = repr(value)

    for key, value in debug.items():
        visit(key, value)
    return arrays, scalars


def norm_params(norm_module) -> tuple[np.ndarray, np.ndarray]:
    return as_float_array(norm_module.weight).reshape(-1), as_float_array(norm_module.bias).reshape(-1)


def temporal_params(temporal_module) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(temporal_module, "conv"):
        raise RuntimeError(f"Unsupported temporal module for HLS export: {type(temporal_module)!r}")
    conv = temporal_module.conv
    return as_float_array(conv.weight), as_float_array(conv.bias)


def collect_weights(model) -> dict[str, np.ndarray]:
    norm_gamma = np.ones((16, 8), dtype=np.float32)
    norm_beta = np.zeros((16, 8), dtype=np.float32)

    norm_slots = [
        model.shared_attention.norm,
        *[block.norm for block in model.fusion_blocks],
        model.final_block.norm,
    ]
    for slot, norm in enumerate(norm_slots):
        gamma, beta = norm_params(norm)
        norm_gamma[slot, : gamma.shape[0]] = gamma
        norm_beta[slot, : beta.shape[0]] = beta

    fusion_temporal_w = []
    fusion_temporal_b = []
    for block in model.fusion_blocks:
        weight, bias = temporal_params(block.temporal)
        fusion_temporal_w.append(weight)
        fusion_temporal_b.append(bias)

    final_temporal_w, final_temporal_b = temporal_params(model.final_block.temporal)

    return {
        "phat_stem_w": as_float_array(model.phat_branch.stem.weight),
        "phat_stem_b": as_float_array(model.phat_branch.stem.bias),
        "lms_stem_w": as_float_array(model.aux_branch.stem.weight),
        "lms_stem_b": as_float_array(model.aux_branch.stem.bias),
        "phat_res_w": np.stack(
            [
                as_float_array(model.phat_branch.residual.conv1.weight),
                as_float_array(model.phat_branch.residual.conv2.weight),
            ],
            axis=0,
        ),
        "phat_res_b": np.stack(
            [
                as_float_array(model.phat_branch.residual.conv1.bias),
                as_float_array(model.phat_branch.residual.conv2.bias),
            ],
            axis=0,
        ),
        "lms_res_w": np.stack(
            [
                as_float_array(model.aux_branch.residual.conv1.weight),
                as_float_array(model.aux_branch.residual.conv2.weight),
            ],
            axis=0,
        ),
        "lms_res_b": np.stack(
            [
                as_float_array(model.aux_branch.residual.conv1.bias),
                as_float_array(model.aux_branch.residual.conv2.bias),
            ],
            axis=0,
        ),
        "attn_w": np.stack(
            [
                as_float_array(model.shared_attention.conv1.weight),
                as_float_array(model.shared_attention.conv2.weight),
            ],
            axis=0,
        ),
        "attn_b": np.stack(
            [
                as_float_array(model.shared_attention.conv1.bias),
                as_float_array(model.shared_attention.conv2.bias),
            ],
            axis=0,
        ),
        "fusion_w": np.stack([as_float_array(block.conv.weight) for block in model.fusion_blocks], axis=0),
        "fusion_b": np.stack([as_float_array(block.conv.bias) for block in model.fusion_blocks], axis=0),
        "fusion_temporal_w": np.stack(fusion_temporal_w, axis=0),
        "fusion_temporal_b": np.stack(fusion_temporal_b, axis=0),
        "final_w": as_float_array(model.final_block.conv.weight),
        "final_b": as_float_array(model.final_block.conv.bias),
        "final_temporal_w": final_temporal_w,
        "final_temporal_b": final_temporal_b,
        "norm_gamma": norm_gamma,
        "norm_beta": norm_beta,
    }


def collect_geometry(model) -> dict[str, np.ndarray]:
    geometry = {
        "reorder_r2_stem": tensor_to_numpy(model.phat_branch.stem.padding.reorder_idx).astype(np.int64),
        "reorder_r2_main": tensor_to_numpy(model.phat_branch.residual.conv1.padding.reorder_idx).astype(np.int64),
        "reorder_r1": tensor_to_numpy(model.fusion_blocks[0].conv.padding.reorder_idx).astype(np.int64),
        "kernel_idx_stem": tensor_to_numpy(model.phat_branch.stem.kernel_expansion_idx).astype(np.int64),
        "kernel_idx_main": tensor_to_numpy(model.phat_branch.residual.conv1.kernel_expansion_idx).astype(np.int64),
    }
    if model.pre_fusion_pool is not None:
        geometry["pool_r2_to_r1_neighbors"] = tensor_to_numpy(model.pre_fusion_pool.neighbors).astype(np.int64)
    return geometry


def collect_feature_maba_weights(refiner) -> dict[str, np.ndarray]:
    if refiner is None:
        return {}

    weights = {
        "in_proj_weight": as_float_array(refiner.in_proj.weight),
        "in_proj_bias": as_float_array(refiner.in_proj.bias),
        "dw_conv_weight": as_float_array(refiner.dw_conv.weight),
        "dw_conv_bias": as_float_array(refiner.dw_conv.bias),
        "mix_norm_weight": as_float_array(refiner.mix_norm.weight),
        "mix_norm_bias": as_float_array(refiner.mix_norm.bias),
        "state_proj_weight": as_float_array(refiner.state_proj.weight),
        "state_proj_bias": as_float_array(refiner.state_proj.bias),
        "state_back_weight": as_float_array(refiner.state_back.weight),
        "state_back_bias": as_float_array(refiner.state_back.bias),
        "out_proj_weight": as_float_array(refiner.out_proj.weight),
        "out_proj_bias": as_float_array(refiner.out_proj.bias),
    }
    if getattr(refiner, "alpha_logit", None) is not None:
        weights["alpha_logit"] = as_float_array(refiner.alpha_logit)
    return weights


def feature_maba_position_index(*, regions: int, charts: int, height: int, width: int) -> np.ndarray:
    index = np.zeros((regions * charts * height * width, 4), dtype=np.int64)
    flat = 0
    for r in range(regions):
        for chart in range(charts):
            for h in range(height):
                for w in range(width):
                    index[flat] = (r, chart, h, w)
                    flat += 1
    return index


def collect_feature_maba_tensors(refiner, x: torch.Tensor) -> dict[str, np.ndarray]:
    if refiner is None:
        return {}
    if x.dim() != 7:
        raise RuntimeError(f"FeatureMABA expects [B,T,C,R,charts,H,W], got {tuple(x.shape)}")

    bsz, tlen, channels, regions, charts, height, width = x.shape
    if channels != refiner.channels:
        raise RuntimeError(f"FeatureMABA channel mismatch: {channels} != {refiner.channels}")

    tensors: dict[str, torch.Tensor] = {}
    tensors["input"] = x
    tensors["position_index"] = torch.from_numpy(
        feature_maba_position_index(regions=regions, charts=charts, height=height, width=width)
    )

    z_positions = x.permute(0, 3, 4, 5, 6, 1, 2).reshape(-1, tlen, channels)
    tensors["input_positions"] = z_positions

    z_in_proj = refiner.in_proj(z_positions)
    tensors["in_proj_out"] = z_in_proj

    z_t = z_in_proj.transpose(1, 2)
    tensors["dw_conv_input"] = z_t
    if refiner.conv_kernel > 1:
        z_t_padded = F.pad(z_t, (refiner.conv_kernel - 1, 0))
    else:
        z_t_padded = z_t
    tensors["dw_conv_input_padded"] = z_t_padded

    z_conv_raw = refiner.dw_conv(z_t_padded)
    tensors["dw_conv_out_raw"] = z_conv_raw
    z_conv_sliced = z_conv_raw[..., :tlen]
    tensors["dw_conv_out_sliced"] = z_conv_sliced
    z_conv = z_conv_sliced.transpose(1, 2)
    tensors["dw_conv_out"] = z_conv

    z_mix_pre_norm = z_in_proj + z_conv
    tensors["mix_pre_norm"] = z_mix_pre_norm
    z_mixed = refiner.mix_norm(z_mix_pre_norm)
    tensors["mix_norm_out"] = z_mixed
    z_after_dropout = refiner.dropout(z_mixed)
    tensors["mix_dropout_out"] = z_after_dropout

    state_input = refiner.state_proj(z_after_dropout)
    tensors["state_input"] = state_input
    if refiner.use_gate:
        q, gate = state_input.chunk(2, dim=-1)
        alpha = torch.sigmoid(gate)
        tensors["q"] = q
        tensors["gate"] = gate
        tensors["alpha"] = alpha
    else:
        q = state_input
        alpha = torch.sigmoid(refiner.alpha_logit).view(1, 1, -1).expand_as(q)
        tensors["q"] = q
        tensors["alpha"] = alpha

    if refiner.use_state:
        h = torch.zeros(z_after_dropout.shape[0], refiner.state_dim, dtype=q.dtype, device=q.device)
        h_seq = []
        for t in range(tlen):
            a_t = alpha[:, t, :]
            q_t = q[:, t, :]
            h = a_t * h + (1.0 - a_t) * q_t
            h_seq.append(h)
        state_sequence = torch.stack(h_seq, dim=1)
    else:
        state_sequence = q
    tensors["state_sequence"] = state_sequence

    state_back_out = refiner.state_back(state_sequence)
    tensors["state_back_out"] = state_back_out
    refined_pre_dropout = z_after_dropout + state_back_out
    tensors["refined_pre_dropout"] = refined_pre_dropout
    refined_latent = refiner.dropout(refined_pre_dropout)
    tensors["refined_latent"] = refined_latent

    delta_flat = refiner.out_proj(refined_latent)
    tensors["delta_flat"] = delta_flat
    delta_regions = delta_flat.reshape(bsz, regions, charts, height, width, tlen, channels)
    tensors["delta_regions"] = delta_regions
    delta = delta_regions.permute(0, 5, 6, 1, 2, 3, 4).contiguous()
    tensors["delta"] = delta
    tensors["output"] = x + delta if refiner.use_residual else delta

    arrays: dict[str, np.ndarray] = {}
    for name, value in tensors.items():
        arrays[name] = tensor_to_numpy(value, drop_batch=name in {"input", "delta", "output"})
    return arrays


def collect_readout_weights(model) -> dict[str, np.ndarray]:
    return {
        "channel_readout_weight": as_float_array(model.channel_readout.proj.weight),
        "channel_readout_bias": as_float_array(model.channel_readout.proj.bias),
    }


def collect_post_maba_tensors(model, debug: dict[str, Any], coords: torch.Tensor) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for key in (
        "pre_readout_refined_logits",
        "channel_readout_logits",
        "post_final_pool_logits",
        "map_refined_logits",
        "softargmax_input",
    ):
        value = debug.get(key)
        if torch.is_tensor(value):
            tensors[key] = tensor_to_numpy(value, drop_batch=True)

    post_final_pool = debug.get("post_final_pool_logits")
    if torch.is_tensor(post_final_pool):
        region_input = post_final_pool.squeeze(2)
        region_max, region_argmax = region_input.max(dim=2)
        tensors["region_max_logits"] = tensor_to_numpy(region_max, drop_batch=True)
        tensors["region_argmax_idx"] = tensor_to_numpy(region_argmax, drop_batch=True).astype(np.int64)

    softargmax_input = debug.get("softargmax_input")
    if torch.is_tensor(softargmax_input):
        probs = torch.exp(softargmax_input - softargmax_input.max())
        probs = probs / (probs.sum(dim=tuple(range(probs.ndim - model.sam.n_dim_in, probs.ndim)), keepdim=True) + 1e-12)
        tensors["softargmax_prob"] = tensor_to_numpy(probs, drop_batch=True)
        indexes = model.sam.indexes
        tensors["softargmax_indexes"] = tensor_to_numpy(indexes).astype(np.float32)
        weighted_xyz = probs.unsqueeze(-model.sam.n_dim_in - 1) * indexes
        tensors["softargmax_weighted_xyz"] = tensor_to_numpy(weighted_xyz, drop_batch=True)

    tensors["clean_vertices_mask"] = tensor_to_numpy(model.clean_vertices.mask).astype(np.float32)
    tensors["coords"] = tensor_to_numpy(coords, drop_batch=True)
    return tensors


def feature_maba_config(refiner) -> dict[str, Any] | None:
    if refiner is None:
        return None
    return {
        "kind": "FeatureMABATemporalRefiner",
        "channels": int(refiner.channels),
        "d_model": int(refiner.d_model),
        "state_dim": int(refiner.state_dim),
        "conv_kernel": int(refiner.conv_kernel),
        "use_residual": bool(refiner.use_residual),
        "use_gate": bool(refiner.use_gate),
        "use_state": bool(refiner.use_state),
    }


def validate_bundle(
    stage1_input: np.ndarray,
    final_head_logits: np.ndarray,
    weights: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    maba_weights: dict[str, np.ndarray],
    maba_tensors: dict[str, np.ndarray],
    maba_expected_output: np.ndarray,
) -> None:
    expected_weights = {
        "phat_stem_w": (8, 1, 1, 7),
        "phat_stem_b": (8,),
        "lms_stem_w": (8, 1, 1, 7),
        "lms_stem_b": (8,),
        "phat_res_w": (2, 8, 8, 6, 7),
        "phat_res_b": (2, 8),
        "lms_res_w": (2, 8, 8, 6, 7),
        "lms_res_b": (2, 8),
        "attn_w": (2, 8, 8, 6, 7),
        "attn_b": (2, 8),
        "fusion_w": (4, 8, 8, 6, 7),
        "fusion_b": (4, 8),
        "fusion_temporal_w": (4, 8, 8, 5),
        "fusion_temporal_b": (4, 8),
        "final_w": (8, 8, 6, 7),
        "final_b": (8,),
        "final_temporal_w": (8, 8, 5),
        "final_temporal_b": (8,),
        "norm_gamma": (16, 8),
        "norm_beta": (16, 8),
    }
    expected_geometry = {
        "reorder_r2_stem": (1, 5, 6, 10),
        "reorder_r2_main": (6, 5, 6, 10),
        "reorder_r1": (6, 5, 4, 6),
        "kernel_idx_stem": (8, 6, 1, 1, 9, 4),
        "kernel_idx_main": (8, 6, 8, 6, 9, 4),
    }

    if tuple(stage1_input.shape) != STAGE1_INPUT_SHAPE:
        raise RuntimeError(f"stage1_input shape mismatch: {stage1_input.shape} != {STAGE1_INPUT_SHAPE}")
    if tuple(final_head_logits.shape) != STAGE1_GOLDEN_SHAPE:
        raise RuntimeError(f"final_head_logits shape mismatch: {final_head_logits.shape} != {STAGE1_GOLDEN_SHAPE}")
    if not np.isfinite(stage1_input).all() or not np.isfinite(final_head_logits).all():
        raise RuntimeError("Input or golden output contains non-finite values.")

    for name, shape in expected_weights.items():
        if name not in weights or tuple(weights[name].shape) != shape:
            got = None if name not in weights else tuple(weights[name].shape)
            raise RuntimeError(f"Weight {name} shape mismatch: {got} != {shape}")
    for name, shape in expected_geometry.items():
        if name not in geometry or tuple(geometry[name].shape) != shape:
            got = None if name not in geometry else tuple(geometry[name].shape)
            raise RuntimeError(f"Geometry {name} shape mismatch: {got} != {shape}")

    expected_maba_weights = {
        "in_proj_weight": (16, 8),
        "in_proj_bias": (16,),
        "dw_conv_weight": (16, 1, 3),
        "dw_conv_bias": (16,),
        "mix_norm_weight": (16,),
        "mix_norm_bias": (16,),
        "state_proj_weight": (16, 16),
        "state_proj_bias": (16,),
        "state_back_weight": (16, 8),
        "state_back_bias": (16,),
        "out_proj_weight": (8, 16),
        "out_proj_bias": (8,),
    }
    expected_maba_tensors = {
        "input": STAGE1_GOLDEN_SHAPE,
        "input_positions": FEATURE_MABA_POSITION_SHAPE,
        "in_proj_out": FEATURE_MABA_LATENT_SHAPE,
        "dw_conv_input": (240, 16, 6),
        "dw_conv_input_padded": (240, 16, 8),
        "dw_conv_out_raw": (240, 16, 6),
        "dw_conv_out_sliced": (240, 16, 6),
        "dw_conv_out": FEATURE_MABA_LATENT_SHAPE,
        "mix_pre_norm": FEATURE_MABA_LATENT_SHAPE,
        "mix_norm_out": FEATURE_MABA_LATENT_SHAPE,
        "mix_dropout_out": FEATURE_MABA_LATENT_SHAPE,
        "state_input": FEATURE_MABA_LATENT_SHAPE,
        "q": FEATURE_MABA_STATE_SHAPE,
        "gate": FEATURE_MABA_STATE_SHAPE,
        "alpha": FEATURE_MABA_STATE_SHAPE,
        "state_sequence": FEATURE_MABA_STATE_SHAPE,
        "state_back_out": FEATURE_MABA_LATENT_SHAPE,
        "refined_pre_dropout": FEATURE_MABA_LATENT_SHAPE,
        "refined_latent": FEATURE_MABA_LATENT_SHAPE,
        "delta_flat": FEATURE_MABA_POSITION_SHAPE,
        "delta_regions": (1, 6, 5, 2, 4, 6, 8),
        "delta": STAGE1_GOLDEN_SHAPE,
        "output": STAGE1_GOLDEN_SHAPE,
        "position_index": (240, 4),
    }
    for name, shape in expected_maba_weights.items():
        if name not in maba_weights or tuple(maba_weights[name].shape) != shape:
            got = None if name not in maba_weights else tuple(maba_weights[name].shape)
            raise RuntimeError(f"MABA weight {name} shape mismatch: {got} != {shape}")
    for name, shape in expected_maba_tensors.items():
        if name not in maba_tensors or tuple(maba_tensors[name].shape) != shape:
            got = None if name not in maba_tensors else tuple(maba_tensors[name].shape)
            raise RuntimeError(f"MABA tensor {name} shape mismatch: {got} != {shape}")
    if not np.isfinite(maba_tensors["output"]).all():
        raise RuntimeError("MABA output contains non-finite values.")
    max_diff = float(np.max(np.abs(maba_tensors["output"] - maba_expected_output)))
    if max_diff > 1e-6:
        raise RuntimeError(f"Manual FeatureMABA output mismatch: max_diff={max_diff}")


def save_array_group(group_dir: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    group_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    for name, array in sorted(arrays.items()):
        save_npy_txt(group_dir / name, array, write_text=True)
        result[name] = {
            "npy": str((group_dir / f"{name}.npy").name),
            "txt": str((group_dir / f"{name}.txt").name),
            "stats": array_stats(array),
        }
    return result


def max_abs_diff(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.shape != rhs.shape:
        raise RuntimeError(f"Cannot compare arrays with different shapes: {lhs.shape} vs {rhs.shape}")
    if lhs.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))))


def collect_layer2_5_convico_datasets(
    model,
    debug: dict[str, Any],
    weights: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    fusion_feature = debug.get("fusion_feature")
    fusion_head_blocks = debug.get("fusion_head_blocks")
    final_head_logits = debug.get("final_head_logits")

    if not torch.is_tensor(fusion_feature):
        raise RuntimeError("Model debug output does not contain fusion_feature tensor.")
    if not isinstance(fusion_head_blocks, list) or len(fusion_head_blocks) != len(model.fusion_blocks):
        raise RuntimeError(
            "Model debug output does not contain the expected fusion_head_blocks list for layer2-5 export."
        )
    if not torch.is_tensor(final_head_logits):
        raise RuntimeError("Model debug output does not contain final_head_logits tensor.")
    if "kernel_idx_main" not in geometry or "reorder_r1" not in geometry:
        raise RuntimeError("Missing kernel_idx_main or reorder_r1 geometry required for layer2-5 export.")

    kernel_idx_main = np.ascontiguousarray(geometry["kernel_idx_main"].astype(np.int64, copy=False))
    reorder_r1 = np.ascontiguousarray(geometry["reorder_r1"].astype(np.int64, copy=False))
    datasets: dict[str, dict[str, np.ndarray]] = {}
    manifest: dict[str, Any] = {}

    current = fusion_feature
    with torch.no_grad():
        for index, block in enumerate(model.fusion_blocks):
            dataset_name = f"fusion{index}"
            conv_output = block.conv(current)
            full_output = block(current)
            debug_output = fusion_head_blocks[index]
            if not torch.is_tensor(debug_output):
                raise RuntimeError(f"fusion_head_blocks[{index}] is not a tensor.")

            input_array = tensor_to_numpy(current, drop_batch=True)
            output_array = tensor_to_numpy(conv_output, drop_batch=True)
            full_output_array = tensor_to_numpy(full_output, drop_batch=True)
            debug_output_array = tensor_to_numpy(debug_output, drop_batch=True)
            replay_max_diff = max_abs_diff(full_output_array, debug_output_array)
            if replay_max_diff > 1e-6:
                raise RuntimeError(
                    f"Replay mismatch for {dataset_name}: max_diff={replay_max_diff} exceeds tolerance."
                )

            datasets[dataset_name] = {
                "input_rearranged": input_array,
                "weight": np.ascontiguousarray(weights["fusion_w"][index].astype(np.float32, copy=False)),
                "bias": np.ascontiguousarray(weights["fusion_b"][index].astype(np.float32, copy=False)),
                "kernel_expansion_idx": kernel_idx_main,
                "reorder_idx": reorder_r1,
                "output": output_array,
            }
            manifest[dataset_name] = {
                "input_node": "debug['fusion_feature']" if index == 0 else f"debug['fusion_head_blocks'][{index - 1}]",
                "conv_output_node": f"model.fusion_blocks[{index}].conv(input_node)",
                "next_stage_node": f"debug['fusion_head_blocks'][{index}]",
                "full_block_replay_max_abs_diff": replay_max_diff,
            }
            current = debug_output

        final_conv_output = model.final_block.conv(current)
        final_full_output = model.final_block(current)
        final_full_array = tensor_to_numpy(final_full_output, drop_batch=True)
        final_debug_array = tensor_to_numpy(final_head_logits, drop_batch=True)
        final_replay_max_diff = max_abs_diff(final_full_array, final_debug_array)
        if final_replay_max_diff > 1e-6:
            raise RuntimeError(
                f"Replay mismatch for final block: max_diff={final_replay_max_diff} exceeds tolerance."
            )

    datasets["final"] = {
        "input_rearranged": tensor_to_numpy(current, drop_batch=True),
        "weight": np.ascontiguousarray(weights["final_w"].astype(np.float32, copy=False)),
        "bias": np.ascontiguousarray(weights["final_b"].astype(np.float32, copy=False)),
        "kernel_expansion_idx": kernel_idx_main,
        "reorder_idx": reorder_r1,
        "output": tensor_to_numpy(final_conv_output, drop_batch=True),
    }
    manifest["final"] = {
        "input_node": "debug['fusion_head_blocks'][3]",
        "conv_output_node": "model.final_block.conv(input_node)",
        "next_stage_node": "debug['final_head_logits']",
        "full_block_replay_max_abs_diff": final_replay_max_diff,
    }
    return datasets, manifest


def validate_layer2_5_convico_datasets(datasets: dict[str, dict[str, np.ndarray]]) -> None:
    expected_shapes = {
        "input_rearranged": STAGE1_GOLDEN_SHAPE,
        "weight": (8, 8, 6, 7),
        "bias": (8,),
        "kernel_expansion_idx": (8, 6, 8, 6, 9, 4),
        "reorder_idx": (6, 5, 4, 6),
        "output": STAGE1_GOLDEN_SHAPE,
    }
    for dataset_name in LAYER2_5_DATASET_NAMES:
        if dataset_name not in datasets:
            raise RuntimeError(f"Missing layer2-5 dataset {dataset_name}.")
        payload = datasets[dataset_name]
        for field, shape in expected_shapes.items():
            if field not in payload or tuple(payload[field].shape) != shape:
                got = None if field not in payload else tuple(payload[field].shape)
                raise RuntimeError(f"{dataset_name}.{field} shape mismatch: {got} != {shape}")
            if payload[field].dtype.kind in "fc" and not np.isfinite(payload[field]).all():
                raise RuntimeError(f"{dataset_name}.{field} contains non-finite values.")


def save_layer2_5_convico_datasets(
    output_dir: Path,
    datasets: dict[str, dict[str, np.ndarray]],
    dataset_manifest: dict[str, Any],
    *,
    root: Path,
    stage1_output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    root_manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_stage1_bundle": relpath(stage1_output_dir, root),
        "datasets": {},
        "notes": [
            "Each subdirectory mirrors the file contract expected by hls_src/HLS/layer2-5/test_ico_conv_layer2_5.cpp.",
            "output.txt is the raw ConvIco output before the per-block ReLU/temporal Conv1d/LNorm path.",
            "input_rearranged.txt is the exact tensor consumed by each fusion/final ConvIco block inside IFAN C8 R2.",
        ],
    }

    for dataset_name in LAYER2_5_DATASET_NAMES:
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        payload = datasets[dataset_name]

        files = {
            "input_rearranged": save_npy_txt(dataset_dir / "input_rearranged", payload["input_rearranged"]),
            "weight": save_npy_txt(dataset_dir / "weight", payload["weight"]),
            "bias": save_npy_txt(dataset_dir / "bias", payload["bias"]),
            "kernel_expansion_idx": save_npy_txt(dataset_dir / "kernel_expansion_idx", payload["kernel_expansion_idx"]),
            "reorder_idx": save_npy_txt(dataset_dir / "reorder_idx", payload["reorder_idx"]),
            "output": save_npy_txt(dataset_dir / "output", payload["output"]),
        }

        dataset_stats = {name: array_stats(array) for name, array in payload.items()}
        manifest = {
            "dataset": dataset_name,
            "kind": "layer2-5_real_convico_golden",
            "layout": {
                "input_rearranged": "[T=6, C=8, R=6, charts=5, H=2, W=4]",
                "weight": "[Cout=8, Cin=8, Rin=6, K=7]",
                "bias": "[Cout=8]",
                "kernel_expansion_idx": "[Cout=8, Rout=6, Cin=8, Rin=6, K3x3=9, fields=4]",
                "reorder_idx": "[R=6, charts=5, H+2=4, W+2=6]",
                "output": "[T=6, C=8, R=6, charts=5, H=2, W=4]",
            },
            "source": {
                "stage1_bundle": relpath(stage1_output_dir, root),
                **dataset_manifest[dataset_name],
            },
            "files": files,
            "stats": dataset_stats,
        }
        with (dataset_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")

        root_manifest["datasets"][dataset_name] = {
            "path": relpath(dataset_dir, root),
            "source": manifest["source"],
            "stats": {
                "input_rearranged": dataset_stats["input_rearranged"],
                "output": dataset_stats["output"],
            },
        }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(root_manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return root_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export IFAN C8 R2 Stage-1 HLS golden data.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layer2-5-output-dir", type=Path, default=DEFAULT_LAYER2_5_OUTPUT_DIR)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = repo_root()
    checkpoint_path = resolve_path(args.checkpoint, root)
    input_path = resolve_path(args.input, root)
    output_dir = resolve_path(args.output_dir, root)
    layer2_5_output_dir = resolve_path(args.layer2_5_output_dir, root)

    torch.set_num_threads(max(1, int(args.torch_threads)))

    if args.frames != 6:
        raise RuntimeError("Current Stage-1 HLS project expects IFAN_STAGE1_T=6.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    model, config, checkpoint = load_model(checkpoint_path, root)

    dual_maps = np.load(input_path)
    if dual_maps.ndim == 6:
        if args.sample_index < 0 or args.sample_index >= dual_maps.shape[0]:
            raise IndexError(f"sample-index {args.sample_index} out of range for input shape {dual_maps.shape}")
        stage1_input = dual_maps[
            args.sample_index,
            :,
            args.start_frame : args.start_frame + args.frames,
            :,
            :,
            :,
        ]
    elif dual_maps.ndim == 5:
        stage1_input = dual_maps[:, args.start_frame : args.start_frame + args.frames, :, :, :]
    else:
        raise RuntimeError(f"Expected input [B,2,T,5,4,8] or [2,T,5,4,8], got {dual_maps.shape}")

    stage1_input = np.ascontiguousarray(stage1_input.astype(np.float32))
    model_input = torch.from_numpy(stage1_input[None, ...])

    with torch.no_grad():
        coords, debug = model(model_input, return_debug=True)

    if not isinstance(debug, dict) or "final_head_logits" not in debug:
        raise RuntimeError("Model debug output does not contain final_head_logits.")

    final_head_logits = tensor_to_numpy(debug["final_head_logits"], drop_batch=True)
    pre_readout_refined = tensor_to_numpy(debug["pre_readout_refined_logits"], drop_batch=True)
    coords_np = tensor_to_numpy(coords, drop_batch=True)
    debug_arrays, debug_scalars = collect_debug_arrays(debug)
    weights = collect_weights(model)
    geometry = collect_geometry(model)
    maba_weights = collect_feature_maba_weights(model.feature_refiner)
    maba_tensors = collect_feature_maba_tensors(model.feature_refiner, debug["final_head_logits"])
    readout_weights = collect_readout_weights(model)
    post_maba_tensors = collect_post_maba_tensors(model, debug, coords)
    layer2_5_datasets, layer2_5_dataset_manifest = collect_layer2_5_convico_datasets(
        model, debug, weights, geometry
    )

    validate_bundle(
        stage1_input,
        final_head_logits,
        weights,
        geometry,
        maba_weights,
        maba_tensors,
        pre_readout_refined,
    )
    validate_layer2_5_convico_datasets(layer2_5_datasets)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_npy_txt(output_dir / "stage1_input", stage1_input, write_text=True)
    save_npy_txt(output_dir / "final_head_logits", final_head_logits, write_text=True)
    save_npy_txt(output_dir / "pre_readout_refined_logits", pre_readout_refined, write_text=True)
    save_npy_txt(output_dir / "coords", coords_np, write_text=True)

    np.savez(output_dir / "stage1_weights.npz", **weights)
    np.savez(output_dir / "stage1_geometry.npz", **geometry)
    np.savez(output_dir / "stage1_debug_tensors.npz", **debug_arrays)
    np.savez(output_dir / "maba_weights.npz", **maba_weights)
    np.savez(output_dir / "maba_debug_tensors.npz", **maba_tensors)
    np.savez(output_dir / "readout_weights.npz", **readout_weights)
    np.savez(output_dir / "post_maba_tensors.npz", **post_maba_tensors)

    weight_files = save_array_group(output_dir / "weights", weights)
    geometry_files = save_array_group(output_dir / "geometry", geometry)
    maba_weight_files = save_array_group(output_dir / "maba" / "weights", maba_weights)
    maba_tensor_files = save_array_group(output_dir / "maba" / "tensors", maba_tensors)
    readout_weight_files = save_array_group(output_dir / "post_maba" / "weights", readout_weights)
    post_maba_tensor_files = save_array_group(output_dir / "post_maba" / "tensors", post_maba_tensors)
    layer2_5_manifest = save_layer2_5_convico_datasets(
        layer2_5_output_dir,
        layer2_5_datasets,
        layer2_5_dataset_manifest,
        root=root,
        stage1_output_dir=output_dir,
    )

    tensor_stats = {
        "stage1_input": array_stats(stage1_input),
        "final_head_logits": array_stats(final_head_logits),
        "pre_readout_refined_logits": array_stats(pre_readout_refined),
        "coords": array_stats(coords_np),
    }
    debug_stats = {name: array_stats(array) for name, array in sorted(debug_arrays.items())}
    maba_weight_stats = {name: array_stats(array) for name, array in sorted(maba_weights.items())}
    maba_tensor_stats = {name: array_stats(array) for name, array in sorted(maba_tensors.items())}
    post_maba_tensor_stats = {name: array_stats(array) for name, array in sorted(post_maba_tensors.items())}

    manifest = {
        "export": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": relpath(Path(__file__), root),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "torch_threads": int(args.torch_threads),
        },
        "source": {
            "checkpoint": relpath(checkpoint_path, root),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "input": relpath(input_path, root),
            "input_sha256": file_sha256(input_path),
            "input_original_shape": list(dual_maps.shape),
            "sample_index": int(args.sample_index),
            "start_frame": int(args.start_frame),
            "frames": int(args.frames),
            "input_source_kind": "precomputed_dual_maps",
        },
        "checkpoint": {
            "epoch": checkpoint.get("epoch"),
            "metrics": checkpoint.get("metrics"),
            "model_config": checkpoint.get("model_config"),
            "training_config": checkpoint.get("training_config"),
            "load_notes": checkpoint.get("_stage1_export_load_notes", []),
        },
        "stage1_contract": {
            "input_shape": list(STAGE1_INPUT_SHAPE),
            "golden_shape": list(STAGE1_GOLDEN_SHAPE),
            "input_layout": "[C=2, T=6, charts=5, H=4, W=8]",
            "golden_layout": "[T=6, C=8, R=6, charts=5, H=2, W=4]",
            "golden_node": "debug['final_head_logits']",
            "refined_node": "debug['pre_readout_refined_logits']",
        },
        "maba_contract": {
            "enabled": model.feature_refiner is not None,
            "config": feature_maba_config(model.feature_refiner),
            "input_node": "debug['final_head_logits']",
            "output_node": "debug['pre_readout_refined_logits']",
            "input_shape": list(STAGE1_GOLDEN_SHAPE),
            "output_shape": list(STAGE1_GOLDEN_SHAPE),
            "position_shape": list(FEATURE_MABA_POSITION_SHAPE),
            "latent_shape": list(FEATURE_MABA_LATENT_SHAPE),
            "state_shape": list(FEATURE_MABA_STATE_SHAPE),
            "position_layout": "[position=R*charts*H*W=240, T=6, C=8]",
            "feature_layout": "[T=6, C=8, R=6, charts=5, H=2, W=4]",
        },
        "post_maba_contract": {
            "channel_readout_logits_shape": list(post_maba_tensors["channel_readout_logits"].shape),
            "region_max_logits_shape": list(post_maba_tensors["region_max_logits"].shape),
            "region_argmax_idx_shape": list(post_maba_tensors["region_argmax_idx"].shape),
            "softargmax_input_shape": list(post_maba_tensors["softargmax_input"].shape),
            "softargmax_prob_shape": list(post_maba_tensors["softargmax_prob"].shape),
            "softargmax_indexes_shape": list(post_maba_tensors["softargmax_indexes"].shape),
            "coords_shape": list(coords_np.shape),
            "channel_readout_layout": "[T=6, 1, R=6, charts=5, H=2, W=4]",
            "region_max_layout": "[T=6, charts=5, H=2, W=4]",
            "softargmax_input_layout": "[T=6, charts=5, H=2, W=4]",
            "softargmax_indexes_layout": "[xyz=3, charts=5, H=2, W=4]",
            "coords_layout": "[T=6, xyz=3]",
        },
        "files": {
            "stage1_input": {"npy": "stage1_input.npy", "txt": "stage1_input.txt"},
            "final_head_logits": {"npy": "final_head_logits.npy", "txt": "final_head_logits.txt"},
            "pre_readout_refined_logits": {
                "npy": "pre_readout_refined_logits.npy",
                "txt": "pre_readout_refined_logits.txt",
            },
            "coords": {"npy": "coords.npy", "txt": "coords.txt"},
            "weights_npz": "stage1_weights.npz",
            "geometry_npz": "stage1_geometry.npz",
            "debug_tensors_npz": "stage1_debug_tensors.npz",
            "maba_weights_npz": "maba_weights.npz",
            "maba_debug_tensors_npz": "maba_debug_tensors.npz",
            "readout_weights_npz": "readout_weights.npz",
            "post_maba_tensors_npz": "post_maba_tensors.npz",
            "layer2_5_root_manifest": relpath(layer2_5_output_dir / "manifest.json", output_dir),
            "weights_flat": weight_files,
            "geometry_flat": geometry_files,
            "maba_weights_flat": maba_weight_files,
            "maba_tensors_flat": maba_tensor_files,
            "readout_weights_flat": readout_weight_files,
            "post_maba_tensors_flat": post_maba_tensor_files,
        },
        "layer2_5_contract": {
            "root_dir": relpath(layer2_5_output_dir, root),
            "datasets": layer2_5_manifest["datasets"],
            "kernel_kind": "R1 shared ConvIco only",
            "excludes": [
                "per-block ReLU",
                "temporal Conv1d",
                "LNormIco",
                "FeatureMABA",
                "channel readout and SoftArgMax path",
            ],
        },
        "tensor_stats": tensor_stats,
        "debug_tensor_stats": debug_stats,
        "maba_weight_stats": maba_weight_stats,
        "maba_tensor_stats": maba_tensor_stats,
        "post_maba_tensor_stats": post_maba_tensor_stats,
        "debug_scalars": debug_scalars,
        "notes": [
            "final_head_logits is the Stage-1 HLS golden target before FeatureMABA/channel readout.",
            "pre_readout_refined_logits is the FeatureMABA output and is exported as the next-stage MABA golden target.",
            "weights_flat and geometry_flat are flat text mirrors of the NPZ payloads for later C++/HLS testbench ingestion.",
            "maba/tensors contains a manual step-by-step FeatureMABA decomposition for later MABA HLS module checks.",
            "post_maba/tensors contains channel readout, region max, clean-vertices/softargmax input, and final coords references.",
        ],
    }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("Exported IFAN C8 R2 Stage-1 HLS data")
    print(f"Output: {relpath(output_dir, root)}")
    print(f"Input shape: {tuple(stage1_input.shape)}")
    print(f"Golden shape: {tuple(final_head_logits.shape)}")
    print(f"Final logits min/max: {final_head_logits.min():.6g} / {final_head_logits.max():.6g}")
    print(f"MABA output shape: {tuple(pre_readout_refined.shape)}")
    print(f"MABA output min/max: {pre_readout_refined.min():.6g} / {pre_readout_refined.max():.6g}")
    print(f"layer2-5 ConvIco real datasets: {relpath(layer2_5_output_dir, root)}")
    print(f"layer2-5 datasets: {', '.join(LAYER2_5_DATASET_NAMES)}")


if __name__ == "__main__":
    main()
