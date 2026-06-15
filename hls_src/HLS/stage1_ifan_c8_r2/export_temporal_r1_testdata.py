from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DATASET_NAMES = ("fusion0", "fusion1", "fusion2", "fusion3", "final")
DEFAULT_STAGE1_DIR = Path("hls_testdata/stage1_ifan_c8_r2/scene_1_t6")
DEFAULT_CONVICO_DIR = Path("hls_testdata/layer2-5_c8_t6")
DEFAULT_OUTPUT_DIR = Path("hls_testdata/temporal_r1_c8_t6")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def array_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    finite = np.isfinite(arr) if arr.dtype.kind in "fc" else np.ones(arr.shape, dtype=bool)
    payload: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "finite": bool(finite.all()),
    }
    if arr.size and arr.dtype.kind in "fciu":
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


def save_txt(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%d" if np.issubdtype(array.dtype, np.integer) else "%.9e"
    np.savetxt(path, array.reshape(-1), fmt=fmt, header=f"Shape: {tuple(array.shape)}")


def save_npy_txt(base_path: Path, array: np.ndarray) -> dict[str, str]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(base_path.with_suffix(".npy"), array)
    save_txt(base_path.with_suffix(".txt"), array)
    return {"npy": base_path.with_suffix(".npy").name, "txt": base_path.with_suffix(".txt").name}


def max_abs_diff(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.shape != rhs.shape:
        raise RuntimeError(f"Shape mismatch: {lhs.shape} vs {rhs.shape}")
    if lhs.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))))


def rmse(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.shape != rhs.shape:
        raise RuntimeError(f"Shape mismatch: {lhs.shape} vs {rhs.shape}")
    if lhs.size == 0:
        return 0.0
    diff = lhs.astype(np.float64) - rhs.astype(np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def temporal_conv1d_r1(
    input_tensor: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    t_steps, channels, regions, charts, height, width = input_tensor.shape
    kernel = weight.shape[2]
    output = np.zeros_like(input_tensor, dtype=np.float32)

    for t in range(t_steps):
        for co in range(channels):
            acc = np.full((regions, charts, height, width), bias[co], dtype=np.float32)
            for ci in range(channels):
                for k in range(kernel):
                    src_t = t - (kernel - 1) + k
                    if src_t >= 0:
                        acc += input_tensor[src_t, ci] * weight[co, ci, k]
            output[t, co] = acc
    return np.ascontiguousarray(output)


def lnorm_r1(
    input_tensor: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    mean = input_tensor.mean(axis=(1, 2), keepdims=True)
    var = ((input_tensor - mean) ** 2).mean(axis=(1, 2), keepdims=True)
    inv_std = 1.0 / np.sqrt(var + 1.0e-5)
    gamma_view = gamma.reshape(1, -1, 1, 1, 1, 1)
    beta_view = beta.reshape(1, -1, 1, 1, 1, 1)
    output = (input_tensor - mean) * inv_std
    output = output * gamma_view + beta_view
    return np.ascontiguousarray(output.astype(np.float32))


def collect_temporal_datasets(
    stage1_dir: Path,
    convico_dir: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    weights = np.load(stage1_dir / "stage1_weights.npz")
    debug = np.load(stage1_dir / "stage1_debug_tensors.npz")

    datasets: dict[str, dict[str, np.ndarray]] = {}
    manifest: dict[str, Any] = {}

    for index, dataset_name in enumerate(DATASET_NAMES):
        conv_output = np.load(convico_dir / dataset_name / "output.npy").astype(np.float32)
        temporal_input = np.ascontiguousarray(np.maximum(conv_output, 0.0).astype(np.float32))

        if dataset_name == "final":
            temporal_weight = np.ascontiguousarray(weights["final_temporal_w"].astype(np.float32))
            temporal_bias = np.ascontiguousarray(weights["final_temporal_b"].astype(np.float32))
            gamma = np.ascontiguousarray(weights["norm_gamma"][5].astype(np.float32))
            beta = np.ascontiguousarray(weights["norm_beta"][5].astype(np.float32))
            expected_block_output = np.ascontiguousarray(debug["final_head_logits"].astype(np.float32))
        else:
            temporal_weight = np.ascontiguousarray(weights["fusion_temporal_w"][index].astype(np.float32))
            temporal_bias = np.ascontiguousarray(weights["fusion_temporal_b"][index].astype(np.float32))
            gamma = np.ascontiguousarray(weights["norm_gamma"][1 + index].astype(np.float32))
            beta = np.ascontiguousarray(weights["norm_beta"][1 + index].astype(np.float32))
            expected_block_output = np.ascontiguousarray(debug[f"fusion_head_blocks_{index}"].astype(np.float32))

        temporal_output = temporal_conv1d_r1(temporal_input, temporal_weight, temporal_bias)
        replay_output = lnorm_r1(temporal_output, gamma, beta)
        if dataset_name != "final":
            replay_output = np.ascontiguousarray(np.maximum(replay_output, 0.0).astype(np.float32))

        datasets[dataset_name] = {
            "input": temporal_input,
            "weight": temporal_weight,
            "bias": temporal_bias,
            "output": temporal_output,
        }
        manifest[dataset_name] = {
            "convico_dataset": relpath(convico_dir / dataset_name, repo_root()),
            "input_node": f"relu(layer2-5_c8_t6/{dataset_name}/output)",
            "temporal_output_node": f"{dataset_name}_temporal_pre_norm",
            "block_output_node": "debug['final_head_logits']" if dataset_name == "final" else f"debug['fusion_head_blocks'][{index}]",
            "post_block_replay_max_abs_diff": max_abs_diff(replay_output, expected_block_output),
            "post_block_replay_rmse": rmse(replay_output, expected_block_output),
        }
    return datasets, manifest


def validate_temporal_datasets(datasets: dict[str, dict[str, np.ndarray]], manifest: dict[str, Any]) -> None:
    expected_shapes = {
        "input": (6, 8, 6, 5, 2, 4),
        "weight": (8, 8, 5),
        "bias": (8,),
        "output": (6, 8, 6, 5, 2, 4),
    }
    for dataset_name in DATASET_NAMES:
        if dataset_name not in datasets:
            raise RuntimeError(f"Missing dataset {dataset_name}")
        payload = datasets[dataset_name]
        for field, shape in expected_shapes.items():
            if tuple(payload[field].shape) != shape:
                raise RuntimeError(f"{dataset_name}.{field} shape mismatch: {payload[field].shape} != {shape}")
            if not np.isfinite(payload[field]).all():
                raise RuntimeError(f"{dataset_name}.{field} contains non-finite values")
        if manifest[dataset_name]["post_block_replay_max_abs_diff"] > 1e-5:
            raise RuntimeError(
                f"{dataset_name} replay mismatch too large: {manifest[dataset_name]['post_block_replay_max_abs_diff']}"
            )


def save_temporal_datasets(
    output_dir: Path,
    datasets: dict[str, dict[str, np.ndarray]],
    dataset_manifest: dict[str, Any],
    *,
    root: Path,
    stage1_dir: Path,
    convico_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root_manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_stage1_bundle": relpath(stage1_dir, root),
        "source_convico_bundle": relpath(convico_dir, root),
        "datasets": {},
        "notes": [
            "Each dataset isolates the causal TemporalConv1d block used after R1 ConvIco in IFAN C8 R2.",
            "input is ReLU(ConvIco output) with shape [T, C, R, charts, H, W].",
            "output is the raw temporal Conv1d result before LNormIco.",
        ],
    }

    for dataset_name in DATASET_NAMES:
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        payload = datasets[dataset_name]
        files = {
            "input": save_npy_txt(dataset_dir / "input", payload["input"]),
            "weight": save_npy_txt(dataset_dir / "weight", payload["weight"]),
            "bias": save_npy_txt(dataset_dir / "bias", payload["bias"]),
            "output": save_npy_txt(dataset_dir / "output", payload["output"]),
        }
        manifest = {
            "dataset": dataset_name,
            "kind": "temporal_r1_real_golden",
            "layout": {
                "input": "[T=6, C=8, R=6, charts=5, H=2, W=4]",
                "weight": "[Cout=8, Cin=8, K=5]",
                "bias": "[Cout=8]",
                "output": "[T=6, C=8, R=6, charts=5, H=2, W=4]",
            },
            "source": dataset_manifest[dataset_name],
            "files": files,
            "stats": {name: array_stats(array) for name, array in payload.items()},
        }
        with (dataset_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")

        root_manifest["datasets"][dataset_name] = {
            "path": relpath(dataset_dir, root),
            "source": manifest["source"],
            "stats": {
                "input": manifest["stats"]["input"],
                "output": manifest["stats"]["output"],
            },
        }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(root_manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export IFAN C8 R2 temporal_r1 golden data.")
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--convico-dir", type=Path, default=DEFAULT_CONVICO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = repo_root()
    stage1_dir = resolve_path(args.stage1_dir, root)
    convico_dir = resolve_path(args.convico_dir, root)
    output_dir = resolve_path(args.output_dir, root)

    if not stage1_dir.exists():
        raise FileNotFoundError(stage1_dir)
    if not convico_dir.exists():
        raise FileNotFoundError(convico_dir)

    datasets, dataset_manifest = collect_temporal_datasets(stage1_dir, convico_dir)
    validate_temporal_datasets(datasets, dataset_manifest)
    save_temporal_datasets(
        output_dir,
        datasets,
        dataset_manifest,
        root=root,
        stage1_dir=stage1_dir,
        convico_dir=convico_dir,
    )

    summary = {
        "export": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": relpath(Path(__file__), root),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "source": {
            "stage1_dir": relpath(stage1_dir, root),
            "stage1_weights_sha256": file_sha256(stage1_dir / "stage1_weights.npz"),
            "stage1_debug_sha256": file_sha256(stage1_dir / "stage1_debug_tensors.npz"),
            "convico_dir": relpath(convico_dir, root),
        },
        "datasets": dataset_manifest,
    }
    with (output_dir / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("Exported temporal_r1 real datasets")
    print(f"Output: {relpath(output_dir, root)}")
    for dataset_name in DATASET_NAMES:
        print(
            f"{dataset_name}: replay_max={dataset_manifest[dataset_name]['post_block_replay_max_abs_diff']:.6g} "
            f"replay_rmse={dataset_manifest[dataset_name]['post_block_replay_rmse']:.6g}"
        )


if __name__ == "__main__":
    main()
