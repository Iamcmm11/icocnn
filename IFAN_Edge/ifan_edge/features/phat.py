from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..bridges import at_modules, icoCNN


def ensure_mic_tensor(mic_sig_batch) -> torch.Tensor:
    """Normalize microphone batches to [B, 1, T, N, K]."""

    if isinstance(mic_sig_batch, np.ndarray):
        mic_sig_batch = torch.from_numpy(mic_sig_batch.astype(np.float32))
    elif not isinstance(mic_sig_batch, torch.Tensor):
        raise TypeError(f"Unsupported microphone batch type: {type(mic_sig_batch)!r}")

    if mic_sig_batch.ndim == 4:
        mic_sig_batch = mic_sig_batch.unsqueeze(1)
    if mic_sig_batch.ndim != 5:
        raise ValueError(f"Expected [B, 1, T, N, K] or [B, T, N, K], got {tuple(mic_sig_batch.shape)}")
    return mic_sig_batch


def _lag_to_gcc_index(lag: np.ndarray, tau_max: int) -> tuple[np.ndarray, np.ndarray]:
    lag = lag.astype(np.int64, copy=False)
    valid = (lag >= -tau_max) & (lag <= tau_max)
    lag = np.clip(lag, -tau_max, tau_max)
    count = 2 * tau_max + 1
    index = np.where(lag >= 0, lag, lag + count)
    return index.astype(np.int64, copy=False), valid


def _normalize_weights(weights: np.ndarray, valid: np.ndarray) -> np.ndarray:
    weights = weights.astype(np.float32, copy=False) * valid.astype(np.float32, copy=False)
    denom = weights.sum(axis=-1, keepdims=True)
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    return weights / denom


def _normalize_maps(maps: torch.Tensor) -> torch.Tensor:
    original_shape = maps.shape
    maps = maps + 1e-12
    maps = maps.reshape(maps.shape[:-3] + (-1,))
    maps = maps / maps.amax(dim=-1, keepdim=True).clamp_min(1e-12)
    return maps.reshape(original_shape)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _named_buffer_stats(module: nn.Module) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for name, tensor in module.named_buffers():
        stats[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes": _tensor_bytes(tensor),
        }
    return stats


@dataclass(frozen=True)
class PHATVariantMetadata:
    srp_variant: str
    pair_count: int
    full_pair_count: int
    diagonal_pair_count: int
    interpolation_taps: int
    grid_shape: tuple[int, int, int]
    unique_pairs_only: bool
    symmetric_pair_factor: float
    buffer_stats: dict[str, dict[str, Any]]
    complexity_proxy: dict[str, int | float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "srp_variant": self.srp_variant,
            "pair_count": int(self.pair_count),
            "full_pair_count": int(self.full_pair_count),
            "diagonal_pair_count": int(self.diagonal_pair_count),
            "interpolation_taps": int(self.interpolation_taps),
            "grid_shape": list(self.grid_shape),
            "unique_pairs_only": bool(self.unique_pairs_only),
            "symmetric_pair_factor": float(self.symmetric_pair_factor),
            "cache_table_bytes": int(sum(row["bytes"] for row in self.buffer_stats.values())),
            "cache_tables": self.buffer_stats,
            "complexity_proxy": self.complexity_proxy,
        }


class _InterpolatedSRPBase(nn.Module):
    def __init__(
        self,
        *,
        N: int,
        K: int,
        r: int,
        rn: np.ndarray,
        fs: int,
        c: float,
        tau_max: int,
        normalize: bool,
        unique_pairs_only: bool,
        sinc_half_width: int,
    ):
        super().__init__()
        self.N = int(N)
        self.K = int(K)
        self.r = int(r)
        self.fs = float(fs)
        self.c = float(c)
        self.tau_max = int(tau_max)
        self.normalize = bool(normalize)
        self.unique_pairs_only = bool(unique_pairs_only)
        self.sinc_half_width = int(sinc_half_width)

        grid = torch.from_numpy(icoCNN.icosahedral_grid_coordinates(r)).float()
        grid_shape = tuple(int(value) for value in grid.shape[:-1])
        grid_flat = grid.reshape(-1, 3).numpy()
        self.grid_shape = grid_shape
        self.num_points = int(grid_flat.shape[0])
        self.register_buffer("grid_xyz", grid, persistent=False)

        if unique_pairs_only:
            pairs = [(i, j) for i in range(self.N) for j in range(i + 1, self.N)]
        else:
            pairs = [(i, j) for i in range(self.N) for j in range(self.N)]
        pair_i = np.asarray([row[0] for row in pairs], dtype=np.int64)
        pair_j = np.asarray([row[1] for row in pairs], dtype=np.int64)
        self.register_buffer("pair_i", torch.from_numpy(pair_i), persistent=False)
        self.register_buffer("pair_j", torch.from_numpy(pair_j), persistent=False)
        self.full_pair_count = int(self.N * self.N)
        self.diagonal_pair_count = int(self.N)

        rn = np.asarray(rn, dtype=np.float32)
        tdoa_samples = np.empty((len(pairs), self.num_points), dtype=np.float32)
        for pair_index, (i, j) in enumerate(pairs):
            delay_seconds = np.dot(grid_flat, rn[j, :] - rn[i, :]) / self.c
            tdoa_samples[pair_index, :] = delay_seconds * self.fs
        self.register_buffer("tdoa_lookup", torch.from_numpy(tdoa_samples), persistent=False)

    def _reshape_maps(self, maps_flat: torch.Tensor) -> torch.Tensor:
        return maps_flat.reshape(maps_flat.shape[:-1] + self.grid_shape)

    def _add_diagonal_constant(self, maps_flat: torch.Tensor, gcc: torch.Tensor) -> torch.Tensor:
        if not self.unique_pairs_only:
            return maps_flat
        diag_index = torch.arange(self.N, device=gcc.device)
        diag = gcc[..., diag_index, diag_index, 0].sum(dim=-1, keepdim=True)
        return maps_flat + diag

    def metadata(self, *, srp_variant: str, symmetric_pair_factor: float) -> PHATVariantMetadata:
        return PHATVariantMetadata(
            srp_variant=srp_variant,
            pair_count=int(self.pair_i.numel()),
            full_pair_count=int(self.full_pair_count),
            diagonal_pair_count=int(self.diagonal_pair_count),
            interpolation_taps=int(self.interpolation_taps()),
            grid_shape=self.grid_shape,
            unique_pairs_only=bool(self.unique_pairs_only),
            symmetric_pair_factor=float(symmetric_pair_factor),
            buffer_stats=_named_buffer_stats(self),
            complexity_proxy=self.complexity_proxy(symmetric_pair_factor=symmetric_pair_factor),
        )

    def interpolation_taps(self) -> int:
        raise NotImplementedError

    def complexity_proxy(self, *, symmetric_pair_factor: float) -> dict[str, int | float]:
        taps = int(self.interpolation_taps())
        pair_count = int(self.pair_i.numel())
        diag_count = int(self.diagonal_pair_count if self.unique_pairs_only else 0)
        sample_reads_per_grid = pair_count * taps + diag_count
        weight_multiplies_per_grid = pair_count * taps
        pair_reduction_adds_per_grid = pair_count * max(taps - 1, 0)
        pair_sum_outputs = pair_count + diag_count
        grid_accumulation_adds_per_grid = max(pair_sum_outputs - 1, 0)
        symmetry_multiplies_per_grid = pair_count if self.unique_pairs_only and symmetric_pair_factor != 1.0 else 0
        total_arithmetic_per_grid = (
            weight_multiplies_per_grid
            + pair_reduction_adds_per_grid
            + grid_accumulation_adds_per_grid
            + symmetry_multiplies_per_grid
        )
        grid_points = int(self.num_points)
        return {
            "grid_points": grid_points,
            "sample_reads_per_grid_point": sample_reads_per_grid,
            "weight_multiplies_per_grid_point": weight_multiplies_per_grid,
            "pair_reduction_adds_per_grid_point": pair_reduction_adds_per_grid,
            "grid_accumulation_adds_per_grid_point": grid_accumulation_adds_per_grid,
            "symmetry_multiplies_per_grid_point": symmetry_multiplies_per_grid,
            "total_arithmetic_ops_per_grid_point": total_arithmetic_per_grid,
            "sample_reads_per_frame": sample_reads_per_grid * grid_points,
            "weight_multiplies_per_frame": weight_multiplies_per_grid * grid_points,
            "pair_reduction_adds_per_frame": pair_reduction_adds_per_grid * grid_points,
            "grid_accumulation_adds_per_frame": grid_accumulation_adds_per_grid * grid_points,
            "symmetry_multiplies_per_frame": symmetry_multiplies_per_grid * grid_points,
            "total_arithmetic_ops_per_frame": total_arithmetic_per_grid * grid_points,
        }


class _InterpolatedSRPReference(_InterpolatedSRPBase):
    def __init__(self, **kwargs):
        super().__init__(unique_pairs_only=False, **kwargs)
        offsets = np.arange(-self.sinc_half_width, self.sinc_half_width + 1, dtype=np.int64)
        center = np.rint(self.tdoa_lookup.cpu().numpy()).astype(np.int64)
        support = center[..., np.newaxis] + offsets.reshape(1, 1, -1)
        index, valid = _lag_to_gcc_index(support, self.tau_max)
        weights = np.sinc(self.tdoa_lookup.cpu().numpy()[..., np.newaxis] - support.astype(np.float32))
        weights = _normalize_weights(weights, valid)
        self.register_buffer("tap_offsets", torch.from_numpy(offsets), persistent=False)
        self.register_buffer("sample_index", torch.from_numpy(index), persistent=False)
        self.register_buffer("sample_weight", torch.from_numpy(weights), persistent=False)

    def interpolation_taps(self) -> int:
        return int(self.sample_index.shape[-1])

    def forward(self, gcc: torch.Tensor) -> torch.Tensor:
        pair_gcc = gcc[..., self.pair_i, self.pair_j, :]
        samples = torch.take_along_dim(
            pair_gcc.unsqueeze(-2),
            self.sample_index.view(1, 1, 1, *self.sample_index.shape),
            dim=-1,
        )
        weighted = samples * self.sample_weight.view(1, 1, 1, *self.sample_weight.shape)
        maps_flat = weighted.sum(dim=-1).sum(dim=-2)
        maps = self._reshape_maps(maps_flat.float())
        if self.normalize:
            maps = _normalize_maps(maps)
        return maps


class _InterpolatedSRPEdge(_InterpolatedSRPBase):
    def __init__(self, **kwargs):
        super().__init__(unique_pairs_only=True, **kwargs)
        center = np.rint(self.tdoa_lookup.cpu().numpy()).astype(np.int64)
        fractional = self.tdoa_lookup.cpu().numpy() - center.astype(np.float32)
        center_index, center_valid = _lag_to_gcc_index(center, self.tau_max)

        nonnegative_offsets = np.arange(0, self.sinc_half_width + 1, dtype=np.int64)
        if self.sinc_half_width > 0:
            left_support = center[..., np.newaxis] - nonnegative_offsets[1:].reshape(1, 1, -1)
            right_support = center[..., np.newaxis] + nonnegative_offsets[1:].reshape(1, 1, -1)
            left_index, left_valid = _lag_to_gcc_index(left_support, self.tau_max)
            right_index, right_valid = _lag_to_gcc_index(right_support, self.tau_max)
            left_weight = np.sinc(fractional[..., np.newaxis] + nonnegative_offsets[1:].reshape(1, 1, -1).astype(np.float32))
            right_weight = np.sinc(fractional[..., np.newaxis] - nonnegative_offsets[1:].reshape(1, 1, -1).astype(np.float32))
            center_weight = np.sinc(fractional)
            total_weight = np.concatenate(
                [
                    center_weight[..., np.newaxis] * center_valid[..., np.newaxis].astype(np.float32),
                    left_weight * left_valid.astype(np.float32),
                    right_weight * right_valid.astype(np.float32),
                ],
                axis=-1,
            )
            total_weight = _normalize_weights(total_weight, np.ones_like(total_weight, dtype=np.bool_))
            center_weight = total_weight[..., 0]
            left_weight = total_weight[..., 1 : 1 + left_weight.shape[-1]]
            right_weight = total_weight[..., 1 + left_weight.shape[-1] :]
        else:
            left_index = np.empty(center.shape + (0,), dtype=np.int64)
            right_index = np.empty(center.shape + (0,), dtype=np.int64)
            left_weight = np.empty(center.shape + (0,), dtype=np.float32)
            right_weight = np.empty(center.shape + (0,), dtype=np.float32)
            center_weight = np.ones_like(fractional, dtype=np.float32)

        self.register_buffer("nonnegative_offsets", torch.from_numpy(nonnegative_offsets), persistent=False)
        self.register_buffer("center_index", torch.from_numpy(center_index), persistent=False)
        self.register_buffer("center_weight", torch.from_numpy(center_weight.astype(np.float32)), persistent=False)
        self.register_buffer("left_index", torch.from_numpy(left_index), persistent=False)
        self.register_buffer("right_index", torch.from_numpy(right_index), persistent=False)
        self.register_buffer("left_weight", torch.from_numpy(left_weight.astype(np.float32)), persistent=False)
        self.register_buffer("right_weight", torch.from_numpy(right_weight.astype(np.float32)), persistent=False)

    def interpolation_taps(self) -> int:
        return int(1 + self.left_index.shape[-1] + self.right_index.shape[-1])

    def forward(self, gcc: torch.Tensor) -> torch.Tensor:
        pair_gcc = gcc[..., self.pair_i, self.pair_j, :]
        center = torch.take_along_dim(
            pair_gcc.unsqueeze(-2),
            self.center_index.view(1, 1, 1, *self.center_index.shape, 1),
            dim=-1,
        ).squeeze(-1)
        interp = center * self.center_weight.view(1, 1, 1, *self.center_weight.shape)
        if self.left_index.shape[-1] > 0:
            left = torch.take_along_dim(
                pair_gcc.unsqueeze(-2),
                self.left_index.view(1, 1, 1, *self.left_index.shape),
                dim=-1,
            )
            right = torch.take_along_dim(
                pair_gcc.unsqueeze(-2),
                self.right_index.view(1, 1, 1, *self.right_index.shape),
                dim=-1,
            )
            interp = interp + (left * self.left_weight.view(1, 1, 1, *self.left_weight.shape)).sum(dim=-1)
            interp = interp + (right * self.right_weight.view(1, 1, 1, *self.right_weight.shape)).sum(dim=-1)
        maps_flat = 2.0 * interp.sum(dim=-2)
        maps_flat = self._add_diagonal_constant(maps_flat, gcc)
        maps = self._reshape_maps(maps_flat.float())
        if self.normalize:
            maps = _normalize_maps(maps)
        return maps


class SRPPHATIcoMapAdapter(nn.Module):
    """Stage-1 PHAT feature adapter with switchable SRP backends."""

    SUPPORTED_VARIANTS = ("paper_original", "lc_reference", "lc_edge")

    def __init__(
        self,
        N: int,
        K: int,
        r: int,
        rn,
        fs: int,
        c: float = 343.0,
        normalize: bool = True,
        srp_variant: str = "paper_original",
        sinc_half_width: int = 0,
    ):
        super().__init__()
        self.N = int(N)
        self.K = int(K)
        self.r = int(r)
        self.fs = int(fs)
        self.c = float(c)
        self.normalize = bool(normalize)
        self.srp_variant = str(srp_variant)
        self.sinc_half_width = int(sinc_half_width)
        if self.srp_variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported srp_variant={self.srp_variant!r}; expected one of {self.SUPPORTED_VARIANTS}.")
        if self.sinc_half_width < 0:
            raise ValueError(f"sinc_half_width must be >= 0, got {self.sinc_half_width}")

        rn = np.asarray(rn, dtype=np.float32)
        dist_max = np.max([np.max([np.linalg.norm(rn[n, :] - rn[m, :]) for m in range(N)]) for n in range(N)])
        tau_max = int(np.ceil(dist_max / c * fs))
        self.tau_max = int(tau_max)
        self.gcc = at_modules.GCC(N, K, tau_max=tau_max, transform="PHAT")

        self.paper_srp: nn.Module | None = None
        self.interpolated_srp: _InterpolatedSRPBase | None = None
        if self.srp_variant == "paper_original":
            self.paper_srp = at_modules.SRP_icosahedral_map(
                N,
                K,
                r,
                rn,
                fs,
                c=c,
                normalize=normalize,
            )
        elif self.srp_variant == "lc_reference":
            self.interpolated_srp = _InterpolatedSRPReference(
                N=N,
                K=K,
                r=r,
                rn=rn,
                fs=fs,
                c=c,
                tau_max=tau_max,
                normalize=normalize,
                sinc_half_width=self.sinc_half_width,
            )
        else:
            self.interpolated_srp = _InterpolatedSRPEdge(
                N=N,
                K=K,
                r=r,
                rn=rn,
                fs=fs,
                c=c,
                tau_max=tau_max,
                normalize=normalize,
                sinc_half_width=self.sinc_half_width,
            )

    def frontend_profile(self) -> dict[str, Any]:
        if self.srp_variant == "paper_original":
            tau0 = torch.from_numpy(self.paper_srp.tau0.copy())
            grid_points = int(np.prod(tau0.shape[2:]))
            pair_count = int(self.N * self.N)
            sample_reads_per_grid = pair_count
            grid_accumulation_adds_per_grid = max(pair_count - 1, 0)
            return {
                "srp_variant": self.srp_variant,
                "pair_count": pair_count,
                "full_pair_count": pair_count,
                "diagonal_pair_count": int(self.N),
                "interpolation_taps": 1,
                "grid_shape": list(tau0.shape[2:]),
                "unique_pairs_only": False,
                "symmetric_pair_factor": 1.0,
                "cache_table_bytes": int(tau0.numel() * tau0.element_size()),
                "cache_tables": {
                    "tau0": {
                        "shape": list(tau0.shape),
                        "dtype": str(tau0.dtype),
                        "bytes": int(tau0.numel() * tau0.element_size()),
                    }
                },
                "complexity_proxy": {
                    "grid_points": grid_points,
                    "sample_reads_per_grid_point": sample_reads_per_grid,
                    "weight_multiplies_per_grid_point": 0,
                    "pair_reduction_adds_per_grid_point": 0,
                    "grid_accumulation_adds_per_grid_point": grid_accumulation_adds_per_grid,
                    "symmetry_multiplies_per_grid_point": 0,
                    "total_arithmetic_ops_per_grid_point": grid_accumulation_adds_per_grid,
                    "sample_reads_per_frame": sample_reads_per_grid * grid_points,
                    "weight_multiplies_per_frame": 0,
                    "pair_reduction_adds_per_frame": 0,
                    "grid_accumulation_adds_per_frame": grid_accumulation_adds_per_grid * grid_points,
                    "symmetry_multiplies_per_frame": 0,
                    "total_arithmetic_ops_per_frame": grid_accumulation_adds_per_grid * grid_points,
                },
            }
        if self.srp_variant == "lc_reference":
            return self.interpolated_srp.metadata(srp_variant=self.srp_variant, symmetric_pair_factor=1.0).to_dict()
        return self.interpolated_srp.metadata(srp_variant=self.srp_variant, symmetric_pair_factor=2.0).to_dict()

    def forward(self, mic_sig_batch) -> torch.Tensor:
        mic_sig_batch = ensure_mic_tensor(mic_sig_batch)
        gcc = self.gcc(mic_sig_batch)
        if self.paper_srp is not None:
            return self.paper_srp(gcc)
        if self.interpolated_srp is None:
            raise RuntimeError("Interpolated SRP backend is not initialized.")
        return self.interpolated_srp(gcc)
