from __future__ import annotations

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional

from ..bridges import icoCNN
from .phat import ensure_mic_tensor


@torch.jit.script
def _estimate_pair_filters_script(
    source: torch.Tensor,
    target: torch.Tensor,
    lms_order: int,
    step_size: float,
    normalized_lms: bool,
    eps: float,
    initial_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    source = source.reshape(-1, source.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    filter_len = lms_order * 2 + 1
    padded = F.pad(source, (lms_order, lms_order))

    if initial_weights is None:
        weights = source.new_zeros((source.shape[0], filter_len))
    else:
        weights = initial_weights.reshape(-1, filter_len).to(device=source.device, dtype=source.dtype)

    for sample_idx in range(source.shape[1]):
        x_vec = padded[:, sample_idx : sample_idx + filter_len]
        y_hat = (weights * x_vec).sum(dim=-1)
        error = target[:, sample_idx] - y_hat

        if normalized_lms:
            denom = x_vec.square().sum(dim=-1).clamp_min(eps)
            update = (step_size * error / denom).unsqueeze(-1)
        else:
            update = (step_size * error).unsqueeze(-1)

        weights = weights + update * x_vec

    return weights


class SRPLMSIcoMap(nn.Module):
    """Compute SRP-LMS icosahedral maps with the same chart layout as SRP-PHAT.

    This stage-1 implementation favors readability and interface stability so it can
    be validated and iterated on in the training server environment.
    """

    def __init__(
        self,
        N: int,
        K: int,
        r: int,
        rn,
        fs: int,
        c: float = 343.0,
        lms_order: int = 64,
        step_size: float = 0.01,
        normalize: bool = True,
        map_mode: str = "tau_sample",
        peak_sigma: float = 2.0,
        update_mode: str = "frame_reset",
        normalized_lms: bool = True,
        include_self_pairs: bool = True,
        lms_backend: Literal["time_reference", "frequency_block"] = "time_reference",
        lms_block_size: int = 256,
        lms_fft_size: int | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.N = N
        self.K = K
        self.r = r
        self.fs = float(fs)
        self.c = c
        self.lms_order = int(lms_order)
        self.filter_len = self.lms_order * 2 + 1
        self.step_size = float(step_size)
        self.normalize = normalize
        self.map_mode = str(map_mode)
        self.peak_sigma = float(peak_sigma)
        self.update_mode = str(update_mode)
        self.include_self_pairs = bool(include_self_pairs)
        self.lms_backend = str(lms_backend)
        self.lms_block_size = int(lms_block_size)
        self.eps = float(eps)
        self.normalized_lms = bool(normalized_lms)
        if self.map_mode not in {"tau_sample", "peak_proximity"}:
            raise ValueError(f"Unsupported LMS map_mode: {self.map_mode}")
        if self.update_mode not in {"frame_reset", "trajectory_tracking"}:
            raise ValueError(f"Unsupported LMS update_mode: {self.update_mode}")
        if self.lms_backend not in {"time_reference", "frequency_block"}:
            raise ValueError(f"Unsupported LMS backend: {self.lms_backend}")
        if self.lms_block_size <= 0:
            raise ValueError(f"lms_block_size must be positive, got {self.lms_block_size}")
        if self.peak_sigma <= 0:
            raise ValueError(f"peak_sigma must be positive, got {self.peak_sigma}")
        self.lms_fft_size = self._resolve_fft_size(lms_fft_size)

        grid = icoCNN.icosahedral_grid_coordinates(r)
        self.grid_shape = tuple(grid.shape[:-1])

        tau = np.concatenate([range(0, K // 2 + 1), range(-K // 2 + 1, 0)]) / float(fs)
        tau0 = np.empty(grid.shape[:-1] + (N, N), dtype=np.int64)
        for k in range(N):
            for l in range(N):
                imtdf = np.dot(grid, rn[l, :] - rn[k, :]) / c
                tau0[..., k, l] = np.argmin(np.abs(imtdf[..., np.newaxis] - tau), axis=-1)
        tau0[tau0 > K // 2] -= K
        tau0 = einops.rearrange(tau0, "... Ni Nj -> Ni Nj ...", Ni=N, Nj=N)
        self.register_buffer("tau0", torch.from_numpy(tau0).long(), persistent=False)
        pair_i, pair_j = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
        if not self.include_self_pairs:
            pair_mask = pair_i != pair_j
            pair_i = pair_i[pair_mask]
            pair_j = pair_j[pair_mask]
        self.register_buffer("pair_i", pair_i.reshape(-1).long(), persistent=False)
        self.register_buffer("pair_j", pair_j.reshape(-1).long(), persistent=False)
        tau_idx = (torch.from_numpy(tau0).long() + self.lms_order).clamp(0, self.filter_len - 1)
        if not self.include_self_pairs:
            tau_idx = tau_idx[pair_mask]
        self.register_buffer("tau_idx", tau_idx.reshape(-1, int(np.prod(self.grid_shape))), persistent=False)

    def _resolve_fft_size(self, lms_fft_size: int | None) -> int:
        min_fft = self.lms_block_size + self.filter_len - 1
        if lms_fft_size is None:
            return 1 << int(np.ceil(np.log2(max(min_fft, 1))))
        fft_size = int(lms_fft_size)
        if fft_size < min_fft:
            raise ValueError(
                "lms_fft_size must be at least lms_block_size + filter_len - 1. "
                f"Got {fft_size}, expected >= {min_fft}."
            )
        return fft_size

    @staticmethod
    def _fft_valid_correlation(
        signal_context: torch.Tensor,
        kernel: torch.Tensor,
        fft_size: int,
    ) -> torch.Tensor:
        signal_len = signal_context.shape[-1]
        kernel_len = kernel.shape[-1]
        if kernel_len > signal_len:
            raise ValueError(f"Kernel length {kernel_len} exceeds signal length {signal_len}.")
        if fft_size < signal_len:
            raise ValueError(f"fft_size must cover the signal context. Got {fft_size} < {signal_len}.")

        signal_fft = torch.fft.rfft(signal_context, n=fft_size, dim=-1)
        kernel_fft = torch.fft.rfft(kernel.flip(-1), n=fft_size, dim=-1)
        circular = torch.fft.irfft(signal_fft * kernel_fft, n=fft_size, dim=-1)
        out_len = signal_len - kernel_len + 1
        return circular[..., kernel_len - 1 : kernel_len - 1 + out_len]

    @staticmethod
    def _sliding_window_energy(signal_context: torch.Tensor, window: int) -> torch.Tensor:
        squared = signal_context.square()
        cumsum = F.pad(squared.cumsum(dim=-1), (1, 0))
        return cumsum[..., window:] - cumsum[..., :-window]

    def _estimate_pair_filters(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        initial_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Estimate a centered FIR filter per frame with LMS/NLMS.

        Parameters
        ----------
        source, target:
            Tensors with shape [BW, K].
        """

        if source.shape != target.shape:
            raise ValueError(f"Expected source and target to share shape, got {source.shape} vs {target.shape}")
        original_shape = source.shape[:-1]
        flat_batch = int(np.prod(original_shape))
        if initial_weights is not None:
            expected_shape = (flat_batch, self.filter_len)
            reshaped = initial_weights.reshape(-1, self.filter_len)
            if tuple(reshaped.shape) != expected_shape:
                raise ValueError(
                    "initial_weights must match flattened source shape. "
                    f"Expected {expected_shape}, got {tuple(reshaped.shape)}"
                )

        if self.lms_backend == "time_reference":
            weights = _estimate_pair_filters_script(
                source=source,
                target=target,
                lms_order=self.lms_order,
                step_size=self.step_size,
                normalized_lms=self.normalized_lms,
                eps=self.eps,
                initial_weights=initial_weights,
            )
        else:
            weights = self._estimate_pair_filters_frequency_block(
                source=source.reshape(flat_batch, source.shape[-1]),
                target=target.reshape(flat_batch, target.shape[-1]),
                initial_weights=initial_weights.reshape(flat_batch, self.filter_len) if initial_weights is not None else None,
            )
        return weights.reshape(original_shape + (self.filter_len,))

    def _estimate_pair_filters_frequency_block(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        initial_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        padded = F.pad(source, (self.lms_order, self.lms_order))
        if initial_weights is None:
            weights = source.new_zeros((source.shape[0], self.filter_len))
        else:
            weights = initial_weights.to(device=source.device, dtype=source.dtype)

        for start in range(0, source.shape[-1], self.lms_block_size):
            stop = min(start + self.lms_block_size, source.shape[-1])
            block_len = stop - start
            source_context = padded[:, start : stop + self.filter_len - 1]
            fft_size = max(self.lms_fft_size, source_context.shape[-1])

            estimate = self._fft_valid_correlation(
                signal_context=source_context,
                kernel=weights,
                fft_size=fft_size,
            )
            error = target[:, start:stop] - estimate

            if self.normalized_lms:
                denom = self._sliding_window_energy(source_context, self.filter_len).clamp_min(self.eps)
                scaled_error = error / denom
            else:
                scaled_error = error

            gradient = self._fft_valid_correlation(
                signal_context=source_context,
                kernel=scaled_error,
                fft_size=fft_size,
            )
            weights = weights + self.step_size * gradient

        return weights

    def _sample_pair_filters(self, pair_filters: torch.Tensor) -> torch.Tensor:
        tau_idx = self.tau_idx.to(pair_filters.device)
        if self.map_mode == "tau_sample":
            return pair_filters.gather(
                dim=-1,
                index=tau_idx.unsqueeze(1).expand(-1, pair_filters.shape[1], -1),
            )

        peak_idx = pair_filters.abs().argmax(dim=-1, keepdim=True)
        peak_val = pair_filters.gather(dim=-1, index=peak_idx).abs()
        distance = tau_idx.unsqueeze(1) - peak_idx.expand(-1, -1, tau_idx.shape[-1])
        return peak_val * torch.exp(-0.5 * (distance.float() / self.peak_sigma).square())

    def _project_filters_to_maps(self, pair_filters: torch.Tensor) -> torch.Tensor:
        sampled = self._sample_pair_filters(pair_filters)
        return sampled.sum(dim=0).reshape((pair_filters.shape[1],) + self.grid_shape)

    def _normalize_maps(self, maps: torch.Tensor) -> torch.Tensor:
        original_shape = maps.shape
        maps = maps.reshape(maps.shape[:-3] + (-1,))
        maps = maps + 1e-12
        denom = maps.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        maps = maps / denom
        return maps.reshape(original_shape)

    def forward(self, mic_sig_batch) -> torch.Tensor:
        mic_sig_batch = ensure_mic_tensor(mic_sig_batch).float()

        if self.update_mode == "frame_reset":
            batch_frames = mic_sig_batch.reshape(-1, self.N, self.K)
            sources = batch_frames[:, self.pair_i, :].permute(1, 0, 2).contiguous()
            targets = batch_frames[:, self.pair_j, :].permute(1, 0, 2).contiguous()
            pair_filters = self._estimate_pair_filters(sources, targets)
            maps = self._project_filters_to_maps(pair_filters)
        else:
            batch_size = mic_sig_batch.shape[0]
            frame_count = mic_sig_batch.shape[2]
            trajectory_frames = mic_sig_batch[:, 0, :, :, :]
            weights = mic_sig_batch.new_zeros((self.pair_i.numel() * batch_size, self.filter_len))
            maps_per_frame = []

            for frame_idx in range(frame_count):
                frame_sig = trajectory_frames[:, frame_idx, :, :]
                sources = frame_sig[:, self.pair_i, :].permute(1, 0, 2).contiguous()
                targets = frame_sig[:, self.pair_j, :].permute(1, 0, 2).contiguous()
                weights = self._estimate_pair_filters(sources, targets, initial_weights=weights).reshape(-1, self.filter_len)
                pair_filters = weights.reshape(self.pair_i.numel(), batch_size, self.filter_len)
                maps_per_frame.append(self._project_filters_to_maps(pair_filters))

            maps = torch.stack(maps_per_frame, dim=1)

        if self.normalize:
            maps = self._normalize_maps(maps)

        return maps.reshape(mic_sig_batch.shape[:-2] + self.grid_shape)
