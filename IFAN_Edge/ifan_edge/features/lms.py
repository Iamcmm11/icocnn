from __future__ import annotations

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bridges import icoCNN
from .phat import ensure_mic_tensor


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
        avoid_negatives: bool = False,
        eps: float = 1e-8,
        normalized_lms: bool = True,
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
        self.avoid_negatives = avoid_negatives
        self.eps = float(eps)
        self.normalized_lms = normalized_lms

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

    def _estimate_pair_filters(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Estimate a centered FIR filter per frame with LMS/NLMS.

        Parameters
        ----------
        source, target:
            Tensors with shape [BW, K].
        """

        padded = F.pad(source, (self.lms_order, self.lms_order))
        weights = source.new_zeros((source.shape[0], self.filter_len))

        for sample_idx in range(self.K):
            x_vec = padded[:, sample_idx : sample_idx + self.filter_len]
            y_hat = (weights * x_vec).sum(dim=-1)
            error = target[:, sample_idx] - y_hat

            if self.normalized_lms:
                denom = x_vec.square().sum(dim=-1).clamp_min(self.eps)
                update = (self.step_size * error / denom).unsqueeze(-1)
            else:
                update = (self.step_size * error).unsqueeze(-1)

            weights = weights + update * x_vec

        return weights

    def _normalize_maps(self, maps: torch.Tensor) -> torch.Tensor:
        original_shape = maps.shape
        maps = maps + 1e-12
        maps = maps.reshape(maps.shape[:-3] + (-1,))
        denom = maps.amax(dim=-1, keepdim=True).clamp_min(1e-12)
        maps = maps / denom
        return maps.reshape(original_shape)

    def forward(self, mic_sig_batch) -> torch.Tensor:
        mic_sig_batch = ensure_mic_tensor(mic_sig_batch).float()

        batch_frames = mic_sig_batch.reshape(-1, self.N, self.K)
        tau0 = self.tau0.to(batch_frames.device)
        maps = batch_frames.new_zeros((batch_frames.shape[0],) + self.grid_shape)

        for n in range(self.N):
            for m in range(self.N):
                pair_filters = self._estimate_pair_filters(batch_frames[:, n, :], batch_frames[:, m, :])
                tau_idx = (tau0[n, m] + self.lms_order).clamp(0, self.filter_len - 1)
                sampled = pair_filters[:, tau_idx.reshape(-1)]
                maps = maps + sampled.reshape((batch_frames.shape[0],) + self.grid_shape)

        if self.avoid_negatives:
            maps = torch.relu(maps)

        if self.normalize:
            maps = self._normalize_maps(maps)

        return maps.reshape(mic_sig_batch.shape[:-2] + self.grid_shape)
