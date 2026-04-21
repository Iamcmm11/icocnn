from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..bridges import at_modules


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


class SRPPHATIcoMapAdapter(nn.Module):
    """Stage-1 PHAT feature adapter with the same output contract as the LMS branch."""

    def __init__(
        self,
        N: int,
        K: int,
        r: int,
        rn,
        fs: int,
        c: float = 343.0,
        normalize: bool = True,
    ):
        super().__init__()
        dist_max = np.max([np.max([np.linalg.norm(rn[n, :] - rn[m, :]) for m in range(N)]) for n in range(N)])
        tau_max = int(np.ceil(dist_max / c * fs))
        self.gcc = at_modules.GCC(N, K, tau_max=tau_max, transform="PHAT")
        self.srp = at_modules.SRP_icosahedral_map(
            N,
            K,
            r,
            rn,
            fs,
            c=c,
            normalize=normalize,
        )

    def forward(self, mic_sig_batch) -> torch.Tensor:
        mic_sig_batch = ensure_mic_tensor(mic_sig_batch)
        return self.srp(self.gcc(mic_sig_batch))
