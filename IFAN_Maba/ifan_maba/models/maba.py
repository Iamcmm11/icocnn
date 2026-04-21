from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MABATemporalConfig:
    channels: int = 16
    d_model: int = 32
    state_dim: int = 16
    conv_kernel: int = 3
    dropout: float = 0.1
    use_residual: bool = True
    use_gate: bool = True
    use_state: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | "MABATemporalConfig" | None) -> "MABATemporalConfig":
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        return cls(
            channels=int(payload.get("channels", cls.channels)),
            d_model=int(payload.get("d_model", cls.d_model)),
            state_dim=int(payload.get("state_dim", cls.state_dim)),
            conv_kernel=int(payload.get("conv_kernel", cls.conv_kernel)),
            dropout=float(payload.get("dropout", cls.dropout)),
            use_residual=bool(payload.get("use_residual", cls.use_residual)),
            use_gate=bool(payload.get("use_gate", cls.use_gate)),
            use_state=bool(payload.get("use_state", cls.use_state)),
        )

    def with_channels(self, channels: int) -> "MABATemporalConfig":
        payload = asdict(self)
        payload["channels"] = int(channels)
        return type(self)(**payload)


class MABAChannelTemporalBlock(nn.Module):
    """MABA-style temporal block matching a Conv1d-like (N, C, T) interface."""

    def __init__(
        self,
        channels: int,
        d_model: int = 32,
        state_dim: int = 16,
        conv_kernel: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_gate: bool = True,
        use_state: bool = True,
    ):
        super().__init__()
        if conv_kernel < 1:
            raise ValueError("conv_kernel must be >= 1")
        self.channels = int(channels)
        self.d_model = int(d_model)
        self.state_dim = int(state_dim)
        self.conv_kernel = int(conv_kernel)
        self.use_residual = bool(use_residual)
        self.use_gate = bool(use_gate)
        self.use_state = bool(use_state)

        self.in_proj = nn.Linear(self.channels, self.d_model)
        self.dw_conv = nn.Conv1d(
            self.d_model,
            self.d_model,
            kernel_size=self.conv_kernel,
            groups=self.d_model,
            bias=True,
        )
        self.mix_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

        state_in_dim = self.state_dim * 2 if self.use_gate else self.state_dim
        self.state_proj = nn.Linear(self.d_model, state_in_dim)
        self.state_back = nn.Linear(self.state_dim, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.channels)

        if not self.use_gate:
            self.alpha_logit = nn.Parameter(torch.zeros(self.state_dim))
        else:
            self.alpha_logit = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Expected input with shape (N, C, T)")
        bsz, channels, tlen = x.shape
        if channels != self.channels:
            raise ValueError(
                "Input channel mismatch: expected {}, got {}".format(self.channels, channels)
            )

        x_time = x.transpose(1, 2)
        z = self.in_proj(x_time)

        z_t = z.transpose(1, 2)
        if self.conv_kernel > 1:
            z_t = F.pad(z_t, (self.conv_kernel - 1, 0))
        z_conv = self.dw_conv(z_t)[..., :tlen].transpose(1, 2)
        z = self.mix_norm(z + z_conv)
        z = self.dropout(z)

        state_input = self.state_proj(z)
        if self.use_gate:
            q, g = state_input.chunk(2, dim=-1)
            alpha = torch.sigmoid(g)
        else:
            q = state_input
            alpha = torch.sigmoid(self.alpha_logit).view(1, 1, -1).expand_as(q)

        if self.use_state:
            h = torch.zeros(
                bsz,
                self.state_dim,
                dtype=q.dtype,
                device=q.device,
            )
            h_seq = []
            for t in range(tlen):
                a_t = alpha[:, t, :]
                q_t = q[:, t, :]
                h = a_t * h + (1.0 - a_t) * q_t
                h_seq.append(h)
            s = torch.stack(h_seq, dim=1)
        else:
            s = q

        z_refined = z + self.state_back(s)
        z_refined = self.dropout(z_refined)
        delta = self.out_proj(z_refined).transpose(1, 2)
        return x + delta if self.use_residual else delta
