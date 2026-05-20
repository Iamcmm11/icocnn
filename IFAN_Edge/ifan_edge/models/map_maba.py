from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MapMABATemporalConfig:
    charts: int = 5
    height: int = 4
    width: int = 8
    d_model: int = 16
    state_dim: int = 8
    conv_kernel: int = 3
    dropout: float = 0.1
    use_residual: bool = True
    use_gate: bool = True
    use_state: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | "MapMABATemporalConfig" | None) -> "MapMABATemporalConfig":
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        return cls(
            charts=int(payload.get("charts", cls.charts)),
            height=int(payload.get("height", cls.height)),
            width=int(payload.get("width", cls.width)),
            d_model=int(payload.get("d_model", cls.d_model)),
            state_dim=int(payload.get("state_dim", cls.state_dim)),
            conv_kernel=int(payload.get("conv_kernel", cls.conv_kernel)),
            dropout=float(payload.get("dropout", cls.dropout)),
            use_residual=bool(payload.get("use_residual", cls.use_residual)),
            use_gate=bool(payload.get("use_gate", cls.use_gate)),
            use_state=bool(payload.get("use_state", cls.use_state)),
        )

    def with_grid(self, *, charts: int, height: int, width: int) -> "MapMABATemporalConfig":
        payload = asdict(self)
        payload["charts"] = int(charts)
        payload["height"] = int(height)
        payload["width"] = int(width)
        return type(self)(**payload)


class MABATemporalRefiner(nn.Module):
    """Map-level temporal refiner applied immediately before SoftArgMax."""

    def __init__(
        self,
        *,
        charts: int,
        height: int,
        width: int,
        d_model: int = 16,
        state_dim: int = 8,
        conv_kernel: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_gate: bool = True,
        use_state: bool = True,
    ):
        super().__init__()
        if conv_kernel < 1:
            raise ValueError("conv_kernel must be >= 1")
        self.charts = int(charts)
        self.height = int(height)
        self.width = int(width)
        self.map_size = self.charts * self.height * self.width
        self.d_model = int(d_model)
        self.state_dim = int(state_dim)
        self.conv_kernel = int(conv_kernel)
        self.use_residual = bool(use_residual)
        self.use_gate = bool(use_gate)
        self.use_state = bool(use_state)

        self.in_proj = nn.Linear(self.map_size, self.d_model)
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
        self.out_proj = nn.Linear(self.d_model, self.map_size)

        if not self.use_gate:
            self.alpha_logit = nn.Parameter(torch.zeros(self.state_dim))
        else:
            self.alpha_logit = None

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        if maps.dim() != 5:
            raise ValueError(f"Expected maps with shape (B, T, charts, H, W), got {tuple(maps.shape)}")
        bsz, tlen, charts, height, width = maps.shape
        if charts != self.charts or height != self.height or width != self.width:
            raise ValueError(
                "Input map size mismatch: expected ({}, {}, {}), got ({}, {}, {})".format(
                    self.charts,
                    self.height,
                    self.width,
                    charts,
                    height,
                    width,
                )
            )

        x = maps.reshape(bsz, tlen, self.map_size)
        x = self.in_proj(x)

        x_t = x.transpose(1, 2)
        if self.conv_kernel > 1:
            x_t = F.pad(x_t, (self.conv_kernel - 1, 0))
        x_conv = self.dw_conv(x_t)[..., :tlen].transpose(1, 2)
        z = self.mix_norm(x + x_conv)
        z = self.dropout(z)

        state_input = self.state_proj(z)
        if self.use_gate:
            q, g = state_input.chunk(2, dim=-1)
            alpha = torch.sigmoid(g)
        else:
            q = state_input
            alpha = torch.sigmoid(self.alpha_logit).view(1, 1, -1).expand_as(q)

        if self.use_state:
            h = torch.zeros(bsz, self.state_dim, dtype=q.dtype, device=q.device)
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
        delta = self.out_proj(z_refined).reshape(bsz, tlen, charts, height, width)
        return maps + delta if self.use_residual else delta

    def mac_proxy(self, time_steps: int) -> int:
        p = self.map_size
        d = self.d_model
        s = self.state_dim
        k = self.conv_kernel
        gate_term = 2 * d * s if self.use_gate else d * s
        per_step = p * d + d * k + gate_term + s + s * d + d * p
        return int(time_steps) * int(per_step)


class FeatureMABATemporalRefiner(nn.Module):
    """Temporal refiner over feature maps before channel readout."""

    def __init__(
        self,
        *,
        channels: int,
        d_model: int = 16,
        state_dim: int = 8,
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
        if x.dim() != 7:
            raise ValueError(f"Expected features with shape (B, T, C, R, charts, H, W), got {tuple(x.shape)}")
        bsz, tlen, channels, regions, charts, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"Input channel mismatch: expected {self.channels}, got {channels}")

        z = x.permute(0, 3, 4, 5, 6, 1, 2).reshape(-1, tlen, channels)
        z = self.in_proj(z)

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
            h = torch.zeros(z.shape[0], self.state_dim, dtype=q.dtype, device=q.device)
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
        delta = self.out_proj(z_refined).reshape(bsz, regions, charts, height, width, tlen, channels)
        delta = delta.permute(0, 5, 6, 1, 2, 3, 4).contiguous()
        return x + delta if self.use_residual else delta

    def mac_proxy(self, *, time_steps: int, regions: int, charts: int, height: int, width: int) -> int:
        positions = int(regions) * int(charts) * int(height) * int(width)
        c = self.channels
        d = self.d_model
        s = self.state_dim
        k = self.conv_kernel
        gate_term = 2 * d * s if self.use_gate else d * s
        per_step = c * d + d * k + gate_term + s + s * d + d * c
        return int(time_steps) * int(positions) * int(per_step)
