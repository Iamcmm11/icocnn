from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .placeholders import IFANModel, IFANModelConfig


@dataclass
class DcaseAzimuthHeadConfig:
    hidden_dim: int = 32
    dropout: float = 0.1

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | "DcaseAzimuthHeadConfig" | None) -> "DcaseAzimuthHeadConfig":
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        return cls(
            hidden_dim=int(payload.get("hidden_dim", cls.hidden_dim)),
            dropout=float(payload.get("dropout", cls.dropout)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DcaseAzimuthOnlyIFANModel(nn.Module):
    """IFAN backbone with a dedicated folded-azimuth sin/cos regression head."""

    def __init__(self, backbone_config: IFANModelConfig, head_config: DcaseAzimuthHeadConfig):
        super().__init__()
        self.backbone_config = backbone_config
        self.head_config = head_config
        self.backbone = IFANModel(backbone_config)

        charts = 5
        output_r = self.backbone.output_r
        height = 2**output_r
        width = 2 ** (output_r + 1)
        self.map_size = int(charts * height * width)
        hidden_dim = int(head_config.hidden_dim)
        dropout = float(head_config.dropout)
        self.azimuth_head = nn.Sequential(
            nn.LayerNorm(self.map_size),
            nn.Linear(self.map_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def count_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (param for param in self.parameters() if param.requires_grad)
        return sum(param.numel() for param in params)

    def parameter_breakdown(self) -> dict[str, int]:
        backbone_breakdown = self.backbone.parameter_breakdown()
        head_params = sum(param.numel() for param in self.azimuth_head.parameters())
        breakdown = dict(backbone_breakdown)
        breakdown["azimuth_head"] = int(head_params)
        breakdown["total"] = int(backbone_breakdown["total"] + head_params)
        return breakdown

    def mac_proxy(self, input_shape: tuple[int, int, int, int, int, int], kernel_neighbors: int = 7) -> dict[str, int]:
        backbone = self.backbone.mac_proxy(input_shape, kernel_neighbors=kernel_neighbors)
        time_steps = int(input_shape[2])
        hidden_dim = int(self.head_config.hidden_dim)
        head = int(time_steps) * (self.map_size * hidden_dim + hidden_dim * 2)
        breakdown = dict(backbone)
        total = int(backbone["total"] + head)
        breakdown["azimuth_head"] = head
        breakdown["total"] = total
        return breakdown

    def forward_sincos(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone.forward_map_logits(x)
        flat = logits.reshape(logits.shape[0], logits.shape[1], -1)
        sincos = self.azimuth_head(flat)
        return F.normalize(sincos, dim=-1, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sincos = self.forward_sincos(x)
        sin_phi = sincos[..., 0]
        cos_phi = sincos[..., 1]
        zeros = torch.zeros_like(sin_phi)
        coords = torch.stack((cos_phi, sin_phi, zeros), dim=-1)
        return coords
