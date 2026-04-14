from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

from ..bridges import at_modules, icoCNN


class IFANEdgeVariant(str, Enum):
    FULL = "ifan"
    LARGE = "ifan_edge_large"
    MEDIUM = "ifan_edge_medium"
    SMALL = "ifan_edge_small"


@dataclass
class IFANModelConfig:
    r: int = 2
    phat_in_channels: int = 1
    aux_in_channels: int = 1
    branch_channels: int = 16
    fused_channels: int = 16
    use_residual_block: bool = False
    smooth_vertices: bool = True
    variant: IFANEdgeVariant = IFANEdgeVariant.FULL


class ResidualIcoBlock(nn.Module):
    """Optional residual learning block over icosahedral features."""

    def __init__(self, r: int, channels: int, smooth_vertices: bool = True, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.conv1 = icoCNN.ConvIco(r, channels, channels, 6, 6, smooth_vertices=smooth_vertices)
            self.conv2 = icoCNN.ConvIco(r, channels, channels, 6, 6, smooth_vertices=smooth_vertices)
            self.norm = icoCNN.LNormIco(channels, 6)
        else:
            self.conv1 = None
            self.conv2 = None
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        residual = x
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.norm(x)
        return torch.relu(x + residual)


class SharedAttentionFusion(nn.Module):
    """Shared-weight attention fusion for PHAT and LMS branches."""

    def __init__(self, r: int, channels: int, smooth_vertices: bool = True):
        super().__init__()
        self.norm = icoCNN.LNormIco(channels, 6)
        self.conv1 = icoCNN.ConvIco(r, channels, channels, 6, 6, smooth_vertices=smooth_vertices)
        self.conv2 = icoCNN.ConvIco(r, channels, channels, 6, 6, smooth_vertices=smooth_vertices)

    def _attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = torch.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x

    def forward(
        self,
        phat_feat: torch.Tensor,
        lms_feat: torch.Tensor,
        return_attention: bool = False,
    ):
        phat_weight = self._attention_weights(phat_feat)
        lms_weight = self._attention_weights(lms_feat)

        phat_weighted = phat_feat * phat_weight
        lms_weighted = lms_feat * lms_weight
        fused = phat_weighted + lms_weighted

        if return_attention:
            return fused, phat_weight, lms_weight
        return fused


class _IFANBranch(nn.Module):
    """Single lightweight branch for one input feature."""

    def __init__(self, config: IFANModelConfig, in_channels: int):
        super().__init__()
        self.stem = icoCNN.ConvIco(
            config.r,
            in_channels,
            config.branch_channels,
            1,
            6,
            smooth_vertices=config.smooth_vertices,
        )
        self.residual = ResidualIcoBlock(
            r=config.r,
            channels=config.branch_channels,
            smooth_vertices=config.smooth_vertices,
            enabled=config.use_residual_block,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.stem(x))
        x = self.residual(x)
        return x


class IFANModel(nn.Module):
    """Stage-2 IFAN backbone with dual branches and shared attention fusion."""

    def __init__(self, config: IFANModelConfig):
        super().__init__()
        self.config = config
        if config.branch_channels != config.fused_channels:
            raise ValueError(
                "This stage-2 implementation keeps channels constant across branches and fusion. "
                f"Got branch_channels={config.branch_channels}, fused_channels={config.fused_channels}."
            )

        self.phat_branch = _IFANBranch(config, in_channels=config.phat_in_channels)
        self.aux_branch = _IFANBranch(config, in_channels=config.aux_in_channels)
        self.fusion = SharedAttentionFusion(
            r=config.r,
            channels=config.branch_channels,
            smooth_vertices=config.smooth_vertices,
        )

        pooled_r = max(config.r - 1, 1)
        self.pool = icoCNN.PoolIco(config.r, 6, smooth_vertices=config.smooth_vertices) if config.r > 1 else None
        self.fusion_conv = icoCNN.ConvIco(
            pooled_r,
            config.fused_channels,
            config.fused_channels,
            6,
            6,
            smooth_vertices=config.smooth_vertices,
        )
        self.fusion_norm = icoCNN.LNormIco(config.fused_channels, 6)
        self.output_conv = icoCNN.ConvIco(
            pooled_r,
            config.fused_channels,
            1,
            6,
            6,
            smooth_vertices=config.smooth_vertices,
        )
        self.clean_vertices = icoCNN.CleanVertices(pooled_r)

        ico_grid = torch.from_numpy(icoCNN.icosahedral_grid_coordinates(pooled_r)).float()
        ico_grid = ico_grid.permute(3, 0, 1, 2).contiguous()
        self.sam = at_modules.SoftArgMax(ico_grid.shape[1:], indexes=ico_grid, include_exp=True)

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if x.ndim != 6:
            raise ValueError(f"Expected [B, C, T, 5, H, W], got {tuple(x.shape)}")
        if x.shape[3] != 5:
            raise ValueError(f"Expected 5 icosahedral charts, got shape {tuple(x.shape)}")

    @staticmethod
    def _branch_input(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(3)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        self._validate_input(x)
        expected_channels = self.config.phat_in_channels + self.config.aux_in_channels
        if x.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} input channels "
                f"({self.config.phat_in_channels} PHAT + {self.config.aux_in_channels} aux), "
                f"got shape {tuple(x.shape)}"
            )

        phat = self._branch_input(x[:, : self.config.phat_in_channels, ...].transpose(1, 2))
        aux = self._branch_input(x[:, self.config.phat_in_channels :, ...].transpose(1, 2))

        phat_feat = self.phat_branch(phat)
        aux_feat = self.aux_branch(aux)

        if return_attention:
            fused, phat_weight, aux_weight = self.fusion(phat_feat, aux_feat, return_attention=True)
        else:
            fused = self.fusion(phat_feat, aux_feat, return_attention=False)

        if self.pool is not None:
            fused = self.pool(fused)

        fused = torch.relu(self.fusion_conv(fused))
        fused = torch.relu(self.fusion_norm(fused))
        logits = self.output_conv(fused)
        logits = logits.max(dim=2).values
        logits = logits.max(dim=2).values
        logits = self.clean_vertices(logits)
        coords = self.sam(logits)

        if return_attention:
            attention = {"phat": phat_weight, "aux": aux_weight}
            if self.config.aux_in_channels == 1:
                attention["lms"] = aux_weight
            return coords, attention
        return coords
