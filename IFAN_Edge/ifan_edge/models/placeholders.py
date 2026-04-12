from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch.nn as nn


class IFANEdgeVariant(str, Enum):
    FULL = "ifan"
    LARGE = "ifan_edge_large"
    MEDIUM = "ifan_edge_medium"
    SMALL = "ifan_edge_small"


@dataclass
class IFANModelConfig:
    r: int = 2
    in_channels_per_branch: int = 1
    branch_channels: int = 16
    fused_channels: int = 16
    variant: IFANEdgeVariant = IFANEdgeVariant.FULL


class ResidualIcoBlock(nn.Module):
    """Reserved for stage-2 residual learning over icosahedral features."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Stage 2 will implement ResidualIcoBlock.")


class SharedAttentionFusion(nn.Module):
    """Reserved for stage-2 shared attention feature fusion."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Stage 2 will implement SharedAttentionFusion.")


class IFANModel(nn.Module):
    """Reserved for the stage-2 IFAN architecture."""

    def __init__(self, config: IFANModelConfig):
        super().__init__()
        self.config = config
        raise NotImplementedError("Stage 2 will implement IFANModel.")
