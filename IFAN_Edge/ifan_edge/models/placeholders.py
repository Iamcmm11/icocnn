from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from ..bridges import at_modules, icoCNN


PAPER_IFAN_BRANCH_CHANNELS = 16
PAPER_IFAN_FUSION_BLOCKS = 4
PAPER_IFAN_PARAM_TARGET = 125_457


@dataclass
class IFANModelConfig:
    r: int = 2
    phat_in_channels: int = 1
    aux_in_channels: int = 1
    branch_channels: int = PAPER_IFAN_BRANCH_CHANNELS
    smooth_vertices: bool = True
    final_head_pooling: bool = False
    temporal_conv_variant: str = "standard_1d"

    @property
    def fused_channels(self) -> int:
        return self.branch_channels

    @property
    def fusion_head_channels(self) -> int:
        return self.branch_channels

    def __post_init__(self) -> None:
        if int(self.branch_channels) <= 0:
            raise ValueError(f"branch_channels must be positive, got {self.branch_channels}")
        if self.temporal_conv_variant not in {"standard_1d", "depthwise_separable_1d"}:
            raise ValueError(
                "temporal_conv_variant must be 'standard_1d' or 'depthwise_separable_1d', "
                f"got {self.temporal_conv_variant!r}."
            )


class ResidualLearningModule(nn.Module):
    """Residual feature enhancement module used inside each frontend branch."""

    def __init__(self, r: int, channels: int, smooth_vertices: bool = True):
        super().__init__()
        self.conv1 = icoCNN.ConvIco(r, channels, channels, 6, 6, smooth_vertices=smooth_vertices)
        self.conv2 = icoCNN.ConvIco(r, channels, channels, 6, 6, smooth_vertices=smooth_vertices)
        self.norm = icoCNN.LNormIco(channels, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.norm(x)
        return torch.relu(x + residual)


class FeatureAttentionWeightModule(nn.Module):
    """Shared-weight attention over enhanced branch features."""

    def __init__(self, r: int, channels: int, smooth_vertices: bool = True):
        super().__init__()
        self.norm = icoCNN.LNormIco(channels, 6)
        self.conv1 = icoCNN.ConvIco(r, channels, channels, 6, 6, smooth_vertices=smooth_vertices)
        self.conv2 = icoCNN.ConvIco(r, channels, channels, 6, 6, smooth_vertices=smooth_vertices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = torch.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x


class FrontendFeatureBranch(nn.Module):
    """Paper-style branch: keep direct feature and enhanced residual feature."""

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
        self.residual = ResidualLearningModule(
            r=config.r,
            channels=config.branch_channels,
            smooth_vertices=config.smooth_vertices,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        direct = torch.relu(self.stem(x))
        enhanced = self.residual(direct)
        return direct, enhanced


class DepthwiseSeparableCausConv1d(nn.Module):
    """Causal depthwise-separable 1D convolution used in lightweight temporal blocks."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=self.pad,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        if self.pad > 0:
            x = x[:, :, :-self.pad]
        return self.pointwise(x)


def build_temporal_conv(variant: str, channels: int, kernel_size: int = 5, dilation: int = 1) -> nn.Module:
    if variant == "standard_1d":
        return at_modules.CausConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
    if variant == "depthwise_separable_1d":
        return DepthwiseSeparableCausConv1d(channels, kernel_size=kernel_size, dilation=dilation)
    raise ValueError(f"Unsupported temporal_conv_variant {variant!r}")


class FusionTemporalBlock(nn.Module):
    """One paper-style fusion block: IcoConv -> ReLU -> Conv1d -> LNorm -> optional ReLU."""

    def __init__(
        self,
        r: int,
        in_channels: int,
        out_channels: int,
        *,
        smooth_vertices: bool = True,
        apply_relu: bool = True,
        temporal_conv_variant: str = "standard_1d",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.apply_relu = apply_relu
        self.conv = icoCNN.ConvIco(r, in_channels, out_channels, 6, 6, smooth_vertices=smooth_vertices)
        self.temporal = build_temporal_conv(temporal_conv_variant, out_channels, kernel_size=5, dilation=1)
        self.norm = icoCNN.LNormIco(out_channels, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        bsz, time_steps, channels, regions, charts, height, width = x.shape
        x = rearrange(x, "b t c r ch h w -> (b r ch h w) c t")
        x = self.temporal(x)
        x = rearrange(
            x,
            "(b r ch h w) c t -> b t c r ch h w",
            b=bsz,
            r=regions,
            ch=charts,
            h=height,
            w=width,
        )
        x = self.norm(x)
        if self.apply_relu:
            x = torch.relu(x)
        return x


class FinalFusionBlock(nn.Module):
    """Final paper-style block: keep branch_channels through the last temporal layer."""

    def __init__(
        self,
        r: int,
        channels: int,
        *,
        smooth_vertices: bool = True,
        temporal_conv_variant: str = "standard_1d",
    ):
        super().__init__()
        self.conv = icoCNN.ConvIco(r, channels, channels, 6, 6, smooth_vertices=smooth_vertices)
        self.temporal = build_temporal_conv(temporal_conv_variant, channels, kernel_size=5, dilation=1)
        self.norm = icoCNN.LNormIco(channels, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        bsz, time_steps, channels, regions, charts, height, width = x.shape
        x = rearrange(x, "b t c r ch h w -> (b r ch h w) c t")
        x = self.temporal(x)
        x = rearrange(
            x,
            "(b r ch h w) c t -> b t c r ch h w",
            b=bsz,
            r=regions,
            ch=charts,
            h=height,
            w=width,
        )
        x = self.norm(x)
        return x


class ChannelReadout(nn.Module):
    """Learned per-position readout from 16 feature channels to one score map."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b t c r ch h w -> b t r ch h w c")
        x = self.proj(x)
        return rearrange(x, "b t r ch h w c -> b t c r ch h w")


# Backward-compatible alias for older imports/docs.
ResidualIcoBlock = ResidualLearningModule
SharedAttentionFusion = FeatureAttentionWeightModule


class IFANModel(nn.Module):
    """Paper-faithful dual-input IFAN backbone using shared-weight attention and a deep fusion head."""

    def __init__(self, config: IFANModelConfig):
        super().__init__()
        self.config = config
        if config.r < 1:
            raise ValueError(f"IFAN expects r >= 1, got {config.r}")
        if config.phat_in_channels != 1 or config.aux_in_channels != 1:
            raise ValueError(
                "This paper-style IFAN implementation expects one PHAT and one LMS channel. "
                f"Got phat_in_channels={config.phat_in_channels}, aux_in_channels={config.aux_in_channels}."
            )

        self.phat_branch = FrontendFeatureBranch(config, in_channels=config.phat_in_channels)
        self.aux_branch = FrontendFeatureBranch(config, in_channels=config.aux_in_channels)
        self.shared_attention = FeatureAttentionWeightModule(
            r=config.r,
            channels=config.branch_channels,
            smooth_vertices=config.smooth_vertices,
        )

        self.pre_fusion_pool = icoCNN.PoolIco(config.r, 6, smooth_vertices=config.smooth_vertices) if config.r > 1 else None
        self.fusion_r = config.r - 1 if config.r > 1 else config.r

        self.fusion_blocks = nn.ModuleList(
            [
                FusionTemporalBlock(
                    r=self.fusion_r,
                    in_channels=config.branch_channels,
                    out_channels=config.branch_channels,
                    smooth_vertices=config.smooth_vertices,
                    apply_relu=True,
                    temporal_conv_variant=config.temporal_conv_variant,
                )
                for _ in range(PAPER_IFAN_FUSION_BLOCKS)
            ]
        )
        self.final_block = FinalFusionBlock(
            r=self.fusion_r,
            channels=config.branch_channels,
            smooth_vertices=config.smooth_vertices,
            temporal_conv_variant=config.temporal_conv_variant,
        )
        self.channel_readout = ChannelReadout(config.branch_channels)

        self.final_pool = None
        output_r = self.fusion_r
        if config.final_head_pooling:
            if self.fusion_r < 1:
                raise ValueError("final_head_pooling requires a fusion resolution >= 1.")
            self.final_pool = icoCNN.PoolIco(self.fusion_r, 6, smooth_vertices=False)
            output_r = self.fusion_r - 1
        self.output_r = output_r
        self.clean_vertices = icoCNN.CleanVertices(output_r)
        ico_grid = torch.from_numpy(icoCNN.icosahedral_grid_coordinates(output_r)).float()
        ico_grid = ico_grid.permute(3, 0, 1, 2).contiguous()
        self.sam = at_modules.SoftArgMax(ico_grid.shape[1:], indexes=ico_grid, include_exp=True)

    @staticmethod
    def _count_params(module: Optional[nn.Module]) -> int:
        if module is None:
            return 0
        return sum(param.numel() for param in module.parameters())

    def count_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (param for param in self.parameters() if param.requires_grad)
        return sum(param.numel() for param in params)

    def expected_input_channels(self) -> int:
        return self.config.phat_in_channels + self.config.aux_in_channels

    def parameter_breakdown(self) -> dict[str, int]:
        breakdown = {
            "phat_stem": self._count_params(self.phat_branch.stem),
            "lms_stem": self._count_params(self.aux_branch.stem),
            "phat_residual": self._count_params(self.phat_branch.residual),
            "lms_residual": self._count_params(self.aux_branch.residual),
            "shared_attention": self._count_params(self.shared_attention),
            "fusion_blocks": sum(self._count_params(block) for block in self.fusion_blocks),
            "final_head": self._count_params(self.final_block),
            "channel_readout": self._count_params(self.channel_readout),
        }
        breakdown["total"] = sum(breakdown.values())
        return breakdown

    @staticmethod
    def _convico_mac_proxy(
        time_steps: int,
        charts: int,
        height: int,
        width: int,
        cin: int,
        cout: int,
        rin: int,
        rout: int,
        kernel_neighbors: int = 7,
    ) -> int:
        return int(time_steps) * int(charts) * int(height) * int(width) * int(cin) * int(cout) * int(rin) * int(rout) * int(kernel_neighbors)

    @staticmethod
    def _conv1d_mac_proxy(
        time_steps: int,
        charts: int,
        height: int,
        width: int,
        regions: int,
        cin: int,
        cout: int,
        kernel_size: int = 5,
        variant: str = "standard_1d",
    ) -> int:
        positions = int(charts) * int(height) * int(width) * int(regions)
        if variant == "standard_1d":
            per_step = int(cin) * int(cout) * int(kernel_size)
        elif variant == "depthwise_separable_1d":
            per_step = int(cin) * int(kernel_size) + int(cin) * int(cout)
        else:
            raise ValueError(f"Unsupported temporal conv variant {variant!r} for MAC proxy.")
        return int(time_steps) * int(positions) * per_step

    def mac_proxy(self, input_shape: tuple[int, int, int, int, int, int], kernel_neighbors: int = 7) -> dict[str, int]:
        batch, channels, time_steps, charts, height, width = input_shape
        expected_channels = self.expected_input_channels()
        if batch != 1:
            raise ValueError(f"MAC proxy expects batch size 1, got {input_shape}")
        if channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} feature channels for MAC proxy, got {input_shape}")

        branch_channels = self.config.branch_channels
        stem = self._convico_mac_proxy(time_steps, charts, height, width, 1, branch_channels, 1, 6, kernel_neighbors)
        residual_conv = self._convico_mac_proxy(
            time_steps,
            charts,
            height,
            width,
            branch_channels,
            branch_channels,
            6,
            6,
            kernel_neighbors,
        )
        fusion_height = max(height // 2, 1) if self.pre_fusion_pool is not None else height
        fusion_width = max(width // 2, 1) if self.pre_fusion_pool is not None else width
        temporal_block_conv = self._convico_mac_proxy(
            time_steps,
            charts,
            fusion_height,
            fusion_width,
            branch_channels,
            branch_channels,
            6,
            6,
            kernel_neighbors,
        )
        temporal_block_1d = self._conv1d_mac_proxy(
            time_steps,
            charts,
            fusion_height,
            fusion_width,
            6,
            branch_channels,
            branch_channels,
            variant=self.config.temporal_conv_variant,
        )
        final_conv = temporal_block_conv
        final_1d = self._conv1d_mac_proxy(
            time_steps,
            charts,
            fusion_height,
            fusion_width,
            6,
            branch_channels,
            branch_channels,
            variant=self.config.temporal_conv_variant,
        )
        channel_readout = int(time_steps) * 6 * int(charts) * int(fusion_height) * int(fusion_width) * branch_channels

        breakdown = {
            "phat_stem": stem,
            "lms_stem": stem,
            "phat_residual": 2 * residual_conv,
            "lms_residual": 2 * residual_conv,
            "shared_attention_conv1": residual_conv,
            "shared_attention_conv2": residual_conv,
            "fusion_block_conv": 4 * temporal_block_conv,
            "fusion_block_temporal": 4 * temporal_block_1d,
            "final_head_conv": final_conv,
            "final_head_temporal": final_1d,
            "channel_readout": channel_readout,
        }
        breakdown["total"] = sum(breakdown.values())
        return breakdown

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if x.ndim != 6:
            raise ValueError(f"Expected [B, C, T, 5, H, W], got {tuple(x.shape)}")
        if x.shape[3] != 5:
            raise ValueError(f"Expected 5 icosahedral charts, got shape {tuple(x.shape)}")

    @staticmethod
    def _branch_input(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(3)

    def _fuse_branch(self, direct: torch.Tensor, enhanced: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.shared_attention(enhanced)
        fused = direct + enhanced * weight
        return fused, weight

    def forward(self, x: torch.Tensor, return_attention: bool = False, return_debug: bool = False):
        self._validate_input(x)
        expected_channels = self.expected_input_channels()
        if x.shape[1] != expected_channels:
            raise ValueError(f"Expected {expected_channels} input channels, got shape {tuple(x.shape)}")
        debug: dict[str, torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor]] | None = {} if return_debug else None

        phat = self._branch_input(x[:, : self.config.phat_in_channels, ...].transpose(1, 2))
        aux = self._branch_input(x[:, self.config.phat_in_channels :, ...].transpose(1, 2))

        phat_direct, phat_enhanced = self.phat_branch(phat)
        aux_direct, aux_enhanced = self.aux_branch(aux)
        phat_fused, phat_weight = self._fuse_branch(phat_direct, phat_enhanced)
        aux_fused, aux_weight = self._fuse_branch(aux_direct, aux_enhanced)
        attention = {
            "phat": phat_weight,
            "aux": aux_weight,
            "lms": aux_weight,
        }
        if debug is not None:
            debug["phat_stem"] = phat_direct
            debug["phat_enhanced"] = phat_enhanced
            debug["phat_fused"] = phat_fused
            debug["lms_stem"] = aux_direct
            debug["lms_enhanced"] = aux_enhanced
            debug["lms_fused"] = aux_fused

        fused = phat_fused + aux_fused
        if debug is not None:
            debug["post_second_fusion"] = fused
        if self.pre_fusion_pool is not None:
            fused = self.pre_fusion_pool(fused)
        if debug is not None:
            debug["fusion_feature"] = fused

        fusion_head_blocks: list[torch.Tensor] = []
        for block in self.fusion_blocks:
            fused = block(fused)
            if debug is not None:
                fusion_head_blocks.append(fused)
        if debug is not None:
            debug["fusion_head_blocks"] = fusion_head_blocks
        logits = self.final_block(fused)
        if debug is not None:
            debug["final_head_logits"] = logits
        logits = self.channel_readout(logits)
        if debug is not None:
            debug["channel_readout_logits"] = logits

        if self.final_pool is not None:
            logits = self.final_pool(logits)
        if debug is not None:
            debug["post_final_pool_logits"] = logits

        logits = logits.squeeze(2)
        logits = logits.max(dim=2).values
        logits = self.clean_vertices(logits)
        coords = self.sam(logits)
        if debug is not None:
            debug["attention"] = attention
            debug["softargmax_input"] = logits

        if return_debug:
            return coords, debug
        if return_attention:
            return coords, attention
        return coords
