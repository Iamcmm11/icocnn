"""Lightweight MABA modules for DOA map refinement."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import acousticTrackingModels as at_models


class MABATemporalRefiner(nn.Module):
    """Map-level temporal refiner with lightweight selective state updates."""

    def __init__(
        self,
        charts=5,
        height=2,
        width=4,
        d_model=64,
        state_dim=16,
        conv_kernel=3,
        dropout=0.1,
        use_residual=True,
        use_gate=True,
        use_state=True,
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
            # Fixed, learnable forget factor used only when dynamic gate is disabled.
            self.alpha_logit = nn.Parameter(torch.zeros(self.state_dim))
        else:
            self.alpha_logit = None

    def forward(self, maps):
        if maps.dim() != 5:
            raise ValueError("Expected maps with shape (B, T, charts, H, W)")
        bsz, tlen, charts, height, width = maps.shape
        if charts * height * width != self.map_size:
            raise ValueError(
                "Input map size mismatch: expected {} elements, got {}".format(
                    self.map_size, charts * height * width
                )
            )

        x = maps.reshape(bsz, tlen, self.map_size)
        x = self.in_proj(x)

        # Causal depthwise temporal mixing with left padding only.
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
        delta = self.out_proj(z_refined).reshape(bsz, tlen, charts, height, width)
        return maps + delta if self.use_residual else delta

    def mac_proxy(self, time_steps):
        """Rough MAC proxy for reporting and model comparison."""
        p = self.map_size
        d = self.d_model
        s = self.state_dim
        k = self.conv_kernel
        gate_term = 2 * d * s if self.use_gate else d * s
        per_step = p * d + d * k + gate_term + s + s * d + d * p
        return int(time_steps) * int(per_step)


class MABAChannelTemporalBlock(nn.Module):
    """MABA-style temporal block that matches a Conv1d-like (N, C, T) interface."""

    def __init__(
        self,
        channels,
        d_model=32,
        state_dim=16,
        conv_kernel=3,
        dropout=0.1,
        use_residual=True,
        use_gate=True,
        use_state=True,
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

    def forward(self, x):
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


class IcoTempCNNWithMABA(at_models.IcoTempCNN):
    """IcoTempCNN + map-level MABA before SoftArgMax."""

    def __init__(
        self,
        r,
        C,
        Cin=1,
        smooth_vertices=True,
        maba_d_model=64,
        maba_state_dim=16,
        maba_conv_kernel=3,
        dropout=0.1,
        use_residual=True,
        use_gate=True,
        use_state=True,
    ):
        super().__init__(r, C, Cin=Cin, smooth_vertices=smooth_vertices)
        charts, height, width = self.sam.input_shape
        self.maba = MABATemporalRefiner(
            charts=charts,
            height=height,
            width=width,
            d_model=maba_d_model,
            state_dim=maba_state_dim,
            conv_kernel=maba_conv_kernel,
            dropout=dropout,
            use_residual=use_residual,
            use_gate=use_gate,
            use_state=use_state,
        )

    def forward(self, x, return_maps=False):
        maps_before = self.apply_cnn(x)
        maps_after = self.maba(maps_before)
        maps_clean = self.clean_vertices(maps_after)
        coords = self.sam(maps_clean)
        if return_maps:
            maps_dict = {
                "maps_before": maps_before,
                "maps_after": maps_after,
                "maps_clean": maps_clean,
            }
            return coords, maps_dict
        return coords


class IcoTempCNNReplaceTemporalMABA(at_models.IcoTempCNN):
    """IcoTempCNN with its first temporal Conv1d blocks replaced by MABA blocks."""

    def __init__(
        self,
        r,
        C,
        Cin=1,
        smooth_vertices=True,
        replace_d_model=32,
        replace_state_dim=16,
        replace_conv_kernel=3,
        dropout=0.1,
        use_residual=True,
    ):
        super().__init__(r, C, Cin=Cin, smooth_vertices=smooth_vertices)
        replaced_blocks = []
        for _ in range(len(self.temp_cnn) - 1):
            replaced_blocks.append(
                MABAChannelTemporalBlock(
                    channels=C,
                    d_model=replace_d_model,
                    state_dim=replace_state_dim,
                    conv_kernel=replace_conv_kernel,
                    dropout=dropout,
                    use_residual=use_residual,
                    use_gate=True,
                    use_state=True,
                )
            )
        replaced_blocks.append(self.temp_cnn[-1])
        self.temp_cnn = nn.ModuleList(replaced_blocks)
