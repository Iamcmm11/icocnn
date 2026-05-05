from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..bridges import icoCNN


STAGE1_SCENARIOS = (
    {"name": "scene_1", "snr_db": 30.0, "t60_s": 0.2},
    {"name": "scene_2", "snr_db": 30.0, "t60_s": 0.8},
    {"name": "scene_3", "snr_db": 5.0, "t60_s": 0.8},
    {"name": "scene_4", "snr_db": 5.0, "t60_s": 1.4},
)


def _to_numpy(feature_map: np.ndarray | "torch.Tensor") -> np.ndarray:
    if hasattr(feature_map, "detach"):
        feature_map = feature_map.detach().cpu().numpy()
    return np.asarray(feature_map)


def _select_frame(feature_map: np.ndarray, frame_idx: int) -> np.ndarray:
    if feature_map.ndim == 6:
        return feature_map[0, 0, frame_idx]
    if feature_map.ndim == 5:
        return feature_map[0, frame_idx]
    if feature_map.ndim == 4:
        return feature_map[frame_idx]
    if feature_map.ndim == 3:
        return feature_map
    raise ValueError(f"Unsupported feature map shape: {feature_map.shape}")


def save_dual_feature_figure(
    phat_map,
    lms_map,
    output_path: str | Path,
    frame_idx: int = 0,
    title: str | None = None,
) -> Path:
    """Save a chart-by-chart view of PHAT and LMS icosahedral feature maps."""

    phat_frame = _select_frame(_to_numpy(phat_map), frame_idx)
    lms_frame = _select_frame(_to_numpy(lms_map), frame_idx)

    if phat_frame.shape != lms_frame.shape:
        raise ValueError(f"Feature shapes differ: {phat_frame.shape} vs {lms_frame.shape}")
    if phat_frame.ndim != 3 or phat_frame.shape[0] != 5:
        raise ValueError(f"Expected [5, H, W] chart layout, got {phat_frame.shape}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), constrained_layout=True)
    if title:
        fig.suptitle(title)

    for chart_idx in range(5):
        axes[0, chart_idx].imshow(phat_frame[chart_idx], aspect="auto", cmap="viridis")
        axes[0, chart_idx].set_title(f"PHAT chart {chart_idx}")
        axes[0, chart_idx].axis("off")

        axes[1, chart_idx].imshow(lms_frame[chart_idx], aspect="auto", cmap="magma")
        axes[1, chart_idx].set_title(f"LMS chart {chart_idx}")
        axes[1, chart_idx].axis("off")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _infer_r_from_frame(feature_frame: np.ndarray) -> int:
    if feature_frame.ndim != 3 or feature_frame.shape[0] != 5:
        raise ValueError(f"Expected [5, H, W] chart layout, got {feature_frame.shape}")

    height = feature_frame.shape[1]
    if height <= 0 or height & (height - 1):
        raise ValueError(f"Expected power-of-two height for icosahedral charts, got {height}")
    return int(math.log2(height))


@lru_cache(maxsize=8)
def _get_projector(r: int, res_theta: int, res_phi: int):
    return icoCNN.plots.SphProjector(r=r, res_theta=res_theta, res_phi=res_phi)


def _get_projection(feature_frame: np.ndarray, res_theta: int, res_phi: int) -> np.ndarray:
    r = _infer_r_from_frame(feature_frame)
    projector = _get_projector(r=r, res_theta=res_theta, res_phi=res_phi)
    return projector.get_projection(feature_frame)


def _plot_projection(
    ax,
    projection: np.ndarray,
    title: str,
    cmap: str = "plasma",
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    theta = np.linspace(0, 180, projection.shape[0])
    theta_step = theta[1] - theta[0]
    phi = np.linspace(-180, 180, projection.shape[1])
    phi_step = phi[1] - phi[0]
    im = ax.imshow(
        projection,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(phi[0] - phi_step / 2, phi[-1] + phi_step / 2, theta[-1] + theta_step / 2, theta[0] - theta_step / 2),
        aspect="auto",
    )
    ax.set_xlabel("Azimuth [$^\\circ$]")
    ax.set_ylabel("Polar angle [$^\\circ$]")
    ax.set_title(title)
    return im


def _projection_limits(
    projection: np.ndarray,
    adaptive_contrast: bool,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[float, float]:
    if not adaptive_contrast:
        return 0.0, 1.0

    vmin = float(np.percentile(projection, lower_percentile))
    vmax = float(np.percentile(projection, upper_percentile))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return float(projection.min()), float(projection.max())
    return vmin, vmax


def save_single_projection_figure(
    feature_map,
    output_path: str | Path,
    frame_idx: int = 0,
    title: str | None = None,
    cmap: str = "plasma",
    res_theta: int = 181,
    res_phi: int = 360,
    adaptive_contrast: bool = False,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> Path:
    """Save one spherical projection with angular axes."""

    feature_frame = _select_frame(_to_numpy(feature_map), frame_idx)
    projection = _get_projection(feature_frame, res_theta=res_theta, res_phi=res_phi)
    vmin, vmax = _projection_limits(
        projection,
        adaptive_contrast=adaptive_contrast,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    im = _plot_projection(ax, projection, title=title or "", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_dual_feature_projection_figure(
    phat_map,
    lms_map,
    output_path: str | Path,
    frame_idx: int = 0,
    title: str | None = None,
    cmap: str = "plasma",
    res_theta: int = 181,
    res_phi: int = 360,
    adaptive_contrast: bool = False,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> Path:
    """Save PHAT and LMS angular projections for side-by-side comparison."""

    phat_frame = _select_frame(_to_numpy(phat_map), frame_idx)
    lms_frame = _select_frame(_to_numpy(lms_map), frame_idx)

    if phat_frame.shape != lms_frame.shape:
        raise ValueError(f"Feature shapes differ: {phat_frame.shape} vs {lms_frame.shape}")

    phat_projection = _get_projection(phat_frame, res_theta=res_theta, res_phi=res_phi)
    lms_projection = _get_projection(lms_frame, res_theta=res_theta, res_phi=res_phi)
    phat_vmin, phat_vmax = _projection_limits(
        phat_projection,
        adaptive_contrast=adaptive_contrast,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    lms_vmin, lms_vmax = _projection_limits(
        lms_projection,
        adaptive_contrast=adaptive_contrast,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), constrained_layout=True)
    if title:
        fig.suptitle(title)

    im0 = _plot_projection(axes[0], phat_projection, title="SRP-PHAT", cmap=cmap, vmin=phat_vmin, vmax=phat_vmax)
    im1 = _plot_projection(axes[1], lms_projection, title="SRP-LMS", cmap=cmap, vmin=lms_vmin, vmax=lms_vmax)
    fig.colorbar(im1, ax=axes, fraction=0.035, pad=0.02)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
