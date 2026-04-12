from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


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
