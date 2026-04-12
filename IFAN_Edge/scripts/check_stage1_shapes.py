from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ifan_edge.bridges import at_dataset
from ifan_edge.features import DualFeatureIcoPreprocessor


def main() -> None:
    r = 2
    batch = 2
    frames = 3
    n_mics = 12
    k = 4096
    fs = 16000

    preprocessor = DualFeatureIcoPreprocessor(
        N=n_mics,
        K=k,
        r=r,
        rn=at_dataset.benchmark2_array_setup.mic_pos,
        fs=fs,
        apply_vad=False,
    )

    mic_sig_batch = np.random.randn(batch, frames, n_mics, k).astype(np.float32)
    dual_maps = preprocessor.data_transformation(mic_sig_batch=mic_sig_batch)
    phat_maps, lms_maps = preprocessor.split_features(dual_maps)

    print("PHAT shape:", tuple(phat_maps.shape))
    print("LMS shape:", tuple(lms_maps.shape))
    print("Dual shape:", tuple(dual_maps.shape))

    expected_single = (batch, 1, frames, 5, 4, 8)
    expected_dual = (batch, 2, frames, 5, 4, 8)

    assert tuple(phat_maps.shape) == expected_single, (phat_maps.shape, expected_single)
    assert tuple(lms_maps.shape) == expected_single, (lms_maps.shape, expected_single)
    assert tuple(dual_maps.shape) == expected_dual, (dual_maps.shape, expected_dual)
    assert np.isfinite(dual_maps.detach().cpu().numpy()).all()

    print("Stage-1 shape sanity check passed.")


if __name__ == "__main__":
    main()
