from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ifan_edge.models import IFANModel, IFANModelConfig, PAPER_IFAN_PARAM_TARGET
from scripts.profile_stage2_model import baseline_summary, ifan_summary


def test_icocnn_parameter_anchor() -> None:
    summary = baseline_summary()
    assert summary["trainable_params"] == 290_017
    assert summary["total_params"] == 290_017
    assert summary["matches_expected_anchor"] is True
    assert summary["paper_style_complexity"]["mac_proxy_total"] == 34_736_640
    assert summary["paper_style_complexity"]["flops_proxy_total"] == 69_473_280


def test_ifan_parameter_anchors() -> None:
    summary = ifan_summary(
        IFANModelConfig(
            r=2,
            phat_in_channels=1,
            aux_in_channels=1,
            final_head_pooling=False,
        )
    )

    assert abs(summary["trainable_params"] - PAPER_IFAN_PARAM_TARGET) <= 64
    assert summary["paper_style_complexity"]["mac_proxy_total"] == 17_621_760
    assert summary["paper_style_complexity"]["flops_proxy_total"] == 35_243_520


def test_ifan_breakdown_matches_model_total() -> None:
    model = IFANModel(IFANModelConfig())
    breakdown = model.parameter_breakdown()
    mac_proxy = model.mac_proxy((1, 2, 3, 5, 4, 8))

    assert breakdown["total"] == model.count_parameters(trainable_only=True)
    assert breakdown["shared_attention"] > 0
    assert breakdown["fusion_blocks"] > 0
    assert breakdown["final_head"] > 0
    assert mac_proxy["total"] == sum(value for key, value in mac_proxy.items() if key != "total")
