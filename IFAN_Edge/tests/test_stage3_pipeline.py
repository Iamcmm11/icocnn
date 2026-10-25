from __future__ import annotations

import csv
import py_compile
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ifan_edge.eval.stage3 import (
    compute_prediction_details,
    resolve_librispeech_split,
    resolve_stage3_scenario,
    select_model_inputs,
)
from ifan_edge.features import SRPLMSIcoMap, SRPPHATIcoMapAdapter
from ifan_edge.models import IFANModel, IFANModelConfig, MapMABATemporalConfig, PAPER_IFAN_PARAM_TARGET
from ifan_edge.pruning import SAFLitePruner, iter_saf_lite_targets
from ifan_edge.training import IFANTrainingConfig, IFANTrainingPipeline
from scripts.assess_stage3_readiness import assess_readiness
from scripts.audit_stage3_protocol import build_protocol_rows
from utils import sph2cart


def test_stage3_sources_parse() -> None:
    targets = (
        PROJECT_ROOT / "ifan_edge" / "eval" / "stage3.py",
        PROJECT_ROOT / "ifan_edge" / "pruning" / "saf_lite.py",
        PROJECT_ROOT / "ifan_edge" / "training" / "pipeline.py",
        PROJECT_ROOT / "scripts" / "train_stage3_ifan.py",
        PROJECT_ROOT / "scripts" / "compare_stage3_baseline.py",
        PROJECT_ROOT / "scripts" / "compare_stage3_runs.py",
        PROJECT_ROOT / "scripts" / "evaluate_stage3_simulated.py",
        PROJECT_ROOT / "scripts" / "evaluate_stage3_locata.py",
        PROJECT_ROOT / "scripts" / "audit_stage3_protocol.py",
        PROJECT_ROOT / "scripts" / "assess_stage3_readiness.py",
        PROJECT_ROOT / "scripts" / "compare_stage3_lms_backends.py",
        PROJECT_ROOT / "scripts" / "compare_stage3_phat_variants.py",
        PROJECT_ROOT / "scripts" / "analyze_stage3_scene.py",
        PROJECT_ROOT / "scripts" / "diagnose_stage3_lms_peak.py",
    )
    for target in targets:
        py_compile.compile(str(target), doraise=True)


def test_stage3_librispeech_path_resolution(tmp_path: Path) -> None:
    direct_root = tmp_path / "direct"
    nested_root = tmp_path / "nested"
    (direct_root / "train-clean-100").mkdir(parents=True)
    (nested_root / "LibriSpeech" / "train-clean-100").mkdir(parents=True)

    assert resolve_librispeech_split(direct_root, "train-clean-100") == direct_root / "train-clean-100"
    assert resolve_librispeech_split(direct_root / "train-clean-100", "train-clean-100") == direct_root / "train-clean-100"
    assert resolve_librispeech_split(nested_root, "train-clean-100") == nested_root / "LibriSpeech" / "train-clean-100"


def test_stage3_default_contract_matches_locked_mainline() -> None:
    config = IFANTrainingConfig.from_toml(PROJECT_ROOT / "configs" / "stage3_default.toml")

    assert config.epochs == 40
    assert config.phase1_epochs == 20
    assert config.lms_backend == "frequency_block"
    assert config.lms_update_mode == "trajectory_tracking"
    assert config.lms_normalized is False
    assert config.srp_variant == "paper_original"
    assert config.temporal_conv_variant == "standard_1d"
    assert config.temporal_module == "conv"
    assert config.pre_fusion_pooling is True

    contract = config.experiment_contract()
    assert contract["experiment_role"] == "mainline_baseline"
    assert contract["pre_fusion_pooling"] is True
    assert contract["lightweight_gate"]["ready_delta_deg"] == pytest.approx(0.3)


def test_stage3_protocol_audit_marks_epoch_budget_as_gap_for_40_epoch_mainline() -> None:
    config = IFANTrainingConfig.from_toml(PROJECT_ROOT / "configs" / "stage3_default.toml")
    rows = build_protocol_rows(config, locata_report=None)
    row_map = {row["item"]: row for row in rows}

    assert row_map["Training schedule"]["status"] == "gap"
    assert row_map["LMS backend implementation"]["status"] == "context"
    assert row_map["Sampling rate"]["status"] == "match"


def test_stage3_ifan_paper_forward_variants() -> None:
    variants = (
        IFANModelConfig(
            final_head_pooling=False,
        ),
        IFANModelConfig(
            final_head_pooling=True,
        ),
    )

    for config in variants:
        model = IFANModel(config)
        channels = model.expected_input_channels()
        x = torch.randn(2, channels, 6, 5, 4, 8)
        y = model(x)
        assert y.shape == (2, 6, 3)
        assert torch.isfinite(y).all()


def test_stage3_ifan_debug_shapes_and_parameter_target() -> None:
    model = IFANModel(IFANModelConfig(r=2, final_head_pooling=False))
    x = torch.randn(2, model.expected_input_channels(), 6, 5, 4, 8)

    coords, debug = model(x, return_debug=True)

    assert coords.shape == (2, 6, 3)
    assert debug["phat_stem"].shape == (2, 6, 16, 6, 5, 4, 8)
    assert debug["phat_enhanced"].shape == (2, 6, 16, 6, 5, 4, 8)
    assert debug["phat_fused"].shape == (2, 6, 16, 6, 5, 4, 8)
    assert debug["lms_stem"].shape == (2, 6, 16, 6, 5, 4, 8)
    assert debug["lms_enhanced"].shape == (2, 6, 16, 6, 5, 4, 8)
    assert debug["lms_fused"].shape == (2, 6, 16, 6, 5, 4, 8)
    assert debug["post_second_fusion"].shape == (2, 6, 16, 6, 5, 4, 8)
    assert debug["fusion_feature"].shape == (2, 6, 16, 6, 5, 2, 4)
    fusion_head_blocks = debug["fusion_head_blocks"]
    assert isinstance(fusion_head_blocks, list)
    assert len(fusion_head_blocks) == 4
    assert all(block.shape == (2, 6, 16, 6, 5, 2, 4) for block in fusion_head_blocks)
    assert debug["final_head_logits"].shape == (2, 6, 16, 6, 5, 2, 4)
    assert debug["channel_readout_logits"].shape == (2, 6, 1, 6, 5, 2, 4)
    assert debug["post_final_pool_logits"].shape == (2, 6, 1, 6, 5, 2, 4)
    assert debug["attention"]["phat"].shape == (2, 6, 16, 6, 5, 4, 8)
    assert debug["attention"]["lms"].shape == (2, 6, 16, 6, 5, 4, 8)
    assert abs(model.count_parameters(trainable_only=True) - PAPER_IFAN_PARAM_TARGET) <= 64


def test_stage3_ifan_can_keep_fusion_at_input_resolution_without_poolico() -> None:
    pooled = IFANModel(IFANModelConfig(r=2, final_head_pooling=False))
    model = IFANModel(IFANModelConfig(r=2, pre_fusion_pooling=False, final_head_pooling=False))
    x = torch.randn(2, model.expected_input_channels(), 6, 5, 4, 8)

    coords, debug = model(x, return_debug=True)

    assert coords.shape == (2, 6, 3)
    assert model.pre_fusion_pool is None
    assert model.fusion_r == 2
    assert model.output_r == 2
    assert debug["post_second_fusion"].shape == (2, 6, 16, 6, 5, 4, 8)
    assert debug["fusion_feature"].shape == (2, 6, 16, 6, 5, 4, 8)
    assert all(block.shape == (2, 6, 16, 6, 5, 4, 8) for block in debug["fusion_head_blocks"])
    assert debug["channel_readout_logits"].shape == (2, 6, 1, 6, 5, 4, 8)
    assert debug["softargmax_input"].shape == (2, 6, 5, 4, 8)
    assert model.count_parameters(trainable_only=True) == pooled.count_parameters(trainable_only=True)
    assert model.mac_proxy((1, 2, 6, 5, 4, 8))["total"] > pooled.mac_proxy((1, 2, 6, 5, 4, 8))["total"]


def test_stage3_ifan_backward_produces_finite_nonzero_gradients() -> None:
    model = IFANModel(IFANModelConfig(r=2))
    x = torch.randn(2, model.expected_input_channels(), 6, 5, 4, 8)
    target = torch.randn(2, 6, 3)

    output = model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    assert torch.isfinite(output).all()
    assert bool(torch.isfinite(loss).item())
    core_modules = (
        model.phat_branch.stem,
        model.aux_branch.stem,
        model.shared_attention,
        model.fusion_blocks[0],
        model.final_block,
    )
    for module in core_modules:
        grads = [param.grad for param in module.parameters() if param.grad is not None]
        assert grads
        assert all(torch.isfinite(grad).all() for grad in grads)
        assert any(torch.count_nonzero(grad).item() > 0 for grad in grads)


def test_stage3_ifan_lightweight_temporal_variant_and_channel_scaling_work() -> None:
    baseline = IFANModel(IFANModelConfig(r=2))
    config = IFANModelConfig(r=2, branch_channels=8, temporal_conv_variant="depthwise_separable_1d")
    model = IFANModel(config)
    x = torch.randn(2, model.expected_input_channels(), 6, 5, 4, 8)
    target = torch.randn(2, 6, 3)

    output = model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    assert output.shape == (2, 6, 3)
    assert torch.isfinite(output).all()
    assert bool(torch.isfinite(loss).item())
    assert model.count_parameters(trainable_only=True) < baseline.count_parameters(trainable_only=True)
    assert model.mac_proxy((1, 2, 6, 5, 4, 8))["total"] < baseline.mac_proxy((1, 2, 6, 5, 4, 8))["total"]
    grads = [param.grad for param in model.final_block.temporal.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert any(torch.count_nonzero(grad).item() > 0 for grad in grads)


def test_stage3_saf_lite_prunes_only_c16_convico_blocks_and_keeps_fixed_zero_weights() -> None:
    torch.manual_seed(0)
    model = IFANModel(IFANModelConfig(r=2, branch_channels=16))
    target_names = [name for name, _module in iter_saf_lite_targets(model, target_channels=16, block_size=8)]

    pruner = SAFLitePruner.from_model(model, keep_per_block=3, block_size=8, target_channels=16)
    pruner.apply(model)
    pruner.register_gradient_hooks(model)

    assert pruner.masks
    assert all("stem" not in name for name in pruner.masks)
    assert sorted(pruner.masks) == sorted(target_names)
    for name, mask in pruner.masks.items():
        assert mask.shape == dict(model.named_modules())[name].weight.shape
        channel_mask = mask[:, :, 0, 0]
        for co in range(channel_mask.shape[0]):
            for block_start in range(0, channel_mask.shape[1], 8):
                assert int(channel_mask[co, block_start : block_start + 8].sum().item()) == 3

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(1, model.expected_input_channels(), 6, 5, 4, 8)
    loss = model(x).square().mean()
    loss.backward()
    pruner.optimizer_step(model, optimizer)

    for name, mask in pruner.masks.items():
        weight = dict(model.named_modules())[name].weight.detach().cpu()
        assert torch.count_nonzero(weight[mask == 0]).item() == 0
        if dict(model.named_modules())[name].weight.grad is not None:
            grad = dict(model.named_modules())[name].weight.grad.detach().cpu()
            assert torch.count_nonzero(grad[mask == 0]).item() == 0


def test_stage3_saf_lite_checkpoint_reload_preserves_masked_zeros(tmp_path: Path) -> None:
    model = IFANModel(IFANModelConfig(r=2, branch_channels=16))
    pruner = SAFLitePruner.from_model(model, keep_per_block=4, block_size=8, target_channels=16)
    pruner.apply(model)
    state_path = tmp_path / "pruned.pt"
    torch.save(model.state_dict(), state_path)

    reloaded = IFANModel(IFANModelConfig(r=2, branch_channels=16))
    reloaded.load_state_dict(torch.load(state_path, map_location="cpu"))

    for name, mask in pruner.masks.items():
        weight = dict(reloaded.named_modules())[name].weight.detach().cpu()
        assert torch.count_nonzero(weight[mask == 0]).item() == 0

    summary = pruner.summary(model, time_steps=6, charts=5)
    assert summary["pruned_layer_count"] == len(pruner.masks)
    assert summary["ico_conv_mac_keep_ratio"] == pytest.approx(0.5)


def test_stage3_ifan_map_maba_refiner_preserves_output_contract_and_adds_mac() -> None:
    baseline = IFANModel(IFANModelConfig(r=2, branch_channels=8))
    config = IFANModelConfig(
        r=2,
        branch_channels=8,
        map_refiner="maba",
        map_maba=MapMABATemporalConfig(d_model=16, state_dim=8, conv_kernel=3),
    )
    model = IFANModel(config)
    x = torch.randn(2, model.expected_input_channels(), 6, 5, 4, 8)

    coords, debug = model(x, return_debug=True)

    assert coords.shape == (2, 6, 3)
    assert debug["map_refiner"] == "maba"
    assert debug["map_refined_logits"].shape == (2, 6, 5, 2, 4)
    assert model.map_refiner is not None
    assert model.parameter_breakdown()["map_refiner"] > 0
    assert model.mac_proxy((1, 2, 6, 5, 4, 8))["map_refiner"] > 0
    assert model.mac_proxy((1, 2, 6, 5, 4, 8))["total"] > baseline.mac_proxy((1, 2, 6, 5, 4, 8))["total"]


def test_stage3_input_ablation_modes_zero_expected_branch() -> None:
    maps = torch.arange(2 * 2 * 3 * 5 * 4 * 8, dtype=torch.float32).reshape(2, 2, 3, 5, 4, 8)
    config = IFANModelConfig()

    phat_only = select_model_inputs(maps, config, "phat_only")
    lms_only = select_model_inputs(maps, config, "lms_only")

    assert torch.allclose(phat_only[:, 0:1, ...], maps[:, 0:1, ...])
    assert torch.count_nonzero(phat_only[:, 1:2, ...]).item() == 0
    assert torch.count_nonzero(lms_only[:, 0:1, ...]).item() == 0
    assert torch.allclose(lms_only[:, 1:2, ...], maps[:, 1:2, ...])


def test_stage3_phat_variants_match_output_contract_and_cache_metadata() -> None:
    torch.manual_seed(0)
    rn = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
        ],
        dtype=np.float32,
    )
    mic = torch.randn(2, 1, 3, 3, 64)
    outputs = {}
    profiles = {}

    for variant in ("paper_original", "lc_reference", "lc_edge"):
        frontend = SRPPHATIcoMapAdapter(N=3, K=64, r=1, rn=rn, fs=16000, srp_variant=variant)
        maps = frontend(mic)
        profile = frontend.frontend_profile()
        outputs[variant] = maps
        profiles[variant] = profile

        assert maps.shape == (2, 1, 3, 5, 2, 4)
        assert maps.dtype == mic.dtype
        assert maps.device == mic.device
        assert torch.isfinite(maps).all()
        assert profile["srp_variant"] == variant
        assert profile["pair_count"] > 0
        assert profile["cache_table_bytes"] > 0
        assert profile["complexity_proxy"]["sample_reads_per_frame"] > 0

    ref = outputs["paper_original"].reshape(2, 3, -1).float()
    lc_ref = outputs["lc_reference"].reshape(2, 3, -1).float()
    lc_edge = outputs["lc_edge"].reshape(2, 3, -1).float()
    ref_cosine = torch.nn.functional.cosine_similarity(ref, lc_ref, dim=-1)
    edge_cosine = torch.nn.functional.cosine_similarity(lc_ref, lc_edge, dim=-1)
    ref_peak = ref.argmax(dim=-1)
    lc_ref_peak = lc_ref.argmax(dim=-1)
    lc_edge_peak = lc_edge.argmax(dim=-1)

    assert ref_cosine.mean().item() >= 0.99
    assert edge_cosine.mean().item() >= 0.99
    assert ((ref_peak - lc_ref_peak).abs() <= 1).float().mean().item() >= 0.95
    assert ((lc_ref_peak - lc_edge_peak).abs() <= 1).float().mean().item() >= 0.95
    assert profiles["lc_edge"]["pair_count"] == 3
    assert profiles["lc_edge"]["full_pair_count"] == 9
    assert profiles["lc_edge"]["unique_pairs_only"] is True
    assert profiles["lc_reference"]["pair_count"] == 9
    assert (
        profiles["lc_edge"]["complexity_proxy"]["sample_reads_per_frame"]
        < profiles["lc_reference"]["complexity_proxy"]["sample_reads_per_frame"]
    )


class _ExactDoaModel(nn.Module):
    def __init__(self, doa_batch: torch.Tensor):
        super().__init__()
        self.register_buffer("coords", sph2cart(doa_batch).contiguous())

    def forward(self, x):
        batch, _, time_steps, _, _, _ = x.shape
        coords = self.coords
        assert coords.shape[0] == batch
        assert coords.shape[1] == time_steps
        return coords


def test_stage3_compute_prediction_details_returns_zero_for_exact_predictions() -> None:
    doa_batch = torch.tensor(
        [
            [
                [0.50, -0.20],
                [0.55, -0.15],
                [0.60, -0.10],
                [0.65, -0.05],
                [0.70, 0.00],
                [0.75, 0.05],
                [0.80, 0.10],
            ]
        ],
        dtype=torch.float32,
    )
    inputs = torch.zeros(1, 2, doa_batch.shape[1], 5, 4, 8)
    model = _ExactDoaModel(doa_batch)

    details = compute_prediction_details(model, inputs, doa_batch)

    assert details["offset_frames"] == 5
    assert details["frame_errors_deg"].shape == (1, 2)
    assert torch.allclose(details["frame_errors_deg"], torch.zeros_like(details["frame_errors_deg"]), atol=1e-4)
    assert torch.allclose(details["trajectory_rmsae_deg"], torch.zeros_like(details["trajectory_rmsae_deg"]), atol=1e-4)
    assert details["rmsae_deg"] == pytest.approx(0.0, abs=1e-4)


def test_stage3_resolve_scenario_returns_scene_metadata() -> None:
    scenario = resolve_stage3_scenario("scene_2")

    assert scenario["name"] == "scene_2"
    assert scenario["snr_db"] == pytest.approx(30.0)
    assert scenario["t60_s"] == pytest.approx(0.8)


def test_stage3_readiness_gate_prefers_lightweighting_after_stable_locata_win() -> None:
    current_summary = {
        "best_val_rmsae_deg": 6.7,
        "baseline_compare": {
            "mean_rmsae_deg": {"delta": 0.10},
            "hard_scenarios_mean_rmsae_deg": {"delta": -0.05},
        },
    }
    previous_summary = {
        "best_val_rmsae_deg": 6.8,
        "baseline_compare": {
            "mean_rmsae_deg": {"delta": 0.25},
            "hard_scenarios_mean_rmsae_deg": {"delta": 0.10},
        },
    }
    current_locata = {
        "overall": {
            "ifan": {
                "with_silences_rmsae_deg": {"mean": 7.30},
                "without_silences_rmsae_deg": {"mean": 6.60},
            },
            "baseline": {
                "with_silences_rmsae_deg": {"mean": 7.70},
                "without_silences_rmsae_deg": {"mean": 7.10},
            },
            "delta_vs_baseline": {
                "with_silences_rmsae_deg": {"mean": -0.40},
                "without_silences_rmsae_deg": {"mean": -0.50},
            },
        },
        "per_task": {
            "task3": {
                "delta_vs_baseline": {
                    "with_silences_rmsae_deg": {"mean": 0.20},
                    "without_silences_rmsae_deg": {"mean": 0.25},
                }
            },
            "task5": {
                "delta_vs_baseline": {
                    "with_silences_rmsae_deg": {"mean": -0.10},
                    "without_silences_rmsae_deg": {"mean": 0.15},
                }
            },
        },
    }
    previous_locata = {
        "overall": {
            "ifan": {
                "with_silences_rmsae_deg": {"mean": 7.45},
                "without_silences_rmsae_deg": {"mean": 6.72},
            },
            "baseline": {
                "with_silences_rmsae_deg": {"mean": 7.70},
                "without_silences_rmsae_deg": {"mean": 7.10},
            },
            "delta_vs_baseline": {
                "with_silences_rmsae_deg": {"mean": -0.25},
                "without_silences_rmsae_deg": {"mean": -0.38},
            },
        },
        "per_task": {
            "task3": {
                "delta_vs_baseline": {
                    "with_silences_rmsae_deg": {"mean": 0.24},
                    "without_silences_rmsae_deg": {"mean": 0.28},
                }
            },
            "task5": {
                "delta_vs_baseline": {
                    "with_silences_rmsae_deg": {"mean": -0.05},
                    "without_silences_rmsae_deg": {"mean": 0.18},
                }
            },
        },
    }

    report = assess_readiness(
        current_summary,
        current_locata,
        previous_summary=previous_summary,
        previous_locata=previous_locata,
        improvement_threshold_deg=0.3,
        task_regression_tolerance_deg=0.3,
    )

    assert report["verdict"] == "ready_for_lightweighting"
    assert report["reasons"]["overall_locata_win"] is True
    assert report["reasons"]["task3_task5_stable"] is True
    assert report["reasons"]["diminishing_returns"] is True


class _DummyStage3Dataset:
    def __len__(self) -> int:
        return 2

    def get_batch(self, start: int, stop: int):
        return torch.zeros(stop - start, 1), [None] * (stop - start)


class _DummyStage3Preprocessor:
    def data_transformation(self, mic_sig_batch, acoustic_scene_batch):
        batch = len(acoustic_scene_batch)
        maps = torch.zeros(batch, 1, 1, 5, 4, 8)
        doa = torch.zeros(batch, 1, 2)
        return maps, doa


class _DummyStage3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        batch, _, time_steps, _, _, _ = x.shape
        return self.weight.expand(batch, time_steps, 3)


def test_stage3_train_epoch_uses_average_gradient_semantics() -> None:
    config = IFANTrainingConfig(
        device="cpu",
        batch_size_phase1=2,
        micro_batch_size_phase1=1,
    )
    pipeline = IFANTrainingPipeline(config)
    model = _DummyStage3Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    train_loss = pipeline.train_epoch(
        model=model,
        optimizer=optimizer,
        dataset=_DummyStage3Dataset(),
        preprocessor=_DummyStage3Preprocessor(),
        batch_size=2,
        micro_batch_size=1,
        epoch=1,
        total_epochs=1,
    )

    assert train_loss == pytest.approx(1.0 / 3.0)
    assert torch.isclose(model.weight.detach(), torch.tensor(2.0 / 3.0), atol=1e-6)


def test_stage3_lms_options_control_pairing_and_normalization() -> None:
    rn = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
        ],
        dtype=np.float32,
    )

    with_self = SRPLMSIcoMap(N=3, K=16, r=1, rn=rn, fs=16000)
    no_self_plain = SRPLMSIcoMap(
        N=3,
        K=16,
        r=1,
        rn=rn,
        fs=16000,
        normalized_lms=False,
        include_self_pairs=False,
    )

    assert with_self.pair_i.numel() == 9
    assert no_self_plain.pair_i.numel() == 6
    assert with_self.normalized_lms is True
    assert no_self_plain.normalized_lms is False


def test_stage3_lms_frequency_block_auto_fft_and_validation() -> None:
    rn = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
        ],
        dtype=np.float32,
    )

    auto_fft = SRPLMSIcoMap(
        N=3,
        K=64,
        r=1,
        rn=rn,
        fs=16000,
        lms_order=4,
        lms_backend="frequency_block",
        lms_block_size=8,
    )
    assert auto_fft.lms_fft_size == 16

    with pytest.raises(ValueError, match="lms_fft_size must be at least"):
        SRPLMSIcoMap(
            N=3,
            K=64,
            r=1,
            rn=rn,
            fs=16000,
            lms_order=4,
            lms_backend="frequency_block",
            lms_block_size=8,
            lms_fft_size=8,
        )


def test_stage3_lms_map_normalize_option_changes_output_scale() -> None:
    rn = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
        ],
        dtype=np.float32,
    )
    mic = torch.randn(1, 1, 1, 3, 16)

    with_norm = SRPLMSIcoMap(N=3, K=16, r=1, rn=rn, fs=16000, normalize=True)
    no_norm = SRPLMSIcoMap(N=3, K=16, r=1, rn=rn, fs=16000, normalize=False)

    maps_norm = with_norm(mic)
    maps_raw = no_norm(mic)

    assert maps_norm.shape == maps_raw.shape
    assert maps_norm.dtype == maps_raw.dtype
    assert torch.isfinite(maps_norm).all()
    assert torch.isfinite(maps_raw).all()
    assert not torch.allclose(maps_norm, maps_raw)


def test_stage3_lms_peak_proximity_mode_changes_maps() -> None:
    rn = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
        ],
        dtype=np.float32,
    )
    mic = torch.randn(1, 1, 1, 3, 16)

    tau_sample = SRPLMSIcoMap(N=3, K=16, r=1, rn=rn, fs=16000, map_mode="tau_sample")
    peak_proximity = SRPLMSIcoMap(N=3, K=16, r=1, rn=rn, fs=16000, map_mode="peak_proximity", peak_sigma=2.0)

    maps_tau = tau_sample(mic)
    maps_peak = peak_proximity(mic)

    assert maps_tau.shape == maps_peak.shape
    assert maps_tau.dtype == maps_peak.dtype
    assert torch.isfinite(maps_tau).all()
    assert torch.isfinite(maps_peak).all()
    assert not torch.allclose(maps_tau, maps_peak)


@pytest.mark.parametrize("update_mode", ("frame_reset", "trajectory_tracking"))
@pytest.mark.parametrize("map_mode", ("tau_sample", "peak_proximity"))
def test_stage3_lms_frequency_block_matches_reference_backend(update_mode: str, map_mode: str) -> None:
    torch.manual_seed(0)
    rn = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
        ],
        dtype=np.float32,
    )
    mic = torch.randn(2, 1, 3, 3, 64)

    reference = SRPLMSIcoMap(
        N=3,
        K=64,
        r=1,
        rn=rn,
        fs=16000,
        lms_order=4,
        map_mode=map_mode,
        update_mode=update_mode,
    )
    frequency_block = SRPLMSIcoMap(
        N=3,
        K=64,
        r=1,
        rn=rn,
        fs=16000,
        lms_order=4,
        map_mode=map_mode,
        update_mode=update_mode,
        lms_backend="frequency_block",
        lms_block_size=8,
        lms_fft_size=16,
    )

    with torch.no_grad():
        ref_maps = reference(mic)
        freq_maps = frequency_block(mic)

    assert ref_maps.shape == freq_maps.shape
    assert ref_maps.dtype == freq_maps.dtype
    assert ref_maps.device == freq_maps.device
    assert torch.isfinite(freq_maps).all()

    ref_flat = ref_maps.reshape(ref_maps.shape[:-3] + (-1,)).float()
    freq_flat = freq_maps.reshape(freq_maps.shape[:-3] + (-1,)).float()
    cosine = torch.nn.functional.cosine_similarity(ref_flat, freq_flat, dim=-1)
    assert cosine.mean().item() >= 0.99

    ref_peak = ref_flat.argmax(dim=-1)
    freq_peak = freq_flat.argmax(dim=-1)
    within_one = ((ref_peak - freq_peak).abs() <= 1).float().mean().item()
    assert within_one >= 0.95


def test_stage3_training_config_parses_frequency_block_options(tmp_path: Path) -> None:
    config_path = tmp_path / "stage3.toml"
    config_path.write_text(
        """
[data]
lms_backend = "frequency_block"
lms_block_size = 128
lms_fft_size = 512
""".strip(),
        encoding="utf-8",
    )

    config = IFANTrainingConfig.from_toml(config_path)

    assert config.lms_backend == "frequency_block"
    assert config.lms_block_size == 128
    assert config.lms_fft_size == 512


def test_stage3_train_script_parser_accepts_frequency_block_options() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "train_stage3_ifan.py"))
    parser = module["build_parser"]()
    args = parser.parse_args(
        [
            "--lms-backend",
            "frequency_block",
            "--lms-block-size",
            "128",
            "--lms-fft-size",
            "512",
        ]
    )

    assert args.lms_backend == "frequency_block"
    assert args.lms_block_size == 128
    assert args.lms_fft_size == 512


def test_stage3_train_script_parser_accepts_phat_and_lightweight_model_overrides() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "train_stage3_ifan.py"))
    parser = module["build_parser"]()
    args = parser.parse_args(
        [
            "--srp-variant",
            "lc_edge",
            "--phat-sinc-half-width",
            "2",
            "--branch-channels",
            "8",
            "--temporal-conv-variant",
            "depthwise_separable_1d",
        ]
    )

    assert args.srp_variant == "lc_edge"
    assert args.phat_sinc_half_width == 2
    assert args.branch_channels == 8
    assert args.temporal_conv_variant == "depthwise_separable_1d"


def test_stage3_train_script_parser_accepts_saf_lite_options() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "train_stage3_ifan.py"))
    parser = module["build_parser"]()
    args = parser.parse_args(
        [
            "--ifan-init-checkpoint",
            "IFAN_Edge/outputs/stage3/checkpoints/best_rmsae.pt",
            "--saf-lite",
            "--saf-lite-keep-per-8",
            "3",
        ]
    )

    assert args.ifan_init_checkpoint == "IFAN_Edge/outputs/stage3/checkpoints/best_rmsae.pt"
    assert args.saf_lite is True
    assert args.saf_lite_keep_per_8 == 3


def test_stage3_train_script_parser_accepts_map_maba_options() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "train_stage3_ifan.py"))
    parser = module["build_parser"]()
    args = parser.parse_args(
        [
            "--map-refiner",
            "maba",
            "--map-maba-d-model",
            "16",
            "--map-maba-state-dim",
            "8",
            "--map-maba-conv-kernel",
            "3",
            "--map-maba-dropout",
            "0.0",
            "--map-maba-no-gate",
        ]
    )

    assert args.map_refiner == "maba"
    assert args.map_maba_d_model == 16
    assert args.map_maba_state_dim == 8
    assert args.map_maba_conv_kernel == 3
    assert args.map_maba_dropout == pytest.approx(0.0)
    assert args.map_maba_no_gate is True


def test_stage3_train_script_parser_accepts_schedule_overrides() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "train_stage3_ifan.py"))
    parser = module["build_parser"]()
    args = parser.parse_args(
        [
            "--seed",
            "7",
            "--phase1-epochs",
            "30",
            "--train-snr-min-phase2",
            "5",
            "--train-snr-max-phase2",
            "15",
            "--train-t60-min",
            "0.3",
            "--train-t60-max",
            "1.1",
        ]
    )

    assert args.seed == 7
    assert args.phase1_epochs == 30
    assert args.train_snr_min_phase2 == pytest.approx(5.0)
    assert args.train_snr_max_phase2 == pytest.approx(15.0)
    assert args.train_t60_min == pytest.approx(0.3)
    assert args.train_t60_max == pytest.approx(1.1)


def test_stage3_lms_backend_compare_parser_accepts_mode_scenario_and_overrides() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "compare_stage3_lms_backends.py"))
    parser = module["build_parser"]()
    args = parser.parse_args(
        [
            "--mode",
            "scenario",
            "--scenario",
            "scene_4",
            "--size",
            "3",
            "--batch-size",
            "1",
            "--trajectory-seconds",
            "5",
            "--device",
            "cpu",
            "--lms-block-size",
            "128",
            "--lms-fft-size",
            "512",
        ]
    )

    assert args.mode == "scenario"
    assert args.scenario == "scene_4"
    assert args.size == 3
    assert args.batch_size == 1
    assert args.trajectory_seconds == 5
    assert args.device == "cpu"
    assert args.lms_block_size == 128
    assert args.lms_fft_size == 512


def test_stage3_lms_backend_compare_parser_accepts_scenario_suite_mode() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "compare_stage3_lms_backends.py"))
    parser = module["build_parser"]()
    args = parser.parse_args(
        [
            "--mode",
            "scenario_suite",
            "--size",
            "2",
            "--batch-size",
            "1",
        ]
    )

    assert args.mode == "scenario_suite"
    assert args.size == 2
    assert args.batch_size == 1


def test_stage3_phat_variant_compare_parser_accepts_variants_and_repeats() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "compare_stage3_phat_variants.py"))
    parser = module["build_parser"]()
    args = parser.parse_args(
        [
            "--mode",
            "scenario",
            "--scenario",
            "scene_4",
            "--size",
            "3",
            "--batch-size",
            "1",
            "--repeats",
            "5",
            "--lms-backend",
            "time_reference",
            "--phat-sinc-half-width",
            "2",
            "--variant",
            "paper_original",
            "--variant",
            "lc_edge",
        ]
    )

    assert args.mode == "scenario"
    assert args.scenario == "scene_4"
    assert args.size == 3
    assert args.batch_size == 1
    assert args.repeats == 5
    assert args.lms_backend == "time_reference"
    assert args.phat_sinc_half_width == 2
    assert args.variant == ["paper_original", "lc_edge"]


def test_stage3_simulated_eval_parser_accepts_multi_checkpoint_and_seeds() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "evaluate_stage3_simulated.py"))
    parser = module["build_parser"]()
    args = parser.parse_args(
        [
            "--checkpoint",
            "ckpt_a.pt",
            "--checkpoint",
            "ckpt_b.pt",
            "--label",
            "best",
            "--label",
            "last",
            "--validation-size",
            "128",
            "--scenario-eval-size",
            "64",
            "--seeds",
            "42",
            "43",
            "44",
        ]
    )

    assert args.checkpoint == ["ckpt_a.pt", "ckpt_b.pt"]
    assert args.label == ["best", "last"]
    assert args.validation_size == 128
    assert args.scenario_eval_size == 64
    assert args.seeds == [42, 43, 44]


def test_stage3_simulated_eval_aggregate_reports_mean_and_std() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "evaluate_stage3_simulated.py"))
    aggregate = module["aggregate_simulated_runs"]
    runs = [
        {
            "validation": {"loss": 0.1, "rmsae_deg": 6.0},
            "baseline_compare": {
                "mean_rmsae_deg": {"ifan": 8.0, "baseline": 7.5, "delta": 0.5},
                "hard_scenarios_mean_rmsae_deg": {"ifan": 9.0, "baseline": 8.5, "delta": 0.5},
                "scenarios": [
                    {
                        "name": "scene_2",
                        "snr_db": 30.0,
                        "t60_s": 0.8,
                        "ifan": {"loss": 0.2, "rmsae_deg": 9.0},
                        "baseline": {"loss": 0.15, "rmsae_deg": 8.0},
                        "rmsae_delta_deg": 1.0,
                    }
                ],
            },
        },
        {
            "validation": {"loss": 0.2, "rmsae_deg": 8.0},
            "baseline_compare": {
                "mean_rmsae_deg": {"ifan": 7.0, "baseline": 7.0, "delta": 0.0},
                "hard_scenarios_mean_rmsae_deg": {"ifan": 8.0, "baseline": 8.5, "delta": -0.5},
                "scenarios": [
                    {
                        "name": "scene_2",
                        "snr_db": 30.0,
                        "t60_s": 0.8,
                        "ifan": {"loss": 0.1, "rmsae_deg": 7.0},
                        "baseline": {"loss": 0.15, "rmsae_deg": 8.0},
                        "rmsae_delta_deg": -1.0,
                    }
                ],
            },
        },
    ]

    report = aggregate(runs)

    assert report["validation"]["rmsae_deg"]["mean"] == pytest.approx(7.0)
    assert report["mean_rmsae_deg"]["delta"]["mean"] == pytest.approx(0.25)
    assert report["scenarios"][0]["rmsae_delta_deg"]["std"] == pytest.approx(1.0)


def test_stage3_scene_analysis_parser_accepts_output_dir_and_size() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "analyze_stage3_scene.py"))
    parser = module["build_parser"]()
    args = parser.parse_args(
        [
            "--checkpoint",
            "best.pt",
            "--scenario",
            "scene_2",
            "--size",
            "64",
            "--output-dir",
            "outdir",
        ]
    )

    assert args.checkpoint == "best.pt"
    assert args.scenario == "scene_2"
    assert args.size == 64
    assert args.output_dir == "outdir"


def test_stage3_run_compare_parser_accepts_labels_and_output() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "compare_stage3_runs.py"))
    parser = module["build_parser"]()
    args = parser.parse_args(
        [
            "--before",
            "run_a",
            "--after",
            "run_b",
            "--before-label",
            "pool_off",
            "--after-label",
            "pool_on",
            "--output",
            "report.json",
        ]
    )

    assert args.before == "run_a"
    assert args.after == "run_b"
    assert args.before_label == "pool_off"
    assert args.after_label == "pool_on"
    assert args.output == "report.json"


def test_stage3_run_compare_classifies_hard_gain_with_easy_cost() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "compare_stage3_runs.py"))
    classify = module["classify_transition"]
    before = {
        "baseline_compare": {
            "mean_rmsae_deg": {"delta": 0.30},
            "hard_scenarios_mean_rmsae_deg": {"delta": 1.00},
            "scenarios": [
                {"name": "scene_1", "snr_db": 30.0, "t60_s": 0.2, "rmsae_delta_deg": -0.5},
                {"name": "scene_2", "snr_db": 30.0, "t60_s": 0.8, "rmsae_delta_deg": 0.0},
                {"name": "scene_3", "snr_db": 5.0, "t60_s": 0.8, "rmsae_delta_deg": 0.8},
                {"name": "scene_4", "snr_db": 5.0, "t60_s": 1.4, "rmsae_delta_deg": 1.2},
            ],
        }
    }
    after = {
        "baseline_compare": {
            "mean_rmsae_deg": {"delta": 0.45},
            "hard_scenarios_mean_rmsae_deg": {"delta": 0.70},
            "scenarios": [
                {"name": "scene_1", "snr_db": 30.0, "t60_s": 0.2, "rmsae_delta_deg": 0.1},
                {"name": "scene_2", "snr_db": 30.0, "t60_s": 0.8, "rmsae_delta_deg": 0.2},
                {"name": "scene_3", "snr_db": 5.0, "t60_s": 0.8, "rmsae_delta_deg": 0.5},
                {"name": "scene_4", "snr_db": 5.0, "t60_s": 1.4, "rmsae_delta_deg": 0.9},
            ],
        }
    }

    report = classify(before, after)

    assert report["classification"]["improves_hard_scenes"] is True
    assert report["classification"]["harms_easy_scenes"] is True
    assert report["classification"]["net_improves_overall"] is False
    assert report["classification"]["verdict"] == "hard_scene_gain_with_easy_scene_cost"


def test_stage3_run_compare_marks_identical_runs_as_no_material_change() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "compare_stage3_runs.py"))
    classify = module["classify_transition"]
    summary = {
        "baseline_compare": {
            "mean_rmsae_deg": {"delta": 0.30},
            "hard_scenarios_mean_rmsae_deg": {"delta": 0.90},
            "scenarios": [
                {"name": "scene_1", "snr_db": 30.0, "t60_s": 0.2, "rmsae_delta_deg": -0.4},
                {"name": "scene_2", "snr_db": 30.0, "t60_s": 0.8, "rmsae_delta_deg": 0.1},
                {"name": "scene_3", "snr_db": 5.0, "t60_s": 0.8, "rmsae_delta_deg": 0.7},
                {"name": "scene_4", "snr_db": 5.0, "t60_s": 1.4, "rmsae_delta_deg": 1.1},
            ],
        }
    }

    report = classify(summary, summary)

    assert report["classification"]["verdict"] == "no_material_change"


def test_stage3_locata_parser_accepts_subset_array_tasks_and_recording() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "evaluate_stage3_locata.py"))
    parser = module["build_parser"]()
    args = parser.parse_args(
        [
            "--checkpoint",
            "best.pt",
            "--subset",
            "eval",
            "--array",
            "benchmark2",
            "--tasks",
            "1",
            "3",
            "5",
            "--recording",
            "recording1",
            "--device",
            "cpu",
        ]
    )

    assert args.checkpoint == "best.pt"
    assert args.subset == "eval"
    assert args.array == "benchmark2"
    assert args.tasks == [1, 3, 5]
    assert args.recording == ["recording1"]
    assert args.device == "cpu"


def test_stage3_locata_normalize_tasks_rejects_non_single_source_tasks() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "evaluate_stage3_locata.py"))
    normalize = module["normalize_tasks"]

    assert normalize([1, 3, 5]) == (1, 3, 5)
    with pytest.raises(ValueError, match="Only single-source LOCATA tasks"):
        normalize([1, 2])


def test_stage3_locata_paper_reference_payload_contains_tables() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "evaluate_stage3_locata.py"))
    payload = module["paper_reference_payload"]((1, 3, 5), "eval", "benchmark2")

    assert payload["tables"]["with_silences"]["table"] == "Table III"
    assert payload["tables"]["without_silences"]["table"] == "Table IV"
    assert payload["tables"]["with_silences"]["reported_ifan_rmsae_deg"]["task1"] is None


def test_stage3_locata_markdown_summary_mentions_training_dataset() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "evaluate_stage3_locata.py"))
    render = module["markdown_summary"]
    report = {
        "checkpoint": "/tmp/best.pt",
        "subset": "eval",
        "array": "benchmark2",
        "tasks": [1, 3, 5],
        "overall": {
            "ifan": {
                "with_silences_rmsae_deg": {"mean": 4.0},
                "without_silences_rmsae_deg": {"mean": 3.0},
            },
            "baseline": {
                "with_silences_rmsae_deg": {"mean": 5.0},
                "without_silences_rmsae_deg": {"mean": 4.0},
            },
        },
        "per_task": {},
        "paper_reference": {
            "tables": {
                "with_silences": {"table": "Table III"},
                "without_silences": {"table": "Table IV"},
            }
        },
    }

    text = render(report)

    assert "LibriSpeech train-clean-100" in text
    assert "Table III" in text
    assert "Table IV" in text


def test_stage3_lms_trajectory_tracking_mode_keeps_cross_frame_state() -> None:
    rn = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
        ],
        dtype=np.float32,
    )
    frame = torch.randn(1, 1, 1, 3, 16)
    mic = frame.repeat(1, 1, 2, 1, 1)

    reset = SRPLMSIcoMap(N=3, K=16, r=1, rn=rn, fs=16000, normalized_lms=False, update_mode="frame_reset")
    tracking = SRPLMSIcoMap(N=3, K=16, r=1, rn=rn, fs=16000, normalized_lms=False, update_mode="trajectory_tracking")

    maps_reset = reset(mic)
    maps_tracking = tracking(mic)

    assert maps_reset.shape == maps_tracking.shape
    assert torch.allclose(maps_reset[:, :, 0, ...], maps_tracking[:, :, 0, ...], atol=1e-6)
    assert torch.allclose(maps_reset[:, :, 0, ...], maps_reset[:, :, 1, ...], atol=1e-6)
    assert not torch.allclose(maps_tracking[:, :, 0, ...], maps_tracking[:, :, 1, ...])


def test_stage3_smoke_training_outputs(tmp_path: Path) -> None:
    config = IFANTrainingConfig(
        output_root=str(tmp_path / "stage3_outputs"),
        output_suffix="pytest",
        device="cpu",
        epochs=2,
        phase1_epochs=20,
        train_dataset_size=1,
        validation_dataset_size=1,
        validation_batch_size=1,
        scenario_eval_size=1,
        scenario_eval_batch_size=1,
        trajectory_seconds=1,
        checkpoint_every=1,
        final_head_pooling=False,
    )

    summary = IFANTrainingPipeline(config).run()

    output_dir = Path(summary["output_dir"])
    history_path = Path(summary["history_path"])
    baseline_compare_path = Path(summary["baseline_compare_path"])
    best_checkpoint_path = Path(summary["best_checkpoint_path"])
    last_checkpoint_path = Path(summary["last_checkpoint_path"])

    assert output_dir.is_dir()
    assert history_path.is_file()
    assert baseline_compare_path.is_file()
    assert best_checkpoint_path.is_file()
    assert last_checkpoint_path.is_file()
    assert Path(output_dir / "summary.json").is_file()
    assert Path(output_dir / "resolved_config.json").is_file()

    with history_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert float(rows[0]["train_loss"]) >= 0.0
    assert float(rows[-1]["val_rmsae_deg"]) >= 0.0

    baseline_report = baseline_compare_path.read_text(encoding="utf-8")
    assert "scene_1" in baseline_report
    assert "scene_4" in baseline_report
    assert summary["model_topology"] == "paper_dual_mainline"
    assert summary["srp_variant"] == "paper_original"
    assert summary["temporal_conv_variant"] == "standard_1d"
    assert summary["baseline_compare"]["srp_variant"] == "paper_original"
    assert summary["frontend_profile"]["phat"]["srp_variant"] == "paper_original"
