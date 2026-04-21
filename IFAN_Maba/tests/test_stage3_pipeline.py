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

from ifan_maba.eval.stage3 import (
    compute_prediction_details,
    resolve_librispeech_split,
    resolve_stage3_scenario,
    select_model_inputs,
)
from ifan_maba.features import SRPLMSIcoMap
from ifan_maba.models import (
    IFANModel,
    IFANModelConfig,
    MABAChannelTemporalBlock,
    MABATemporalConfig,
    PAPER_IFAN_PARAM_TARGET,
    build_temporal_module,
)
from ifan_maba.training import IFANTrainingConfig, IFANTrainingPipeline
from utils import sph2cart


def test_stage3_sources_parse() -> None:
    targets = (
        PROJECT_ROOT / "ifan_maba" / "eval" / "__init__.py",
        PROJECT_ROOT / "ifan_maba" / "eval" / "stage3.py",
        PROJECT_ROOT / "ifan_maba" / "models" / "maba.py",
        PROJECT_ROOT / "ifan_maba" / "models" / "placeholders.py",
        PROJECT_ROOT / "ifan_maba" / "training" / "pipeline.py",
        PROJECT_ROOT / "scripts" / "train_stage3_ifan_maba.py",
        PROJECT_ROOT / "scripts" / "evaluate_stage3_simulated.py",
        PROJECT_ROOT / "scripts" / "evaluate_stage3_locata.py",
        PROJECT_ROOT / "scripts" / "compare_locata_four_models.py",
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


def test_stage3_ifan_paper_forward_variants() -> None:
    variants = (
        IFANModelConfig(
            temporal_backend="conv1d",
            final_head_pooling=False,
        ),
        IFANModelConfig(
            temporal_backend="maba",
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


def test_stage3_temporal_backend_factory_supports_conv1d_and_maba() -> None:
    conv = build_temporal_module("conv1d", 16, MABATemporalConfig())
    maba = build_temporal_module("maba", 16, MABATemporalConfig())

    assert type(conv).__name__ == "CausConv1d"
    assert isinstance(maba, MABAChannelTemporalBlock)


def test_stage3_ifan_debug_shapes_and_parameter_target() -> None:
    model = IFANModel(IFANModelConfig(r=2, final_head_pooling=False, temporal_backend="conv1d"))
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
    assert debug["temporal_backend"] == "conv1d"
    assert debug["fusion_temporal_module_types"] == ["CausConv1d"] * 4
    assert debug["final_temporal_module_type"] == "CausConv1d"
    assert abs(model.count_parameters(trainable_only=True) - PAPER_IFAN_PARAM_TARGET) <= 64


def test_stage3_ifan_maba_debug_marks_backend() -> None:
    model = IFANModel(IFANModelConfig(r=2, temporal_backend="maba"))
    x = torch.randn(2, model.expected_input_channels(), 6, 5, 4, 8)

    coords, debug = model(x, return_debug=True)

    assert coords.shape == (2, 6, 3)
    assert debug["temporal_backend"] == "maba"
    assert debug["fusion_temporal_module_types"] == ["MABAChannelTemporalBlock"] * 4
    assert debug["final_temporal_module_type"] == "MABAChannelTemporalBlock"


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


def test_stage3_input_ablation_modes_zero_expected_branch() -> None:
    maps = torch.arange(2 * 2 * 3 * 5 * 4 * 8, dtype=torch.float32).reshape(2, 2, 3, 5, 4, 8)
    config = IFANModelConfig()

    phat_only = select_model_inputs(maps, config, "phat_only")
    lms_only = select_model_inputs(maps, config, "lms_only")

    assert torch.allclose(phat_only[:, 0:1, ...], maps[:, 0:1, ...])
    assert torch.count_nonzero(phat_only[:, 1:2, ...]).item() == 0
    assert torch.count_nonzero(lms_only[:, 0:1, ...]).item() == 0
    assert torch.allclose(lms_only[:, 1:2, ...], maps[:, 1:2, ...])


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

[model]
temporal_backend = "maba"

[maba]
d_model = 48
state_dim = 24
conv_kernel = 5
dropout = 0.2
use_residual = true
use_gate = false
use_state = true
""".strip(),
        encoding="utf-8",
    )

    config = IFANTrainingConfig.from_toml(config_path)

    assert config.lms_backend == "frequency_block"
    assert config.lms_block_size == 128
    assert config.lms_fft_size == 512
    assert config.temporal_backend == "maba"
    assert config.maba.d_model == 48
    assert config.maba.state_dim == 24
    assert config.maba.conv_kernel == 5
    assert config.maba.dropout == pytest.approx(0.2)
    assert config.maba.use_gate is False


def test_stage3_train_script_parser_accepts_frequency_block_options() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "train_stage3_ifan_maba.py"))
    parser = module["build_parser"]()
    args = parser.parse_args(
        [
            "--lms-backend",
            "frequency_block",
            "--lms-block-size",
            "128",
            "--lms-fft-size",
            "512",
            "--temporal-backend",
            "maba",
        ]
    )

    assert args.lms_backend == "frequency_block"
    assert args.lms_block_size == 128
    assert args.lms_fft_size == 512
    assert args.temporal_backend == "maba"


def test_stage3_train_script_parser_accepts_schedule_overrides() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "train_stage3_ifan_maba.py"))
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


def test_stage3_locata_four_model_compare_parser_accepts_report_paths() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts" / "compare_locata_four_models.py"))
    parser = module["build_parser"]()
    args = parser.parse_args(
        [
            "--baseline-report",
            "base.json",
            "--replace1d-report",
            "replace.json",
            "--ablation-report",
            "ablation.json",
            "--ifan-report",
            "ifan.json",
            "--ifan-maba-report",
            "ifan_maba.json",
            "--output-json",
            "out.json",
        ]
    )

    assert args.baseline_report == "base.json"
    assert args.replace1d_report == "replace.json"
    assert args.ablation_report == "ablation.json"
    assert args.ifan_report == "ifan.json"
    assert args.ifan_maba_report == "ifan_maba.json"
    assert args.output_json == "out.json"


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
