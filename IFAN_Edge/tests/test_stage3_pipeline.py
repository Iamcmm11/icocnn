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

from ifan_edge.eval.stage3 import resolve_librispeech_split, select_model_inputs
from ifan_edge.features import SRPLMSIcoMap
from ifan_edge.models import IFANModel, IFANModelConfig, PAPER_IFAN_PARAM_TARGET
from ifan_edge.training import IFANTrainingConfig, IFANTrainingPipeline


def test_stage3_sources_parse() -> None:
    targets = (
        PROJECT_ROOT / "ifan_edge" / "eval" / "stage3.py",
        PROJECT_ROOT / "ifan_edge" / "training" / "pipeline.py",
        PROJECT_ROOT / "scripts" / "train_stage3_ifan.py",
        PROJECT_ROOT / "scripts" / "compare_stage3_baseline.py",
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
