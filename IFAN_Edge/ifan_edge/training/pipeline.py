from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tomli

import acousticTrackingDataset as at_dataset
import acousticTrackingLearners as at_learners
import acousticTrackingModels as at_models
from acousticTrackingDataset import Parameter

from ..eval.stage3 import (
    build_librispeech_dataset,
    build_random_trajectory_dataset,
    build_scenario_caches,
    cache_ifan_batches,
    evaluate_model_on_cache,
    select_model_inputs,
    temporary_seed,
)
from ..features import DualFeatureIcoPreprocessor
from ..models import IFANModel, IFANModelConfig, PAPER_IFAN_BRANCH_CHANNELS

DEFAULT_STAGE3_MAINLINE_ANCHOR_RUN = (
    "IFAN_Edge/outputs/stage3/ifan_stage3_full20_freqblock_paper_original_20260419_005314"
)
DEFAULT_STAGE3_MAINLINE_LOCATA_REPORT = (
    "IFAN_Edge/outputs/stage3/analysis/locata_eval_benchmark2_best.json"
)
DEFAULT_STAGE3_BENCHMARK_SUITE = "simulated_4scene+hard_scenes+locata_eval_benchmark2_task1_3_5"
DEFAULT_STAGE3_LIGHTWEIGHT_READY_DELTA_DEG = 0.3


def _validate_paper_mainline_model_section(model: dict[str, Any]) -> None:
    deprecated_equal_expected = {
        "use_residual_block": True,
        "feature_mode": "dual",
        "fusion_mode": "shared_attention_sum",
        "head_mode": "paper_temporal",
    }
    for key, expected in deprecated_equal_expected.items():
        if key in model and model[key] != expected:
            value = model[key]
            raise ValueError(
                f"Stage-3 paper-mainline no longer supports model.{key}={value!r}. "
                f"Only the paper default {expected!r} is supported."
            )

    branch_channels = int(model.get("branch_channels", PAPER_IFAN_BRANCH_CHANNELS))
    for key in ("fused_channels", "fusion_head_channels"):
        if key in model and int(model[key]) != branch_channels:
            raise ValueError(
                f"Stage-3 paper-mainline expects model.{key} to match model.branch_channels={branch_channels}, "
                f"got {model[key]!r}."
            )


@dataclass
class IFANTrainingConfig:
    stage_name: str = "stage_03"
    experiment_name: str = "ifan_stage3"
    output_root: str = "IFAN_Edge/outputs/stage3"
    output_suffix: str = ""
    input_ablation_mode: str = "none"
    seed: int = 42
    device: str = "cuda"
    librispeech_path: str = "datasets/LibriSpeech"
    train_split: str = "train-clean-100"
    test_split: str = "test-clean"
    trajectory_seconds: int = 20
    fs: int = 16000
    k: int = 4096
    step: int = 3072
    r: int = 2
    branch_channels: int = PAPER_IFAN_BRANCH_CHANNELS
    final_head_pooling: bool = False
    smooth_vertices: bool = True
    apply_vad: bool = True
    lms_order: int = 64
    lms_step_size: float = 0.01
    lms_map_normalize: bool = True
    lms_map_mode: str = "tau_sample"
    lms_peak_sigma: float = 2.0
    lms_update_mode: str = "trajectory_tracking"
    lms_normalized: bool = False
    lms_include_self_pairs: bool = True
    lms_backend: str = "frequency_block"
    lms_block_size: int = 256
    lms_fft_size: int | None = None
    nb_points: int = 156
    epochs: int = 40
    phase1_epochs: int = 20
    batch_size_phase1: int = 1
    batch_size_phase2: int = 10
    micro_batch_size_phase1: int = 1
    micro_batch_size_phase2: int = 1
    lr_phase1: float = 1e-4
    lr_phase2: float = 1e-5
    train_dataset_size: int | None = None
    validation_dataset_size: int = 32
    validation_batch_size: int = 1
    scenario_eval_size: int = 8
    scenario_eval_batch_size: int = 1
    checkpoint_every: int = 10
    train_t60_min: float = 0.2
    train_t60_max: float = 1.3
    train_snr_min_phase1: float = 30.0
    train_snr_max_phase1: float = 30.0
    train_snr_min_phase2: float = 5.0
    train_snr_max_phase2: float = 30.0
    validation_snr_min: float = 5.0
    validation_snr_max: float = 30.0
    baseline_checkpoint_path: str = "models/1sourceTracking_icoCNN_robot_K4096_r2_model.bin"
    experiment_role: str = "mainline_baseline"
    srp_variant: str = "paper_original"
    phat_sinc_half_width: int = 0
    temporal_conv_variant: str = "standard_1d"
    temporal_module: str = "conv"
    primary_benchmark_suite: str = DEFAULT_STAGE3_BENCHMARK_SUITE
    mainline_anchor_run: str = DEFAULT_STAGE3_MAINLINE_ANCHOR_RUN
    mainline_anchor_locata_report: str = DEFAULT_STAGE3_MAINLINE_LOCATA_REPORT
    lightweight_ready_delta_deg: float = DEFAULT_STAGE3_LIGHTWEIGHT_READY_DELTA_DEG

    @classmethod
    def from_toml(cls, path: str | Path) -> "IFANTrainingConfig":
        with Path(path).open("rb") as handle:
            raw = tomli.load(handle)

        project = raw.get("project", {})
        runtime = raw.get("runtime", {})
        paths = raw.get("paths", {})
        data = raw.get("data", {})
        model = raw.get("model", {})
        _validate_paper_mainline_model_section(model)
        training = raw.get("training", {})
        evaluation = raw.get("evaluation", {})
        checkpoints = raw.get("checkpoints", {})
        contract = raw.get("contract", {})
        gates = raw.get("gates", {})

        return cls(
            stage_name=str(project.get("stage_name", cls.stage_name)),
            experiment_name=str(project.get("experiment_name", cls.experiment_name)),
            output_root=str(paths.get("output_root", cls.output_root)),
            output_suffix=str(runtime.get("output_suffix", "")),
            input_ablation_mode=str(runtime.get("input_ablation_mode", cls.input_ablation_mode)),
            seed=int(runtime.get("seed", cls.seed)),
            device=str(runtime.get("device", cls.device)),
            librispeech_path=str(paths.get("librispeech_root", cls.librispeech_path)),
            train_split=str(paths.get("train_split", cls.train_split)),
            test_split=str(paths.get("test_split", cls.test_split)),
            trajectory_seconds=int(data.get("trajectory_seconds", cls.trajectory_seconds)),
            fs=int(data.get("fs", cls.fs)),
            k=int(data.get("k", cls.k)),
            step=int(data.get("step", cls.step)),
            r=int(model.get("r", cls.r)),
            branch_channels=int(model.get("branch_channels", cls.branch_channels)),
            final_head_pooling=bool(model.get("final_head_pooling", cls.final_head_pooling)),
            smooth_vertices=bool(model.get("smooth_vertices", cls.smooth_vertices)),
            apply_vad=bool(data.get("apply_vad", cls.apply_vad)),
            lms_order=int(data.get("lms_order", cls.lms_order)),
            lms_step_size=float(data.get("lms_step_size", cls.lms_step_size)),
            lms_map_normalize=bool(data.get("lms_map_normalize", cls.lms_map_normalize)),
            lms_map_mode=str(data.get("lms_map_mode", cls.lms_map_mode)),
            lms_peak_sigma=float(data.get("lms_peak_sigma", cls.lms_peak_sigma)),
            lms_update_mode=str(data.get("lms_update_mode", cls.lms_update_mode)),
            lms_normalized=bool(data.get("lms_normalized", cls.lms_normalized)),
            lms_include_self_pairs=bool(data.get("lms_include_self_pairs", cls.lms_include_self_pairs)),
            lms_backend=str(data.get("lms_backend", cls.lms_backend)),
            lms_block_size=int(data.get("lms_block_size", cls.lms_block_size)),
            lms_fft_size=int(data["lms_fft_size"]) if data.get("lms_fft_size") is not None else None,
            nb_points=int(data.get("nb_points", cls.nb_points)),
            epochs=int(training.get("epochs", cls.epochs)),
            phase1_epochs=int(training.get("phase1_epochs", cls.phase1_epochs)),
            batch_size_phase1=int(training.get("batch_size_phase1", cls.batch_size_phase1)),
            batch_size_phase2=int(training.get("batch_size_phase2", cls.batch_size_phase2)),
            micro_batch_size_phase1=int(training.get("micro_batch_size_phase1", cls.micro_batch_size_phase1)),
            micro_batch_size_phase2=int(training.get("micro_batch_size_phase2", cls.micro_batch_size_phase2)),
            lr_phase1=float(training.get("lr_phase1", cls.lr_phase1)),
            lr_phase2=float(training.get("lr_phase2", cls.lr_phase2)),
            train_dataset_size=int(training["train_dataset_size"]) if training.get("train_dataset_size") is not None else None,
            validation_dataset_size=int(evaluation.get("validation_dataset_size", cls.validation_dataset_size)),
            validation_batch_size=int(evaluation.get("validation_batch_size", cls.validation_batch_size)),
            scenario_eval_size=int(evaluation.get("scenario_eval_size", cls.scenario_eval_size)),
            scenario_eval_batch_size=int(evaluation.get("scenario_eval_batch_size", cls.scenario_eval_batch_size)),
            checkpoint_every=int(checkpoints.get("every_n_epochs", cls.checkpoint_every)),
            train_t60_min=float(training.get("train_t60_min", cls.train_t60_min)),
            train_t60_max=float(training.get("train_t60_max", cls.train_t60_max)),
            train_snr_min_phase1=float(training.get("train_snr_min_phase1", cls.train_snr_min_phase1)),
            train_snr_max_phase1=float(training.get("train_snr_max_phase1", cls.train_snr_max_phase1)),
            train_snr_min_phase2=float(training.get("train_snr_min_phase2", cls.train_snr_min_phase2)),
            train_snr_max_phase2=float(training.get("train_snr_max_phase2", cls.train_snr_max_phase2)),
            validation_snr_min=float(evaluation.get("validation_snr_min", cls.validation_snr_min)),
            validation_snr_max=float(evaluation.get("validation_snr_max", cls.validation_snr_max)),
            baseline_checkpoint_path=str(paths.get("baseline_checkpoint_path", cls.baseline_checkpoint_path)),
            experiment_role=str(contract.get("experiment_role", cls.experiment_role)),
            srp_variant=str(contract.get("srp_variant", cls.srp_variant)),
            phat_sinc_half_width=int(contract.get("phat_sinc_half_width", cls.phat_sinc_half_width)),
            temporal_conv_variant=str(contract.get("temporal_conv_variant", cls.temporal_conv_variant)),
            temporal_module=str(contract.get("temporal_module", cls.temporal_module)),
            primary_benchmark_suite=str(contract.get("primary_benchmark_suite", cls.primary_benchmark_suite)),
            mainline_anchor_run=str(contract.get("mainline_anchor_run", cls.mainline_anchor_run)),
            mainline_anchor_locata_report=str(
                contract.get("mainline_anchor_locata_report", cls.mainline_anchor_locata_report)
            ),
            lightweight_ready_delta_deg=float(
                gates.get("lightweight_ready_delta_deg", cls.lightweight_ready_delta_deg)
            ),
        )

    def model_config(self) -> IFANModelConfig:
        return IFANModelConfig(
            r=self.r,
            phat_in_channels=1,
            aux_in_channels=1,
            branch_channels=self.branch_channels,
            smooth_vertices=self.smooth_vertices,
            final_head_pooling=self.final_head_pooling,
            temporal_conv_variant=self.temporal_conv_variant,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def experiment_contract(self) -> dict[str, Any]:
        return {
            "stage": self.stage_name,
            "experiment_role": self.experiment_role,
            "model_topology": "paper_dual_mainline",
            "branch_channels": self.branch_channels,
            "feature_pair": "phat+lms",
            "srp_variant": self.srp_variant,
            "phat_sinc_half_width": self.phat_sinc_half_width,
            "lms_backend": self.lms_backend,
            "temporal_conv_variant": self.temporal_conv_variant,
            "temporal_module": self.temporal_module,
            "primary_benchmark_suite": self.primary_benchmark_suite,
            "baseline_anchor": {
                "run_dir": self.mainline_anchor_run,
                "locata_report": self.mainline_anchor_locata_report,
                "baseline_checkpoint_path": self.baseline_checkpoint_path,
            },
            "lightweight_gate": {
                "ready_delta_deg": self.lightweight_ready_delta_deg,
                "requires_locata_overall_win": True,
                "requires_task3_task5_no_material_regression": True,
            },
        }


class IFANTrainingPipeline:
    """Stage-3 training loop, validation cache, and baseline comparison orchestration."""

    def __init__(
        self,
        config: IFANTrainingConfig,
        *,
        resume_checkpoint_path: str | None = None,
        resume_output_dir: str | None = None,
        resume_log_path: str | None = None,
    ):
        self.config = config
        self.model_config = config.model_config()
        self.output_dir: Path | None = None
        self.checkpoint_dir: Path | None = None
        self.resume_checkpoint_path = None if resume_checkpoint_path is None else Path(resume_checkpoint_path)
        self.resume_output_dir = None if resume_output_dir is None else Path(resume_output_dir)
        self.resume_log_path = None if resume_log_path is None else Path(resume_log_path)

    @staticmethod
    def set_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def resolve_device(self) -> torch.device:
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def prepare_output_dir(self) -> Path:
        if self.resume_output_dir is not None:
            output_dir = self.resume_output_dir
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir = output_dir
            self.checkpoint_dir = checkpoint_dir
            return output_dir
        root = Path(self.config.output_root)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{self.config.output_suffix}" if self.config.output_suffix else ""
        run_name = f"{self.config.experiment_name}{suffix}_{stamp}"
        output_dir = root / run_name
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        return output_dir

    def build_model_profile(self, model: IFANModel) -> dict[str, Any]:
        input_shape = (
            1,
            model.expected_input_channels(),
            6,
            5,
            2**self.config.r,
            2 ** (self.config.r + 1),
        )
        mac_proxy = model.mac_proxy(input_shape)
        return {
            "trainable_params": int(model.count_parameters(trainable_only=True)),
            "total_params": int(model.count_parameters(trainable_only=False)),
            "parameter_breakdown": {key: int(value) for key, value in model.parameter_breakdown().items()},
            "mac_proxy_total": int(mac_proxy["total"]),
            "mac_proxy_breakdown": {key: int(value) for key, value in mac_proxy.items() if key != "total"},
            "mac_proxy_input_shape": list(input_shape),
        }

    @staticmethod
    def _move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    @staticmethod
    def _load_epoch_history_from_log(log_path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not log_path.exists():
            return rows
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") != "epoch_complete":
                continue
            rows.append(
                {
                    "epoch": int(payload["epoch"]),
                    "phase": int(payload["phase"]),
                    "lr": float(payload["lr"]),
                    "batch_size": int(payload["batch_size"]),
                    "micro_batch_size": int(payload["micro_batch_size"]),
                    "train_loss": float(payload["train_loss"]),
                    "val_loss": float(payload["val_loss"]),
                    "val_rmsae_deg": float(payload["val_rmsae_deg"]),
                    "epoch_time_s": float(payload["epoch_time_s"]),
                }
            )
        return rows

    def _load_resume_state(
        self,
        *,
        model: IFANModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[int, list[dict[str, Any]], float, str]:
        if self.resume_checkpoint_path is None:
            return 0, [], float("inf"), ""
        if self.checkpoint_dir is None:
            raise RuntimeError("Checkpoint directory is not initialized.")
        if not self.resume_checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {self.resume_checkpoint_path}")

        checkpoint = torch.load(self.resume_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._move_optimizer_to_device(optimizer, device)
        start_epoch = int(checkpoint.get("epoch", 0))

        history_rows: list[dict[str, Any]] = []
        if self.resume_log_path is not None:
            history_rows = self._load_epoch_history_from_log(self.resume_log_path)
            history_rows = [row for row in history_rows if int(row["epoch"]) <= start_epoch]

        best_checkpoint_path = ""
        best_val_rmsae = float("inf")
        existing_best_path = self.checkpoint_dir / "best_rmsae.pt"
        if existing_best_path.exists():
            best_checkpoint_path = str(existing_best_path)
            best_state = torch.load(existing_best_path, map_location="cpu")
            best_metrics = best_state.get("metrics", {})
            if "val_rmsae_deg" in best_metrics:
                best_val_rmsae = float(best_metrics["val_rmsae_deg"])
        elif isinstance(checkpoint.get("metrics"), dict) and "val_rmsae_deg" in checkpoint["metrics"]:
            best_val_rmsae = float(checkpoint["metrics"]["val_rmsae_deg"])

        return start_epoch, history_rows, best_val_rmsae, best_checkpoint_path

    @staticmethod
    def move_ifan_preprocessor(preprocessor: DualFeatureIcoPreprocessor, device: torch.device) -> None:
        if device.type == "cuda":
            preprocessor.cuda_activated = True
            preprocessor.phat.cuda()
            preprocessor.lms.cuda()
        else:
            preprocessor.cuda_activated = False
            preprocessor.phat.cpu()
            preprocessor.lms.cpu()

    @staticmethod
    def move_baseline_preprocessor(preprocessor, device: torch.device) -> None:
        if device.type == "cuda":
            preprocessor.cuda_activated = True
            preprocessor.gcc.cuda()
            preprocessor.srp.cuda()
        else:
            preprocessor.cuda_activated = False
            preprocessor.gcc.cpu()
            preprocessor.srp.cpu()

    def phase_settings(self, epoch_index: int) -> dict[str, float | int]:
        if epoch_index < self.config.phase1_epochs:
            return {
                "phase": 1,
                "lr": self.config.lr_phase1,
                "batch_size": self.config.batch_size_phase1,
                "micro_batch_size": self.config.micro_batch_size_phase1,
                "snr_min": self.config.train_snr_min_phase1,
                "snr_max": self.config.train_snr_max_phase1,
            }
        return {
            "phase": 2,
            "lr": self.config.lr_phase2,
            "batch_size": self.config.batch_size_phase2,
            "micro_batch_size": self.config.micro_batch_size_phase2,
            "snr_min": self.config.train_snr_min_phase2,
            "snr_max": self.config.train_snr_max_phase2,
        }

    def build_validation_cache(
        self,
        *,
        source_dataset,
        preprocessor: DualFeatureIcoPreprocessor,
        progress_callback=None,
    ) -> list[dict[str, torch.Tensor]]:
        callback = self._on_validation_cache_progress if progress_callback is None else progress_callback
        with temporary_seed(self.config.seed + 101):
            dataset = build_random_trajectory_dataset(
                source_dataset=source_dataset,
                k=self.config.k,
                step=self.config.step,
                size=self.config.validation_dataset_size,
                t60=Parameter(self.config.train_t60_min, self.config.train_t60_max),
                snr=Parameter(self.config.validation_snr_min, self.config.validation_snr_max),
                nb_points=self.config.nb_points,
            )
            return cache_ifan_batches(
                dataset=dataset,
                preprocessor=preprocessor,
                model_config=self.model_config,
                input_ablation_mode=self.config.input_ablation_mode,
                batch_size=self.config.validation_batch_size,
                progress_callback=callback,
            )

    def build_training_dataset(self, *, source_dataset, snr_min: float, snr_max: float):
        train_size = self.config.train_dataset_size if self.config.train_dataset_size is not None else len(source_dataset)
        return build_random_trajectory_dataset(
            source_dataset=source_dataset,
            k=self.config.k,
            step=self.config.step,
            size=train_size,
            t60=Parameter(self.config.train_t60_min, self.config.train_t60_max),
            snr=Parameter(snr_min, snr_max),
            nb_points=self.config.nb_points,
        )

    @staticmethod
    def _format_hms(seconds: float) -> str:
        total = max(int(seconds), 0)
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _format_progress_line(
        *,
        epoch: int,
        total_epochs: int,
        processed: int,
        total: int,
        elapsed_s: float,
        avg_loss: float,
        bar_width: int = 10,
    ) -> str:
        total = max(total, 1)
        processed = min(max(processed, 0), total)
        progress = processed / float(total)
        filled = min(int(progress * bar_width), bar_width)
        bar = "#" * filled + "." * (bar_width - filled)
        rate = processed / elapsed_s if elapsed_s > 0 else 0.0
        eta_s = (total - processed) / rate if rate > 0 else 0.0
        percent = int(round(progress * 100.0))
        return (
            f"Epoch {epoch}: {percent:3d}%|{bar}| "
            f"{processed}/{total} [{IFANTrainingPipeline._format_hms(elapsed_s)}<"
            f"{IFANTrainingPipeline._format_hms(eta_s)},  {rate:0.2f}it/s, loss={avg_loss:0.4f}]"
        )

    def train_epoch(
        self,
        *,
        model: IFANModel,
        optimizer: torch.optim.Optimizer,
        dataset,
        preprocessor: DualFeatureIcoPreprocessor,
        batch_size: int,
        micro_batch_size: int,
        epoch: int,
        total_epochs: int,
    ) -> float:
        if batch_size <= 0 or micro_batch_size <= 0:
            raise ValueError("Batch sizes must be positive.")
        if batch_size % micro_batch_size != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by micro_batch_size ({micro_batch_size}).")

        model.train()
        total_loss = 0.0
        sample_count = 0
        epoch_start = time.perf_counter()
        last_progress_emit = epoch_start
        progress_emit_interval_s = 5.0
        total_items = len(dataset)

        print(f"Epoch {epoch}/{total_epochs}:", flush=True)

        for group_start in range(0, len(dataset), batch_size):
            group_stop = min(group_start + batch_size, len(dataset))
            group_items = group_stop - group_start
            optimizer.zero_grad(set_to_none=True)

            for start in range(group_start, group_stop, micro_batch_size):
                stop = min(start + micro_batch_size, group_stop)
                mic_sig_batch, acoustic_scene_batch = dataset.get_batch(start, stop)
                with torch.no_grad():
                    maps, doa_batch = preprocessor.data_transformation(mic_sig_batch, acoustic_scene_batch)
                inputs = select_model_inputs(maps, self.model_config, self.config.input_ablation_mode)
                coords = model(inputs).contiguous()
                doa_cart = at_learners.sph2cart(doa_batch.contiguous())
                loss = torch.nn.functional.mse_loss(coords.reshape(-1, 3), doa_cart.reshape(-1, 3))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Encountered non-finite training loss at batch {start}:{stop}")

                batch_items = stop - start
                scaled_loss = loss * (float(batch_items) / float(group_items))
                scaled_loss.backward()

                total_loss += float(loss.item()) * float(batch_items)
                sample_count += batch_items
                now = time.perf_counter()
                if now - last_progress_emit >= progress_emit_interval_s or stop == total_items:
                    avg_loss = total_loss / float(sample_count)
                    print(
                        self._format_progress_line(
                            epoch=epoch,
                            total_epochs=total_epochs,
                            processed=stop,
                            total=total_items,
                            elapsed_s=now - epoch_start,
                            avg_loss=avg_loss,
                        ),
                        flush=True,
                    )
                    last_progress_emit = now

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if sample_count == 0:
            return 0.0
        return total_loss / float(sample_count)

    def save_checkpoint(
        self,
        *,
        model: IFANModel,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        tag: str,
        metrics: dict[str, Any],
    ) -> str:
        if self.checkpoint_dir is None:
            raise RuntimeError("Checkpoint directory is not initialized.")
        path = self.checkpoint_dir / f"{tag}.pt"
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_config": self.config.to_dict(),
            "model_config": {
                "r": self.model_config.r,
                "branch_channels": self.model_config.branch_channels,
                "final_head_pooling": self.model_config.final_head_pooling,
                "smooth_vertices": self.model_config.smooth_vertices,
                "temporal_conv_variant": self.model_config.temporal_conv_variant,
            },
            "metrics": metrics,
        }
        torch.save(payload, path)
        return str(path)

    def compare_against_baseline(
        self,
        *,
        model: IFANModel,
        scenario_caches: dict[str, dict[str, object]],
        device: torch.device,
    ) -> dict[str, Any]:
        baseline_model = at_models.IcoTempCNN(self.config.r, 32, Cin=1, smooth_vertices=self.config.smooth_vertices)
        checkpoint_path = Path(self.config.baseline_checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Baseline checkpoint does not exist: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location="cpu")
        incompatible = baseline_model.load_state_dict(state, strict=False)
        non_buffer_missing = [key for key in incompatible.missing_keys if not key.endswith(".mask")]
        if non_buffer_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Baseline checkpoint is incompatible with the current icoCNN model.\n"
                f"Unexpected keys: {list(incompatible.unexpected_keys)}\n"
                f"Missing non-buffer keys: {non_buffer_missing}"
            )
        baseline_model.to(device)

        scenario_reports: list[dict[str, Any]] = []
        ifan_means = []
        baseline_means = []
        for scenario_name, payload in scenario_caches.items():
            ifan_metrics = evaluate_model_on_cache(model, payload["ifan_batches"])
            baseline_metrics = evaluate_model_on_cache(baseline_model, payload["baseline_batches"])
            scenario_reports.append(
                {
                    "name": scenario_name,
                    "snr_db": payload["scenario"]["snr_db"],
                    "t60_s": payload["scenario"]["t60_s"],
                    "ifan": ifan_metrics,
                    "baseline": baseline_metrics,
                    "rmsae_delta_deg": ifan_metrics["rmsae_deg"] - baseline_metrics["rmsae_deg"],
                }
            )
            ifan_means.append(ifan_metrics["rmsae_deg"])
            baseline_means.append(baseline_metrics["rmsae_deg"])

        ifan_mean = float(np.mean(ifan_means)) if ifan_means else 0.0
        baseline_mean = float(np.mean(baseline_means)) if baseline_means else 0.0
        ifan_hard_mean = float(np.mean([row["ifan"]["rmsae_deg"] for row in scenario_reports if row["name"] in {"scene_3", "scene_4"}]))
        baseline_hard_mean = float(np.mean([row["baseline"]["rmsae_deg"] for row in scenario_reports if row["name"] in {"scene_3", "scene_4"}]))

        return {
            "baseline_checkpoint_path": str(checkpoint_path),
            "srp_variant": self.config.srp_variant,
            "temporal_conv_variant": self.config.temporal_conv_variant,
            "scenarios": scenario_reports,
            "mean_rmsae_deg": {
                "ifan": ifan_mean,
                "baseline": baseline_mean,
                "delta": ifan_mean - baseline_mean,
            },
            "hard_scenarios_mean_rmsae_deg": {
                "ifan": ifan_hard_mean,
                "baseline": baseline_hard_mean,
                "delta": ifan_hard_mean - baseline_hard_mean,
            },
        }

    def run(self) -> dict[str, Any]:
        self.set_seed(self.config.seed)
        device = self.resolve_device()
        output_dir = self.prepare_output_dir()
        print(
            json.dumps(
                {
                    "event": "stage3_boot",
                    "output_dir": str(output_dir),
                    "device_requested": self.config.device,
                    "device_resolved": str(device),
                    "epochs": self.config.epochs,
                    "model_topology": "paper_dual_mainline",
                    "branch_channels": self.model_config.branch_channels,
                    "final_head_pooling": self.model_config.final_head_pooling,
                    "input_ablation_mode": self.config.input_ablation_mode,
                    "srp_variant": self.config.srp_variant,
                    "temporal_conv_variant": self.config.temporal_conv_variant,
                    "resume_checkpoint_path": None if self.resume_checkpoint_path is None else str(self.resume_checkpoint_path),
                    "resume_output_dir": None if self.resume_output_dir is None else str(self.resume_output_dir),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        train_source, train_split_path = build_librispeech_dataset(
            self.config.librispeech_path,
            self.config.train_split,
            self.config.trajectory_seconds,
        )
        val_source, val_split_path = build_librispeech_dataset(
            self.config.librispeech_path,
            self.config.test_split,
            self.config.trajectory_seconds,
        )
        print(
            json.dumps(
                {
                    "event": "datasets_ready",
                    "train_split_path": str(train_split_path),
                    "validation_split_path": str(val_split_path),
                    "train_source_size": len(train_source),
                    "validation_source_size": len(val_source),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        ifan_preprocessor = DualFeatureIcoPreprocessor(
            N=at_dataset.benchmark2_array_setup.mic_pos.shape[0],
            K=self.config.k,
            r=self.config.r,
            rn=at_dataset.benchmark2_array_setup.mic_pos,
            fs=self.config.fs,
            apply_vad=self.config.apply_vad,
            lms_order=self.config.lms_order,
            lms_step_size=self.config.lms_step_size,
            lms_map_normalize=self.config.lms_map_normalize,
            lms_map_mode=self.config.lms_map_mode,
            lms_peak_sigma=self.config.lms_peak_sigma,
            lms_update_mode=self.config.lms_update_mode,
            lms_normalized=self.config.lms_normalized,
            lms_include_self_pairs=self.config.lms_include_self_pairs,
            lms_backend=self.config.lms_backend,
            lms_block_size=self.config.lms_block_size,
            lms_fft_size=self.config.lms_fft_size,
            srp_variant=self.config.srp_variant,
            phat_sinc_half_width=self.config.phat_sinc_half_width,
        )
        baseline_preprocessor = at_learners.TrackingFromIcoMapsPreprocessor(
            N=at_dataset.benchmark2_array_setup.mic_pos.shape[0],
            K=self.config.k,
            r=self.config.r,
            rn=at_dataset.benchmark2_array_setup.mic_pos,
            fs=self.config.fs,
            apply_vad=self.config.apply_vad,
        )
        self.move_ifan_preprocessor(ifan_preprocessor, device)
        self.move_baseline_preprocessor(baseline_preprocessor, device)
        print(json.dumps({"event": "preprocessors_ready"}, ensure_ascii=False), flush=True)
        frontend_profile = ifan_preprocessor.frontend_profile()

        print(
            json.dumps(
                {
                    "event": "validation_cache_build_start",
                    "validation_dataset_size": self.config.validation_dataset_size,
                    "validation_batch_size": self.config.validation_batch_size,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        validation_cache = self.build_validation_cache(source_dataset=val_source, preprocessor=ifan_preprocessor)
        print(
            json.dumps(
                {
                    "event": "validation_cache_build_complete",
                    "cached_batches": len(validation_cache),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        print(
            json.dumps(
                {
                    "event": "scenario_cache_build_start",
                    "scenario_eval_size": self.config.scenario_eval_size,
                    "scenario_eval_batch_size": self.config.scenario_eval_batch_size,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        scenario_caches = build_scenario_caches(
            source_dataset=val_source,
            ifan_preprocessor=ifan_preprocessor,
            baseline_preprocessor=baseline_preprocessor,
            model_config=self.model_config,
            input_ablation_mode=self.config.input_ablation_mode,
            k=self.config.k,
            step=self.config.step,
            batch_size=self.config.scenario_eval_batch_size,
            scenario_size=self.config.scenario_eval_size,
            seed=self.config.seed,
            nb_points=self.config.nb_points,
            progress_callback=self._on_scenario_cache_progress,
        )
        print(
            json.dumps(
                {
                    "event": "scenario_cache_build_complete",
                    "scenario_count": len(scenario_caches),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        model = IFANModel(self.model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr_phase1)
        model_profile = self.build_model_profile(model)
        print(
            json.dumps(
                {
                    "event": "optimizer_ready",
                    "initial_lr": self.config.lr_phase1,
                    "trainable_params": model.count_parameters(trainable_only=True),
                    "mac_proxy_total": model_profile["mac_proxy_total"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        start_epoch, history_rows, best_val_rmsae, best_checkpoint_path = self._load_resume_state(
            model=model,
            optimizer=optimizer,
            device=device,
        )
        if start_epoch >= self.config.epochs:
            raise ValueError(
                f"Resume checkpoint epoch={start_epoch} is already at or beyond configured epochs={self.config.epochs}."
            )

        config_path = output_dir / "resolved_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "training_config": self.config.to_dict(),
                    "model_config": {
                        "r": self.model_config.r,
                        "branch_channels": self.model_config.branch_channels,
                        "final_head_pooling": self.model_config.final_head_pooling,
                        "smooth_vertices": self.model_config.smooth_vertices,
                        "temporal_conv_variant": self.model_config.temporal_conv_variant,
                        "input_ablation_mode": self.config.input_ablation_mode,
                    },
                    "experiment_contract": self.config.experiment_contract(),
                    "model_profile": model_profile,
                    "frontend_profile": frontend_profile,
                    "train_split_path": str(train_split_path),
                    "validation_split_path": str(val_split_path),
                    "device": str(device),
                    "resume_checkpoint_path": None if self.resume_checkpoint_path is None else str(self.resume_checkpoint_path),
                    "resume_output_dir": None if self.resume_output_dir is None else str(self.resume_output_dir),
                    "resume_log_path": None if self.resume_log_path is None else str(self.resume_log_path),
                    "resume_start_epoch": int(start_epoch),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "event": "stage3_start",
                    "output_dir": str(output_dir),
                    "device": str(device),
                    "train_split_path": str(train_split_path),
                    "validation_split_path": str(val_split_path),
                    "model_topology": "paper_dual_mainline",
                    "branch_channels": self.model_config.branch_channels,
                    "final_head_pooling": self.model_config.final_head_pooling,
                    "input_ablation_mode": self.config.input_ablation_mode,
                    "experiment_role": self.config.experiment_role,
                    "srp_variant": self.config.srp_variant,
                    "temporal_conv_variant": self.config.temporal_conv_variant,
                    "temporal_module": self.config.temporal_module,
                    "epochs": self.config.epochs,
                    "resume_start_epoch": int(start_epoch),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        for epoch_index in range(start_epoch, self.config.epochs):
            settings = self.phase_settings(epoch_index)
            epoch_start = time.perf_counter()
            optimizer.param_groups[0]["lr"] = float(settings["lr"])
            train_dataset = self.build_training_dataset(
                source_dataset=train_source,
                snr_min=float(settings["snr_min"]),
                snr_max=float(settings["snr_max"]),
            )
            train_loss = self.train_epoch(
                model=model,
                optimizer=optimizer,
                dataset=train_dataset,
                preprocessor=ifan_preprocessor,
                batch_size=int(settings["batch_size"]),
                micro_batch_size=int(settings["micro_batch_size"]),
                epoch=epoch_index + 1,
                total_epochs=self.config.epochs,
            )
            val_metrics = evaluate_model_on_cache(model, validation_cache)
            print(
                f"Test loss: {val_metrics['loss']:.4f}, Test rmsae: {val_metrics['rmsae_deg']:.2f}deg",
                flush=True,
            )
            row = {
                "epoch": epoch_index + 1,
                "phase": int(settings["phase"]),
                "lr": float(settings["lr"]),
                "batch_size": int(settings["batch_size"]),
                "micro_batch_size": int(settings["micro_batch_size"]),
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]),
                "val_rmsae_deg": float(val_metrics["rmsae_deg"]),
                "epoch_time_s": time.perf_counter() - epoch_start,
            }
            history_rows.append(row)
            print(json.dumps({"event": "epoch_complete", **row}, ensure_ascii=False), flush=True)

            if row["val_rmsae_deg"] < best_val_rmsae:
                best_val_rmsae = row["val_rmsae_deg"]
                best_checkpoint_path = self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch_index + 1,
                    tag="best_rmsae",
                    metrics=row,
                )
                print(
                    json.dumps(
                        {
                            "event": "checkpoint_saved",
                            "tag": "best_rmsae",
                            "epoch": epoch_index + 1,
                            "path": best_checkpoint_path,
                            "val_rmsae_deg": row["val_rmsae_deg"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            if (epoch_index + 1) % self.config.checkpoint_every == 0:
                checkpoint_path = self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch_index + 1,
                    tag=f"epoch_{epoch_index + 1:03d}",
                    metrics=row,
                )
                print(
                    json.dumps(
                        {
                            "event": "checkpoint_saved",
                            "tag": f"epoch_{epoch_index + 1:03d}",
                            "epoch": epoch_index + 1,
                            "path": checkpoint_path,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        last_checkpoint_path = self.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=self.config.epochs,
            tag="last",
            metrics=history_rows[-1] if history_rows else {},
        )

        if best_checkpoint_path:
            best_state = torch.load(best_checkpoint_path, map_location="cpu")
            model.load_state_dict(best_state["model_state_dict"])
            model.to(device)

        baseline_compare = self.compare_against_baseline(
            model=model,
            scenario_caches=scenario_caches,
            device=device,
        )
        baseline_compare_path = output_dir / "baseline_compare.json"
        baseline_compare_path.write_text(json.dumps(baseline_compare, indent=2, ensure_ascii=False), encoding="utf-8")

        history_path = output_dir / "history.csv"
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "epoch",
                "phase",
                "lr",
                "batch_size",
                "micro_batch_size",
                "train_loss",
                "val_loss",
                "val_rmsae_deg",
                "epoch_time_s",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history_rows)

        summary = {
            "output_dir": str(output_dir),
            "history_path": str(history_path),
            "baseline_compare_path": str(baseline_compare_path),
            "best_checkpoint_path": best_checkpoint_path,
            "last_checkpoint_path": last_checkpoint_path,
            "device": str(device),
            "train_split_path": str(train_split_path),
            "validation_split_path": str(val_split_path),
            "epochs": self.config.epochs,
            "model_topology": "paper_dual_mainline",
            "branch_channels": self.model_config.branch_channels,
            "final_head_pooling": self.model_config.final_head_pooling,
            "input_ablation_mode": self.config.input_ablation_mode,
            "srp_variant": self.config.srp_variant,
            "temporal_conv_variant": self.config.temporal_conv_variant,
            "experiment_contract": self.config.experiment_contract(),
            "model_profile": model_profile,
            "frontend_profile": frontend_profile,
            "resume_checkpoint_path": None if self.resume_checkpoint_path is None else str(self.resume_checkpoint_path),
            "resume_output_dir": None if self.resume_output_dir is None else str(self.resume_output_dir),
            "resume_log_path": None if self.resume_log_path is None else str(self.resume_log_path),
            "resume_start_epoch": int(start_epoch),
            "best_val_rmsae_deg": best_val_rmsae,
            "final_epoch": history_rows[-1] if history_rows else {},
            "baseline_compare": baseline_compare,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"event": "stage3_complete", **summary}, ensure_ascii=False), flush=True)
        return summary

    @staticmethod
    def _on_validation_cache_progress(stage: str, batch_index: int, total_batches: int, start: int, stop: int) -> None:
        print(
            json.dumps(
                {
                    "event": "validation_cache_progress",
                    "pid": os.getpid(),
                    "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "stage": stage,
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                    "sample_start": start,
                    "sample_stop": stop,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    @staticmethod
    def _on_scenario_cache_progress(
        scenario_name: str,
        scenario_index: int,
        total_scenarios: int,
        batch_index: int,
        total_batches: int,
        start: int,
        stop: int,
    ) -> None:
        print(
            json.dumps(
                {
                    "event": "scenario_cache_progress",
                    "scenario_name": scenario_name,
                    "scenario_index": scenario_index,
                    "total_scenarios": total_scenarios,
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                    "sample_start": start,
                    "sample_stop": stop,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
