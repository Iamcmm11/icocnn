from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tomli

from ..dcase_stereo import (
    PreparedDcaseClip,
    aggregate_reports,
    build_active_mask_tensor,
    build_folded_azimuth_sincos_tensor,
    build_model_config,
    build_report,
    checkpoint_model_kind,
    create_preprocessor,
    evaluate_model_on_prepared,
    load_ifan_backbone_into_azimuth_model,
    load_ifan_checkpoint,
    load_manifest,
    masked_coordinate_loss,
    masked_sincos_loss,
    prepare_rows,
    select_rows_by_split,
    split_manifest_rows,
    stereo_proxy_positions,
    write_report,
)
from ..models import (
    DcaseAzimuthHeadConfig,
    DcaseAzimuthOnlyIFANModel,
    IFANModel,
    IFANModelConfig,
    MapMABATemporalConfig,
)


@dataclass
class DcaseStereoTrainingConfig:
    stage_name: str = "dcase_stereo_stage_01"
    experiment_name: str = "dcase_stereo_ifan"
    output_root: str = "IFAN_Edge/outputs/dcase_stage3"
    analysis_output_root: str = "IFAN_Edge/outputs/stage3/analysis"
    output_suffix: str = "dcase_stereo_folded_azimuth_c8_r2_maba_pre_readout_init_from_frozen"
    seed: int = 42
    device: str = "cuda"
    dataset_root: str = "datasets/DCASE2025_Task3"
    train_manifest: str = "datasets/DCASE2025_Task3/locata_like_strict/manifest_all.csv"
    test_manifest: str = "datasets/DCASE2025_Task3/locata_like_devtest_strict/manifest_all.csv"
    train_splits: tuple[str, ...] = ("dev-train-sony", "dev-train-tau")
    validation_fraction: float = 0.1
    validation_min_per_bucket: int = 1
    fs: int = 16000
    k: int = 4096
    step: int = 3072
    stereo_baseline_m: float = 0.08
    exclude_initial_windows: int = 5
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
    srp_variant: str = "paper_original"
    phat_sinc_half_width: int = 0
    r: int = 2
    branch_channels: int = 8
    final_head_pooling: bool = False
    smooth_vertices: bool = True
    map_refiner: str = "maba"
    map_refiner_position: str = "pre_readout"
    map_maba: MapMABATemporalConfig = field(default_factory=MapMABATemporalConfig)
    weak_map_refiner: str = "none"
    weak_map_maba: MapMABATemporalConfig = field(default_factory=lambda: MapMABATemporalConfig(d_model=8, state_dim=4, dropout=0.0, use_state=False))
    ifan_init_checkpoint_path: str = "IFAN_Edge/outputs/stage3/ifan_stage3_long80_c8_r2_maba_pre_readout_20260518_215408/checkpoints/best_rmsae.pt"
    epochs: int = 20
    batch_size: int = 8
    micro_batch_size: int = 2
    eval_batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.0
    checkpoint_every: int = 5
    experiment_role: str = "dcase_stereo_folded_azimuth_c8_r2_maba_pre_readout_init_from_frozen"

    @classmethod
    def from_toml(cls, path: str | Path) -> "DcaseStereoTrainingConfig":
        with Path(path).open("rb") as handle:
            raw = tomli.load(handle)
        project = raw.get("project", {})
        runtime = raw.get("runtime", {})
        paths = raw.get("paths", {})
        data = raw.get("data", {})
        model = raw.get("model", {})
        map_maba = raw.get("map_maba", {})
        weak_map_maba = raw.get("weak_map_maba", {})
        training = raw.get("training", {})
        return cls(
            stage_name=str(project.get("stage_name", cls.stage_name)),
            experiment_name=str(project.get("experiment_name", cls.experiment_name)),
            output_root=str(paths.get("output_root", cls.output_root)),
            analysis_output_root=str(paths.get("analysis_output_root", cls.analysis_output_root)),
            output_suffix=str(runtime.get("output_suffix", cls.output_suffix)),
            seed=int(runtime.get("seed", cls.seed)),
            device=str(runtime.get("device", cls.device)),
            dataset_root=str(paths.get("dataset_root", cls.dataset_root)),
            train_manifest=str(paths.get("train_manifest", cls.train_manifest)),
            test_manifest=str(paths.get("test_manifest", cls.test_manifest)),
            train_splits=tuple(paths.get("train_splits", list(cls.train_splits))),
            validation_fraction=float(data.get("validation_fraction", cls.validation_fraction)),
            validation_min_per_bucket=int(data.get("validation_min_per_bucket", cls.validation_min_per_bucket)),
            fs=int(data.get("fs", cls.fs)),
            k=int(data.get("k", cls.k)),
            step=int(data.get("step", cls.step)),
            stereo_baseline_m=float(data.get("stereo_baseline_m", cls.stereo_baseline_m)),
            exclude_initial_windows=int(data.get("exclude_initial_windows", cls.exclude_initial_windows)),
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
            srp_variant=str(data.get("srp_variant", cls.srp_variant)),
            phat_sinc_half_width=int(data.get("phat_sinc_half_width", cls.phat_sinc_half_width)),
            r=int(model.get("r", cls.r)),
            branch_channels=int(model.get("branch_channels", cls.branch_channels)),
            final_head_pooling=bool(model.get("final_head_pooling", cls.final_head_pooling)),
            smooth_vertices=bool(model.get("smooth_vertices", cls.smooth_vertices)),
            map_refiner=str(model.get("map_refiner", cls.map_refiner)),
            map_refiner_position=str(model.get("map_refiner_position", cls.map_refiner_position)),
            map_maba=MapMABATemporalConfig.from_mapping(
                {
                    "d_model": map_maba.get("d_model", MapMABATemporalConfig.d_model),
                    "state_dim": map_maba.get("state_dim", MapMABATemporalConfig.state_dim),
                    "conv_kernel": map_maba.get("conv_kernel", MapMABATemporalConfig.conv_kernel),
                    "dropout": map_maba.get("dropout", MapMABATemporalConfig.dropout),
                    "use_residual": map_maba.get("use_residual", MapMABATemporalConfig.use_residual),
                    "use_gate": map_maba.get("use_gate", MapMABATemporalConfig.use_gate),
                    "use_state": map_maba.get("use_state", MapMABATemporalConfig.use_state),
                }
            ),
            weak_map_refiner=str(model.get("weak_map_refiner", cls.weak_map_refiner)),
            weak_map_maba=MapMABATemporalConfig.from_mapping(
                {
                    "d_model": weak_map_maba.get("d_model", 8),
                    "state_dim": weak_map_maba.get("state_dim", 4),
                    "conv_kernel": weak_map_maba.get("conv_kernel", MapMABATemporalConfig.conv_kernel),
                    "dropout": weak_map_maba.get("dropout", 0.0),
                    "use_residual": weak_map_maba.get("use_residual", MapMABATemporalConfig.use_residual),
                    "use_gate": weak_map_maba.get("use_gate", MapMABATemporalConfig.use_gate),
                    "use_state": weak_map_maba.get("use_state", False),
                }
            ),
            ifan_init_checkpoint_path=str(paths.get("ifan_init_checkpoint_path", cls.ifan_init_checkpoint_path)),
            epochs=int(training.get("epochs", cls.epochs)),
            batch_size=int(training.get("batch_size", cls.batch_size)),
            micro_batch_size=int(training.get("micro_batch_size", cls.micro_batch_size)),
            eval_batch_size=int(training.get("eval_batch_size", cls.eval_batch_size)),
            lr=float(training.get("lr", cls.lr)),
            weight_decay=float(training.get("weight_decay", cls.weight_decay)),
            checkpoint_every=int(training.get("checkpoint_every", cls.checkpoint_every)),
            experiment_role=str(training.get("experiment_role", cls.experiment_role)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_config(self) -> IFANModelConfig:
        charts = 5
        fusion_r = self.r - 1 if self.r > 1 else self.r
        output_r = fusion_r - 1 if self.final_head_pooling else fusion_r
        spatial_h = 2**output_r
        spatial_w = 2 ** (output_r + 1)
        return IFANModelConfig(
            r=self.r,
            phat_in_channels=1,
            aux_in_channels=1,
            branch_channels=self.branch_channels,
            smooth_vertices=self.smooth_vertices,
            final_head_pooling=self.final_head_pooling,
            temporal_conv_variant="standard_1d",
            map_refiner=self.map_refiner,
            map_refiner_position=self.map_refiner_position,
            map_maba=self.map_maba.with_grid(charts=charts, height=spatial_h, width=spatial_w),
            weak_map_refiner=self.weak_map_refiner,
            weak_map_maba=self.weak_map_maba.with_grid(charts=charts, height=spatial_h, width=spatial_w),
        )

    def experiment_contract(self) -> dict[str, Any]:
        return {
            "stage": self.stage_name,
            "experiment_role": self.experiment_role,
            "model_topology": "paper_dual_mainline_3d_head",
            "feature_pair": "phat+lms",
            "input_geometry": "stereo_proxy",
            "target_mode": "horizontal_folded_azimuth_unit_vector",
            "srp_variant": self.srp_variant,
            "map_refiner": self.map_refiner,
            "map_refiner_position": self.map_refiner_position,
            "lms_backend": self.lms_backend,
            "train_manifest": self.train_manifest,
            "test_manifest": self.test_manifest,
            "train_splits": list(self.train_splits),
            "init_checkpoint": self.ifan_init_checkpoint_path,
        }


@dataclass
class DcaseStereoAzimuthOnlyTrainingConfig(DcaseStereoTrainingConfig):
    stage_name: str = "dcase_stereo_stage_02"
    experiment_name: str = "dcase_stereo_ifan_azimuth_only"
    output_suffix: str = "dcase_stereo_azimuth_only_c8_r2_maba_pre_readout_init_from_stage1_dcase80"
    ifan_init_checkpoint_path: str = "IFAN_Edge/outputs/dcase_stage3/dcase_stereo_ifan_run80_bg_20260609_131300/checkpoints/best_doa_error.pt"
    epochs: int = 30
    experiment_role: str = "dcase_stereo_azimuth_only_c8_r2_maba_pre_readout_init_from_stage1_dcase80"
    azimuth_head: DcaseAzimuthHeadConfig = field(default_factory=DcaseAzimuthHeadConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> "DcaseStereoAzimuthOnlyTrainingConfig":
        config = DcaseStereoTrainingConfig.from_toml.__func__(cls, path)
        with Path(path).open("rb") as handle:
            raw = tomli.load(handle)
        config.azimuth_head = DcaseAzimuthHeadConfig.from_mapping(raw.get("azimuth_head", {}))
        return config

    def experiment_contract(self) -> dict[str, Any]:
        payload = super().experiment_contract()
        payload.update(
            {
                "model_topology": "dcase_azimuth_only_ifan",
                "target_mode": "folded_azimuth_sincos",
                "azimuth_head": self.azimuth_head.to_dict(),
            }
        )
        return payload


def serialize_model_config(model_config: IFANModelConfig) -> dict[str, Any]:
    return {
        "r": model_config.r,
        "branch_channels": model_config.branch_channels,
        "final_head_pooling": model_config.final_head_pooling,
        "smooth_vertices": model_config.smooth_vertices,
        "temporal_conv_variant": model_config.temporal_conv_variant,
        "map_refiner": model_config.map_refiner,
        "map_refiner_position": model_config.map_refiner_position,
        "map_maba": asdict(model_config.map_maba),
        "weak_map_refiner": model_config.weak_map_refiner,
        "weak_map_maba": asdict(model_config.weak_map_maba),
    }


class DcaseStereoTrainer:
    def __init__(
        self,
        config: DcaseStereoTrainingConfig,
        *,
        train_limit: int | None = None,
        validation_limit: int | None = None,
        test_limit: int | None = None,
    ):
        self.config = config
        self.model_config = config.model_config()
        self.train_limit = train_limit
        self.validation_limit = validation_limit
        self.test_limit = test_limit
        self.output_dir: Path | None = None
        self.checkpoint_dir: Path | None = None

    def resolve_device(self) -> torch.device:
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _make_output_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{self.config.output_suffix}" if self.config.output_suffix else ""
        output_dir = Path(self.config.output_root) / f"{self.config.experiment_name}{suffix}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / "checkpoints"
        return output_dir

    @staticmethod
    def _set_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model_profile(self, model: IFANModel, sample_time_steps: int) -> dict[str, Any]:
        input_shape = (1, 2, int(sample_time_steps), 5, 4, 8)
        mac_proxy = model.mac_proxy(input_shape)
        return {
            "trainable_params": int(model.count_parameters(trainable_only=True)),
            "total_params": int(model.count_parameters(trainable_only=False)),
            "parameter_breakdown": model.parameter_breakdown(),
            "mac_proxy_total": int(mac_proxy["total"]),
            "mac_proxy_breakdown": {key: int(value) for key, value in mac_proxy.items() if key != "total"},
            "mac_proxy_input_shape": list(input_shape),
        }

    def _load_rows(self) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
        train_manifest_rows = load_manifest(Path(self.config.train_manifest))
        train_manifest_rows = select_rows_by_split(train_manifest_rows, set(self.config.train_splits))
        train_rows, validation_rows = split_manifest_rows(
            train_manifest_rows,
            validation_fraction=self.config.validation_fraction,
            validation_min_per_bucket=self.config.validation_min_per_bucket,
            seed=self.config.seed,
        )
        if self.train_limit is not None:
            train_rows = train_rows[: self.train_limit]
        if self.validation_limit is not None:
            validation_rows = validation_rows[: self.validation_limit]
        test_rows = load_manifest(Path(self.config.test_manifest), limit=self.test_limit)
        return train_rows, validation_rows, test_rows

    def _prepare_all_clips(self) -> tuple[list[PreparedDcaseClip], list[PreparedDcaseClip], list[PreparedDcaseClip]]:
        dataset_root = Path(self.config.dataset_root)
        train_rows, validation_rows, test_rows = self._load_rows()
        common_kwargs = {
            "dataset_root": dataset_root,
            "fs": int(self.config.fs),
            "k": int(self.config.k),
            "step": int(self.config.step),
        }
        train_clips = prepare_rows(train_rows, progress_label="train_prepare", **common_kwargs)
        validation_clips = prepare_rows(validation_rows, progress_label="validation_prepare", **common_kwargs)
        test_clips = prepare_rows(test_rows, progress_label="test_prepare", **common_kwargs)
        return train_clips, validation_clips, test_clips

    def _train_epoch(
        self,
        *,
        model: IFANModel,
        optimizer: torch.optim.Optimizer,
        preprocessor: DualFeatureIcoPreprocessor,
        train_clips: list[PreparedDcaseClip],
        device: torch.device,
        epoch: int,
    ) -> float:
        order = np.random.permutation(len(train_clips))
        model.train()
        total_loss = 0.0
        total_items = 0
        batch_size = int(self.config.batch_size)
        micro_batch_size = int(self.config.micro_batch_size)
        epoch_start = time.perf_counter()

        for group_start in range(0, len(order), batch_size):
            group_indices = order[group_start : group_start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            group_items = len(group_indices)
            for micro_start in range(0, group_items, micro_batch_size):
                micro_indices = group_indices[micro_start : micro_start + micro_batch_size]
                micro_clips = [train_clips[int(index)] for index in micro_indices]
                mic_batch = np.stack([clip.windows for clip in micro_clips], axis=0)
                scenes = [clip.scene for clip in micro_clips]
                with torch.no_grad():
                    maps, doa_batch = preprocessor.data_transformation(mic_batch, scenes)
                coords = model(maps.to(device)).contiguous()
                active_mask = build_active_mask_tensor(
                    micro_clips,
                    exclude_initial_windows=self.config.exclude_initial_windows,
                    device=coords.device,
                )
                loss = masked_coordinate_loss(coords=coords, doa_batch=doa_batch.to(coords.device), active_mask=active_mask)
                scaled_loss = loss * (float(len(micro_clips)) / float(group_items))
                scaled_loss.backward()
                total_loss += float(loss.item()) * float(len(micro_clips))
                total_items += len(micro_clips)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            processed = min(group_start + batch_size, len(order))
            if processed == len(order) or processed % max(batch_size * 5, 1) == 0:
                print(
                    json.dumps(
                        {
                            "event": "dcase_train_progress",
                            "epoch": int(epoch),
                            "processed": int(processed),
                            "total": int(len(order)),
                            "avg_loss": total_loss / max(total_items, 1),
                            "elapsed_s": time.perf_counter() - epoch_start,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        return total_loss / max(total_items, 1)

    def _save_checkpoint(
        self,
        *,
        model: IFANModel,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        tag: str,
        metrics: dict[str, Any],
    ) -> Path:
        if self.checkpoint_dir is None:
            raise RuntimeError("Checkpoint directory is not initialized.")
        path = self.checkpoint_dir / f"{tag}.pt"
        torch.save(
            {
                "model_kind": "ifan_coords",
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_config": self.config.to_dict(),
                "model_config": serialize_model_config(self.model_config),
                "metrics": metrics,
                "experiment_contract": self.config.experiment_contract(),
            },
            path,
        )
        return path

    def _history_path(self) -> Path:
        if self.output_dir is None:
            raise RuntimeError("Output directory is not initialized.")
        return self.output_dir / "history.csv"

    def _write_history(self, rows: list[dict[str, Any]]) -> Path:
        path = self._history_path()
        fieldnames = list(rows[0].keys()) if rows else ["epoch", "train_loss", "val_doa_error_deg", "val_folded_azimuth_rmse_deg", "val_horizontal_assumption_rmsae_deg", "epoch_time_s"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path

    def run(self) -> dict[str, Any]:
        self._set_seed(self.config.seed)
        output_dir = self._make_output_dir()
        device = self.resolve_device()
        train_clips, validation_clips, test_clips = self._prepare_all_clips()
        if not train_clips:
            raise RuntimeError("No training clips available for DCASE stereo training.")
        if not validation_clips:
            raise RuntimeError("No validation clips available for DCASE stereo training.")
        if not test_clips:
            raise RuntimeError("No test clips available for DCASE stereo evaluation.")

        model = IFANModel(self.model_config).to(device)
        init_checkpoint = torch.load(self.config.ifan_init_checkpoint_path, map_location="cpu")
        init_load_notes = load_ifan_checkpoint(model, init_checkpoint, self.model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        rn = stereo_proxy_positions(self.config.stereo_baseline_m)
        preprocessor = create_preprocessor(
            self.config.to_dict(),
            rn=rn,
            device=device,
            apply_vad=self.config.apply_vad,
        )
        model_profile = self._build_model_profile(model, sample_time_steps=int(train_clips[0].windows.shape[0]))
        frontend_profile = preprocessor.frontend_profile()

        history_rows: list[dict[str, Any]] = []
        best_val_doa_error = float("inf")
        best_checkpoint_path: Path | None = None
        last_checkpoint_path: Path | None = None

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = self._train_epoch(
                model=model,
                optimizer=optimizer,
                preprocessor=preprocessor,
                train_clips=train_clips,
                device=device,
                epoch=epoch,
            )
            validation_reports = evaluate_model_on_prepared(
                model=model,
                preprocessor=preprocessor,
                prepared=validation_clips,
                device=device,
                batch_size=self.config.eval_batch_size,
                exclude_initial_windows=self.config.exclude_initial_windows,
                progress_callback=lambda done, total: print(
                    json.dumps(
                        {
                            "event": "dcase_validation_progress",
                            "epoch": int(epoch),
                            "processed": int(done),
                            "total": int(total),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                ),
            )
            validation_summary = aggregate_reports(validation_reports)
            row = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_doa_error_deg": float(validation_summary["doa_error_deg"]["mean"]),
                "val_folded_azimuth_rmse_deg": float(validation_summary["folded_azimuth_rmse_deg"]["mean"]),
                "val_horizontal_assumption_rmsae_deg": float(validation_summary["horizontal_assumption_rmsae_deg"]["mean"]),
                "epoch_time_s": float(time.perf_counter() - epoch_start),
            }
            history_rows.append(row)
            last_checkpoint_path = self._save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                tag="last",
                metrics=row,
            )
            if epoch % self.config.checkpoint_every == 0:
                self._save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    tag=f"epoch_{epoch:03d}",
                    metrics=row,
                )
            if row["val_doa_error_deg"] < best_val_doa_error:
                best_val_doa_error = row["val_doa_error_deg"]
                best_checkpoint_path = self._save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    tag="best_doa_error",
                    metrics=row,
                )
            self._write_history(history_rows)
            print(json.dumps({"event": "dcase_epoch_complete", **row}, ensure_ascii=False), flush=True)

        if best_checkpoint_path is None:
            raise RuntimeError("Failed to produce a best checkpoint.")

        best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        best_model = IFANModel(self.model_config).to(device)
        load_ifan_checkpoint(best_model, best_checkpoint, self.model_config)
        best_model.eval()
        test_reports = evaluate_model_on_prepared(
            model=best_model,
            preprocessor=preprocessor,
            prepared=test_clips,
            device=device,
            batch_size=self.config.eval_batch_size,
            exclude_initial_windows=self.config.exclude_initial_windows,
            progress_callback=lambda done, total: print(
                json.dumps(
                    {
                        "event": "dcase_test_progress",
                        "processed": int(done),
                        "total": int(total),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            ),
        )
        analysis_output_name = f"dcase2025_locata_like_devtest_strict_{output_dir.name}.json"
        analysis_output_path = Path(self.config.analysis_output_root) / analysis_output_name
        test_report = build_report(
            reports=test_reports,
            checkpoint=str(best_checkpoint_path.resolve()),
            manifest=str(Path(self.config.test_manifest).resolve()),
            dataset_root=str(Path(self.config.dataset_root).resolve()),
            device=str(device),
            limit=self.test_limit,
            exclude_initial_windows=int(self.config.exclude_initial_windows),
            stereo_baseline_m=float(self.config.stereo_baseline_m),
            rn=rn,
            config={
                "fs": int(self.config.fs),
                "k": int(self.config.k),
                "step": int(self.config.step),
                "r": int(self.config.r),
                "apply_vad": bool(self.config.apply_vad),
                "lms_backend": str(self.config.lms_backend),
                "lms_map_mode": str(self.config.lms_map_mode),
                "lms_update_mode": str(self.config.lms_update_mode),
                "srp_variant": str(self.config.srp_variant),
            },
            model_payload={
                "epoch": int(best_checkpoint.get("epoch", -1)),
                "model_config": best_checkpoint["model_config"],
                "metrics": best_checkpoint.get("metrics", {}),
                "load_notes": init_load_notes,
            },
        )
        report_json_path, report_markdown_path = write_report(analysis_output_path, test_report)

        summary = {
            "kind": "dcase_stereo_folded_azimuth_training",
            "output_dir": str(output_dir),
            "device_requested": self.config.device,
            "device_resolved": str(device),
            "training_config": self.config.to_dict(),
            "experiment_contract": self.config.experiment_contract(),
            "dataset_counts": {
                "train": len(train_clips),
                "validation": len(validation_clips),
                "test": len(test_clips),
            },
            "model_profile": model_profile,
            "frontend_profile": frontend_profile,
            "init_checkpoint": {
                "path": self.config.ifan_init_checkpoint_path,
                "epoch": int(init_checkpoint.get("epoch", -1)),
                "load_notes": init_load_notes,
            },
            "best_val_doa_error_deg": float(best_val_doa_error),
            "best_checkpoint_path": str(best_checkpoint_path),
            "last_checkpoint_path": None if last_checkpoint_path is None else str(last_checkpoint_path),
            "history_path": str(self._history_path()),
            "test_report_json": str(report_json_path),
            "test_report_markdown": str(report_markdown_path),
            "test_overall": test_report["overall"],
            "final_epoch": history_rows[-1],
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary


class DcaseStereoAzimuthOnlyTrainer(DcaseStereoTrainer):
    config: DcaseStereoAzimuthOnlyTrainingConfig

    def __init__(
        self,
        config: DcaseStereoAzimuthOnlyTrainingConfig,
        *,
        train_limit: int | None = None,
        validation_limit: int | None = None,
        test_limit: int | None = None,
    ):
        super().__init__(
            config,
            train_limit=train_limit,
            validation_limit=validation_limit,
            test_limit=test_limit,
        )

    def _train_epoch(
        self,
        *,
        model: DcaseAzimuthOnlyIFANModel,
        optimizer: torch.optim.Optimizer,
        preprocessor: DualFeatureIcoPreprocessor,
        train_clips: list[PreparedDcaseClip],
        device: torch.device,
        epoch: int,
    ) -> float:
        order = np.random.permutation(len(train_clips))
        model.train()
        total_loss = 0.0
        total_items = 0
        batch_size = int(self.config.batch_size)
        micro_batch_size = int(self.config.micro_batch_size)
        epoch_start = time.perf_counter()

        for group_start in range(0, len(order), batch_size):
            group_indices = order[group_start : group_start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            group_items = len(group_indices)
            for micro_start in range(0, group_items, micro_batch_size):
                micro_indices = group_indices[micro_start : micro_start + micro_batch_size]
                micro_clips = [train_clips[int(index)] for index in micro_indices]
                mic_batch = np.stack([clip.windows for clip in micro_clips], axis=0)
                scenes = [clip.scene for clip in micro_clips]
                with torch.no_grad():
                    maps, _ = preprocessor.data_transformation(mic_batch, scenes)
                pred_sincos = model.forward_sincos(maps.to(device)).contiguous()
                active_mask = build_active_mask_tensor(
                    micro_clips,
                    exclude_initial_windows=self.config.exclude_initial_windows,
                    device=pred_sincos.device,
                )
                target_sincos = build_folded_azimuth_sincos_tensor(
                    micro_clips,
                    device=pred_sincos.device,
                )
                loss = masked_sincos_loss(
                    pred_sincos=pred_sincos,
                    target_sincos=target_sincos,
                    active_mask=active_mask,
                )
                scaled_loss = loss * (float(len(micro_clips)) / float(group_items))
                scaled_loss.backward()
                total_loss += float(loss.item()) * float(len(micro_clips))
                total_items += len(micro_clips)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            processed = min(group_start + batch_size, len(order))
            if processed == len(order) or processed % max(batch_size * 5, 1) == 0:
                print(
                    json.dumps(
                        {
                            "event": "dcase_azimuth_train_progress",
                            "epoch": int(epoch),
                            "processed": int(processed),
                            "total": int(len(order)),
                            "avg_loss": total_loss / max(total_items, 1),
                            "elapsed_s": time.perf_counter() - epoch_start,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        return total_loss / max(total_items, 1)

    def _save_checkpoint(
        self,
        *,
        model: DcaseAzimuthOnlyIFANModel,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        tag: str,
        metrics: dict[str, Any],
    ) -> Path:
        if self.checkpoint_dir is None:
            raise RuntimeError("Checkpoint directory is not initialized.")
        path = self.checkpoint_dir / f"{tag}.pt"
        backbone_config = serialize_model_config(self.model_config)
        torch.save(
            {
                "model_kind": "dcase_azimuth_only_ifan",
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_config": self.config.to_dict(),
                "model_config": backbone_config,
                "backbone_model_config": backbone_config,
                "azimuth_head_config": self.config.azimuth_head.to_dict(),
                "metrics": metrics,
                "experiment_contract": self.config.experiment_contract(),
            },
            path,
        )
        return path

    def _load_initial_weights(
        self,
        model: DcaseAzimuthOnlyIFANModel,
        checkpoint: dict[str, Any],
    ) -> list[str]:
        if checkpoint_model_kind(checkpoint) == "dcase_azimuth_only_ifan":
            return load_ifan_checkpoint(
                model,
                checkpoint,
                self.model_config,
                state_key_prefix="backbone.",
            )
        return load_ifan_backbone_into_azimuth_model(model, checkpoint, self.model_config)

    def run(self) -> dict[str, Any]:
        self._set_seed(self.config.seed)
        output_dir = self._make_output_dir()
        device = self.resolve_device()
        train_clips, validation_clips, test_clips = self._prepare_all_clips()
        if not train_clips:
            raise RuntimeError("No training clips available for DCASE stereo azimuth-only training.")
        if not validation_clips:
            raise RuntimeError("No validation clips available for DCASE stereo azimuth-only training.")
        if not test_clips:
            raise RuntimeError("No test clips available for DCASE stereo evaluation.")

        model = DcaseAzimuthOnlyIFANModel(self.model_config, self.config.azimuth_head).to(device)
        init_checkpoint = torch.load(self.config.ifan_init_checkpoint_path, map_location="cpu")
        init_load_notes = self._load_initial_weights(model, init_checkpoint)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        rn = stereo_proxy_positions(self.config.stereo_baseline_m)
        preprocessor = create_preprocessor(
            self.config.to_dict(),
            rn=rn,
            device=device,
            apply_vad=self.config.apply_vad,
        )
        model_profile = self._build_model_profile(model, sample_time_steps=int(train_clips[0].windows.shape[0]))
        frontend_profile = preprocessor.frontend_profile()

        history_rows: list[dict[str, Any]] = []
        best_val_doa_error = float("inf")
        best_checkpoint_path: Path | None = None
        last_checkpoint_path: Path | None = None

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = self._train_epoch(
                model=model,
                optimizer=optimizer,
                preprocessor=preprocessor,
                train_clips=train_clips,
                device=device,
                epoch=epoch,
            )
            validation_reports = evaluate_model_on_prepared(
                model=model,
                preprocessor=preprocessor,
                prepared=validation_clips,
                device=device,
                batch_size=self.config.eval_batch_size,
                exclude_initial_windows=self.config.exclude_initial_windows,
                progress_callback=lambda done, total: print(
                    json.dumps(
                        {
                            "event": "dcase_azimuth_validation_progress",
                            "epoch": int(epoch),
                            "processed": int(done),
                            "total": int(total),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                ),
            )
            validation_summary = aggregate_reports(validation_reports)
            row = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_doa_error_deg": float(validation_summary["doa_error_deg"]["mean"]),
                "val_folded_azimuth_rmse_deg": float(validation_summary["folded_azimuth_rmse_deg"]["mean"]),
                "val_horizontal_assumption_rmsae_deg": float(validation_summary["horizontal_assumption_rmsae_deg"]["mean"]),
                "epoch_time_s": float(time.perf_counter() - epoch_start),
            }
            history_rows.append(row)
            last_checkpoint_path = self._save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                tag="last",
                metrics=row,
            )
            if epoch % self.config.checkpoint_every == 0:
                self._save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    tag=f"epoch_{epoch:03d}",
                    metrics=row,
                )
            if row["val_doa_error_deg"] < best_val_doa_error:
                best_val_doa_error = row["val_doa_error_deg"]
                best_checkpoint_path = self._save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    tag="best_doa_error",
                    metrics=row,
                )
            self._write_history(history_rows)
            print(json.dumps({"event": "dcase_azimuth_epoch_complete", **row}, ensure_ascii=False), flush=True)

        if best_checkpoint_path is None:
            raise RuntimeError("Failed to produce a best checkpoint.")

        best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        best_model = DcaseAzimuthOnlyIFANModel(self.model_config, self.config.azimuth_head).to(device)
        best_load_notes = load_ifan_checkpoint(
            best_model,
            best_checkpoint,
            self.model_config,
            state_key_prefix="backbone.",
        )
        best_model.eval()
        test_reports = evaluate_model_on_prepared(
            model=best_model,
            preprocessor=preprocessor,
            prepared=test_clips,
            device=device,
            batch_size=self.config.eval_batch_size,
            exclude_initial_windows=self.config.exclude_initial_windows,
            progress_callback=lambda done, total: print(
                json.dumps(
                    {
                        "event": "dcase_azimuth_test_progress",
                        "processed": int(done),
                        "total": int(total),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            ),
        )
        analysis_output_name = f"dcase2025_locata_like_devtest_strict_{output_dir.name}.json"
        analysis_output_path = Path(self.config.analysis_output_root) / analysis_output_name
        test_report = build_report(
            reports=test_reports,
            checkpoint=str(best_checkpoint_path.resolve()),
            manifest=str(Path(self.config.test_manifest).resolve()),
            dataset_root=str(Path(self.config.dataset_root).resolve()),
            device=str(device),
            limit=self.test_limit,
            exclude_initial_windows=int(self.config.exclude_initial_windows),
            stereo_baseline_m=float(self.config.stereo_baseline_m),
            rn=rn,
            config={
                "fs": int(self.config.fs),
                "k": int(self.config.k),
                "step": int(self.config.step),
                "r": int(self.config.r),
                "apply_vad": bool(self.config.apply_vad),
                "lms_backend": str(self.config.lms_backend),
                "lms_map_mode": str(self.config.lms_map_mode),
                "lms_update_mode": str(self.config.lms_update_mode),
                "srp_variant": str(self.config.srp_variant),
            },
            model_payload={
                "epoch": int(best_checkpoint.get("epoch", -1)),
                "model_kind": checkpoint_model_kind(best_checkpoint),
                "model_config": best_checkpoint["model_config"],
                "backbone_model_config": best_checkpoint["backbone_model_config"],
                "azimuth_head_config": best_checkpoint["azimuth_head_config"],
                "metrics": best_checkpoint.get("metrics", {}),
                "load_notes": best_load_notes,
                "init_load_notes": init_load_notes,
            },
        )
        report_json_path, report_markdown_path = write_report(analysis_output_path, test_report)

        summary = {
            "kind": "dcase_stereo_azimuth_only_training",
            "output_dir": str(output_dir),
            "device_requested": self.config.device,
            "device_resolved": str(device),
            "training_config": self.config.to_dict(),
            "experiment_contract": self.config.experiment_contract(),
            "dataset_counts": {
                "train": len(train_clips),
                "validation": len(validation_clips),
                "test": len(test_clips),
            },
            "model_profile": model_profile,
            "frontend_profile": frontend_profile,
            "init_checkpoint": {
                "path": self.config.ifan_init_checkpoint_path,
                "model_kind": checkpoint_model_kind(init_checkpoint),
                "epoch": int(init_checkpoint.get("epoch", -1)),
                "load_notes": init_load_notes,
            },
            "best_val_doa_error_deg": float(best_val_doa_error),
            "best_checkpoint_path": str(best_checkpoint_path),
            "last_checkpoint_path": None if last_checkpoint_path is None else str(last_checkpoint_path),
            "history_path": str(self._history_path()),
            "test_report_json": str(report_json_path),
            "test_report_markdown": str(report_markdown_path),
            "test_overall": test_report["overall"],
            "final_epoch": history_rows[-1],
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary
