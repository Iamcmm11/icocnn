from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IFANTrainingConfig:
    stage_name: str = "stage_03"
    epochs: int = 80
    batch_size_phase1: int = 1
    batch_size_phase2: int = 10
    lr_phase1: float = 1e-4
    lr_phase2: float = 1e-5


class IFANTrainingPipeline:
    """Reserved for stage-3 training and ablation orchestration."""

    def __init__(self, config: IFANTrainingConfig):
        self.config = config

    def run(self):
        raise NotImplementedError("Stage 3 will implement the IFAN training pipeline.")
