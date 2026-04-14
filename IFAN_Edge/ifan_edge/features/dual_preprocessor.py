from __future__ import annotations

import numpy as np
import torch

from ..bridges import at_learners
from .lms import SRPLMSIcoMap
from .phat import SRPPHATIcoMapAdapter, ensure_mic_tensor


class DualFeatureIcoPreprocessor(at_learners.Preprocessor):
    """IFAN stage-1 preprocessor producing PHAT+LMS dual icosahedral maps."""

    feature_names = ("phat", "lms")

    def __init__(
        self,
        N: int,
        K: int,
        r: int,
        rn,
        fs: int,
        c: float = 343.0,
        apply_vad: bool = False,
        lms_order: int = 64,
        lms_step_size: float = 0.01,
    ):
        super().__init__()
        self.N = N
        self.K = K
        self.r = r
        self.fs = fs
        self.apply_vad = apply_vad

        self.phat = SRPPHATIcoMapAdapter(
            N=N,
            K=K,
            r=r,
            rn=rn,
            fs=fs,
            c=c,
        )
        self.lms = SRPLMSIcoMap(
            N=N,
            K=K,
            r=r,
            rn=rn,
            fs=fs,
            c=c,
            lms_order=lms_order,
            step_size=lms_step_size,
        )

    @staticmethod
    def split_features(maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if maps.shape[1] != 2:
            raise ValueError(f"Expected feature dimension of size 2, got {maps.shape}")
        return maps[:, 0:1, ...], maps[:, 1:2, ...]

    def data_transformation(self, mic_sig_batch=None, acoustic_scene_batch=None, vad_batch=None):
        output = []

        if mic_sig_batch is not None:
            mic_sig_batch = ensure_mic_tensor(mic_sig_batch)
            if self.cuda_activated:
                mic_sig_batch = mic_sig_batch.cuda()

            phat_maps = self.phat(mic_sig_batch)
            lms_maps = self.lms(mic_sig_batch)
            maps = torch.cat((phat_maps, lms_maps), dim=1)
            maps = self.apply_extras(maps, acoustic_scene_batch, vad_batch)
            output.append(maps)

        if acoustic_scene_batch is not None:
            doa_batch = torch.tensor(
                np.stack(
                    [
                        np.stack(
                            [
                                acoustic_scene_batch[i].DOAw[n].astype(np.float32)
                                for n in range(len(acoustic_scene_batch[i].DOAw))
                            ]
                        )
                        for i in range(len(acoustic_scene_batch))
                    ]
                )
            )
            if self.cuda_activated:
                doa_batch = doa_batch.cuda()
            output.append(doa_batch)

        return output[0] if len(output) == 1 else output

    def apply_extras(self, maps, acoustic_scene_batch, vad_batch=None):
        if self.apply_vad:
            if acoustic_scene_batch is not None:
                vad_batch = np.array([acoustic_scene_batch[i].vad for i in range(len(acoustic_scene_batch))])
            assert vad_batch is not None
            vad_output_th = vad_batch.mean(axis=-1) > 2 / 3
            vad_output_th = vad_output_th[:, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
            vad_output_th = torch.from_numpy(vad_output_th.astype(float)).to(maps.device)
            repeat_factor = np.array(maps.shape)
            repeat_factor[:-3] = 1
            maps = maps * vad_output_th.float().repeat(repeat_factor.tolist())
        return maps
