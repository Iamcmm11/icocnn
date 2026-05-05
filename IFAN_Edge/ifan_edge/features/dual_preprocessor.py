from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

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
        lms_normalized: bool = True,
        lms_include_self_pairs: bool = True,
        lms_map_normalize: bool = True,
        lms_map_mode: str = "tau_sample",
        lms_peak_sigma: float = 2.0,
        lms_update_mode: str = "frame_reset",
        lms_backend: str = "time_reference",
        lms_block_size: int = 256,
        lms_fft_size: int | None = None,
        srp_variant: str = "paper_original",
        phat_sinc_half_width: int = 0,
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
            srp_variant=srp_variant,
            sinc_half_width=phat_sinc_half_width,
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
            normalize=lms_map_normalize,
            map_mode=lms_map_mode,
            peak_sigma=lms_peak_sigma,
            update_mode=lms_update_mode,
            normalized_lms=lms_normalized,
            include_self_pairs=lms_include_self_pairs,
            lms_backend=lms_backend,
            lms_block_size=lms_block_size,
            lms_fft_size=lms_fft_size,
        )

    def frontend_profile(self) -> dict[str, object]:
        return {
            "phat": self.phat.frontend_profile(),
            "lms": {
                "backend": self.lms.lms_backend,
                "include_self_pairs": bool(self.lms.include_self_pairs),
                "map_mode": self.lms.map_mode,
                "update_mode": self.lms.update_mode,
                "block_size": int(self.lms.lms_block_size) if hasattr(self.lms, "lms_block_size") else None,
                "fft_size": int(self.lms.lms_fft_size) if getattr(self.lms, "lms_fft_size", None) is not None else None,
            },
        }

    @staticmethod
    def split_features(maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if maps.shape[1] != 2:
            raise ValueError(f"Expected feature dimension of size 2, got {maps.shape}")
        return maps[:, 0:1, ...], maps[:, 1:2, ...]

    @staticmethod
    def _debug_mode() -> str:
        value = os.getenv("IFAN_DEBUG_PREPROCESS", "")
        value = value.lower().strip()
        if value in ("", "0", "false", "no", "off"):
            return "off"
        if value in ("verbose", "detail", "detailed", "2"):
            return "verbose"
        return "summary"

    @classmethod
    def _debug_enabled(cls) -> bool:
        return cls._debug_mode() != "off"

    @classmethod
    def _debug_verbose(cls) -> bool:
        return cls._debug_mode() == "verbose"

    @classmethod
    def _debug_event(cls, stage: str, **fields) -> None:
        if not cls._debug_verbose():
            return
        payload = {
            "event": "dual_preprocessor_debug",
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stage": stage,
            **fields,
        }
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    @classmethod
    def _debug_summary(cls, **fields) -> None:
        if not cls._debug_enabled():
            return
        payload = {
            "event": "dual_preprocessor_summary",
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **fields,
        }
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    def data_transformation(self, mic_sig_batch=None, acoustic_scene_batch=None, vad_batch=None):
        output = []
        debug_enabled = self._debug_enabled()
        total_start = time.perf_counter() if debug_enabled else None
        phat_ms = None
        lms_ms = None
        extras_ms = None
        doa_ms = None

        if mic_sig_batch is not None:
            self._debug_event("before_ensure_mic_tensor")
            mic_sig_batch = ensure_mic_tensor(mic_sig_batch)
            self._debug_event("after_ensure_mic_tensor", shape=tuple(mic_sig_batch.shape))
            if self.cuda_activated:
                mic_sig_batch = mic_sig_batch.cuda()
                self._debug_event("after_mic_to_cuda", device=str(mic_sig_batch.device))

            self._debug_event("before_phat")
            phat_start = time.perf_counter() if debug_enabled else None
            phat_maps = self.phat(mic_sig_batch)
            if self._debug_verbose() and phat_maps.is_cuda:
                torch.cuda.synchronize(phat_maps.device)
            if phat_start is not None:
                phat_ms = (time.perf_counter() - phat_start) * 1000.0
            self._debug_event("after_phat", shape=tuple(phat_maps.shape), device=str(phat_maps.device))

            self._debug_event("before_lms")
            lms_start = time.perf_counter() if debug_enabled else None
            lms_maps = self.lms(mic_sig_batch)
            if self._debug_verbose() and lms_maps.is_cuda:
                torch.cuda.synchronize(lms_maps.device)
            if lms_start is not None:
                lms_ms = (time.perf_counter() - lms_start) * 1000.0
            self._debug_event("after_lms", shape=tuple(lms_maps.shape), device=str(lms_maps.device))

            maps = torch.cat((phat_maps, lms_maps), dim=1)
            self._debug_event("after_concat", shape=tuple(maps.shape))
            extras_start = time.perf_counter() if debug_enabled else None
            maps = self.apply_extras(maps, acoustic_scene_batch, vad_batch)
            if self._debug_verbose() and maps.is_cuda:
                torch.cuda.synchronize(maps.device)
            if extras_start is not None:
                extras_ms = (time.perf_counter() - extras_start) * 1000.0
            self._debug_event("after_apply_extras", shape=tuple(maps.shape), device=str(maps.device))
            output.append(maps)

        if acoustic_scene_batch is not None:
            self._debug_event("before_doa_tensor", batch_size=len(acoustic_scene_batch))
            doa_start = time.perf_counter() if debug_enabled else None
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
            if doa_start is not None:
                doa_ms = (time.perf_counter() - doa_start) * 1000.0
            self._debug_event("after_doa_tensor", shape=tuple(doa_batch.shape), device=str(doa_batch.device))
            output.append(doa_batch)

        if total_start is not None:
            total_ms = (time.perf_counter() - total_start) * 1000.0
            maps_shape = tuple(output[0].shape) if output and isinstance(output[0], torch.Tensor) else None
            self._debug_summary(
                mode=self._debug_mode(),
                mic_shape=tuple(mic_sig_batch.shape) if mic_sig_batch is not None else None,
                maps_shape=maps_shape,
                doa_shape=tuple(doa_batch.shape) if acoustic_scene_batch is not None else None,
                phat_ms=round(phat_ms, 3) if phat_ms is not None else None,
                lms_ms=round(lms_ms, 3) if lms_ms is not None else None,
                apply_extras_ms=round(extras_ms, 3) if extras_ms is not None else None,
                doa_ms=round(doa_ms, 3) if doa_ms is not None else None,
                total_ms=round(total_ms, 3),
            )

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
