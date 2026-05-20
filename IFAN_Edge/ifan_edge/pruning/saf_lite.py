from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn


def _is_convico_module(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return (
        isinstance(weight, torch.nn.Parameter)
        and weight.ndim == 4
        and all(hasattr(module, attr) for attr in ("Cin", "Cout", "Rin", "Rout", "r"))
    )


def iter_saf_lite_targets(
    model: nn.Module,
    *,
    target_channels: int = 16,
    block_size: int = 8,
) -> Iterable[tuple[str, nn.Module]]:
    """Yield hardware-regular CxC ConvIco layers; stem Cin=1 layers are skipped."""

    for name, module in model.named_modules():
        if not _is_convico_module(module):
            continue
        cin = int(getattr(module, "Cin"))
        cout = int(getattr(module, "Cout"))
        weight = getattr(module, "weight")
        if cin != target_channels or cout != target_channels:
            continue
        if weight.shape[0] != cout or weight.shape[1] != cin:
            continue
        if cin % block_size != 0:
            continue
        yield name, module


def build_saf_lite_mask(weight: torch.Tensor, *, keep_per_block: int, block_size: int = 8) -> torch.Tensor:
    if weight.ndim != 4:
        raise ValueError(f"Expected ConvIco weight [Cout, Cin, Rin, 7], got {tuple(weight.shape)}")
    cout, cin, _rin, _kernel = weight.shape
    if cin % block_size != 0:
        raise ValueError(f"Input channels {cin} must be divisible by block_size={block_size}.")
    if keep_per_block <= 0 or keep_per_block > block_size:
        raise ValueError(f"keep_per_block must be in [1, {block_size}], got {keep_per_block}.")

    scores = weight.detach().abs().sum(dim=(2, 3))
    mask = torch.zeros_like(weight, dtype=weight.dtype)
    for co in range(cout):
        for block_start in range(0, cin, block_size):
            block_scores = scores[co, block_start : block_start + block_size]
            keep_local = torch.topk(block_scores, k=keep_per_block, largest=True, sorted=True).indices
            keep_indices = block_start + keep_local
            mask[co, keep_indices, :, :] = 1
    return mask


def _retained_indices_from_mask(mask: torch.Tensor, *, block_size: int) -> list[list[list[int]]]:
    cout, cin, _rin, _kernel = mask.shape
    retained: list[list[list[int]]] = []
    channel_mask = mask[:, :, 0, 0].detach().cpu()
    for co in range(cout):
        row: list[list[int]] = []
        for block_start in range(0, cin, block_size):
            kept = torch.nonzero(channel_mask[co, block_start : block_start + block_size], as_tuple=False)
            row.append((kept.flatten() + block_start).tolist())
        retained.append(row)
    return retained


def _convico_dense_mac_proxy(module: nn.Module, *, time_steps: int, charts: int = 5, kernel_neighbors: int = 7) -> int:
    height = 2 ** int(getattr(module, "r"))
    width = 2 ** (int(getattr(module, "r")) + 1)
    return (
        int(time_steps)
        * int(charts)
        * int(height)
        * int(width)
        * int(getattr(module, "Cin"))
        * int(getattr(module, "Cout"))
        * int(getattr(module, "Rin"))
        * int(getattr(module, "Rout"))
        * int(kernel_neighbors)
    )


def _convico_effective_mac_proxy(
    module: nn.Module,
    mask: torch.Tensor,
    *,
    time_steps: int,
    charts: int = 5,
    kernel_neighbors: int = 7,
) -> int:
    height = 2 ** int(getattr(module, "r"))
    width = 2 ** (int(getattr(module, "r")) + 1)
    retained_pairs = int(mask[:, :, 0, 0].detach().sum().item())
    return (
        int(time_steps)
        * int(charts)
        * int(height)
        * int(width)
        * retained_pairs
        * int(getattr(module, "Rin"))
        * int(getattr(module, "Rout"))
        * int(kernel_neighbors)
    )


def build_saf_lite_pruning_summary(
    model: nn.Module,
    masks: dict[str, torch.Tensor],
    *,
    keep_per_block: int,
    block_size: int = 8,
    target_channels: int = 16,
    time_steps: int = 6,
    charts: int = 5,
    kernel_neighbors: int = 7,
) -> dict[str, Any]:
    module_map = dict(model.named_modules())
    layers: list[dict[str, Any]] = []
    dense_total = 0
    effective_total = 0
    weight_total = 0
    retained_weight_total = 0
    for name, mask in masks.items():
        module = module_map[name]
        dense_mac = _convico_dense_mac_proxy(
            module,
            time_steps=time_steps,
            charts=charts,
            kernel_neighbors=kernel_neighbors,
        )
        effective_mac = _convico_effective_mac_proxy(
            module,
            mask,
            time_steps=time_steps,
            charts=charts,
            kernel_neighbors=kernel_neighbors,
        )
        dense_total += dense_mac
        effective_total += effective_mac
        weights = int(mask.numel())
        retained_weights = int(mask.detach().sum().item())
        weight_total += weights
        retained_weight_total += retained_weights
        layers.append(
            {
                "name": name,
                "shape": list(mask.shape),
                "r": int(getattr(module, "r")),
                "Cin": int(getattr(module, "Cin")),
                "Cout": int(getattr(module, "Cout")),
                "Rin": int(getattr(module, "Rin")),
                "Rout": int(getattr(module, "Rout")),
                "retained_weights": retained_weights,
                "total_weights": weights,
                "weight_keep_ratio": retained_weights / float(weights),
                "dense_mac_proxy": dense_mac,
                "effective_mac_proxy": effective_mac,
                "mac_keep_ratio": effective_mac / float(dense_mac),
                "retained_input_indices_by_output_and_block": _retained_indices_from_mask(mask, block_size=block_size),
            }
        )

    return {
        "method": "saf_lite",
        "target": "icoCNN.ConvIco.weight",
        "target_channels": int(target_channels),
        "block_size": int(block_size),
        "keep_per_block": int(keep_per_block),
        "sparsity_per_block": 1.0 - (float(keep_per_block) / float(block_size)),
        "pruned_layer_count": len(layers),
        "layers": layers,
        "pruned_weight_total": weight_total,
        "retained_weight_total": retained_weight_total,
        "pruned_weight_keep_ratio": retained_weight_total / float(weight_total) if weight_total else 1.0,
        "pruned_weight_sparsity": 1.0 - (retained_weight_total / float(weight_total)) if weight_total else 0.0,
        "dense_ico_conv_mac_proxy": dense_total,
        "effective_ico_conv_mac_proxy": effective_total,
        "ico_conv_mac_keep_ratio": effective_total / float(dense_total) if dense_total else 1.0,
        "theoretical_pruned_ico_conv_compression": dense_total / float(effective_total) if effective_total else 0.0,
    }


@dataclass
class SAFLitePruner:
    masks: dict[str, torch.Tensor]
    keep_per_block: int
    block_size: int = 8
    target_channels: int = 16

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        keep_per_block: int,
        block_size: int = 8,
        target_channels: int = 16,
    ) -> "SAFLitePruner":
        masks = {
            name: build_saf_lite_mask(module.weight, keep_per_block=keep_per_block, block_size=block_size).detach().cpu()
            for name, module in iter_saf_lite_targets(
                model,
                target_channels=target_channels,
                block_size=block_size,
            )
        }
        if not masks:
            raise ValueError(
                "SAF-lite pruning found no target ConvIco layers. "
                f"Expected Cin=Cout={target_channels} and Cin divisible by {block_size}."
            )
        return cls(masks=masks, keep_per_block=keep_per_block, block_size=block_size, target_channels=target_channels)

    def apply(self, model: nn.Module) -> None:
        module_map = dict(model.named_modules())
        with torch.no_grad():
            for name, mask in self.masks.items():
                module = module_map[name]
                module.weight.mul_(mask.to(device=module.weight.device, dtype=module.weight.dtype))

    def mask_gradients(self, model: nn.Module) -> None:
        module_map = dict(model.named_modules())
        for name, mask in self.masks.items():
            weight = module_map[name].weight
            if weight.grad is not None:
                weight.grad.mul_(mask.to(device=weight.grad.device, dtype=weight.grad.dtype))

    def register_gradient_hooks(self, model: nn.Module) -> None:
        module_map = dict(model.named_modules())
        for name, mask in self.masks.items():
            weight = module_map[name].weight
            local_mask = mask.detach()
            weight.register_hook(
                lambda grad, local_mask=local_mask: grad
                * local_mask.to(device=grad.device, dtype=grad.dtype)
            )

    def optimizer_step(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.mask_gradients(model)
        optimizer.step()
        self.apply(model)

    def summary(self, model: nn.Module, *, time_steps: int = 6, charts: int = 5) -> dict[str, Any]:
        return build_saf_lite_pruning_summary(
            model,
            self.masks,
            keep_per_block=self.keep_per_block,
            block_size=self.block_size,
            target_channels=self.target_channels,
            time_steps=time_steps,
            charts=charts,
        )

    def save_artifacts(self, model: nn.Module, output_dir: str | Path, *, time_steps: int = 6, charts: int = 5) -> dict[str, str]:
        output_path = Path(output_dir)
        pruning_dir = output_path / "pruning"
        pruning_dir.mkdir(parents=True, exist_ok=True)
        mask_path = pruning_dir / "mask.pt"
        summary_path = pruning_dir / "pruning_summary.json"
        torch.save({name: mask.cpu() for name, mask in self.masks.items()}, mask_path)
        summary = self.summary(model, time_steps=time_steps, charts=charts)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "mask_path": str(mask_path),
            "summary_path": str(summary_path),
        }
