from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gpuRIR

import acousticTrackingModels as at_models

from ifan_edge.eval import run_engineering_check
from ifan_edge.models import IFANModel, IFANModelConfig, PAPER_IFAN_BRANCH_CHANNELS


BASELINE_PARAM_TARGET = 290_017
IFAN_PARAM_TARGET = 125_440
IFAN_PARAM_TOLERANCE = 0.05
ENGINEERING_INPUT_SHAPE = (1, 2, 3, 5, 4, 8)
PAPER_STYLE_CONVICO_KERNEL_TAPS = 9
PAPER_STYLE_FLOPS_PER_MAC = 2
PAPER_STYLE_BASELINE_REFERENCE = 74_770_000
PAPER_STYLE_IFAN_REFERENCE = 6_360_000


def count_named_parameters(model, trainable_only: bool = True) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        rows.append(
            {
                "name": name,
                "shape": list(param.shape),
                "numel": int(param.numel()),
            }
        )
    return rows


def summarize_top_level(rows: list[dict[str, object]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        prefix = str(row["name"]).split(".", 1)[0]
        summary[prefix] = summary.get(prefix, 0) + int(row["numel"])
    return dict(sorted(summary.items()))


def ratio_dict(counts: dict[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {key: 0.0 for key in counts}
    return {key: float(value) / float(total) for key, value in counts.items()}


def target_gap_percent(actual: int, target: int) -> float:
    return 100.0 * (float(actual) - float(target)) / float(target)


def within_target_band(actual: int, target: int, tolerance: float) -> bool:
    lower = target * (1.0 - tolerance)
    upper = target * (1.0 + tolerance)
    return lower <= actual <= upper


def paper_style_convico_mac(charts: int, height: int, width: int, cin: int, cout: int, rin: int) -> int:
    return int(charts) * int(height) * int(width) * int(cin) * int(cout) * int(rin) * PAPER_STYLE_CONVICO_KERNEL_TAPS


def paper_style_temporal_conv_mac(positions: int, cin: int, cout: int, kernel_size: int = 5) -> int:
    return int(positions) * int(cin) * int(cout) * int(kernel_size)


def paper_style_depthwise_separable_temporal_conv_mac(
    positions: int,
    cin: int,
    cout: int,
    kernel_size: int = 5,
) -> int:
    return int(positions) * (int(cin) * int(kernel_size) + int(cin) * int(cout))


def paper_style_totals(breakdown: dict[str, int], reference: int) -> dict[str, object]:
    mac_total = sum(breakdown.values())
    flops_total = mac_total * PAPER_STYLE_FLOPS_PER_MAC
    return {
        "convention": {
            "scope": "DNN backbone only",
            "count_unit": "single output frame",
            "convico_kernel_taps": PAPER_STYLE_CONVICO_KERNEL_TAPS,
            "convico_counts_rout_expansion": False,
            "includes_frontend": False,
            "includes_norm_pool_activation_softargmax": False,
            "flops_per_mac": PAPER_STYLE_FLOPS_PER_MAC,
        },
        "reference_value": reference,
        "mac_proxy_total": mac_total,
        "flops_proxy_total": flops_total,
        "reference_gap_percent_vs_mac": target_gap_percent(mac_total, reference),
        "reference_gap_percent_vs_flops": target_gap_percent(flops_total, reference),
        "breakdown": breakdown,
    }


def baseline_paper_style_summary() -> dict[str, object]:
    channels = 32
    breakdown = {
        "ico_conv_0": paper_style_convico_mac(charts=5, height=4, width=8, cin=1, cout=channels, rin=1),
        "ico_conv_1": paper_style_convico_mac(charts=5, height=4, width=8, cin=channels, cout=channels, rin=6),
        "ico_conv_2_to_6": 5 * paper_style_convico_mac(charts=5, height=2, width=4, cin=channels, cout=channels, rin=6),
        "temp_conv_0": paper_style_temporal_conv_mac(positions=6 * 5 * 4 * 8, cin=channels, cout=channels),
        "temp_conv_1": paper_style_temporal_conv_mac(positions=6 * 5 * 4 * 8, cin=channels, cout=channels),
        "temp_conv_2_to_5": 4 * paper_style_temporal_conv_mac(positions=6 * 5 * 2 * 4, cin=channels, cout=channels),
        "temp_conv_6": paper_style_temporal_conv_mac(positions=6 * 5 * 2 * 4, cin=channels, cout=1),
    }
    return paper_style_totals(breakdown, reference=PAPER_STYLE_BASELINE_REFERENCE)


def ifan_paper_style_summary(config: IFANModelConfig) -> dict[str, object]:
    charts = 5
    height = 2**config.r
    width = 2 ** (config.r + 1)
    pooled_height = max(height // 2, 1) if config.pre_fusion_pooling and config.r > 1 else height
    pooled_width = max(width // 2, 1) if config.pre_fusion_pooling and config.r > 1 else width
    channels = config.branch_channels

    temporal_fn = (
        paper_style_depthwise_separable_temporal_conv_mac
        if config.temporal_conv_variant == "depthwise_separable_1d"
        else paper_style_temporal_conv_mac
    )

    breakdown = {
        "phat_stem": paper_style_convico_mac(charts=charts, height=height, width=width, cin=config.phat_in_channels, cout=channels, rin=1),
        "lms_stem": paper_style_convico_mac(charts=charts, height=height, width=width, cin=config.aux_in_channels, cout=channels, rin=1),
        "phat_residual": 2 * paper_style_convico_mac(charts=charts, height=height, width=width, cin=channels, cout=channels, rin=6),
        "lms_residual": 2 * paper_style_convico_mac(charts=charts, height=height, width=width, cin=channels, cout=channels, rin=6),
        "shared_attention_conv1": paper_style_convico_mac(charts=charts, height=height, width=width, cin=channels, cout=channels, rin=6),
        "shared_attention_conv2": paper_style_convico_mac(charts=charts, height=height, width=width, cin=channels, cout=channels, rin=6),
        "fusion_blocks_conv": 4 * paper_style_convico_mac(charts=charts, height=pooled_height, width=pooled_width, cin=channels, cout=channels, rin=6),
        "fusion_blocks_temporal": 4 * temporal_fn(positions=6 * charts * pooled_height * pooled_width, cin=channels, cout=channels),
        "final_head_conv": paper_style_convico_mac(charts=charts, height=pooled_height, width=pooled_width, cin=channels, cout=channels, rin=6),
        "final_head_temporal": temporal_fn(positions=6 * charts * pooled_height * pooled_width, cin=channels, cout=channels),
        "channel_readout": 6 * charts * pooled_height * pooled_width * channels,
    }

    return paper_style_totals(breakdown, reference=PAPER_STYLE_IFAN_REFERENCE)


def baseline_summary() -> dict[str, object]:
    model = at_models.IcoTempCNN(r=2, C=32, Cin=1, smooth_vertices=True)
    named = count_named_parameters(model)
    trainable = sum(int(row["numel"]) for row in named)
    total = sum(param.numel() for param in model.parameters())
    top_level = summarize_top_level(named)
    return {
        "model_name": "IcoTempCNN",
        "config": {
            "r": 2,
            "C": 32,
            "Cin": 1,
            "smooth_vertices": True,
        },
        "trainable_params": trainable,
        "total_params": total,
        "target_params": BASELINE_PARAM_TARGET,
        "target_gap_percent": target_gap_percent(trainable, BASELINE_PARAM_TARGET),
        "matches_expected_anchor": trainable == BASELINE_PARAM_TARGET,
        "top_level_breakdown": top_level,
        "top_level_ratio": ratio_dict(top_level, trainable),
        "paper_style_complexity": baseline_paper_style_summary(),
        "named_parameters": named,
    }


def ifan_summary(config: IFANModelConfig) -> dict[str, object]:
    model = IFANModel(config)
    named = count_named_parameters(model)
    trainable = model.count_parameters(trainable_only=True)
    total = model.count_parameters(trainable_only=False)
    breakdown = model.parameter_breakdown()
    breakdown_total = breakdown.pop("total")
    top_level = summarize_top_level(named)
    mac_proxy = model.mac_proxy(ENGINEERING_INPUT_SHAPE)
    mac_total = mac_proxy.pop("total")
    return {
        "model_name": "IFANModel",
        "config": {
            "r": config.r,
            "phat_in_channels": config.phat_in_channels,
            "aux_in_channels": config.aux_in_channels,
            "aux_feature_name": "LMS",
            "branch_channels": config.branch_channels,
            "pre_fusion_pooling": config.pre_fusion_pooling,
            "final_head_pooling": config.final_head_pooling,
            "smooth_vertices": config.smooth_vertices,
            "temporal_conv_variant": config.temporal_conv_variant,
        },
        "engineering_input_shape": list(ENGINEERING_INPUT_SHAPE),
        "trainable_params": trainable,
        "total_params": total,
        "target_params": IFAN_PARAM_TARGET,
        "target_gap_percent": target_gap_percent(trainable, IFAN_PARAM_TARGET),
        "within_target_band": within_target_band(trainable, IFAN_PARAM_TARGET, IFAN_PARAM_TOLERANCE),
        "module_breakdown": breakdown,
        "module_ratio": ratio_dict(breakdown, breakdown_total),
        "top_level_breakdown": top_level,
        "top_level_ratio": ratio_dict(top_level, trainable),
        "mac_proxy_total": mac_total,
        "mac_proxy_breakdown": mac_proxy,
        "paper_style_complexity": ifan_paper_style_summary(config),
        "named_parameters": named,
    }


def print_human_report(summary: dict[str, object]) -> None:
    print(f'Model: {summary["model_name"]}')
    print("Config:", json.dumps(summary["config"], ensure_ascii=False, sort_keys=True))
    print(f'Trainable params: {summary["trainable_params"]}')
    print(f'Total params: {summary["total_params"]}')
    if "target_params" in summary:
        print(f'Target params: {summary["target_params"]}')
        print(f'Target gap (%): {summary["target_gap_percent"]:.3f}')
    if "matches_expected_anchor" in summary:
        print(f'Anchor match: {summary["matches_expected_anchor"]}')
    if "within_target_band" in summary:
        print(f'Within target band: {summary["within_target_band"]}')
    if "module_breakdown" in summary:
        print("Module breakdown:")
        for name, value in summary["module_breakdown"].items():
            ratio = summary["module_ratio"][name] * 100.0
            print(f"  {name}: {value} ({ratio:.2f}%)")
    print("Top-level breakdown:")
    for name, value in summary["top_level_breakdown"].items():
        ratio = summary["top_level_ratio"].get(name, 0.0) * 100.0
        print(f"  {name}: {value} ({ratio:.2f}%)")
    if "paper_style_complexity" in summary:
        paper = summary["paper_style_complexity"]
        print("Paper-style complexity:")
        print(f'  MAC proxy total: {paper["mac_proxy_total"]}')
        print(f'  FLOPs proxy total: {paper["flops_proxy_total"]}')
        print(f'  Reference value: {paper["reference_value"]}')
        print(f'  Gap vs reference (MAC): {paper["reference_gap_percent_vs_mac"]:.3f}%')
        print(f'  Gap vs reference (FLOPs): {paper["reference_gap_percent_vs_flops"]:.3f}%')
        print("  Breakdown:")
        for name, value in paper["breakdown"].items():
            print(f"    {name}: {value}")
    if "mac_proxy_total" in summary:
        print(f'MAC proxy total: {summary["mac_proxy_total"]}')
        print("MAC proxy breakdown:")
        for name, value in summary["mac_proxy_breakdown"].items():
            print(f"  {name}: {value}")
    if "engineering_check" in summary:
        print("Engineering check:")
        engineering = summary["engineering_check"]
        print(f'  source_dataset: {engineering["source_dataset"]}')
        print(f'  input_shape: {engineering["input_shape"]}')
        print(f'  output_shape: {engineering["output_shape"]}')
        print(f'  loss: {engineering["loss"]}')
        print(f'  finite_output: {engineering["finite_output"]}')
        print(f'  finite_gradients: {engineering["finite_gradients"]}')
        print(f'  nonzero_gradient_params: {engineering["nonzero_gradient_params"]}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile stage-2 IFAN and root-level icoCNN parameter baselines.")
    parser.add_argument("--model", choices=("all", "baseline", "ifan"), default="all")
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--branch-channels", type=int, default=PAPER_IFAN_BRANCH_CHANNELS)
    parser.add_argument("--no-pre-fusion-pooling", action="store_true")
    parser.add_argument("--temporal-conv-variant", choices=("standard_1d", "depthwise_separable_1d"), default="standard_1d")
    parser.add_argument("--librispeech-path", default="datasets/LibriSpeech")
    parser.add_argument("--signal-length", type=int, default=2)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--step", type=int, default=3072)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--skip-engineering-check", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of the human-readable report.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reports: dict[str, dict[str, object]] = {}

    if args.model in ("all", "baseline"):
        reports["baseline"] = baseline_summary()
    if args.model in ("all", "ifan"):
        config = IFANModelConfig(
            r=args.r,
            phat_in_channels=1,
            aux_in_channels=1,
            branch_channels=args.branch_channels,
            pre_fusion_pooling=not args.no_pre_fusion_pooling,
            final_head_pooling=False,
            temporal_conv_variant=args.temporal_conv_variant,
        )
        reports["ifan"] = ifan_summary(config)
        if not args.skip_engineering_check:
            reports["ifan"]["engineering_check"] = run_engineering_check(
                config,
                librispeech_path=args.librispeech_path,
                signal_length=args.signal_length,
                k=args.k,
                step=args.step,
                fs=args.fs,
            )

    if args.json:
        print(json.dumps(reports, indent=2, ensure_ascii=False, sort_keys=True))
        return

    for name in ("baseline", "ifan"):
        if name not in reports:
            continue
        print_human_report(reports[name])
        if name == "baseline" and "ifan" in reports:
            print()


if __name__ == "__main__":
    main()
