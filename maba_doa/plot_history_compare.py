"""Plot and summarize training histories across multiple MABA-DOA runs."""

import argparse
import csv
import glob
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

KNOWN_VARIANTS = (
    "baseline",
    "maba",
    "ablation_no_gate",
    "ablation_no_state",
)


def _read_history_csv(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": float(row["epoch"]),
                    "test_loss": float(row["test_loss"]),
                    "test_rmsae_deg": float(row["test_rmsae_deg"]),
                    "epoch_time_s": float(row.get("epoch_time_s", 0.0)),
                    "lr": float(row.get("lr", 0.0)),
                }
            )
    return rows


def _read_summary(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_variant_from_dirname(run_dir: str) -> Optional[str]:
    base = os.path.basename(os.path.normpath(run_dir))
    for variant in sorted(KNOWN_VARIANTS, key=len, reverse=True):
        token = "_{}_".format(variant)
        if token in "_{}_".format(base):
            return variant
    return None


def _variant_for_run(run_dir: str, summary: Dict) -> Optional[str]:
    variant = summary.get("variant")
    if variant:
        return str(variant)
    return _infer_variant_from_dirname(run_dir)


def _history_mtime(run_dir: str) -> float:
    history_path = os.path.join(run_dir, "history.csv")
    if os.path.exists(history_path):
        return os.path.getmtime(history_path)
    return os.path.getmtime(run_dir)


def _find_latest_run_for_variant(output_root: str, variant: str) -> str:
    pattern = os.path.join(output_root, "*")
    candidates = []
    for p in glob.glob(pattern):
        if not os.path.isdir(p):
            continue
        history_path = os.path.join(p, "history.csv")
        if not os.path.exists(history_path):
            continue
        summary = _read_summary(os.path.join(p, "summary.json"))
        if _variant_for_run(p, summary) == variant:
            candidates.append(p)
    if not candidates:
        return ""
    candidates.sort(key=_history_mtime, reverse=True)
    return candidates[0]


def _collect_run_dirs(args) -> List[str]:
    run_dirs: List[str] = []
    if args.run_dirs:
        run_dirs.extend(args.run_dirs)
    if args.glob_pattern:
        run_dirs.extend([p for p in glob.glob(args.glob_pattern) if os.path.isdir(p)])
    if not run_dirs:
        for v in args.variants:
            run_dir = _find_latest_run_for_variant(args.output_root, v)
            if run_dir:
                run_dirs.append(run_dir)
    # De-duplicate while preserving order.
    unique = []
    seen = set()
    for d in run_dirs:
        abs_d = os.path.abspath(d)
        if abs_d not in seen:
            seen.add(abs_d)
            unique.append(abs_d)
    return unique


def _label_for_run(run_dir: str, summary: Dict) -> str:
    return _variant_for_run(run_dir, summary) or os.path.basename(os.path.normpath(run_dir))


def _display_labels(records: List[Tuple[str, str, List[Dict[str, float]], Dict]]) -> List[str]:
    raw_labels = [label for label, _, _, _ in records]
    counts = Counter(raw_labels)
    display_labels = []
    for label, run_dir, _, _ in records:
        if counts[label] > 1:
            display_labels.append("{} ({})".format(label, os.path.basename(os.path.normpath(run_dir))))
        else:
            display_labels.append(label)
    return display_labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple maba_doa run histories and draw learning curves."
    )
    parser.add_argument("--output-root", type=str, default="maba_doa/outputs")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(KNOWN_VARIANTS),
        help="Variants to auto-pick latest runs when --run-dirs is not provided.",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=None,
        help="Explicit run directories containing history.csv.",
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default=None,
        help="Optional glob pattern to collect run directories.",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="maba_doa/outputs/history_compare.png",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="maba_doa/outputs/history_compare_summary.csv",
    )
    parser.add_argument(
        "--out-history-csv",
        type=str,
        default="maba_doa/outputs/history_compare_merged.csv",
        help="Long-form merged per-epoch history table.",
    )
    args = parser.parse_args()

    run_dirs = _collect_run_dirs(args)
    if not run_dirs:
        raise RuntimeError("No run directories found. Provide --run-dirs or valid --output-root.")

    records: List[Tuple[str, str, List[Dict[str, float]], Dict]] = []
    for run_dir in run_dirs:
        history_path = os.path.join(run_dir, "history.csv")
        if not os.path.exists(history_path):
            continue
        summary = _read_summary(os.path.join(run_dir, "summary.json"))
        label = _label_for_run(run_dir, summary)
        history = _read_history_csv(history_path)
        if history:
            records.append((label, run_dir, history, summary))

    if not records:
        raise RuntimeError("Found run directories but no valid history.csv data.")

    os.makedirs(os.path.dirname(args.out_plot), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_history_csv), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_loss, ax_rmsae = axes
    ax_loss.set_title("Test Loss vs Epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Test Loss")
    ax_loss.grid(True, alpha=0.25)

    ax_rmsae.set_title("Test RMSAE vs Epoch")
    ax_rmsae.set_xlabel("Epoch")
    ax_rmsae.set_ylabel("RMSAE (deg)")
    ax_rmsae.grid(True, alpha=0.25)

    summary_rows = []
    merged_history_rows = []
    display_labels = _display_labels(records)
    for display_label, (label, run_dir, history, summary) in zip(display_labels, records):
        variant = _variant_for_run(run_dir, summary) or label
        epochs = [r["epoch"] for r in history]
        losses = [r["test_loss"] for r in history]
        rmsaes = [r["test_rmsae_deg"] for r in history]
        ax_loss.plot(epochs, losses, marker="o", label=display_label)
        ax_rmsae.plot(epochs, rmsaes, marker="o", label=display_label)

        best_idx = min(range(len(rmsaes)), key=lambda i: rmsaes[i])
        avg_epoch_time_s = sum(r["epoch_time_s"] for r in history) / len(history)
        summary_rows.append(
            {
                "label": display_label,
                "variant": variant,
                "run_dir": run_dir,
                "epochs": int(epochs[-1]),
                "final_loss": losses[-1],
                "best_loss": min(losses),
                "final_rmsae_deg": rmsaes[-1],
                "best_rmsae_deg": rmsaes[best_idx],
                "best_epoch": int(epochs[best_idx]),
                "avg_epoch_time_s": avg_epoch_time_s,
                "latency_step_ms": summary.get("latency_step_ms", ""),
                "param_count": summary.get("param_count", ""),
                "maba_mac_proxy": summary.get("maba_mac_proxy", ""),
            }
        )
        for row in history:
            merged_history_rows.append(
                {
                    "label": display_label,
                    "variant": variant,
                    "run_dir": run_dir,
                    **row,
                }
            )

    ax_loss.legend()
    ax_rmsae.legend()
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=220)

    fieldnames = [
        "label",
        "variant",
        "run_dir",
        "epochs",
        "final_loss",
        "best_loss",
        "final_rmsae_deg",
        "best_rmsae_deg",
        "best_epoch",
        "avg_epoch_time_s",
        "latency_step_ms",
        "param_count",
        "maba_mac_proxy",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    merged_fieldnames = [
        "label",
        "variant",
        "run_dir",
        "epoch",
        "test_loss",
        "test_rmsae_deg",
        "epoch_time_s",
        "lr",
    ]
    with open(args.out_history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
        writer.writeheader()
        writer.writerows(merged_history_rows)

    print("Saved plot:", args.out_plot)
    print("Saved summary csv:", args.out_csv)
    print("Saved merged history csv:", args.out_history_csv)
    for row in summary_rows:
        print(
            "[{label}] epochs={epochs} final_rmsae={final_rmsae_deg:.4f} "
            "best_rmsae={best_rmsae_deg:.4f}@{best_epoch} final_loss={final_loss:.6f}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
