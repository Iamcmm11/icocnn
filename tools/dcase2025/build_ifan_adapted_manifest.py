#!/usr/bin/env python3
"""Build an IFAN-adapted DCASE training manifest by rebalancing clip distribution."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ANGLE_BINS: tuple[tuple[float, float, str], ...] = (
    (-90.0, -60.0, "[-90,-60)"),
    (-60.0, -30.0, "[-60,-30)"),
    (-30.0, 0.0, "[-30,0)"),
    (0.0, 30.0, "[0,30)"),
    (30.0, 60.0, "[30,60)"),
    (60.0, 90.0, "[60,90]"),
)

MANIFEST_FIELDS = [
    "subset",
    "clip_id",
    "split",
    "audio_relpath",
    "metadata_relpath",
    "active_frames",
    "active_duration_s",
    "classes",
    "class_names",
    "sources",
    "source_count",
    "max_frame_source_count",
    "azimuth_mean_deg",
    "azimuth_min_deg",
    "azimuth_max_deg",
    "azimuth_span_deg",
    "distance_mean_cm",
    "distance_min_cm",
    "distance_max_cm",
    "distance_span_cm",
    "rows",
    "angle_bin",
    "onscreen_ratio",
    "resample_source",
    "repeat_index",
]


def add_bool_argument(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help_text: str,
) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an IFAN-adapted DCASE manifest by oversampling moving clips "
            "and balancing folded-azimuth bins."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets/DCASE2025_Task3",
        help="Dataset root containing metadata_dev and audio referenced by the manifest.",
    )
    parser.add_argument(
        "--input-manifest",
        default="datasets/DCASE2025_Task3/locata_like_all_classes/manifest_all.csv",
        help="Source manifest to rebalance.",
    )
    parser.add_argument(
        "--output-root",
        default="datasets/DCASE2025_Task3/ifan_adapted_all_classes_balanced",
        help="Directory where the adapted manifest and summary are written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev-train-sony", "dev-train-tau"],
        help="Allowed splits from the source manifest.",
    )
    add_bool_argument(
        parser,
        "speech-only",
        default=False,
        help_text="Keep only female/male speech clips from the source manifest.",
    )
    parser.add_argument(
        "--onscreen-min-ratio",
        type=float,
        default=0.0,
        help="Drop clips whose active-frame onscreen ratio is lower than this threshold.",
    )
    add_bool_argument(
        parser,
        "balance-angle-bins",
        default=True,
        help_text="Oversample each subset so all folded-azimuth bins reach the subset maximum.",
    )
    parser.add_argument(
        "--target-moving-ratio",
        type=float,
        default=0.5,
        help="Target moving fraction after oversampling. Values > current moving ratio trigger moving oversampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for oversampling remainder draws.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clip_angle_bin(azimuth_mean_deg: float) -> str:
    angle = float(max(-90.0, min(90.0, azimuth_mean_deg)))
    for low, high, label in ANGLE_BINS:
        if low <= angle < high or (high == 90.0 and angle <= high):
            return label
    return "out_of_range"


def metadata_onscreen_ratio(dataset_root: Path, metadata_relpath: str) -> float:
    metadata_path = dataset_root / metadata_relpath
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 0.0
    onscreen = sum(int(row["onscreen"]) for row in rows)
    return float(onscreen) / float(len(rows))


def duplicate_rows(
    rows: list[dict[str, str]],
    *,
    target_count: int,
    rng: random.Random,
    resample_source: str,
) -> list[dict[str, str]]:
    if not rows:
        return []
    if len(rows) >= target_count:
        out: list[dict[str, str]] = []
        for repeat_index, row in enumerate(rows[:target_count]):
            item = dict(row)
            item["repeat_index"] = str(repeat_index)
            item["resample_source"] = resample_source
            out.append(item)
        return out
    repeats, remainder = divmod(target_count, len(rows))
    out: list[dict[str, str]] = []
    for repeat_index in range(repeats):
        for row in rows:
            item = dict(row)
            item["repeat_index"] = str(repeat_index)
            item["resample_source"] = resample_source
            out.append(item)
    if remainder > 0:
        for row in rng.sample(rows, remainder):
            item = dict(row)
            item["repeat_index"] = str(repeats)
            item["resample_source"] = resample_source
            out.append(item)
    return out


def balance_subset_angle_bins(
    rows: list[dict[str, str]],
    *,
    rng: random.Random,
    resample_source: str,
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["angle_bin"]].append(row)
    if not grouped:
        return []
    target_count = max(len(group) for group in grouped.values())
    balanced: list[dict[str, str]] = []
    for _, _, label in ANGLE_BINS:
        balanced.extend(
            duplicate_rows(
                grouped.get(label, []),
                target_count=target_count,
                rng=rng,
                resample_source=resample_source,
            )
        )
    return balanced


def oversample_moving(
    rows: list[dict[str, str]],
    *,
    target_ratio: float,
    rng: random.Random,
) -> list[dict[str, str]]:
    moving = [row for row in rows if row["subset"] == "moving_single_source"]
    static = [row for row in rows if row["subset"] == "static_single_source"]
    if not moving or target_ratio <= 0.0 or target_ratio >= 1.0:
        return rows
    desired_moving = int(round(len(static) * (target_ratio / max(1.0 - target_ratio, 1e-6))))
    if desired_moving <= len(moving):
        return rows
    extra = duplicate_rows(
        moving,
        target_count=desired_moving,
        rng=rng,
        resample_source="moving_ratio_balance",
    )
    kept = static + extra
    kept.sort(key=lambda row: (row["subset"], row["split"], row["clip_id"], row["repeat_index"]))
    return kept


def write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    by_subset = Counter(row["subset"] for row in rows)
    by_split = Counter(row["split"] for row in rows)
    by_class = Counter(row["class_names"] for row in rows)
    by_angle = Counter(f"{row['subset']}/{row['angle_bin']}" for row in rows)
    return {
        "count": len(rows),
        "by_subset": dict(sorted(by_subset.items())),
        "by_split": dict(sorted(by_split.items())),
        "top_class_names": by_class.most_common(10),
        "by_subset_and_angle_bin": dict(sorted(by_angle.items())),
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    input_manifest = Path(args.input_manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    rng = random.Random(args.seed)

    rows = load_manifest(input_manifest)
    selected: list[dict[str, str]] = []
    rejected = Counter()
    for row in rows:
        if row["split"] not in set(args.splits):
            rejected["split"] += 1
            continue
        class_names = set(str(row["class_names"]).split("|"))
        if args.speech_only and not class_names.issubset({"female_speech", "male_speech"}):
            rejected["not_speech_only"] += 1
            continue
        onscreen_ratio = metadata_onscreen_ratio(dataset_root, str(row["metadata_relpath"]))
        if onscreen_ratio < float(args.onscreen_min_ratio):
            rejected["onscreen_ratio"] += 1
            continue
        item = dict(row)
        item["angle_bin"] = clip_angle_bin(float(item["azimuth_mean_deg"]))
        item["onscreen_ratio"] = f"{onscreen_ratio:.6f}"
        item["repeat_index"] = "0"
        item["resample_source"] = "original"
        selected.append(item)

    original_summary = summarize_rows(selected)

    if args.balance_angle_bins:
        balanced: list[dict[str, str]] = []
        for subset in ("static_single_source", "moving_single_source"):
            subset_rows = [row for row in selected if row["subset"] == subset]
            balanced.extend(
                balance_subset_angle_bins(
                    subset_rows,
                    rng=rng,
                    resample_source=f"angle_balance_{subset}",
                )
            )
        selected = balanced

    selected = oversample_moving(
        selected,
        target_ratio=float(args.target_moving_ratio),
        rng=rng,
    )
    selected.sort(key=lambda row: (row["subset"], row["split"], row["clip_id"], row["repeat_index"]))

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest_all.csv"
    write_manifest(manifest_path, selected)

    summary = {
        "kind": "dcase2025_ifan_adapted_manifest",
        "dataset_root": str(dataset_root),
        "input_manifest": str(input_manifest),
        "output_root": str(output_root),
        "parameters": {
            "splits": list(args.splits),
            "speech_only": bool(args.speech_only),
            "onscreen_min_ratio": float(args.onscreen_min_ratio),
            "balance_angle_bins": bool(args.balance_angle_bins),
            "target_moving_ratio": float(args.target_moving_ratio),
            "seed": int(args.seed),
        },
        "rejected": dict(sorted(rejected.items())),
        "source_summary": original_summary,
        "adapted_summary": summarize_rows(selected),
        "manifest": str(manifest_path),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
