#!/usr/bin/env python3
"""Build LOCATA-like single-source manifests from DCASE2025 Task3 dev data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


CLASS_NAMES = {
    0: "female_speech",
    1: "male_speech",
    2: "clapping",
    3: "telephone",
    4: "laughter",
    5: "domestic_sounds",
    6: "footsteps",
    7: "door",
    8: "music",
    9: "musical_instrument",
    10: "water_tap",
    11: "bell",
    12: "knock",
}
SPEECH_CLASSES = {0, 1}


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
            "Filter DCASE2025 Task3 metadata into LOCATA-like single-source "
            "static/moving clip manifests."
        )
    )
    parser.add_argument(
        "--root",
        default="datasets/DCASE2025_Task3",
        help="Dataset root containing metadata_dev and stereo_dev.",
    )
    parser.add_argument(
        "--output-root",
        default="datasets/DCASE2025_Task3/locata_like",
        help="Where manifests and symlink directories are written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev-test-sony", "dev-test-tau", "dev-train-sony", "dev-train-tau"],
        help="DCASE splits to scan.",
    )
    parser.add_argument(
        "--min-active-frames",
        type=int,
        default=45,
        help="Require at least this many annotated 100 ms frames.",
    )
    parser.add_argument(
        "--static-azimuth-span-deg",
        type=float,
        default=5.0,
        help="Static clips must have azimuth span no larger than this.",
    )
    parser.add_argument(
        "--static-distance-span-cm",
        type=float,
        default=20.0,
        help="Static clips must have distance span no larger than this.",
    )
    parser.add_argument(
        "--moving-azimuth-span-deg",
        type=float,
        default=10.0,
        help="Moving clips must have azimuth span at least this, unless distance qualifies.",
    )
    parser.add_argument(
        "--moving-distance-span-cm",
        type=float,
        default=50.0,
        help="Moving clips must have distance span at least this, unless azimuth qualifies.",
    )
    add_bool_argument(
        parser,
        "speech-only",
        default=True,
        help_text="Keep only clips whose metadata classes are female/male speech.",
    )
    add_bool_argument(
        parser,
        "single-source-id",
        default=False,
        help_text=(
            "Require one source ID across the whole clip. This is stricter than "
            "single_position_per_frame and better matches LOCATA single-source tracks."
        ),
    )
    add_bool_argument(
        parser,
        "make-links",
        default=True,
        help_text="Create symlink trees for selected clips.",
    )
    return parser.parse_args()


def read_metadata(path: Path) -> list[dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{key: int(value) for key, value in row.items()} for row in csv.DictReader(f)]


def unique_positions(rows: Iterable[dict[str, int]]) -> set[tuple[int, int]]:
    return {(row["azimuth"], row["distance"]) for row in rows}


def circular_span_deg(values: list[float]) -> float:
    """Return the minimum angular span covering all values, in degrees."""
    if not values:
        return 0.0
    wrapped = sorted(value % 360.0 for value in values)
    if len(wrapped) == 1:
        return 0.0
    gaps = [
        wrapped[index + 1] - wrapped[index]
        for index in range(len(wrapped) - 1)
    ]
    gaps.append(wrapped[0] + 360.0 - wrapped[-1])
    return 360.0 - max(gaps)


def circular_mean_deg(values: list[float]) -> float:
    if not values:
        return 0.0
    radians = [math.radians(value) for value in values]
    sin_mean = sum(math.sin(value) for value in radians) / len(radians)
    cos_mean = sum(math.cos(value) for value in radians) / len(radians)
    return ((math.degrees(math.atan2(sin_mean, cos_mean)) + 180.0) % 360.0) - 180.0


def analyze_metadata(metadata_path: Path, audio_path: Path, root: Path) -> dict[str, object]:
    rows = read_metadata(metadata_path)
    by_frame: dict[int, list[dict[str, int]]] = defaultdict(list)
    for row in rows:
        by_frame[row["frame"]].append(row)

    active_frames = sorted(by_frame)
    classes = sorted({row["class"] for row in rows})
    sources = sorted({row["source"] for row in rows})
    per_frame_positions = {
        frame: unique_positions(frame_rows)
        for frame, frame_rows in by_frame.items()
    }
    single_position_per_frame = bool(active_frames) and all(
        len(points) == 1 for points in per_frame_positions.values()
    )

    azimuths: list[float] = []
    distances: list[float] = []
    if single_position_per_frame:
        for frame in active_frames:
            azimuth, distance = next(iter(per_frame_positions[frame]))
            azimuths.append(float(azimuth))
            distances.append(float(distance))

    class_counts = Counter(row["class"] for row in rows)
    frame_source_counts = [
        len({row["source"] for row in by_frame[frame]})
        for frame in active_frames
    ]

    return {
        "clip_id": metadata_path.stem,
        "split": metadata_path.parent.name,
        "metadata_path": str(metadata_path),
        "audio_path": str(audio_path),
        "metadata_relpath": str(metadata_path.relative_to(root)),
        "audio_relpath": str(audio_path.relative_to(root)),
        "rows": len(rows),
        "active_frames": len(active_frames),
        "active_duration_s": len(active_frames) * 0.1,
        "classes": classes,
        "class_names": [CLASS_NAMES.get(class_id, str(class_id)) for class_id in classes],
        "class_counts": {str(key): int(value) for key, value in sorted(class_counts.items())},
        "sources": sources,
        "source_count": len(sources),
        "max_frame_source_count": max(frame_source_counts) if frame_source_counts else 0,
        "single_position_per_frame": single_position_per_frame,
        "speech_only": bool(rows) and set(classes).issubset(SPEECH_CLASSES),
        "azimuth_mean_deg": circular_mean_deg(azimuths),
        "azimuth_min_deg": min(azimuths) if azimuths else None,
        "azimuth_max_deg": max(azimuths) if azimuths else None,
        "azimuth_span_deg": circular_span_deg(azimuths),
        "distance_mean_cm": sum(distances) / len(distances) if distances else None,
        "distance_min_cm": min(distances) if distances else None,
        "distance_max_cm": max(distances) if distances else None,
        "distance_span_cm": max(distances) - min(distances) if distances else None,
    }


def classify_record(record: dict[str, object], args: argparse.Namespace) -> str | None:
    if not record["single_position_per_frame"]:
        return None
    if int(record["active_frames"]) < args.min_active_frames:
        return None
    if args.speech_only and not record["speech_only"]:
        return None
    if args.single_source_id and int(record["source_count"]) != 1:
        return None

    az_span = float(record["azimuth_span_deg"])
    dist_span = float(record["distance_span_cm"])
    is_static = (
        az_span <= args.static_azimuth_span_deg
        and dist_span <= args.static_distance_span_cm
    )
    is_moving = (
        az_span >= args.moving_azimuth_span_deg
        or dist_span >= args.moving_distance_span_cm
    )

    if is_moving:
        return "moving_single_source"
    if is_static:
        return "static_single_source"
    return None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
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
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, tuple)):
                    value = "|".join(str(item) for item in value)
                out[key] = value
            writer.writerow(out)


def recreate_link(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and Path(os.readlink(link_path)) == target_path:
            return
        link_path.unlink()
    link_path.symlink_to(target_path)


def make_links(output_root: Path, dataset_root: Path, records: list[dict[str, object]]) -> None:
    for row in records:
        subset = str(row["subset"])
        split = str(row["split"])
        audio_path = dataset_root / str(row["audio_relpath"])
        metadata_path = dataset_root / str(row["metadata_relpath"])
        stem = str(row["clip_id"])
        recreate_link(
            output_root / subset / "audio" / split / f"{stem}.wav",
            audio_path.resolve(),
        )
        recreate_link(
            output_root / subset / "metadata" / split / f"{stem}.csv",
            metadata_path.resolve(),
        )


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    metadata_root = dataset_root / "metadata_dev"
    audio_root = dataset_root / "stereo_dev"

    if not metadata_root.is_dir():
        raise FileNotFoundError(f"Missing metadata directory: {metadata_root}")
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Missing stereo audio directory: {audio_root}")

    selected: list[dict[str, object]] = []
    scanned = 0
    missing_audio = 0
    rejected = Counter()
    by_split = Counter()

    for split in args.splits:
        split_metadata_root = metadata_root / split
        if not split_metadata_root.is_dir():
            raise FileNotFoundError(f"Missing split metadata directory: {split_metadata_root}")
        for metadata_path in sorted(split_metadata_root.glob("*.csv")):
            scanned += 1
            audio_path = audio_root / split / f"{metadata_path.stem}.wav"
            if not audio_path.is_file():
                missing_audio += 1
                continue
            record = analyze_metadata(metadata_path, audio_path, dataset_root)
            subset = classify_record(record, args)
            if subset is None:
                if not record["single_position_per_frame"]:
                    rejected["multi_position"] += 1
                elif int(record["active_frames"]) < args.min_active_frames:
                    rejected["short_active"] += 1
                elif args.speech_only and not record["speech_only"]:
                    rejected["not_speech_only"] += 1
                elif args.single_source_id and int(record["source_count"]) != 1:
                    rejected["multiple_source_ids"] += 1
                else:
                    rejected["middle_motion"] += 1
                continue
            record["subset"] = subset
            selected.append(record)
            by_split[(subset, split)] += 1

    selected.sort(key=lambda row: (str(row["subset"]), str(row["split"]), str(row["clip_id"])))
    static_rows = [row for row in selected if row["subset"] == "static_single_source"]
    moving_rows = [row for row in selected if row["subset"] == "moving_single_source"]

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "manifest_all.csv", selected)
    write_csv(output_root / "manifest_static_single_source.csv", static_rows)
    write_csv(output_root / "manifest_moving_single_source.csv", moving_rows)

    summary = {
        "kind": "dcase2025_task3_locata_like_filter",
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "parameters": {
            "splits": args.splits,
            "min_active_frames": args.min_active_frames,
            "static_azimuth_span_deg": args.static_azimuth_span_deg,
            "static_distance_span_cm": args.static_distance_span_cm,
            "moving_azimuth_span_deg": args.moving_azimuth_span_deg,
            "moving_distance_span_cm": args.moving_distance_span_cm,
            "speech_only": args.speech_only,
            "single_source_id": args.single_source_id,
        },
        "scanned_metadata_files": scanned,
        "missing_audio_files": missing_audio,
        "selected_total": len(selected),
        "selected_by_subset": {
            "static_single_source": len(static_rows),
            "moving_single_source": len(moving_rows),
        },
        "selected_by_subset_and_split": {
            f"{subset}/{split}": count
            for (subset, split), count in sorted(by_split.items())
        },
        "rejected": dict(sorted(rejected.items())),
        "manifests": {
            "all": str(output_root / "manifest_all.csv"),
            "static_single_source": str(output_root / "manifest_static_single_source.csv"),
            "moving_single_source": str(output_root / "manifest_moving_single_source.csv"),
        },
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.make_links:
        make_links(output_root, dataset_root, selected)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
