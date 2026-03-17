import argparse
import csv
import datetime
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def find_first(path: Path, patterns):
    for pattern in patterns:
        found = sorted(path.glob(pattern))
        if found:
            return found[0]
    return None


def text_or_na(root, xpath):
    node = root.find(xpath)
    if node is None or node.text is None:
        return "N/A"
    txt = node.text.strip()
    return txt if txt else "N/A"


def parse_csynth_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    return {
        "target_clock_ns": text_or_na(root, "./PerformanceEstimates/SummaryOfTimingAnalysis/TargetClockPeriod"),
        "estimated_clock_ns": text_or_na(root, "./PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod"),
        "latency_best": text_or_na(root, "./PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency"),
        "latency_avg": text_or_na(root, "./PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency"),
        "latency_worst": text_or_na(root, "./PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency"),
        "interval_best": text_or_na(root, "./PerformanceEstimates/SummaryOfOverallLatency/Interval-min"),
        "interval_worst": text_or_na(root, "./PerformanceEstimates/SummaryOfOverallLatency/Interval-max"),
        "bram_18k": text_or_na(root, "./AreaEstimates/Resources/BRAM_18K"),
        "dsp": text_or_na(root, "./AreaEstimates/Resources/DSP"),
        "ff": text_or_na(root, "./AreaEstimates/Resources/FF"),
        "lut": text_or_na(root, "./AreaEstimates/Resources/LUT"),
        "uram": text_or_na(root, "./AreaEstimates/Resources/URAM"),
        "available_bram_18k": text_or_na(root, "./AreaEstimates/AvailableResources/BRAM_18K"),
        "available_dsp": text_or_na(root, "./AreaEstimates/AvailableResources/DSP"),
        "available_ff": text_or_na(root, "./AreaEstimates/AvailableResources/FF"),
        "available_lut": text_or_na(root, "./AreaEstimates/AvailableResources/LUT"),
        "available_uram": text_or_na(root, "./AreaEstimates/AvailableResources/URAM"),
    }


def percent(used, avail):
    try:
        u = float(used)
        a = float(avail)
        if a <= 0:
            return "N/A"
        return f"{(u / a) * 100.0:.2f}%"
    except Exception:
        return "N/A"


def write_summary_md(summary_path: Path, meta, summary, report_paths):
    lines = []
    lines.append("# Vitis HLS Synthesis Summary")
    lines.append("")
    lines.append(f"- Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Project: `{meta['project']}`")
    lines.append(f"- Solution: `{meta['solution']}`")
    lines.append(f"- Top: `{meta['top']}`")
    lines.append("")
    lines.append("## Timing")
    lines.append("")
    lines.append(f"- Target Clock (ns): `{summary['target_clock_ns']}`")
    lines.append(f"- Estimated Clock (ns): `{summary['estimated_clock_ns']}`")
    lines.append("")
    lines.append("## Latency")
    lines.append("")
    lines.append(f"- Best: `{summary['latency_best']}`")
    lines.append(f"- Avg: `{summary['latency_avg']}`")
    lines.append(f"- Worst: `{summary['latency_worst']}`")
    lines.append(f"- II Min: `{summary['interval_best']}`")
    lines.append(f"- II Max: `{summary['interval_worst']}`")
    lines.append("")
    lines.append("## Resource")
    lines.append("")
    lines.append("| Resource | Used | Available | Utilization |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| BRAM_18K | {summary['bram_18k']} | {summary['available_bram_18k']} | {percent(summary['bram_18k'], summary['available_bram_18k'])} |")
    lines.append(f"| DSP | {summary['dsp']} | {summary['available_dsp']} | {percent(summary['dsp'], summary['available_dsp'])} |")
    lines.append(f"| FF | {summary['ff']} | {summary['available_ff']} | {percent(summary['ff'], summary['available_ff'])} |")
    lines.append(f"| LUT | {summary['lut']} | {summary['available_lut']} | {percent(summary['lut'], summary['available_lut'])} |")
    lines.append(f"| URAM | {summary['uram']} | {summary['available_uram']} | {percent(summary['uram'], summary['available_uram'])} |")
    lines.append("")
    lines.append("## Raw Report Paths")
    lines.append("")
    for name, path in report_paths.items():
        lines.append(f"- {name}: `{path.as_posix()}`" if path else f"- {name}: `N/A`")
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def append_history(history_csv: Path, meta, summary, snapshot_dir: Path):
    fieldnames = [
        "run_time", "project", "solution", "top",
        "target_clock_ns", "estimated_clock_ns",
        "latency_best", "latency_avg", "latency_worst",
        "interval_best", "interval_worst",
        "bram_18k", "available_bram_18k",
        "dsp", "available_dsp",
        "ff", "available_ff",
        "lut", "available_lut",
        "uram", "available_uram",
        "snapshot_dir",
    ]
    row = {
        "run_time": datetime.datetime.now().isoformat(timespec="seconds"),
        "project": meta["project"],
        "solution": meta["solution"],
        "top": meta["top"],
        "snapshot_dir": snapshot_dir.as_posix(),
    }
    row.update(summary)
    write_header = not history_csv.exists()
    with history_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Parse Vitis HLS csynth reports and export a summary.")
    parser.add_argument("--project", default="layer2_5_hls_prj")
    parser.add_argument("--solution", default="sol1")
    parser.add_argument("--top", default="conv_ico_layer2_5")
    args = parser.parse_args()

    root = Path(".").resolve()
    report_dir = root / args.project / args.solution / "syn" / "report"
    if not report_dir.exists():
        print(f"[ERROR] Report directory not found: {report_dir}")
        return 1

    csynth_xml = find_first(report_dir, [f"{args.top}_csynth.xml", "*_csynth.xml"])
    csynth_rpt = find_first(report_dir, [f"{args.top}_csynth.rpt", "*_csynth.rpt"])
    cosim_rpt = find_first(root / args.project / args.solution / "sim" / "report", [f"{args.top}_cosim.rpt", "*_cosim.rpt"])

    if csynth_xml is None:
        print(f"[ERROR] csynth XML not found under: {report_dir}")
        return 1

    summary = parse_csynth_xml(csynth_xml)
    reports_root = root.parent.parent / "hls_reports"
    reports_root.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = reports_root / f"{args.project}_{args.solution}_{stamp}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied = {}
    for name, src in {"csynth_xml": csynth_xml, "csynth_rpt": csynth_rpt, "cosim_rpt": cosim_rpt}.items():
        if src and src.exists():
            dst = snapshot_dir / src.name
            shutil.copy2(src, dst)
            copied[name] = dst
        else:
            copied[name] = None

    meta = {"project": args.project, "solution": args.solution, "top": args.top}
    summary_path = snapshot_dir / "summary.md"
    write_summary_md(summary_path, meta, summary, copied)

    latest_path = reports_root / "layer2_5_latest_summary.md"
    shutil.copy2(summary_path, latest_path)
    append_history(reports_root / "summary_history.csv", meta, summary, snapshot_dir)

    print("=== HLS Summary ===")
    print(f"Clock(ns): target={summary['target_clock_ns']}, estimated={summary['estimated_clock_ns']}")
    print(f"Resource: BRAM={summary['bram_18k']}/{summary['available_bram_18k']} DSP={summary['dsp']}/{summary['available_dsp']} LUT={summary['lut']}/{summary['available_lut']} FF={summary['ff']}/{summary['available_ff']}")
    print(f"Summary file: {latest_path}")
    print(f"Snapshot dir: {snapshot_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
