import argparse
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

    summary = {
        "target_clock_ns": text_or_na(
            root, "./PerformanceEstimates/SummaryOfTimingAnalysis/TargetClockPeriod"
        ),
        "estimated_clock_ns": text_or_na(
            root, "./PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod"
        ),
        "latency_best": text_or_na(
            root, "./PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency"
        ),
        "latency_avg": text_or_na(
            root, "./PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency"
        ),
        "latency_worst": text_or_na(
            root, "./PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency"
        ),
        "interval_best": text_or_na(
            root, "./PerformanceEstimates/SummaryOfOverallLatency/Interval-min"
        ),
        "interval_worst": text_or_na(
            root, "./PerformanceEstimates/SummaryOfOverallLatency/Interval-max"
        ),
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
    return summary


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
    lines.append(
        f"| BRAM_18K | {summary['bram_18k']} | {summary['available_bram_18k']} | {percent(summary['bram_18k'], summary['available_bram_18k'])} |"
    )
    lines.append(
        f"| DSP | {summary['dsp']} | {summary['available_dsp']} | {percent(summary['dsp'], summary['available_dsp'])} |"
    )
    lines.append(
        f"| FF | {summary['ff']} | {summary['available_ff']} | {percent(summary['ff'], summary['available_ff'])} |"
    )
    lines.append(
        f"| LUT | {summary['lut']} | {summary['available_lut']} | {percent(summary['lut'], summary['available_lut'])} |"
    )
    lines.append(
        f"| URAM | {summary['uram']} | {summary['available_uram']} | {percent(summary['uram'], summary['available_uram'])} |"
    )
    lines.append("")
    lines.append("## Raw Report Paths")
    lines.append("")
    for name, path in report_paths.items():
        if path:
            lines.append(f"- {name}: `{path.as_posix()}`")
        else:
            lines.append(f"- {name}: `N/A`")
    lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Parse Vitis HLS csynth reports and export a summary.")
    parser.add_argument("--project", default="layer0_hls_prj")
    parser.add_argument("--solution", default="sol1")
    parser.add_argument("--top", default="conv_ico_layer0")
    args = parser.parse_args()

    root = Path(".").resolve()
    report_dir = root / args.project / args.solution / "syn" / "report"
    if not report_dir.exists():
        print(f"[ERROR] Report directory not found: {report_dir}")
        return 1

    csynth_xml = find_first(report_dir, [f"{args.top}_csynth.xml", "*_csynth.xml"])
    csynth_rpt = find_first(report_dir, [f"{args.top}_csynth.rpt", "*_csynth.rpt"])
    cosim_rpt = find_first(
        root / args.project / args.solution / "sim" / "report",
        [f"{args.top}_cosim.rpt", "*_cosim.rpt"],
    )

    if csynth_xml is None:
        print(f"[ERROR] csynth XML not found under: {report_dir}")
        return 1

    summary = parse_csynth_xml(csynth_xml)

    reports_root = root.parent.parent / "hls_reports"
    reports_root.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_dir = reports_root / f"{args.project}_{args.solution}_{stamp}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    copied = {}
    for name, src in {
        "csynth_xml": csynth_xml,
        "csynth_rpt": csynth_rpt,
        "cosim_rpt": cosim_rpt,
    }.items():
        if src and src.exists():
            dst = snap_dir / src.name
            shutil.copy2(src, dst)
            copied[name] = dst
        else:
            copied[name] = None

    summary_path = snap_dir / "summary.md"
    write_summary_md(
        summary_path,
        meta={"project": args.project, "solution": args.solution, "top": args.top},
        summary=summary,
        report_paths=copied,
    )

    latest_path = reports_root / "latest_summary.md"
    shutil.copy2(summary_path, latest_path)

    print("=== HLS Summary ===")
    print(f"Clock(ns): target={summary['target_clock_ns']}, estimated={summary['estimated_clock_ns']}")
    print(
        "Resource: "
        f"BRAM={summary['bram_18k']}/{summary['available_bram_18k']} "
        f"DSP={summary['dsp']}/{summary['available_dsp']} "
        f"LUT={summary['lut']}/{summary['available_lut']} "
        f"FF={summary['ff']}/{summary['available_ff']}"
    )
    print(f"Summary file: {latest_path}")
    print(f"Snapshot dir: {snap_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
