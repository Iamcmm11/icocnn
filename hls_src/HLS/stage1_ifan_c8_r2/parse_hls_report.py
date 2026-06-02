import argparse
import csv
import datetime
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def sanitize_name(text):
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "report"


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


def text_first_or_na(root, xpaths):
    for xpath in xpaths:
        value = text_or_na(root, xpath)
        if value != "N/A":
            return value
    return "N/A"


def percent(used, avail):
    try:
        u = float(used)
        a = float(avail)
        if a <= 0:
            return "N/A"
        return f"{(u / a) * 100.0:.2f}%"
    except Exception:
        return "N/A"


def parse_csynth_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    return {
        "target_clock_ns": text_first_or_na(root, [
            "./PerformanceEstimates/SummaryOfTimingAnalysis/TargetClockPeriod",
            "./UserAssignments/TargetClockPeriod",
        ]),
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


def split_design_size_column(text):
    if text is None:
        return "", "", ""
    parts = [part.strip() for part in text.split(",", 2)]
    while len(parts) < 3:
        parts.append("")
    return parts[0], parts[1], parts[2]


def parse_design_size_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    incomplete = root.find("./item[@name='C-Synthesis has not completed!']") is not None

    phase_rows = []
    phase_table = root.find("./item[@name='Total Instructions per Compilation Phase']/table")
    if phase_table is not None:
        for column in phase_table.findall("column"):
            phase_name = (column.get("name") or "").strip()
            step, instructions, description = split_design_size_column(column.text)
            if phase_name or step or instructions or description:
                phase_rows.append({
                    "phase": phase_name,
                    "step": step,
                    "instructions": instructions,
                    "description": description,
                })

    top_functions = []
    rows = root.find("./item[@name='Instructions per Function for each Compilation Phase']/hiertable/rows")
    if rows is not None:
        top_row = rows.find("row")
        if top_row is not None:
            for child in top_row.findall("row"):
                top_functions.append({
                    "function": child.get("col0", "N/A"),
                    "location": child.get("col1", "N/A"),
                    "compile_link": child.get("col2_disp", child.get("col2", "N/A")),
                    "unroll_inline": child.get("col3_disp", child.get("col3", "N/A")),
                })

    return {
        "status": "incomplete" if incomplete else "complete",
        "phase_rows": phase_rows,
        "top_functions": top_functions,
    }


def write_final_summary(summary_path: Path, meta, summary, report_paths):
    lines = []
    lines.append("# Stage1 IFAN C8 R2 HLS Summary")
    lines.append("")
    lines.append(f"- Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Project: `{meta['project']}`")
    lines.append(f"- Solution: `{meta['solution']}`")
    lines.append(f"- Top: `{meta['top']}`")
    lines.append(f"- Status: `csynth complete`")
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


def write_design_size_summary(summary_path: Path, meta, summary, report_paths):
    lines = []
    lines.append("# Stage1 IFAN C8 R2 HLS Summary")
    lines.append("")
    lines.append(f"- Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Project: `{meta['project']}`")
    lines.append(f"- Solution: `{meta['solution']}`")
    lines.append(f"- Top: `{meta['top']}`")
    lines.append(f"- Status: `csynth {summary['status']}`")
    lines.append("- Note: final resource/timing report is not available yet; this summary is based on `csynth_design_size`.")
    lines.append("")
    lines.append("## Phase Instructions")
    lines.append("")
    lines.append("| Phase | Step | Instructions | Description |")
    lines.append("|---|---|---:|---|")
    for row in summary["phase_rows"]:
        phase = row["phase"] if row["phase"] else " "
        step = row["step"] if row["step"] else " "
        instructions = row["instructions"] if row["instructions"] else " "
        description = row["description"] if row["description"] else " "
        lines.append(f"| {phase} | {step} | {instructions} | {description} |")
    lines.append("")
    lines.append("## Top-Level Function Pressure")
    lines.append("")
    lines.append("| Function | Location | Compile/Link | Unroll/Inline |")
    lines.append("|---|---|---:|---:|")
    for row in summary["top_functions"]:
        lines.append(
            f"| {row['function']} | {row['location']} | {row['compile_link']} | {row['unroll_inline']} |"
        )
    lines.append("")
    lines.append("## Raw Report Paths")
    lines.append("")
    for name, path in report_paths.items():
        lines.append(f"- {name}: `{path.as_posix()}`" if path else f"- {name}: `N/A`")
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def append_history(history_csv: Path, meta, mode, summary, snapshot_dir: Path):
    fieldnames = [
        "run_time",
        "project",
        "solution",
        "top",
        "summary_mode",
        "target_clock_ns",
        "estimated_clock_ns",
        "latency_best",
        "latency_avg",
        "latency_worst",
        "interval_best",
        "interval_worst",
        "bram_18k",
        "available_bram_18k",
        "dsp",
        "available_dsp",
        "ff",
        "available_ff",
        "lut",
        "available_lut",
        "uram",
        "available_uram",
        "snapshot_dir",
    ]
    row = {
        "run_time": datetime.datetime.now().isoformat(timespec="seconds"),
        "project": meta["project"],
        "solution": meta["solution"],
        "top": meta["top"],
        "summary_mode": mode,
        "snapshot_dir": snapshot_dir.as_posix(),
        "target_clock_ns": summary.get("target_clock_ns", "N/A"),
        "estimated_clock_ns": summary.get("estimated_clock_ns", "N/A"),
        "latency_best": summary.get("latency_best", "N/A"),
        "latency_avg": summary.get("latency_avg", "N/A"),
        "latency_worst": summary.get("latency_worst", "N/A"),
        "interval_best": summary.get("interval_best", "N/A"),
        "interval_worst": summary.get("interval_worst", "N/A"),
        "bram_18k": summary.get("bram_18k", "N/A"),
        "available_bram_18k": summary.get("available_bram_18k", "N/A"),
        "dsp": summary.get("dsp", "N/A"),
        "available_dsp": summary.get("available_dsp", "N/A"),
        "ff": summary.get("ff", "N/A"),
        "available_ff": summary.get("available_ff", "N/A"),
        "lut": summary.get("lut", "N/A"),
        "available_lut": summary.get("available_lut", "N/A"),
        "uram": summary.get("uram", "N/A"),
        "available_uram": summary.get("available_uram", "N/A"),
    }
    write_header = not history_csv.exists()
    with history_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Parse Stage1 IFAN HLS reports and export a summary.")
    parser.add_argument("--project", default="stage1_ifan_c8_r2_frontend_hls_prj")
    parser.add_argument("--solution", default="sol1")
    parser.add_argument("--top", default="ifan_dual_frontend_top")
    parser.add_argument("--latest-name", default="stage1_ifan_c8_r2_latest_summary.md")
    args = parser.parse_args()

    root = Path(".").resolve()
    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = root / project_path
    report_dir = project_path / args.solution / "syn" / "report"
    if not report_dir.exists():
        print(f"[ERROR] Report directory not found: {report_dir}")
        return 1

    csynth_xml = find_first(report_dir, [f"{args.top}_csynth.xml", "*_csynth.xml"])
    csynth_rpt = find_first(report_dir, [f"{args.top}_csynth.rpt", "*_csynth.rpt"])
    design_size_xml = find_first(report_dir, ["csynth_design_size.xml"])
    design_size_rpt = find_first(report_dir, ["csynth_design_size.rpt"])

    reports_root = root.parent.parent / "hls_reports"
    reports_root.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    project_tag = sanitize_name(project_path.name)
    snapshot_dir = reports_root / f"{project_tag}_{args.solution}_{stamp}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied = {}
    for name, src in {
        "csynth_xml": csynth_xml,
        "csynth_rpt": csynth_rpt,
        "design_size_xml": design_size_xml,
        "design_size_rpt": design_size_rpt,
    }.items():
        if src and src.exists():
            dst = snapshot_dir / src.name
            shutil.copy2(src, dst)
            copied[name] = dst
        else:
            copied[name] = None

    meta = {"project": str(project_path), "solution": args.solution, "top": args.top}
    summary_path = snapshot_dir / "summary.md"
    latest_path = reports_root / args.latest_name

    if csynth_xml is not None:
        final_summary = parse_csynth_xml(csynth_xml)
        write_final_summary(summary_path, meta, final_summary, copied)
        shutil.copy2(summary_path, latest_path)
        append_history(reports_root / "stage1_ifan_c8_r2_summary_history.csv", meta, "csynth", final_summary, snapshot_dir)
        print("=== Stage1 HLS Summary ===")
        print(f"Clock(ns): target={final_summary['target_clock_ns']}, estimated={final_summary['estimated_clock_ns']}")
        print(
            "Resource: "
            f"BRAM={final_summary['bram_18k']}/{final_summary['available_bram_18k']} "
            f"DSP={final_summary['dsp']}/{final_summary['available_dsp']} "
            f"LUT={final_summary['lut']}/{final_summary['available_lut']} "
            f"FF={final_summary['ff']}/{final_summary['available_ff']}"
        )
        print(f"Summary file: {latest_path}")
        print(f"Snapshot dir: {snapshot_dir}")
        return 0

    if design_size_xml is not None:
        design_summary = parse_design_size_xml(design_size_xml)
        write_design_size_summary(summary_path, meta, design_summary, copied)
        shutil.copy2(summary_path, latest_path)
        append_history(reports_root / "stage1_ifan_c8_r2_summary_history.csv", meta, "design_size", {}, snapshot_dir)
        print("=== Stage1 HLS Design-Size Summary ===")
        print(f"Status: csynth {design_summary['status']}")
        if design_summary["phase_rows"]:
            first_rows = design_summary["phase_rows"][:4]
            for row in first_rows:
                phase = row["phase"] or "-"
                step = row["step"] or "-"
                instructions = row["instructions"] or "-"
                print(f"{phase} / {step}: {instructions}")
        print(f"Summary file: {latest_path}")
        print(f"Snapshot dir: {snapshot_dir}")
        return 0

    print(f"[ERROR] No csynth or design-size reports found under: {report_dir}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
