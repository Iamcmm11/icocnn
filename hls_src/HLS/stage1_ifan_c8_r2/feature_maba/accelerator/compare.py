"""One-change-at-a-time comparisons. Default: use existing synthesis, never start RTL."""
from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
from run import HERE,ART,report,sha

PAIRS={
    "tiling":(0,1,"位置分块","完整位置张量 → 16 位置块；比较存储与搬运代价"),
    "fusion":(1,2,"流水融合","相同位置块，阶段数组 → FIFO 流水；比较周期与并行资源"),
    "interleave":(2,3,"位置交错","位置优先 → 时间优先；比较状态依赖是否限制吞吐"),
    "fold":(3,4,"投影折叠","原残差读出 → 离线仿射折叠；比较局部读出与整体瓶颈"),
}
def tag(v):return f"v{v}_t16_p1_T6_f8"

def module_stats(project,needle):
    d=project/"sol1/syn/report"
    p=d/(needle+"_csynth.xml")
    if not p.exists():return None
    t=ET.parse(p)
    return dict(latency_max=int(t.findtext(".//Worst-caseLatency")),
                report=str(p),sha256=sha(p))

def collect(v):
    project=HERE/"_work"/tag(v)
    syn=report(project)
    if not syn:raise RuntimeError(f"Missing synthesis for {tag(v)}; run compare.py --refresh-synth for this pair.")
    tree=ET.parse(Path(syn["report"]))
    part=tree.findtext(".//Part")
    clock=tree.findtext(".//TargetClockPeriod")
    native=ART/(tag(v)+"_native.json")
    n=json.loads(native.read_text())
    if n["returncode"] or not n["cases"] or not all(c["passed"] for c in n["cases"]):
        raise RuntimeError(f"Numerical gate failed for {tag(v)}")
    staged=HERE/"_work"/("sources_"+tag(v))
    sources={k:sha(staged/k) for k in ["maba_top.cpp","maba_top.hpp"] if (staged/k).exists()}
    return dict(tag=tag(v),variant=v,synthesis=syn,part=part,target_clock_ns=clock,
                staged_source_sha256=sources,
                native_report=str(native),native_report_sha256=sha(native),native_case_count=len(n["cases"]),
                native_max_abs=max(c["max_abs"] for c in n["cases"]),native_max_rmse=max(c["rmse"] for c in n["cases"]),
                data_manifest_sha256=n["data_manifest_sha256"],
                modules={k:module_stats(project,k) for k in ["project_conv","normalize","scan","readout"]})

def paired_error(a,b):
    data=ART/"data_full"
    manifest=json.loads((data/"manifest.json").read_text())
    result=[]
    for case in manifest["cases"]:
        name=case["name"];valid=case["valid_frames"]
        paths=[ART/"outputs"/tag(v)/(name+".bin") for v in (a,b)]
        if not all(p.exists() for p in paths):raise RuntimeError(f"Missing paired output {name}")
        x,y=[np.fromfile(p,"<f4").reshape(6,8,240)[:valid].astype(np.float64) for p in paths]
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise RuntimeError("Nonfinite paired output")
        e=x-y
        result.append(dict(name=name,max_abs=float(abs(e).max()),rmse=float(np.sqrt(np.mean(e*e)))))
    return dict(cases=len(result),max_abs=max(r["max_abs"] for r in result),
                max_rmse=max(r["rmse"] for r in result),rows=result)

def interpret(r):
    br=r["resources"]["BRAM_18K"]
    delta=r["cycle_change_percent"]
    if r["step"]=="tiling":
        return ("缓存减少；" if br["after"]<br["before"] else "未减少缓存；")+("周期略增。" if delta>0 else "周期下降。")
    if r["step"]=="fusion":
        return ("周期下降；" if delta<0 else "未降低周期；")+("相比分块版增加存储。" if br["after"]>br["before"] else "未增加 BRAM。")
    if r["step"]=="interleave":
        return "周期变化不足 1%，无显著加速证据。" if abs(delta)<1 else "按实际周期变化评估调度收益。"
    a,b=r["before"]["modules"],r["after"]["modules"]
    if a["readout"] and b["readout"] and b["readout"]["latency_max"]<a["readout"]["latency_max"]:
        return "读出阶段变快；整体收益有限。" if abs(delta)<1 else "读出与整体周期均下降。"
    return "尚未显示读出阶段加速。"

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step",choices=["all",*PAIRS],default="all")
    ap.add_argument("--verify",action="store_true",help="Re-run full native correctness before comparisons")
    ap.add_argument("--refresh-synth",action="store_true",help="Serial C simulation + synthesis only; no RTL")
    ap.add_argument("--timeout",type=int,default=900)
    args=ap.parse_args()
    chosen=list(PAIRS) if args.step=="all" else [args.step]
    variants=sorted({v for k in chosen for v in PAIRS[k][:2]})
    if args.verify:
        subprocess.run([sys.executable,str(HERE/"run.py"),"native","--variants",",".join(map(str,variants)),
            "--data",str(ART/"data_full"),"--timeout",str(args.timeout)],check=True)
    if args.refresh_synth:
        for v in variants:
            subprocess.run([sys.executable,str(HERE/"run.py"),"synth","--variants",str(v),
                "--data",str(ART/"data_hls"),"--timeout",str(args.timeout)],check=True)
    out=ART/"comparisons";out.mkdir(exist_ok=True)
    cases={v:collect(v) for v in variants}
    records=[]
    for key in chosen:
        av,bv,title,purpose=PAIRS[key];a,b=cases[av],cases[bv]
        if a["part"]!=b["part"] or a["target_clock_ns"]!=b["target_clock_ns"]:
            raise RuntimeError("Device/clock mismatch")
        if a["data_manifest_sha256"]!=b["data_manifest_sha256"]:
            raise RuntimeError("Numerical comparisons used different data")
        if a["staged_source_sha256"] and b["staged_source_sha256"] and a["staged_source_sha256"]!=b["staged_source_sha256"]:
            raise RuntimeError("Paired syntheses used different source snapshots")
        ca,cb=[int(r["synthesis"]["latency_max"]) for r in (a,b)]
        resources={k:dict(before=int(a["synthesis"]["resources"][k]),after=int(b["synthesis"]["resources"][k])) for k in ["DSP","BRAM_18K","LUT","FF"]}
        row=dict(step=key,title=title,purpose=purpose,before=a,after=b,cycles_before=ca,cycles_after=cb,
                 synthesis_speedup=ca/cb,cycle_change_percent=100*(cb/ca-1),resources=resources,
                 paired_output=paired_error(av,bv),evidence_level="HLS synthesis estimates + native numerical validation; not RTL or board speed",
                 source_policy="Existing synthesized snapshots are preserved; --refresh-synth regenerates both sides from current sources.")
        row["paired_output"]["passed"]=row["paired_output"]["max_abs"]<=1e-5 and row["paired_output"]["max_rmse"]<=1e-6
        if not row["paired_output"]["passed"]:
            raise RuntimeError("Paired output change exceeds the unchanged gate")
        records.append(row)
        (out/(key+".json")).write_text(json.dumps(row,indent=2),encoding="utf-8")
    # Report includes completed comparisons, without inferring missing stages.
    allrows=[json.loads((out/(k+".json")).read_text()) for k in PAIRS if (out/(k+".json")).exists()]
    lines=["# 按设计改动逐项验证","",
        "统一条件：VC707、目标 5 ns、FP32、T=6、P=240、tile=16、PAR=1、FIFO=8。每对只改变一个机制。",
        "周期为 HLS 综合估计；本页不将其称为 RTL 实测或板级加速。完整 RTL 已停止，不会由本命令自动启动。","",
        "| 改动 | 前→后周期 | 估计加速比 | BRAM18K 前→后 | DSP 前→后 | 解释 |",
        "|---|---:|---:|---:|---:|---|"]
    for r in allrows:
        br=r["resources"]["BRAM_18K"];dsp=r["resources"]["DSP"]
        lines.append(f"| {r['title']} | {r['cycles_before']:,} → {r['cycles_after']:,} | {r['synthesis_speedup']:.6f}× | {br['before']} → {br['after']} | {dsp['before']} → {dsp['after']} | {interpret(r)} |")
    lines+=["","## 局部阶段与整体瓶颈","",
        "| 改动 | 阶段 | 前→后估计周期 | 说明 |","|---|---|---:|---|"]
    for r in allrows:
        for key in ["project_conv","normalize","scan","readout"]:
            a,b=r["before"]["modules"][key],r["after"]["modules"][key]
            if a and b:
                lines.append(f"| {r['title']} | {key} | {a['latency_max']:,} → {b['latency_max']:,} | 同名 HLS 子模块；以局部调度作用域为准 |")
    lines+=["","子模块是流水重叠执行的，局部延迟不能直接相加成为顶层延迟，也不能把局部加速比当成整模块加速比。","",
        "## 数值前后对照","",
        "| 改动 | 配对用例 | 前后输出最大差值 | 最坏 RMSE |","|---|---:|---:|---:|"]
    for r in allrows:
        e=r["paired_output"]
        lines.append(f"| {r['title']} | {e['cases']} | {e['max_abs']:.9g} | {e['max_rmse']:.9g} |")
    lines+=["","每个版本还分别通过原始 PyTorch golden 的 1e-5/1e-6 门限。配对差异和对参考误差分别统计。",
        "JSON 保留综合 XML、子模块报告和数值报告的路径及哈希；表中缺少数据的机制不推断通过。",
        "默认读取已保存的完整规模综合证据；--refresh-synth 会按同一份源码串行刷新两侧结果。运行日期和源码身份以原始报告及 JSON 哈希为准。","",
        "## 下一步验真顺序","",
        "1. 分块：数值一致性、BRAM 是否减少、搬运是否抵消周期收益。",
        "2. 融合：在相同分块和精度下看周期、FIFO 与资源代价。",
        "3. 交错：检查 scan 阶段及顶层周期；若无收益，保留负结果。",
        "4. 折叠：检查 readout 的局部收益及 scan 瓶颈，避免宣称整模块同等加速。",
        "5. 仅对确定保留的设计安排小规模 RTL 功能/死锁测试；完整规模 RTL 作为之后独立的一步。",
        "小规模 RTL 的周期不能直接换算完整模型加速比。"]
    (HERE/"COMPARISON.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    with (out/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=["step","cycles_before","cycles_after","synthesis_speedup","cycle_change_percent"])
        writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in allrows)
    print("Wrote "+str(HERE/"COMPARISON.md"))
    for r in records:
        print(f"{r['step']}: {r['cycles_before']} -> {r['cycles_after']} cycles, estimated {r['synthesis_speedup']:.6f}x")
if __name__=="__main__":main()
