"""Reproducible native/HLS matrix runner with raw logs and report provenance."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from job_control import execute,hardware_lock

HERE=Path(__file__).resolve().parent
ART=HERE/"artifacts"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_data(data):
    manifest_path=data/"manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    for name,expected in manifest.get("files",{}).items():
        path=data/name
        if not path.is_file():raise FileNotFoundError(f"Missing fixture file: {path}")
        if path.stat().st_size!=expected["bytes"] or sha(path)!=expected["sha256"]:
            raise RuntimeError(f"Fixture integrity check failed: {path}")
    return manifest
def report(project):
    p=project/"sol1/syn/report/maba_top_csynth.xml"
    if not p.exists():return {}
    tree=ET.parse(p)
    def txt(path):return tree.findtext(path)
    stages=[]
    for child in p.parent.glob("*csynth.xml"):
        if child==p or "Pipeline" in child.name:continue
        t=ET.parse(child)
        stages.append(dict(name=child.stem,latency_max=t.findtext(".//Worst-caseLatency"),
                           interval_max=t.findtext(".//Interval-max")))
    return dict(report=str(p),sha256=sha(p),clock_ns=txt(".//EstimatedClockPeriod"),stages=stages,
                latency_min=txt(".//Best-caseLatency"),latency_max=txt(".//Worst-caseLatency"),
                interval_min=txt(".//Interval-min"),interval_max=txt(".//Interval-max"),
                resources={k:txt(".//AreaEstimates/Resources/"+k) for k in ["DSP","LUT","FF","BRAM_18K","URAM"]},
                available={k:txt(".//AreaEstimates/AvailableResources/"+k) for k in ["DSP","LUT","FF","BRAM_18K","URAM"]})
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=["native","csim","synth","cosim","all"])
    ap.add_argument("--variants",default="3")
    ap.add_argument("--tiles",default="16")
    ap.add_argument("--pars",default="1")
    ap.add_argument("--frames",type=int,default=6)
    ap.add_argument("--timeout",type=int,default=900,help="Maximum wall seconds per job; timeout stops the queue")
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--rtl-limit",type=int,default=3)
    ap.add_argument("--fifo",type=int,default=8)
    ap.add_argument("--data",type=Path,default=ART/"data")
    args=ap.parse_args()
    if args.timeout<1:ap.error("--timeout must be positive")
    verify_data(args.data)
    for p in [ART,ART/"logs",ART/"bin",ART/"outputs"]:p.mkdir(parents=True,exist_ok=True)
    results=[]
    for v in map(int,args.variants.split(",")):
      for tile in map(int,args.tiles.split(",")):
       for par in map(int,args.pars.split(",")):
        tag=f"v{v}_t{tile}_p{par}_T{args.frames}_f{args.fifo}"
        log=ART/"logs"/f"{tag}_{args.mode}.log"
        if log.exists():
            history=ART/"history";history.mkdir(exist_ok=True)
            stamp=time.strftime("%Y%m%d_%H%M%S")
            shutil.copy2(log,history/(stamp+"_"+log.name))
            prior=ART/f"{tag}_{args.mode}.json"
            if prior.exists():shutil.copy2(prior,history/(stamp+"_"+prior.name))
        project=HERE/"_work"/tag
        if args.mode!="native" and (project/"sol1/syn/report").exists():
            archive=ART/"history"/(time.strftime("%Y%m%d_%H%M%S")+"_"+tag+"_reports")
            archive.mkdir(parents=True,exist_ok=True)
            for prior_report in (project/"sol1/syn/report").glob("*csynth*"):
                shutil.copy2(prior_report,archive/prior_report.name)
            prior_rtl=project/"sol1/sim/report/maba_top_cosim.rpt"
            if prior_rtl.exists():shutil.copy2(prior_rtl,archive/prior_rtl.name)
        launch=HERE/"_work"/("launch_"+tag)
        launch.mkdir(parents=True,exist_ok=True)
        defs=[f"-DMABA_VARIANT={v}",f"-DMABA_TILE={tile}",f"-DMABA_PAR={par}",f"-DMABA_T={args.frames}"]
        start=time.time()
        hashes={p.name:sha(p) for p in [HERE/"maba_top.cpp",HERE/"maba_top.hpp",HERE/"test_maba.cpp",HERE/"run_hls.tcl"]}
        data_hash=sha(args.data/"manifest.json")
        if args.mode=="native":
            exe=ART/"bin"/(tag+(".exe" if os.name=="nt" else ""))
            cmd=["g++","-std=c++14","-O2","-ffp-contract=off","-Wno-unknown-pragmas","-DMABA_NATIVE"]+defs+[str(HERE/"maba_top.cpp"),str(HERE/"test_maba.cpp"),"-o",str(exe)]
            build=subprocess.run(cmd,capture_output=True,text=True)
            if build.returncode:log.write_text(build.stdout+build.stderr);raise RuntimeError(build.stderr)
            output=ART/"outputs"/tag;output.mkdir(exist_ok=True)
            cmd=[str(exe),str(args.data.resolve()),str(args.limit),str(output)]
            env=os.environ.copy()
        else:
            if args.frames!=6:raise ValueError("HLS performance matrix is fixed at T=6")
            env=dict(os.environ,MABA_VARIANT=str(v),MABA_TILE=str(tile),MABA_PAR=str(par),MABA_MODE=args.mode,
                     MABA_PROJECT=str(project),MABA_DATA=str(args.data.resolve()),MABA_CASE_LIMIT=str(args.limit),
                     MABA_RTL_CASE_LIMIT=str(args.rtl_limit),MABA_FIFO=str(args.fifo))
            hls=shutil.which("vitis_hls")
            if not hls:raise RuntimeError("vitis_hls is not on PATH")
            cmd=[hls,"-nolog","-f",str(HERE/"run_hls.tcl")]
        print(f"START {tag} {args.mode}",flush=True)
        code,state=execute(cmd,env,launch if args.mode!="native" else HERE,log,args.timeout,tag)
        text=log.read_text(encoding="utf-8",errors="replace")
        case_rows=[dict(name=n,max_abs=float(a),rmse=float(b),passed=c=="1")
                   for n,a,b,c in re.findall(r"CASE (\S+) max_abs=(\S+) rmse=(\S+) pass=([01])",text)]
        row=dict(tag=tag,mode=args.mode,variant=v,tile=tile,par=par,fifo=args.fifo,frames=args.frames,
                 execution_state=state,returncode=code,seconds=time.time()-start,log=str(log),log_sha256=sha(log),
                 source_sha256=hashes,
                 data_manifest_sha256=data_hash,cases=case_rows)
        if args.mode!="native":
            row["synthesis"]=report(project)
            cp=project/"sol1/sim/report/maba_top_cosim.rpt"
            if cp.exists():row["cosim"]=dict(report=str(cp),sha256=sha(cp),text=cp.read_text())
            archive=ART/"reports"/tag;archive.mkdir(parents=True,exist_ok=True)
            for source in (project/"sol1/syn/report").glob("*csynth*"):
                shutil.copy2(source,archive/source.name)
            if cp.exists():shutil.copy2(cp,archive/cp.name)
            expected_csim="CSim done with 0 errors" in text
            expected_synth=args.mode=="csim" or bool(row["synthesis"])
            expected_rtl=args.mode not in ("all","cosim") or ("Pass" in row.get("cosim",{}).get("text","") and "C/RTL co-simulation finished" in text)
            if not (expected_csim and expected_synth and expected_rtl) or "ERROR:" in text:
                row["returncode"]=row["returncode"] or 1
        results.append(row)
        (ART/f"{tag}_{args.mode}.json").write_text(json.dumps(row,indent=2),encoding="utf-8")
        print(f"END {tag} rc={row['returncode']} seconds={row['seconds']:.1f}",flush=True)
        if row["returncode"]:sys.exit(row["returncode"])
    if any(r["returncode"] for r in results):sys.exit(1)
if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1] in ["csim","synth","cosim","all"]:
        with hardware_lock():main()
    else:main()
