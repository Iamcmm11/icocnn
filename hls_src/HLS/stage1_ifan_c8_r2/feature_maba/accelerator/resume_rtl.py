"""Re-run RTL on an existing synthesized project without changing its source."""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from run import HERE,ART,sha,report
from job_control import execute,hardware_lock
ap=argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--data",type=Path,default=ART/"data_hls")
ap.add_argument("--limit",type=int,default=3)
ap.add_argument("--profile",action="store_true")
ap.add_argument("--timeout",type=int,default=900)
args=ap.parse_args()
tag=args.tag;project=HERE/"_work"/tag
if not (project/"sol1/syn/report/maba_top_csynth.xml").exists():raise RuntimeError("Synthesis required")
log=ART/"logs"/(tag+"_rtl_resume.log")
launch=HERE/"_work"/("resume_"+tag);launch.mkdir(exist_ok=True)
env=dict(os.environ,MABA_PROJECT=str(project),MABA_DATA=str(args.data.resolve()),
         MABA_RTL_CASE_LIMIT=str(args.limit),MABA_PROFILE=str(int(args.profile)))
start=time.time()
print("RTL START "+tag,flush=True)
vitis_hls=shutil.which("vitis_hls")
if not vitis_hls:raise RuntimeError("vitis_hls is not on PATH")
with hardware_lock():
    code,state=execute([vitis_hls,"-nolog","-f",str(HERE/"resume_rtl.tcl")],env,launch,log,args.timeout,tag)
text=log.read_text(errors="replace");cp=project/"sol1/sim/report/maba_top_cosim.rpt"
co=cp.read_text() if cp.exists() else ""
ok=code==0 and "Pass" in co and "C/RTL co-simulation finished" in text and "ERROR:" not in text
parts=tag.split("_")
row=dict(tag=tag,mode="cosim",variant=int(parts[0][1:]),tile=int(parts[1][1:]),par=int(parts[2][1:]),
         frames=int(parts[3][1:]),fifo=int(parts[4][1:]),returncode=0 if ok else (code or 1),execution_state=state,
         seconds=time.time()-start,profile=args.profile,log=str(log),log_sha256=sha(log),
         data_manifest_sha256=sha(args.data/"manifest.json"),synthesis=report(project),
         cosim=dict(report=str(cp),text=co,sha256=sha(cp) if cp.exists() else None))
archive=ART/"reports"/tag;archive.mkdir(parents=True,exist_ok=True)
for f in (project/"sol1/syn/report").glob("*csynth*"):shutil.copy2(f,archive/f.name)
if cp.exists():shutil.copy2(cp,archive/cp.name)
(ART/(tag+"_cosim.json")).write_text(json.dumps(row,indent=2))
print("RTL END "+tag+" pass="+str(ok),flush=True)
raise SystemExit(0 if ok else 1)
