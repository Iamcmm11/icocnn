"""Monitor already-launched jobs and refresh evidence; never invent a pass."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import psutil
from run import HERE,ART,report,sha
REQUIRED=["v0_t16_p1_T6_f8","v1_t16_p1_T6_f8","v2_t16_p1_T6_f8","v3_t16_p1_T6_f8","v4_t16_p1_T6_f8"]
def snapshot():
    jobs=[]
    for p in psutil.process_iter(["pid","name","cmdline","exe","cpu_times"]):
        try:
            cmd=" ".join(p.info["cmdline"] or [])
            exe=p.info["exe"] or ""
            if str(HERE).lower().replace("\\","/") not in (cmd+" "+exe).lower().replace("\\","/"):continue
            if p.pid==psutil.Process().pid:continue
            process_name=(p.info["name"] or "").lower().replace(".exe", "")
            if process_name in ["xsimk","vitis_hls","xelab","csim","python"]:
                jobs.append(dict(pid=p.pid,name=p.info["name"],command=cmd,cpu_seconds=sum(p.info["cpu_times"][:2])))
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    inventory=[];rtl={}
    for d in (HERE/"_work").glob("v*"):
        s=report(d)
        if s:
            inventory.append(dict(tag=d.name,synthesis=s,status="synthesis_only_inventory",
                preprocessed_sha256=sha(d/"sol1/.autopilot/db/a.pp.bc") if (d/"sol1/.autopilot/db/a.pp.bc").exists() else None))
        cp=d/"sol1/sim/report/maba_top_cosim.rpt"
        rtl[d.name]="PASS" if cp.exists() and "Pass" in cp.read_text() else "NOT_COMPLETED"
    (ART/"synthesis_inventory.json").write_text(json.dumps(inventory,indent=2))
    status=dict(updated_local=time.strftime("%Y-%m-%d %H:%M:%S"),jobs=jobs,rtl=rtl,
                required_complete=all(rtl.get(k)=="PASS" for k in REQUIRED),
                outstanding=[k for k in REQUIRED if rtl.get(k)!="PASS"])
    (ART/"runtime_status.json").write_text(json.dumps(status,indent=2))
    return status
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--once",action="store_true")
    args=ap.parse_args();count=0
    while True:
        status=snapshot()
        if args.once or count%5==0 or not status["jobs"]:
            with (ART/"logs/auto_report.log").open("w") as log:
                for name in ["analyze.py","write_report.py"]:
                    subprocess.run([sys.executable,str(HERE/name)],cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
        if args.once or not status["jobs"]:break
        count+=1;time.sleep(30)
if __name__=="__main__":main()
