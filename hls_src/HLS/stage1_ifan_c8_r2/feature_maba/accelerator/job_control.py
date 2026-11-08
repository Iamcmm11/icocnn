"""Run at most one HLS job, with a wall-clock bound and explicit status."""
import contextlib
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import psutil

ART=Path(__file__).resolve().parent/"artifacts"

@contextlib.contextmanager
def hardware_lock():
    ART.mkdir(exist_ok=True)
    lock=ART/"hardware_job.lock"
    if lock.exists():
        owner=json.loads(lock.read_text())
        try:
            live=abs(psutil.Process(owner["pid"]).create_time()-owner["created"])<.01
        except psutil.NoSuchProcess:
            live=False
        if live:
            raise RuntimeError(f"Hardware job PID {owner['pid']} is active; parallel HLS execution is disabled.")
        lock.unlink()
    owner=dict(pid=os.getpid(),created=psutil.Process().create_time())
    with lock.open("x") as f:json.dump(owner,f)
    try:
        yield
    finally:
        if lock.exists() and json.loads(lock.read_text())==owner:lock.unlink()

def stop_tree(pid):
    try:
        root=psutil.Process(pid);children=root.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    try:root.suspend()
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    for p in list(reversed(children))+[root]:
        try:p.kill()
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    psutil.wait_procs(children+[root],timeout=5)

def execute(cmd,env,cwd,log,timeout,tag):
    start=time.monotonic();state="RUNNING";last_notice=0
    with log.open("w",encoding="utf-8") as f:
        proc=subprocess.Popen(cmd,env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                elapsed=time.monotonic()-start
                status=dict(updated=datetime.datetime.now().isoformat(timespec="seconds"),
                    state=state,pid=proc.pid,tag=tag,elapsed_seconds=round(elapsed,1),timeout_seconds=timeout,
                    log=str(log))
                (ART/"current_job.json").write_text(json.dumps(status,indent=2))
                if elapsed-last_notice>=30:
                    print(f"RUNNING {tag}: {elapsed:.0f}s / {timeout}s",flush=True)
                    last_notice=elapsed
                if elapsed>=timeout:
                    state="TIMEOUT";stop_tree(proc.pid);break
                time.sleep(1)
        except KeyboardInterrupt:
            state="INTERRUPTED";stop_tree(proc.pid)
        rc=proc.wait()
    state=state if state!="RUNNING" else ("FINISHED" if rc==0 else "FAILED")
    status=dict(updated=datetime.datetime.now().isoformat(timespec="seconds"),state=state,
        pid=proc.pid,tag=tag,elapsed_seconds=round(time.monotonic()-start,1),timeout_seconds=timeout,
        returncode=rc,log=str(log))
    (ART/"current_job.json").write_text(json.dumps(status,indent=2))
    return (124 if state=="TIMEOUT" else 130 if state=="INTERRUPTED" else rc),state
