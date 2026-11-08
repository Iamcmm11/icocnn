"""Read-only experiment health viewer. Activity is not proof of RTL progress."""
from pathlib import Path
import argparse
import datetime
import json
import re
import time
import psutil

HERE = Path(__file__).resolve().parent

def sample():
    jobs = {}
    current_path=HERE/"artifacts/current_job.json"
    current={}
    if current_path.exists():
        try:current=json.loads(current_path.read_text())
        except (json.JSONDecodeError,OSError):pass
    if current.get("state")=="RUNNING" and psutil.pid_exists(current.get("pid",-1)):
        jobs[current["tag"]]=dict(pid=current["pid"])
    for p in psutil.process_iter(['pid', 'name', 'exe', 'cpu_times']):
        try:
            if (p.info['name'] or '').lower().replace('.exe', '') != 'xsimk':
                continue
            exe = Path(p.info['exe'] or '')
            if HERE not in exe.parents:
                continue
            tag = next((s for s in exe.parts if re.fullmatch(r'v\d+_t\d+_p\d+_T\d+_f\d+', s)), None)
            if tag:
                jobs[tag] = dict(pid=p.pid, cpu=sum(p.info['cpu_times'][:2]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    rows = {}
    for project in sorted((HERE / '_work').glob('v*')):
        if not project.is_dir():
            continue
        tag = project.name
        sim = project / 'sol1/sim/verilog'
        report = project / 'sol1/sim/report/maba_top_cosim.rpt'
        state = 'NO_ACTIVE_RTL'
        if tag in jobs:
            state = 'RUNNING_UNCONFIRMED' if 'cpu' in jobs[tag] else 'HLS_SERIAL_RUNNING'
        elif report.exists():
            content = report.read_text(errors='replace')
            if re.search(r'\|\s*Verilog\s*\|\s*Pass\s*\|', content):
                state = 'RTL_PASS_REPORT'
            elif re.search(r'\|\s*Verilog\s*\|\s*Fail\s*\|', content):
                state = 'RTL_FAIL_REPORT'
        elif (project / 'sol1/syn/report/maba_top_csynth.xml').exists():
            state = 'SYNTH_ONLY_OR_QUEUED'
        log = sim / 'xsim.log'
        events = []
        if log.exists():
            events = re.findall(r'RTL Simulation : (\d+\s*/\s*\d+)\s*\[([^]]+)\]\s*@\s*"([^"]+)"', log.read_text(errors='replace'))
        wave = sim / 'maba_top.wdb'
        rows[tag] = dict(state=state, **jobs.get(tag, {}),
                         wave_bytes=wave.stat().st_size if wave.exists() else None,
                         event=events[-1] if events else None)
    return rows

def display(before, after, seconds, show_all):
    print()
    print(datetime.datetime.now().isoformat(timespec='seconds'))
    print('Configuration          State                   PID   CPU delta   WDB delta   Last logged transaction / time')
    for tag, row in after.items():
        if not show_all and 'pid' not in row and row['state'] not in ('RTL_PASS_REPORT', 'RTL_FAIL_REPORT'):
            continue
        old = before.get(tag, {})
        cpu = f"{row['cpu']-old['cpu']:.2f}s" if row.get('pid') == old.get('pid') and 'cpu' in old else '-'
        wave = str(row['wave_bytes']-old['wave_bytes'])+'B' if row['wave_bytes'] is not None and old.get('wave_bytes') is not None else '-'
        event = ' / '.join(row['event']) if row['event'] else '-'
        print(f"{tag:22} {row['state']:23} {row.get('pid', '-'):>6} {cpu:>11} {wave:>11}   {event}")
    vanished = [str(row['pid']) for tag,row in before.items() if 'pid' in row and row['pid'] != after.get(tag,{}).get('pid')]
    if vanished:
        print('Exited/replaced PIDs: '+', '.join(vanished)+'; inspect reports for success/failure.')
    print(f"Sample interval: {seconds:.1f}s; available RAM: {psutil.virtual_memory().available/2**30:.2f} GiB")
    print('CPU/WDB changes indicate activity only. Logged simulation time may be stale.')
    print('SYNTH_ONLY_OR_QUEUED does not mean a hang. Ctrl+C stops this viewer only.')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--interval', type=float, default=10)
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--all', action='store_true', dest='show_all')
    args = ap.parse_args()
    if not 1 <= args.interval <= 60:
        ap.error('--interval must be between 1 and 60 seconds')
    before = sample()
    print('Sampling activity; no process or experiment file will be changed.', flush=True)
    try:
        while True:
            start = time.monotonic()
            time.sleep(args.interval)
            after = sample()
            display(before, after, time.monotonic()-start, args.show_all)
            if not args.watch:
                break
            before = after
    except KeyboardInterrupt:
        print('Viewer stopped. Experiments remain running.')

if __name__ == '__main__':
    main()
