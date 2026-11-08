"""Run the original node checks and paired full-chain MABA replacement checks."""
from pathlib import Path
import argparse
import json
import subprocess
import hashlib
import re
import os
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
STAGE=HERE.parents[1]
V14=ROOT/"paper_rewriting_output/v14/experiments"
ART=HERE/"artifacts"
def run(cmd,log):
    p=subprocess.run([str(x) for x in cmd],cwd=ROOT,capture_output=True,text=True)
    log.write_text(p.stdout+p.stderr,encoding="utf-8")
    return p.returncode
def default_vitis_include():
    for key in ("XILINX_HLS", "XILINX_VITIS"):
        value=os.environ.get(key)
        if value:
            candidate=Path(value)/"include"
            if (candidate/"ap_fixed.h").exists():return candidate
    return None
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--variant",type=int,default=4)
    ap.add_argument("--vitis-include",type=Path,default=default_vitis_include(),
                    help="Vitis include directory; required for --mixed")
    ap.add_argument("--mixed",action="store_true");args=ap.parse_args()
    (ART/"bin").mkdir(parents=True,exist_ok=True);(ART/"logs").mkdir(exist_ok=True)
    readers=ART/"bin/chain_readers";readers.mkdir(exist_ok=True)
    for ns,folder,name in [("stage","full_stage1_legacy","test_ifan_stage1.cpp"),
                           ("maba","feature_maba","test_feature_maba.cpp"),
                           ("post","post_maba","test_post_maba.cpp")]:
        source=(STAGE/folder/name).read_text()
        source=re.sub(r"#ifdef __HLS_CSIM__.*?#endif","",source,flags=re.S)
        source=re.sub(r"^#include[^\n]*\n","",source,flags=re.M)
        source=source.replace("int main()","int original_test()")
        (readers/(ns+"_reader.inc")).write_text(source)
    flags=["g++","-std=c++14","-O2","-ffp-contract=off","-Wno-unknown-pragmas","-DMABA_NATIVE",f"-DMABA_VARIANT={args.variant}","-I"+str(readers)]
    if args.mixed:
        if not args.vitis_include or not (args.vitis_include/"ap_fixed.h").exists():
            ap.error("--mixed requires --vitis-include (or XILINX_HLS/XILINX_VITIS) containing ap_fixed.h")
        flags+=["-I"+str(args.vitis_include.resolve()),"-DMABA_REQUIRE_FIXED"]
    else:flags+=["-DIFAN_FORCE_FP32"]
    sources=[HERE/"maba_top.cpp",HERE/"test_chain.cpp",
             STAGE/"full_stage1_legacy/ifan_stage1.cpp",STAGE/"full_stage1_legacy/ifan_stage1_engines.cpp",
             STAGE/"feature_maba/ifan_stage1_maba.cpp",STAGE/"post_maba/ifan_stage1_post.cpp"]
    tag=f"chain_v{args.variant}_{'mixed' if args.mixed else 'fp32'}"
    exe=ART/"bin"/(tag+(".exe" if os.name=="nt" else ""))
    rc=run(flags+sources+["-o",exe],ART/"logs"/(tag+"_build.log"))
    if rc:raise RuntimeError((ART/"logs"/(tag+"_build.log")).read_text())
    rc=run([exe,V14/"golden/scene_1_t6"],ART/"logs"/(tag+".log"))
    text=(ART/"logs"/(tag+".log")).read_text()
    print(text[-1800:])
    (ART/(tag+".json")).write_text(json.dumps(dict(returncode=rc,variant=args.variant,mixed=args.mixed,
        log=text,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2))
    raise SystemExit(rc)
if __name__=="__main__":main()
