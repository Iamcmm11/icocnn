"""Freeze FP32 weights and independent references; never overwrite V14 evidence."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
GOLD = ROOT / "paper_rewriting_output/v14/experiments/golden/scene_1_t6"
KEYS = ["in_proj_weight","in_proj_bias","dw_conv_weight","dw_conv_bias",
        "mix_norm_weight","mix_norm_bias","state_proj_weight","state_proj_bias",
        "state_back_weight","state_back_bias","out_proj_weight","out_proj_bias"]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def pack(weights):
    # Deterministic scalar double accumulation, then one FP32 rounding per coefficient.
    wo=weights["out_proj_weight"].astype(np.float64)
    wb=weights["state_back_weight"].astype(np.float64)
    bb=weights["state_back_bias"].astype(np.float64)
    wh=np.zeros((8,8),np.float32);bf=np.zeros(8,np.float32)
    for c in range(8):
        for s in range(8):
            wh[c,s]=sum(float(wo[c,d])*float(wb[d,s]) for d in range(16))
        bf[c]=float(weights["out_proj_bias"][c])+sum(float(wo[c,d])*float(bb[d]) for d in range(16))
    return np.concatenate([weights[k].reshape(-1) for k in KEYS]+[wh.reshape(-1),bf]).astype("<f4")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scenes", action="store_true")
    parser.add_argument("--frames",type=int,default=6)
    parser.add_argument("--output",type=Path,default=HERE/"artifacts/data")
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    weights=dict(np.load(GOLD/"maba_weights.npz"))
    frozen_source=json.loads((GOLD/"manifest.json").read_text(encoding="utf-8"))["source"]
    checkpoint=ROOT/frozen_source["checkpoint"]
    if sha(checkpoint)!=frozen_source["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from frozen V14 golden identity")
    packed=pack(weights);assert packed.size==864
    packed.tofile(out/"weights.bin")
    m=module("maba_reference",ROOT/"IFAN_Edge/ifan_edge/models/map_maba.py")
    ref=m.FeatureMABATemporalRefiner(channels=8,dropout=0).eval()
    mapping={"in_proj_weight":"in_proj.weight","in_proj_bias":"in_proj.bias",
             "dw_conv_weight":"dw_conv.weight","dw_conv_bias":"dw_conv.bias",
             "mix_norm_weight":"mix_norm.weight","mix_norm_bias":"mix_norm.bias",
             "state_proj_weight":"state_proj.weight","state_proj_bias":"state_proj.bias",
             "state_back_weight":"state_back.weight","state_back_bias":"state_back.bias",
             "out_proj_weight":"out_proj.weight","out_proj_bias":"out_proj.bias"}
    ref.load_state_dict({mapping[k]:torch.from_numpy(weights[k]) for k in KEYS})
    rows=[]
    def emit(name,x,valid=None,expected=None,role="synthetic",override=None):
        x=np.ascontiguousarray(x,dtype=np.float32)
        if override is not None:
            ref.load_state_dict({mapping[k]:torch.from_numpy(override[k]) for k in KEYS})
            pack(override).tofile(out/(name+".weights.bin"))
        if expected is None:
            with torch.no_grad():expected=ref(torch.from_numpy(x[None])).numpy()[0]
        if override is not None:ref.load_state_dict({mapping[k]:torch.from_numpy(weights[k]) for k in KEYS})
        np.asarray(x,dtype="<f4").tofile(out/(name+".input.bin"))
        np.asarray(expected,dtype="<f4").tofile(out/(name+".output.bin"))
        rows.append(dict(name=name,valid_frames=valid or args.frames,role=role,
                         input_sha256=sha(out/(name+".input.bin")),
                         output_sha256=sha(out/(name+".output.bin")),
                         weights_sha256=sha(out/(name+".weights.bin")) if override is not None else sha(out/"weights.bin")))
    shape=(args.frames,8,6,5,2,4)
    if args.frames==6:
        original=np.load(GOLD/"final_head_logits.npy")
        emit("frozen_v14",original,expected=np.load(GOLD/"pre_readout_refined_logits.npy"),role="frozen")
    emit("zeros",np.zeros(shape))
    if args.frames==6:emit("frozen_repeat",original,expected=np.load(GOLD/"pre_readout_refined_logits.npy"),role="reset")
    emit("constant",np.full(shape,0.125))
    x=np.zeros(shape,np.float32);x[0,0,0,0,0,0]=1;emit("impulse",x)
    # Same pulse in the final position tests tile tails and spatial isolation.
    x=np.zeros(shape,np.float32);x[0,0,-1,-1,-1,-1]=1;emit("last_position",x)
    rng=np.random.default_rng(42)
    emit("near_constant",np.full(shape,0.125)+rng.normal(0,1e-7,shape))
    emit("random",rng.normal(0,0.5,shape))
    # Large inputs stress normalization; bounded gates are separately checked analytically.
    emit("large",rng.choice([-20.,20.],size=shape))
    for sign in [-1,1]:
        special={k:v.copy() for k,v in weights.items()}
        special["state_proj_weight"][8:]=0
        special["state_proj_bias"][8:]=80*sign
        emit("gate_positive" if sign>0 else "gate_negative",rng.normal(0,.5,shape),override=special)
    special={k:v.copy() for k,v in weights.items()}
    for k in ["in_proj_weight","in_proj_bias","dw_conv_weight","dw_conv_bias"]:special[k][:]=0
    emit("zero_variance",np.zeros(shape),override=special)
    scene_sources={}
    if args.scenes:
        e=module("maba_stage_export",ROOT/"IFAN_Edge/scripts/export_stage1_hls_golden.py")
        # PyTorch >= 2.6 changed the default; this is the user's trusted local checkpoint.
        original_load=torch.load
        def trusted_load(*a,**kw):
            kw.setdefault("weights_only",False);return original_load(*a,**kw)
        torch.load=trusted_load
        model,_,_=e.load_model(ROOT/e.DEFAULT_CHECKPOINT,ROOT)
        torch.load=original_load
        for scene in range(1,5):
            p=ROOT/f"IFAN_Edge/outputs/stage1_features/scene_{scene}/dual_maps.npy"
            scene_sources[str(p.relative_to(ROOT))]=sha(p)
            maps=np.load(p);maps=maps[0] if maps.ndim==6 else maps
            for start in range(0,maps.shape[1],args.frames):
                valid=min(args.frames,maps.shape[1]-start)
                window=np.zeros((2,args.frames,5,4,8),np.float32)
                window[:,:valid]=maps[:,start:start+valid]
                with torch.no_grad():
                    _,debug=model(torch.from_numpy(window[None]),return_debug=True)
                x=debug["final_head_logits"].numpy()[0]
                expected=debug["pre_readout_refined_logits"].numpy()[0]
                emit(f"scene{scene}_{start:03d}",x,valid,expected,"development" if scene==1 else "held_out")
    (out/"cases.txt").write_text("".join(f"{r['name']} {r['valid_frames']}\n" for r in rows))
    source=json.loads((GOLD/"manifest.json").read_text(encoding="utf-8"))["source"]
    manifest=dict(schema=1,frames=args.frames,reset_per_call=True,precision="FP32",
                  torch=torch.__version__,checkpoint=source,weights_order=KEYS+["folded_weight","folded_bias"],
                  weight_sha256=sha(out/"weights.bin"),golden_manifest_sha256=sha(GOLD/"manifest.json"),
                  reference_sha256=sha(ROOT/"IFAN_Edge/ifan_edge/models/map_maba.py"),
                  exporter_sha256=sha(__file__),scene_sources=scene_sources,cases=rows)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (HERE/"artifacts").mkdir(exist_ok=True)
    (HERE/"artifacts"/f"data_manifest_t{args.frames}.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Exported {len(rows)} cases to {out}")

if __name__=="__main__":main()
