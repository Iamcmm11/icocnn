"""Aggregate evidence, paired direction drift, and export publication-ready figures."""
from pathlib import Path
import csv
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ART=HERE/"artifacts";ROOT=HERE.parents[4]
GOLD=ROOT/"paper_rewriting_output/v14/experiments/golden/scene_1_t6"
NAMES=["Materialized","Tiled","Fused","Interleaved","Folded"]
def post(x):
    w=np.load(GOLD/"readout_weights.npz");p=np.load(GOLD/"post_maba_tensors.npz")
    a=np.einsum("tcrahw,c->trahw",x.astype(np.float64),w["channel_readout_weight"].reshape(8))+float(w["channel_readout_bias"].reshape(-1)[0])
    a=a.max(axis=1)*p["clean_vertices_mask"]
    a=a.reshape(len(x),40);a-=a.max(axis=1,keepdims=True)
    prob=np.exp(a);prob/=prob.sum(axis=1,keepdims=True)
    return prob@p["softargmax_indexes"].reshape(3,40).T
def main():
    rows=[];directions=[]
    manifest=json.loads((ART/"data_full/manifest.json").read_text())
    for v in range(5):
        tag=f"v{v}_t16_p1_T6_f8";p=ART/(tag+"_native.json")
        if not p.exists():continue
        r=json.loads(p.read_text());rows.append(dict(variant=v,max_abs=max(c["max_abs"] for c in r["cases"]),rmse=max(c["rmse"] for c in r["cases"]),cases=len(r["cases"]),passed=r["returncode"]==0))
        for case in manifest["cases"]:
            name=case["name"];valid=case["valid_frames"]
            actual=ART/"outputs"/tag/(name+".bin")
            if not actual.exists():continue
            a=np.fromfile(actual,"<f4").reshape(6,8,6,5,2,4)[:valid]
            b=np.fromfile(ART/"data_full"/(name+".output.bin"),"<f4").reshape(6,8,6,5,2,4)[:valid]
            ca,cb=post(a),post(b);err=ca-cb
            na,nb=np.linalg.norm(ca,axis=1),np.linalg.norm(cb,axis=1)
            defined=(na>1e-6)&(nb>1e-6)
            angles=np.degrees(np.arccos(np.clip(np.sum(ca[defined]*cb[defined],axis=1)/(na[defined]*nb[defined]),-1,1)))
            directions.append(dict(variant=v,name=name,role=case["role"],max_abs=float(abs(err).max()),rmse=float(np.sqrt(np.mean(err**2))),max_angle_deg=float(angles.max()) if angles.size else None,undefined_direction_frames=int((~defined).sum()),pass_increment=bool(abs(err).max()<=1e-4 and np.sqrt(np.mean(err**2))<=1e-5)))
    (ART/"numerical_summary.json").write_text(json.dumps(dict(module=rows,direction=directions,post_reference="FP64 software readout; paired comparison, not HLS post implementation"),indent=2))
    fig,ax=plt.subplots(figsize=(6.5,3.5))
    ax.bar([NAMES[r["variant"]] for r in rows],[r["max_abs"] for r in rows])
    ax.axhline(1e-5,color="red",linestyle="--",label="Unchanged max-abs threshold")
    ax.set_yscale("log");ax.set_ylabel("Maximum absolute error");ax.legend(fontsize=8);fig.tight_layout()
    for ext in ["pdf","png"]:fig.savefig(ART/("numerical_error."+ext),dpi=200)
    plt.close(fig)
    long_manifest=ART/"data_long/manifest.json"
    if long_manifest.exists():
        lm=json.loads(long_manifest.read_text());long_rows=[]
        fig,ax=plt.subplots(figsize=(6.5,3.5))
        for v in range(5):
            worst=np.zeros(96);present=False
            for case in lm["cases"]:
                p=ART/"outputs"/f"v{v}_t16_p1_T96_f8"/(case["name"]+".bin")
                if not p.exists():continue
                a=np.fromfile(p,"<f4").reshape(96,-1)
                b=np.fromfile(ART/"data_long"/(case["name"]+".output.bin"),"<f4").reshape(96,-1)
                worst=np.maximum(worst,np.max(abs(a.astype(float)-b),axis=1));present=True
            if present:
                ax.plot(np.arange(1,97),worst,label=NAMES[v])
                long_rows.append(dict(variant=v,max_abs_by_frame=worst.tolist()))
        ax.axhline(1e-5,color="red",linestyle="--")
        ax.set_xlabel("Time step (one invocation, no inter-window persistence)")
        ax.set_ylabel("Worst absolute error over stress cases");ax.legend(fontsize=7);fig.tight_layout()
        for ext in ["pdf","png"]:fig.savefig(ART/("long_sequence_error."+ext),dpi=200)
        plt.close(fig)
        (ART/"long_sequence_error.json").write_text(json.dumps(long_rows,indent=2))
    # Standalone vector figure: no claimed hardware measurements.
    fig,ax=plt.subplots(figsize=(10,3));ax.axis("off")
    labels=["AXI load +\nposition tile","8 -> 16\nprojection + DWConv","LayerNorm\nD=16","Gate + state\nS=8","Folded readout\n+ residual","Tile restore\n+ AXI store"]
    for i,label in enumerate(labels):
        x=.02+i*.165
        ax.text(x,.65,label,ha="left",va="center",fontsize=8,bbox=dict(boxstyle="round,pad=.5",facecolor="#e8f1fb",edgecolor="#34699a"),transform=ax.transAxes)
        if i<5:ax.annotate("",xy=(x+.159,.65),xytext=(x+.133,.65),arrowprops=dict(arrowstyle="->"),xycoords="axes fraction")
    ax.text(.02,.25,"Tile-local history: B x 2 x 16 FP32; recurrent state: B x 8 FP32.\nTime remains causal; independent positions interleave inside each tile. FIFO depths are measured in RTL.",fontsize=9,transform=ax.transAxes)
    fig.tight_layout()
    for ext in ["pdf","png"]:fig.savefig(ART/("architecture."+ext),dpi=200)
    plt.close(fig)
    synth=[]
    inventory_path=ART/"synthesis_inventory.json"
    evidence=[]
    if inventory_path.exists():
        for item in json.loads(inventory_path.read_text()):
            parts=item["tag"].split("_")
            evidence.append(dict(tag=item["tag"],mode="synth",variant=int(parts[0][1:]),
                tile=int(parts[1][1:]),par=int(parts[2][1:]),synthesis=item["synthesis"]))
    for p in sorted(ART.glob("v*_*.json")):
        r=json.loads(p.read_text())
        evidence.append(r)
    for r in evidence:
        if r.get("mode") not in ["all","synth","cosim"]:continue
        s=r.get("synthesis",{})
        if not s:continue
        co=r.get("cosim",{}).get("text","")
        match=re.search(r"\|\s*Verilog\s*\|\s*Pass\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)",co)
        synth.append(dict(tag=r["tag"],variant=r["variant"],tile=r["tile"],par=r["par"],
            clock_ns=float(s["clock_ns"]),cycles=int(s["latency_max"]) if s["latency_max"].isdigit() else None,
            rtl_pass=bool(match),rtl_cycles=int(match[3]) if match else None,
            **{k:int(v) for k,v in s["resources"].items() if v is not None}))
    # Prefer RTL-complete record if there are multiple modes for one tag.
    unique={}
    for r in synth:
        if r["tag"] not in unique or r["rtl_pass"]:unique[r["tag"]]=r
    synth=list(unique.values())
    (ART/"hardware_summary.json").write_text(json.dumps(synth,indent=2))
    if synth:
        with (ART/"hardware_summary.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(synth[0]));w.writeheader();w.writerows(synth)
        plotted=sorted([r for r in synth if r["cycles"] is not None],key=lambda r:(r["variant"],r["tile"],r["par"]))
        fig,axes=plt.subplots(1,3,figsize=(10,max(4,len(plotted)*.34)),sharey=True)
        labels=[r["tag"].split("_T")[0] for r in plotted]
        for ax,key,title in zip(axes,["cycles","BRAM_18K","DSP"],["HLS worst-case latency (ms)","BRAM 18K","DSP"]):
            values=[r[key]*5e-6 if key=="cycles" else r[key] for r in plotted]
            ax.barh(range(len(plotted)),values,color="#3c78a8",height=.65)
            ax.set_xlabel(title,fontsize=9);ax.grid(axis="x",alpha=.2);ax.set_axisbelow(True)
            ax.spines[["top","right"]].set_visible(False)
        axes[0].set_yticks(range(len(labels)),labels,fontsize=8);axes[0].invert_yaxis()
        fig.suptitle("VC707, 5 ns target; synthesis estimates include interface overhead",fontsize=10)
        fig.tight_layout()
        for ext in ["pdf","png"]:fig.savefig(ART/("resource_latency."+ext),dpi=200)
        plt.close(fig)
    qualified=[]
    for r in synth:
        n=ART/(r["tag"]+"_native.json")
        fits=all(r[k]<=cap for k,cap in dict(DSP=2800,BRAM_18K=2060,LUT=303600,FF=607200).items())
        if r["rtl_pass"] and r["clock_ns"]<=5 and fits and n.exists() and json.loads(n.read_text())["returncode"]==0:qualified.append(r)
    # Full resource-vector Pareto filter, then deterministic latency tie-breaking.
    keys=["rtl_cycles","BRAM_18K","DSP","LUT","FF"]
    pareto=[a for a in qualified if not any(all(b[k]<=a[k] for k in keys) and any(b[k]<a[k] for k in keys) for b in qualified)]
    best=min(pareto,key=lambda r:tuple(r[k] for k in keys)) if pareto else None
    (ART/"selection.json").write_text(json.dumps(dict(qualified=qualified,pareto=pareto,selected=best),indent=2))
    print(json.dumps(dict(module=rows,hardware_configurations=len(synth),selected=best),indent=2))
if __name__=="__main__":main()
