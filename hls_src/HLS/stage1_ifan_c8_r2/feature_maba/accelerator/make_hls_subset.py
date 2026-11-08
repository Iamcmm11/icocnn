"""Curated hardware regression, separate from the complete 84-case native suite."""
import json
import shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
src=HERE/"artifacts/data_full";dst=HERE/"artifacts/data_hls";dst.mkdir(exist_ok=True)
names=["frozen_v14","zeros","frozen_repeat","last_position","zero_variance",
       "gate_positive","gate_negative","scene2_000","scene3_000","scene4_000"]
m=json.loads((src/"manifest.json").read_text())
lookup={c["name"]:c for c in m["cases"]}
shutil.copy2(src/"weights.bin",dst/"weights.bin")
for name in names:
    for suffix in [".input.bin",".output.bin",".weights.bin"]:
        p=src/(name+suffix)
        if p.exists():shutil.copy2(p,dst/p.name)
m["cases"]=[lookup[n] for n in names];m["subset_of"]="data_full"
(dst/"manifest.json").write_text(json.dumps(m,indent=2))
(dst/"cases.txt").write_text("".join(f"{n} {lookup[n]['valid_frames']}\n" for n in names))
(HERE/"artifacts/data_manifest_hls.json").write_text(json.dumps(m,indent=2))
print(f"Prepared {len(names)} hardware cases")
