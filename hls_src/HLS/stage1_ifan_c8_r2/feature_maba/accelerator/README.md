# VC707 FeatureMABA accelerator

## Current workflow: paired comparisons, one job at a time

The previous parallel RTL batch and monitor were stopped at the user's request.
See artifacts/interrupted_parallel_jobs.json; interruption is not a numerical failure.
Start with COMPARISON.md, which compares each change against its immediate predecessor.
No command in this section launches RTL.

From the repository root:

~~~powershell
$maba = "hls_src/HLS/stage1_ifan_c8_r2/feature_maba/accelerator"
python "$maba/compare.py" --step tiling --verify
python "$maba/compare.py" --step fusion --verify
python "$maba/compare.py" --step interleave --verify
python "$maba/compare.py" --step fold --verify
~~~

Add --refresh-synth to regenerate both sides using serial C simulation and synthesis.
Add --timeout 600 to stop any individual job after ten minutes.
Defaults use existing synthesis reports and re-run native tests only when --verify is supplied.
run_all.ps1 now defaults to the tiling pair, not a whole RTL matrix.
The low-level HLS/resume runners enforce a single-job lock and stop on timeout/failure.

View the active serial job with status.py --watch or artifacts/current_job.json.
After assessing these comparisons, small RTL tests and one full RTL configuration are separate later steps.
Reduced RTL models must never be mixed into the full-shape performance table.

This is the existing IFAN_Edge pre-readout **custom MABA**, not a standard Mamba/Mamba2 implementation.
It preserves the trained checkpoint, LayerNorm epsilon 1e-5, causal K=3 convolution, two residual connections,
and zero state/zero convolution history at every invocation. No training or cross-window state change is introduced.

## Interface and variants

`maba_top(input, weights, output)` exposes three AXI memory ports and an AXI-Lite control interface.
Input/output are contiguous FP32 [T,8,6,5,2,4], T=6 for hardware comparisons.
The packing order is the original 792 model parameters followed by 64 folded state-to-output weights
and 8 folded biases (864 FP32 values). All variants use the same external shape and weight-load code.
`export_data.py` generates the binary weights and records their ordering/hash.
`legacy_adapter.hpp` adapts the existing FeatureMabaWeights/tensor interface without modifying its callers.

| MABA_VARIANT | Behavior |
|---|---|
| 0 | Materialized tensors at all main computational boundaries, 240 positions |
| 1 | Same stages and arithmetic in a position tile |
| 2 | Load/project+DWConv/norm/scan/readout/store dataflow, position then time |
| 3 | Same dataflow with time then position within each tile |
| 4 | Variant 3 plus offline residual projection folding |

Tile sizes 8,16,32 and arithmetic unroll factors 1,2,4 are supported.
The last tile has a dynamic valid count; tile 32 intentionally tests a 16-position remainder.
Every frame is written in the original external layout. Load/store transpose buffers and AXI transfers
are inside the measured top. The native FIFO model is unbounded and proves only functional behavior;
finite FIFO liveness must be checked using RTL co-simulation.

Explicit FP32 arithmetic is independent of the legacy header's optional ap_fixed detection.
The legacy header additionally accepts IFAN_FORCE_FP32 for reproducible original-node checks.
No dependency pragma suppresses a real recurrent dependency. Neither requested unroll nor loop ordering
is evidence of achieved hardware parallelism: consult synthesis and co-simulation reports.

## Reproduce

### Linux server (Vitis HLS 2024.2)

The HLS source, TCL flow, and Python runner are platform independent. On the server, install the
Linux Vitis HLS/Vivado 2024.2 distribution supported by AMD, make `vitis_hls` available on `PATH`,
and install `g++`, Python 3, and `psutil` (used by the serial job guard). No PyTorch install, training checkpoint, or scene data is needed for
the checked-in RTL smoke fixture.

From the repository root, run one design at a time:

```bash
maba=hls_src/HLS/stage1_ifan_c8_r2/feature_maba/accelerator
source /tools/Xilinx/Vitis_HLS/2024.2/settings64.sh  # adjust to the server install path
python3 -m venv "$HOME/.venvs/maba-hls"
source "$HOME/.venvs/maba-hls/bin/activate"
python3 -m pip install psutil
bash "$maba/run_linux.sh" cosim
```

For synthesis-only, use `bash "$maba/run_linux.sh" synth`; `cosim` already includes synthesis, so
there is no need to run both commands before the RTL check.

The default is fused variant 2, tile 16, PAR 1. `cosim` performs C simulation and HLS synthesis,
then runs three RTL transactions (frozen input, zeros, and the repeated frozen input). A successful
run must end with a Verilog `Pass` report. HLS projects and logs are written under the accelerator's
ignored `_work/` and `artifacts/` directories. To test all ten curated transactions in RTL, append
prefix the command with `MABA_RTL_LIMIT=0` to run all cases; this can take substantially longer.
Set `MABA_DATA_DIR` to use a different dataset,
or set `MABA_VARIANTS`, `MABA_TILE`, and `MABA_PAR` to select another configuration. For example,
`MABA_VARIANT=3 MABA_TILE=16 MABA_PAR=1 bash "$maba/run_linux.sh" cosim` selects the interleaved variant.
The default per-job timeout is 48 hours to avoid cutting off a long RTL run; override it with `MABA_TIMEOUT`.

The smoke inputs, expected outputs, packed weights, and SHA-256 manifest are in
`testdata/rtl_smoke/`; they are independent of the source machine's absolute paths. The default
fixture is intentionally small, and is suitable for functional RTL validation rather than reporting
full-scene accuracy. The complete 84-case native suite and optional scene-based exporter still need
the repository's frozen golden/checkpoint and scene files, respectively.

The Vitis install path can be supplied through `XILINX_HLS` or `XILINX_VITIS` when running the
optional mixed-precision chain check; `validate_chain.py --mixed --vitis-include /path/to/include`
is also supported. The ordinary MABA RTL flow does not require that check.

### Windows numerical workflow

Use Python with numpy, torch, einops and matplotlib, g++ for native checks, and Vitis HLS 2024.2 on PATH.
The project-local .maba_env created for this work contains the Python dependencies.
Run from the repository root (PowerShell):

```powershell
$maba = "hls_src/HLS/stage1_ifan_c8_r2/feature_maba/accelerator"
./.maba_env/Scripts/python.exe "$maba/export_data.py" --scenes --output "$maba/artifacts/data_full"
python "$maba/run.py" native --data "$maba/artifacts/data_full" --tiles 8,16,32 --pars 1,2,4
python "$maba/make_hls_subset.py"
python "$maba/compare.py" --step all --verify --refresh-synth --timeout 600
python "$maba/validate_chain.py"
python "$maba/validate_chain.py" --mixed
python "$maba/analyze.py"
```

The mixed-chain script looks for the Vitis include directory from `XILINX_HLS` or `XILINX_VITIS`;
it fails compilation if `ap_fixed.h` is unavailable instead of silently falling back to float.
The runner supports csim, synth, cosim and all, records return codes, and checks tool completion markers.
Each HLS invocation rebuilds its named project; archive artifacts before deliberately repeating a configuration.
The original V14 manuscript and its frozen evidence are not modified.

## Running-job status

artifacts/runtime_status.json records active tool PIDs and missing RTL results.
monitor.py refreshes that status every 30 seconds and regenerates figures/REPORT.md periodically.
The monitor only observes existing jobs; it does not convert synthesis completion into an RTL pass.
The original XSIM full-FP32 runs are slow on this host: a 10 us diagnostic (1977 clock cycles)
took about 16.5 seconds of simulation CPU time; O3 did not improve it.
Therefore no completion-time promise or RTL throughput result should be inferred from a running process.
The old long simulations and background monitor have now been explicitly stopped.
For the current serial workflow, artifacts/current_job.json is the live status.
runtime_status.json is only refreshed when monitor.py is explicitly run; its timestamp must be checked.

Long-sequence regression:

```powershell
./.maba_env/Scripts/python.exe "$maba/export_data.py" --frames 96 --output "$maba/artifacts/data_long"
python "$maba/run.py" native --frames 96 --data "$maba/artifacts/data_long"
```

The default C simulation uses all exported cases; default RTL simulation uses the first three transactions
(frozen input, zeros, repeated frozen input) to check back-to-back isolation. Use --rtl-limit 0 to run all
cases, or a separate manifest ordering for focused boundary/held-out RTL checks. C/RTL comparison is distinct
from the 1e-5 max-abs / 1e-6 RMSE comparison against PyTorch. Invalid/nonfinite outputs fail explicitly.

The complete 84-case native suite is complemented by ten curated HLS cases (three reset transactions,
tile tail, zero variance, both saturated gates, and one window from each held-out scene).
Parameter-only synthesis sweeps may use --limit 3 after the same configuration passes the full native suite.
For finite FIFO occupancy profiling, use resume_rtl.py TAG --profile on an existing synthesis;
ordinary co-simulation omits profiling for speed. Interrupted runs are not passing evidence.

## Evidence and selection

artifacts contains numerical JSON, raw logs, copied synthesis/co-simulation reports, source hashes,
data manifests, references, figures, hardware_summary.csv and selection.json.
Generated binary inputs, executables, output arrays and tool build directories are ignored by Git.
analyze.py selects only numerical-pass, RTL-pass, <=5 ns configurations, filters resource/latency Pareto
candidates, and breaks ties by RTL latency, BRAM, DSP, LUT, then FF.

Whole-chain tests report both the incremental change caused by replacing MABA and the total error against
the software reference. The historical mixed-precision backbone still fails the strict total-error gate.
An incremental pass does not turn that total failure into a pass.

FP64 software readout comparisons across scenes measure paired direction sensitivity; they are not new
HLS post-processing implementations or dataset-level localization-accuracy results.
No board power, achieved place-and-route timing, or full-network RTL result is claimed.

## Attribution

Design ideas were studied in [LightMamba](https://github.com/PKU-SEC-Lab/LightMamba)
([DATE 2025 paper](https://doi.org/10.23919/DATE64628.2025.10993079)),
[ViM-Q](https://github.com/shengzhelyu65/ViM-Q-FCCM-2026), and
[mamba790m_hls](https://github.com/todi-mih/mamba790m_hls).
Exact repository revisions are recorded in artifacts/references.json.
This implementation is independently written; no source code was copied from those repositories.
Generic tiling/FIFO fusion, position interleaving, and affine folding are established techniques.
The contribution under investigation is their measured combination for 240 independent short
directional sequences and the specific MABA residual projection, not a claim of inventing those techniques.
