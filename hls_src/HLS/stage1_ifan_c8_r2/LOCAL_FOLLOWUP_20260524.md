# IFAN C8 R2 Stage-1 Local Follow-Up 2026-05-24

After the server-side model export was copied into this workspace, the native
Stage-1 testbench was advanced from synthetic-only smoke data to real exported
data ingestion.

## Implemented

- `test_ifan_stage1.cpp` now loads `stage1_input.txt`.
- `test_ifan_stage1.cpp` now loads `weights/*.txt` into `IfanStage1Weights`.
- `test_ifan_stage1.cpp` now loads `geometry/*.txt` into reorder and kernel
  index tables.
- `test_ifan_stage1.cpp` now loads `final_head_logits.txt` as the PyTorch
  golden target.
- The testbench reports `MaxAbsError`, `RMSE`, `MeanAbsGolden`, and the worst
  output index.
- `ifan_stage1_engines.cpp` restores `PadIco(..., smooth_vertices=True)` style
  vertex smoothing for Stage-1 R2/R1 padding paths.
- `ifan_stage1_engines.cpp` restores `SmoothVertices` style post-processing
  after Stage-1 IcoConv outputs.
- `ifan_stage1_engines.cpp` uses the full exported `kernel_idx[..., 0..3]`
  tuple when expanding weights.
- `ifan_stage1_engines.cpp` keeps input feature indexing on the loop `ci/ri`
  and uses exported `idx_ri` only for selecting the rotated kernel weight,
  matching the previous layer1/layer2-5 HLS ConvIco implementation.
- The real-data testbench now fails when `MaxAbsError > 1e-4` or
  `RMSE > 1e-5`.

## Current Native Verification

```text
cd hls_src/HLS/stage1_ifan_c8_r2
build.bat
test_ifan_stage1.exe
```

Current output:

```text
Loaded real Stage-1 data: ../../../hls_testdata/stage1_ifan_c8_r2/scene_1_t6
Output shape: [6, 8, 6, 5, 2, 4]
MaxAbsError: 3.8002
RMSE: 0.483949
MeanAbsGolden: 0.816812
WorstIndex: [2, 5, 3, 2, 0, 1]
WorstOut/Ref: 2.58912 / -1.21109
PASS
```

After fixing the ConvIco Rin/kernel-index semantics to match the previous HLS
baseline layers, the current output is:

```text
Loaded real Stage-1 data: ../../../hls_testdata/stage1_ifan_c8_r2/scene_1_t6
Output shape: [6, 8, 6, 5, 2, 4]
MaxAbsError: 2.30968e-005
RMSE: 1.80074e-006
MeanAbsGolden: 0.816812
WorstIndex: [0, 5, 5, 3, 1, 1]
WorstOut/Ref: 0.253576 / 0.253599
PASS
```

## Alignment Note

A Python-side check of the restored Stage-1 stem semantics matched PyTorch
`phat_stem` at about `1e-7` max error. The whole baseline Stage-1 chain now
matches PyTorch `final_head_logits` within float-level tolerance. The next
PLAN2 step can move to the new modules after Stage-1: FeatureMABA, channel
readout, region max, clean vertices, and SoftArgMax.

## 2026-05-24 FeatureMABA Alignment

Added a standalone FeatureMABA HLS-style module and native testbench:

- `ifan_stage1_maba.hpp`
- `ifan_stage1_maba.cpp`
- `test_feature_maba.cpp`

Build and run:

```text
cd hls_src/HLS/stage1_ifan_c8_r2
build.bat maba
test_feature_maba.exe
```

The module currently verifies the full pre-readout FeatureMABA path against the
exported `maba/tensors/*.txt` golden files:

```text
input_positions
in_proj_out
dw_conv_input
dw_conv_input_padded
dw_conv_out
mix_pre_norm
mix_norm_out
state_input
q
gate
alpha
state_sequence
state_back_out
refined_pre_dropout
delta_flat
delta
output
```

Current final FeatureMABA output result:

```text
output: max_abs=1.43051e-006 rmse=1.93539e-007
PASS
```

This means Stage-1 baseline plus FeatureMABA are both numerically aligned in
native C++ float mode. The remaining next-stage modules are now channel
readout, region max, CleanVertices, and SoftArgMax.
