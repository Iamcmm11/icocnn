# Layer2-5 Verification Guide

## Directory Layout

```text
icocnn/
├── hls_src/
│   └── HLS/
│       └── layer2-5/
│           ├── ico_conv_layer2_5.hpp
│           ├── ico_conv_layer2_5.cpp
│           ├── test_ico_conv_layer2_5.cpp
│           ├── test_ico_conv_layer2_5_debug.cpp
│           ├── build_layer2_5.bat
│           ├── run_hls.bat
│           ├── run_hls.tcl
│           └── parse_hls_report.py
├── tools/
│   └── layer2-5/
│       ├── generate_layer2_5_testdata.py
│       ├── debug_layer2_5_intermediate.py
│       └── compare_intermediate_layer2_5.py
└── hls_testdata/
    └── layer2-5/
        ├── layer2/
        ├── layer3/
        ├── layer4/
        └── layer5/
```

## Design Choice

`layer2-5` share the same ConvIco spatial shape:

1. `Cin = 32`
2. `Cout = 32`
3. `Rin = 6`
4. `Rout = 6`
5. `r = 1`, so `H = 2`, `W = 4`

So this directory provides a shared verification implementation for the repeated spatial block.

## Run Order

1. Generate per-layer testdata for `layer2-5`
2. Generate Python intermediates for one target layer
3. Build C executables
4. Run C full verification for one target layer
5. Run C intermediate dump for one target layer
6. Compare Python/C intermediates for that layer

## Commands

### 1) Generate testdata for layers 2-5

```powershell
python .\tools\layer2-5\generate_layer2_5_testdata.py `
  --model models\1sourceTracking_icoCNN_robot_K4096_r2_model.bin `
  --layer0-input hls_testdata\layer0\input_rearranged.npy `
  --out-dir hls_testdata\layer2-5 `
  --layers 2,3,4,5 `
  --time-steps 52
```

### 2) Generate Python intermediates for one layer (example: layer2)

```powershell
python .\tools\layer2-5\debug_layer2_5_intermediate.py --layer 2
```

### 3) Build C executables

```powershell
cd .\hls_src\HLS\layer2-5
.\build_layer2_5.bat
cd ..\..\..
```

### 4) Run C full verification for one layer

```powershell
cd .\hls_src\HLS\layer2-5
.\test_ico_conv_layer2_5.exe 2
cd ..\..\..
```

### 5) Run C intermediate dump for one layer

```powershell
cd .\hls_src\HLS\layer2-5
.\test_ico_conv_layer2_5_debug.exe 2
cd ..\..\..
```

### 6) Compare Python/C intermediates

```powershell
python .\tools\layer2-5\compare_intermediate_layer2_5.py --layer 2
```

## HLS

The HLS scripts use the shared `layer2-5` ConvIco block implementation.

```powershell
cd .\hls_src\HLS\layer2-5
.\run_hls.bat
.\run_hls.bat csim
.\run_hls.bat synth
```

Notes:

1. The testbench defaults to `layer2` if no layer id is passed.
2. Because layers 2-5 share the same spatial block shape, the HLS synthesis result is representative for the repeated ConvIco block itself.
