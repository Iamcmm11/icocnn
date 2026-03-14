# Layer1 Verification Guide

## Directory Layout

```text
icocnn/
├── hls_src/
│   └── HLS/
│       ├── common/
│       │   └── utils.hpp
│       └── layer1/
│           ├── ico_conv_layer1.hpp
│           ├── ico_conv_layer1.cpp
│           ├── test_ico_conv_layer1.cpp
│           ├── test_ico_conv_layer1_debug.cpp
│           ├── Makefile
│           └── build_layer1.bat
├── tools/
│   └── layer1/
│       ├── generate_layer1_testdata.py
│       ├── debug_layer1_intermediate.py
│       └── compare_intermediate_layer1.py
└── hls_testdata/
    └── layer1/
        ├── input_rearranged.txt/.npy
        ├── output_layer1.txt/.npy
        ├── weight.txt/.npy
        ├── bias.txt/.npy
        ├── kernel_expansion_idx.txt/.npy
        ├── reorder_idx.txt/.npy
        ├── debug_intermediate/
        └── debug_intermediate_cpp/
```

## Run Order

1. Generate layer1 testdata from model.
2. Generate Python intermediate files for frame0.
3. Build C executables.
4. Run C full verification and C intermediate dump.
5. Compare Python/C intermediate files.

## Commands (PowerShell, repo root)

### 1) Generate layer1 testdata

```powershell
python .\tools\layer1\generate_layer1_testdata.py `
  --model models\1sourceTracking_icoCNN_robot_K4096_r2_model.bin `
  --layer0-input hls_testdata\layer0\input_rearranged.npy `
  --out-dir hls_testdata\layer1 `
  --time-steps 52
```

### 2) Generate Python side layer1 intermediate outputs

```powershell
python .\tools\layer1\debug_layer1_intermediate.py
```

### 3) Build C layer1 tests

```powershell
cd .\hls_src\HLS\layer1
.\build_layer1.bat
cd ..\..\..
```

### 4) Run C full verification

```powershell
cd .\hls_src\HLS\layer1
.\test_ico_conv_layer1.exe
cd ..\..\..
```

### 5) Run C intermediate dump

```powershell
cd .\hls_src\HLS\layer1
.\test_ico_conv_layer1_debug.exe
cd ..\..\..
```

### 6) Compare Python/C intermediate outputs

```powershell
python .\tools\layer1\compare_intermediate_layer1.py
```

## Notes

1. `generate_layer1_testdata.py` uses `layer0` input as source and computes `layer0 -> layer1` in PyTorch, so layer1 input/output are topology-consistent with model behavior.
2. Current C implementation is verification-oriented (clear mapping first), then you can apply HLS pragmas progressively.
3. Layer1 debug outputs use MATLAB-style matrix slices with shape headers, matching Layer0 readability.
4. If your model path changes, pass `--model` explicitly.
