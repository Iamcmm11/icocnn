# Layer1 Verification Guide

## Directory Layout

```text
icocnn/
©À©¤©¤ hls_src/
©¦   ©À©¤©¤ layer1/
©¦   ©¦   ©À©¤©¤ ico_conv_layer1.hpp
©¦   ©¦   ©À©¤©¤ ico_conv_layer1.cpp
©¦   ©¦   ©À©¤©¤ test_ico_conv_layer1.cpp
©¦   ©¦   ©À©¤©¤ test_ico_conv_layer1_debug.cpp
©¦   ©¦   ©À©¤©¤ Makefile
©¦   ©¦   ©¸©¤©¤ build_layer1.bat
©¦   ©¸©¤©¤ utils.hpp
©À©¤©¤ tools/
©¦   ©¸©¤©¤ layer1/
©¦       ©À©¤©¤ generate_layer1_testdata.py
©¦       ©À©¤©¤ debug_layer1_intermediate.py
©¦       ©¸©¤©¤ compare_intermediate_layer1.py
©¸©¤©¤ hls_testdata/
    ©¸©¤©¤ layer1/
        ©À©¤©¤ input_rearranged.txt/.npy
        ©À©¤©¤ output_layer1.txt/.npy
        ©À©¤©¤ weight.txt/.npy
        ©À©¤©¤ bias.txt/.npy
        ©À©¤©¤ kernel_expansion_idx.txt/.npy
        ©À©¤©¤ reorder_idx.txt/.npy
        ©À©¤©¤ debug_intermediate/
        ©¸©¤©¤ debug_intermediate_cpp/
```

## Run Order

1. Generate layer1 testdata from model
2. Generate Python intermediate files for frame0
3. Build C executables
4. Run C full verification and C intermediate dump
5. Compare Python/C intermediate files

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
cd .\hls_src\layer1
.\build_layer1.bat
cd ..\..
```

### 4) Run C full verification

```powershell
cd .\hls_src\layer1
.\test_ico_conv_layer1.exe
cd ..\..
```

### 5) Run C intermediate dump

```powershell
cd .\hls_src\layer1
.\test_ico_conv_layer1_debug.exe
cd ..\..
```

### 6) Compare Python/C intermediate outputs

```powershell
python .\tools\layer1\compare_intermediate_layer1.py
```

## Notes

1. `generate_layer1_testdata.py` uses `layer0` input as source and computes `layer0 -> layer1` in PyTorch, so layer1 input/output are topology-consistent with model behavior.
2. Current C implementation is verification-oriented (clear mapping first), then you can apply HLS pragmas progressively.
3. If your model path changes, pass `--model` explicitly.
