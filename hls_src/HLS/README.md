# HLS Verification Workspace

This directory is the organized HLS simulation/verification workspace for multi-layer integration.

## Directory Layout

- `layer0/`: Layer0 HLS source, C simulation testbench, debug testbench, bat/tcl scripts.
- `layer1/`: Layer1 HLS source, C simulation testbench, debug testbench, build script.
- `common/`: shared helper headers (currently `utils.hpp`).

## Why split by layer

Use per-layer folders (`layer0`, `layer1`, ...):

1. Better isolation of per-layer parameters and test vectors.
2. Easier staged bring-up before integrating a unified top-level IP.
3. Lower regression risk when optimizing one layer (pragma/pipeline/dataflow changes).

For a future single IP core, keep this structure and add:

1. `top/` with an integration wrapper (`conv_ico_top.cpp/.hpp`).
2. Shared interfaces in `common/`.
3. Layer-level csim as smoke tests + top-level csim/cosim for system checks.

## Build and Verify

### Layer0

```bash
cd hls_src/HLS/layer0
make clean && make test_ico_conv
./test_ico_conv

g++ -std=c++11 -O2 -I. -Wall -o test_ico_conv_debug test_ico_conv_debug.cpp ico_conv_layer0.cpp
./test_ico_conv_debug
```

### Layer1

```bash
cd hls_src/HLS/layer1
make clean && make test_ico_conv_layer1 test_ico_conv_layer1_debug
./test_ico_conv_layer1
./test_ico_conv_layer1_debug
```

## Numeric Alignment Notes (Layer0 vs Layer1)

Current measured full-output errors:

- Layer0: `Max Error = 9.53674e-07`, `RMSE = 7.03355e-08`
- Layer1: `Max Error = 1.23978e-05`, `RMSE = 1.00145e-06`

Reason layer1 usually differs in the last digits while layer0 can look nearly exact:

1. Layer1 accumulation depth is much larger.
2. Layer1 uses `Cin=32, Rin=6` (many more MAC terms per output) while Layer0 uses `Cin=1, Rin=1`.
3. Floating-point non-associativity makes tiny rounding differences grow with operation count.
4. Debug text export precision can amplify apparent tiny mismatch if too few decimals are written.

After fixing layer1 indexing/smoothing logic, remaining difference is in normal float tolerance and now passes verification.
