# Layer0 / Layer1 Numeric Difference Analysis

## Observation

- Layer0 can be almost bit-identical to reference in many intermediate tensors.
- Layer1 usually matches within tolerance, but may differ in the last few decimal digits.

## Measured Results

1. Layer0 full-output compare:
- Max Error: `9.53674e-07`
- RMSE: `7.03355e-08`

2. Layer1 full-output compare:
- Max Error: `1.23978e-05`
- RMSE: `1.00145e-06`

3. Layer1 frame0 intermediate compare:
- Input: max `0`
- After PadIco: max `7.2e-07`
- Final output: max `8.59e-06`

## Why Layer1 Has More Tail-Digit Error

1. Higher accumulation depth:
- Layer0 effective per-output accumulation is small (`Cin=1`, `Rin=1`).
- Layer1 accumulation is much deeper (`Cin=32`, `Rin=6`), so floating-point rounding accumulates.

2. Floating-point non-associativity:
- Even mathematically equivalent expressions can produce tiny drift when operation count rises.

3. Debug text precision effects:
- If text export precision is low, parsed comparison may show inflated tail differences.

## Important: Fixed Logic Errors vs Normal FP Drift

Before this update, Layer1 had two real C-side logic mismatches:

1. `reorder_idx` decode missed `RIN` dimension (`src_ri`).
2. `SmoothVertices` was averaged per `ri` instead of averaging across `R` and neighbors (Python behavior).

Those are fixed. Current residual is normal floating-point tolerance level.
