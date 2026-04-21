# IFAN_Maba

Independent IFAN experiment fork for validating a full MABA replacement of the stage-3 temporal stack.

## Scope

- Keep the current root-level `acousticTracking*` and `icoCNN-master` code as the baseline.
- Keep `IFAN_Edge/` untouched as the current IFAN mainline reference.
- Run all MABA replacement work under `IFAN_Maba/`.
- Share only the stage-3 code needed for the fork, while keeping outputs under `IFAN_Maba/outputs/`.

## Layout

- `docs/`: phased reproduction notes
- `ifan_maba/`: reusable Python package
- `scripts/`: runnable helper scripts
- `configs/`: default configuration files
- `tests/`: lightweight sanity checks

## Fixed Frontend

- `channel 0`: SRP-PHAT icosahedral map
- `channel 1`: SRP-LMS icosahedral map

The shared tensor contract is:

`[B, 2, T, 5, H, W]`

For `r=2`, this becomes:

`[B, 2, T, 5, 4, 8]`

## Stage 3 Entry Points

- Default config: `configs/stage3_maba_default.toml`
- Training entry: `scripts/train_stage3_ifan_maba.py`
- Larger-cache simulated analysis: `scripts/evaluate_stage3_simulated.py`
- LOCATA analysis: `scripts/evaluate_stage3_locata.py`
- Model mainline: paper-faithful dual-input `PHAT + LMS` IFAN with shared attention, fixed `16`-channel width, deep fusion head, and `temporal_backend = "maba"` by default
