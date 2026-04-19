# IFAN_Edge

Incremental IFAN reproduction workspace built on top of the existing `icocnn` repository.

## Scope

- Keep the current root-level `acousticTracking*` and `icoCNN-master` code as the baseline.
- Add all IFAN and IFAN-Edge work under `IFAN_Edge/`.
- Reproduce the project in phases, with `before/after` notes for every stage.

## Layout

- `docs/`: phased reproduction notes
- `ifan_edge/`: reusable Python package
- `scripts/`: runnable helper scripts
- `configs/`: default configuration files
- `tests/`: lightweight sanity checks

## Stage 1 Goal

Stage 1 introduces a dual-feature front-end:

- `channel 0`: SRP-PHAT icosahedral map
- `channel 1`: SRP-LMS icosahedral map

The shared tensor contract is:

`[B, 2, T, 5, H, W]`

For `r=2`, this becomes:

`[B, 2, T, 5, 4, 8]`

## Stage 3 Entry Points

- Default config: `configs/stage3_default.toml`
- Training entry: `scripts/train_stage3_ifan.py`
- Baseline comparison entry: `scripts/compare_stage3_baseline.py`
- Model mainline: paper-faithful dual-input `PHAT + LMS` IFAN with shared attention, fixed `16`-channel width, deep fusion head, and optional `final_head_pooling`
