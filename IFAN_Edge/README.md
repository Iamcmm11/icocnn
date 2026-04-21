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
- Long-budget config: `configs/stage3_long_budget.toml`
- LMS reference probe config: `configs/stage3_reference_backend_probe.toml`
- Training entry: `scripts/train_stage3_ifan.py`
- Baseline comparison entry: `scripts/compare_stage3_baseline.py`
- Larger-cache simulated analysis: `scripts/evaluate_stage3_simulated.py`
- Paper-vs-local protocol audit: `scripts/audit_stage3_protocol.py`
- Transition gate assessment: `scripts/assess_stage3_readiness.py`
- Scenario-focused analysis: `scripts/analyze_stage3_scene.py`
- Run-to-run gate comparison: `scripts/compare_stage3_runs.py`
- Model mainline: paper-faithful dual-input `PHAT + LMS` IFAN with shared attention, fixed `16`-channel width, deep fusion head, and optional `final_head_pooling`

The current default stage-3 config is now the locked IFAN mainline used for reproduction-gap analysis:

- `paper_dual_mainline`
- `PHAT + LMS`
- `paper_original`
- `lms_backend = frequency_block`
- `final_head_pooling = false`
- `epochs = 40`
