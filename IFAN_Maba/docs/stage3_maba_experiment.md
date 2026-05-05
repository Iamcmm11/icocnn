# IFAN_Maba Stage-3 Experiment

## Goal

`IFAN_Maba/` is an isolated fork for one question: does replacing every stage-3 `CausConv1d(16 -> 16)` temporal block with a MABA-style channel-temporal module improve hard simulated scenes and LOCATA dynamic tasks without changing the rest of the current IFAN mainline?

## Why Full Replacement

The stage-3 fusion head has five temporal Conv1d sites:

- four `FusionTemporalBlock` temporal layers
- one `FinalFusionBlock` temporal layer

This fork replaces all five so the result answers a clean structural question instead of mixing partial replacement effects.

## Why Keep 40 Epochs

This branch is a structure-validation fork, not a new training-mainline search. We therefore keep the currently effective IFAN training budget:

- `epochs = 40`
- `phase1_epochs = 20`
- `batch_size_phase1 = 1`
- `batch_size_phase2 = 10`
- `lr_phase1 = 1e-4`
- `lr_phase2 = 1e-5`

That keeps any metric change attributable to the temporal backend rather than to a different schedule.

## Why Keep `final_head_pooling = false`

The current IFAN line already found implementation-level regression risk around `final_head_pooling`. This fork is meant to isolate the temporal-module swap, so it keeps the accepted mainline setting:

- `final_head_pooling = false`

## Expected Hypothesis

MABA adds per-channel temporal state and gating while preserving the existing `(N, C, T)` contract used by IFAN's reshape path. The working hypothesis is:

- easy scenes and LOCATA Task 1 may move only slightly
- dynamic and harder conditions, especially LOCATA Task 3 and Task 5, have the best chance to benefit

The reason is that Tasks 3 and 5 emphasize temporal continuity, motion, and nontrivial sequence context more strongly than static or easier settings.
