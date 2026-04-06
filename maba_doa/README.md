# MABA-DOA Experiments

This directory contains a reproducible sandbox for:

1. `Baseline`: original `IcoTempCNN`
2. `+MABA`: `apply_cnn() -> MABA -> SoftArgMax`
3. `Ablation`: no-gate / no-state variants

The implementation is intentionally pure PyTorch (no Triton/CUDA custom kernels), so it can be trained in the current repository environment.

## Structure

- `maba_doa/models.py`: `MABATemporalRefiner` and `IcoTempCNNWithMABA`
- `maba_doa/train_maba_doa.py`: single-run training and evaluation
- `maba_doa/run_ablation.py`: fixed baseline + ablation suite
- `maba_doa/plot_history_compare.py`: compare `history.csv` curves and export summary table
- `maba_doa/visualize_maps.py`: map-level before/after visualization and jitter report
- `maba_doa/configs/default.yaml`: default experiment config
- `maba_doa/tests/test_maba_doa.py`: shape and integration smoke tests

## Environment

Required:

1. Existing project dependencies for `acousticTracking*` pipeline
2. `pyyaml`

Install:

```bash
pip install pyyaml
```

## One-command training

```bash
python maba_doa/train_maba_doa.py --config maba_doa/configs/default.yaml
```

Useful overrides:

```bash
python maba_doa/train_maba_doa.py --config maba_doa/configs/default.yaml --variant baseline --epochs 2 --cpu
python maba_doa/train_maba_doa.py --config maba_doa/configs/default.yaml --variant maba --epochs 2
```

## One-command ablation suite

```bash
python maba_doa/run_ablation.py --config maba_doa/configs/default.yaml --epochs 2
```

The suite runs:

1. `baseline`
2. `maba`
3. `ablation_no_gate`
4. `ablation_no_state`

## Outputs

All outputs are written to `maba_doa/outputs/`:

1. `config.yaml`: frozen runtime config
2. `history.csv`: per-epoch metrics (`test_loss`, `test_rmsae_deg`)
3. `summary.json`: final stats (`param_count`, `maba_mac_proxy`, `latency_step_ms`)
4. `model.bin`: trained checkpoint
5. `ablation_summary.json`: combined report (for ablation script)

## Visualization

```bash
python maba_doa/visualize_maps.py --config maba_doa/configs/default.yaml --checkpoint maba_doa/outputs/<run_dir>/model.bin --frame 0 --output maba_doa/outputs/map_refinement.png
```

## History comparison

Auto-pick latest `baseline/maba/ablation_*` runs under outputs:

```bash
python maba_doa/plot_history_compare.py --output-root maba_doa/outputs
```

The script skips incomplete runs automatically, exports:

1. `history_compare.png`: Loss/RMSAE vs epoch curves
2. `history_compare_summary.csv`: per-run final/best metrics
3. `history_compare_merged.csv`: merged per-epoch history rows

Or compare explicit run dirs:

```bash
python maba_doa/plot_history_compare.py --run-dirs maba_doa/outputs/<run_a> maba_doa/outputs/<run_b>
```

## Testing

Run smoke tests:

```bash
python -m unittest maba_doa.tests.test_maba_doa
```
