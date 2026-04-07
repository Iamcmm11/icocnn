# MABA-DOA Experiments

该目录提供了一套可复现的实验沙盒，用于比较以下方案：

1. `Baseline`: 原始 `IcoTempCNN`
2. `+MABA`: `apply_cnn() -> MABA -> SoftArgMax`
3. `Ablation`: `no-gate` / `no-state` 两个消融变体

其中，两个消融变体与完整 `+MABA` 的区别如下：

1. `maba`: 完整时序精炼结构，包含 `Linear In -> causal DW-Conv -> gated state scan -> Linear Out`，并在输出端与原始响应图做残差相加。
2. `ablation_no_gate`: 保留时序状态扫描，但去掉动态门控 `alpha_t = sigmoid(G_t)`。此时遗忘系数退化为一个跨时间共享的可学习常量向量，因此模型仍然有递推记忆，但失去了逐帧自适应调节能力。
3. `ablation_no_state`: 保留输入投影、因果深度卷积和输出投影，但跳过递推状态更新。此时模块退化为一个没有选择性记忆累积的前馈式时序混合块。

可以把这三者理解为：

1. `+MABA` 用来验证完整设计是否有效；
2. `no-gate` 用来验证动态门控是否是性能提升的关键来源；
3. `no-state` 用来验证递推状态路径本身是否带来了收益。

实现刻意保持为纯 PyTorch 版本，不依赖 Triton 或自定义 CUDA kernel，因此可以直接在当前仓库环境中训练和复现实验。

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
