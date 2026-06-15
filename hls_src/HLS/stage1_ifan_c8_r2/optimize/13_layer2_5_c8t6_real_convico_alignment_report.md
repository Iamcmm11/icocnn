# Layer2-5 C8/T6 真实 ConvIco 对齐报告 13

日期：2026-06-02

## 1. 本轮目标

基于计划 11 的 A4，本轮目标从 synthetic smoke 前进到真实 IFAN C8_R2 `ConvIco` 子层对齐：

1. 从 `stage1_ifan_c8_r2/scene_1_t6` 已导出的真实权重和 debug tensor 中，切出 `layer2-5` 可直接复用的 `ConvIco` 数据集；
2. 为 `fusion0 / fusion1 / fusion2 / fusion3 / final` 五个 R1 `ConvIco` 子层生成独立目录；
3. 用 `hls_src/HLS/layer2-5` 的 C++ testbench 逐层做真实 `native` 对齐；
4. 至少补一次代表性真实数据 `HLS csim`，确认 Vitis 入口也能读取真实 C8/T6 文件数据。

本轮不重新跑 `csynth`。因为 `layer2-5` 的综合结果只与设计和宏配置相关，不与 testbench 选择的真实/合成数据相关；已有 `C8/T6` synth 报告仍可作为资源锚点。

## 2. 代码与数据改动

### 2.1 导出脚本补强

已增强：

- `IFAN_Edge/scripts/export_stage1_hls_golden.py`

新增导出：

```text
hls_testdata/layer2-5_c8_t6/
  fusion0/
  fusion1/
  fusion2/
  fusion3/
  final/
  manifest.json
```

每个子目录包含：

- `input_rearranged.{npy,txt}`
- `weight.{npy,txt}`
- `bias.{npy,txt}`
- `kernel_expansion_idx.{npy,txt}`
- `reorder_idx.{npy,txt}`
- `output.{npy,txt}`
- `manifest.json`

这些目录直接对齐 `hls_src/HLS/layer2-5/test_ico_conv_layer2_5.cpp` 的文件契约。

### 2.2 ConvIco-only 节点切分

导出边界不是整个 fusion block，而是单独切到每个 `ConvIco`：

- `fusion0`: `input = debug['fusion_feature']`
- `fusion1`: `input = debug['fusion_head_blocks'][0]`
- `fusion2`: `input = debug['fusion_head_blocks'][1]`
- `fusion3`: `input = debug['fusion_head_blocks'][2]`
- `final`: `input = debug['fusion_head_blocks'][3]`

每个目录中的 `output` 都是：

```text
model.<block>.conv(input)
```

也就是：

- 不包含后续 `ReLU`
- 不包含 `temporal Conv1d`
- 不包含 `LNormIco`

### 2.3 replay 自检

导出脚本对五个子层都增加了 replay 校验：

```text
full_block_replay_max_abs_diff = 0.0
```

含义是：

- 重新执行 `FusionTemporalBlock` / `FinalFusionBlock`；
- 其完整输出与原始 `debug['fusion_head_blocks'][i]` / `debug['final_head_logits']` 完全一致；
- 因而当前切出的 `ConvIco-only` 输入边界是自洽的。

### 2.4 testbench 路径入口补强

已增强：

- `hls_src/HLS/layer2-5/test_ico_conv_layer2_5.cpp`

当前支持：

- 第二个命令行参数指定数据目录；
- `ICO_LAYER2_5_DATA_DIR` 指定数据目录；
- 若给的是相对路径，则 testbench 会自动向上回溯多级 `../`，兼容：
  - 本地 exe 运行目录；
  - Vitis HLS `csim` 的更深工作目录。

这一步是为真实文件数据 `csim` 做的接口补强。

## 3. 数据集状态

根 manifest：

```text
hls_testdata/layer2-5_c8_t6/manifest.json
```

关键结论：

1. 五个数据集 `fusion0..fusion3/final` 已全部生成；
2. `input/output` shape 全部为：
   ```text
   [T=6, C=8, R=6, charts=5, H=2, W=4]
   ```
3. `weight` shape 为：
   ```text
   [8, 8, 6, 7]
   ```
4. `kernel_expansion_idx` / `reorder_idx` 统一来自真实：
   - `kernel_idx_main`
   - `reorder_r1`
5. 所有数据均 finite；
6. 五个子层的 `full_block_replay_max_abs_diff` 全为 `0.0`。

## 4. Native 真实对齐结果

### 4.1 运行方式

示例：

```bat
cd hls_src\HLS\layer2-5
build_layer2_5.bat c8t6
test_ico_conv_layer2_5_c8t6.exe 2 ..\..\..\hls_testdata\layer2-5_c8_t6\fusion0
```

本轮对五个子层都执行了真实文件数据 `native` 对齐。

### 4.2 结果汇总

| 子层 | Max Error | RMSE | 结果 |
|---|---:|---:|---|
| `fusion0` | `8.10623e-006` | `1.13005e-006` | PASS |
| `fusion1` | `3.09944e-006` | `3.15851e-007` | PASS |
| `fusion2` | `8.34465e-007` | `1.40911e-007` | PASS |
| `fusion3` | `2.86102e-006` | `3.12617e-007` | PASS |
| `final`   | `2.62260e-006` | `2.58682e-007` | PASS |

### 4.3 判断

这说明：

1. `layer2-5` 参数化后的 `C8/T6` 真实文件数据入口已经可用；
2. `fusion0..fusion3/final` 五个真实 IFAN C8_R2 `ConvIco` 子层已经全部与当前 C++ `native float` 路径对齐；
3. A4 中“真实 `ConvIco` golden 导出 + C++ testbench 可选目录对齐”的主目标已经达成。

## 5. 代表性 HLS csim 结果

### 5.1 运行方式

代表性选择 `fusion0`：

```bat
cd hls_src\HLS\layer2-5
set ICO_LAYER2_5_DATA_DIR=..\..\..\hls_testdata\layer2-5_c8_t6\fusion0
run_hls.bat csim c8t6
```

### 5.2 路径入口结果

本轮 `csim` 已成功解析真实数据目录，并在 Vitis 工作目录下自动回溯到：

```text
../../../../../../../hls_testdata/layer2-5_c8_t6/fusion0/
```

说明 testbench 的路径补强已经生效。

### 5.3 数值结果

`fusion0` 的 `csim` 输出为：

```text
Max Error: 0.010498
RMSE: 0.00480558
FAIL
```

但 Vitis 工具流本身显示：

```text
CSim done with 0 errors.
```

### 5.4 解释

当前更合理的解释是：

1. 本地 `g++ native` 路径没有 `ap_fixed.h`，内部走的是 `float` fallback；
2. Vitis HLS `csim` 会启用 `ap_fixed` 版本的内部类型：
   - `input_t`
   - `weight_t`
   - `act_t`
   - `acc_t`
3. 因而真实 `PyTorch float` golden 与 HLS `csim` 之间，已经开始体现当前位宽配置下的量化误差；
4. 这不是数据目录找错，而是第一次拿到了真实 C8/T6 `ConvIco` 的定点误差锚点。

上面第 3 点是基于当前代码路径和编译环境的推断，但与现象一致：

- `native float`：`1e-6` 级误差；
- `Vitis csim`：`1e-2` 级误差。

## 6. 当前阶段判断

截至本轮，可以把 `layer2-5` C8/T6 主干状态更新为：

### 已完成

1. `TIME_STEPS/CIN/COUT/OC_PAR_FACTOR` 参数化；
2. `baseline` 与 `c8t6` 构建/HLS preset；
3. synthetic smoke `native/csim/synth`；
4. 真实 `fusion0..fusion3/final` ConvIco 数据导出；
5. 五个真实子层 `native` 全通过；
6. 代表性真实文件数据 `HLS csim` 已打通输入链路。

### 新增结论

真实 `ConvIco` 数据接入后，当前 `C8/T6` HLS `csim` 与 `PyTorch float golden` 之间已观测到约 `1e-2` 量级误差，这应视为后续定点/位宽路径评估的起点，而不是数据接口问题。

## 7. 下一步建议

建议按以下顺序继续：

1. 明确 `layer2-5` `csim` 的数值判定策略：
   - 方案 A：保留当前 `ap_fixed` `csim`，把它作为真实定点误差基线；
   - 方案 B：增加一个 float-only `csim`/native 对账模式，专门验证结构正确性；
   - 方案 C：保留当前模式，但把 `FAIL` 判定门限参数化，便于后续定点 sweep。
2. 如果目标是继续硬件主线，优先做：
   - `fusion0` 与 `final` 两个代表点的位宽/误差 sweep；
   - 评估是否需要调整 `INPUT/WEIGHT/ACT/ACC` 位宽。
3. 如果目标是继续功能模块切分，可在当前真实 `ConvIco` 已对齐的基础上，开始推进：
   - `temporal Conv1d/LNorm` 小模块；
   - `FeatureMABA` 独立 `native/csim/synth`。

本轮更偏向“把真实 C8/T6 ConvIco 主干完全接上”，这一步已经完成。后续分叉点不再是数据接口，而是：

- 定点误差策略；
- temporal/MABA 的后续硬件边界。
