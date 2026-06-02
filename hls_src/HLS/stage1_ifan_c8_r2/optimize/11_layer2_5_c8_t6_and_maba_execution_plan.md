# Layer2-5 C8/T6 主干复用与 MABA 推进执行计划 11

日期：2026-06-02

## 1. 当前判断

下一阶段先推进主干网络 `layer2-5` 是正确主线。原因是：

1. `stage1_ifan_c8_r2` 已经完成分区重构，不再适合继续复制一套 fusion/final 主干。
2. C8_R2 的 fusion head 和 final head 中，核心 IcoConv 与已有 `hls_src/HLS/layer2-5` 同型：

```text
ConvIco(Cin=Cout, Rin=Rout=6, R=1, H=2, W=4)
```

3. 差异主要是规模参数和外围算子：

| 对象 | C | T | 说明 |
|---|---:|---:|---|
| baseline `layer2-5` | 32 | 52 | 现有可 synth 的 R1 ConvIco 基线 |
| IFAN C8_R2 fusion/final ConvIco | 8 | 6 | 当前 Stage-1 HLS golden 窗口 |

已有 `layer2-5` 可完整生成 `csynth` 报告，当前记录为：

| 指标 | 数值 |
|---|---:|
| Target clock | 5.00 ns |
| Estimated clock | 4.209 ns |
| Latency | 261,686,621 cycles |
| BRAM_18K | 64 / 890 |
| DSP | 72 / 840 |
| FF | 43,639 / 407,600 |
| LUT | 69,382 / 203,800 |

这组结果应作为后续 C8/T6 版本的对照基线。

## 2. 关于 `T=6` 的结论

`T=6` 不是 Torch 算法层面的原始全序列要求。

当前证据是：

1. 原始/训练配置仍使用 `trajectory_seconds = 20`。
2. 已归档的 `dual_maps.npy` 原始 shape 是 `[1, 2, 103, 5, 4, 8]`。
3. `export_stage1_hls_golden.py` 默认 `--frames=6`，并显式限制当前 Stage-1 HLS 工程期望 `IFAN_STAGE1_T=6`。
4. 当前 HLS golden 目录是 `hls_testdata/stage1_ifan_c8_r2/scene_1_t6/`，表示从完整特征序列中截取 6 帧窗口。
5. 旧 baseline HLS `layer0/layer1/layer2-5` 仍使用 `TIME_STEPS=52`。

因此，论文和报告中建议这样表述：

```text
T=6 是当前 C8_R2 HLS 子模块验证窗口和 golden 导出窗口，
不是原始 PyTorch 训练/推理的完整时间长度。
后续硬件评估以 6-frame streaming/windowed PL kernel 为单位，
完整序列可通过 PS 侧调度或滑窗/分块方式调用。
```

如果最终论文需要面向完整 20 秒轨迹，应在报告中补充从 `T=6` kernel 到完整序列的调度假设和吞吐估算，不能直接把 6 帧结果说成完整序列端到端结果。

## 3. 阶段目标

本阶段目标不是恢复完整 Stage-1 纯 PL top，而是形成三组可比较证据：

1. `layer2-5` baseline `C=32,T=52`：保留已有报告作为对照。
2. `layer2-5` C8/T6 复用版：完成 native、HLS `csim`、HLS `synth`。
3. `feature_maba`：完成 native、HLS `csim`、HLS `synth`，作为 PL 创新模块候选。

阶段完成后，应能回答：

- C8/T6 ConvIco 主干复用后资源、延迟相比 C32/T52 下降多少？
- FeatureMABA 独立上 PL 的资源和延迟是否可接受？
- post-MABA head 是否值得继续上 PL，还是留在 PS 更合理？

## 4. 执行路线 A：优先推进 `layer2-5` C8/T6

### A1. 参数化现有 `layer2-5`

修改目标：

- `hls_src/HLS/layer2-5/ico_conv_layer2_5.hpp`
- `hls_src/HLS/layer2-5/build_layer2_5.bat`
- `hls_src/HLS/layer2-5/run_hls.bat`
- `hls_src/HLS/layer2-5/run_hls.tcl`
- 必要时同步 `Makefile` 和 testbench 参数入口

建议把硬编码：

```cpp
#define TIME_STEPS  52
#define CIN         32
#define COUT        32
#define OC_PAR_FACTOR 2
```

改成保留默认值的可覆盖宏：

```cpp
#ifndef ICO_LAYER2_5_TIME_STEPS
#define ICO_LAYER2_5_TIME_STEPS 52
#endif

#ifndef ICO_LAYER2_5_CIN
#define ICO_LAYER2_5_CIN 32
#endif

#ifndef ICO_LAYER2_5_COUT
#define ICO_LAYER2_5_COUT 32
#endif

#ifndef ICO_LAYER2_5_OC_PAR_FACTOR
#define ICO_LAYER2_5_OC_PAR_FACTOR 2
#endif

#define TIME_STEPS    ICO_LAYER2_5_TIME_STEPS
#define CIN           ICO_LAYER2_5_CIN
#define COUT          ICO_LAYER2_5_COUT
#define OC_PAR_FACTOR ICO_LAYER2_5_OC_PAR_FACTOR
```

验收：

- 默认不传宏时，baseline C32/T52 native 仍通过。
- 传入 `-DICO_LAYER2_5_TIME_STEPS=6 -DICO_LAYER2_5_CIN=8 -DICO_LAYER2_5_COUT=8` 时可以编译。

### A2. 增加 C8/T6 构建和 HLS preset

建议新增两个 preset：

```bat
build_layer2_5.bat baseline
build_layer2_5.bat c8t6

run_hls.bat synth c8t6
run_hls.bat csim c8t6
```

如果不想改命令格式，也至少支持环境变量：

```bat
set ICO_HLS_CPPFLAGS=-DICO_LAYER2_5_TIME_STEPS=6 -DICO_LAYER2_5_CIN=8 -DICO_LAYER2_5_COUT=8
run_hls.bat synth
```

验收：

- C8/T6 生成独立 project/solution，避免覆盖 baseline 报告。
- `parse_hls_report.py` 能输出 C8/T6 summary。

### A3. 先用 synthetic 数据跑 native smoke

第一步不要立刻接真实 IFAN 权重，先用 deterministic synthetic input/weight/kernel/reorder 跑通：

```text
input  : [T=6, C=8, R=6, charts=5, H=2, W=4]
weight : [Cout=8, Cin=8, Rin=6, 7]
bias   : [8]
kernel : [8, 6, 8, 6, 9, 4]
output : [6, 8, 6, 5, 2, 4]
```

验收：

- g++ native 可以运行并输出 finite 结果。
- HLS `csim` 不因数组维度或宏替换失败。

### A4. 导出真实 IFAN C8_R1 ConvIco golden

在真实数据对齐时，不要把完整 fusion block 混在一起。先只对齐 ConvIco 核：

1. 从 `stage1_weights.npz` 读取：
   - `fusion_w[0..3]`
   - `fusion_b[0..3]`
   - `final_w`
   - `final_b`
   - `kernel_idx_main`
   - `reorder_r1`
2. 从 `stage1_debug_tensors.npz` 读取：
   - `fusion_feature`
   - `fusion_head_blocks`
   - `final_head_logits`
3. 为每个 ConvIco 子层导出 layer2-5 格式数据：

```text
hls_testdata/layer2-5_c8_t6/
  fusion0/
  fusion1/
  fusion2/
  fusion3/
  final/
```

注意边界：

- `fusion_head_blocks[i]` 是完整 `ConvIco -> ReLU -> temporal Conv1d -> LNorm` 后的输出，不一定能直接作为单个 ConvIco 的 golden。
- 如果当前 debug 没有 ConvIco 后、temporal 前的中间节点，需要先在 Torch 模型或导出脚本中补充 debug hook。
- 第一轮真实对齐只做 ConvIco；temporal Conv1d/LNorm 作为后续小模块单独实现。

验收：

- 每个子层都有 manifest，记录 input/golden 对应的 Torch 节点。
- C++ testbench 可选择 `fusion0..final` 任一目录运行。
- float native max error 建议先控制在 `1e-4` 量级；定点 sweep 后再放宽。

### A5. C8/T6 HLS synth 与报告

完成 native 和 csim 后，运行 C8/T6 `csynth`。

需要记录：

- target clock / estimated clock
- latency best/avg/worst
- II
- BRAM_18K / DSP / FF / LUT
- 关键 warning，尤其是 memory port、clock violation、fanout
- 与 C32/T52 baseline 的比例对比

验收：

- 生成 top-level `conv_ico_layer2_5_csynth.rpt/xml`。
- 生成 summary md，并归档到阶段报告。
- 如果 synth 失败，必须记录失败阶段和 design-size，而不是继续扩大 top。

## 5. 执行路线 B：推进 FeatureMABA 独立 PL 候选

FeatureMABA 已有真实数据和 native 回归，是下一条硬件创新线。

### B1. 保持独立 top，不并入 Stage-1 full top

当前 top 继续使用：

```text
stage1_ifan_c8_r2/feature_maba/ifan_feature_maba_top
```

输入/输出：

```text
input/output : [T=6, C=8, R=6, charts=5, H=2, W=4]
positions    : 6*5*2*4 = 240
state_dim    : 8
d_model      : 16
```

### B2. 先跑 HLS csim

使用现有 `maba/tensors` 中逐节点 golden：

```text
input_positions
in_proj_out
dw_conv_out
mix_norm_out
q / gate / alpha
state_sequence
state_back_out
delta
output
```

验收：

- top output 对 `pre_readout_refined_logits` 通过。
- 如果失败，优先按上述节点逐级定位，不直接看最终 coords。

### B3. synth 前的结构约束

MABA 的主要风险是 `[positions=240] * [T=6] * [d_model/state]` 的数组展开。synth 前建议明确：

- 以 position tile 流式推进，而不是完整展开 240 个 position。
- state scan 沿 `T` 顺序进行，保留每个 position 的小状态。
- depthwise temporal conv 可以先保守实现，确认资源后再优化。
- LayerNorm 先保留 float/fixed 兼容路径，定点化放到资源闭合之后。

验收：

- 能进入 `csynth_design` 并生成 top-level report。
- 若无法闭合，保留最小 design-size 报告，并缩小 tile/unroll。

### B4. MABA 资源闭合后再决定 post-MABA

post-MABA 包含：

- channel readout
- region max
- CleanVertices
- SoftArgMax

这些模块控制简单，但对论文创新贡献弱。建议只有在 FeatureMABA 资源可接受后，才评估是否把 post-MABA 作为 PL head；否则保留在 PS 侧更干净。

## 6. 推荐排期

### 第 1 轮：主干参数化闭环

1. 参数化 `layer2-5` 宏，保留 baseline 默认行为。
2. 增加 C8/T6 build/HLS preset。
3. 跑 baseline native，确认无回归。
4. 跑 C8/T6 synthetic native。

产物：

- 参数化代码
- C8/T6 native smoke 记录

### 第 2 轮：主干真实数据对齐

1. 补齐 Torch debug hook 或导出脚本，使 ConvIco 单层 golden 可得。
2. 导出 `fusion0..fusion3/final` 的 C8/T6 layer2-5 格式数据。
3. C++ testbench 支持选择数据目录。
4. 跑 native 数值对齐。

产物：

- `hls_testdata/layer2-5_c8_t6/*`
- native max error / RMSE 表

### 第 3 轮：主干 HLS 报告

1. 跑 C8/T6 `csim`。
2. 跑 C8/T6 `synth`。
3. 解析报告并与 C32/T52 baseline 对比。

产物：

- C8/T6 `csynth.rpt/xml`
- C32/T52 vs C8/T6 资源/延迟对比表

### 第 4 轮：FeatureMABA HLS 报告

1. 跑 FeatureMABA `csim`。
2. 跑 FeatureMABA `synth`。
3. 如果资源过高，做 tile/unroll 调整。
4. 形成 MABA 独立资源表。

产物：

- FeatureMABA `csynth.rpt/xml`
- MABA 节点级误差记录
- 是否继续 post-MABA PL 的结论

## 7. 停止条件与转向条件

继续推进条件：

- `layer2-5` C8/T6 能生成 top-level `csynth` 报告。
- FeatureMABA 至少能完成 `csim`，并能进入 HLS 后端。

需要转向/降级条件：

- C8/T6 主干虽然参数化成功，但真实 ConvIco golden 无法稳定对齐：先回到 Torch debug hook，不进入 synth。
- FeatureMABA synth 出现与完整 Stage-1 类似的长时间无 report：立即转为 position tile 或保留 PS。
- post-MABA 资源收益不明显：保持 PS 侧实现，不纳入 PL 主线。

## 8. 论文表述建议

后续论文硬件章节建议拆成三个层次：

1. 原始全量 Stage-1 纯 PL 尝试：作为路线收敛证据，说明为什么转向 PS/PL 协同。
2. C8/T6 主干 ConvIco PL kernel：作为可复用主干算子的核心硬件评估。
3. FeatureMABA PL kernel：作为本课题新增轻量时序建模模块的硬件创新评估。

避免表述：

```text
完整 IFAN C8_R2 全部在 PL 上完成了综合。
```

建议表述：

```text
本文将 IFAN C8_R2 拆分为 PS 侧特征/调度与 PL 侧可复用计算 kernel。
PL 侧重点评估 C8/T6 R1 ConvIco 主干 kernel 与 FeatureMABA temporal refiner。
```

