# IFAN IcoConv 主干轻量化与 C=8 实验计划与结果总结

## 1. 问题定位

当前 IFAN/icoCNN 主干与 Cross3D 的计算瓶颈不同。Cross3D-Edge 中采用 depthwise separable convolution 的主要对象是输出端的 `Output_Conv1`，该层来自 Cross-Conv 后的空间特征展平：

```text
Cin = 空间位置数 * feature channels * 双分支
Cout = 128 或 4C
K = 5
```

因此在 Cross3D 中会出现 `Cin > 1000, Cout > 100` 的大通道 1D 卷积，替换为 depthwise separable convolution 可以显著降低参数量和权重访存。

但 IFAN/icoCNN 已经改变了这一结构。当前网络保留二十面体空间维度 `R, charts, H, W`，时间卷积只在每个空间位置上处理 `C` 个通道：

```text
(B, T, C, R, charts, H, W)
-> rearrange
((B * R * charts * H * W), C, T)
-> Conv1d(C -> C)
```

因此，Cross3D 中的大 `Output_Conv1(Cin > 1000)` 在当前 IFAN 主干中没有直接等价层。当前主要计算量已经转移到 IcoConv、residual learning、shared attention 和 fusion block 中的二十面体邻域卷积。

## 2. 理论依据

当前 IcoConv 的核心权重可抽象为：

```text
W[Cout, Cin, Rin, 7]
```

在 forward 中会展开为：

```text
(Cout * Rout) x (Cin * Rin) x 3 x 3
```

其主要计算复杂度近似为：

```text
MAC_IcoConv ~= T * charts * H * W * Cin * Cout * Rin * Rout * Kico
```

其中 `Kico` 可按有效 7 邻域估算。对于 IFAN 主干中大量等通道 IcoConv，通常有：

```text
Cin = Cout = C
Rin = Rout = 6
```

因此主项近似随 `C^2` 缩放。将 `C=16` 缩减到 `C=8`，理论上 IcoConv 主计算量约变为：

```text
(8 / 16)^2 = 25%
```

这比单独替换 1D temporal Conv 更贴合当前 IFAN 的真实瓶颈。

## 3. 实验设置

本轮轻量化实验只改变 IFAN 主干宽度：

```text
branch_channels: 16 -> 8
```

保持以下设置不变：

- `srp_variant = paper_original`
- `temporal_conv_variant = standard_1d`
- `temporal_module = conv`
- `epochs = 80`
- `phase1_epochs = 20`
- `lr_phase1 = 1e-4`
- `lr_phase2 = 1e-5`
- `train_snr_phase1 = 30 dB`
- `train_snr_phase2 = 5~30 dB`
- `r = 2`
- `final_head_pooling = false`
- `smooth_vertices = true`

这样可以保证实验因果归因清楚：结果差异主要来自 IcoConv 主干通道数缩减，而不是前端、时序模块或训练策略变化。

## 4. 资源与结构对照

以 Stage-3 常用的 `MAC proxy` 口径 `[1, 2, 6, 5, 4, 8]` 统计：

| 配置 | 输入 shape | trainable params | backend MAC proxy | 相对 `IFAN_80` backend MAC | frontend grid points |
|---|---|---:|---:|---:|---:|
| `IFAN_80 (C=16, r=2)` | `[1, 2, 6, 5, 4, 8]` | 125,457 | 459,532,800 | baseline | 160 |
| `IFAN_C8_R2` | `[1, 2, 6, 5, 4, 8]` | 31,561 | 115,211,520 | 0.251x | 160 |
| `IFAN_C8_R3` | `[1, 2, 6, 5, 8, 16]` | 31,561 | 460,846,080 | 1.003x | 640 |

分项上，`C=8` 后主要模块的 `MAC proxy` 预期为：

| 模块 | `C=8` MAC proxy |
|---|---:|
| PHAT stem | 322,560 |
| LMS stem | 322,560 |
| PHAT residual | 30,965,760 |
| LMS residual | 30,965,760 |
| Shared attention conv1 | 15,482,880 |
| Shared attention conv2 | 15,482,880 |
| Fusion block IcoConv | 15,482,880 |
| Fusion block temporal | 1,843,200 |
| Final head IcoConv | 3,870,720 |
| Final head temporal | 460,800 |
| Channel readout | 11,520 |

补充说明：

- `profile_stage2_model.py` 的快速工程检查使用短序列 `T=3`，因此其直接打印的 `MAC proxy total = 57,605,760`。
- 按 Stage-3 summary 的 `T=6` 口径翻倍后，对应 `115,211,520`。

## 5. 结果锚点

当前文档统一使用以下事实来源：

- `IFAN_80` 的验证集、模拟四场景、参数量与 `MAC`：
  - `IFAN_Edge/outputs/stage3/logs/long80_freqblock_paper_original_20260426_155329.log`
  - 以上指标取自该日志中的 `stage3_complete` 事件
- `IFAN_C8_R2`：
  - `IFAN_Edge/outputs/stage3/ifan_stage3_long80_c8_paper_original_20260505_222115/summary.json`
- `IFAN_C8_R3`：
  - `IFAN_Edge/outputs/stage3/ifan_stage3_long80_c8_r3_paper_original_20260506_220735/summary.json`
- LOCATA 上的统一平均值比较：
  - `IFAN_Edge/outputs/stage3/analysis/locata_four_model_compare.md`

其中，轻量化主线的平均 LOCATA 判定统一以 `locata_four_model_compare.md` 为准，用来保证 `baseline / IFAN_80 / IFAN_C8_R2 / IFAN_C8_R3` 处于同一张对比表内。

## 6. 实际结果

### 6.1 验证集与模拟场景

| 配置 | best validation RMSAE | final validation RMSAE | four-scene IFAN mean | four-scene delta vs baseline | hard-scene IFAN mean | hard-scene delta vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| `IFAN_80` | 7.1806 | 7.3641 | 7.8608 | +0.0539 | 9.2202 | -0.2488 |
| `IFAN_C8_R2` | 7.9372 | 7.9851 | 9.9787 | +1.5095 | 14.2679 | +2.8102 |
| `IFAN_C8_R3` | 6.6482 | 6.6937 | 8.2471 | +0.5567 | 11.8861 | +1.5956 |

### 6.2 LOCATA 统一对比口径

以下平均值全部来自 `IFAN_Edge/outputs/stage3/analysis/locata_four_model_compare.md`。

| 比较项 | 参数量变化 | `MAC` 变化 | with silences average delta | without silences average delta | 当前解释 |
|---|---:|---:|---:|---:|---|
| `IFAN_80` vs `baseline` | `290017 -> 125457` | `n/a -> 459532800` | `-1.3310 deg` | `-0.9283 deg` | 当前最强 accuracy-oriented reference |
| `IFAN_C8_R2` vs `baseline` | `290017 -> 31561` | `n/a -> 115211520` | `-0.7136 deg` | `-0.1221 deg` | 激进压缩后，LOCATA 平均仍优于 baseline |
| `IFAN_C8_R2` vs `IFAN_80` | `125457 -> 31561` | `459532800 -> 115211520` | `+0.6174 deg` | `+0.8062 deg` | 约 `75%` 资源压缩换来可接受的平均精度损失 |
| `IFAN_C8_R3` vs `IFAN_80` | `125457 -> 31561` | `459532800 -> 460846080` | `+1.3464 deg` | `+1.3132 deg` | `MAC` 基本不降，LOCATA 退化更明显 |

### 6.3 当前定位

- `IFAN_80`：当前复现主线、最佳精度主线、论文 gap 解释主线。
- `IFAN_C8_R2`：当前主轻量化结果、主边缘结果、默认硬件映射网络候选。
- `IFAN_C8_R3`：固定保留为失败参考，不再作为候选主线。

## 7. 当前结论

### 7.1 关于 `IFAN_C8_R2`

- `IFAN_C8_R2` 不应表述成“提出新的通道裁剪算法”。
- `IFAN_C8_R2` 可以明确表述为：
  - 面向 `IcoConv` 主瓶颈的结构化轻量化分析；
  - 资源-精度折中设计；
  - 面向 FPGA/IP 映射的默认网络候选。
- 在当前 LOCATA 统一比较口径下，`IFAN_C8_R2` 仍优于 `baseline`，因此它可以升格为 **IFAN-Edge 轻量化主线**。

### 7.2 关于 `IFAN_C8_R3`

- `IFAN_C8_R3` 在验证集与模拟场景上并非最差，但它没有形成有意义的 edge trade-off。
- 其主要问题是：
  - `MAC` 与 `IFAN_80` 基本持平；
  - LOCATA 平均值明显弱于 `IFAN_C8_R2`；
  - 在统一 LOCATA 对比口径下不再优于 baseline。
- 因此，`IFAN_C8_R3` 只保留为失败参考，不继续扩展。

## 8. 论文表述建议

不建议写成：

```text
提出一种新的通道裁剪算法。
```

更合适的表述是：

```text
针对 IFAN/icoCNN 主干中 IcoConv 计算占比高、复杂度随通道宽度近似二次增长的问题，设计并验证了一种保持二十面体拓扑和时序建模结构不变的宽度裁剪方案。实验以 IFAN_80 为 accuracy-oriented reference，以 IFAN_C8_R2 为 edge-oriented reference，在 LOCATA 上分析通道宽度对参数量、MAC、真实数据定位精度和硬件映射成本的影响。
```

该贡献属于：

- 面向特定网络主瓶颈的结构化轻量化分析；
- 面向 FPGA/IP 映射前的架构级资源压缩；
- 为后续球面 `ConvIco` 硬件映射提供默认网络候选。

## 9. 与硬件映射的关系

`IFAN_C8_R2` 当前成为默认硬件候选，不是因为它在所有指标上都最强，而是因为它同时满足：

- 参数量和 `MAC` 相对 `IFAN_80` 均下降约 `74.8% ~ 74.9%`；
- 在统一 LOCATA 对比口径下，相对 `baseline` 仍保持平均优势；
- 比 `IFAN_C8_R3` 更有意义，因为它保住了 `MAC` 优势。

因此，后续 FPGA 方向默认围绕以下网络候选展开：

- `IFAN_80`：精度参考网络
- `IFAN_C8_R2`：边缘实现参考网络

## 10. 后续备选

当前默认任务不再新增训练实验。若后续答辩反馈必须补强算法新意，再按以下顺序考虑：

1. `C=12` 中间宽度实验
2. Bottleneck IcoConv
3. Separable IcoConv
4. Orientation 压缩

这些方向当前只保留为后续备选，不作为本轮默认执行项。
