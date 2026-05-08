# IFAN IcoConv 主干轻量化与 C=8 实验计划

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

## 3. C=8 宽度裁剪实验设计

本实验只改变 IFAN 主干宽度：

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

## 4. 资源预期

以 Stage-3 summary 常用的 MAC proxy 输入口径 `[1, 2, 6, 5, 4, 8]` 估算：

| 配置 | 参数量 | MAC proxy | 参数下降 | MAC 下降 |
|---|---:|---:|---:|---:|
| IFAN C=16 | 125,457 | 459,532,800 | - | - |
| IFAN C=8 | 31,561 | 115,211,520 | 74.8% | 74.9% |

分项上，C=8 后主要模块的 MAC proxy 预期为：

| 模块 | C=8 MAC proxy |
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

补充说明：`profile_stage2_model.py` 的快速工程检查使用短序列 `T=3`，因此其直接打印的 `MAC proxy total = 57,605,760`；按 Stage-3 summary 的 `T=6` 口径翻倍后为 `115,211,520`。

## 4.1 宽度-分辨率折中假设

仅做 `C=8` 宽度裁剪虽然最直接，但它会同时降低参数量和空间建模容量。作为第二阶段候选方案，值得把“降低 `C` 的同时提高输入分辨率 `r`”纳入同一张资源表中，单独验证是否存在更好的精度/复杂度平衡点。

这里先统一两类口径：

- `backend MAC proxy`：只统计 IFAN 主干网络，不包含 PHAT/LMS 前端。
- `frontend grid points`：只反映 SRP 特征图候选点数量，可近似理解为前端复杂度的一级指标。

在当前实现中，`r=3` 会让输入 chart 从 `4 x 8` 变为 `8 x 16`；同时 fusion 前仍有一次 `PoolIco`，因此 fusion head 实际工作分辨率会从 `r=1` 提升到 `r=2`。这意味着 `r=3` 不是“只把输入做大”，而是确实会把更多空间细节送入后续融合模块。

按 Stage-3 summary 常用的 `T=6` 口径估算，得到下表：

| 配置 | 输入 shape | trainable params | backend MAC proxy | 相对 C=16,r=2 backend MAC | frontend grid points | frontend 复杂度粗估 | 预期精度风险 |
|---|---|---:|---:|---:|---:|---:|---|
| IFAN C=16, r=2 | `[1, 2, 6, 5, 4, 8]` | 125,457 | 459,532,800 | baseline | 160 | 1.0x | 当前主线基线 |
| IFAN C=8, r=2 | `[1, 2, 6, 5, 4, 8]` | 31,561 | 115,211,520 | 0.251x | 160 | 1.0x | 宽度压缩，存在明显精度回退风险 |
| IFAN C=8, r=3 | `[1, 2, 6, 5, 8, 16]` | 31,561 | 460,846,080 | 1.003x | 640 | 4.0x | 有机会补偿部分 `C` 损失，但不能预设无损 |

从表中可以直接看到：

- `C=8, r=3` 的 `backend MAC proxy` 与当前 `C=16, r=2` 主线几乎持平，仅高约 `0.29%`。
- `C=8, r=3` 的参数量仍保持 `31,561`，相对主线下降 `74.8%`。
- 但 `frontend grid points` 会从 `160` 升到 `640`，因此前端 SRP 复杂度、缓存和端到端时延不会像 backend 那样“基本持平”。

因此，`C=8, r=3` 的意义不是“免费提高分辨率”，而是：

- 用几乎不增加 backend CNN 计算量的方式，换取更高的空间采样密度；
- 检查空间分辨率提升能否补偿通道宽度裁剪带来的精度损失；
- 为后续 `IFAN-Edge-M` 级别方案提供一个比 `C=8, r=2` 更有希望的折中点。

需要特别强调：这只是资源与结构层面的合理假设，不是已验证结论。当前仓库里还没有完成 `C=8, r=3` 的训练与 LOCATA 验收，因此不能先验地把它表述成“无精度损失方案”。

## 5. 对照组与评价指标

固定 C=16 对照组：

```text
IFAN_Edge/outputs/stage3/ifan_stage3_long80_freqblock_paper_original_20260426_155330/summary.json
```

当前对照指标：

| 指标 | C=16 long80 |
|---|---:|
| best validation RMSAE | 7.1806 deg |
| four-scenario mean RMSAE | 7.8608 deg |
| hard-scenario mean RMSAE | 9.2202 deg |
| trainable params | 125,457 |
| MAC proxy | 459,532,800 |

C=8 训练完成后回填：

| 指标 | C=8 long80 | 相对 C=16 差值 |
|---|---:|---:|
| best validation RMSAE | TBD | TBD |
| four-scenario mean RMSAE | TBD | TBD |
| hard-scenario mean RMSAE | TBD | TBD |
| trainable params | 31,561 | -74.8% |
| MAC proxy | 115,211,520 | -74.9% |

判定口径：

- `RMSAE delta <= 0.3 deg`：可作为主轻量化方案。
- `0.3 deg < RMSAE delta <= 1.0 deg`：可作为资源/精度折中方案。
- `RMSAE delta > 1.0 deg`：只作为消融结果，不宜作为主方案。

如果追加 `C=8, r=3` 试验，建议把它作为独立对照组加入相同判定口径，并额外记录：

- SRP 前端耗时；
- 单 batch 推理耗时；
- 端到端总时延；
- LOCATA Task1/3/5 是否仍保持稳定。

## 6. 执行命令

训练环境：

```text
/home/cmm/miniconda3/envs/icocnn/bin/python
```

已确认：

```text
torch = 2.3.1+cu121
cuda_available = True
```

后台训练命令：

```bash
TS=$(date +%Y%m%d_%H%M%S)
nohup /home/cmm/miniconda3/envs/icocnn/bin/python IFAN_Edge/scripts/train_stage3_ifan.py \
  --config IFAN_Edge/configs/stage3_long_budget.toml \
  --branch-channels 8 \
  --output-suffix long80_c8_paper_original \
  --experiment-role c8_width_ablation \
  --srp-variant paper_original \
  --temporal-conv-variant standard_1d \
  --temporal-module conv \
  --device cuda \
  > IFAN_Edge/outputs/stage3/logs/long80_c8_paper_original_${TS}.log 2>&1 &
echo $!
```

训练前快速检查命令：

```bash
/home/cmm/miniconda3/envs/icocnn/bin/python IFAN_Edge/scripts/profile_stage2_model.py \
  --branch-channels 8 \
  --temporal-conv-variant standard_1d
```

已完成快速检查：

```text
IFANModel C=8 trainable params = 31,561
finite_output = True
finite_gradients = True
nonzero_gradient_params = 54
```

## 7. 论文表述建议

不建议把该实验表述为：

```text
提出一种新的通道裁剪算法。
```

更合适的表述是：

```text
针对 IFAN/icoCNN 主干中 IcoConv 计算占比高、复杂度随通道宽度近似二次增长的问题，设计并验证了一种保持二十面体拓扑和时序建模结构不变的宽度裁剪方案。实验通过 C=16 与 C=8 的严格对照，分析通道宽度对参数量、MAC、定位精度和硬件映射成本的影响。
```

该贡献属于：

- 面向特定网络主瓶颈的结构化轻量化分析；
- 面向 FPGA/IP 映射前的架构级资源压缩；
- 后续 IcoConv 专用轻量化设计的基线实验。

## 8. 第二阶段可扩展方向

如果 C=8 精度损失较小，可以直接进入硬件映射评估。

如果 C=8 精度损失明显，则建议按以下顺序推进：

1. `C=12` 中间宽度实验：验证是否存在更合适的资源/精度折中点。
2. Bottleneck IcoConv：在高开销 IcoConv 中引入 `16 -> b -> 16` 的低秩通道瓶颈。
3. Separable IcoConv：将二十面体邻域 filtering 与 channel/orientation mixing 分离。
4. Orientation 压缩：研究 `R=6` 方向通道是否存在冗余，评估 `R=6 -> R'=3/1` 的可行性。

其中第 2~4 项比 1D temporal DWConv 更贴合当前 IFAN 的真实计算瓶颈。
