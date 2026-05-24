# IFAN-Edge 前期工作总结与结果呈现

## 1. 研究目标与问题背景

本课题围绕多通道麦克风阵列声源定位任务展开，目标是在已有 icoCNN baseline 的基础上复现 IFAN 论文主线，并进一步探索适合边缘部署的轻量化网络候选。整体工作不再以开放式扩展多个模型分支为主，而是收束为两条明确主线：

- `IFAN_80`：作为论文复现、精度参考和论文 gap 解释主线。
- `IFAN_C8_R2`：作为 IFAN-Edge 轻量化、边缘部署和后续硬件映射候选主线。

当前阶段的核心成果是：已经建立从 `PHAT + LMS` 双特征前端、IFAN 主干重构、训练闭环、LOCATA 统一验收到轻量化比较的完整工程链路。后续 HLS/FPGA 工作保留为并列创新方向，但不作为本阶段 IFAN-Edge 算法主线的已完成核心成果。

## 2. 前期计划完成情况对照

| 阶段 | 原始目标 | 当前状态 | 说明 |
| --- | --- | --- | --- |
| Stage 1 | SRP-LMS 特征生成，复用二十面体网格，形成 PHAT + LMS 双特征 | 已完成 | 已实现 `SRPPHATIcoMapAdapter`、`SRPLMSIcoMap`、`DualFeatureIcoPreprocessor`，并导出四场景可视化。 |
| Stage 2 | 在 IcoCNN 基础上增量加入 IFAN 结构模块 | 已完成 | 已完成双分支、残差学习、共享注意力、融合 head 和前向/反向工程检查。 |
| Stage 3 | 建立 IFAN 训练与收敛验证 | 已完成主闭环 | 已完成训练入口、checkpoint、history、baseline compare 和长训练结果整理。 |
| Stage 4 | LOCATA 单源任务验收 | 已完成本地统一口径验收 | 当前口径为 `benchmark2 / eval / task1,3,5 / total=23 recordings`。 |
| Stage 5 | IFAN-Edge 轻量化探索 | 已形成主结果 | `IFAN_C8_R2` 成为默认轻量化主线，`IFAN_C8_R3` 固定为失败参考。 |
| HLS/FPGA | 网络硬件映射与资源受限优化 | 后续衔接方向 | 已有 layer0/layer1/layer2-5 HLS 基础，但不在当前算法主线中过度表述。 |

## 3. 已完成工程链路

### 3.1 Stage 1：双特征前端

当前前端已经从单一 SRP-PHAT 扩展为 `PHAT + LMS` 双特征输入，并统一映射到二十面体网格。张量约定为：

- `channel 0 = PHAT`
- `channel 1 = LMS`

已有可视化产物位于：

- `IFAN_Edge/outputs/stage1_features/scene_1/feature_maps_projection_contrast.png`
- `IFAN_Edge/outputs/stage1_features/scene_2/feature_maps_projection_contrast.png`
- `IFAN_Edge/outputs/stage1_features/scene_3/feature_maps_projection_contrast.png`
- `IFAN_Edge/outputs/stage1_features/scene_4/feature_maps_projection_contrast.png`

这些图片可用于中期答辩展示前端链路已经能在二十面体空间上形成可观察的响应。需要注意的是，可视化只证明前端工程链路和响应形态，不应直接替代最终精度结论。

### 3.2 Stage 2：IFAN 主干重构

当前 IFAN 主线已经从早期旧简化主干切换为论文理解版结构，核心模块包括：

- PHAT / LMS 双输入分支
- residual learning module
- shared attention weight module
- branch-local fusion
- second-stage feature fusion
- 深层 fusion head
- `CleanVertices -> SoftArgMax` 输出

结构对比结论来自 `IFAN_Edge/docs/stage_03_architecture_compare.md`。当前更稳妥的表述是：代码主线已经符合现阶段采用的论文 IFAN 结构理解，但仍存在少量图示歧义和训练效果差距需要继续解释。

### 3.3 Stage 3：训练闭环与模拟实验

训练主线已完成以下闭环：

- 两阶段训练流程
- 笛卡尔坐标 MSE 损失
- Adam 优化器
- 前期高 SNR、后期混合 SNR
- fixed validation cache
- checkpoint、history、baseline compare 输出

当前模拟场景层面的结论是：完整重构后的 IFAN 主线已经接近 icoCNN baseline，不再是旧简化网络显著落后的状态。已有结果中，`IFAN_80` 的四场景平均差距为 `+0.0539 deg`，高混响低 SNR 场景平均差距为 `-0.2488 deg`。这说明主干方向已基本稳定，后续重点应转向论文 gap 解释与真实数据集验收，而不是回退到旧简化主干。

### 3.4 Stage 4：LOCATA 统一验收

当前 LOCATA 统一评测口径为：

- subset：`eval`
- array：`benchmark2`
- tasks：`task1, task3, task5`
- available recordings：`task1=13, task3=5, task5=5, total=23`
- 指标：recording-level RMSAE
- 统计方式：with silences 与 without silences

跨模型平均值比较统一以 `IFAN_Edge/outputs/stage3/analysis/locata_four_model_compare.md` 为准，避免不同单模型评测报告的 baseline 口径不一致。

### 3.5 Stage 5：IFAN-Edge 轻量化主线

当前轻量化工作围绕 IcoConv 主瓶颈展开。由于 IcoConv 的参数量和 MAC 随通道宽度近似二次变化，降低主干宽度能够直接影响模型规模和计算量。

最终收束为：

- `IFAN_80`：复现主线、精度主线、论文 gap 解释主线。
- `IFAN_C8_R2`：轻量化主线、边缘主线、默认硬件映射候选。
- `IFAN_C8_R3`：失败参考，不继续扩展。

需要注意：`IFAN_C8_R2` 不应包装成独立的通道裁剪理论，而应表述为“面向 IcoConv 主瓶颈的结构化轻量化与边缘折中设计”。

## 4. 关键实验结果

### 4.1 LOCATA 三模型核心结果

| Model | Params | MAC | With Silences Avg | Without Silences Avg |
| --- | ---: | ---: | ---: | ---: |
| baseline | 290017 | - | 8.5718 | 7.1976 |
| IFAN_80 | 125457 | 459532800 | 7.2407 | 6.2693 |
| IFAN_C8_R2 | 31561 | 115211520 | 7.8581 | 7.0755 |

从 LOCATA 统一口径看，`IFAN_80` 是当前核心模型中最强的精度参考，with silences average 相对 baseline 改善 `1.3310 deg`，without silences average 改善 `0.9283 deg`。这说明当前 IFAN 复现主线已经在真实数据集上形成有效优势。

`IFAN_C8_R2` 在大幅压缩后仍保持对 baseline 的平均优势：with silences average 改善 `0.7136 deg`，without silences average 改善 `0.1221 deg`。因此它可以作为 IFAN-Edge 当前阶段的主要轻量化结果。

### 4.2 资源与精度折中

| Comparison | Params Change | MAC Change | With Silences Avg Delta | Without Silences Avg Delta | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| IFAN_80 vs baseline | 56.7% reduction | n/a | -1.3310 deg | -0.9283 deg | 精度参考主线，真实数据集平均优于 baseline。 |
| IFAN_C8_R2 vs baseline | 89.1% reduction | n/a | -0.7136 deg | -0.1221 deg | 激进压缩后仍保持 LOCATA 平均优势。 |
| IFAN_C8_R2 vs IFAN_80 | 74.8% reduction | 74.9% reduction | +0.6174 deg | +0.8062 deg | 约 75% 参数与 MAC 压缩换取可接受平均精度损失。 |

`IFAN_C8_R2` 的意义不在于所有场景都优于 `IFAN_80`，而在于它在保留 LOCATA 平均优势的同时，大幅降低了模型参数量和 MAC。这使它比 `IFAN_80` 更适合作为后续硬件映射候选。

### 4.3 C8_R3 失败参考

`IFAN_C8_R3` 与 `IFAN_C8_R2` 参数量相同，但由于输入网格分辨率变化，MAC 基本没有下降。统一 LOCATA 对比显示，它的平均退化比 `C8_R2` 更明显，且失去了关键计算优势。因此当前将 `IFAN_C8_R3` 固定为失败参考，不再作为候选主线。

## 5. 当前结论

当前已经可以明确写入中期报告的结论包括：

- 已完成 `PHAT + LMS` 双特征前端的工程链路和可视化验证。
- 已完成 IFAN 主干的论文理解版重构，当前不再应描述为旧简化主干。
- 已建立训练、评估、baseline compare 和 LOCATA 验收闭环。
- `IFAN_80` 是当前复现主线和精度参考主线。
- `IFAN_C8_R2` 是当前 IFAN-Edge 轻量化主线和默认硬件映射候选。
- `IFAN_C8_R2` 在 LOCATA 统一口径下仍优于 baseline，并相对 `IFAN_80` 降低约 `74.8%` 参数和 `74.9%` MAC。
- `IFAN_C8_R3` 固定为失败参考，不继续扩展。

## 6. 限制与风险

当前仍不能过度表述的部分包括：

- 不能宣称论文最终效果已经完全复现。
- 不能把 `IFAN_C8_R2` 包装成独立的通道裁剪理论或网络结构理论创新。
- LOCATA 结果仍受本地可运行 recording 口径约束，即 `benchmark2 / eval / task1,3,5 / total=23 recordings`。
- HLS/FPGA 尚未完成 IFAN_C8_R2 整网资源闭合，当前只能作为后续衔接方向。
- 论文中 `final_head_pooling` 位置、`32 kernels` 含义、LMS backend 等细节仍需在后续论文 gap 解释中继续说明。

## 7. 后续计划

后续工作建议按照以下顺序推进：

1. 固化中期报告和答辩 PPT 中的模型口径：`IFAN_80 = 复现主线`，`IFAN_C8_R2 = 边缘主线`。
2. 继续围绕 `IFAN_80` 解释与论文最好结果之间的剩余 gap，重点检查训练预算、评测口径、LMS backend 和结构图示歧义。
3. 围绕 `IFAN_C8_R2` 补充更清晰的硬件映射预算，将参数量、MAC、HLS 资源和数据流设计连接起来。
4. 将当前 Stage 1/2/3/LOCATA/轻量化结果整理进论文实验章节，形成可复查的数据来源链。

## 8. 中期答辩可用图表清单

| 图表 | 建议用途 | 文件或来源 |
| --- | --- | --- |
| 双特征前端流程图 | 展示 PHAT + LMS 输入链路 | 可手绘或用 PPT 画图 |
| 四场景特征投影图 | 展示 Stage 1 可视化结果 | `IFAN_Edge/outputs/stage1_features/scene_*/feature_maps_projection_contrast.png` |
| IFAN 与 icoCNN 结构对比表 | 展示主干重构差异 | `IFAN_Edge/docs/stage_03_architecture_compare.md` |
| LOCATA 三模型结果表 | 展示核心实验结果 | `IFAN_Edge/outputs/stage3/analysis/locata_four_model_compare.md` |
| 参数量 / MAC 压缩图 | 展示 IFAN_C8_R2 edge trade-off | 根据本文第 4.2 节绘制 |
| HLS/FPGA 后续数据流图 | 展示后续硬件衔接方向 | 可参考已有 layer2-5 HLS 架构文档 |
