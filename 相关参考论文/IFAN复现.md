# IFAN 复现总纲与当前结论

## 1. 文档定位

本文档作为当前唯一复现总纲，统一回答 5 个问题：

1. IFAN 原论文到底要求复现什么。
2. 当前工程已经完成到哪一步。
3. 当前准确率主线应以哪条结果为准。
4. 当前轻量化/边缘主线应以哪条结果为准。
5. 接下来资源应放在论文 gap 解释、边缘收束和 FPGA 创新中的什么位置。

本文档中的信息固定分为 3 类：

- `论文事实`：能直接从论文正文、图表或公式确认的信息
- `当前实现`：当前仓库和输出目录中已经落地的事实
- `当前结论`：基于当前有效结果锚点给出的阶段判断

当前总路线固定为：

`icoCNN baseline -> IFAN 论文主线复现 -> LOCATA 验收闭环 -> IFAN_C8_R2 边缘主线 -> FPGA 并列创新`

## 2. 当前固定口径

### 2.1 双主线定义

当前阶段统一采用双主线，不再争论单一默认主线：

- `IFAN_80`：
  - 复现主线
  - 最佳精度主线
  - 论文 gap 解释主线
- `IFAN_C8_R2`：
  - IFAN-Edge 轻量化主线
  - 边缘部署主线
  - 默认硬件映射网络候选

### 2.2 其他结果线的当前定位

- `IFAN_Maba`：保留为候选增强线，不进入当前默认主线
- `IFAN_C8_R3`：固定保留为失败参考，不再作为候选主线
- `IFAN_LC`：作为前端方法探索线保留，不进入当前默认主线

### 2.3 LOCATA 与模拟四场景的角色分工

- 模拟四场景仍然保留：
  - 用于复现链证据
  - 用于论文口径核对
- LOCATA 在当前阶段承担更高优先级：
  - `IFAN_80` 的论文级可用性判断看 LOCATA
  - `IFAN_C8_R2` 的 edge 主线判定也主看 LOCATA
- `IFAN_C8_R2` 不再因模拟四场景回退而被自动否决

## 3. 论文事实提炼

### 3.1 输入、数据与训练

- `论文事实`：IFAN 输入由 `icosahedral SRP-PHAT maps + icosahedral SRP-LMS maps` 组成。
- `论文事实`：训练数据来源于 `LibriSpeech train-clean-100`。
- `论文事实`：模拟测试数据来源于 `LibriSpeech test-clean`。
- `论文事实`：采样率为 `16 kHz`。
- `论文事实`：帧长 `K = 4096`，帧移 `step = 3072`。
- `论文事实`：随机轨迹时长为 `20 s`。
- `论文事实`：训练共 `80 epoch`。
- `论文事实`：前 `20 epoch` 采用 `SNR = 30 dB`、`batch size = 1`、`lr = 1e-4`。
- `论文事实`：后 `60 epoch` 采用 `SNR = 5~30 dB`、`batch size = 10`、`lr = 1e-5`。
- `论文事实`：训练时 `T60` 在 `0.2~1.3 s` 范围内随机采样。
- `论文事实`：损失函数是笛卡尔坐标 `(x, y, z)` 上的 MSE。
- `论文事实`：优化器是 Adam。
- `论文事实`：训练时使用 VAD 过滤静音段。

### 3.2 评测口径

- `论文事实`：模拟实验使用 RMSAE。
- `论文事实`：模拟评测忽略前 `5` 帧。
- `论文事实`：LOCATA 只评测单源 `Task 1 / Task 3 / Task 5`。
- `论文事实`：论文强调测试口径与所比较方法保持一致。

### 3.3 结构事实

- `论文事实`：`Fig.1` 展示的是特征图示例，不是四场景跟踪图。
- `论文事实`：`Fig.10` 才是四组模拟跟踪场景。
- `论文事实`：论文张量口径写作 `B x T x C x R x 5 x H x W`。
- `论文事实`：输入通道初值为 `1`，分支通道扩展到 `16`。
- `论文事实`：网络核心模块包括：
  - 特征提取
  - 特征残差学习
  - 特征注意力权重学习
  - 融合特征学习
- `论文事实`：融合后的空间分辨率等价于 `r=1`，即 `5 x 2 x 4`。
- `论文事实`：融合头包含 `32` 个 icosahedral kernels 和 `1-D time convolution`。

## 4. 当前工程状态

### 4.1 Stage 1

- `当前实现`：已完成 `SRPPHATIcoMapAdapter`、`SRPLMSIcoMap`、`DualFeatureIcoPreprocessor`。
- `当前结论`：双特征前端工程链路稳定可用。

### 4.2 Stage 2

- `当前实现`：已完成 `IFANModelConfig` 与 `IFANModel` 的论文主线骨架重构。
- `当前结论`：当前 stage-2 已不再处于“是否还是旧简化主干”的状态。

### 4.3 Stage 3

- `当前实现`：已完成训练闭环、固定 validation cache、四场景 matched baseline compare。
- `当前实现`：已完成 `paper_original` LMS 行为口径与 `frequency_block` backend。
- `当前实现`：已完成 `IFAN_80`、`IFAN_C8_R2`、`IFAN_C8_R3` 与 LOCATA 四模型统一比较。

## 5. 当前双主线结果

### 5.1 `IFAN_80`：复现主线 / 精度主线

事实锚点：

- 模拟与验证集：
  - `IFAN_Edge/outputs/stage3/logs/long80_freqblock_paper_original_20260426_155329.log`
  - 指标取自该日志中的 `stage3_complete` 事件
- LOCATA 统一对比：
  - `IFAN_Edge/outputs/stage3/analysis/locata_four_model_compare.md`

当前指标：

- `best_val_rmsae_deg = 7.1806`
- four-scene mean delta vs baseline `= +0.0539 deg`
- hard-scene mean delta vs baseline `= -0.2488 deg`
- LOCATA `with silences` average `= 7.2407 deg`
- LOCATA `without silences` average `= 6.2693 deg`
- 相对 LOCATA baseline 的平均提升：
  - `with silences = -1.3310 deg`
  - `without silences = -0.9283 deg`
- 资源口径：
  - `trainable params = 125,457`
  - `MAC proxy = 459,532,800`

当前定位：

- 它是当前最强的 accuracy-oriented reference。
- 它负责承担“IFAN 是否已经站住论文主线复现”的主要证据。
- 论文剩余 gap 仍然围绕这条线解释，而不是围绕 `C8` 解释。

### 5.2 `IFAN_C8_R2`：轻量化主线 / 边缘主线

事实锚点：

- `IFAN_Edge/outputs/stage3/ifan_stage3_long80_c8_paper_original_20260505_222115/summary.json`
- `IFAN_Edge/outputs/stage3/analysis/locata_four_model_compare.md`

当前指标：

- `best_val_rmsae_deg = 7.9372`
- `trainable params = 31,561`
- `MAC proxy = 115,211,520`
- 相对 `IFAN_80`：
  - 参数下降 `74.8%`
  - `MAC` 下降 `74.9%`
- LOCATA `with silences` average `= 7.8581 deg`
- LOCATA `without silences` average `= 7.0755 deg`
- 相对 LOCATA baseline 的平均提升：
  - `with silences = -0.7136 deg`
  - `without silences = -0.1221 deg`
- 相对 `IFAN_80` 的平均损失：
  - `with silences = +0.6174 deg`
  - `without silences = +0.8062 deg`

当前定位：

- 它不是新的算法理论主干。
- 它是当前最有意义的 edge point。
- 它已经足以承担 **IFAN-Edge 边缘主线**，并作为默认硬件映射网络候选。

### 5.3 失败参考与候选增强线

`IFAN_C8_R3`

- 相对 `IFAN_80` 参数下降同样是 `74.8%`，但 `MAC` 基本不降。
- 在统一 LOCATA 对比表中，平均值明显弱于 `IFAN_C8_R2`，因此固定保留为失败参考。

`IFAN_Maba`

- 仍明显优于本地 baseline。
- 但现阶段没有形成足够稳定、足够全面的整体替代优势。
- 继续保留为候选增强线，不升级为默认主线。

## 6. 当前结论

### 6.1 已经可以明确写死的结论

- `当前结论`：当前 IFAN 主线已经是完整重构后的正确主干，不再应被描述成“可能仍是简化主干”。
- `当前结论`：`IFAN_80` 是当前复现主线与最佳精度主线。
- `当前结论`：`IFAN_C8_R2` 是当前轻量化主线与边缘主线。
- `当前结论`：`IFAN_C8_R2` 不应表述为“新的通道裁剪算法”，而应表述为“面向 IcoConv 主瓶颈的结构化轻量化与边缘折中设计”。
- `当前结论`：`IFAN_C8_R3` 已经固定为失败参考，不再作为候选主线。

### 6.2 仍不能过度表述的部分

- `当前结论`：不能表述为“已完全复现论文最终效果”。
- `当前结论`：不能把 `IFAN_C8_R2` 直接写成“新的网络结构理论创新”。
- `当前结论`：当前 LOCATA 结论仍受本地可运行 recording 口径约束。

## 7. 当前剩余差距

当前剩余问题已经从“主干正确性”转移到以下三个方向：

1. `IFAN_80` 与论文 IFAN 的剩余 gap 到底来自训练预算、评测口径还是 backend 细节
2. 如何把 `IFAN_C8_R2` 组织成一个更站得住的“边缘主线贡献”，而不是单个 ablation
3. 如何把现有 HLS 结果与 `IFAN_C8_R2` 对接，形成并列硬件创新

## 8. 当前策略

### 8.1 默认不新增训练实验

当前默认任务不再新增算法训练实验。

若后续答辩反馈必须补强算法新意，再考虑：

- `C=12`
- Bottleneck IcoConv
- Separable IcoConv

这些方向当前只保留为后续备选，不属于本轮默认执行项。

### 8.2 当前文档与结果同步优先级

1. 先同步 `IFAN_IcoConv主干轻量化与C8实验计划.md`
2. 再同步本文件 `IFAN复现.md`
3. 最后同步 `IFAN复现总体大纲.md`

### 8.3 当前硬件方向的角色

硬件方向不再被表述成“以后有空再接”。

当前更合适的定位是：

- 与算法线并列的第二创新点
- 重点是面向球面 `ConvIco` 的网络适配型 FPGA 架构
- 默认网络候选是 `IFAN_C8_R2`
- 精度参考网络是 `IFAN_80`

## 9. LOCATA 口径边界

当前本地 `benchmark2 / eval / Task 1, 3, 5` 实际跑通并用于汇总的 recording 数是：

- `task1 = 13`
- `task3 = 5`
- `task5 = 5`
- `total = 23`

因此，当前总纲中关于 LOCATA 的所有判断，都建立在这套本地可运行口径上。

需要特别固定两点：

- 单模型 LOCATA 报告可作为补充证据保留。
- 跨模型平均值比较统一以 `IFAN_Edge/outputs/stage3/analysis/locata_four_model_compare.md` 为准，以保证 baseline 口径一致。

## 10. 相关文档

- 轻量化结果总结：
  - `相关参考论文/IFAN_IcoConv主干轻量化与C8实验计划.md`
- 当前阶段总体大纲：
  - `相关参考论文/IFAN复现总体大纲.md`
- 四模型 LOCATA 对比：
  - `IFAN_Edge/outputs/stage3/analysis/locata_four_model_compare.md`
