# IFAN 复现总纲与当前结论

## 1. 文档定位

本文档作为当前唯一总纲，统一回答 4 个问题：

1. IFAN 原论文到底要求复现什么。
2. 当前工程已经完成到哪一步。
3. 当前结果与 baseline、与论文最好结果还差多少。
4. 接下来应该把资源放在什么地方，而不再回退到旧无效主线。

本文档中的信息固定分为 3 类：

- `论文事实`：能直接从论文正文、图表或公式确认的信息
- `当前实现`：当前仓库和输出目录中已经落地的事实
- `当前结论`：基于当前唯一有效主结果给出的阶段判断

当前总路线固定为：

`icoCNN baseline -> IFAN 论文主线复现 -> 论文剩余差距分析 -> LOCATA 验收 -> 再考虑轻量化迁移`

当前唯一正确结论也固定为：

`完整重构后的 IFAN 已经接近当前工程中的 icoCNN r=2 baseline，但与论文最好结果仍有差距；旧简化主干实验不再计入 IFAN 复现结论。`

## 2. 有效结果边界

从本文件开始，统一采用以下保留标准：

- 只有“完整重构后的 IFAN 主线”产生的实验结果才计入 IFAN 复现结论。
- 任何“旧简化主干 / 重构前主干 / 未完整复现架构”阶段的实验，只保留一句历史说明，不再保留指标表或结论段。

当前唯一有效主结果来源：

- `IFAN_Edge/outputs/stage3/ifan_stage3_full20_freqblock_paper_original_20260419_005314`

当前唯一有效 baseline 对比来源：

- `IFAN_Edge/outputs/stage3/ifan_stage3_full20_freqblock_paper_original_20260419_005314/baseline_compare.json`

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

### 3.3 图表与结构事实

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

### 4.1 Stage 1 当前状态

- `当前实现`：已完成 `SRPPHATIcoMapAdapter`、`SRPLMSIcoMap`、`DualFeatureIcoPreprocessor`。
- `当前实现`：已完成 stage-1 特征导出与可视化链路。
- `当前结论`：双特征前端工程链路已经稳定可用。

### 4.2 Stage 2 当前状态

- `当前实现`：已完成 `IFANModelConfig` 与 `IFANModel` 的论文主线骨架重构。
- `当前实现`：当前主线是双分支、共享注意力、深 fusion head 的 paper-mainline。
- `当前结论`：当前 stage-2 不再处于“是否还是旧简化主干”的状态。

### 4.3 Stage 3 当前状态

- `当前实现`：已完成训练闭环、固定 validation cache、四场景 matched baseline compare。
- `当前实现`：已完成 `paper_original` LMS 行为口径与 `frequency_block` backend。
- `当前实现`：stage-3 当前默认主线为：
  - `paper_dual_mainline`
  - `PHAT + LMS`
  - `branch_channels = 16`
  - `shared_attention = true`
  - `fusion_head = 4 + 1 blocks`
  - `final_head_pooling = false`
  - `lms_backend = frequency_block`
  - `paper_original`

## 5. 当前唯一有效主结果

### 5.1 主结果来源

- 输出目录：
  - `IFAN_Edge/outputs/stage3/ifan_stage3_full20_freqblock_paper_original_20260419_005314`

### 5.2 验证集结果

- `best_val_rmsae_deg = 6.6669847294688225`
- `final val_rmsae_deg = 7.358573332428932`
- `final val_loss = 0.02454297846998088`
- 当前最优点出现在 `epoch 20`
- `epoch 21-40` 基本保持在 `7.25~7.46 deg`

### 5.3 四场景 matched baseline compare

- `IFAN mean RMSAE = 8.170234978199005 deg`
- `icoCNN mean RMSAE = 7.8069010972976685 deg`
- `delta = +0.36333388090133667 deg`

### 5.4 高混响低 SNR 两场景

- `IFAN mean RMSAE = 9.679717183113098 deg`
- `icoCNN mean RMSAE = 9.469039112329483 deg`
- `delta = +0.2106780707836151 deg`

### 5.5 分场景结果

- `scene_1: 4.198238372802734 vs 4.905608534812927 deg`
- `scene_2: 9.12326717376709 vs 7.3839176297187805 deg`
- `scene_3: 9.501166939735413 vs 9.228352904319763 deg`
- `scene_4: 9.858267426490784 vs 9.709725320339203 deg`

## 6. 当前结论

### 6.1 已经可以明确写死的结论

- `当前结论`：当前 IFAN 主线已经是完整重构后的正确主干，不再应被描述成“可能仍是简化主干”。
- `当前结论`：当前 IFAN 已经接近 `icoCNN r=2` baseline，而不是像旧无效主线那样明显落后。
- `当前结论`：这次结果已经证明 IFAN 不是旧简化网络的轻量替代，而是在 baseline 基础上做了完整深层结构与前端行为联合改造的网络。
- `当前结论`：旧简化主干阶段的实验结果不再具有 IFAN 复现证明力。

### 6.2 仍不能过度表述的部分

- `当前结论`：不能表述为“已完全复现论文最终效果”。
- `当前结论`：不能把“接近 baseline”直接外推为“已达到论文最好结果”。
- `当前结论`：在 LOCATA 单源任务完成前，论文级验收仍未闭环。

## 7. 当前剩余差距

当前剩余问题已经从“主干正确性”转移到以下几个方向：

1. 论文口径与当前评测口径是否仍有差异
2. `40 epoch` 是否已经代表这条主线的上限
3. `frequency_block` 与论文参考时域 LMS 是否还存在会影响上限的小差异
4. LOCATA 单源任务验收尚未完成

这也是当前后续工作的优先级，而不是回头继续解释旧简化主干为什么失败。

## 8. 当前阶段门槛表

| 项目 | 论文目标 | 当前工程状态 | 当前判断 |
| --- | --- | --- | --- |
| 输入特征 | `SRP-PHAT + SRP-LMS` | 已完成 | 已达成 |
| 主干结构 | 论文 IFAN 主线 | 已完成完整重构 | 已达成 |
| 训练闭环 | 可稳定正式训练 | 已完成 `paper_original + frequency_block` 扩训 | 已达成 |
| 模拟场景表现 | 论文中优于对比方法 | 当前已接近 baseline | 已显著收敛，但未宣称达到论文最好结果 |
| LOCATA 单源任务 | 论文级验收 | 尚未完成 | 未达成 |
| 轻量化迁移前提 | IFAN 基线已解释清楚 | 差距仍待继续分析 | 暂不进入主线 |

## 9. 下一步建议

- 优先继续做：
  - 论文口径与当前评测口径的差异核对
  - 更长训练预算或更大评测口径的稳定性验证
  - LOCATA 单源任务验收准备
- 暂不再做：
  - 继续在正文中保留旧简化主干实验统计
  - 把旧无效主线当成 IFAN 复现失败证据
  - 过早进入轻量化迁移主线

## 10. 相关文档

- 阶段 1：[`../IFAN_Edge/docs/stage_01_after.md`](../IFAN_Edge/docs/stage_01_after.md)
- 阶段 2：[`../IFAN_Edge/docs/stage_02_after.md`](../IFAN_Edge/docs/stage_02_after.md)
- 阶段 3 当前事实：[`../IFAN_Edge/docs/stage_03_after.md`](../IFAN_Edge/docs/stage_03_after.md)
- 阶段 3 剩余差距分析：[`../IFAN_Edge/docs/stage_03_analysis.md`](../IFAN_Edge/docs/stage_03_analysis.md)
- 阶段 3 收尾路线：[`../IFAN_Edge/docs/stage_03_recovery_plan.md`](../IFAN_Edge/docs/stage_03_recovery_plan.md)
