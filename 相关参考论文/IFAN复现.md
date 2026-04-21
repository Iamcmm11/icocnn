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
- `当前结论`：当前本地 LOCATA 单源任务验收已经完成，但论文级闭环仍未完全完成。

## 7. 当前剩余差距

当前剩余问题已经从“主干正确性”转移到以下几个方向：

1. 论文口径与当前评测口径是否仍有差异
2. `40 epoch` 是否已经代表这条主线的上限
3. `frequency_block` 与论文参考时域 LMS 是否还存在会影响上限的小差异
4. 已完成的 LOCATA 本地验收与论文级结论之间仍需继续解释和复核

这也是当前后续工作的优先级，而不是回头继续解释旧简化主干为什么失败。

## 8. 当前阶段门槛表

| 项目 | 论文目标 | 当前工程状态 | 当前判断 |
| --- | --- | --- | --- |
| 输入特征 | `SRP-PHAT + SRP-LMS` | 已完成 | 已达成 |
| 主干结构 | 论文 IFAN 主线 | 已完成完整重构 | 已达成 |
| 训练闭环 | 可稳定正式训练 | 已完成 `paper_original + frequency_block` 扩训 | 已达成 |
| 模拟场景表现 | 论文中优于对比方法 | 当前已接近 baseline | 已显著收敛，但未宣称达到论文最好结果 |
| LOCATA 单源任务 | 论文级验收 | 已完成当前本地 `benchmark2 / eval / task1,3,5` 验收 | 已达成当前本地验收，但论文级闭环仍待继续解释 |
| 轻量化迁移前提 | IFAN 基线已解释清楚 | 差距仍待继续分析 | 暂不进入主线 |

## 9. 下一步建议

- 优先继续做：
  - 论文口径与当前评测口径的差异核对
  - 更长训练预算或更大评测口径的稳定性验证
  - 已完成 LOCATA 结果的论文级解释与复核
- 暂不再做：
  - 继续在正文中保留旧简化主干实验统计
  - 把旧无效主线当成 IFAN 复现失败证据
  - 过早进入轻量化迁移主线

### 9.1 已锁定的当前主基线

当前仓库中后续所有判断统一以以下配置为唯一主基线：

- 配置文件：`IFAN_Edge/configs/stage3_default.toml`
- 主线标签：
  - `experiment_role = mainline_baseline`
  - `srp_variant = paper_original`
  - `temporal_conv_variant = standard_1d`
  - `temporal_module = conv`
- 核心行为：
  - `paper_dual_mainline`
  - `PHAT + LMS`
  - `lms_backend = frequency_block`
  - `final_head_pooling = false`
  - `epochs = 40`

所有新实验都必须相对这条主基线解释差异，不再允许“边改主线边比较”。

### 9.2 已落地的三类推进工具

为落实“先补复现、再做轻量化”，当前仓库已经补齐以下入口：

- 论文口径审计：
  - `IFAN_Edge/scripts/audit_stage3_protocol.py`
  - 用于输出 `paper vs local` 差异表，先判断差距来自模型还是来自评测/训练口径
- 更长预算训练：
  - `IFAN_Edge/configs/stage3_long_budget.toml`
  - 用于在不改结构前提下直接验证 `40 epoch` 是否只是当前工程预算上限
- LMS backend 等价性抽检：
  - `IFAN_Edge/configs/stage3_reference_backend_probe.toml`
  - 用于在小预算下只切换 `time_reference`，判断 `frequency_block` 是否可能解释尾部差距
- 转轻量化门槛评估：
  - `IFAN_Edge/scripts/assess_stage3_readiness.py`
  - 用于基于 `summary.json + LOCATA report` 输出“继续补复现”还是“可以进入轻量化”的结论

### 9.3 当前推荐执行顺序

1. 先用 `stage3_default.toml` 保持主线不漂移，继续产出可比结果。
2. 先运行 `audit_stage3_protocol.py`，把论文口径差异表补齐。
3. 再运行 `stage3_long_budget.toml`，验证训练预算是否仍能带来实质改善。
4. 若仍有疑问，再运行 `stage3_reference_backend_probe.toml`，隔离 LMS backend 的影响。
5. 每轮结束后同时更新 simulated 与 LOCATA 结果，再用 `assess_stage3_readiness.py` 判断是否进入轻量化。

### 9.4 当前转入轻量化门槛

当前工程已把默认门槛固定为：

- `LOCATA` 总体结果稳定优于本地 `icoCNN`
- `Task 3 / Task 5` 不出现明显恶化
- 连续一轮或多轮“只补复现”的平均收益低于约 `0.3 deg`

在满足上述条件前，不进入 `SRP` 替换、`DSConv` 替换或 `IFAN-lite-Maba` 组合实验。

### 9.5 当前主线选择

结合当前 LOCATA 四模型对比，当前主线选择应明确写成：

- `IFAN` 和 `IFAN_Maba` 都已明显优于本地 baseline
- 但现阶段更稳、更适合作为继续补论文 gap 主线的是 `IFAN`

原因是：

- `IFAN` 在 `with silences` 总体上更强
- `IFAN_Maba` 虽然仍有价值，但没有形成足够确定、足够稳定的整体替代优势
- 当前剩余问题的核心仍是：
  - `IFAN` 与论文 IFAN 的差距解释
  - 尤其是更复杂、更动态 task 上的剩余差距

因此当前默认策略固定为：

- `IFAN` 继续承担复现主线
- `IFAN_Maba` 暂不删除
- 但它当前只保留为后续候选增强线，而不是默认主线

### 9.6 当前 LOCATA 口径边界

当前本地 `benchmark2 / eval / Task 1, 3, 5` 实际跑通并用于汇总的 recording 数是：

- `task1 = 13`
- `task3 = 5`
- `task5 = 5`
- `total = 23`

因此，当前总纲中关于 LOCATA 的所有判断，都建立在这套本地可运行口径上。

若后续环境补全到 `task1 = 26 / task3 = 5 / task5 = 10`，应单独记为“口径扩大后的复核结果”，而不是直接与当前结果混写。

## 10. 相关文档

- 阶段 1：[`../IFAN_Edge/docs/stage_01_after.md`](../IFAN_Edge/docs/stage_01_after.md)
- 阶段 2：[`../IFAN_Edge/docs/stage_02_after.md`](../IFAN_Edge/docs/stage_02_after.md)
- 阶段 3 当前事实：[`../IFAN_Edge/docs/stage_03_after.md`](../IFAN_Edge/docs/stage_03_after.md)
- 阶段 3 剩余差距分析：[`../IFAN_Edge/docs/stage_03_analysis.md`](../IFAN_Edge/docs/stage_03_analysis.md)
- 阶段 3 收尾路线：[`../IFAN_Edge/docs/stage_03_recovery_plan.md`](../IFAN_Edge/docs/stage_03_recovery_plan.md)
- 阶段 4 当前结论：[`../IFAN_Edge/docs/stage_04_after.md`](../IFAN_Edge/docs/stage_04_after.md)

## 11. 历史完整执行大纲（同步自 `5b5b51c`）

本章节同步保留项目早期形成的完整执行蓝图。

它的用途是：

- 作为长期路线和原始分阶段规划的保留版本
- 方便后续论文写作、系统化回顾和远期扩展时回看

但当前需要明确：

- 本章节不是当前唯一执行顺序
- 当前实际优先级仍以前文“当前总纲与当前结论”为准

### 11.1 当前如何使用这份历史大纲

为避免再次丢失长期方向，以下标记固定采用：

- `当前主线`：
  - 与当前 IFAN 复现闭环直接相关，优先执行
- `中期候选`：
  - 在 IFAN 主线解释清楚后可重新启用
- `暂停项`：
  - 当前不删，但暂不占主线资源

其中：

- `SRP-LMS 前端`
- `IFAN 结构改造`
- `训练与收敛验证`
- `LOCATA 验收`

仍属于 `当前主线` 的历史来源。

- 极致轻量化
- IFAN-Edge 系列
- HLS / FPGA 回接

当前属于 `中期候选` 或 `暂停项`，但不再删除。

### 11.2 历史阶段 1：SRP-LMS 特征生成与验证

`当前主线`

原始规划要点：

- 基于麦克风阵列信号实现 `SRP-LMS` 计算
- 复用 `IcoCNN` 中已有的二十面体网格生成与投影代码
- 数据管道同时输出 `PHAT + LMS` 双特征
- 保持训练 / 测试集划分、帧长 `4096`、帧移 `3072`
- 做特征维度一致性校验和四种典型声学场景可视化

这部分当前已经完成并固化为：

- `SRPPHATIcoMapAdapter`
- `SRPLMSIcoMap`
- `DualFeatureIcoPreprocessor`

### 11.3 历史阶段 2：IFAN 模型结构增量改造

`当前主线`

原始规划要点：

- 在 `IcoCNN` 基础上增量加入：
  - 特征残差学习模块
  - 双特征输入分支
  - 共享注意力权重模块
  - 融合特征学习模块
- 验证前向、反向、参数量与复杂度
- 保留论文主线骨架，不做无根据的大改动

这部分当前已经完成并进入正式训练主线。

### 11.4 历史阶段 3：IFAN 训练与收敛验证

`当前主线`

原始规划要点：

- 严格按 IFAN 论文做两阶段训练
- 损失函数采用笛卡尔坐标 MSE
- 优化器采用 Adam
- 前期高 SNR、后期混合 SNR
- 做模拟数据测试与消融验证

当前对应状态：

- 训练闭环已经建立
- `paper_original + frequency_block` 已形成正式主线
- 当前后续重点是继续解释与论文的剩余差距，而不是回退到旧简化主干

### 11.5 历史阶段 4：LOCATA 数据集测试与 IFAN 基线验收

`当前主线`

原始规划要点：

- 预处理 `LOCATA` 单源任务 `Task 1 / 3 / 5`
- 分别测试：
  - 含静音帧
  - 不含静音帧
- 计算每任务 RMSAE 与平均 RMSAE
- 进行轨迹可视化与论文级验收

当前对应状态：

- LOCATA 已经同步回当前结论链
- 当前更稳的主线是 `IFAN`
- `IFAN_Maba` 暂保留为后续候选分支
- 当前 LOCATA 结论仍受本地可用 recording 口径约束

### 11.6 历史阶段 5：IFAN 的轻量化与 IFAN-Edge 系列

`中期候选`

原始规划要点：

- `SRP` 计算轻量化
- 更细的训练策略调整
- 全局通道缩放
- 深度可分离卷积替换
- 提前下采样
- 注意力模块简化
- 量化与极端边缘部署

当前处理原则：

- 这条长期路线继续保留
- 但当前不作为主线优先级
- 只有在 IFAN 主线差距解释清楚后，才重新进入主线讨论

### 11.7 HLS / FPGA 方向的当前定位

`暂停项`

项目早期的完整蓝图里还包含：

- HLS 工作流熟悉与加速链路回接
- 端侧部署与资源受限优化

这一方向当前不删除，但阶段性定位已固定为：

- 先完成网络层面的理论与算法主线
- 后续再把更稳定的优化网络重新回接到 HLS / FPGA
