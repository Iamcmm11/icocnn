# 阶段 04 论文结构与口径差异清单

## 目标

本清单只保留当前仍值得继续核对的论文差异点，不再回退到“主干可能整体仍错”的旧叙事。

当前默认用途：

- 作为 `final_head_pooling` 结果和 `LMS backend / hard-scene` 分析后的第三优先级核对项
- 只做结构/口径差异解释，不直接扩展成新的重训练主线

当前阶段还需要补一条优先级说明：

- 由于 LOCATA 当前对比下 `IFAN > IFAN_Maba` 的主线判断已经成立
- 本清单现在继续服务于“解释 `IFAN` 与论文剩余 gap”
- 不再优先服务于“是否把 `MABA` 升级为 IFAN 主线增强路线”

## 结构差异

### 1. `final_head_pooling` 是否应默认开启

当前状态：

- 当前正式主结果使用 `final_head_pooling = false`
- 当前代码末端已经有 baseline 风格的 `R-pooling`
  - 具体是 `channel readout -> max over R -> CleanVertices -> SoftArgMax`
  - 这层只聚合 `R=6` 个 orientation channels，不降低空间分辨率
- 当前代码另外还支持在最终 `LNorm` 之后、`SoftArgMax` 之前增加一层 optional icosahedral pooling
- 这一点在当前结构理解里仍是显式歧义

论文证据：

- `Fig. 6` 的融合特征学习模块图示容易被读成最后一层存在 pooling
- 当前结构对照文档已把这一点列为最高优先级歧义

当前判断：

- 这是最值得先用实验确认的结构点
- 但必须明确：论文图里的尾部 `pooling` 不能直接等同于 baseline 式 `R-pooling`
- 当前歧义点是“是否存在额外的 icosahedral pooling”，不是“末端有没有聚合”
- 若小预算对照显示只表现为 hard-scene gain with easy-scene cost，则保留为 trade-off，不改默认值

### 2. `32 kernels` 是否仅代表卷积核数量

当前状态：

- 当前实现按“Fusion Feature 后主干仍保持 16 channels”理解落地
- 目前没有把正文中的 `32 kernels` 解释成“通道数升到 32”

论文证据：

- 正文描述更像“icosahedral convolution of 32 nuclei”
- 当前更合理解释是卷积核/邻域表达，而不是主干 feature width

当前判断：

- 在没有更强证据前，不应把它上升为新的主干分叉
- 只保留为文字解释核对项

## 口径差异

### 3. 当前 hard scenes 是否已足够代表论文描述

当前状态：

- 更大 cache、多 seed 评测表明稳定 gap 主要来自 `scene_3/4`
- `scene_4 (5 dB, T60=1.4)` 超出当前训练 `T60 <= 1.3` 上限

论文事实：

- 论文图示与文字强调低 SNR、较强混响下的鲁棒性
- 论文示例场景包含 `SNR = 5 dB, T60 = 1.4 s`

当前判断：

- `scene_4` 仍是有代表性的论文式 hard scene
- 但它也带有“超出训练上限”的额外因素，解释结果时必须单独注明

### 4. 当前大 cache 结论与论文叙事是否冲突

当前状态：

- 更大口径下 `scene_2` 已基本持平，不再是主矛盾
- 当前主要差距集中在 `scene_3/4`

论文叙事：

- 论文强调 IFAN 在复杂室内声学条件下的鲁棒性优势
- 论文图示更偏 hard-scene 成功案例

当前判断：

- 当前复现并未否定 IFAN 方向本身
- 但还不能表述为“已达到论文 hard-scene 优势”
- 这正是 `LMS backend / hard-scene` 机制需要优先解释的原因

## 当前结论

- 结构层面最值得继续验证的只剩 `final_head_pooling`
- 文字层面最值得继续核对的只剩 `32 kernels`
- 口径层面最关键的是：当前 hard-scene gap 是否来自 LMS backend、最终读出鲁棒性，还是论文/训练口径差异
- 当前这份 checklist 的角色是解释 `IFAN` 主线剩余差距
- 不是扩展 `IFAN_Maba` 主线的优先入口

## 不再优先做的事

- 不再把 `scene_2` 当作当前最大 gap 主矛盾
- 不再把“phase-2 调度”列为前三优先级主线
- 不再把论文结构核对扩展成新的大规模主干重构任务
