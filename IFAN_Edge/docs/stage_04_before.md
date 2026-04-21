# 阶段 04 开始前

阶段 4 的原始目标是补充：

- LOCATA 评估
- 轨迹可视化
- IFAN 基线验收

但在进入阶段 4 后，任务重心发生了调整。

## 当时的起点事实

- 当前 IFAN 主线已经完成 paper-mainline 重构，并得到一版接近 baseline 的有效主结果。
- 当时仍未完成 LOCATA 单源任务验收，因此论文级闭环尚未完成。
- 当时对尾部结构仍存在一个关键歧义：
  - 论文图示里的尾部 `pooling` 到底表示什么
  - 它是否应作为额外的 `final_head_pooling` 保留

## 当时最容易混淆的点

在阶段 4 开始时，需要先区分两件不同的操作：

- 末端已有的 `R-pooling`
  - 具体是 `channel readout -> max over R -> CleanVertices -> SoftArgMax`
  - 这一步只聚合 `R=6` 个 orientation channels，不降低 `icosahedral` 空间分辨率
- 额外的 `final_head_pooling`
  - 这是可选的 `icosahedral pooling`
  - 它会继续降低空间分辨率
  - 它不是 baseline 式 `R-pooling` 的同义词

阶段 4 后续的结构验证、论文图核对和 gap 分析，都建立在这个区分之上。

## 阶段 4 进入时的默认方向

进入阶段 4 时，默认方向应表述为：

- 先核对 gap 的真实来源
- 再验证 `final_head_pooling` 这种尾部歧义
- 最后才决定是否需要把重点重新拉回 LOCATA 或其他扩训动作

## 当前说明

本文件只保留“阶段进入前”的起点语境。

阶段 4 的后续判断、LOCATA 同步结论与当前主线优先级，现统一由以下文档接管：

- [`stage_04_after.md`](/home/cmm/icocnn/IFAN_Edge/docs/stage_04_after.md)
- [`../../相关参考论文/IFAN复现.md`](/home/cmm/icocnn/相关参考论文/IFAN复现.md)

也就是说：

- 本文件仍有历史说明价值
- 但它不再代表当前阶段的实际优先级
