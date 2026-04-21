# 阶段 04 完成后

## 当前定位

阶段 4 现在不再沿用早期的“结构 gate / hard-scene 机制 / 论文差异并列推进”叙事。

在补上 LOCATA 同步验收后，当前最准确的阶段表述应为：

- `IFAN` 仍是当前最稳的复现主线
- `IFAN_Maba` 先保留为后续候选增强线
- 阶段 4 的主要任务已经从“扩展更多结构分支”切换为“解释 IFAN 与论文剩余 gap”

## 当前固定事实

- 当前唯一有效 IFAN 主结果仍是：
  - `IFAN_Edge/outputs/stage3/ifan_stage3_full20_freqblock_paper_original_20260419_005314`
- 当前四模型 LOCATA 对比结果：
  - `IFAN_Maba/outputs/stage3/analysis/locata_four_model_compare.json`
  - `IFAN_Maba/outputs/stage3/analysis/locata_four_model_compare.md`
- 当前 IFAN LOCATA 单模型结果：
  - `IFAN_Edge/outputs/stage3/analysis/locata_eval_benchmark2_best_rerun.json`
- 当前 IFAN_Maba LOCATA 单模型结果：
  - `IFAN_Maba/outputs/stage3/analysis/locata_eval_benchmark2_best.json`

## LOCATA 当前结论

### 当前口径边界

当前本地 `benchmark2 / eval / Task 1, 3, 5` 实际跑通并用于汇总的 recording 数是：

- `task1 = 13`
- `task3 = 5`
- `task5 = 5`
- `total = 23`

因此，阶段 4 当前所有 LOCATA 结论都建立在这套本地可用口径上。

若后续数据补全到 `task1 = 26 / task3 = 5 / task5 = 10`，应作为“口径扩大后的复核结果”单独记录，而不是直接与当前表混写。

### 四模型对比下的主判断

当前 LOCATA 四模型对比表明：

- `IFAN` 和 `IFAN_Maba` 都已经明显优于本地 `icoCNN` baseline
- 但现阶段更稳、更适合作为继续补论文 gap 主线的是 `IFAN`

原因不是单看一个平均值，而是综合当前 task 结构后的判断：

- `IFAN` 的 `with silences` 总体更优：
  - average `7.4228 deg`
  - baseline `8.5718 deg`
  - delta `-1.1489 deg`
- `IFAN_Maba` 的 `with silences` 总体略弱于 `IFAN`：
  - average `7.5172 deg`
  - delta `-1.0546 deg`
- `without silences` 下 `IFAN_Maba` 的 overall average 略低于 `IFAN`
  - 但差距很小
  - 同时 `Task 3 / Task 5` 并没有形成比 `IFAN` 更明确、更稳定的优势

因此当前更合理的阶段判断是：

- `IFAN` 继续承担复现主线
- `IFAN_Maba` 暂不删除
- 但它当前没有提供足够确定的增益来替代 `IFAN`

### 分任务观察

当前 LOCATA 结果下，`IFAN` 的最重要特点是：

- `Task 1` 已经表现较强，是整体 gain 的主要来源
- `Task 3` 仍是当前最主要短板
- `Task 5` 有改善，但还不能视为“论文级显著优势”

这也解释了为什么阶段 4 现在不能写成“IFAN_Maba 已经成为新的主线”：

- 当前最需要解决的问题仍是 `IFAN` 在更复杂、更动态任务上的剩余差距
- 而不是继续扩展更多结构分支

## 当前优先级重排

阶段 4 当前优先级应统一改写为：

1. 固定 `IFAN` 为当前复现主线
2. 继续解释 `IFAN` 与论文 IFAN 的剩余差距
3. 继续核对论文结构与评测口径差异
4. `IFAN_Maba` 暂列为后续候选线

这意味着以下叙事不再作为当前阶段主表述：

- `final_head_pooling` 与 `LMS backend` 仍是阶段 4 的并列主线
- `IFAN_Maba` 已经足够强到可以直接接替 `IFAN`

更准确的说法是：

- `final_head_pooling`
- `LMS backend`
- 论文结构与口径差异

这些工作仍有解释价值，但现在都应服务于 `IFAN` 主线，而不是替代主线。

## 对 `IFAN_Maba` 的当前处理

当前对 `IFAN_Maba` 的阶段性处理固定为：

- 保留已有训练与 LOCATA 结果
- 保留其作为后续候选增强路线
- 暂不再把它升级为默认主线

原因：

- 它没有在当前 LOCATA 口径下形成足够明确的整体优势
- 当前最缺的仍是 `IFAN` 主线与论文 gap 的解释
- 继续在此时切主线，只会增加解释复杂度

## 当前阶段结论

阶段 4 当前最准确的总结应为：

- LOCATA 已经同步补入阶段结论链
- `IFAN` 仍是当前更稳的主线
- `IFAN_Maba` 暂缓，不删除
- 当前阶段重点不再是展开更多结构分支，而是围绕 `IFAN` 主线解释论文剩余差距

## 相关文档

- 当前总纲：
  - [`../../相关参考论文/IFAN复现.md`](/home/cmm/icocnn/相关参考论文/IFAN复现.md)
- 阶段 4 开始前：
  - [`stage_04_before.md`](/home/cmm/icocnn/IFAN_Edge/docs/stage_04_before.md)
- 阶段 4 论文结构与口径差异：
  - [`stage_04_paper_gap_checklist.md`](/home/cmm/icocnn/IFAN_Edge/docs/stage_04_paper_gap_checklist.md)
