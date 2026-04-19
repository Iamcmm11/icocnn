# 阶段 03 完成后

## 当前结论

阶段 3 当前唯一有效的结果口径固定为：

- 只有“完整重构后的 IFAN 主线”产生的实验结果才计入 IFAN 复现结论。
- 任何“旧简化主干 / 重构前主干 / 未完整复现架构”的实验只保留一句历史说明，不再在正文中保留指标表或对比结论。

截至 `2026-04-19`，当前 stage-3 主线已经明确收口为：

- `paper_dual_mainline`
- `paper_original` LMS 行为口径
- `frequency_block` LMS backend

当前最准确的阶段性表述应为：

“已完成 IFAN 正确主干的重构、训练闭环与扩训验证。最新主线结果已经接近 `icoCNN r=2` baseline，但与论文最好结果之间仍有剩余差距；这说明 IFAN 不是旧简化网络可以代表的轻量变体，而是在 baseline 之上做了完整深层结构与前端行为联合调整的网络。”

## 有效结果保留标准

- 正文只保留完整重构后 IFAN 主线的结果。
- 当前唯一主结果来源：
  - `outputs/stage3/ifan_stage3_full20_freqblock_paper_original_20260419_005314`
- 当前唯一主对比来源：
  - `outputs/stage3/ifan_stage3_full20_freqblock_paper_original_20260419_005314/baseline_compare.json`
- 旧简化主干阶段的实验结果不再作为 IFAN 复现证据引用。

## 当前主线配置

- 当前主配置：
  - `configs/stage3_default.toml`
- 当前默认主线：
  - `model_topology = paper_dual_mainline`
  - `PHAT + LMS`
  - `branch_channels = 16`
  - `shared_attention = true`
  - `fusion_head = 4 + 1 blocks`
  - `final_head_pooling = false`
  - `lms_backend = frequency_block`
  - `paper_original` 口径：plain LMS、跨帧连续更新、`tau_sample` 读出、全麦对
  - `K = 4096`
  - `step = 3072`
  - `fs = 16000`
  - `trajectory_seconds = 20`
  - `epoch 1-20: lr=1e-4, batch=1, SNR=30dB`
  - `epoch 21-40: lr=1e-5, batch=10, micro_batch=1, SNR=5~30dB`

## 最新有效主结果

- 主结果输出目录：
  - `outputs/stage3/ifan_stage3_full20_freqblock_paper_original_20260419_005314`
- 关键产物：
  - `history.csv`
  - `summary.json`
  - `baseline_compare.json`
  - `checkpoints/best_rmsae.pt`
  - `checkpoints/last.pt`

### 验证集结果

- `best_val_rmsae_deg = 6.6669847294688225`
- `final train_loss = 0.025400243774374835`
- `final val_loss = 0.02454297846998088`
- `final val_rmsae_deg = 7.358573332428932`
- 最优点出现在 `epoch 20`
- `epoch 21-40` 基本稳定在 `val_rmsae ≈ 7.25~7.46 deg`

### 四场景 matched baseline compare

- `IFAN mean RMSAE = 8.170234978199005 deg`
- `icoCNN mean RMSAE = 7.8069010972976685 deg`
- `delta = +0.36333388090133667 deg`

### 高混响低 SNR 两场景

- `IFAN mean RMSAE = 9.679717183113098 deg`
- `icoCNN mean RMSAE = 9.469039112329483 deg`
- `delta = +0.2106780707836151 deg`

### 分场景结果

- `scene_1: 4.198238372802734 vs 4.905608534812927 deg`
- `scene_2: 9.12326717376709 vs 7.3839176297187805 deg`
- `scene_3: 9.501166939735413 vs 9.228352904319763 deg`
- `scene_4: 9.858267426490784 vs 9.709725320339203 deg`

## 当前工程性结论

- `frequency_block` 已经把 LMS 前处理成本压到可支撑正式训练的范围，完整主线不再受早期时域实现的训练速度瓶颈限制。
- 完整重构后的 IFAN 主线已经可以稳定接近 baseline，而不是像旧无效主线那样显著落后。
- `scene_1` 已经优于 baseline，其他三个场景只剩小幅差距，说明当前差距已经不再是“主干根本没复现对”的量级。
- 这次结果已经足够证明：
  - IFAN 不是“在 baseline 上随手删减后留下的轻量网络”
  - IFAN 的有效增益来自完整主干、双特征输入和 LMS 动态行为的联合设计
  - 旧简化主干阶段的实验结果不应继续被当成 IFAN 复现结论

## 证据链保留说明

- 当前仍保留少量 smoke / 小口径实验名称，只用于说明恢复路径已经走通过。
- 这些历史实验不再作为正文指标结论来源。
- 当前正文只保留一条历史说明：
  - 在完整主线稳定前，项目曾经历过旧简化主干与 LMS 行为修正阶段；这些实验已完成其排障价值，但不再计入 IFAN 复现结果。

## 与论文目标的关系

- 当前不能表述为“已完全复现论文最终效果”。
- 当前更准确的表述是：
  - 已完成正确主干的重构与训练闭环
  - 已得到一版接近 baseline 的稳定主结果
  - 已验证 IFAN 的完整结构方向是成立的
  - 但与论文最好结果、论文级 LOCATA 验收之间仍有剩余差距

## 当前未完成事项

- 尚未完成论文模拟实验口径与当前工程评测口径的逐项对齐说明。
- 尚未完成更长训练预算下的收敛边界验证。
- 尚未完成 `final_head_pooling` 是否值得继续保留为分叉项的最终结论。
- 尚未完成 LOCATA 单源任务的论文级验收。
- 尚未回补阶段 2 的 FLOPs / MAC 论文口径校准。

## 下一步建议

- 继续把后续分析集中在“为什么已经接近 baseline，但仍未达到论文最好结果”这一问题上。
- 优先保留以下三个方向：
  - 论文口径与当前评测口径的差异核对
  - 更长训练预算或更稳评测口径下的收敛验证
  - LOCATA 单源任务验收
- 不再把“旧简化主干结果为什么差”作为主线问题继续占用正文空间。
