# IFAN 复现与轻量化迭代总体大纲

## 1. 文档定位

本文档用于指导当前 IFAN 工程的后续复现、对齐、轻量化与边缘部署实验。当前项目已经完成了基于 `IcoCNN` 的 IFAN Stage-1/2/3 主线工程搭建，并形成了 PHAT + LMS 双特征、IFAN 双分支注意力融合、模拟场景评估与 LOCATA 评估链路。后续工作不再以“从零复现 IFAN”为主，而是以“稳定现有 IFAN 主线 → 引入 Cross3D-Edge 论文启发的轻量化改造 → 做可回滚消融实验 → 输出 IFAN-Edge 系列方案”为核心。

重点参考论文为《CNN-based Robust Sound Source Localization with SRP-PHAT for the Extreme Edge》。该论文从 Cross3D 出发，提出 LC-SRP-Edge 与 Cross3D-Edge，给出 SRP 特征、输入分辨率、深度可分离卷积、通道缩放和硬件资源评估方面的系统结论。本文档仅作为工程计划与实验设计依据，不直接要求修改当前正在训练的代码。

---

## 2. 当前工程基线与前期基础工作

### 2.1 已完成基础

- **IcoCNN 基线**：已具备 `r=2` IcoCNN 模型、IcoConv、G-Padding、二十面体网格映射、SoftArgMax 坐标回归链路。
- **数据与评估链路**：已具备 LibriSpeech + gpuRIR 随机轨迹模拟、Benchmark2 12 麦阵列配置、LOCATA 单源任务评估、RMSAE 指标统计。
- **IFAN 双特征预处理**：已形成 SRP-PHAT 与 SRP-LMS 双特征输出，默认对齐为 `[B, 2, T, 5, 4, 8]` 的二十面体特征图。
- **LMS 工程优化**：已有 `time_reference` 与 `frequency_block` 后端，当前主线倾向使用 `frequency_block` 降低前处理成本。
- **IFAN 主线模型**：已实现 PHAT 分支、LMS 分支、残差增强、共享注意力权重、双特征融合、融合头与坐标输出。
- **Stage-3 训练框架**：已具备配置化训练、阶段式 SNR 课程、固定验证缓存、四场景模拟对比、LOCATA benchmark2 报告和 readiness assessment。

### 2.2 当前正在保留的主线假设

- 主线配置仍以当前训练使用的 `paper_original + frequency_block + r=2` 方案作为全精度/工程基线。
- 轻量化相关改动必须作为新实验分支或新配置引入，不直接覆盖当前训练主线。
- 每项改动必须可单独开关，支持与当前 anchor run 做公平对比。
- 任何改动进入默认配置前，必须先完成小样本前向、参数量、MAC proxy、模拟场景、LOCATA 的分层验证。

---

## 3. 论文关键结论提炼及对 IFAN 的启发

### 3.1 SRP-PHAT 特征处理

论文从 Cross3D 的 SRP-PHAT 输入出发，强调 SRP-PHAT 在噪声与混响环境下的鲁棒性，并指出传统 TD-SRP 在高分辨率候选网格下存在明显计算压力。论文提出 LC-SRP-Edge：

- 用低复杂度 SRP 插值思想替代原始 TD-SRP 的高成本搜索。
- 利用 sinc 插值项的对称性，将插值索引范围从正负对称区间压缩到非负区间。
- 在保持与 LC-SRP 数学等价的前提下降低插值计算量和片上系数缓存。
- 论文报告 LC-SRP-Edge 相对 LC-SRP 可显著减少插值计算和片上存储；在 16 kHz、8×16 等典型设置下可降低 SRP 端延迟。

对 IFAN 的启发：当前 PHAT 分支仍可保留 SRP-PHAT 作为鲁棒空间特征，但 SRP 模块应从“直接复用旧 TD-SRP/全量 GCC 索引”逐步重构为“候选点预计算 + 麦克风对压缩 + 插值系数缓存 + 向量化聚合”的边缘友好版本。

### 3.2 输入特征图尺寸

论文比较了 4×8、8×16、16×32、32×64 多种 SRP 候选空间分辨率，结论是：

- 4×8 过粗，难以充分描述声场。
- 8×16 是 Cross3D/Cross3D-Edge 中较均衡的精度-复杂度折中点。
- 16×32 与 32×64 虽有更细网格，但模型复杂度和运行延迟快速膨胀，且高分辨率收益出现饱和。
- 论文最终将 16 kHz + 8×16 + Cross3D-Edge-Medium 作为典型平衡方案，并在 Raspberry Pi 4B 上给出实时性验证。

对 IFAN 的启发：IFAN 现有二十面体输入并非规则 Elevation×Azimuth 图，但应统一一个“工程默认分辨率”并围绕该分辨率做消融。当前 `r=2` 对应每 chart `4×8`、共 5 个 chart 的二十面体特征；不能机械等同于论文的 4×8，但局部特征图尺寸、候选点数量和后端计算量都接近轻量化目标。后续可设计两个层级：

- **默认基线层级**：保持当前 `r=2`，保证与已训练主线、IcoCNN checkpoint、LOCATA 结果兼容。
- **轻量实验层级**：新增紧凑输入消融，如 `r=1` 或规则 8×16 SRP 投影适配，仅作为 IFAN-Edge 输入分辨率实验，不覆盖主线。

### 3.3 深度可分离卷积与通道缩放

论文指出 Cross3D 的主要瓶颈来自 Cross_Conv 计算量与 Output_Conv1 参数量，因此采用两类优化：

- 压缩中间通道数 `C`，形成 Large/Medium/Small 系列。
- 在输出时序卷积部分引入 depth-wise separable convolution，显著减少参数量和带宽需求。

论文的 Cross3D-Edge 系列在多分辨率下相对 baseline 可降低计算复杂度与参数量，并给出 Large/Medium/Small 的硬件资源对比。对 IFAN 的启发是：

- IFAN 中重复出现的 IcoConv 与 Temporal Conv 是优先轻量化对象。
- 深度可分离卷积应先从时序卷积、读出头等低风险模块引入，再评估是否推广到 IcoConv。
- 全局通道宽度可从当前 `C=16` 做 `16 → 12 → 8 → 4` 消融，不应一次性替换默认主线。

---

## 4. 总体技术路线

```text
已复现 IcoCNN / 当前 IFAN 主线
        │
        ├─ 基线冻结：paper_original + frequency_block + r=2
        │
        ├─ SRP 模块优化：TD-SRP / 旧 PHAT → LC-SRP-Edge 风格 SRP-PHAT
        │
        ├─ 模型轻量化：标准卷积 → 深度可分离卷积 + 通道缩放
        │
        ├─ 输入尺寸消融：r=2 保持主线，新增 r=1 / 规则 8×16 映射实验
        │
        └─ IFAN-Edge 系列：Large / Medium / Small，完成模拟 + LOCATA + 资源评估
```

原则：

1. **先冻结当前主线**：当前正在运行的 IFAN 训练不受轻量化实验影响。
2. **所有改动配置化**：SRP 后端、输入分辨率、卷积类型、通道数必须可开关。
3. **先消融再合并**：SRP、卷积、输入尺寸三类改动单独验证后，再做组合实验。
4. **优先复现实验结论，不盲目追求最小模型**：以 RMSAE、参数量、MAC proxy、前处理耗时、LOCATA 稳定性共同决策。

---

## 5. 分阶段执行计划

### 阶段 0：冻结当前 IFAN 主线与训练记录

目标：保护当前正在跑的训练，不让轻量化计划污染基线。

执行内容：

- 保持当前 `stage3_default.toml`、`stage3_long_budget.toml` 等训练配置不变。
- 记录当前训练 run 目录、日志、checkpoint、summary、LOCATA 报告路径。
- 将当前主线标记为 `IFAN-Full / paper_original / frequency_block / r=2`。
- 后续所有轻量化配置新建文件，例如 `stage3_edge_lc_srp.toml`、`stage3_edge_dwconv.toml`、`stage3_edge_resolution.toml`。

验收：

- 当前训练不中断，输出目录持续写入正常。
- 新轻量化计划不会覆盖默认训练命令。

### 阶段 1：SRP-PHAT 模块重构计划

目标：借鉴 LC-SRP-Edge 思想，重构 PHAT 分支的 SRP 计算逻辑，降低前处理复杂度。

设计方向：

- **候选点预计算**：对二十面体网格候选点预计算每个麦克风对的 TDOA。
- **麦克风对压缩**：只计算唯一麦克风对 `m > m'`，避免当前可能存在的 `N×N` 冗余累加。
- **插值系数缓存**：为每个候选点和麦克风对缓存 sinc 插值系数。
- **LC-SRP-Edge 对称化**：利用正负插值点的对称关系，将缓存和计算集中到非负索引区间。
- **向量化聚合**：尽量用 PyTorch tensor gather/einsum 或 batched matmul 替代 Python 双重循环。

建议配置开关：

- `phat_srp_backend = "td_reference"`：旧版基准。
- `phat_srp_backend = "lc_reference"`：完整 sinc 插值参考实现。
- `phat_srp_backend = "lc_edge"`：LC-SRP-Edge 风格优化实现。

必须验证：

- 与旧 PHAT 输出 shape 完全一致。
- `td_reference` 与 `lc_edge` 的峰值方向误差、map 相关性、RMSAE 差异可量化。
- 前处理耗时、缓存大小、GPU/CPU 内存占用可统计。
- 先在小 batch、小场景验证，再进入完整训练。

交付物：

- SRP 后端对比报告。
- SRP 前处理 profile 表。
- 是否进入 Stage-3 训练的结论。

### 阶段 2：深度可分离卷积轻量化计划

目标：在不改变 IFAN 主干语义的前提下，压缩模型参数量与计算量。

改造顺序：

1. **只改时序卷积**：将融合头中的 `CausConv1d(C,C,k)` 替换为 `DepthwiseConv1d(C,C,k) + PointwiseConv1d(C,C,1)`。
2. **改读出头**：将末端线性/逐点 readout 替换为更边缘友好的 pointwise 形式，保持输出 map 语义不变。
3. **评估 IcoConv 可分离化**：如需进一步压缩，再设计 `DepthwiseIcoConv + PointwiseChannelMixing`，但该步骤风险较高，应单独实验。
4. **全局通道缩放**：在标准 IFAN 与 DW-IFAN 上分别测试 `C=16,12,8,4`。

建议模型版本：

| 版本 | 通道数 | 卷积策略 | 目标用途 |
|---|---:|---|---|
| IFAN-Full | 16 | 标准卷积 | 当前全精度工程基线 |
| IFAN-Edge-L | 16 | 时序深度可分离 | 低风险轻量化 |
| IFAN-Edge-M | 8 或 12 | 时序深度可分离 + 通道缩放 | 精度/资源折中 |
| IFAN-Edge-S | 4 或 8 | 更激进通道缩放，可选 IcoConv 可分离 | 极端边缘探索 |

必须验证：

- 前向输出 shape、梯度、NaN 检查。
- 参数量、MAC proxy、显存占用对比。
- 四场景模拟 RMSAE 与当前主线差异。
- LOCATA Task1/3/5 含静音与去静音结果。

### 阶段 3：输入特征图分辨率统一与消融计划

目标：结合论文 8×16 最优折中结论和 IFAN 二十面体工程实际，明确后续默认输入尺寸策略。

实施原则：

- 当前训练主线维持 `r=2`，不在训练中途切换。
- 轻量化实验单独比较 `r=1`、`r=2`、必要时 `r=3`。
- 如要贴近论文 8×16，应新增规则球面 SRP map 或规则 Elevation×Azimuth 到 Ico map 的适配层，而不是直接把 `r=2` 简化叫作论文 8×16。

建议实验矩阵：

| 输入设置 | 候选/图尺寸含义 | 用途 |
|---|---|---|
| Ico `r=1` | 每 chart 2×4，极低成本 | IFAN-Edge-S 输入探索 |
| Ico `r=2` | 每 chart 4×8，当前工程主线 | 默认基线与主要轻量实验 |
| Ico `r=3` | 每 chart 8×16，更高精度候选 | 上限/饱和性检查 |
| Rule 8×16 | 论文 Cross3D-Edge 推荐分辨率 | 论文对齐专用实验 |

决策建议：

- 若目标是稳定复现 IFAN：默认保留 `r=2`。
- 若目标是最小边缘成本：优先测试 `r=1 + LC-SRP-Edge + DW temporal conv`。
- 若目标是贴合论文输入尺寸：新增 Rule 8×16 分支，不强行改动 IFAN 主线 Ico 输入。

### 阶段 4：组合实验与 IFAN-Edge 系列定型

单项改动通过后，按以下顺序组合：

1. `IFAN-Full`：当前主线基线。
2. `IFAN + LC-SRP-Edge`：只换 PHAT SRP 后端。
3. `IFAN + DW temporal conv`：只换时序卷积。
4. `IFAN + r/input ablation`：只改输入分辨率。
5. `IFAN-Edge-L/M/S`：组合 SRP、DW conv、通道缩放、输入分辨率。

每个组合都输出：

- 配置文件。
- 参数量与 MAC proxy。
- SRP 前处理耗时。
- 单 epoch 训练耗时。
- 四场景模拟 RMSAE。
- LOCATA Task1/3/5 RMSAE。
- 与主线 anchor 的 delta。

---

## 6. 实验与验收指标

### 6.1 当前 IFAN-Full 基线

- 不以论文 Cross3D-Edge 指标直接约束 IFAN-Full。
- 以当前工程已跑出的 Stage-3 anchor 为主线对照。
- 重点比较 IFAN 与本地 IcoCNN baseline、论文 IFAN 表格、LOCATA benchmark2 报告。

### 6.2 SRP 模块验收

| 指标 | 要求 |
|---|---|
| shape | 与旧 PHAT map 完全一致 |
| map 数值 | 峰值方向差异可解释，相关性稳定 |
| 速度 | CPU/GPU 前处理耗时下降或至少不劣化 |
| 缓存 | sinc/TDOA 缓存大小可报告 |
| 精度 | 四场景 RMSAE 不出现系统性退化 |

### 6.3 轻量模型验收

| 版本 | 目标 | 验收重点 |
|---|---|---|
| IFAN-Edge-L | 低风险轻量化 | RMSAE 接近 IFAN-Full，参数/MAC 下降 |
| IFAN-Edge-M | 平衡版 | 明显降参降算，LOCATA 平均退化可控 |
| IFAN-Edge-S | 极限版 | 极低资源，允许更高精度损失但需稳定 |

建议采用相对阈值而非固定绝对阈值：

- 模拟四场景平均 RMSAE 相对主线退化优先控制在 `0.3° ~ 0.5°` 内。
- LOCATA 平均 RMSAE 相对主线退化需单独标注 Task1/3/5，不能只看总体均值。
- 若轻量版在高混响低 SNR 场景明显优于主线，应单独记录为鲁棒性收益。

---

## 7. 风险点与工程注意事项

- **不要覆盖正在训练的主线代码和配置**：轻量化必须新建配置或分支。
- **不要混淆论文 8×16 与 Ico `r=2`**：两者图拓扑不同，只能作为复杂度和候选密度参考。
- **SRP 数值等价性要单独验证**：LC-SRP-Edge 是数学推导明确的优化，但工程实现容易因索引、归一化、麦克风对方向导致偏差。
- **深度可分离 IcoConv 风险高于 Conv1d**：建议先从 temporal conv 开始。
- **通道缩放要全局一致**：残差、注意力、融合头、readout 的通道数必须同步。
- **LOCATA 是最终稳定性门槛**：模拟场景收益不能直接代表真实数据可用。
- **训练预算需区分主线与轻量实验**：轻量模型可能需要重新调学习率和 warmup，不能直接沿用所有 IFAN-Full 结论。

---

## 8. 推荐下一步 TODO

1. 等当前 `long80_freqblock_paper_original` 训练结束，冻结该 run 的 summary、best checkpoint、LOCATA 报告。
2. 新建只读分析报告，对比当前 long80 与 full20 主线差异，判断主线是否收敛稳定。
3. 新建轻量化实验配置，不修改默认主线：先做 `LC-SRP-Edge PHAT backend` 小样本 feature parity。
4. 通过 SRP parity 后，再做 `Depthwise Temporal Conv` 的 forward/profile smoke test。
5. 最后启动小预算组合训练，形成 IFAN-Edge-L/M/S 的初版 tradeoff 表。

