# IFAN-Edge 中期答辩 PPT 大纲

> 建议规模：20 页。  
> 汇报主线：以 `icoCNN` 为 baseline，围绕 `PHAT + LMS` 双特征、注意力融合网络、轻量化边缘候选和 FPGA/HLS 适配形成 IFAN-Edge 工作闭环。  
> 参考定位：IFAN 原论文作为相关工作、结构启发和图表风格参考，不作为本次汇报的主线标题。  
> 结果来源：`IFAN_Edge/outputs/stage3/analysis/locata_four_model_compare.md`、`IFAN_Edge/docs/stage_03_architecture_compare.md`、`相关参考论文/毕业设计最初设想计划.md`。

## 1. 题目页：面向边缘部署的二十面体声源定位网络设计与优化

- 标题建议：面向边缘部署的二十面体声源定位网络设计与优化。
- 副标题：中期检查汇报。
- 信息：姓名、专业、导师、日期。
- 一句话定位：以 icoCNN 为基线，构建双特征注意力融合网络，并探索面向 FPGA 的轻量化部署路径。

图表建议：

- 放一张简化系统图：麦克风阵列 -> PHAT/LMS 特征 -> IFAN-Edge 网络 -> DOA 输出 -> FPGA/HLS 部署。


## 2. 研究背景：多通道声源定位与边缘部署需求

- 声源定位任务目标：从多通道麦克风信号估计声源方向。
- 二十面体网格适合把球面方向估计转化为规则化的局部空间学习问题。
- 边缘端部署要求模型在保持定位精度的同时降低参数量、MAC 和片上资源压力。
- 本课题的核心问题：如何在 icoCNN baseline 上增强特征表达，并进一步形成可映射到 FPGA 的轻量化候选。

图表建议：

- 麦克风阵列 + 球面 DOA + 二十面体网格示意图。


## 3. Baseline 与问题切入：icoCNN 的优势、瓶颈和改进空间

- 本课题的 baseline 明确为 `icoCNN` 网络。
- icoCNN 优势：
  - 已有二十面体卷积与球面定位流程。
  - 作为本地工程中稳定可运行的对照模型。
- icoCNN 局限：
  - 输入以单一 PHAT 特征为主，动态延迟信息利用不足。
  - 主干缺少双特征交互与显式注意力融合。
  - IcoConv 计算量随通道宽度近似二次增长，边缘部署压力较大。
- 改进切入点：
  - 双特征输入。
  - 注意力融合。
  - 通道宽度轻量化。
  - 面向 FPGA 的 ConvIco 数据流组织。

图表建议：

- 表格对比：icoCNN baseline 的已有能力、瓶颈、对应改进方向。


## 4. 本课题工作总览：双特征、融合网络、轻量化、FPGA 映射

- 算法前端：构建 `PHAT + LMS` 双特征二十面体输入。
- 网络结构：从 icoCNN 单特征主干扩展为双分支注意力融合网络。
- 轻量化：提出 `DFA-IcoNet-Edge` 作为边缘轻量化候选模型。
- 实验验证：在模拟场景与 LOCATA 单源任务上比较 baseline、完整宽度模型和轻量化模型。
- 硬件方向：以后续 FPGA/HLS 映射为目标，围绕 IcoConv / ConvIco 数据流和资源闭合展开。

图表建议：

- 四段式路线图：Feature -> Network -> Edge -> Evaluation -> FPGA。

备注：

- 这一页是全篇目录页，算法与硬件都要出现。

## 5. PHAT + LMS 双特征前端

- 已实现双特征前端：
  - `SRPPHATIcoMapAdapter`
  - `SRPLMSIcoMap`
  - `DualFeatureIcoPreprocessor`
- 张量约定：
  - `channel 0 = PHAT`
  - `channel 1 = LMS`
- PHAT 提供鲁棒的相位加权空间响应，LMS 补充动态时延估计信息。
- 双特征共同投影到二十面体网格，作为后续网络输入。

图表建议：

- 多通道音频 -> SRP-PHAT / SRP-LMS -> 二十面体特征图的流程图。

备注：

- 这一页保留现有说明，突出这是相对 icoCNN 单特征输入的前端增强。

## 6. 双特征可视化：二十面体空间响应

- 展示 4 个典型场景的双特征投影结果。
- 已有图片：
  - `IFAN_Edge/outputs/stage1_features/scene_1/feature_maps_projection_contrast.png`
  - `IFAN_Edge/outputs/stage1_features/scene_2/feature_maps_projection_contrast.png`
  - `IFAN_Edge/outputs/stage1_features/scene_3/feature_maps_projection_contrast.png`
  - `IFAN_Edge/outputs/stage1_features/scene_4/feature_maps_projection_contrast.png`
- 讲述重点：前端已经能在二十面体网格上形成可观察、可比较的空间响应。

图表建议：

- 四宫格放 4 张 `feature_maps_projection_contrast.png`。
- 每张图标注 scene 编号，图下注明“用于展示前端响应，不作为最终精度结论”。

备注：

- 这页图比较大，建议单独一页。

## 7. 我们的网络结构设计：从 icoCNN 到双分支注意力融合

- 设计目标：在保留 icoCNN 二十面体卷积能力的基础上，增强双特征融合能力。
- 结构改进：
  - PHAT / LMS 双输入分支。
  - 每个分支加入 residual learning module。
  - 使用 shared attention weight module 建模特征权重。
  - branch-local fusion 后再做 second-stage feature fusion。
  - 融合后进入深层 fusion head 完成方向估计。
- 当前完整宽度精度参考模型：`DFA-IcoNet`。

图表建议：

- 画一张“icoCNN 单主干 -> 双分支注意力融合主干”的结构演化图。
- 可参考 IFAN 原论文结构图风格，但图名写成“本课题双特征注意力融合网络结构”。

备注：

- 不要把该页写成“复现 IFAN 论文结构”，要写成“基于相关工作启发的结构改造与工程实现”。

## 8. 轻量化设计：IcoConv 主瓶颈与 C8_R2 边缘候选

- IcoConv 的参数量与 MAC 对通道宽度敏感，通道宽度下降可显著降低计算量。
- 轻量化策略：
  - 保持二十面体拓扑和时序建模流程。
  - 将主干宽度收缩到 `C=8`。
  - 保持 `r=2` 网格分辨率，避免 `r=3` 带来的 MAC 回升。
- 当前边缘轻量化候选模型：`DFA-IcoNet-Edge`。
- 表述边界：
  - 这是面向 IcoConv 主瓶颈的结构化轻量化与边缘折中设计。
  - 不包装成新的通道裁剪算法或新的网络结构理论。

图表建议：

- 放一个小公式或示意：`MAC_IcoConv ~ Cin * Cout * grid_size`。
- 旁边放模型定位卡片：`DFA-IcoNet = 完整宽度精度参考`，`DFA-IcoNet-Edge = 边缘轻量化候选`。

备注：

- 本页把原先参数量、C8、C8_R3 的部分压缩，为实验和硬件页让出空间。

## 9. 实验设置：模拟实验 + LOCATA 真实数据评测

- 模拟实验：
  - 用于验证训练闭环、收敛趋势和不同声学场景下的模型行为。
  - 可保留 four-scene / hard-scene 指标作为辅助证据。
- LOCATA 评测：
  - subset：`eval`
  - array：`benchmark2`
  - tasks：`task1, task3, task5`
  - available recordings：`task1=13, task3=5, task5=5, total=23`
  - 指标：recording-level RMSAE，分 with silences / without silences。
- 对比模型：
  - `baseline = icoCNN`
  - `DFA-IcoNet = 完整宽度精度参考模型`
  - `DFA-IcoNet-Edge = 边缘轻量化候选模型`

图表建议：

- 放一张紧凑实验设置表：数据集、任务、指标、模型。

备注：

- 训练策略、评测协议和数据口径合并到一页，不再拆成多页。

## 10. LOCATA 总体结果：三模型核心对比

| Model | Params | MAC | With Silences Avg | Without Silences Avg |
| --- | ---: | ---: | ---: | ---: |
| icoCNN baseline | 290017 | - | 8.5718 | 7.1976 |
| DFA-IcoNet | 125457 | 459532800 | 7.2407 | 6.2693 |
| DFA-IcoNet-Edge | 31561 | 115211520 | 7.8581 | 7.0755 |

- `DFA-IcoNet` 相对 icoCNN baseline：
  - with silences average 改善 `1.3310 deg`
  - without silences average 改善 `0.9283 deg`
- `DFA-IcoNet-Edge` 相对 icoCNN baseline：
  - with silences average 改善 `0.7136 deg`
  - without silences average 改善 `0.1221 deg`

图表建议：

- 主图：with / without silences RMSAE 双柱状图，越低越好。
- 角落保留参数量和 MAC 小表。

备注：

- 该页是算法实验核心结果页，建议保留完整数字。

## 11. 分任务结果：Task1 / Task3 / Task5 稳定性观察

- LOCATA 单源任务：
  - Task1：静态或相对简单场景，recording 数最多。
  - Task3 / Task5：更能体现真实数据中场景差异。
- 展示重点：
  - 不只看总体平均，还要看不同任务上的波动。
  - `DFA-IcoNet` 作为完整宽度模型更稳定。
  - `DFA-IcoNet-Edge` 在轻量化后仍保持总体平均优势，但部分任务损失需要解释。

图表建议：

- 每个 task 一组柱状图：icoCNN baseline / DFA-IcoNet / DFA-IcoNet-Edge。
- 指标优先用 mean RMSAE，可分 with silences 和 without silences 两张小图。

备注：

- 这页用于支撑“真实数据上的稳定性”，图会比较大，单独展示。

## 12. Tracking 可视化：真实轨迹与预测轨迹对比

- 目标：参考 IFAN 原论文 tracking 图的表达方式，生成本课题自己的轨迹可视化。
- 优先方案：
  - 从 LOCATA `Task1 / Task3 / Task5` 选 1-2 个 recording。
  - 绘制 ground truth、icoCNN baseline、DFA-IcoNet、DFA-IcoNet-Edge 的时间序列轨迹。
  - 可分别画 azimuth / elevation，或画球面角度误差随时间变化。
- 当前状态：
  - LOCATA 评估已有 recording 级 RMSAE JSON/MD。
  - 现有脚本尚未直接输出 LOCATA tracking PNG，需要后续补充脚本或手动基于预测结果生成。
- 替代方案：
  - 若 LOCATA tracking 图来不及生成，先用 `IFAN_Edge/scripts/analyze_stage3_scene.py` 生成模拟场景 `trajectory_rmsae.png`，作为轨迹误差分布展示。

图表建议：

- 一页放 2 张大图：
  - 图 1：azimuth tracking，ground truth vs models。
  - 图 2：frame-level angular error 或 elevation tracking。

备注：

- 这页要明确“参考论文图风格”，但展示的是本课题结果，不要直接搬论文图作为结果。

## 13. 资源-精度折中：参数量、MAC 与 RMSAE

| Comparison | Params Change | MAC Change | With Silences Avg Delta | Without Silences Avg Delta |
| --- | ---: | ---: | ---: | ---: |
| DFA-IcoNet vs icoCNN baseline | 56.7% reduction | n/a | -1.3310 deg | -0.9283 deg |
| DFA-IcoNet-Edge vs icoCNN baseline | 89.1% reduction | n/a | -0.7136 deg | -0.1221 deg |
| DFA-IcoNet-Edge vs DFA-IcoNet | 74.8% reduction | 74.9% reduction | +0.6174 deg | +0.8062 deg |

- 关键结论：`DFA-IcoNet-Edge` 用约 `75%` 的参数量与 MAC 压缩，换取可接受的平均精度损失。
- 该结果支撑后续 FPGA/HLS 默认以 `DFA-IcoNet-Edge` 作为边缘实现候选。

图表建议：

- Pareto 散点图：
  - 横轴：MAC 或 Params。
  - 纵轴：LOCATA average RMSAE。
  - 点：icoCNN baseline / DFA-IcoNet / DFA-IcoNet-Edge。

备注：

- 这一页承接算法实验到硬件映射的转场。

## 14. 消融与失败参考：非主线结果如何收束

- `IFAN_C8_R3`：
  - 参数量与 `C8_R2` 相同，但 MAC 基本不降。
  - LOCATA 平均退化更明显。
  - 固定为失败参考，不进入候选主线。
- `IFAN_Maba` 等候选分支：
  - 保留已有结果作为参考。
  - 不升为当前默认主线，避免汇报分散。
- 当前收束策略：
  - `DFA-IcoNet`：完整宽度精度参考模型。
  - `DFA-IcoNet-Edge`：边缘轻量化候选模型。
  - 其他模型只作为消融或失败参考。

图表建议：

- 小表或决策矩阵：模型、是否保留、原因、当前定位。

备注：

- 这页不是展示更多模型，而是展示为什么主线已经收束。

## 15. FPGA/HLS 设计目标：为什么算法收束后需要硬件映射

- 边缘部署不只看模型参数，还要看：
  - MAC 压力。
  - 片上缓存。
  - 数据搬运。
  - DSP / BRAM / LUT / FF 资源。
  - 延迟与吞吐。
- 算法收束后的硬件目标：
  - 以 `DFA-IcoNet` 作为精度参考。
  - 以 `DFA-IcoNet-Edge` 作为默认边缘实现候选。
  - 围绕 ConvIco / IcoConv 主瓶颈做数据流与资源优化。

图表建议：

- 算法指标 -> 硬件指标映射图：Params/MAC/RMSAE -> DSP/BRAM/LUT/Latency。

备注：

- 这一页开始进入后续工作，不要讲成已完成整网上板。

## 16. FPGA 整体架构预期：前端、缓存、IcoConv 加速、输出

- 预期系统分层：
  - 输入与特征准备：接收 PHAT/LMS 特征或其片上缓存表示。
  - 几何预处理：二十面体邻接、PadIco、局部窗口组织。
  - ConvIco 加速核心：权重读取、局部 MAC、通道归并。
  - 输出模块：归一化、方向读出、SoftArgMax / 后处理。
- 设计原则：
  - memory-first，先解决数据组织与缓存，再谈算子并行。
  - 用多个可调度小模块，而不是单一不可控大阵列。
  - 让硬件结构匹配二十面体卷积的数据访问特点。

图表建议：

- FPGA 顶层框图：Feature Buffer -> Geometry / PadIco -> ConvIco Engine -> Output Head。

备注：

- 这一页讲“预期架构”，可以用论文式框图风格。

## 17. ConvIco 硬件数据流：PadIco、局部缓冲、权重展开、DSP MAC

- 数据流重点：
  - `PadIco` 与重排映射器：把复杂几何访问规则化。
  - 局部输入缓冲：减少直接访问大数组导致的端口冲突。
  - 紧凑权重表示与展开：服务局部卷积窗口。
  - DSP MAC 阵列：承担热点乘加。
  - output tile 与局部部分和归并：控制累加路径和输出写回。
- 与当前 HLS 文档的对接：
  - layer2-5 共享 ConvIco 块已有 DSP48E1-aware 数据流描述。
  - 后续要把 `DFA-IcoNet-Edge` 的网络规模映射到该类硬件骨架上。

图表建议：

- 参考 `hls_src/layer2-5论文插图增强版架构图-中文.md` 或 `layer2-5_DSP48E1_aware_bilingual_flowchart.md`，重画简化版。

备注：

- 这页是硬件技术核心页，建议放大数据流图。

## 18. 当前 HLS 基础与资源风险

- 已有基础：
  - layer0 / layer1 / layer2-5 的 HLS 工程与资源报告。
  - layer2-5 中 `PadIco` 相关 pipeline 已有 `Final II = 1` 的阶段性记录。
  - 已有定点量化、局部缓冲、DSP-aware 路径分析。
- 资源风险：
  - layer1 仍可能存在 LUT 超限或资源压力。
  - 整网级资源闭合尚未完成。
  - 前端特征生成是否上板仍需明确边界。
- 后续硬件验证重点：
  - `DFA-IcoNet` 与 `DFA-IcoNet-Edge` 的整网预算对比。
  - ConvIco 主路径延迟、资源和精度影响。

图表建议：

- 放 HLS 资源快照表：layer0 / layer1 / layer2-5 的 BRAM、DSP、FF、LUT、Latency。
- 对超限或风险项用颜色标注。

备注：

- 主动说明风险，比把硬件写成已经完成更稳。

## 19. 后续计划：补图、gap 解释、硬件预算、论文撰写

- 算法侧：
  - 补 LOCATA tracking 可视化图。
  - 继续解释完整宽度模型与相关论文最佳结果之间的 gap。
  - 固化 `DFA-IcoNet` 和 `DFA-IcoNet-Edge` 的最终结果表。
- 图表侧：
  - 生成 LOCATA tracking 图。
  - 生成 LOCATA 分任务柱状图。
  - 生成参数量 / MAC / RMSAE Pareto 图。
- 硬件侧：
  - 以 `DFA-IcoNet-Edge` 做整网资源预算。
  - 对接 layer1 / layer2-5 的 HLS 数据流。
  - 明确前端是否进入硬件实现边界。
- 论文侧：
  - 完成算法实验章节。
  - 完成硬件架构与后续实现章节。

图表建议：

- 甘特图或任务矩阵：算法补充、图表生成、HLS 预算、论文撰写。

备注：

- 这一页可以直接作为中期后任务安排。

## 20. 总结页：当前完成度、创新点、下一阶段交付

- 当前完成：
  - 以 icoCNN 为 baseline 的双特征注意力融合网络工程链路。
  - PHAT + LMS 双特征可视化。
  - LOCATA 统一口径下的三模型核心比较。
  - `DFA-IcoNet-Edge` 边缘轻量化候选验证。
- 当前创新表述：
  - 双特征二十面体输入与注意力融合结构的工程实现。
  - 面向 IcoConv 主瓶颈的轻量化边缘折中设计。
  - 面向 FPGA/HLS 的 ConvIco 数据流与资源闭合规划。
- 下一阶段交付：
  - LOCATA tracking 图。
  - 完整实验图表。
  - `DFA-IcoNet-Edge` 整网硬件预算。
  - 毕业论文实验与硬件章节。

图表建议：

- 三栏总结：已完成 / 正在补充 / 下一阶段交付。

备注：

- 最后一页回扣主线：不是单纯复现 IFAN，而是在 icoCNN baseline 上完成 IFAN-Edge 的算法与硬件收束。
