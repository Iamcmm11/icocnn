# DFA-IcoNet-Edge 中期答辩 PPT 生成大纲（优化版）

> 用途：本文件用于直接交给 PPT 生成器，生成一份学术风格、适合现场汇报的硕士中期答辩 PPT。  
> 建议页数：18 页。  
> 汇报主线：以 `icoCNN` 为 baseline，围绕 `PHAT + LMS` 双特征输入、双分支注意力融合网络、边缘轻量化候选模型和 FPGA/HLS 映射规划，形成“算法改进 -> 实验验证 -> 边缘部署”的闭环。  
> 重要限制：PPT 生成器不要自动生成实验相关表格、柱状图、折线图、tracking 图、HLS 资源表。所有标记为【实验图表占位】的地方只需要留出空白区域，并在空白框内标明后续应插入的图片或表格名称。

## 全局生成要求

- 语言：中文。
- 场景：电子信息/嵌入式方向硕士研究生中期答辩。
- 风格：学术、简约、严谨、克制，重点突出结构清晰和信息层级，不做科技展板式装饰。
- 配色：白色背景为主，深蓝仅用于标题、页眉或少量重点强调；正文使用黑色/深灰，辅助线条使用浅灰。
- 装饰限制：不要使用渐变背景、发光效果、复杂纹理、3D 图标、大面积深色背景或强烈科技感元素。
- 版式：每页只保留 1 个核心结论，正文尽量短句化，避免大段文字堆砌。
- 字号：标题清晰，正文适合投影阅读；每页正文不超过 5 个主 bullet。
- 图表规则：流程图、结构图、对比矩阵可以由 PPT 生成器绘制；实验数据图、实验表、资源表必须保留空白占位框。
- 占位框规则：使用浅灰虚线边框，框内写清“此处放入：文件名/图表名/数据来源”，不要生成假数据。
- 术语统一：PPT 展示名使用 `DFA-IcoNet`、`DFA-IcoNet-Edge`、`DFA-IcoNet-Edge-MABA`；实验代号可在小字备注中对应为 `IFAN_80`、`IFAN_C8_R2`、`ifan_c8_r2_maba_pre_readout_best`。

## 模型命名对照

| PPT 展示名 | 实验代号 | 角色定位 |
| --- | --- | --- |
| `icoCNN baseline` | `baseline` | 原始二十面体 DOA 估计基线 |
| `DFA-IcoNet` | `IFAN_80` | 完整宽度精度参考模型 |
| `DFA-IcoNet-Edge` | `IFAN_C8_R2` | 边缘轻量化候选模型 |
| `DFA-IcoNet-Edge-MABA` | `ifan_c8_r2_maba_pre_readout_best` | 轻量化基础上的时序增强扩展 |

## 1. 题目页：面向边缘部署的二十面体声源定位网络设计与优化

### 页面目的

建立答辩主题和技术主线，让评委一眼知道研究对象、算法基础和边缘部署方向。

### 页面内容

- 标题：面向边缘部署的二十面体声源定位网络设计与优化。
- 副标题：硕士研究生中期检查汇报。
- 信息：姓名、专业、导师、学院、日期。
- 一句话定位：以 `icoCNN` 为基线，构建 `PHAT + LMS` 双特征注意力融合网络，并探索面向 FPGA 的轻量化部署路径。

### 版式与占位

- 左侧或居中放标题信息。
- 右侧放一条简化技术链路图：`麦克风阵列 -> PHAT/LMS 特征 -> DFA-IcoNet-Edge -> DOA 输出 -> FPGA/HLS 部署`。
- 该链路图可由 PPT 生成器绘制，不需要实验图片。

## 2. 研究背景：多通道声源定位与边缘部署需求

### 页面目的

说明为什么研究纯 DOA 估计，以及为什么需要考虑边缘部署。

### 页面内容

- 多通道麦克风阵列声源定位是智能感知、机器人听觉和人机交互中的关键基础能力；相比 DCASE/SELD 多任务建模，纯 DOA 估计仍需要高精度、低成本的专项优化。
- 原始 `icoCNN` 的核心选择不是普通平面 CNN，而是把 SRP-PHAT 方向图放到二十面体球面网格上建模。
- 选择二十面体网格的原因：DOA 和 SRP-PHAT 图天然服从球面旋转几何，等角投影存在极点过采样，普通平面 CNN 的平移等变性与问题不完全匹配。
- `icoCNN` 优势：二十面体旋转等变近似球面旋转，IcoConv 可用标准 2D 卷积实现，并通过 SoftArgMax 直接输出 DOA，适合作为低成本纯 DOA baseline。
- 边缘部署不仅关注 RMSAE 精度，还需要控制参数量、MAC、片上缓存和数据搬运。

### 版式与占位

- 左侧：背景问题三段式卡片，分别为“纯 DOA 估计”“复杂声学环境”“边缘部署约束”。
- 右侧：绘制“算法指标 -> 硬件指标”的映射图：`RMSAE / Params / MAC -> DSP / BRAM / LUT / Latency`。
- 该页不需要实验图表占位。

## 3. 问题定义：baseline 瓶颈与本课题切入点

### 页面目的

把问题从宏观背景收束到本课题的具体技术切入点。

### 页面内容

- `icoCNN baseline` 优点：任务边界清晰，面向纯 DOA 估计；二十面体网格更匹配球面方向几何，避免等角投影极点过采样和普通平面卷积的几何错配。
- 特征瓶颈：单一 PHAT 响应在低 SNR、强混响和运动场景下容易出现峰值扩散或伪峰。
- 时序稳定性问题：baseline 已有与 IcoConv 交替的因果 1D 卷积，可利用短时上下文；但在声源/麦克风运动、静音片段和强混响伪峰场景下，读出前特征序列仍可能出现跨帧响应波动，影响 DOA 轨迹稳定性。
- 结构瓶颈：baseline 只处理单一空间响应，不能建模“特征可靠性随环境变化”；简单拼接或相加也难以根据 SNR、混响和运动状态保留可靠峰、抑制误峰。
- 部署瓶颈：IcoConv 计算量随通道宽度近似二次增长，FPGA/HLS 映射存在资源压力。

### 版式与占位

- 使用三列表格：`Baseline 优势`、`主要瓶颈`、`本文改进方向`。
- 页脚加一句边界说明：当前主线聚焦纯 DOA 与网络后端硬件映射，不宣称完成完整音频前端全链路 FPGA 实现。
- 该页不需要实验图表占位。

## 4. 研究目标与技术路线总览

### 页面目的

给出全文逻辑地图，让后续算法、实验和硬件部分形成闭环。

### 页面内容

- 目标 1：构建 `PHAT + LMS` 双特征二十面体输入，补强复杂声学环境下的空间响应表达。
- 目标 2：设计双分支残差增强与共享注意力融合网络，形成 `DFA-IcoNet`。
- 目标 3：围绕 IcoConv 主瓶颈进行规则稠密宽度收缩，形成 `DFA-IcoNet-Edge`。
- 目标 4：在已有局部时序卷积基础上，引入 `pre_readout MABA` 作为轻量化网络的时序增强扩展。
- 目标 5：面向 FPGA/HLS 规划 ConvIco 数据流、缓存结构和资源闭合路径。

### 版式与占位

- 绘制六段式技术路线图：`Problem -> Feature -> Network -> Edge -> MABA -> Hardware`。
- 下方用两条验证线标注：`模拟四场景 = 声学环境鲁棒性`，`LOCATA Task1/3/5 = 运动状态适应性`。
- 该页不需要实验图表占位。

## 5. 双特征前端：PHAT + LMS 二十面体输入

### 页面目的

说明为什么不是只用 PHAT，而是引入 LMS 作为互补特征。

### 页面内容

- `PHAT`：在高 SNR、低混响场景下峰值更清晰，是稳健的空间定位基准线索。
- `LMS`：来自自适应滤波的时延估计，在大混响或低 SNR 场景中可能保留与 DOA 相关的另一类证据，但复杂混响下也可能误估。
- 因此 LMS 不是替代 PHAT，而是补充 PHAT 在不同声学条件下失稳时的互补空间证据。
- 双特征统一映射到二十面体网格，形成两通道输入：`channel 0 = PHAT`，`channel 1 = LMS`。
- 已实现模块：`SRPPHATIcoMapAdapter`、`SRPLMSIcoMap`、`DualFeatureIcoPreprocessor`。

### 版式与占位

- 左侧绘制前端流程图：`多通道音频 -> PHAT 分支 / LMS 分支 -> 二十面体投影 -> 双通道特征图`。
- 右侧留出 2 个小图占位框。
- 【实验图表占位】右上框：此处放入 `IFAN_Edge/outputs/stage1_features/scene_1/feature_maps_projection_contrast.png`。
- 【实验图表占位】右下框：此处放入 `IFAN_Edge/outputs/stage1_features/scene_4/feature_maps_projection_contrast.png`。
- 占位框只需标明图片路径，不要让 PPT 生成器自动绘制热力图。

## 6. 网络结构：双分支残差增强与注意力融合

### 页面目的

展示 `DFA-IcoNet` 的核心结构创新，并和 baseline 的单特征主干形成对比。

### 页面内容

- 从单特征 `icoCNN` 扩展为 PHAT/LMS 双输入分支。
- 每个分支保留直通特征，并通过 residual learning module 得到增强特征。
- 共享 attention weight module 根据输入声学环境学习特征权重，避免对两类响应做固定比例融合。
- 两级融合后进入深层 fusion head，最后通过 `CleanVertices -> SoftArgMax` 输出 DOA。
- 完整宽度模型 `DFA-IcoNet` 作为精度参考模型。

### 版式与占位

- 中央绘制网络结构图：`PHAT branch` 与 `LMS branch` 左右并行，经过 residual、attention、fusion head 后输出。
- 右下角放小注释：实验代号 `IFAN_80`。
- 该页结构图可由 PPT 生成器绘制，不需要实验图表占位。

## 7. 边缘轻量化：IcoConv 主瓶颈与 C8_R2 设计

### 页面目的

解释 `DFA-IcoNet-Edge` 为什么采用规则稠密宽度收缩，而不是把重点放在不规则稀疏剪枝。

### 页面内容

- IcoConv 的计算量近似满足：`MAC_IcoConv ~ Cin * Cout * grid_size`。
- 因此主干通道宽度是影响参数量和 MAC 的关键变量。
- 轻量化策略：保留双特征输入和二十面体拓扑，在融合后通过 `PoolIco: r=2 -> r=1` 降低后端空间计算压力。
- 将主干宽度从 `C=16` 收缩到 `C=8`，使核心计算块更接近单个规则 `8 x 8 DSP tile`。
- 当前边缘候选模型为 `DFA-IcoNet-Edge`，实验代号 `IFAN_C8_R2`。

### 版式与占位

- 上方绘制三段式轻量化图：`双特征融合 at r=2 -> PoolIco r=2 to r=1 -> C16 to C8 dense slimming`。
- 下方绘制对比矩阵：`C16 dense`、`SAF-lite 2-of-8`、`C8_R2 dense`。
- 该页表格是结构解释表，可由 PPT 生成器绘制，不属于实验结果表。

## 8. MABA 时序增强：pre_readout 插入位置

### 页面目的

说明为什么在轻量化网络中加入 MABA，以及为什么选择 `pre_readout` 位置。

### 页面内容

- `DFA-IcoNet-Edge` 仍保留 IcoConv 后的因果 1D 卷积，但轻量化后在低 SNR、高混响和动态场景下可能出现跨帧响应不稳定。
- `pre_readout MABA` 在 `final_block` 之后、`channel_readout` 之前进行 feature 级时序重整，用状态扫描补充固定窗口 1D 卷积的局部建模。
- 该位置仍保留多通道和多 region 信息，能在读出前抑制跨帧尖峰和伪峰。
- 当前确定扩展模型为 `DFA-IcoNet-Edge-MABA`。
- MABA 是可选时序增强扩展，后续硬件映射需单独评估其资源代价。

### 版式与占位

- 中央绘制插入位置流程图：`Final block -> pre_readout MABA -> channel_readout -> region max -> CleanVertices -> SoftArgMax`。
- 右侧用小模块图展示 MABA 内部：`Linear In -> Temporal Conv -> Gate/State Scan -> Linear Out -> Residual`。
- 该页结构图可由 PPT 生成器绘制，不需要实验图表占位。

## 9. 实验设置：模拟四场景 + LOCATA 真实数据

### 页面目的

统一说明实验口径，避免评委对数据来源和评价指标产生疑问。

### 页面内容

- 模拟四场景：用于验证不同 SNR/T60 条件下的声学环境鲁棒性。
- 四种场景：`30dB/T60=0.2s`、`30dB/T60=0.8s`、`5dB/T60=0.8s`、`5dB/T60=1.4s`。
- LOCATA 评测口径：`eval / benchmark2 / task1, task3, task5`。
- 可用 recording：`task1=13`，`task3=5`，`task5=5`，合计 `23`。
- 指标：recording-level RMSAE，分别统计 with silences 与 without silences。

### 版式与占位

- 左侧绘制“数据集与任务”信息卡。
- 右侧绘制“模型对比对象”列表：`icoCNN baseline`、`DFA-IcoNet`、`DFA-IcoNet-Edge`、`DFA-IcoNet-Edge-MABA`。
- 该页是实验设置说明表，可由 PPT 生成器绘制，不属于实验结果表。

## 10. 核心结果：LOCATA 总体性能对比

### 页面目的

展示最重要的实验结论：完整模型提升精度，边缘模型大幅降规模后仍保持平均优势。

### 页面内容

- `DFA-IcoNet` 相对 `icoCNN baseline`：with silences average 改善 `1.3310 deg`，without silences average 改善 `0.9283 deg`。
- `DFA-IcoNet-Edge` 相对 `icoCNN baseline`：with silences average 改善 `0.7136 deg`，without silences average 改善 `0.1221 deg`。
- `DFA-IcoNet-Edge-MABA` 相对 `DFA-IcoNet-Edge`：without silences average 进一步改善 `0.1625 deg`。
- 结论：`DFA-IcoNet-Edge` 是当前边缘候选主线，`MABA` 是小资源增量下的时序增强扩展。

### 版式与占位

- 【实验图表占位】页面中央偏上放一张大表占位框：此处放入“LOCATA 四模型总体结果表”。
- 占位框内标明数据来源：`IFAN_Edge/outputs/stage3/analysis/locata_four_model_compare.md`。
- 【实验图表占位】页面中央偏下放柱状图占位框：此处放入“with / without silences RMSAE 双柱状图，四个模型对比”。
- PPT 生成器不要自动绘制表格或柱状图，只保留两个占位框。

## 11. 分任务结果：Task1 / Task3 / Task5 稳定性观察

### 页面目的

说明模型在不同真实任务上的表现差异，突出 Task5 中 MABA 的价值。

### 页面内容

- Task1：相对简单，recording 数最多，适合观察总体稳定性。
- Task3 / Task5：更能体现真实场景中的运动状态和声学差异。
- `DFA-IcoNet` 作为完整宽度模型整体更稳定。
- `DFA-IcoNet-Edge` 轻量化后仍保持总体平均优势，但部分任务存在波动。
- `DFA-IcoNet-Edge-MABA` 的收益主要集中在 Task5：with silences Task5 mean 从 `11.8516` 降至 `11.2159`，without silences Task5 mean 从 `9.9599` 降至 `8.5478`。

### 版式与占位

- 【实验图表占位】整页采用 2 x 1 图表布局。
- 上方占位框：此处放入“LOCATA Task1/Task3/Task5 with silences mean RMSAE 分任务柱状图”。
- 下方占位框：此处放入“LOCATA Task1/Task3/Task5 without silences mean RMSAE 分任务柱状图”。
- 占位框内标明数据来源：`IFAN_Edge/outputs/stage3/analysis/locata_four_model_compare.md`。
- PPT 生成器不要自动绘制分任务柱状图。

## 12. MABA 消融结果：时序增强的收益边界

### 页面目的

说明 MABA 的有效位置和收益边界，避免把 MABA 讲成所有任务无条件提升的模块。

### 页面内容

- `pre_readout MABA` 在读出前保留多通道、多 region 弱证据，更适合抑制复杂场景下的伪峰。
- 模拟 `scene_4 (5dB/T60=1.4s)` 中，RMSAE 从 `17.9796 deg` 降到 `13.9095 deg`。
- 模拟四场景 mean 从 `9.9787 deg` 降到 `8.5406 deg`。
- hard mean 从 `14.2679 deg` 降到 `11.6980 deg`。
- 结论：MABA 更像 hard-scene / 动态场景增强模块，而不是所有任务的均匀增益模块。

### 版式与占位

- 左侧放 `pre_readout MABA` 插入位置小流程图，可由 PPT 生成器绘制。
- 【实验图表占位】右侧放消融结果表占位框：此处放入“DFA-IcoNet-Edge / map_maba / pre_readout / dual_refine 在 scene_4、four-scene mean、hard mean 和 LOCATA average 上的对比表”。
- 占位框内标明数据来源：`IFAN_Edge/outputs/stage3/analysis/locata_four_model_compare.md`。
- PPT 生成器不要自动生成消融表。

## 13. 资源-精度折中：Params、MAC 与 RMSAE

### 页面目的

把算法结果自然过渡到边缘部署，突出 `DFA-IcoNet-Edge` 的取舍价值。

### 页面内容

- `DFA-IcoNet` 相对 baseline：参数量减少 `56.7%`，且 LOCATA 平均精度提升。
- `DFA-IcoNet-Edge` 相对 `DFA-IcoNet`：参数量减少 `74.8%`，MAC 减少 `74.9%`。
- `DFA-IcoNet-Edge` 相对 baseline：参数量减少 `89.1%`，同时 LOCATA 平均仍优于 baseline。
- `DFA-IcoNet-Edge-MABA` 仅增加约 `2.5%` 参数和 `0.9%` MAC，带来一定平均精度补偿。
- 结论：后续 FPGA/HLS 默认以 `DFA-IcoNet-Edge` 为主线，MABA 作为可选扩展评估。

### 版式与占位

- 【实验图表占位】左侧放资源-精度对比表占位框：此处放入“Params / MAC / RMSAE trade-off 表”。
- 【实验图表占位】右侧放 Pareto 散点图占位框：此处放入“横轴 MAC 或 Params，纵轴 LOCATA average RMSAE，四模型 Pareto 图”。
- 占位框内标明数据来源：`IFAN_Edge/outputs/stage3/analysis/locata_four_model_compare.md`。
- PPT 生成器不要自动生成 Pareto 图或资源表。

## 14. Tracking 可视化：真实轨迹与预测轨迹对比

### 页面目的

用直观轨迹图辅助说明模型输出的时间连续性和真实数据表现。

### 页面内容

- 计划从 LOCATA `Task1 / Task3 / Task5` 中选择典型 recording。
- 绘制 ground truth、baseline、`DFA-IcoNet`、`DFA-IcoNet-Edge`、`DFA-IcoNet-Edge-MABA` 的时间序列轨迹。
- 优先展示 azimuth tracking 与 frame-level angular error。
- 若 LOCATA tracking 图尚未生成，可暂用模拟场景 `trajectory_rmsae.png` 作为替代展示。
- 该页用于增强直观性，不替代第 10-13 页的定量结论。

### 版式与占位

- 【实验图表占位】上方大图框：此处放入“LOCATA azimuth tracking，ground truth vs models”。
- 【实验图表占位】下方大图框：此处放入“frame-level angular error 或 elevation tracking”。
- 若未生成 LOCATA tracking 图，框内改写为：此处暂放模拟场景 `trajectory_rmsae.png`。
- PPT 生成器不要自动绘制轨迹曲线。

## 15. FPGA/HLS 设计目标：从算法收束到硬件映射

### 页面目的

说明为什么中期后半部分要进入硬件映射，并明确硬件工作边界。

### 页面内容

- 边缘部署不只看模型大小，还要看 DSP、BRAM、LUT、FF、延迟和吞吐。
- `DFA-IcoNet` 作为精度参考，`DFA-IcoNet-Edge` 作为默认硬件映射候选。
- 当前硬件重点放在二十面体网络后端 ConvIco / IcoConv，不宣称完整音频前端全链路上板。
- 硬件设计围绕三个问题展开：数据组织、局部缓存、DSP MAC 复用。
- 后续需要完成整网级资源预算和关键层 HLS 验证。

### 版式与占位

- 左侧绘制算法到硬件的映射箭头：`DFA-IcoNet-Edge -> ConvIco dataflow -> HLS resource budget -> FPGA deployment`。
- 右侧列出硬件评价指标：`DSP / BRAM / LUT / FF / Latency / II`。
- 该页不需要实验图表占位。

## 16. ConvIco 硬件数据流与当前 HLS 基础

### 页面目的

展示硬件侧的技术核心，以及当前已有基础和风险。

### 页面内容

- ConvIco 数据流重点：`PadIco`、局部输入缓冲、权重展开、DSP MAC、output tile 归并。
- `PadIco` 与重排映射器用于把复杂二十面体几何访问规则化。
- 局部缓冲减少大数组访问导致的端口冲突。
- DSP-aware MAC 路径承担主要乘加计算。
- 当前已有 layer0 / layer1 / layer2-5 的 HLS 工程与阶段性资源报告，但整网级资源闭合尚未完成。

### 版式与占位

- 左侧绘制 ConvIco 数据流框图：`Feature Buffer -> Geometry/PadIco -> Local Buffer -> DSP MAC -> Output Head`。
- 【实验图表占位】右侧放 HLS 资源快照表占位框：此处放入“layer0 / layer1 / layer2-5 的 BRAM、DSP、FF、LUT、Latency 资源表”。
- 占位框内可标明参考来源：`hls_src/hls_reports/latest_summary.md`、`hls_src/hls_reports/layer1_latest_summary.md`、`hls_src/hls_reports/layer2_5_latest_summary.md`。
- PPT 生成器不要自动生成 HLS 资源数据表。

## 17. 后续计划：补图、资源预算与论文撰写

### 页面目的

给出中期之后的工作安排，让评委看到可执行的收尾路径。

### 页面内容

- 算法侧：固化最终模型口径，补充 LOCATA tracking 可视化，继续解释与相关论文最好结果之间的 gap。
- 图表侧：生成分任务柱状图、资源-精度 Pareto 图、MABA 消融表和 tracking 图。
- 硬件侧：以 `DFA-IcoNet-Edge` 做整网资源预算，对接 layer1 / layer2-5 HLS 数据流。
- 论文侧：完成算法实验章节，补充硬件架构与后续实现章节。
- 风险控制：明确前端是否进入硬件实现边界，避免把未完成工作表述成已完成。

### 版式与占位

- 使用四象限任务矩阵：`算法补充`、`图表生成`、`HLS 预算`、`论文撰写`。
- 也可使用横向时间线：`中期后 1-2 周`、`3-4 周`、`5-6 周`、`论文定稿前`。
- 该页不需要实验图表占位。

## 18. 总结页：完成度、创新点与下一阶段交付

### 页面目的

回扣主线，形成答辩结束页。

### 页面内容

- 已完成：`PHAT + LMS` 双特征前端、双分支注意力融合网络、LOCATA 统一口径评测、边缘轻量化候选验证。
- 当前结果：`DFA-IcoNet` 提升定位精度，`DFA-IcoNet-Edge` 在大幅压缩后仍保持平均优势。
- 创新点 1：双特征二十面体输入与注意力融合结构。
- 创新点 2：面向 IcoConv 主瓶颈的规则稠密轻量化设计。
- 创新点 3：面向 FPGA/HLS 的 ConvIco 数据流与资源闭合规划。
- 下一阶段交付：完整实验图表、tracking 可视化、整网 HLS 资源预算、毕业论文实验与硬件章节。

### 版式与占位

- 使用三栏总结：`已完成`、`当前创新`、`下一阶段交付`。
- 页脚回扣一句：本课题不是单纯复现 IFAN，而是在 `icoCNN` baseline 上完成面向边缘部署的算法与硬件收束。
- 该页不需要实验图表占位。

## 实验图表占位索引

| 页码 | 占位内容 | 建议位置 | 来源或后续插入内容 |
| --- | --- | --- | --- |
| 5 | 双特征投影图 | 右侧上下两个小图框 | `IFAN_Edge/outputs/stage1_features/scene_1/feature_maps_projection_contrast.png`，`scene_4/feature_maps_projection_contrast.png` |
| 10 | LOCATA 四模型总体结果表 | 页面中央偏上 | `IFAN_Edge/outputs/stage3/analysis/locata_four_model_compare.md` |
| 10 | with / without silences 双柱状图 | 页面中央偏下 | 后续根据 LOCATA 结果生成图片 |
| 11 | 分任务 RMSAE 柱状图 | 上下两张大图 | 后续根据 LOCATA Task1/3/5 结果生成图片 |
| 12 | MABA 消融结果表 | 右侧大表框 | `IFAN_Edge/outputs/stage3/analysis/locata_four_model_compare.md` |
| 13 | 资源-精度 trade-off 表 | 左侧 | `IFAN_Edge/outputs/stage3/analysis/locata_four_model_compare.md` |
| 13 | Params/MAC-RMSAE Pareto 图 | 右侧 | 后续根据四模型数据生成图片 |
| 14 | LOCATA tracking 图 | 上下两张大图 | 后续生成 azimuth/elevation 或 frame-level error 图片 |
| 16 | HLS 资源快照表 | 右侧 | `hls_src/hls_reports/latest_summary.md` 等 HLS 报告 |

## 给 PPT 生成器的最终提示词
