# DFA-IcoNet-Edge 中期答辩 PPT 大纲（优化版）

> 建议规模：15 页。  
> 使用场景：硕士研究生中期答辩现场汇报。  
> 汇报主线：以 `icoCNN` 为 baseline，围绕 `PHAT + LMS` 双特征输入、注意力融合网络、轻量化边缘候选和 FPGA/HLS 映射规划，形成“算法改进 -> 实验验证 -> 硬件映射”的闭环。  
> 风格要求：简约学术风，白底为主，深蓝仅用于标题和重点强调，正文使用黑色/深灰，辅助线条使用浅灰；不要科技风装饰、渐变背景、发光效果、复杂纹理或大面积深色背景。  
> 图表原则：流程图、结构示意图和框图可以由 PPT 生成器绘制；实验曲线、真实结果图、tracking 图、HLS 资源表必须使用已有图片或留空占位，不要生成虚构数据。

## 0. 模型命名与汇报口径

| PPT 展示名 | 实验代号 | 汇报定位 |
| --- | --- | --- |
| `icoCNN baseline` | `baseline` | 原始二十面体 DOA 估计基线 |
| `DFA-IcoNet` | `IFAN_80` | 完整宽度精度参考模型 |
| `DFA-IcoNet-Edge` | `IFAN_C8_R2` | 边缘轻量化候选模型 |
| `DFA-IcoNet-Edge-MABA` | `ifan_c8_r2_maba_pre_readout_best` | 轻量化基础上的时序增强扩展 |

答辩边界需要主动说明：

- 本课题聚焦纯 DOA 估计，不宣称覆盖完整 SELD 多任务系统。
- 当前硬件部分聚焦网络后端 ConvIco / IcoConv 的 HLS 映射，不宣称已经完成完整音频前端全链路上板。
- IFAN 原论文作为相关工作、结构启发和图表风格参考，不作为本次汇报标题主线。

## 1. 题目页：面向边缘部署的二十面体声源定位网络设计与优化

### 页面目的

明确课题名称、答辩身份和一句话研究定位。

### 页面内容

- 标题：面向边缘部署的二十面体声源定位网络设计与优化。
- 副标题：中期检查汇报。
- 信息：姓名、专业、导师、日期。
- 一句话定位：以 `icoCNN` 为基线，构建双特征注意力融合网络，并探索面向 FPGA 的轻量化部署路径。

### 版式建议

- 白底简约封面，标题居中或左对齐。
- 右下或下方放一条细线式系统链路图：`麦克风阵列 -> PHAT/LMS 特征 -> DFA-IcoNet-Edge -> DOA 输出 -> FPGA/HLS 部署`。
- 链路图使用线框和箭头即可，不要做科技感背景。

## 2. 研究背景：多通道声源定位与边缘部署需求

### 页面目的

说明本课题为什么选择纯 DOA 估计，以及为什么需要考虑边缘部署。

### 页面内容

- 多通道声源定位是机器人听觉、智能会议、人机交互等场景中的基础感知能力。
- 近年 DCASE/SELD 类任务更偏向检测、分类、定位联合建模，但纯 DOA 估计仍有专项优化空间。
- 本课题选择在二十面体空间建模框架下研究方位估计，强调可解释的空间结构和定位精度。
- 边缘端不仅关注定位误差，还需要控制参数量、MAC、片上缓存和数据搬运压力。
- 核心问题：如何在 `icoCNN baseline` 上增强复杂声学环境中的方位估计稳定性，并形成更适合硬件映射的轻量化网络后端。

### 版式建议

- 左侧：三点背景逻辑，分别为“纯 DOA 估计”“复杂声学环境”“边缘部署约束”。
- 右侧：验证矩阵示意图：`模拟四场景 = 声学环境鲁棒性`，`LOCATA Task1/3/5 = 运动状态适应性`。
- 该页为背景说明，不放实验结果图。

## 3. Baseline 与问题切入：icoCNN 的优势、瓶颈和改进空间

### 页面目的

把研究问题从背景收束到 `icoCNN` baseline 的具体瓶颈。

### 页面内容

- 选择 `icoCNN` 的原因：面向纯 DOA 估计，任务边界清晰，二十面体拓扑与空间方向估计目标一致。
- 特征瓶颈：单一 PHAT 响应在低 SNR、强混响和运动场景下容易出现峰值扩散或伪峰。
- 时序瓶颈：声源或麦克风运动会带来跨帧时延变化，单帧响应难以显式保留动态信息。
- 结构瓶颈：baseline 缺少面向互补特征的自适应融合机制。
- 部署瓶颈：IcoConv 计算量随输入/输出通道宽度近似二次增长，边缘部署资源压力较大。

### 版式建议

- 使用三列表格：`baseline 优势`、`主要瓶颈`、`对应改进方向`。
- 页脚放一句答辩口径：当前主线聚焦纯 DOA 与网络后端硬件映射闭环。

## 4. 本课题工作总览：双特征、融合网络、轻量化、FPGA 映射

### 页面目的

给评委一个总览图，建立后续页面的逻辑顺序。

### 页面内容

- 特征补强：构建 `PHAT + LMS` 双特征二十面体输入。
- 网络设计：从单特征 `icoCNN` 扩展为双分支残差增强与注意力融合网络，形成 `DFA-IcoNet`。
- 边缘轻量化：围绕 IcoConv 主瓶颈进行宽度收缩，形成 `DFA-IcoNet-Edge`。
- 时序增强：在轻量化主干上引入 `pre_readout MABA`，形成 `DFA-IcoNet-Edge-MABA`。
- 硬件映射：面向 ConvIco / IcoConv 规划 HLS 数据流、缓存和资源闭合路径。

### 版式建议

- 绘制六段式路线图：`Problem -> Feature -> Network -> Edge -> MABA -> Hardware`。
- 下方标注两条验证线：`模拟四场景` 与 `LOCATA Task1/3/5`。
- 该页只做总览，不放具体实验结果。

## 5. 算法主体设计：PHAT+LMS 双特征与注意力融合网络

### 页面目的

展示算法主体结构，说明 `DFA-IcoNet` 相对 baseline 的核心变化。

### 页面内容

- 前端输入由单一 PHAT 扩展为 `PHAT + LMS` 双特征输入。
- `PHAT` 提供相位加权空间响应，`LMS` 补充跨帧动态时延估计信息。
- 双特征统一映射到二十面体网格：`channel 0 = PHAT`，`channel 1 = LMS`。
- 网络主体采用 PHAT / LMS 双分支、residual learning、shared attention weight module 和 second-stage fusion。
- 融合特征进入深层 fusion head，最后通过 `CleanVertices -> SoftArgMax` 输出 DOA。

### 版式建议

- 中央放算法结构图，图名写为“本课题双特征注意力融合网络结构”。
- 图片使用你新大纲中已经插入的算法结构图：

![算法结构图](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NjM0MDY3Yzk0YmQwN2Q2ZjI3YzFlNTM3YWVlYjY4YjVfZWNiZjI4YTViNWU4ZDAzMzVmYWI4NzIxODViNjVmZDRfSUQ6NzY0ODYyODQxMzM2Nzk3OTE5M18xNzgxNTk1OTI1OjE3ODE2ODIzMjVfVjM)

- 若图片无法被 PPT 生成器读取，则保留一块图片占位框，标注“此处插入双特征注意力融合网络结构图”。

## 6. 轻量化设计：IcoConv 主瓶颈与 C8_R2 边缘候选

### 页面目的

说明 `DFA-IcoNet-Edge` 的轻量化设计不是简单缩小网络，而是面向 IcoConv 主瓶颈和硬件执行结构的规则压缩。

### 页面内容

- IcoConv 的主要计算压力来自通道维度，近似满足：`MAC_IcoConv ~ Cin * Cout * grid_size`。
- 前端 `PHAT + LMS` 双特征需要较高空间分辨率完成互补信息交互，因此融合前不宜过早压缩。
- 在 `r=2` 完成双分支融合后，通过 `PoolIco` 将 Fusion Feature 降到 `r=1`，降低后端空间计算压力。
- 主干宽度从 `C=16` 收缩到 `C=8`，使核心 IcoConv 计算块更接近规则 `8 x 8` 稠密 tile。
- 当前边缘轻量化候选模型为 `DFA-IcoNet-Edge`。

### 版式建议

- 左侧放轻量化路径图：`双特征融合 at r=2 -> PoolIco r=2 to r=1 -> C16 to C8 dense channel slimming`。
- 右侧放“为什么不以稀疏剪枝为主线”的简短对照：
  - SAF-lite 稀疏剪枝理论压缩高；
  - 但需要索引、块内位置记录和不规则调度；
  - 当前验证退化明显，未形成可用主线；
  - 因此回到规则 dense channel slimming。
- 该页可以生成结构示意图，不需要实验图。

## 7. 实验设置与总体结果入口：模拟实验 + LOCATA 真实数据评测

### 页面目的

统一实验口径，并展示 with silences / without silences 两类核心结果图的位置。

### 页面内容

- 模拟四场景用于验证不同 SNR/T60 条件下的声学环境鲁棒性。
- 场景组合：`30dB/T60=0.2s`、`30dB/T60=0.8s`、`5dB/T60=0.8s`、`5dB/T60=1.4s`。
- LOCATA 评测口径：`eval / benchmark2 / task1, task3, task5`。
- 可用 recording：`task1=13`，`task3=5`，`task5=5`，总计 `23`。
- 指标：recording-level RMSAE，分别统计 with silences 与 without silences。

### 版式建议

- 上半部分放紧凑实验设置表：数据集、任务、模型、指标。
- 下半部分放两个真实结果图位置：
  - 左侧：`With Silences` 图表。
  - 右侧：`Without Silences` 图表。
- 如果图表尚未插入，只保留空白占位框，不要生成虚构柱状图或表格。

## 8. Tracking 可视化：真实轨迹与预测轨迹对比

### 页面目的

用直观轨迹图说明模型在真实序列中的预测趋势和误差波动。

### 页面内容

- 参考 IFAN 原论文 tracking 图的表达方式，但展示本课题自己的结果。
- 从 LOCATA `Task1 / Task3 / Task5` 中选择 1-2 个典型 recording。
- 绘制 ground truth、`icoCNN baseline`、`DFA-IcoNet`、`DFA-IcoNet-Edge`、`DFA-IcoNet-Edge-MABA` 的时间序列轨迹。
- 优先展示 azimuth tracking 与 frame-level angular error。
- 若 LOCATA tracking 图尚未完全生成，可用模拟场景 `trajectory_rmsae.png` 作为临时替代图。

### 版式建议

- 一页放两张大图，上下或左右排布。
- 使用你新大纲中已插入的两张 tracking 图：

![Tracking 图 1](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MjI4ZDI0YmRjMWQ3NmI2OTVlYzI3OWJmNTg0NDY1MGFfMjc0NGNlZGFhNTVmMjEwN2JmZGUyMTRlNjNhNWY5ZWNfSUQ6NzY1MTY0NzI4MjUyOTUzNjk2NF8xNzgxNTk1OTI1OjE3ODE2ODIzMjVfVjM)

![Tracking 图 2](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NDcxNWI3OTFmZDVmMGMzZDE0YjRmY2UxMzJhNjQ3YjlfNTA4Y2FmNzkzMGI0Mzc1MWZjYTY2NjBhYjA3ZmY2NzNfSUQ6NzY1MTY0NzMyNzYwNTY0MDQzMF8xNzgxNTk1OTI1OjE3ODE2ODIzMjVfVjM)

- 页脚注明：该页是直观可视化，定量结论以 RMSAE 结果为准。

## 9. 资源-精度折中：参数量、MAC 与 RMSAE

### 页面目的

把算法实验自然过渡到边缘部署，突出 `DFA-IcoNet-Edge` 的取舍价值。

### 页面内容

- `DFA-IcoNet-Edge` 用约 `75%` 的参数量与 MAC 压缩，换取可接受的平均精度损失。
- `DFA-IcoNet-Edge` 相对 baseline 仍保持 LOCATA 平均优势，是当前默认边缘候选。
- `DFA-IcoNet-Edge-MABA` 在极小资源增量下带来一定平均精度补偿，尤其对 hard-scene / Task5 更有价值。
- 后续 FPGA/HLS 默认以 `DFA-IcoNet-Edge` 为基础网络，单独评估 MABA refiner 是否纳入扩展。

### 版式建议

- 主图建议为 Pareto 散点图：横轴为 Params 或 MAC，纵轴为 LOCATA average RMSAE。
- 标出四个点：`icoCNN baseline`、`DFA-IcoNet`、`DFA-IcoNet-Edge`、`DFA-IcoNet-Edge-MABA`。
- 如果 Pareto 图尚未生成，只保留图表占位框，不要让 PPT 生成器根据文字自动造图。

## 10. Pre-readout MABA 时序增强与消融收束

### 页面目的

说明 MABA 加入的意义、插入位置和收益边界。

### 页面内容

- `DFA-IcoNet-Edge` 已完成主干轻量化，但低 SNR、高混响和动态场景下仍可能出现跨帧响应不稳定。
- `pre_readout MABA` 插入在 `final_block` 之后、`channel_readout` 之前。
- 该位置的输入仍保留 channel 和 region 信息，适合在读出前进行 feature 级时序重整。
- 模拟 `scene_4 (5dB/T60=1.4s)` 中，`pre_readout` 将 RMSAE 从 `17.9796 deg` 降到 `13.9095 deg`。
- 结论：MABA 更适合作为 hard-scene / 动态场景增强模块，而不是所有任务的均匀增益模块。

### 版式建议

- 上方或左侧放单一插入位置流程：`Final block -> pre_readout MABA -> channel_readout -> region max -> CleanVertices -> SoftArgMax`。
- 右侧或下方放你新大纲中已插入的 MABA 相关图片：

![MABA 相关图](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=Yjk0ZTI1MDUwYTI4MzRiMzAxZjRhZWVlYTgwYzZlNGZfY2RlOTQ0Mzc3N2NjODUzMTcyYWNhNzJhYjM2YmQ5ZGFfSUQ6NzY1MTY0OTE2Njc1Nzc2MDIxMV8xNzgxNTk1OTI1OjE3ODE2ODIzMjVfVjM)

- 下方可保留两个小图表位：`With Silences` 与 `Without Silences`，用于插入真实对比结果。
- 不要把 MABA 讲成两个位置同时加入的主流程。

## 11. FPGA/HLS 设计目标：为什么算法收束后需要硬件映射

### 页面目的

说明算法轻量化之后为什么仍需要硬件映射分析。

### 页面内容

- 边缘部署不只看模型参数，还需要看 MAC 压力、片上缓存、数据搬运、DSP/BRAM/LUT/FF 资源、延迟和吞吐。
- `DFA-IcoNet` 作为精度参考模型。
- `DFA-IcoNet-Edge` 作为默认边缘实现候选。
- `DFA-IcoNet-Edge-MABA` 作为可选时序增强扩展，后续单独评估其资源和延迟影响。
- 硬件侧重点围绕 ConvIco / IcoConv 主瓶颈做数据流与资源优化。

### 版式建议

- 绘制指标映射图：`Params / MAC / RMSAE -> DSP / BRAM / LUT / Latency / Throughput`。
- 该页是算法到硬件的转场页，避免写成“整网上板已完成”。

## 12. FPGA 整体架构预期：前端、缓存、IcoConv 加速、输出

### 页面目的

给出预期硬件系统分层，说明后续硬件实现如何承接网络结构。

### 页面内容

- 输入与特征准备：接收 PHAT/LMS 特征或其片上缓存表示。
- 几何预处理：二十面体邻接、PadIco、局部窗口组织。
- ConvIco 加速核心：权重读取、局部 MAC、通道归并。
- 输出模块：归一化、方向读出、SoftArgMax / 后处理。
- 设计原则：memory-first，先解决数据组织与缓存，再讨论算子并行。

### 版式建议

- 绘制顶层硬件框图：`Feature Buffer -> Geometry / PadIco -> ConvIco Engine -> Output Head`。
- 图形保持论文式线框风格，不使用复杂三维硬件图。

## 13. ConvIco 硬件数据流：PadIco、局部缓冲、权重展开、DSP MAC

### 页面目的

展示硬件部分的技术核心，即 ConvIco 的数据流如何降低不规则几何访问带来的实现压力。

### 页面内容

- `PadIco` 与重排映射器：把复杂几何访问规则化。
- 局部输入缓冲：减少直接访问大数组导致的端口冲突。
- 紧凑权重表示与展开：服务局部卷积窗口。
- DSP MAC 阵列：承担热点乘加计算。
- output tile 与局部部分和归并：控制累加路径和输出写回。

### 版式建议

- 放大数据流图：`PadIco -> Local Buffer -> Weight Unpack -> DSP MAC -> Output Tile`。
- 可参考 `hls_src/layer2-5论文插图增强版架构图-中文.md` 或 `layer2-5_DSP48E1_aware_bilingual_flowchart.md` 重画简化版。
- 该页是硬件技术核心页，文字要少，图要清楚。

## 14. 当前 HLS 基础与资源风险

### 页面目的

主动说明已完成的硬件基础和仍未闭合的风险，增强答辩可信度。

### 页面内容

- 已有基础：layer0 / layer1 / layer2-5 的 HLS 工程与资源报告。
- layer2-5 中 `PadIco` 相关 pipeline 已有 `Final II = 1` 的阶段性记录。
- 已有定点量化、局部缓冲、DSP-aware 路径分析。
- 资源风险：layer1 仍可能存在 LUT 超限或资源压力，整网级资源闭合尚未完成。
- 后续重点：完成 `DFA-IcoNet-Edge` 整网预算，对比 ConvIco 主路径延迟、资源和精度影响。

### 版式建议

- 放 HLS 资源快照表：layer0 / layer1 / layer2-5 的 BRAM、DSP、FF、LUT、Latency。
- 如果资源表尚未最终整理，只留表格占位，不填虚构数值。
- 对风险项用浅色标注，避免过度强调失败感。

## 15. 总结页：当前完成度、创新点、下一阶段交付

### 页面目的

收束汇报主线，强调已经完成的工作、当前创新和后续交付。

### 页面内容

- 当前完成：以 `icoCNN` 为 baseline 的双特征注意力融合网络工程链路。
- 当前完成：PHAT + LMS 双特征可视化与 LOCATA 统一口径评测。
- 当前完成：`DFA-IcoNet-Edge` 边缘轻量化候选验证。
- 创新点 1：双特征二十面体输入与注意力融合结构的工程实现。
- 创新点 2：面向 IcoConv 主瓶颈的轻量化边缘折中设计。
- 创新点 3：面向 FPGA/HLS 的 ConvIco 数据流与资源闭合规划。
- 下一阶段交付：`DFA-IcoNet-Edge` 整网硬件预算与毕业论文相关章节。

### 版式建议

- 使用三栏总结：`已完成`、`当前创新`、`下一阶段交付`。
- 最后一行回扣主线：本课题不是单纯复现 IFAN，而是在 `icoCNN baseline` 上完成面向边缘部署的算法与硬件收束。

## 图表与素材处理清单

| 页码 | 图表/素材 | 处理方式 |
| --- | --- | --- |
| 1 | 系统链路图 | 可由 PPT 生成器绘制简约箭头图 |
| 4 | 六段式技术路线图 | 可由 PPT 生成器绘制流程图 |
| 5 | 双特征注意力融合网络结构图 | 使用新大纲中已有图片；无法读取时留占位 |
| 7 | With Silences / Without Silences 结果图 | 插入真实图表；未提供时留空占位 |
| 8 | Tracking 两张图 | 使用新大纲中已有图片；无法读取时留占位 |
| 9 | Params/MAC-RMSAE Pareto 图 | 插入真实图；未生成时留空占位 |
| 10 | MABA 相关图片与两个结果图 | 使用已有图片；结果图未提供时留空占位 |
| 12 | FPGA 顶层框图 | 可由 PPT 生成器绘制简约框图 |
| 13 | ConvIco 数据流图 | 可由 PPT 生成器绘制简约框图 |
| 14 | HLS 资源快照表 | 插入真实表；未整理时留空占位 |
