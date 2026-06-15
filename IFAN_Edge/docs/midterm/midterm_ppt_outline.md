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

- 研究现状：近年的声学场景分析更偏向 DCASE/SELD 一类联合任务，同时处理事件检测、事件分类、DOA 估计甚至距离估计；这类模型能力更综合，但并非针对纯 DOA 估计专项设计，方位估计精度和边缘部署代价之间仍有优化空间。
- 选择纯 DOA 的原因：本课题不追求把所有声学任务合并到一个大模型里，而是深耕“方位估计”这一单一但关键的感知能力，在可解释的二十面体空间建模框架下针对性优化模型结构。
- 声源定位任务目标：从多通道麦克风信号估计声源方向，并验证模型在不同声学环境和运动状态下的稳定性。
  - 运动状态包含 3 类单源场景：固定麦克风静态声源、固定麦克风移动声源、移动麦克风移动声源。
  - 声学环境包含 4 种典型难度组合：`30dB/T60=0.2s`、`30dB/T60=0.8s`、`5dB/T60=0.8s`、`5dB/T60=1.4s`。
  - 后续实验验证分为两条线：模拟四场景用于声学环境鲁棒性测试，LOCATA `Task1/3/5` 用于运动状态适应性测试。
- 本课题优势：依托 LOCATA 单源任务覆盖固定/移动麦克风、静态/动态声源等运动状态，同时用四种 SNR/T60 组合验证恶劣声学环境下的鲁棒性；后续可补充 DCASE 片段作为跨数据集专项 DOA 对比。
- 边缘端部署不仅要求模型精度，还要求控制参数量、MAC、片上缓存和数据搬运压力；音频前端可采用软件/异构预处理，硬件侧优先关注网络后端。
- 本课题的核心问题：如何在 icoCNN baseline 上增强纯方位估计能力，并形成更适合边缘硬件映射的轻量化网络后端。

图表建议：

- 右侧补一个验证矩阵：纯 DOA 专项设计 -> 声学环境鲁棒性测试（4 个模拟 scene）+ 运动状态适应性测试（LOCATA Task1/3/5）+ 后续 DCASE 片段交叉对比。


## 3. Baseline 与问题切入：icoCNN 的优势、瓶颈和改进空间

- 为什么选择 `icoCNN` 作为 baseline：
  - `icoCNN` 是面向纯 DOA 估计的二十面体网络，与本课题“方位估计专项优化”的目标一致。
  - 相比 DCASE/SELD 多任务模型，`icoCNN` 的任务边界更清晰，更适合评估纯方位估计性能。
- baseline 的主要瓶颈：
  - 特征侧：单一 PHAT 响应依赖相位加权互相关，在低 SNR、强混响和运动场景下容易出现峰值扩散或伪峰，导致方位读出不稳定。
  - 时序侧：声源或麦克风运动会带来跨帧时延变化，单帧 PHAT 图难以显式保留这种动态时延演化信息。
  - 结构侧：baseline 缺少面向互补特征的自适应融合机制，难以在不同声学条件下选择更稳定的空间证据。
  - 部署侧：IcoConv 计算量随输入/输出通道宽度近似二次增长，边缘部署资源压力较大。
- 本课题的改进方向：
  - 引入 LMS 自适应时延估计特征，补充 PHAT 在复杂声学环境下的动态时延信息。
  - 将 PHAT 的稳健空间响应与 LMS 的时序自适应能力融合，增强二十面体特征表达。
  - 设计双分支残差增强与注意力融合结构，形成 `DFA-IcoNet`。
  - 通过 `DFA-IcoNet-Edge` 做结构化宽度收缩，降低参数量与 MAC。
  - 将硬件侧目标收束为网络后端 ConvIco 数据流与 FPGA 资源闭合。
- 答辩边界主动说明：
  - 当前不宣称横向超越所有 SELD/SOTA 模型，而是在同一二十面体 DOA 框架下验证专项结构改进和边缘映射可行性。
  - 当前不宣称完整音频前端全链路 FPGA 实现，硬件重点优先放在二十面体网络后端。

图表建议：

- 表格对比：`为什么选 icoCNN`、`baseline 瓶颈`、`本文对应改进`。
- 可在页脚加一句防追问口径：`SELD/DCASE 多任务模型作为后续扩展对比，当前主线聚焦纯 DOA 与硬件映射闭环。`


## 4. 本课题工作总览：双特征、融合网络、轻量化、FPGA 映射

- 研究目标：面向纯 DOA 估计任务，在 `icoCNN` 二十面体 baseline 上提升复杂声学环境和运动场景下的方位估计稳定性。
- 特征补强：针对单一 PHAT 在低 SNR、强混响和运动场景下可能出现伪峰或跨帧信息不足的问题，构建 `PHAT + LMS` 双特征二十面体输入。
- 网络设计：从 icoCNN 单特征主干扩展为双分支残差增强与注意力融合网络，形成完整宽度精度参考模型 `DFA-IcoNet`。
- 边缘轻量化：围绕 IcoConv 主瓶颈进行结构化宽度收缩，形成边缘轻量化候选模型 `DFA-IcoNet-Edge`。
- 时序增强：在轻量化主干上进一步引入 feature 级 `pre_readout MABA` temporal refiner，形成当前确定模型 `DFA-IcoNet-Edge-MABA`，用于增强低 SNR / 高混响场景下的跨帧特征稳定性。
- 实验验证：
  - 模拟四场景验证声学环境鲁棒性。
  - LOCATA `Task1/3/5` 验证固定/移动麦克风、静态/动态声源下的运动状态适应性。
  - DCASE-2025挑战赛DOA数据集作为跨数据集纯 DOA 对比。
- 硬件边界：当前不追求完整音频前端全链路 FPGA 实现，优先探索二十面体网络后端 ConvIco / IcoConv 的 HLS 数据流、资源闭合和整网预算。

图表建议：

- 六段式路线图：Problem -> Feature -> Network -> Edge -> MABA Temporal Refine -> Hardware。
- 图中标注两条验证线：`模拟四场景 = 声学环境鲁棒性`，`LOCATA Task1/3/5 = 运动状态适应性`。


## 5. 算法主体设计：PHAT+LMS 双特征与注意力融合网络

- 前端补强：构建 `PHAT + LMS` 双特征二十面体输入。
  - `channel 0 = PHAT`：提供稳健的相位加权空间响应。
  - `channel 1 = LMS`：补充跨帧动态时延估计信息。
  - 已实现 `SRPPHATIcoMapAdapter`、`SRPLMSIcoMap`、`DualFeatureIcoPreprocessor`。
- 可视化证据：四个典型 scene 已导出双特征投影图，用于说明 PHAT/LMS 在二十面体网格上形成可观察的空间响应。
  - `IFAN_Edge/outputs/stage1_features/scene_*/feature_maps_projection_contrast.png`
- 网络主体：在保留 icoCNN 二十面体卷积能力的基础上，扩展为双分支残差增强与注意力融合结构。
  - PHAT / LMS 双输入分支。
  - branch-local fusion + shared attention weight module。
  - second-stage fusion 后进入深层 fusion head 完成方向估计。
- 当前完整宽度精度参考模型：`DFA-IcoNet`。

图表建议：

- 主图：`多通道音频 -> PHAT/LMS 双特征 -> 双分支残差增强 -> shared attention fusion -> fusion head -> DOA`。
- 右下角放 1-2 张 `feature_maps_projection_contrast.png` 小图作为特征可视化证据，不必四张全放。
- 图名写成“本课题双特征注意力融合网络结构”，不要写成复现论文结构。

备注：

- 这一页相当于算法主体总页：把前端、可视化和网络结构压缩到一页，后面直接衔接轻量化。

## 6. 轻量化设计：IcoConv 主瓶颈与 C8_R2 边缘候选

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

## 7. 实验设置：模拟实验 + LOCATA 真实数据评测

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
  - `DFA-IcoNet-Edge-MABA = pre_readout MABA 时序增强模型，对应 ifan_c8_r2_maba_pre_readout_best`

图表建议：

- 放一张紧凑实验设置表：数据集、任务、指标、模型。

备注：

- 训练策略、评测协议和数据口径合并到一页，不再拆成多页。

## 8. LOCATA 总体结果：核心模型对比

| Model | Params | MAC | With Silences Avg | Without Silences Avg |
| --- | ---: | ---: | ---: | ---: |
| icoCNN baseline | 290017 | - | 8.5718 | 7.1976 |
| DFA-IcoNet | 125457 | 459532800 | 7.2407 | 6.2693 |
| DFA-IcoNet-Edge | 31561 | 115211520 | 7.8581 | 7.0755 |
| DFA-IcoNet-Edge-MABA | 32353 | 116213760 | 7.7960 | 6.9130 |

- `DFA-IcoNet` 相对 icoCNN baseline：
  - with silences average 改善 `1.3310 deg`
  - without silences average 改善 `0.9283 deg`
- `DFA-IcoNet-Edge` 相对 icoCNN baseline：
  - with silences average 改善 `0.7136 deg`
  - without silences average 改善 `0.1221 deg`
- `DFA-IcoNet-Edge-MABA` 相对 `DFA-IcoNet-Edge`：
  - 参数量仅从 `31561` 增至 `32353`，MAC 从 `115211520` 增至 `116213760`
  - with silences average 进一步改善 `0.0621 deg`
  - without silences average 进一步改善 `0.1625 deg`
- 结果解释：MABA 的总体平均收益不大，但在不显著增加资源的情况下改善了 LOCATA 平均值，为后续 hard-scene 时序增强提供证据。

图表建议：

- 主图：with / without silences RMSAE 双柱状图，越低越好，加入 `DFA-IcoNet-Edge-MABA` 第四根柱。
- 角落保留参数量和 MAC 小表。

备注：

- 该页是算法实验核心结果页，建议保留完整数字。

## 9. 分任务结果：Task1 / Task3 / Task5 稳定性观察

- LOCATA 单源任务：
  - Task1：静态或相对简单场景，recording 数最多。
  - Task3 / Task5：更能体现真实数据中场景差异。
- 展示重点：
  - 不只看总体平均，还要看不同任务上的波动。
  - `DFA-IcoNet` 作为完整宽度模型更稳定。
  - `DFA-IcoNet-Edge` 在轻量化后仍保持总体平均优势，但部分任务损失需要解释。
  - `DFA-IcoNet-Edge-MABA` 的收益主要集中在 Task5：with silences Task5 mean 从 `11.8516` 降至 `11.2159`，without silences Task5 mean 从 `9.9599` 降至 `8.5478`。
  - Task1/Task3 上 MABA 有小幅波动，说明 pre_readout MABA 更像 hard-scene / 动态场景增强，而不是所有任务的均匀增益模块。

图表建议：

- 每个 task 一组柱状图：icoCNN baseline / DFA-IcoNet / DFA-IcoNet-Edge / DFA-IcoNet-Edge-MABA。
- 指标优先用 mean RMSAE，可分 with silences 和 without silences 两张小图。

备注：

- 这页用于支撑“真实数据上的稳定性”，图会比较大，单独展示。

## 10. Tracking 可视化：真实轨迹与预测轨迹对比

- 目标：参考 IFAN 原论文 tracking 图的表达方式，生成本课题自己的轨迹可视化。
- 优先方案：
  - 从 LOCATA `Task1 / Task3 / Task5` 选 1-2 个 recording。
  - 绘制 ground truth、icoCNN baseline、DFA-IcoNet、DFA-IcoNet-Edge、DFA-IcoNet-Edge-MABA 的时间序列轨迹。
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

## 11. 资源-精度折中：参数量、MAC 与 RMSAE

| Comparison | Params Change | MAC Change | With Silences Avg Delta | Without Silences Avg Delta |
| --- | ---: | ---: | ---: | ---: |
| DFA-IcoNet vs icoCNN baseline | 56.7% reduction | n/a | -1.3310 deg | -0.9283 deg |
| DFA-IcoNet-Edge vs icoCNN baseline | 89.1% reduction | n/a | -0.7136 deg | -0.1221 deg |
| DFA-IcoNet-Edge vs DFA-IcoNet | 74.8% reduction | 74.9% reduction | +0.6174 deg | +0.8062 deg |
| DFA-IcoNet-Edge-MABA vs DFA-IcoNet-Edge | +2.5% params | +0.9% MAC | -0.0621 deg | -0.1625 deg |

- 关键结论：`DFA-IcoNet-Edge` 用约 `75%` 的参数量与 MAC 压缩，换取可接受的平均精度损失。
- `DFA-IcoNet-Edge-MABA` 在极小资源增量下带来平均精度补偿，尤其对 hard-scene / Task5 更有价值。
- 后续 FPGA/HLS 默认以 `DFA-IcoNet-Edge` 为基础网络候选，并单独评估 MABA refiner 是否纳入时序增强扩展。

图表建议：

- Pareto 散点图：
  - 横轴：MAC 或 Params。
  - 纵轴：LOCATA average RMSAE。
  - 点：icoCNN baseline / DFA-IcoNet / DFA-IcoNet-Edge / DFA-IcoNet-Edge-MABA。

备注：

- 这一页承接算法实验到硬件映射的转场。

## 12. Pre-readout MABA 时序增强与消融收束

- 加入 MABA 的原因：
  - `DFA-IcoNet-Edge` 已经完成主干轻量化，但低 SNR / 高混响和动态场景下仍可能出现跨帧响应不稳定。
  - MABA temporal refiner 用于补充轻量化主干的时序建模能力，不改变 `PHAT + LMS` 和二十面体主干的基本接口。
- 当前确定模型：
  - `DFA-IcoNet-Edge-MABA` 对应实验 `ifan_c8_r2_maba_pre_readout_best`。
  - 只在一个位置加入 MABA：`final_block` 之后、`channel_readout` 之前。
  - 输入张量为 `[B, T, C, R, 5, H, W]`，此时 channel 和 region 信息尚未被压缩成单通道响应图。
- 结构作用：
  - `pre_readout MABA` 基本保留完整 MABA 形态：Linear In、depthwise temporal conv、gate、state scan、Linear Out 和 residual。
  - 它负责 feature 级时序重整，在读出前利用多通道、多 region 的弱证据抑制跨帧尖峰和伪峰。
  - 不再把 `pre_softargmax` 作为主模型组成部分；`dual_refine` 只作为消融，说明过晚的响应图级整形可能对 hard scene 产生过平滑风险。
- 实验观察：
  - 模拟 `scene_4 (5dB/T60=1.4s)` 中，`pre_readout` 将 `DFA-IcoNet-Edge` 的 RMSAE 从 `17.9796 deg` 降到 `13.9095 deg`，说明 feature 级时序重整对强噪声强混响场景有明显价值。
  - 模拟四场景 mean 从 `9.9787 deg` 降到 `8.5406 deg`，hard mean 从 `14.2679 deg` 降到 `11.6980 deg`。
  - LOCATA without silences average 从 `7.0755 deg` 降到 `6.9130 deg`，with silences average 从 `7.8581 deg` 降到 `7.7960 deg`。
  - LOCATA Task5 改善最明显，说明该模块更适合补强复杂动态/强干扰场景，而不是追求所有任务平均无波动提升。
- 其他消融收束：
  - `C8_R3` 参数量与 `DFA-IcoNet-Edge` 相近，但 MAC 基本不降，固定为失败参考。
  - `map_maba` 放在响应图压缩之后，scene_4 基本没有改善；`dual_refine` 增加第二个弱 refiner 后 scene_1/2/3 略有收益，但 scene_4 弱于 `pre_readout`，因此不作为当前确定模型。

图表建议：

- 主图画成单一插入位置流程：`Final block -> pre_readout MABA -> channel_readout -> region max -> CleanVertices -> SoftArgMax`。
- 配一个小表：`DFA-IcoNet-Edge / map_maba / pre_readout / dual_refine` 在 `scene_4`、four-scene mean、hard mean 和 LOCATA average 上的对比。

备注：

- 这页的定位是“pre_readout MABA 为什么有效”，不要把 MABA 讲成两个位置同时加入的主流程。

## 13. FPGA/HLS 设计目标：为什么算法收束后需要硬件映射

- 边缘部署不只看模型参数，还要看：
  - MAC 压力。
  - 片上缓存。
  - 数据搬运。
  - DSP / BRAM / LUT / FF 资源。
  - 延迟与吞吐。
- 算法收束后的硬件目标：
  - 以 `DFA-IcoNet` 作为精度参考。
  - 以 `DFA-IcoNet-Edge` 作为默认边缘实现候选。
  - 将 `DFA-IcoNet-Edge-MABA` 作为可选时序增强扩展，单独评估其少量额外参数/MAC 对 HLS 资源和延迟的影响。
  - 围绕 ConvIco / IcoConv 主瓶颈做数据流与资源优化。

图表建议：

- 算法指标 -> 硬件指标映射图：Params/MAC/RMSAE -> DSP/BRAM/LUT/Latency。

备注：

- 这一页开始进入后续工作，不要讲成已完成整网上板。

## 14. FPGA 整体架构预期：前端、缓存、IcoConv 加速、输出

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

## 15. ConvIco 硬件数据流：PadIco、局部缓冲、权重展开、DSP MAC

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

## 16. 当前 HLS 基础与资源风险

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

## 17. 后续计划：补图、gap 解释、硬件预算、论文撰写

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

## 18. 总结页：当前完成度、创新点、下一阶段交付

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
