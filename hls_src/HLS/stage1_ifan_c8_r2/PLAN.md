# IFAN_C8_R2_MABA Pre-Readout 的 HLS 硬件实现计划

## Summary

当前算法架构冻结为 `ifan_c8_r2_maba_pre_readout_best`，第一阶段 HLS 目标只覆盖网络主干到 MABA 前：以已生成的 `[B,2,T,5,4,8]` PHAT/LMS 特征作为输入，完成 layer0-layer5 / 主干功能块化重构；MABA、channel readout、region max、CleanVertices、SoftArgMax 放到第二阶段及后续阶段。

目标平台沿用 `xc7k325t` 资源口径；数值策略采用“外部 float 接口、内部 ap_fixed 定点”。核心路线不是逐层复制旧 baseline HLS，而是把 IcoConv、temporal conv、norm/activation、pool/readout 前处理设计成可复用功能块，由配置和权重驱动不同层调用，实现资源复用和结构压缩。

## Key Implementation Plan

- 冻结硬件输入/输出边界：
  - Stage-1 输入：`input[B=1, C=2, T, 5, 4, 8]`，通道 0 为 PHAT，通道 1 为 LMS。
  - Stage-1 输出：MABA 前的 pre-readout feature tensor，形状按当前模型为 `[B,T,8,6,5,2,4]`。
  - Stage-1 不实现 MABA、channel readout、SoftArgMax；只保证输出能和 PyTorch `final_head_logits` 或 `pre_readout_refined_logits` 前一节点对齐。

- 建立可复用 HLS 功能块库：
  - `IcoConvEngine`：统一支持 `Cin/Cout/Rin/Rout/H/W/r` 参数，通过 layer config 调用，复用已有 `layer2-5` 的 staging、PadIco、kernel 展开、partial-sum、tile writeback 思路。
  - `TemporalConv1dEngine`：支持当前 `standard_1d`，固定优先覆盖 `channels=8, kernel=5, dilation=1`，按 `(R,chart,H,W)` 位置复用同一个时间卷积核。
  - `LNormIcoEngine`：实现当前 IFAN 主干中 ConvIco 后的 LNormIco，先保留 float 对账接口，内部定点统计与缩放可配置。
  - `ElementwiseEngine`：覆盖 ReLU、Sigmoid、residual add、attention multiply-add：`fused = direct + enhanced * sigmoid(attention)`。
  - `PoolIcoEngine`：覆盖 `r=2 -> r=1` 的 pre-fusion pooling，复用几何索引/重排表机制。

- 主干调度按功能块复用组织：
  - PHAT/LMS 两支不复制硬件单元，采用同一套 `IcoConvEngine + ResidualBlock` 分时处理不同权重。
  - Shared attention 只保留一套权重和一套硬件路径，对 PHAT/LMS enhanced feature 分别调用。
  - Fusion head 的 4 个 `IcoConv -> ReLU -> TemporalConv1d -> LNorm -> ReLU` 用同一组功能块循环调用不同权重。
  - Final block 用同一套 `IcoConvEngine + TemporalConv1dEngine + LNormIcoEngine`，输出到 Stage-1 终点 buffer。

- 重构现有 HLS 工程：
  - 以 `hls_src/HLS/layer2-5` 当前 II=1 版本作为 IcoConv 主体模板，抽离成 reusable engine，而不是继续维护 layer-specific 大函数。
  - `layer0/layer1/layer2-5` 旧工程保留为验证基线；新建一个 stage-1 top 工程用于 IFAN_C8_R2 主干串接。
  - 保留外部 `float` 数组接口，内部统一转换为 `input_t/weight_t/act_t/acc_t`，初始位宽沿用 layer2-5 文档中的混合定点策略，再做 sweep。

## Hardware Strategy

- 资源复用优先级高于极限吞吐：
  - 第一版目标是单套或少量并行 IcoConv/TemporalConv 功能块分时跑完整主干。
  - `OC_TILE` 初始建议从 `2` 开始，避免 layer1 当前 LUT 超标问题再次出现。
  - 对 C8_R2 重新设定 tile，不沿用 C16/32 baseline 的通道假设。

- 数据流策略：
  - 不全帧全层缓存；按 `T` 帧、`OC tile`、空间 tile 分块推进。
  - 权重常驻或分批加载到本地 buffer，避免重复解析完整大索引表。
  - PadIco、kernel 展开、partial sum、post-process、writeback 继续采用分层流水结构。
  - 对融合块采用“层间写回 + 下一层读入”的稳健版本作为 v1；确认正确后再评估 top-level dataflow 双缓冲。

- MABA 后续阶段预留接口：
  - Stage-1 输出 buffer 必须保持 `[T,8,6,5,2,4]` 语义，作为后续 FeatureMABA 的输入。
  - 后续 MABA 采用 LightMamba 风格：沿 `T` 顺序 scan，`C/R/chart/H/W` 分组并行，内部使用定点和 PoT scale 优化。

## Test Plan

- Python golden data：
  - 从 `ifan_c8_r2_maba_pre_readout_best` checkpoint 导出 Stage-1 输入、每个关键中间节点、Stage-1 输出。
  - 至少包含普通场景和 hard scene：`scene_1`、`scene_4` 各一组小样本。

- C-sim 分层验证：
  - 单独验证 `IcoConvEngine`、`TemporalConv1dEngine`、`LNormIcoEngine`、`PoolIcoEngine`。
  - 再验证 PHAT branch、LMS branch、shared attention、fusion blocks、final block。
  - 最后验证 Stage-1 top 输出与 PyTorch 对齐。

- 误差标准：
  - float 外接口版本先要求功能 PASS，记录 max error/RMSE。
  - 定点版本先按模块分别给出误差，再给 Stage-1 总误差；若误差影响明显，优先调 `acc_t` 和 LNorm 路径位宽。
  - 报告必须同时记录资源、latency、Estimated Clock、关键 loop II。

- 综合验收：
  - `IcoConvEngine` 关键循环保持 `II=1` 或明确记录无法达成的瓶颈。
  - Stage-1 top 在 `xc7k325t` 口径下不允许 LUT 超 100%；第一轮资源目标优先低于旧 layer1 的 123.95% LUT。
  - 输出一份 stage-1 HLS summary，用于后续论文硬件章节和 MABA 阶段衔接。

## Assumptions

- 第一阶段不包含 PHAT/LMS 特征生成，输入默认是算法侧已经生成的双通道特征图。
- 第一阶段不实现 MABA 和 SoftArgMax，只在接口上为 MABA 保留 pre-readout feature tensor。
- 目标资源按现有文档中的 `xc7k325t`：`LUT=203800, FF=407600, DSP=840, BRAM_18K=890`。
- 第一版重构以正确性和资源闭合为主，不追求整网端到端最低 latency；资源复用是本阶段的核心创新表述。
- 当前模型关键硬件规模以 `branch_channels=8, r=2, fusion_r=1, T` 来自导出测试数据为准。
