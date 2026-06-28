# Mamba/MABA 硬件实现挑战与现有 HLS 基础总结

> 用途：作为中期答辩 PPT 硬件部分的补充讲稿。  
> 口径：`DFA-IcoNet-Edge` 是当前硬件映射主线，`pre_readout MABA` 是可选时序增强扩展；MABA 已有算法实现、golden 数据导出和 HLS 切片雏形，但尚未完成默认 HLS top 集成、综合资源报告和整网资源闭合。

## 1. 汇报定位

- 当前硬件主线仍以 `DFA-IcoNet-Edge / IFAN_C8_R2` 为核心，重点解决 ConvIco / IcoConv 后端的规则化数据流、缓存结构和 DSP MAC 复用。
- `pre_readout MABA` 用于在 `final_block` 之后、`channel_readout` 之前做 feature 级时序重整，目标是补充固定窗口因果 1D 卷积在复杂动态场景中的时序稳定性。
- 答辩中不应把 MABA 讲成已完成 FPGA 上板模块；更稳妥的表述是：已有 PyTorch 模块、逐步 golden 张量和 HLS C++ 对齐切片，后续还需要资源闭合与定点化。

## 2. 当前 MABA 计算结构

当前主线采用 `FeatureMABATemporalRefiner`，不是 SoftArgMax 前的 map-level refiner。它的插入链路为：

```text
final_head_logits
  -> FeatureMABA
  -> pre_readout_refined_logits
  -> channel_readout
  -> region max
  -> CleanVertices
  -> SoftArgMax
  -> DOA coordinates
```

当前导出的 HLS 合同：

| 项目 | 当前值 |
| --- | --- |
| FeatureMABA 输入/输出 | `[T=6, C=8, R=6, charts=5, H=2, W=4]` |
| position 展平 | `R * charts * H * W = 240` |
| position layout | `[position=240, T=6, C=8]` |
| latent shape | `[240, 6, 16]` |
| state shape | `[240, 6, 8]` |
| `d_model` | `16` |
| `state_dim` | `8` |
| `conv_kernel` | `3` |
| 状态更新 | `h_t = alpha_t * h_{t-1} + (1 - alpha_t) * q_t` |

FeatureMABA 内部主要计算为：

```text
flatten positions
  -> in projection: C=8 -> D=16
  -> causal depthwise Conv1d over T
  -> LayerNorm over D
  -> state projection: D=16 -> 2 * state_dim
  -> split q/gate and sigmoid(gate)=alpha
  -> state scan over T
  -> state back projection: state_dim -> D
  -> out projection: D=16 -> C=8
  -> residual add
```

## 3. MABA 在 FPGA/HLS 上的主要挑战

| 挑战 | 具体困难 | 对硬件的影响 |
| --- | --- | --- |
| 状态扫描串行依赖 | `h_t` 依赖 `h_{t-1}`，时间维不能像普通卷积一样完全展开 | 需要按 `T` 顺序推进，吞吐优化依赖 position/channel 维并行和流水调度 |
| 细粒度 element-wise 操作多 | `alpha * h`、`(1-alpha) * q`、残差加法等操作分散 | 不能只依靠大矩阵乘阵列，控制逻辑和中间数据搬运会变重 |
| 非线性函数 | `sigmoid`、`LayerNorm` 中的均值/方差/平方根、SoftArgMax 中的指数归一化 | 浮点实现资源重，定点近似会带来精度验证压力 |
| 量化与 re-quantization | SSM/MABA 内部乘法和状态递推需要频繁缩放 | 普通 scale 乘法会增加 DSP/LUT 开销，后续应参考 PoT 或近似 PoT 缩放 |
| 全特征缓存压力 | 若保存完整 `[T,C,R,charts,H,W]` 以及所有中间 tensor，片上 buffer 会明显膨胀 | 需要按 position/tile 流式处理，避免把所有中间激活落入大数组 |
| 与现有 ConvIco/1D 数据流衔接 | 现有 HLS 主线已经围绕 ConvIco 和 temporal Conv1d 形成切片 | MABA 需要明确放在 PL 还是 PS，以及是否进入默认 top |

LightMamba 的启发是：Mamba/MABA 类模块的瓶颈不只是矩阵乘，而是 SSM 状态更新、element-wise 数据流、量化缩放和中间缓存组织。

## 4. 已有 HLS 基础能解决什么

| 问题类别 | 现有基础 | 当前状态 |
| --- | --- | --- |
| 二十面体几何访问不规则 | ConvIco / PadIco / reorder mapper | 已用于 layer0、layer1、layer2-5 和 Stage1 C8_R2 相关 HLS 工作 |
| 规则卷积 MAC 路径 | 局部输入缓存、权重 staging、DSP-aware MAC | 已有 layer2-5 C8/T6 资源快照，`conv_ico_layer2_5` csynth complete |
| 局部时序卷积 | `temporal_r1` 因果 1D HLS 切片 | 已有 `ifan_temporal_r1_top` csynth 快照，说明固定窗口 Conv1d 映射路径成立 |
| Golden/testbench 合同 | `export_stage1_hls_golden.py` 导出权重、几何、stage1 tensor、MABA tensor | 已形成 `hls_testdata/stage1_ifan_c8_r2/scene_1_t6` 合同 |
| FeatureMABA 算子级复现 | `hls_src/HLS/stage1_ifan_c8_r2/feature_maba` | 已有 C++ engine 和逐步 tensor 对齐 testbench |
| MABA 后处理 | `hls_src/HLS/stage1_ifan_c8_r2/post_maba` | 已有 channel readout、region max、CleanVertices、SoftArgMax 坐标头切片 |

可以在 PPT 中概括为：

```text
已有 ConvIco/1D HLS 基础解决了规则卷积和局部时序卷积映射；
MABA 目前已推进到 golden 合同和 C++ 切片级验证；
真正待攻克的是状态扫描、非线性/量化、tile 化流水和整网资源闭合。
```

## 5. 未解决问题与下一步

当前不能过度表述的部分：

- MABA 尚未并入默认 HLS top；`post_maba` README 明确说明后续需根据 MABA 资源闭合结果决定放在 PS 还是 PL。
- 尚未看到独立 `feature_maba` 的 csynth 资源报告，因此不能宣称 MABA 的 BRAM/DSP/LUT/Latency 已闭合。
- 当前 HLS 类型基础仍以浮点合同和 C++ 对齐为主，尚未形成可汇报的 MABA 定点位宽、PoT scale 或近似 sigmoid/LayerNorm 方案。
- 现有 `feature_maba_engine` 保留大量中间数组，适合对齐验证，但还不是最终 streaming/dataflow 结构。
- 整网级调度仍需决定：ConvIco/temporal 输出是否直接流入 MABA，还是把 MABA 放在 PS 或作为可选 PL IP。

建议下一阶段路线：

1. 固化 MABA 的最小硬件合同：输入 `final_head_logits`，输出 `pre_readout_refined_logits`，先只覆盖 `T=6,C=8,R=6,charts=5,H=2,W=4`。
2. 做 `feature_maba` 独立 csim/csynth，拿到第一版资源和 latency，用于判断是否值得进入 PL。
3. 将验证版中间数组改为 position/tile 流式处理：`in_proj -> DWConv -> state update -> out_proj` 尽量用 FIFO 或小 buffer 串接。
4. 为 `sigmoid`、`LayerNorm` 和状态更新建立定点近似策略，优先考虑 PoT 或近似 PoT 缩放，降低 re-quantization 成本。
5. 根据资源结果决定系统分工：MABA 留在 PS、作为可选 PL IP，或只保留 `DFA-IcoNet-Edge` 主线并把 MABA 作为后续扩展。

## 挑战-已有基础-缺口矩阵

| 层级 | 具体问题 | 已有基础 | 当前缺口 |
| --- | --- | --- | --- |
| 已有方法能覆盖 | 规则卷积映射 | ConvIco / PadIco / local buffer / DSP MAC | 整网级资源预算仍需汇总 |
| 已有方法能覆盖 | 二十面体几何访问规则化 | reorder mapper、kernel expansion、HLS testdata | 与 MABA 本身关系较弱，主要服务 ConvIco |
| 已有方法能覆盖 | 局部 temporal Conv1d | `temporal_r1` HLS 切片和 csynth 快照 | 只能覆盖固定窗口卷积，不能替代状态扫描 |
| 已有方法能覆盖 | golden/testbench 合同 | `stage1_ifan_c8_r2` 导出 final logits、MABA tensors、post-MABA tensors | 还需要把合同转成可综合资源评估 |
| 部分覆盖 | MABA 算子级 C++ 复现 | `feature_maba` engine 与逐步 tensor 对齐 | 未见独立资源报告，结构仍偏验证版 |
| 部分覆盖 | MABA 后处理路径 | `post_maba` channel readout、region max、SoftArgMax 切片 | README 标注暂不并入默认 top |
| 尚未覆盖 | 状态扫描流水化 | 代码已有 `state_scan` 计算关系 | 缺少 tile/dataflow 优化和 II/latency 评估 |
| 尚未覆盖 | 非线性近似 | 当前有 `sigmoid`、`LayerNorm` 浮点表达 | 缺少 LUT/分段/定点近似方案 |
| 尚未覆盖 | 低比特定点与 PoT 缩放 | LightMamba 文档给出参考方向 | 缺少本模型量化实验和误差评估 |
| 尚未覆盖 | MABA 资源闭合 | 有 HLS 源码雏形 | 缺少 csynth/cosim/resource summary |
| 尚未覆盖 | 整网调度 | ConvIco、temporal、MABA/post-MABA 都有局部材料 | 缺少统一 top、PS/PL 分工和数据搬运预算 |

## PPT 可用一句话

MABA 在硬件上的难点不在于单个矩阵乘规模，而在于状态扫描的时间依赖、细粒度 element-wise 数据流、非线性与定点量化、以及全特征中间缓存；我们已有 ConvIco/1D HLS 和 MABA C++ 对齐切片作为基础，但 MABA 仍需要完成 tile 化流水、定点近似和资源闭合后才能作为正式硬件成果汇报。
