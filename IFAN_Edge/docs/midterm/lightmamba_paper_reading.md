# LightMamba 论文精读

> 论文：LightMamba: Efficient Mamba Acceleration on FPGA with Quantization and Hardware Co-design  
> 定位：面向 Mamba2 的 FPGA 推理加速框架，对本课题后端轻量 MABA refiner 的 HLS 映射具有参考价值。

## 1. 论文基本信息

LightMamba 由北京大学相关团队提出，目标是在 FPGA 上高效部署 Mamba2。论文核心是算法量化与硬件架构协同设计，重点解决 Mamba 在 FPGA 上量化困难、SSM 计算复杂、片上缓存压力大的问题。

<details>
<summary>详细内容</summary>

| 项目 | 内容 |
|---|---|
| 标题 | LightMamba: Efficient Mamba Acceleration on FPGA with Quantization and Hardware Co-design |
| 研究对象 | Mamba2 推理加速 |
| 平台 | Xilinx Versal VCK190、Alveo U280 |
| 工具 | Vitis HLS + Vivado Design Flow |
| 关键词 | rotation-assisted quantization、PoT SSM quantization、computation reordering、fine-grained tiling and fusion |
| 主要结果 | VCK190 上达到 7.21 tokens/s；相比 GPU baseline 能效提升 4.65-6.06 倍 |

</details>

## 2. 论文背景

Mamba/SSM 相比 Transformer 的优势是序列长度复杂度为线性，并且不需要随序列增长的 KV cache。但论文指出，Mamba 不能直接套用已有 Transformer/LLM FPGA 加速器，因为它的 SSM 层包含大量 element-wise 操作、非线性函数和状态依赖。

<details>
<summary>详细内容</summary>

Mamba2 block 的主要计算结构可以概括为：

```text
Input projection
  -> Conv1d
  -> SSM layer
  -> RMSNorm / SiLU / Softplus / Exp / Element-wise ops
  -> Output projection
```

FPGA 上的主要负载分为两类：

| 负载 | 特点 | FPGA 映射特点 |
|---|---|---|
| Linear / Matrix Multiplication | 规则矩阵乘 | 适合 DSP 阵列和流水化 |
| SSM / Element-wise | 状态依赖强、操作碎、非线性多 | 访存、重定量化和控制逻辑压力更大 |

论文的关键判断是：Mamba 的瓶颈不只是矩阵乘，而是 SSM 层中细粒度 element-wise 数据流、状态更新和中间缓存组织。

</details>

## 3. FPGA 加速挑战

论文总结了 Mamba 在 FPGA 上的三个主要挑战：分散 outlier 导致低比特量化困难，SSM 层重定量化代价高，输入投影与 SSM 之间存在复杂数据依赖。

<details>
<summary>详细内容</summary>

| 挑战 | 具体问题 | 影响 |
|---|---|---|
| 激活 outlier 分散 | Mamba 的异常激活不固定在某些通道，而是随 token 分散到不同通道 | SmoothQuant、Outlier Suppression 等按通道缩放方法效果有限 |
| SSM 层难量化 | SSM 中大量 element-wise multiplication 需要频繁 re-quantization | 直接量化会带来额外 DSP/LUT 开销，FP 实现又过重 |
| 数据依赖复杂 | Input projection 需要生成 `X, B, C, Delta` 后，SSM 才能继续 | naive 顺序执行硬件利用率低，中间激活缓存占用大 |

论文报告中，naive 顺序实现的硬件利用率不足 60%，SSM 中间激活缓存占 URAM 超过 70%。这说明仅靠增加并行 MAC 阵列不能解决 Mamba/SSM 的 FPGA 映射问题。

</details>

## 4. 论文提出的方法

LightMamba 的方法分为算法侧和硬件侧：算法侧用 rotation-assisted PTQ 和 PoT SSM quantization 降低低比特量化误差与重定量化开销；硬件侧用部分展开架构、计算重排、细粒度 tiling/fusion 提高利用率并降低片上缓存。

<details>
<summary>详细内容</summary>

| 方向 | 方法 | 目的 |
|---|---|---|
| 算法 | Rotation-assisted PTQ | 将分散 outlier 摊平，提高 4-bit 量化精度 |
| 算法 | Power-of-two SSM quantization | 将 re-quantization 中的乘法替换为移位 |
| 硬件 | Partially unfolded spatial architecture | 展开一个 Mamba block，平衡吞吐和资源 |
| 硬件 | MMU | 复用处理 input/output projection |
| 硬件 | SSMU | 对 SSM element-wise 操作做细粒度流水 |
| 硬件 | HTU | 支持 Hadamard rotation |
| 数据流 | Computation reordering | 让 SSM 尽早启动，提高硬件利用率 |
| 数据流 | Fine-grained tiling and fusion | 减少中间缓存，降低 URAM 压力 |

整体数据流可以概括为：

```text
Off-chip DRAM weights
  |
  v
MMU: input/output projection
  |
  +--> HTU: Hadamard rotation
  |
  +--> SSMU: Conv + Softplus + Exp + element-wise scan
          |
          v
Fine-grained FIFO pipeline
```

</details>

## 5. 核心技术 1：Rotation-assisted Quantization

LightMamba 使用 Hadamard/orthogonal rotation 处理 Mamba 中分散的 outlier。其核心思想是在不改变矩阵乘结果的前提下，将异常值摊平，使低比特量化更稳定。

<details>
<summary>详细内容</summary>

普通量化方法常假设 outlier 固定在部分通道，因此可以通过 channel-wise scaling 处理。但 Mamba 的 outlier 会随 token 分散到不同通道，传统方法效果不稳定。

LightMamba 使用：

```text
XW = X Q Q^T W
```

其中 `Q` 是正交矩阵或 Hadamard 变换矩阵。这样理论上不改变计算结果，但旋转后的激活和权重更容易量化。

论文中 Mamba2-2.7B output projection activation 的 4-bit 量化误差：

| 方法 | Quantization Error |
|---|---:|
| RTN | 19.5 |
| SmoothQuant | 18.8 |
| OS+ | 309.8 |
| LightMamba | 13.1 |

这说明 rotation 对 Mamba 的分散 outlier 更有效。

</details>

## 6. 核心技术 2：PoT SSM Quantization

SSM 层包含大量 element-wise multiplication。LightMamba 使用 power-of-two scale，使重定量化由乘法变为移位，从而降低硬件代价。

<details>
<summary>详细内容</summary>

普通 INT 量化中，element-wise 操作后经常需要重新缩放：

```text
requantization: value * scale
```

LightMamba 将 scale 约束为 2 的幂：

```text
scale ~= 2^k
requantization: multiply -> shift
```

这对于 SSM 层尤其重要，因为 SSM 内部有大量 element-wise 运算，而 element-wise 运算不像矩阵乘那样有天然的累加归约，重定量化开销更突出。

该方法对本课题的 MABA 状态更新也有直接启发：

```text
h_t = alpha_t * h_{t-1} + (1 - alpha_t) * q_t
```

后续 HLS 中，`alpha`、`h`、`q` 的定点 scale 应优先考虑 PoT 或近似 PoT 设计。

</details>

## 7. 核心技术 3：Computation Reordering

LightMamba 通过改变 input projection 的输出顺序，让 SSM 不必等待全部投影完成后才启动，从而提高硬件利用率。

<details>
<summary>详细内容</summary>

naive 数据流：

```text
Input projection 全部完成
  -> SSM 开始
  -> Output projection
```

LightMamba 改为：

```text
先生成 Delta, B, C 并缓存
再交替生成 X, Z
SSM head-by-head 提前启动
```

论文报告的收益：

| 指标 | 改善 |
|---|---:|
| 总计算时间 | 降低 32% |
| 硬件利用率 | 58% -> 96% |

对本课题而言，这意味着 Strong MABA 不应等待完整 `[B,T,C,R,5,H,W]` 投影完成后再 scan，而应按 channel、region 或 vertex tile 流式推进。

</details>

## 8. 核心技术 4：Fine-grained Tiling and Fusion

LightMamba 通过 tile-by-tile 计算和算子融合，避免保存大量 SSM 中间张量，显著降低片上缓存需求。

<details>
<summary>详细内容</summary>

SSM naive 实现可能需要保存：

```text
B * X
A * h_{t-1}
h_t
h_t * C
Y
```

LightMamba 将其改成更细粒度的数据流：

```text
B, X -> BX -> h update -> C multiply -> Y
```

中间结果尽量通过 FIFO 直接传给下一个算子，而不是落到大 buffer 中。

论文报告：

| 指标 | 改善 |
|---|---:|
| SSMU URAM 使用 | 降低 4x |
| URAM | 246 -> 61 |

该方法对本课题 HLS 很关键，因为当前主干特征 `[B,T,C,R,5,H,W]` 中间激活规模较大，不能依赖全帧全层片上缓存。

</details>

## 9. 实验结果

LightMamba 在保持可接受精度的同时提升了 FPGA 吞吐和能效。W4A4 配置下，VCK190 上达到 7.21 tokens/s；U280 上模拟达到 93 tokens/s；能效相比 GPU baseline 提升 4.65-6.06 倍。

<details>
<summary>详细内容</summary>

### 量化精度

| 方法 | 精度 | Average acc |
|---|---|---:|
| FP16 | - | 60.2 |
| RTN | W8A8 | 59.6 |
| SQ | W8A8 | 59.7 |
| OS+ | W8A8 | 60.1 |
| LightMamba | W8A8 | 60.2 |
| LightMamba* | W8A8，全模型含 SSM | 60.2 |
| RTN | W4A4 | 51.6 |
| SQ | W4A4 | 55.5 |
| OS+ | W4A4 | 30.3 |
| LightMamba | W4A4 | 56.3 |
| LightMamba* | W4A4，全模型含 SSM | 55.9 |

### 硬件吞吐

| 平台 | 精度 | 频率 | 吞吐 |
|---|---|---:|---:|
| VCK190 | W8A8 | 400 MHz | 3.61 tokens/s |
| VCK190 | W4A4 | 400 MHz | 7.21 tokens/s |
| U280 | W4A4 | 200 MHz | 93 tokens/s |
| RTX 2070 | FP16 | 1.62 GHz | 65 tokens/s |
| RTX 4090 | FP16 | 2.52 GHz | 138 tokens/s |

### FPGA 资源

| 平台/精度 | LUT | FF | DSP | BRAM | URAM |
|---|---:|---:|---:|---:|---:|
| VCK190 W4A4 | 107k | 130k | 228 | 912 | 61 |
| VCK190 W8A8 | 111k | 134k | 228 | 914 | 61 |
| U280 W4A4 | 297k | 394k | 1164 | 912 | 61 |

### 能效

| 对比对象 | LightMamba 能效提升 |
|---|---:|
| RTX 2070 | 6.06x |
| RTX 4090 | 4.65x |

</details>

## 10. 对本课题的启发

LightMamba 对本课题的价值不在于照搬大模型 accelerator，而在于提供了 MABA/SSM 后端在 FPGA 上的设计原则：沿时间维顺序 scan，空间/通道维分块并行，中间结果 FIFO 流水传递，状态更新尽量使用定点和 PoT 缩放。

<details>
<summary>详细内容</summary>

| LightMamba 思路 | 对本课题 MABA refiner 的借鉴 |
|---|---|
| SSM 沿序列顺序 scan | MABA 只沿 `T` 做状态更新 |
| Partially unfolded architecture | 不展开完整大张量，只展开 `D/S/tile` |
| PoT SSM quantization | `alpha, h, q` 的定点缩放尽量用 2 的幂 |
| Computation reordering | Strong MABA 按 region/channel tile 流式生成和扫描 |
| Fine-grained tiling/fusion | 避免保存完整 `[T,C,R,5,H,W]` 中间激活 |
| FIFO pipeline | Linear、DWConv、state update、out projection 串成 dataflow |

建议本课题优先从 Weak MABA 开始 HLS：

```text
Weak MABA pre_softargmax:
[B,T,5,H,W]
  -> flatten P=5HW
  -> P->D projection
  -> causal DWConv
  -> state scan over T
  -> D->P projection
  -> residual
```

Strong MABA 则应采用分组/分块版本：

```text
Strong MABA pre_readout:
[B,T,C,R,5,H,W]
  -> tile by C/R/region
  -> small P or grouped channel projection
  -> state scan over T
  -> write back/refine
```

不建议 Strong MABA 一开始全 flatten，否则 Linear 和 buffer 都会明显膨胀。

</details>

## 11. 汇报用一句话总结

LightMamba 说明，Mamba/MABA 类后端在 FPGA 上的关键不是简单堆矩阵乘并行度，而是通过低比特量化、PoT 状态更新、计算重排和细粒度流水，把 SSM 的时间扫描和中间缓存组织成硬件友好的流式结构。

