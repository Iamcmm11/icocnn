# 分层计算访存与 Latency 瓶颈统一分析

## 0. 2026-03-22 稳定阶段更新

2026-03-22，`layer2-5` 共享 `ConvIco` 块完成了一次可以保留的结构级重构。该版本已经通过
[layer2-5硬件优化与策略跟踪.md](G:/3DSLED/icocnn/hls_src/layer2-5硬件优化与策略跟踪.md)
中的功能回归与 HLS 重新综合，因此这里先补充稳定阶段结论，避免后文的旧 baseline 与当前稳定版本混淆。

### 0.1 `layer2-5` 新稳定版指标

| 指标 | 重构前 baseline | 2026-03-22 稳定版 | 变化 |
|---|---:|---:|---:|
| Estimated Clock | `4.498 ns` | `4.472 ns` | 略优 |
| Total Latency | `6213631061 cycles` | `498988881 cycles` | `-91.97%` |
| 估算执行时间 | `31.068 sec` | `2.495 sec` | `-91.97%` |
| 单帧 iteration latency | `119492905 cycles` | `9595940 cycles` | `-91.97%` |
| BRAM_18K | `66` | `50` | `-24.24%` |
| DSP | `13` | `139` | `+126` |
| FF | `12108` | `37847` | `+25739` |
| LUT | `11337` | `32331` | `+20994` |

### 0.2 对统一分析结论的修正

1. `layer2-5` 当前最关键的有效优化已经被证明是“主数据流重构 + 局部输出 tile 累加”，而不是单纯增加 pragma。
2. 原 baseline 中“主 MAC 串行累加链”和“顶层输出端口读写冲突”这两类问题，已经在 2026-03-22 稳定版中被明显缓解。
3. 因此，`layer2-5` 的主要矛盾已经从“单输出点串行求和 + 顶层输出端口竞争”，转向“`PadIco` 输入端口冲突 + 局部 `output_tile` 累加调度”。
4. 这说明 `layer2-5` 已经进入下一阶段：后续优化重点不再是确认是否需要结构重构，而是继续优化局部缓冲与输入访存组织。

### 0.3 对论文叙事的意义

1. 这次结果验证了 `layer2-5` 适合抽象成“共享参数化主干卷积块”，并且该主干块的性能收益来自结构化的数据复用，而不是仅靠局部指令调优。
2. `layer2-5` 稳定版可以作为“共享主干架构可显著降低总拍数”的直接实验依据。
3. 因为当前版本时序仍满足 `5.00 ns` 目标，而 latency 已下降约 `12.45x`，所以论文中的主结论应继续强调“瓶颈在数据流与存储组织，不在关键路径频率”。

### 0.4 当前统一视角下的下一步

1. 对 `layer2-5`，优先处理 `PadIco` 的输入端口冲突。
2. 对 `layer2-5`，继续压缩局部 `output_tile` 累加阶段的启动间隔。
3. 对 `layer1`，后续可参考这次 `layer2-5` 的局部输出 tile 路径，评估是否存在可迁移的主干结构化改法。

## 1. 文档目的

本文档用于把当前 `layer0`、`layer1`、`layer2-5` 的硬件映射对象统一整理为：

1. 计算量模型
2. 访存量模型
3. 结构特征总结
4. 基于现有 HLS 报告的 latency 来源拆解

目标不是单纯汇总数字，而是为后续论文中的“网络结构适配型硬件架构设计”提供统一分析基础。

---

## 2. 统一分析口径

### 2.1 统一符号

对当前 ConvIco 顶层块，记：

- `T`：时间步数
- `Cin` / `Cout`：输入 / 输出通道数
- `Rin` / `Rout`：输入 / 输出旋转维
- `Charts * H * W`：每帧的 icosahedral 空间位置数
- `7`：紧凑 7 邻域卷积核

### 2.2 统一计算量公式

主卷积 MAC 数可统一写为：

`MACs = T * Cout * Rout * Charts * H * W * Cin * Rin * 7`

该公式只统计主卷积，不额外计入：

1. `PadIco` 的重排和极点处理
2. `kernel_expansion_idx` 的索引展开开销
3. 输出端极点平滑后处理

因此它是“主计算核复杂度”的统一上界表达。

### 2.3 统一访存量口径

本文用两类访存量：

1. 静态张量规模
含输入、输出、权重、bias、索引表本身的存储规模。

2. 主卷积逻辑访问量
假设每次 MAC 都需要一次输入读和一次权重读，则：

- 逻辑输入读次数约为 `MACs`
- 逻辑权重读次数约为 `MACs`
- 输出写次数约为输出元素数

这不是最终硬件真实片上访存次数，而是“若不做充分复用时的逻辑访问需求”，可用于说明为什么必须做缓存、tiling 和数据复用。

---

## 3. 统一结构表

| 层类型 | 角色定位 | T | Cin | Cout | Rin | Rout | H x W | Charts | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `layer0` | 前端特化层 | 52 | 1 | 32 | 1 | 6 | `4 x 8` | 5 | 输入几何特殊，`1 -> 32`、`1 -> 6` 扩展明显 |
| `layer1` | 过渡层 | 52 | 32 | 32 | 6 | 6 | `4 x 8` | 5 | 已进入主干通道规模，但仍保留前端到共享块的过渡性质 |
| `layer2-5` | 共享参数化主干块 | 52 | 32 | 32 | 6 | 6 | `2 x 4` | 5 | 重复出现、结构一致，最适合参数化复用 |

---

## 4. 统一计算量与静态数据规模表

### 4.1 元素数量统计

| 层类型 | 输入元素数 | 输出元素数 | 权重元素数 | bias 元素数 | `kernel_idx` 元素数 | `reorder_idx` 元素数 | 主卷积 MACs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `layer0` | 8,320 | 1,597,440 | 224 | 32 | 6,912 | 300 | 11,182,080 |
| `layer1` | 1,597,440 | 1,597,440 | 43,008 | 32 | 1,327,104 | 1,800 | 2,146,959,360 |
| `layer2-5` | 399,360 | 399,360 | 43,008 | 32 | 1,327,104 | 720 | 536,739,840 |

### 4.2 主要张量静态规模

默认 `float32/int32` 均按 `4 Byte` 估算。

| 层类型 | 输入规模 | 输出规模 | 权重规模 | `kernel_idx` 规模 | 特征 |
|---|---:|---:|---:|---:|---|
| `layer0` | `0.0317 MB` | `6.0938 MB` | `0.00085 MB` | `0.0264 MB` | 输入很小，但输出已经进入较大规模 |
| `layer1` | `6.0938 MB` | `6.0938 MB` | `0.1641 MB` | `5.0625 MB` | 输入/输出与索引规模都显著增加 |
| `layer2-5` | `1.5234 MB` | `1.5234 MB` | `0.1641 MB` | `5.0625 MB` | 空间分辨率下降后，特征图规模降为 `layer1` 的 1/4 |

### 4.3 主卷积逻辑访问量

若不考虑复用，主卷积阶段的逻辑输入/权重读取量约为：

| 层类型 | 逻辑输入读量 | 逻辑权重读量 | 说明 |
|---|---:|---:|---|
| `layer0` | `0.0417 GB` | `0.0417 GB` | 规模较小，主瓶颈不在 MAC 总量 |
| `layer1` | `7.9980 GB` | `7.9980 GB` | 主计算量和主访存需求急剧放大 |
| `layer2-5` | `1.9995 GB` | `1.9995 GB` | 比 `layer1` 降低到约 1/4，但仍明显需要复用 |

---

## 5. 结构特征总结

### 5.1 layer0

`layer0` 的主要特点：

1. `Cin=1`、`Rin=1`，输入规模极小。
2. `Cout=32`、`Rout=6`，输出扩展明显。
3. 空间分辨率仍为 `4 x 8`，保持较大的 chart 空间。
4. 输入端几何结构特殊，`PadIco` 与极点平滑的相对影响更突出。

对应的硬件含义是：

`layer0` 更适合作为前端特化层，而不是共享模板层。

### 5.2 layer1

`layer1` 的主要特点：

1. 已进入 `32 -> 32`、`6 -> 6` 的主干规模。
2. 空间分辨率仍保持在 `4 x 8`，因此计算量仍然很大。
3. 已经具备共享卷积块的多数特征，但保留了从前端过渡过来的结构位置。
4. 代码中已经开始出现 `IC_TILE` 和 `OC_TILE`，说明它是进入参数化架构的自然过渡点。

对应的硬件含义是：

`layer1` 不再适合纯前端特化写法，但也未必应直接与 `layer2-5` 完全同构。

### 5.3 layer2-5

`layer2-5` 的主要特点：

1. 固定为 `Cin=Cout=32`、`Rin=Rout=6`。
2. 空间尺寸降为 `2 x 4`。
3. 结构重复出现，便于统一成共享参数化块。
4. 与 `layer1` 相比，空间位置数缩小为 1/4，因此理论 MACs 也缩小为约 1/4。

对应的硬件含义是：

`layer2-5` 是参数化共享卷积核复用的最佳承载层。

---

## 6. 从 HLS 报告看总 latency 的层间差异

以下数值基于当前 `Target Clock = 5 ns` 估算。

| 层类型 | 总 latency (cycles) | 约合时间 | 顶层每帧 iteration latency (cycles) | 约合时间 |
|---|---:|---:|---:|---:|
| `layer0` | 2,001,222 | `10.006 ms` | 38,444 | `0.192 ms/frame` |
| `layer1` | 24,993,887,933 | `124.969 sec` | 480,651,691 | `2.403 sec/frame` |
| `layer2-5` | 6,213,631,061 | `31.068 sec` | 119,492,905 | `0.597 sec/frame` |

### 6.1 观察

1. `layer1` 的总 latency 极端偏高，说明其主干结构虽然功能正确，但尚未形成高效数据复用与并行映射。
2. `layer2-5` 相比 `layer1` 有显著下降，数量级上接近 1/4，这与其空间位置数从 `4 x 8` 降到 `2 x 4` 是一致的。
3. `layer0` latency 最低，但它不是共享块的代表，而是前端特化层的特例。

因此，从论文表达角度看，最有价值的比较不是“layer0 最快”，而是：

`在保持网络主干结构不变的前提下，如何让 layer1 和 layer2-5 的主干卷积块从高 latency 结构演化为可复用、高吞吐的参数化架构。`

---

## 7. 当前 latency 来源的三类拆解

本节将当前 HLS 报告中的主要瓶颈统一归为三类：

1. 累加相关
2. 输入端口冲突
3. 输出端口冲突

这样做的目的是让后续优化有清晰目标，而不是只看总 latency 数字。

---

## 8. layer0 的 latency 来源拆解

### 8.1 累加相关

对应逻辑：

- `pad_ico` 中 `smooth_north_pole_sum / smooth_south_pole_sum` 的逐项累加
- 代码位置集中在 [ico_conv_layer0.cpp](G:/3DSLED/icocnn/hls_src/HLS/layer0/ico_conv_layer0.cpp) 的 `83-93` 附近

HLS 证据：

- `pad_ico_Pipeline_VITIS_LOOP_87_2`
- `Final II = 9`
- 原因是 `smooth_south_pole_sum` 的 loop-carried dependence

含义：

这一部分不是主卷积本身，而是极点平滑求和逻辑造成的串行依赖。

### 8.2 输入端口冲突

对应逻辑：

- `pad_ico` 中对输入数组的多位置读取
- 代码位置集中在 [ico_conv_layer0.cpp](G:/3DSLED/icocnn/hls_src/HLS/layer0/ico_conv_layer0.cpp) 的 `83-115` 附近

HLS 证据：

- `pad_ico_Pipeline_VITIS_LOOP_51_11`
- 多条 `input_r_load_* ... due to limited memory ports`

含义：

尽管 `layer0` 输入规模很小，但 `PadIco` 的访问模式仍可能在同一拍内需要多个输入读端口。

### 8.3 输出端口冲突

对应逻辑：

- 输出极点清零与最终平滑回写
- 代码位置集中在 [ico_conv_layer0.cpp](G:/3DSLED/icocnn/hls_src/HLS/layer0/ico_conv_layer0.cpp) 的 `269-298` 附近

HLS 证据：

- `conv_ico_layer0_Pipeline_VITIS_LOOP_277_14_VITIS_LOOP_278_15`
- `Final II = 36`
- `output_r_load_*` 和 `output_r_addr_*_write` 都提示 `limited memory ports`

含义：

`layer0` 当前最重的单点瓶颈是输出后处理对 `output_r` 的读写竞争。

### 8.4 小结

`layer0` 的主卷积环节本身不是主要瓶颈，真正拉高 latency 的是：

1. 输出后处理端口冲突
2. 极点求和的串行累加
3. `PadIco` 的输入多端口访问

---

## 9. layer1 的 latency 来源拆解

### 9.1 累加相关

对应逻辑 1：

- `pad_ico` 中极点平滑求和
- 代码位置集中在 [ico_conv_layer1.cpp](G:/3DSLED/icocnn/hls_src/HLS/layer1/ico_conv_layer1.cpp) 的 `85-117` 附近

HLS 证据：

- `pad_ico_Pipeline_VITIS_LOOP_110_4_VITIS_LOOP_111_5`
- `Final II = 9`

对应逻辑 2：

- 主卷积 MAC 内的 `sum += ...`
- 代码位置集中在 [ico_conv_layer1.cpp](G:/3DSLED/icocnn/hls_src/HLS/layer1/ico_conv_layer1.cpp) 的 `264-279` 附近

HLS 证据：

- `conv_ico_layer1_Pipeline_VITIS_LOOP_272_14_...`
- `Final II = 9`
- `sum_1` 的 carried dependence 明确出现

含义：

`layer1` 已经开始表现出真正的主干卷积累加依赖瓶颈，这和 `layer0` 有明显不同。

### 9.2 输入端口冲突

对应逻辑：

- `pad_ico` 读入 `input_r`
- 代码位置集中在 [ico_conv_layer1.cpp](G:/3DSLED/icocnn/hls_src/HLS/layer1/ico_conv_layer1.cpp) 的 `93-145` 附近

HLS 证据：

- `pad_ico_Pipeline_VITIS_LOOP_89_2`
- `Final II = 30`
- 多条 `input_r_load_* ... limited memory ports`

含义：

在 `Cin=32`、`Rin=6` 后，`PadIco` 的输入端口压力已明显变成大瓶颈。

### 9.3 输出端口冲突

对应逻辑：

- 最终输出极点平滑回写
- 代码位置集中在 [ico_conv_layer1.cpp](G:/3DSLED/icocnn/hls_src/HLS/layer1/ico_conv_layer1.cpp) 的 `316-337` 附近

HLS 证据：

- `conv_ico_layer1_Pipeline_VITIS_LOOP_316_26_VITIS_LOOP_317_27`
- `Final II = 36`
- `output_r_load_*` 和 `output_r_addr_*_write` 都提示 `limited memory ports`

含义：

`layer1` 的输出后处理与 `layer0` 一样，仍是最重的端口冲突瓶颈之一。

### 9.4 小结

`layer1` 的 latency 来源已经同时具备三类特征：

1. 主 MAC 累加相关
2. `PadIco` 输入端口冲突
3. 输出后处理端口冲突

因此它是从前端特化层过渡到共享主干块时，最能暴露综合瓶颈的代表层。

---

## 10. layer2-5 的 latency 来源拆解

### 10.1 累加相关

对应逻辑 1：

- `pad_ico` 极点平滑求和
- 代码位置集中在 [ico_conv_layer2_5.cpp](G:/3DSLED/icocnn/hls_src/HLS/layer2-5/ico_conv_layer2_5.cpp) 的 `72-105` 附近

HLS 证据：

- `pad_ico_Pipeline_VITIS_LOOP_97_4_VITIS_LOOP_98_5`
- `Final II = 9`

对应逻辑 2：

- 主卷积 MAC 的 `sum += ...`
- 代码位置集中在 [ico_conv_layer2_5.cpp](G:/3DSLED/icocnn/hls_src/HLS/layer2-5/ico_conv_layer2_5.cpp) 的 `178-190` 附近

HLS 证据：

- `conv_ico_layer2_5_Pipeline_VITIS_LOOP_178_2_...`
- `Final II = 9`
- `sum` 的 carried dependence 明确存在

含义：

共享块已经把空间规模压下来了，但主 MAC 的串行累加依赖并没有自然消失。

### 10.2 输入端口冲突

对应逻辑：

- `pad_ico` 读取输入数组
- 代码位置集中在 [ico_conv_layer2_5.cpp](G:/3DSLED/icocnn/hls_src/HLS/layer2-5/ico_conv_layer2_5.cpp) 的 `80-120` 附近

HLS 证据：

- `pad_ico_Pipeline_VITIS_LOOP_76_2`
- `Final II = 27`
- 多条 `input_r_load_* ... limited memory ports`

含义：

虽然 `H x W` 已降为 `2 x 4`，但 `Cin=32`、`Rin=6` 仍使 `PadIco` 保持较高的读端口压力。

### 10.3 输出端口冲突

对应逻辑：

- 输出极点平滑的读写回写
- 代码位置集中在 [ico_conv_layer2_5.cpp](G:/3DSLED/icocnn/hls_src/HLS/layer2-5/ico_conv_layer2_5.cpp) 的 `212-233` 附近

HLS 证据：

- `conv_ico_layer2_5_Pipeline_VITIS_LOOP_212_14_VITIS_LOOP_213_15`
- `Final II = 33`
- `output_r_load_*` 与 `output_r_addr_*_write` 都提示 `limited memory ports`

含义：

即使主干共享块已经较 `layer1` 更规整，输出后处理仍然是最大的单点结构瓶颈。

### 10.4 小结

`layer2-5` 当前的主要 latency 来源非常清晰：

1. 输出端口冲突最重
2. 输入端口冲突次之
3. 主卷积与极点求和中的累加相关仍然显著

这恰好说明：

`layer2-5` 的下一步优化重点应该是数据流与存储组织，而不是继续追求更低的 Estimated Clock。`

---

## 11. 三类瓶颈的跨层对比表

| 层类型 | 累加相关 | 输入端口冲突 | 输出端口冲突 | 当前最重瓶颈 |
|---|---|---|---|---|
| `layer0` | `PadIco` 极点求和，`II=9` | `PadIco` 输入读冲突 | 输出极点回写，`II=36` | 输出端口冲突 |
| `layer1` | `PadIco` 求和 `II=9`；主 MAC `II=9` | `PadIco` 输入读冲突，`II=30` | 输出极点回写，`II=36` | 输出端口冲突，其次是输入端口冲突 |
| `layer2-5` | `PadIco` 求和 `II=9`；主 MAC `II=9` | `PadIco` 输入读冲突，`II=27` | 输出极点回写，`II=33` | 输出端口冲突，其次是输入端口冲突 |

---

## 12. 当前阶段可直接支撑论文的结论

基于上面的统一表格和 HLS 报告拆解，可以得到以下几条已经比较稳固的论文结论：

1. `layer0`、`layer1`、`layer2-5` 在计算规模和访存结构上差异显著，因此采用分层异构映射策略是合理的。

2. `layer1` 与 `layer2-5` 的主要问题不是时钟不够，而是总周期数过高；说明当前架构瓶颈的核心在数据流和存储组织，而不在关键路径时序。

3. `layer2-5` 相比 `layer1` 在理论 MACs 和顶层每帧 iteration latency 上均接近下降为 1/4，这与其空间规模从 `4 x 8` 降到 `2 x 4` 的结构变化一致，说明共享块建模具有良好的结构一致性基础。

4. 当前最值得投入的优化方向，不是泛化地“加 pragma”，而是围绕三类已明确的 latency 来源做结构化优化：
   1. 主 MAC 累加相关
   2. `PadIco` 输入端口冲突
   3. 输出极点后处理的读写端口冲突

---

## 13. 下一步建议

基于本文档，后续工作建议按以下顺序推进：

1. 先针对 `layer2-5` 主 MAC 累加相关做结构化修改。
2. 再重构输出极点后处理，避免对 `output_r` 的读写冲突。
3. 最后处理 `PadIco` 输入访问的局部缓存或分块组织。

理由是：

1. `layer2-5` 已经是共享参数化块，优化结果最适合上升为论文中的“通用主干架构方法”。
2. 它的结构最规整，最容易从优化中提炼出可复用的设计原则。
3. 其现有正确性链路和 HLS 报告都已较完整，适合形成连续实验。
