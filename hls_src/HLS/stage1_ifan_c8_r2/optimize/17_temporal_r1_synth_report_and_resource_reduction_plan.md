# Temporal R1 综合报告与降资源方案 17

日期：2026-06-02

## 1. 与报告 16 的衔接

报告 16 已经完成：

1. `temporal_r1` 真实 replay 对齐；
2. `temporal_r1` 独立真实数据导出；
3. `temporal_r1` native 对齐；
4. `temporal_r1` 独立 `csim` 跑通，并获得首个真实定点误差锚点。

因此，本轮目标转为：

```text
跑通 temporal_r1 独立综合
-> 拿到资源/时序/II 报告
-> 分析为什么 LUT 高、DSP 低
-> 给出下一轮降资源与 DSP 分担方案
```

## 2. 综合结果

### 2.1 报告路径

本轮 `temporal_r1` 独立综合快照在：

- `hls_src/hls_reports/stage1_ifan_c8_r2_temporal_r1_hls_prj_sol1_20260602_160047/summary.md`
- `hls_src/hls_reports/stage1_ifan_c8_r2_temporal_r1_hls_prj_sol1_20260602_160047/ifan_temporal_r1_top_csynth.rpt`
- `hls_src/hls_reports/stage1_ifan_c8_r2_temporal_r1_hls_prj_sol1_20260602_160047/ifan_temporal_r1_top_csynth.xml`
- `hls_src/hls_reports/stage1_ifan_c8_r2_temporal_r1_hls_prj_sol1_20260602_160047/csynth_design_size.rpt`
- `hls_src/hls_reports/stage1_ifan_c8_r2_temporal_r1_hls_prj_sol1_20260602_160047/csynth_design_size.xml`

最新汇总入口为：

- `hls_src/hls_reports/stage1_ifan_c8_r2_latest_summary.md`

### 2.2 关键指标

从 `summary.md` 读取：

| 指标 | 数值 |
|---|---:|
| Target clock | `5.00 ns` |
| Estimated clock | `5.083 ns` |
| Latency | `330,914 cycles` |
| II | `330,915` |
| BRAM_18K | `0` |
| DSP | `9` |
| FF | `71,440` |
| LUT | `152,949` |

资源利用率：

| 资源 | 使用率 |
|---|---:|
| DSP | `1.07%` |
| FF | `17.53%` |
| LUT | `75.05%` |

### 2.3 初步判断

这组结果说明：

1. `temporal_r1` 已经具备独立综合能力；
2. 但当前结构的 QoR 很差：
   - LUT 极高；
   - DSP 极低；
   - 时钟略超预算；
   - 关键 loop 的 `II` 远未闭合。

因此，本轮综合的价值更多是“问题暴露”和“优化锚点建立”，而不是说明当前结构已经适合直接纳入主线。

## 3. 关键瓶颈分析

### 3.1 为什么 LUT 占用这么多

当前 LUT 很高，不是因为 `TemporalConv1d` 数学上很复杂，而是因为当前实现方式让大量控制、寻址、乘加选择逻辑落在 LUT 上。

本轮最主要的 LUT 膨胀来源有四类：

#### 1. 输入数组 `input_r` 访问端口冲突导致的调度和控制逻辑膨胀

报告中最核心的 warning 是：

```text
Unable to schedule 'load' operation ... on array 'input_r' due to limited memory ports
Final II = 36
```

也就是：

- 内层 pipeline 试图在一个拍内读取很多 `input[src_t][ci][ri][ch][h][w]`
- 但顶层 `input_r` 只是普通 `ap_memory`
- 可用读端口不足
- HLS 被迫插入复杂的调度、仲裁、控制与状态逻辑

这类“读端口不足 + 强行保 pipeline”通常首先炸的是：

- LUT
- FF
- 控制扇出

而不是 DSP。

#### 2. 内层 `(ci, k)` 完全展开，使乘加网络变成超宽组合树

当前核心热点在：

```cpp
for (int ci = 0; ci < 8; ci++) {
    for (int k = 0; k < 5; k++) {
#pragma HLS UNROLL
        ...
        sum += input * staged_weight;
    }
}
```

这里相当于：

- 每个输出通道、每个时间点、每个空间位置
- 一次性展开 `8 * 5 = 40` 个乘法项
- 再做一棵 40 输入加法树

如果这些乘法/加法不能稳定映射到 DSP 或 DSP 链，HLS 就会用大量：

- LUT adder tree
- 多路选择器
- 中间寄存

于是 LUT 和 FF 会急剧上升。

#### 3. `src_t >= 0` 条件与 variable-indexed 访问引入大量条件逻辑

综合日志里有：

```text
variable-indexed range selection may cause suboptimal QoR
Performing if-conversion ...
converting 2477 basic blocks
```

说明当前结构里：

- `src_t = t - (K - 1) + k`
- `if (src_t >= 0)`

这种随时间变化的有效窗判断，被 HLS 展开后形成了大量条件选择逻辑。

这类逻辑通常不会上 DSP，主要会消耗：

- LUT
- 控制网络
- 比较器/选择器

#### 4. 顶层接口被综合成大量分裂 memory 端口

从接口报告可以看到：

- `input_r` 是 `ap_memory`
- `weight_0_0 ... weight_1_4 ...`
- `bias_0 ... bias_1 ...`

这说明权重/偏置在顶层接口侧也被拆成了很多 memory 端口。

虽然这本身不一定直接导致 15 万 LUT，但会放大：

- 地址生成
- 读使能控制
- 端口相关状态逻辑

对整体 LUT 不利。

## 4. 为什么 DSP 只有 9 个

当前 DSP 只有 9，不是因为算子不需要乘法，而是因为现在的结构并没有把大量 MAC 稳定压进 DSP 主路径。

主要原因：

### 4.1 输入端口冲突先限制了并行有效性

虽然 `(ci, k)` 展开了很多乘法候选，但 `input_r` 读端口不足，导致：

- HLS 实际调度无法在同拍真实喂饱所有乘法
- 最终 pipeline 退化
- 大量计算结构变成被控制逻辑包围的稀疏执行

这会降低 DSP 高效映射的机会。

### 4.2 当前乘加树更像“宽组合表达式”，而不是结构清晰的 DSP MAC 阵列

代码现在写法是：

```cpp
sum += input * staged_weight;
```

但由于：

- 完全展开；
- 条件有效窗；
- variable index；
- `acc_t` 位宽较宽；

HLS 更容易得到的是大块组合表达式，而不是规则、均匀、可级联的 DSP MAC 链。

所以结果是：

- 少量乘法上 DSP
- 大量加法/选择/控制落 LUT

### 4.3 加法树本身大量消耗 LUT

即使乘法部分用了部分 DSP，40 项展开后的归并加法树如果没有被设计成明确的分级流水 MAC，也仍然会主要消耗 LUT。

因此“DSP 低、LUT 高”并不矛盾，它恰恰说明：

```text
当前结构没有把 temporal kernel 组织成 DSP-friendly 的数据流
```

## 5. 设计规模信息

从 `csynth_design_size.rpt` 可见：

| 项 | 指令数 |
|---|---:|
| Compile/Link | `3,549` |
| Unroll/Inline peak | `28,813` |
| HW Transforms final | `15,970` |

函数热点集中在：

- `ifan_temporal_r1_engine`
- `stage_temporal_weight_tile`

这说明当前问题并不在 testbench 或外层壳层，而就在 temporal 主核本身。

## 6. 降资源方案

下面按“优先级从高到低”给出建议。

### 方案 A：先解决 `input_r` 端口瓶颈

这是当前最优先的动作。

#### 建议做法

在进入 `t` 主循环前，把当前 `(ri, ch, h, w)` 位置的整条时间序列先搬到本地小缓冲：

```text
input_timebuf[T=6][C=8]
```

或者更进一步：

```text
input_windowbuf[K=5][C=8]
```

这样做的好处：

1. 外部大数组 `input_r` 的访问次数大幅减少；
2. 内层 MAC 读的是本地寄存器/小 buffer，而不是多维大数组；
3. `II` 受 memory port 限制的问题会明显缓和；
4. LUT/FF 会先下降一大截。

这是当前最应该先做的一步。

### 方案 B：把 `(ci, k)` 的完全展开改成分层展开

当前 `8 * 5 = 40` 项完全展开太激进。

建议尝试：

#### B1. 只展开 `k`

```text
UNROLL k=5
PIPELINE ci
```

意思是：

- 保留 5 tap 的时间卷积并行
- 但不把 8 个输入通道也一次性全展开

这样会显著减小：

- 加法树宽度
- 组合扇出
- LUT 压力

代价是延迟增加，但通常比当前 `II=36` 更可控。

#### B2. 对 `ci` 做小因子展开，例如 `factor=2` 或 `4`

这会形成：

```text
5 * 2 = 10
或
5 * 4 = 20
```

规模比 40 项完全展开更容易映射到 DSP + LUT 混合结构。

### 方案 C：显式把乘法路径往 DSP 上推

当前 DSP 利用率太低，可以尝试更明确地引导。

#### C1. 对核心乘法加 DSP 绑定 pragma

例如在乘法热点周围尝试：

```cpp
#pragma HLS BIND_OP variable=... op=mul impl=dsp
```

或者等价的资源约束方式，让：

- `input * staged_weight`

优先映射到 DSP。

#### C2. 缩窄乘法位宽，保持 DSP48E1 友好

当前乘法日志显示是：

```text
mul_40s_25s_65
```

说明乘法前后的位宽已经被拉得比较宽。

建议检查：

- `input_t`
- `weight_t`
- `acc_t`

是否能重新组织为：

- 乘法阶段尽量窄
- 累加阶段分层扩位

不要让所有乘法直接进入超宽位乘法表达式。

更理想的结构是：

```text
narrow input_t * narrow weight_t -> dsp product
partial sums in local acc tree
late widen / final accumulate
```

#### C3. 先做“乘法在 DSP，归并分级流水”

如果想进一步吃 DSP，可以把 40 项 MAC 不写成一个大表达式，而改成：

1. 第一层：小组 partial sums
2. 第二层：组间归并
3. 每层之间插入流水

这样更像 DSP-friendly 的 staged reduction，而不是 LUT-heavy 的大组合树。

### 方案 D：显式做时间窗 staging，去掉 `src_t >= 0` 条件

当前 `if (src_t >= 0)` 会制造很多条件控制逻辑。

建议改为：

1. 先把输入按 causal 规则预填充到一个带零前缀的时间缓冲；
2. 内层 MAC 直接访问：

```text
window[t][k][ci]
```

而不在 MAC 内再做 `if (src_t >= 0)`。

这样能减少：

- 比较器
- 条件选择器
- if-conversion 产生的大量基本块

对 LUT 非常有帮助。

### 方案 E：必要时降低 `OC_TILE`

当前 `IFAN_OC_TILE = IFAN_OC_PAR_FACTOR = 2`。

如果前面几项做完后 LUT 仍高，可以尝试：

```text
OC_TILE: 2 -> 1
```

这会减少：

- staged_weight 的并行宽度
- 同拍输出通道并行数
- 相应的控制与加法树压力

代价是吞吐下降，但很可能对 LUT 有立竿见影的帮助。

## 7. 推荐执行顺序

建议不要同时乱改，按下面顺序做最稳：

1. 先做 `input_r` 本地时间缓冲 staging
2. 去掉内层 `src_t >= 0` 条件判断
3. 将 `ci,k` 的完全展开改成：
   - `k` 全展开
   - `ci` 部分展开或不展开
4. 再尝试乘法绑定到 DSP
5. 如果还不够，再降低 `OC_TILE`

这是因为：

- A/B/D 主要解决 LUT 爆炸和 II 失控
- C 主要解决 DSP 利用率过低
- E 是保守兜底手段

## 8. 当前最合理的解释总结

当前 `temporal_r1` 出现：

```text
DSP 很低
LUT 很高
II 很差
```

本质原因不是 temporal 算法天然不适合硬件，而是：

1. 输入访问没有 staging，本地端口不够；
2. 内层完全展开过度；
3. 条件有效窗逻辑混入 MAC 热路径；
4. 乘加结构没有组织成 DSP-friendly 的分级数据流。

所以，当前 LUT 高主要是：

```text
memory-port 受限 + wide unroll + conditional MAC + LUT adder/control tree
```

而不是因为这 8x8x5 的 temporal 卷积本身计算量不可接受。

## 9. 下一步建议

下一轮建议直接做一个最小结构优化实验：

1. 给 `temporal_r1` 增加本地 `time/channel` staging buffer；
2. 去掉 `src_t >= 0` 的内层条件；
3. 保留 `k=5` 展开，但把 `ci=8` 完全展开改成部分展开；
4. 重跑：
   - native
   - csim
   - synth

重点观察三项：

1. LUT 是否明显下降；
2. DSP 是否上升；
3. `VITIS_LOOP_45_6` 的 `II` 是否从 `36` 明显下降。

如果你愿意，我下一步可以直接开始做这个“temporal_r1 第一轮降资源结构实验”，而不是只停在报告分析。*** End Patch
