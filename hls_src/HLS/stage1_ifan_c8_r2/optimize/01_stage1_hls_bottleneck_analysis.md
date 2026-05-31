# Stage-1 HLS 瓶颈分析阶段文档 01

日期：2026-05-24  
对象：`hls_src/HLS/stage1_ifan_c8_r2`  
目标：解释当前 C8/R2 Stage-1 参数量较小但 HLS 综合卡在前端的原因，并固定后续优化方向。

## 1. 当前现象

当前 native C++ 数值链路已经闭环：

- Stage-1 baseline 已对齐 PyTorch `final_head_logits`：
  - `MaxAbsError = 2.30968e-005`
  - `RMSE = 1.80074e-006`
- FeatureMABA native 已对齐：
  - `output max_abs = 1.43051e-006`
  - `rmse = 1.93539e-007`
- post-MABA native 已对齐：
  - `channel_readout -> region_max -> CleanVertices -> SoftArgMax -> coords` 全链路 PASS

因此当前问题不是数值算法错误，也不是 C simulation 逻辑无法执行，而是 HLS C synthesis 的前端设计规模过大。

当前 `ifan_stage1_top` 的 HLS `csynth` 没有生成最终 `csynth.rpt` 资源表，只生成了 design-size report：

```text
stage1_ifan_c8_r2_hls_prj/sol1/syn/report/csynth_design_size.rpt
```

关键阶段规模如下：

| 阶段 | 指令数 | 说明 |
|---|---:|---|
| Compile/Link | 409,709 | C/C++ 合并后的整体设计规模 |
| Unroll/Inline step 1 | 1,790,028 | 用户 unroll/inline 后达到最高展开规模 |
| Unroll/Inline step 4 | 999,485 | 简化后仍接近百万级 |
| Array/Struct | 1,393,356 | 数组/结构体处理后再次膨胀 |
| Performance | pending | 尚未进入完整调度和资源绑定结果 |

这说明 Vitis HLS 卡住的位置在综合前端的设计展开、数组结构变换和性能调度之前。换句话说，工具还没有真正给出 LUT/DSP/BRAM 的最终估计，就已经被展开后的 IR 规模拖住。

## 2. 为什么参数量小仍然综合困难

这里需要区分三个概念：

1. 参数量  
   C8/R2 网络的权重数量确实比旧 baseline 小很多。

2. 激活和中间缓存  
   Stage-1 当前 top 在 C++ 层保留了大量 full-tensor static buffer，例如 PHAT/LMS direct/enhanced/attention/fused，以及 R2/R1 中间结果。

3. HLS 展开后的硬件结构规模  
   HLS 综合时看的不是 `.txt` 权重文件有多大，而是 C++ top 被 inline、unroll、array transform 之后形成了多少控制路径、访存路径、mux、转换逻辑和运算单元候选。

当前 Stage-1 的 C++ 代码虽然写成了“同类型函数复用”，但这只是在软件函数层复用。对 HLS 来说，`frontend_branch_engine`、`shared_attention_engine`、`fusion_block_engine` 在 top 中被多次调用，而且每个调用都有自己的静态中间数组和权重入口。工具会倾向于把这些调用点展开成多个硬件上下文，而不是自动理解为“同一套硬件按时间顺序复用”。

所以现在资源压力不是由参数量主导，而是由整网 top 的整体展开主导。权重展开是最明显的症状之一，但不是唯一原因。

## 3. 瓶颈归因

从 `csynth_design_size.rpt` 看，主要膨胀点集中在以下模块：

| 模块 | Compile/Link | Unroll/Inline | Array/Struct | 判断 |
|---|---:|---:|---:|---|
| `frontend_branch_engine` | 181,290 | 393,650 | 576,752 | PHAT/LMS 两支路的前端块被展开 |
| `shared_attention_engine` | 124,488 | 382,846 | 560,698 | attention 对 PHAT/LMS enhanced feature 分别调用 |
| `fusion_block_engine` | 73,924 | 219,222 | 250,558 | fusion/final 同型块重复进入 top |
| `ico_conv_r2_main_engine` | 113,768 | 381,524 | 559,376 | R2 主 IcoConv 是最重的内部计算块 |
| `ico_conv_r1_main_engine` | 56,884 | 190,728 | 222,258 | R1 fusion IcoConv 也明显膨胀 |

其中 `to_weight_t` 是权重路径展开最明显的局部证据：

```text
ico_conv_r2_main_engine:
  to_weight_t: 245,376 instructions (1728 calls)

ico_conv_r1_main_engine:
  to_weight_t: 122,688 instructions (864 calls)
```

这说明权重读取、定点转换、kernel index 选择逻辑位于较深的循环和展开路径中，随 `co/ro/ci/ri/kernel` 组合被反复实例化。它是当前综合规模膨胀的重要局部来源。

但更大的结构性问题是：Stage-1 top 没有显式时序调度层来约束复用。当前写法表达的是：

```text
PHAT frontend
LMS frontend
PHAT attention
LMS attention
fusion block x 4
final block
```

这些步骤在算法顺序上是串行的，但在 HLS 结构推断中，并没有被实现成“少量 IcoConv/TemporalConv 引擎按 block id 分时执行”。因此，工具面对的是一个整网级、多个同型调用点叠加的巨大 top，而不是一个小硬件引擎加调度 FSM。

当前瓶颈可以概括为四点：

- 主因：IcoConv 主体计算块在多个调用点被整体展开。
- 次因：权重读取、`to_weight_t`、kernel index 选择逻辑在内层循环中被反复实例化。
- 次因：多个 full-tensor static 中间 buffer 增加 Array/Struct 阶段压力。
- 次因：top-level 缺少显式时序调度器，HLS 无法自动把同型函数调用收敛成单硬件单元分时复用。

## 4. baseline HLS 做了什么

这里的 baseline 主要指之前 `layer2-5` 的 HLS 实现。它的目标不是把整网一次性静态拼接成一个 top，而是先把单层 IcoConv 做成可综合、可观察、可调 tile 的硬件模块。

已记录的 `layer2-5` 关键结果如下：

```text
hls_src/layer2-5_details/2026-03-24_154758_csynth_key.md
```

资源和时延快照：

| 指标 | 数值 |
|---|---:|
| Target Clock | 5.00 ns |
| Estimated Clock | 4.209 ns |
| Total Latency | 261686621 cycles |
| BRAM_18K | 64 |
| DSP | 72 |
| FF | 43639 |
| LUT | 69382 |

`layer2-5` baseline 的核心价值不是参数更多或更少，而是结构上做了以下事情：

- 单层/模块级综合，而不是整网级综合。
- 使用 `OC_TILE=2` 控制输出通道并行度，避免一次铺开所有输出通道。
- 将计算拆成 staging、PadIco、partial sum、post-process、writeback 等阶段。
- 使用局部 tile buffer 和 partial buffer，减少全量中间缓存常驻。
- 将关键循环组织成可以达到 `Final II = 1` 的流水段。
- 明确让一个计算模块在时间上处理多个 tile，而不是让 HLS 自动展开多个同型 block。

这也是为什么 `layer2-5` 参数量更大、通道更多，但能完成 `csynth`；当前 C8/R2 Stage-1 参数量更少，却在整网 top 前端卡住。两者的差别不在参数规模，而在硬件表达粒度和调度方式。

## 5. 后续优化路线

### 优先级 1：建立 Stage-1 显式调度层

把当前整网 top 改为“少量硬件引擎 + block 调度 FSM”的结构。目标是让 PHAT/LMS、attention、fusion/final blocks 分时复用同一类 IcoConv/TemporalConv 引擎，而不是让每个调用点都形成展开后的硬件上下文。

建议调度单位：

```text
FRONTEND_PHAT_STEM
FRONTEND_PHAT_RES0
FRONTEND_PHAT_RES1
FRONTEND_LMS_STEM
FRONTEND_LMS_RES0
FRONTEND_LMS_RES1
ATTN_PHAT_0
ATTN_PHAT_1
ATTN_LMS_0
ATTN_LMS_1
FUSION_0..3
FINAL
```

调度层要显式传入当前 block 的权重、bias、norm 参数、输入 buffer 和输出 buffer。

### 优先级 2：把权重 staging 移出最深展开路径

当前 `to_weight_t` 在 IcoConv 内层循环中大量重复出现。后续应先按当前 tile 或当前 `(co, ro, ci, ri)` 组合预取并转换权重，再进入核心 MAC 循环。

目标是把：

```text
weight float -> weight_t
kernel_idx -> rotated kernel selection
```

从最深的计算路径里拆出来，变成 tile 级或 block 级 staging。

### 优先级 3：减少 top-level full-tensor static buffer

当前 top 中的 PHAT/LMS direct/enhanced/attention/fused 等 R2 full tensor buffer 会增加 Array/Struct 阶段压力。后续应改成：

- R2/R1 ping-pong buffer
- tile buffer
- block 间 writeback/readback
- 只保留跨 block 必需的状态

第一版不追求端到端最低 latency，先让资源闭合和 `csynth.rpt` 稳定生成。

### 优先级 4：MABA 和 post-MABA 分别综合

FeatureMABA 和 post-MABA native 已经对齐，但暂时不应直接塞进当前 Stage-1 top。原因是 Stage-1 baseline 自身还没有稳定完成 `csynth`，直接合并会掩盖瓶颈来源。

建议后续顺序：

1. Stage-1 baseline 独立产出 `csynth.rpt`。
2. FeatureMABA 独立 top 产出 `csynth.rpt`。
3. post-MABA 独立 top 产出 `csynth.rpt`。
4. 再讨论是否顶层串接，或保留 FPGA/CPU 分工边界。

## 6. 验收标准

下一阶段优化不应以“最终 LUT/DSP/BRAM 百分比最低”为第一目标，而应先完成以下闭环：

1. `ifan_stage1_top` 能稳定完成 `csynth_design`。
2. 生成最终 `ifan_stage1_top_csynth.rpt` 或等价 top-level csynth 报告。
3. 报告中记录：
   - Estimated Clock
   - Total Latency
   - BRAM_18K
   - DSP
   - FF
   - LUT
   - 关键 loop II
4. native 数值对齐仍保持：
   - Stage-1 baseline PASS
   - FeatureMABA PASS
   - post-MABA PASS

只有完成这些之后，再比较 LUT/DSP/BRAM 占比是否满足 `xc7k325t` 的落地要求。

## 7. 文档归档约定

后续所有 Stage-1 优化分析、实验记录和阶段报告都保存到：

```text
hls_src/HLS/stage1_ifan_c8_r2/optimize/
```

建议命名规则：

```text
NN_topic_summary.md
```

例如：

```text
01_stage1_hls_bottleneck_analysis.md
02_stage1_scheduler_design.md
03_weight_staging_experiment.md
04_tile_buffer_csynth_report.md
```

每份文档至少记录：

- 本阶段目标
- 修改或实验范围
- 使用的 HLS/native 命令
- 关键报告路径
- 数值对齐结果
- 资源/latency/II 结果
- 当前结论和下一步

## 8. 当前结论

当前 C8/R2 Stage-1 的瓶颈不是模型参数量，而是 HLS 表达方式。软件层“函数复用”没有自动变成硬件层“单元复用”。现在的 top 把多个同型 IcoConv/TemporalConv/elementwise block 串成一个大静态设计，导致 HLS 在 unroll、inline、array transform 阶段形成百万级 IR。

下一步真正要做的不是继续把模块往 top 里接，而是先补上 Stage-1 调度层，把 baseline 中已经验证有效的 tile、staging、partial sum、writeback 和显式资源复用思想移植到当前 C8/R2 整体网络。
