# Stage-1 显式调度与资源复用设计 03

日期：2026-05-31  
对象：`hls_src/HLS/stage1_ifan_c8_r2`  
阶段目标：把 `01_stage1_hls_bottleneck_analysis.md` 中确认的瓶颈，进一步收敛为下一轮可实施的 HLS 重构边界、调度表、buffer 策略和验收标准。

## 1. 本阶段定位

当前 `ifan_stage1_top` 的数值链路已经可以完成 native / C simulation smoke，但 HLS `csynth` 仍停在 design-size 阶段。瓶颈不是 C8/R2 参数量，而是整网 top 的静态表达方式。

本阶段不继续把 MABA 或 post-MABA 接入当前 top，也不直接追求最低 latency。当前阶段的核心工作是固定一版“显式调度层 + 少量可复用 engine + 可逐步收缩 buffer”的设计，使后续代码修改有清晰边界。

## 2. 当前源码中的问题落点

关键源码位置：

```text
ifan_stage1.cpp:43   frontend_branch_engine
ifan_stage1.cpp:71   shared_attention_engine
ifan_stage1.cpp:91   fusion_block_engine
ifan_stage1.cpp:116  ifan_stage1_top

ifan_stage1_engines.cpp:558  ico_conv_r2_main_engine
ifan_stage1_engines.cpp:622  ico_conv_r1_main_engine
ifan_stage1_engines.cpp:737  temporal_conv1d_r1_engine
```

当前 top 仍然表达为：

```text
extract PHAT/LMS
PHAT frontend
LMS frontend
PHAT attention
LMS attention
attention fuse + add
pool R2->R1
fusion block x4
final block
```

这在算法顺序上是正确的，但对 HLS 来说仍像一个整网级静态 top。`csynth_design_size.rpt` 显示：

```text
Compile/Link     409,709 instructions
Unroll step 1  1,790,028 instructions
Unroll/Inline    999,485 instructions after simplification
Array/Struct   1,393,356 instructions
Performance    pending
```

`to_weight_t` 的局部膨胀仍非常明显：

```text
ico_conv_r2_main_engine: to_weight_t 245,376 instructions, 1728 calls
ico_conv_r1_main_engine: to_weight_t 122,688 instructions, 864 calls
```

因此本阶段设计优先解决两件事：

- 让 HLS 看到“同一类 engine 被调度复用”，而不是多个静态调用点叠加。
- 把 weight conversion / kernel index selection 从最内层 MAC 路径中移出，至少形成 tile 级 staging 实验入口。

## 3. 建议的调度单元

调度层建议定义统一的 stage op 枚举，而不是继续在 top 里直接展开一长串函数调用。

```cpp
enum Stage1Op {
    OP_FRONTEND_PHAT_STEM,
    OP_FRONTEND_PHAT_RES0,
    OP_FRONTEND_PHAT_RES1,
    OP_FRONTEND_LMS_STEM,
    OP_FRONTEND_LMS_RES0,
    OP_FRONTEND_LMS_RES1,
    OP_ATTN_PHAT_NORM,
    OP_ATTN_PHAT_CONV0,
    OP_ATTN_PHAT_CONV1,
    OP_ATTN_LMS_NORM,
    OP_ATTN_LMS_CONV0,
    OP_ATTN_LMS_CONV1,
    OP_ATTN_FUSE_PHAT,
    OP_ATTN_FUSE_LMS,
    OP_BRANCH_ADD,
    OP_POOL_R2_TO_R1,
    OP_FUSION_0,
    OP_FUSION_1,
    OP_FUSION_2,
    OP_FUSION_3,
    OP_FINAL
};
```

第一版可以不做复杂动态 FSM，而是用静态调度表驱动：

```cpp
static const Stage1Op STAGE1_SCHEDULE[] = {
    OP_FRONTEND_PHAT_STEM,
    OP_FRONTEND_PHAT_RES0,
    OP_FRONTEND_PHAT_RES1,
    OP_FRONTEND_LMS_STEM,
    OP_FRONTEND_LMS_RES0,
    OP_FRONTEND_LMS_RES1,
    OP_ATTN_PHAT_NORM,
    OP_ATTN_PHAT_CONV0,
    OP_ATTN_PHAT_CONV1,
    OP_ATTN_LMS_NORM,
    OP_ATTN_LMS_CONV0,
    OP_ATTN_LMS_CONV1,
    OP_ATTN_FUSE_PHAT,
    OP_ATTN_FUSE_LMS,
    OP_BRANCH_ADD,
    OP_POOL_R2_TO_R1,
    OP_FUSION_0,
    OP_FUSION_1,
    OP_FUSION_2,
    OP_FUSION_3,
    OP_FINAL
};
```

这样做的目的不是引入运行时复杂度，而是在 C++ 表达上把“当前执行哪个 block、使用哪组权重、读哪个 buffer、写哪个 buffer”显式化。

## 4. 复用 engine 边界

建议下一轮代码重构只保留少量核心 engine 边界：

| Engine | 覆盖任务 | 当前来源 |
|---|---|---|
| `stage1_r2_stem_engine` | PHAT/LMS stem | `ico_conv_r2_stem_engine` |
| `stage1_r2_main_engine` | frontend residual、attention conv | `ico_conv_r2_main_engine` |
| `stage1_r1_main_engine` | fusion/final IcoConv | `ico_conv_r1_main_engine` |
| `stage1_temporal_r1_engine` | fusion/final temporal conv | `temporal_conv1d_r1_engine` |
| `stage1_lnorm_engine` | R2/R1 LNormIco | `lnorm_ico_r2_engine`, `lnorm_ico_r1_engine` |
| `stage1_elementwise_engine` | ReLU、sigmoid、residual、attention fuse、add | 当前 elementwise functions |
| `stage1_pool_engine` | R2 到 R1 pooling | `pool_ico_r2_to_r1_engine` |

重点是：`frontend_branch_engine`、`shared_attention_engine`、`fusion_block_engine` 不再作为 top 中的大粒度静态 block 承载硬件结构，而是退化为调度层中的 op 组合或 wrapper。

## 5. 权重 staging 第一版

当前 `ico_conv_r2_main_engine` 与 `ico_conv_r1_main_engine` 在最内层计算中执行：

```cpp
idx = kernel_idx[co][ro][ci][ri][k][...]
to_weight_t(weight[idx_co][idx_ci][idx_ri][idx_w])
```

下一轮建议先做 tile 级 staging，不直接追求完整权重常驻。

### 5.1 R2 main staging

建议在进入 `ch/h/w` 空间循环前，对当前 `(co, ro)` 预取：

```text
staged_weight[CI=8][RI=6][K=9]
staged_valid [CI=8][RI=6][K=9]
```

预取逻辑处理：

```text
kernel_idx -> idx_co/idx_ci/idx_ri/idx_w
idx_w valid check
float weight -> weight_t
```

MAC 内层只执行：

```text
if staged_valid[ci][ri][k]:
    sum += padded[...] * staged_weight[ci][ri][k]
```

这样 `to_weight_t` 和 index 解码不再被每个空间位置重复实例化。

### 5.2 R1 main staging

R1 main 与 R2 main 同构，但空间尺寸更小。建议复用同一个 staging 结构，只通过 engine variant 或 template 参数区分 `H/W/reorder`。

### 5.3 Temporal staging

`temporal_conv1d_r1_engine` 中也有 `to_weight_t(weight[co][ci][k])`，但规模明显小于 IcoConv。建议第二优先级处理：

```text
staged_temporal_weight[CO=8][CI=8][K=5]
```

第一轮可以只 stage 当前 `co` 的 `[CI][K]`，避免过多局部 partition。

## 6. Buffer 收缩路径

当前 top 保留了大量 full-tensor static buffer：

```text
phat_direct
phat_enhanced
lms_direct
lms_enhanced
phat_attention
lms_attention
phat_fused
lms_fused
fused_r2
fused_r1_a
fused_r1_b
```

建议分三步收缩，不在第一轮一次改到底。

### 6.1 第一轮：R1 ping-pong 固化

保留：

```text
r1_ping
r1_pong
```

fusion block 之间只 ping-pong，不再用语义名隐含 block 级静态输出。

### 6.2 第二轮：attention buffer 合并

PHAT/LMS attention 可以按分支顺序处理：

```text
branch_direct
branch_enhanced
branch_attention_or_tmp
branch_fused
```

PHAT fused 写入 `fused_r2_accum`，LMS fused 直接累加到同一个 `fused_r2_accum`。这样可去掉 `phat_fused`、`lms_fused` 和单独的 `add_feature_r2` 输出 buffer。

### 6.3 第三轮：frontend 分支时分复用

当 scheduler 稳定后，PHAT/LMS frontend 可以共享：

```text
branch_input
branch_direct
branch_enhanced
```

PHAT 的结果需要保留到 attention fuse 完成；LMS 处理完成后再进入 LMS attention。第一版不强制消除全部 R2 buffer，先保证 `csynth.rpt` 能生成。

## 7. HLS pragma 原则

本阶段建议先从“减少 frontend 展开规模”出发调整 pragma，而不是继续增加 unroll。

建议原则：

- top-level wrapper 保持 `#pragma HLS INLINE off`。
- 大 engine 边界保持 `INLINE off`，避免 top 继续吞并全部调用体。
- 对 `kh/kw` 的完全 unroll 暂时保留，但 staging 后复查 design-size。
- 对 staging buffer 小维度可 selective partition，不对 full tensor 做完整 partition。
- 暂不引入 top-level `DATAFLOW`，先让顺序调度生成最终 `csynth.rpt`。

## 8. 验收标准

本阶段后的下一轮代码工作不以 latency 最优为第一目标，而以以下闭环为准：

1. native smoke 仍 PASS。
2. `run_hls.bat csim` 或等价 C simulation 仍 PASS。
3. `run_hls.bat synth` 能生成最终 top-level `csynth.rpt`。
4. 报告中记录：
   - Estimated Clock
   - Total Latency
   - BRAM_18K
   - DSP
   - FF
   - LUT
   - 关键 loop II
5. 若仍只能生成 `csynth_design_size.rpt`，必须比较设计规模是否下降：
   - Compile/Link instructions
   - Unroll/Inline instructions
   - Array/Struct instructions
   - `to_weight_t` call/instruction 数

## 9. 建议下一步代码提交边界

建议下一次实际源码修改拆成三个小提交或三个阶段：

1. **调度表与 wrapper 边界**
   - 新增 `Stage1Op` 和静态 schedule。
   - 先不改数值路径，只把 fusion/final 的 block id、权重选择、输入输出 buffer 显式化。

2. **IcoConv weight staging**
   - 新增 R2/R1 main staging 内核。
   - 先保持函数签名接近当前实现，减少 testbench 改动。
   - 对比 `csynth_design_size.rpt` 中 `to_weight_t` 规模。

3. **R1 ping-pong 与 attention buffer 合并**
   - 优先收缩 R1 fusion/final 中间 buffer。
   - 再处理 PHAT/LMS attention fuse 的 R2 buffer。

## 10. 当前结论

本阶段的关键结论是：Stage-1 下一步不应再沿着“新增模块接入当前 top”推进，而应先把当前 top 改造成显式调度表达。只有当 HLS 能看到少量 engine 被按 op 顺序分时复用，后续的 weight staging、ping-pong buffer 和 MABA 分阶段综合才有稳定基线。

