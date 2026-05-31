# Stage-1 当前优化阶段总结 04

日期：2026-05-31  
对象：`hls_src/HLS/stage1_ifan_c8_r2`  
阶段输入文档：

```text
optimize/01_stage1_hls_bottleneck_analysis.md
optimize/02_stage1_hardware_architecture_diagram_spec.md
```

## 1. 本阶段目标

根据 `01_stage1_hls_bottleneck_analysis.md` 的瓶颈判断，继续推进 Stage-1 HLS 优化阶段工作，明确下一轮实现应从“整网静态串接 top”转向“显式调度层 + engine 复用 + weight staging + buffer 收缩”。

本阶段不修改现有 C++ 数值路径，原因是当前工作区已有多处未提交源码和生成物；本阶段先固化设计边界和验收口径，避免在尚未形成 `csynth.rpt` 的情况下扩大代码改动范围。

## 2. 本阶段完成内容

新增阶段设计文档：

```text
hls_src/HLS/stage1_ifan_c8_r2/optimize/03_stage1_scheduler_reuse_design.md
```

该文档完成了：

- 复核当前 `ifan_stage1_top` 的静态调用结构。
- 固定建议的 `Stage1Op` 调度单元。
- 给出第一版静态调度表。
- 明确下一轮应保留的少量 engine 边界。
- 给出 R2/R1 IcoConv 的 tile 级 weight staging 方案。
- 给出 R1 ping-pong、attention buffer 合并、frontend 分支复用的分阶段 buffer 收缩路径。
- 固定后续 `csynth` 与 design-size 对比的验收标准。

新增本阶段总结文档：

```text
hls_src/HLS/stage1_ifan_c8_r2/optimize/04_stage1_current_phase_summary.md
```

## 3. 当前源码复核结果

当前 top 级结构仍位于：

```text
hls_src/HLS/stage1_ifan_c8_r2/ifan_stage1.cpp
```

关键函数：

```text
frontend_branch_engine       ifan_stage1.cpp:43
shared_attention_engine      ifan_stage1.cpp:71
fusion_block_engine          ifan_stage1.cpp:91
ifan_stage1_top              ifan_stage1.cpp:116
```

当前主要 engine 位于：

```text
hls_src/HLS/stage1_ifan_c8_r2/ifan_stage1_engines.cpp
```

关键函数：

```text
ico_conv_r2_main_engine      ifan_stage1_engines.cpp:558
ico_conv_r1_main_engine      ifan_stage1_engines.cpp:622
temporal_conv1d_r1_engine    ifan_stage1_engines.cpp:737
lnorm_ico_r2_engine          ifan_stage1_engines.cpp:769
lnorm_ico_r1_engine          ifan_stage1_engines.cpp:809
```

复核结论：

- PHAT/LMS frontend 是两个同构调用路径。
- Shared attention 虽然算法上共享权重，但当前 HLS top 中仍分别调用 PHAT/LMS 路径。
- Fusion block x4 与 final block 复用同类 R1 IcoConv / TemporalConv / LNorm 结构，但 top 表达仍没有统一调度层。
- `to_weight_t(weight[...])` 和 `kernel_idx[...]` 仍在 IcoConv main 的最内层 MAC 路径中。

## 4. 当前报告复核结果

现有 HLS design-size 报告路径：

```text
hls_src/HLS/stage1_ifan_c8_r2/stage1_ifan_c8_r2_hls_prj/sol1/syn/report/csynth_design_size.rpt
```

报告状态：

```text
C-Synthesis has not completed
Performance pending
```

关键规模：

| Phase | Instructions |
|---|---:|
| Compile/Link | 409,709 |
| Unroll step 1 | 1,790,028 |
| Unroll/Inline after simplification | 999,485 |
| Array/Struct | 1,393,356 |

关键函数规模：

| Function | Compile/Link | Unroll/Inline | Array/Struct |
|---|---:|---:|---:|
| `frontend_branch_engine` | 181,290 | 393,650 | 576,752 |
| `shared_attention_engine` | 124,488 | 382,846 | 560,698 |
| `fusion_block_engine` | 73,924 | 219,222 | 250,558 |
| `ico_conv_r2_main_engine` | 113,768 | 381,524 | 559,376 |
| `ico_conv_r1_main_engine` | 56,884 | 190,728 | 222,258 |

权重转换路径：

| Function | `to_weight_t` evidence |
|---|---:|
| `ico_conv_r2_main_engine` | 245,376 instructions, 1728 calls |
| `ico_conv_r1_main_engine` | 122,688 instructions, 864 calls |
| `temporal_conv1d_r1_engine` | 11,360 instructions, 80 calls |

结论：`01` 文档中的瓶颈判断仍成立，当前最优先的代码优化不是继续接模块，而是约束 HLS 前端看到的设计规模。

## 5. 当前 C simulation 复核结果

现有 C simulation 报告路径：

```text
hls_src/HLS/stage1_ifan_c8_r2/stage1_ifan_c8_r2_hls_prj/sol1/csim/report/ifan_stage1_top_csim.log
```

当前记录：

```text
Real Stage-1 data not found; using synthetic smoke data.
IFAN Stage-1 HLS smoke test
Output shape: [6, 8, 6, 5, 2, 4]
Checksum: -0.108568
AbsSum: 0.398083
Min/Max: -0.000150789 / 0.000150789
PASS
C Simulation done with 0 errors
```

结论：当前可用的 HLS C simulation 是 synthetic smoke，并非真实 `scene_1_t6` golden 对齐。它能证明调用链和 shape 可跑通，但不能替代真实权重/真实输入的数值验收。

## 6. 本阶段没有改源码的原因

当前 `git status --short` 显示工作区已有未提交改动，包括：

```text
.gitignore
hls_src/HLS/stage1_ifan_c8_r2/Makefile
hls_src/HLS/stage1_ifan_c8_r2/build.bat
hls_src/HLS/stage1_ifan_c8_r2/ifan_stage1_engines.cpp
hls_src/HLS/stage1_ifan_c8_r2/test_ifan_stage1.cpp
```

并且还有多个未跟踪的 MABA/post-MABA 源码、日志和 HLS 工程目录。为避免覆盖已有验证路径，本阶段只新增 optimize 文档，不回滚、不重排、不格式化现有源码。

## 7. 下一阶段建议执行顺序

下一阶段建议按下面顺序推进源码改造：

1. 新增 `Stage1Op` 与静态 schedule wrapper。
2. 先把 fusion/final block 的 block id、权重选择、输入输出 buffer 显式化。
3. 为 `ico_conv_r2_main_engine` 和 `ico_conv_r1_main_engine` 增加 tile 级 weight staging 实验版本。
4. 对比 `csynth_design_size.rpt` 中 `to_weight_t` 指令数和调用数是否下降。
5. 将 R1 fusion/final 中间结果改为 ping-pong buffer。
6. 合并 PHAT/LMS attention fuse 的 full-tensor buffer。
7. 再运行 `run_hls.bat synth`，目标是生成最终 `csynth.rpt`。

## 8. 下一阶段验收口径

下一阶段验收不应只看最终资源占比，而应先看是否从 design-size 阶段进入完整综合报告。

最低验收：

- native smoke PASS。
- HLS `csim` PASS。
- 生成 top-level `csynth.rpt`。
- 记录 Estimated Clock、Total Latency、BRAM_18K、DSP、FF、LUT、关键 loop II。

若仍未生成最终 `csynth.rpt`，则记录 design-size 改善：

- Compile/Link instructions 是否下降。
- Unroll/Inline instructions 是否下降。
- Array/Struct instructions 是否下降。
- `ico_conv_r2_main_engine` / `ico_conv_r1_main_engine` 中 `to_weight_t` 指令数和调用数是否下降。

## 9. 当前阶段结论

当前 Stage-1 的 HLS 优化方向已经从“解释为什么卡住”推进到“明确下一轮如何改”。下一轮实际代码工作应集中在调度表达和权重路径收缩：

```text
whole-network static top
    -> explicit Stage1Op schedule
    -> reusable R2/R1 engines
    -> weight/index staging
    -> ping-pong and reduced full-tensor buffers
    -> stable csynth.rpt
```

FeatureMABA 和 post-MABA 虽然已有 native 对齐结果，但仍应保持为独立综合阶段，等 Stage-1 baseline 能稳定产出 `csynth.rpt` 后再讨论顶层串接或 FPGA/CPU 分工。
