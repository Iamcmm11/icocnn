# Temporal R1 第一轮结构优化报告 18

日期：2026-06-02

## 1. 与报告 17 的衔接

报告 17 的主要结论是：

1. `temporal_r1` 已具备独立综合能力；
2. 旧版结构资源极不理想：
   - LUT 很高；
   - DSP 很低；
   - `II` 很差；
   - 时序略超预算。
3. 主要问题并不在算法本身，而在当前实现方式：
   - `input_r` 外部端口访问压力大；
   - 内层 `(ci, k)` 完全展开过度；
   - `src_t >= 0` 条件混入 MAC 热路径；
   - 乘加结构没有组织成更 DSP-friendly 的局部数据流。

因此，报告 17 给出的第一优先级方案是：

```text
时间维本地 staging
+ 因果窗口提前展开
+ 降低 ci 并行因子
```

本轮就是针对这三点做的第一轮最小结构优化实验。

## 2. 本轮改动

修改文件：

- `hls_src/HLS/stage1_ifan_c8_r2/temporal_r1/ifan_temporal_r1.cpp`

### 2.1 新增本地时间缓冲

为当前 `(ri, ch, h, w)` 位置先搬运整条时间序列：

```text
staged_input[T=6][C=8]
```

并对通道维做完全分区：

```cpp
#pragma HLS ARRAY_PARTITION variable=staged_input complete dim=2
```

含义是：

- 外部 `input_r` 只在 staging 阶段读取；
- 后续 MAC 热路径不再反复直接读顶层大数组。

### 2.2 新增 causal window 预展开

在 MAC 前先构造：

```text
causal_window[T=6][K=5][C=8]
```

即把：

```text
src_t = t - (K - 1) + k
if (src_t >= 0)
```

这套因果访问逻辑提前做完。

这意味着：

- MAC 热路径里不再出现 `src_t >= 0` 条件判断；
- 内层计算变成固定窗口读取；
- HLS 更容易做规则调度与 DSP 映射。

### 2.3 将 `ci` 从完全展开改为部分展开

新增参数：

```cpp
#define IFAN_TEMPORAL_CI_PAR_FACTOR 2
```

当前 MAC 结构由旧版：

```text
ci=8 完全展开
k=5 完全展开
=> 40 项宽组合树
```

改为新版：

```text
ci 按 factor=2 分块
k=5 保持展开
每个 ci 小组内部做局部并行
```

这样做的目的不是追求最小时延，而是先把：

- LUT 压力
- 扇出
- 端口冲突
- 加法树规模

控制到一个更合理的范围。

## 3. 数值验证结果

### 3.1 native

重新构建并运行：

```bat
build.bat temporal
test_ifan_temporal_r1.exe ..\..\..\hls_testdata\temporal_r1_c8_t6\fusion0
```

结果：

```text
Max Error: 0
RMSE: 0
PASS
```

说明结构优化没有破坏真实 `temporal_r1` 功能对齐。

### 3.2 HLS csim

代表性命令：

```bat
set IFAN_TEMPORAL_R1_DATA_DIR=..\..\..\hls_testdata\temporal_r1_c8_t6\fusion0
set IFAN_TEMPORAL_R1_MAX_ERR_TOL=0.002
set IFAN_TEMPORAL_R1_RMSE_TOL=0.001
run_hls.bat csim temporal
```

结果：

```text
Max Error: 0.00139052
RMSE: 0.000369002
PASS
CSim done with 0 errors.
```

这与优化前 `csim` 的误差级别基本一致，说明本轮优化主要改变的是结构与资源行为，没有引入新的定点数值退化。

## 4. 优化后综合结果

### 4.1 报告路径

新版综合快照：

- `hls_src/hls_reports/stage1_ifan_c8_r2_temporal_r1_hls_prj_sol1_20260602_161542/summary.md`
- `hls_src/hls_reports/stage1_ifan_c8_r2_temporal_r1_hls_prj_sol1_20260602_161542/ifan_temporal_r1_top_csynth.rpt`
- `hls_src/hls_reports/stage1_ifan_c8_r2_temporal_r1_hls_prj_sol1_20260602_161542/csynth_design_size.rpt`

### 4.2 关键指标

从最新 `summary.md` 读取：

| 指标 | 新版优化后 |
|---|---:|
| Target clock | `5.00 ns` |
| Estimated clock | `3.867 ns` |
| DSP | `160` |
| LUT | `38,494` |
| FF | `39,465` |
| BRAM_18K | `0` |

并且：

```text
All loop constraints were satisfied
Estimated Fmax: 258.60 MHz
```

## 5. 与优化前的对比

优化前（报告 17 对应快照）：

- Estimated clock: `5.083 ns`
- DSP: `9`
- LUT: `152,949`
- FF: `71,440`
- `II` 极差，loop constraint 不满足

优化后（本轮）：

- Estimated clock: `3.867 ns`
- DSP: `160`
- LUT: `38,494`
- FF: `39,465`
- loop constraint 满足

### 5.1 变化总结

| 指标 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| Estimated clock (ns) | `5.083` | `3.867` | 明显改善 |
| DSP | `9` | `160` | 大幅上升 |
| LUT | `152,949` | `38,494` | 大幅下降 |
| FF | `71,440` | `39,465` | 明显下降 |
| Loop constraints | 不满足 | 满足 | 闭合 |

### 5.2 结论

这轮结构优化是有效的，而且效果非常显著：

1. 说明报告 17 的瓶颈判断是正确的；
2. 说明 temporal 内核的资源问题主要是结构问题，不是算法本身问题；
3. 说明“先做 staging、再削减不规则访存、再控制展开粒度”是当前正确路线。

## 6. 为什么这轮优化有效

### 6.1 `input_r` 端口瓶颈被切断

旧版最核心的问题是外部 `input_r` 在 MAC 热路径里被反复随机访问。

本轮把它改成：

```text
外部 input_r
-> staged_input
-> causal_window
-> MAC
```

这样：

- 外部大数组只承担一次规则读取；
- 热点计算完全基于本地数组；
- HLS 不再需要为外部端口冲突生成大量仲裁和控制逻辑。

### 6.2 `src_t >= 0` 条件不再进入热点乘加路径

旧版在 MAC 内部每次都要判断：

```text
if (src_t >= 0)
```

这会诱导 HLS 产生大量 if-conversion 逻辑。

本轮提前构造 `causal_window` 后：

- MAC 内部只读固定窗口；
- 条件逻辑被移到前置 staging 阶段；
- 热路径更规则。

### 6.3 `ci` 部分展开让乘加结构更接近 DSP-friendly 组织

旧版是 40 项宽组合树；
新版改成较小粒度的局部并行。

结果从综合日志也能看到：

```text
mul_26s_25s_51_5_1 : 80 instance(s)
DSP: 160
```

这说明乘法现在已经明显更稳定地落到 DSP 路径上，而不是继续堆在 LUT 里。

## 7. 当前仍需注意的问题

虽然这轮结果很好，但还不是终点：

1. `causal_window` 被自动推断成多个 RAM 副本：
   ```text
   ... using auto RAMs with 3 copies ...
   ```
   这说明当前是用额外复制换取端口，后续仍可进一步优化其存储组织。

2. 报告中仍有：
   ```text
   variable-indexed range selection may cause suboptimal QoR
   ```
   说明还有进一步规则化访问的空间。

3. 当前 DSP 虽然已经明显上升，但并未明确绑定：
   - 现在更多是结构改好了，工具自然推上去；
   - 还没有显式做 DSP 绑定策略。

## 8. 下一轮建议

下一轮建议分两条：

### A. 继续当前 temporal_r1 降资源线

1. 评估 `IFAN_TEMPORAL_CI_PAR_FACTOR`
   - `2`
   - `4`
   - `8`
2. 对比：
   - DSP
   - LUT
   - FF
   - 时钟
   - `csim` 误差

这能帮助确定当前 temporal 模块最合适的并行因子。

### B. 引入显式 DSP 分担策略

虽然本轮 DSP 已经从 `9` 抬升到 `160`，但下一轮仍值得把乘法稳定落到 DSP 的策略正式引入：

1. 对乘法热点尝试 `BIND_OP op=mul impl=dsp`
2. 尽量保持：
   - `input_t * weight_t` 在窄位宽下乘
   - 扩位延后到 partial sum 之后
3. 必要时把局部归并写成更明确的 staged reduction

这部分建议作为下一轮正式工作计划写入，而不是本轮直接继续堆改动。

## 9. 本轮总结

本轮是 `temporal_r1` 主线上的第一个真正“结构优化见效”的节点。

相较于优化前：

- LUT 大幅下降
- DSP 大幅上升
- FF 下降
- 时钟闭合
- loop constraints 满足

这说明：

```text
参考 layer2-5 的 staging 思路去改 temporal，是有效且必要的。
```

同时也说明，后续完全可以在阶段报告里把“乘法稳定落到 DSP”的方法作为下一轮计划项继续推进。
