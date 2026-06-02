# Stage-1 Directive-First 实验记录 07（TemporalConv）

日期：2026-05-31  
对象：`hls_src/HLS/stage1_ifan_c8_r2`

## 1. 实验目的

在不做架构级代码改造前，先验证 HLS 指令层是否还有明显收益空间。  
本轮只对 `temporal_conv1d_r1_engine` 做 pragma 调整，不改算法路径和接口。

## 2. 实验改动（pragma-only）

实验版在 `temporal_conv1d_r1_engine` 中尝试：

- `#pragma HLS INLINE off`
- `#pragma HLS ARRAY_PARTITION variable=weight complete dim=3`
- `#pragma HLS ARRAY_PARTITION variable=bias complete dim=1`
- 在 `ci/k` 循环增加 `#pragma HLS UNROLL off`

备注：该组合用于抑制自动展开和明确内存分割，验证是否可先通过指令层压缩 design-size。

## 3. 运行方式与环境

为规避 `G:` 路径下偶发 Windows 文件映射报错，本轮 synth 在本地盘工程目录运行：

```text
run_hls.bat synth ... PROJECT_ROOT=C:\hls_tmp\stage1_ifan_c8_r2_hls_runs
```

报告回收到：

```text
hls_src/hls_reports/stage1_ifan_c8_r2_hls_prj_sol1_20260531_164503/
```

## 4. 结果对比（与上一轮基线）

基线（上一轮 latest）：

- Compile/Link: `413,509`
- Array/Struct: `788,676`
- `temporal_conv1d_r1_engine` Array/Struct: `26,988`
- `temporal_conv1d_r1_engine -> to_weight_t`: `11,360 (80 calls)`

实验轮（pragma-only）：

- Compile/Link: `413,513`（+4）
- Array/Struct: `789,021`（+345）
- `temporal_conv1d_r1_engine` Array/Struct: `27,322`（+334）
- `temporal_conv1d_r1_engine -> to_weight_t`: `11,360 (80 calls)`（无变化）

结论：  
该组 pragma 没有带来压缩收益，反而使关键规模指标轻微上升，且 temporal `to_weight_t` 热点未改善。

## 5. 处理决定

已将上述实验 pragma 从主线代码回滚（保留实验记录，不保留负收益改动）。

## 6. 下一步建议（符合 directive-first 原则）

在进入新的架构级修改前，继续完成“指令优先”闭环：

1. 保持当前有效 pragma 组合不变（回到实验前状态）。  
2. 进入下一项“最小结构改动 + 明确指令约束”的 temporal weight staging：
   - 先做 `(co)` 或 `(co, ci)` 级 temporal 权重预取；
   - 明确 `INLINE` / `UNROLL` 策略，避免 helper 被重展开。  
3. 再跑 synth，对比 `temporal_conv1d_r1_engine` 与 `to_weight_t` 指标是否下降。

