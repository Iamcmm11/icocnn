# Stage-1 阶段5完成度对照与下一阶段计划 06

日期：2026-05-31  
对象：`hls_src/HLS/stage1_ifan_c8_r2`  
对照基线：`optimize/05_stage1_weight_staging_implementation.md`

## 1. 本次对照范围

本记录只对照阶段5文档中的“目标、验收、下一步”三部分，结合当前最新证据：

- 代码与脚本：  
  `run_hls.bat` / `run_hls.tcl` / `parse_hls_report.py`
- 最新摘要：  
  `hls_src/hls_reports/stage1_ifan_c8_r2_latest_summary.md`
- 最新 design-size 原始报告：  
  `hls_src/hls_reports/stage1_ifan_c8_r2_hls_prj_sol1_20260531_162633/csynth_design_size.rpt`

## 2. 阶段5需求完成度对照

| 阶段5需求 | 当前状态 | 证据 | 结论 |
|---|---|---|---|
| R2/R1 main 引入 weight/index staging（低扰动改动） | 已完成 | `ifan_stage1_engines.cpp` 中 `stage_ico_main_weights` 与 R2/R1 main 引擎调用 | 达成 |
| 不改 top/testbench 接口 | 已完成 | `ifan_stage1_top` 接口未变；testbench 接口未改 | 达成 |
| native 数值链路保持 PASS | 已完成 | 阶段5文档第4节已有 `scene_1_t6` 对齐 PASS 记录 | 达成（沿用既有证据） |
| HLS `csim` 跑通 | 未完成（且当前阶段非刚需） | 目前聚焦 synth；未形成“本轮脚本链路下的 csim PASS”新证据 | 未达成 |
| 跑 `synth` 并输出可比较的 design-size | 已完成 | 最新 summary 已落盘，含 Compile/Unroll/Array/Performance 状态 | 达成 |
| 对比 `to_weight_t` 收缩效果 | 部分完成 | 新 report 中 R2/R1 main 的 `to_weight_t` 显著下降，temporal 未变 | 部分达成 |
| 形成下一步压资源计划 | 已完成 | 本文第4节 | 达成 |

## 3. 本轮关键量化结果（相对阶段4基线）

阶段4基线（`04` 文档）：

- Compile/Link：`409,709`
- Unroll/Inline（早期阶段）：`1,790,028`
- Array/Struct：`1,393,356`
- `to_weight_t` 证据：
  - `ico_conv_r2_main_engine`: `245,376 instructions, 1728 calls`
  - `ico_conv_r1_main_engine`: `122,688 instructions, 864 calls`
  - `temporal_conv1d_r1_engine`: `11,360 instructions, 80 calls`

当前最新（`stage1_ifan_c8_r2_latest_summary.md` + 对应 `csynth_design_size.rpt`）：

- Compile/Link：`413,509`（轻微上升）
- Unroll/Inline：`683,058 -> 354,700 -> 342,691 -> 323,245`（显著下降）
- Array/Struct：`788,676`（显著下降）
- `to_weight_t`（关键条目）：
  - `ico_conv_r2_main_engine` 路径条目：`568 (4 calls)`（相对阶段4大幅收缩）
  - `ico_conv_r1_main_engine` 路径条目：`284 (2 calls)`（相对阶段4大幅收缩）
  - `temporal_conv1d_r1_engine`：`11,360 (80 calls)`（基本不变）

结论：  
阶段5“把 main IcoConv 的权重转换挪出最内层 MAC”目标已经产生了可观的 design-size 收缩效果；当前最大未收缩热点已转向 frontend/shared attention/fusion 结构规模与 temporal 分支。

## 4. 下一阶段计划（建议执行顺序）

### 阶段6A（先稳住综合链路）

1. 固化“外置 HLS 工程目录 + 源码绝对路径 + 统一回收到 `hls_reports`”流程作为默认 `synth` 路径。  
2. 每次 `synth` 必留两份证据：
   - latest 摘要：`hls_src/hls_reports/stage1_ifan_c8_r2_latest_summary.md`
   - 对应快照目录中的 `csynth_design_size.rpt/xml`

验收口径：

- 能重复运行 `run_hls.bat synth` 并稳定更新 latest 摘要。
- 不要求本阶段 `csim` PASS。

### 阶段6B（压资源主线：继续收缩 design-size）

1. 先做 `temporal_conv1d_r1_engine` 的 temporal weight staging（对应阶段5第7节第4条）。  
2. 复测 design-size，重点观察：
   - `temporal_conv1d_r1_engine` 中 `to_weight_t` 是否从 `11,360 (80 calls)` 下降。
   - `fusion_block_engine` 的 Unroll/Array 指令是否连带下降。  
3. 若下降不明显，检查 helper 被内联重展开的情况，尝试：
   - 在 staging helper 上做 `INLINE off` 实验；
   - 调整 staging buffer 的维度与循环边界，减少 HLS 重复实例化。

验收口径：

- `to_weight_t`（temporal）出现可观下降；
- `Array/Struct` 相比 `788,676` 继续下降；
- latest 摘要可追溯每轮变化。

### 阶段6C（调度层与 buffer 收缩）

在 6B 稳定后推进：

1. 显式 `Stage1Op` / schedule wrapper（先 fusion/final block 显式化）。  
2. R1 fusion/final 中间结果改为 ping-pong。  
3. 合并 PHAT/LMS attention full-tensor buffer。  

验收口径：

- `frontend_branch_engine`、`shared_attention_engine`、`fusion_block_engine` 三个热点函数的 Compile/Unroll/Array 指令同步下降；
- 目标从“仅 design-size 改善”过渡到“可稳定逼近最终 `csynth.rpt`”。

## 5. 当前建议

下一步建议直接执行“阶段6B-1”：  
先改 `temporal_conv1d_r1_engine` 的 temporal weight staging，并立即跑一轮 synth 对比 `to_weight_t` 与 `Array/Struct`。

