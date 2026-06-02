# Stage-1 Synth Smoke 收敛与 ZYNQ 分工转向阶段报告 09

日期：2026-05-31  
对象：`hls_src/HLS/stage1_ifan_c8_r2`

## 1. 本轮收敛结论

本轮对 Stage-1 做了一次 `T=2` 的 HLS synth smoke，目的是快速确认当前代码是否还能进入 `csynth_design`，并评估是否值得继续等待完整资源报告。

最终结论是：

1. `T=2` synth smoke 已经确认可以进入 HLS 后端，不再复现 `_hls_work` 下的 source file 映射失败。
2. 本轮已生成 `csynth_design_size.rpt/xml`，但没有生成最终 top-level `ifan_stage1_top_csynth.rpt/xml`。
3. 长时间运行后仍未得到完整 top 资源报告，说明当前“完整 Stage-1 全部放入 PL 综合”的路线评估成本过高。
4. 后续论文实现路线应从纯 FPGA 全量 top 转向 ZYNQ PS/PL 协同，不再默认把资源占比大的双特征前端与融合部分全部放入 PL。

## 2. 进程与产物状态

本轮 synth smoke 使用外置工程目录：

```text
C:\hls_tmp\stage1_ifan_c8_r2_synth_smoke\stage1_ifan_c8_r2_hls_prj
```

使用的关键配置：

```text
ICO_HLS_MODE=synth
ICO_HLS_CPPFLAGS=-DIFAN_STAGE1_T=2
Top=ifan_stage1_top
Part=xc7k325tffg900-2
Clock=5.0 ns
```

进程状态：

```text
PID=23484
状态：已手动停止
停止前运行时间：接近 2 小时
停止前内存：约 5.6 GB
```

停止前已经生成：

```text
csynth_design_size.rpt
csynth_design_size.xml
若干子模块级 *_csynth.rpt/xml
```

但没有生成：

```text
ifan_stage1_top_csynth.rpt
ifan_stage1_top_csynth.xml
```

因此，本轮不能作为 LUT/FF/DSP/BRAM 的正式资源评估结果，只能作为 design-size 与综合可达性的阶段证据。

## 3. Design-size 关键数据

本轮 `csynth_design_size.rpt` 的 top-level 指令规模如下：

| Phase | Instructions |
|---|---:|
| Compile/Link | 558,357 |
| Unroll/Inline final | 125,750 |
| Array/Struct final | 333,843 |
| Performance peak | 6,254,877 |
| HW Transforms final | 1,482,946 |

主要压力模块如下：

| Module | HW Transforms instructions |
|---|---:|
| `frontend_branch_engine` | 558,396 |
| `fusion_block_engine` | 488,247 |
| `shared_attention_engine` | 368,090 |
| `temporal_conv1d_r1_engine` | 66,345 |

局部日志还显示：

1. `kernel_idx_main` 存在 memory port 限制导致的 II warning。
2. 部分 pipeline 的 estimated clock period 超过 effective delay budget。
3. 停止前 HLS 已经推进到 scheduling / binding 之后，并开始生成若干 RTL 子模块。

这说明当前问题不是“完全不能综合”，而是完整 top 的静态展开、双分支前端、融合链路和 buffer/partition 压力共同导致综合后端代价过大。

## 4. 架构路线调整

此前优化路线默认以完整 `ifan_stage1_top` 放入 PL 为目标，并围绕 IcoConv / temporal / fusion / ping-pong buffer 逐步压缩 design-size。经过本轮 synth smoke 后，该路线需要调整。

新的建议路线：

1. 不再把完整 Stage-1 双特征前端、融合、后处理全部作为纯 FPGA top 的默认目标。
2. 论文实现目标转向 ZYNQ PS/PL 协同。
3. 资源大、控制复杂、收益不明显的前端双特征与融合相关逻辑优先考虑放在 PS 侧。
4. PL 侧聚焦更适合作为硬件创新点的模块：
   - 可流式或可复用的主干网络核心算子；
   - 后端新增 MABA 相关模块；
   - 能形成清晰资源/延迟收益对比的子模块 top。
5. 后续资源评估不再以当前全量 `ifan_stage1_top` 为唯一目标，而应拆分为 PS/PL 边界后的 PL-only top 进行 `csim` 与 `synth`。

## 5. 当前已保留的有效成果

本轮并非失败实验，已经留下以下有效结论：

1. `csim_smoke` 环境链路已经打通，可用于快速确认 HLS 启动、编译与运行链路。
2. 外置工程目录可以绕开 `_hls_work` 下出现过的 source file 映射问题。
3. `T=2` synth smoke 能进入 HLS 后端，并生成 design-size 报告。
4. 已经确认当前全量 top 的综合代价过高，不适合作为下一阶段论文实现的唯一硬件边界。
5. 当前 IcoConv / temporal staging、ping-pong 调度等优化记录仍可作为后续 PL 子模块设计的基础。

## 6. 下一阶段建议

下一阶段不建议继续等待或反复尝试当前全量 top 的完整 `csynth.rpt`。建议转为以下工作：

1. 先定义 ZYNQ PS/PL 分工边界。
2. 确定哪些 Stage-1 前端与融合逻辑留在 PS。
3. 为 PL 侧重新定义更小的 top：
   - 主干网络中可复用 IcoConv/temporal 算子；
   - MABA 或 MABA 相关后端模块；
   - 必要的数据搬运接口。
4. 对新的 PL-only top 分别跑：
   - native 数值对齐；
   - HLS `csim`；
   - HLS `synth`；
   - 最终资源报告 `csynth.rpt`。
5. 后续报告应以 PS/PL 分工后的模块资源作为论文硬件评估主体，而不是以完整 Stage-1 纯 PL 资源作为主线。

## 7. 本阶段结论

本阶段正式收敛当前全量 Stage-1 synth smoke。当前证据已经足够支持路线调整：

```text
完整 Stage-1 纯 PL 综合不是高效论文实现路径。
后续应转向 ZYNQ PS/PL 协同，把 PL 侧重点放在主干核心算子与 MABA 创新模块。
```

