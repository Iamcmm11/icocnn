# Stage-1 环境修复与对照 Layer2-5 的阶段记录 08

日期：2026-05-31  
对象：`hls_src/HLS/stage1_ifan_c8_r2`

## 1. 环境与验证链路

### 1.1 为什么不能直接照搬 `layer2-5` 的 `run_hls.bat`

`layer2-5` 的脚本稳定，不代表 Stage-1 直接复用就能稳定，原因已经在当前仓库内被反复验证：

1. Stage-1 的 HLS testbench 体量更大，真实数据装载、权重、几何索引都更多。
2. Stage-1 曾出现过与 `layer2-5` 不同的独立问题：
   - `tee.exe` 缺失
   - `csim.exe` 返回 `-1073741515`
   - `synth` 日志出现 Windows 文件映射/工程目录异常
3. Stage-1 的 `csim` 工程还额外踩到了一个 HLS 侧问题：
   - `csim.mk` 只编译 testbench，未自动把 design sources 一起带入链接

结论：Stage-1 需要保留自己的 `run_hls.bat` 主入口，并围绕自身 testbench 和工程目录单独修复。

### 1.2 本轮已完成的环境修复

1. `run_hls.bat` 现在保留为唯一主入口，并新增：
   - `csim_smoke`
   - `csim`
   - `synth`
2. `csim_smoke` 固化为：
   - `IFAN_STAGE1_T=2`
   - 只用于验证 HLS 启动、编译、`csim.exe` 运行时链路
3. `run_hls.bat` 增加了启动前自检日志：
   - `where vitis_hls`
   - `where g++`
   - `where python`
   - `where tee.exe`
   - 当前 PATH
4. `run_hls.tcl` 默认工程目录切到仓库内：
   - `hls_src/HLS/stage1_ifan_c8_r2/_hls_work`
5. `run_hls_eval.ps1` 补充了：
   - smoke 编译参数
   - `csim.exe` 复跑与退出码记录
6. `test_ifan_stage1.cpp` 在 `__HLS_CSIM__` 下直接包含 design `.cpp`，用于绕过当前 HLS `csim.mk` 未自动链接 design sources 的问题。

### 1.3 当前验证结果

#### Native

1. 全量 `T=6` native 验证继续 `PASS`。
2. 当前结构改动后真实数据对齐结果：
   - `MaxAbsError = 1.95801e-05`
   - `RMSE = 1.46019e-06`
3. 说明本轮 IcoConv / temporal / ping-pong 改动没有破坏原有数值链路。

#### HLS `csim_smoke`

本轮已经拿到新的、可复现的 `csim_smoke PASS` 证据：

1. `run_hls.bat csim_smoke` 已能完整跑通 Vitis HLS `csim_design`。
2. `csim.exe` 已成功生成并执行。
3. `ifan_stage1_top_csim.log` 已出现：
   - `CSIM start`
   - `Generating csim.exe`
   - `PASS`
   - `CSim done with 0 errors`

这说明当前“无需手工补 PATH 的 Stage-1 smoke 环境链路”已经打通。

### 1.4 当前仍未完全闭环的问题

1. HLS `_hls_work` 目录下的 smoke 仍未命中真实 `scene_1_t6` 数据路径，当前退回 synthetic smoke。
2. `IFAN_STAGE1_T=2` 时，直接用真实 golden 前缀做严格误差阈值判断并不成立。
   - 这不是环境失败
   - 更像是 temporal 因果窗口和真实模型导出边界之间的前缀对齐问题
3. 因此本轮 `csim_smoke` 的验收口径是：
   - HLS 启动成功
   - 编译成功
   - `csim.exe` 成功执行
   - 不要求等同于完整 `scene_1_t6` 的数值闭环

## 2. 相对 Layer2-5 的优化缺口

### 2.1 本轮已补齐

| 项目 | 状态 | 说明 |
|---|---|---|
| 定点类型接口 | 已有 | `input_t/weight_t/act_t/acc_t` 已与 `layer2-5` 同口径 |
| main IcoConv weight staging | 已完成 | 之前已完成，当前继续保留 |
| temporal weight staging | 已完成 | `temporal_conv1d_r1_engine` 已加入 staging |
| `OC_PAR_FACTOR=2` | 已完成 | `ifan_stage1.hpp` 已固定引入 |
| `ico_conv_r2_stem` 输入 staging + kernel tile + output tile | 已完成 | 从单点直接累加改成 tile 骨架 |
| `ico_conv_r2_main` output tile + `ri_partial` | 已完成 | 已贴近 `layer2-5` 的 proven 结构 |
| `ico_conv_r1_main` output tile + `ri_partial` | 已完成 | 已贴近 `layer2-5` 的 proven 结构 |
| `weight` / `padded` / kernel 的结构化 pragma | 已完成 | 已补齐主路径必要 partition/unroll |
| 顶层 ping-pong 调度 | 已完成 | 去掉 block 间全量复制，改为 `fused_r1_a/b` 交替 |

### 2.2 当前仍未做

| 项目 | 状态 | 说明 |
|---|---|---|
| 顶层 `DATAFLOW` | 未做 | 本轮刻意不引入跨模块重叠 |
| unified top 接入 `MABA + post` | 未做 | 当前综合边界仍是 pre-readout `ifan_stage1_top` |
| HLS 下真实 `scene_1_t6` smoke 数据命中 | 未完成 | 仍需补 HLS 工作目录下的真实数据定位 |
| `T=2` 前缀 smoke 的严格数值定义 | 未完成 | 需要后续给出独立 smoke golden 或前缀对齐策略 |
| design-size 量化对比 | 未完成 | 需在新的 `synth` 结果上确认 `to_weight_t` 与 phase 指标变化 |

### 2.3 已做 / 未做 / 下一步

| 类别 | 项目 | 当前状态 | 下一步 |
|---|---|---|---|
| 环境 | `run_hls.bat` 多模式入口 | 已完成 | 继续保留为唯一主入口 |
| 环境 | `csim_smoke` 无手工补 PATH PASS | 已完成 | 将真实数据路径补进 HLS 工作目录 |
| 环境 | `csim.exe` DLL/链接问题 | 已绕过 | 后续可再评估是否去掉 `__HLS_CSIM__` 直包含方案 |
| 结构 | temporal staging | 已完成 | 用 `synth` 报告验证 `to_weight_t` 是否下降 |
| 结构 | IcoConv tile / `ri_partial` | 已完成 | 看资源与 design-size 变化是否可接受 |
| 结构 | ping-pong 顶层调度 | 已完成 | 后续再评估是否值得引入更深一层 schedule wrapper |
| 结构 | unified top with MABA/post | 未做 | 下一阶段再推进 |
| 验证 | `T=2` 真实 prefix 严格 golden | 未完成 | 后续单独定义 smoke golden 或 debug tensor 对齐口径 |

## 3. 本轮结论

1. 本轮最重要目标已经达成：Stage-1 的 `csim_smoke` 环境链路已打通。
2. 当前 Stage-1 已不再停留在“功能串接 + 少量 pragma”，而是已经把 `layer2-5` 的关键结构思路迁入主 IcoConv/temporal 路径。
3. 接下来真正的主线不再是“能不能启动 HLS”，而是两件事：
   - 用新的 `synth` 结果确认 design-size/资源变化
   - 把 HLS smoke 从 synthetic 数据推进到真实 `scene_1_t6` 数据命中
