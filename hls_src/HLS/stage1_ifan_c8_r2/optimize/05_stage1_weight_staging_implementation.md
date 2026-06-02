# Stage-1 IcoConv Weight Staging 实现记录 05

日期：2026-05-31  
对象：`hls_src/HLS/stage1_ifan_c8_r2`  
对应设计文档：`03_stage1_scheduler_reuse_design.md`

## 1. 本轮目标

按 `03_stage1_scheduler_reuse_design.md` 的优先级，先实现一个低扰动的代码推进：

```text
把 ico_conv_r2_main_engine / ico_conv_r1_main_engine 中的
kernel_idx 解码与 to_weight_t(weight[...]) 转换，
从空间 MAC 内层循环移动到 (co, ro) 级 staging。
```

本轮不改 top 接口、不改 testbench 接口、不改权重文件格式，也不接入新的 scheduler/FSM。目标是先验证 weight staging 对数值链路无破坏，并为下一轮 HLS design-size 对比提供明确代码基线。

## 2. 修改范围

修改文件：

```text
hls_src/HLS/stage1_ifan_c8_r2/ifan_stage1_engines.cpp
```

新增 helper：

```cpp
static void stage_ico_main_weights(
    const data_t weight[IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_NEIGHBORS],
    const int kernel_idx[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][9][4],
    int co,
    int ro,
    weight_t staged_weight[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_H * IFAN_KERNEL_W],
    bool staged_valid[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_H * IFAN_KERNEL_W]
);
```

应用位置：

```text
ico_conv_r2_main_engine
ico_conv_r1_main_engine
```

未修改位置：

```text
ico_conv_r2_stem_engine
temporal_conv1d_r1_engine
ifan_stage1_top
test_ifan_stage1.cpp
```

## 3. 实现方式

在每个 `(co, ro)` 输出组合进入空间循环前，预取：

```text
staged_weight[ci][ri][k]
staged_valid[ci][ri][k]
```

预取阶段完成：

```text
kernel_idx[co][ro][ci][ri][k] -> idx_co / idx_ci / idx_ri / idx_w
idx_w valid check
data_t weight -> weight_t
```

空间 MAC 内层从：

```cpp
const int idx_co = kernel_idx[co][ro][ci][ri][k][0];
const int idx_ci = kernel_idx[co][ro][ci][ri][k][1];
const int idx_ri = kernel_idx[co][ro][ci][ri][k][2];
const int idx_w = kernel_idx[co][ro][ci][ri][k][3];
if (idx_w >= 0 && idx_w < IFAN_KERNEL_NEIGHBORS) {
    sum += padded[...] * to_weight_t(weight[idx_co][idx_ci][idx_ri][idx_w]);
}
```

变为：

```cpp
if (staged_valid[ci][ri][k]) {
    sum += padded[...] * staged_weight[ci][ri][k];
}
```

这样做的预期收益是减少 HLS 在每个 `ch/h/w` 空间位置重复实例化 index 解码和 `to_weight_t` 转换。

## 4. Native 验证

命令：

```bat
cd hls_src\HLS\stage1_ifan_c8_r2
build.bat
test_ifan_stage1.exe
```

结果：

```text
Built test_ifan_stage1.exe
Loaded real Stage-1 data: ../../../hls_testdata/stage1_ifan_c8_r2/scene_1_t6
IFAN Stage-1 HLS smoke test
Output shape: [6, 8, 6, 5, 2, 4]
Checksum: -332.325
AbsSum: 9409.68
Min/Max: -2.33815 / 2.57942
Golden: final_head_logits.txt
MaxAbsError: 2.30968e-05
RMSE: 1.80074e-06
MeanAbsGolden: 0.816812
WorstIndex: [0, 5, 5, 3, 1, 1]
WorstOut/Ref: 0.253576 / 0.253599
PASS
```

结论：本轮 staging 改动保持真实 `scene_1_t6` 数据对齐结果不变，数值误差仍处于既有 PASS 量级。

## 5. HLS C Simulation 尝试

第一次运行：

```bat
run_hls.bat csim
```

失败原因：

```text
'tee.exe' is not recognized as an internal or external command
```

定位到本机存在：

```text
G:\PostGraduateFile\Git\usr\bin\tee.exe
```

第二次运行时临时补 PATH：

```powershell
$env:PATH = 'G:\PostGraduateFile\Git\usr\bin;' + $env:PATH
.\run_hls.bat csim
```

结果：

```text
Vitis HLS 2024.2 启动成功
open_project / set_top / add_files 成功
csim_design 启动成功
Compiling ../../../../ifan_stage1_engines.cpp in debug mode
Generating csim.exe
```

但命令在 600 秒超时前没有完成完整 `csim_design`，未得到 HLS C simulation PASS。

生成文件：

```text
stage1_ifan_c8_r2_hls_prj/sol1/csim/build/csim.exe
stage1_ifan_c8_r2_hls_prj/sol1/csim/report/ifan_stage1_top_csim.log
```

当前 `ifan_stage1_top_csim.log` 停在：

```text
INFO: [SIM 2] *************** CSIM start ***************
INFO: [SIM 4] CSIM will launch CLANG as the compiler.
   Compiling ../../../../ifan_stage1_engines.cpp in debug mode
   Generating csim.exe
```

直接运行 HLS 生成的 `csim.exe` 返回：

```text
EXIT=-1073741515
```

该返回码通常对应 Windows DLL 加载失败。因此本轮 HLS 侧证据只能说明：Vitis HLS 前端能够读取修改后的源码并生成新的 `csim.exe`；不能证明 HLS `csim` 已经 PASS。

## 6. 当前结论

本轮完成了 `03` 设计文档中第二优先级的第一步代码落地：

```text
IcoConv R2/R1 main weight/index staging
```

已确认：

- native `g++` 构建通过。
- 真实 `scene_1_t6` Stage-1 输出仍对齐 `final_head_logits.txt`。
- main IcoConv 空间 MAC 内层不再直接执行 `kernel_idx` 四元组读取和 `to_weight_t(weight[idx...])`。

尚未确认：

- HLS `csim_design` 完整 PASS。
- HLS `csynth_design` 是否能生成最终 `csynth.rpt`。
- `csynth_design_size.rpt` 中 `to_weight_t` 指令数是否下降。

## 7. 下一步

建议下一步按以下顺序继续：

1. 修复 HLS C simulation 运行环境，重点是 `csim.exe` 运行时 DLL 路径。
2. 重新运行 `run_hls.bat csim`，确认 HLS C simulation PASS。
3. 运行 `run_hls.bat synth`，对比新的 `csynth_design_size.rpt`：
   - `ico_conv_r2_main_engine` 中 `to_weight_t` 指令数；
   - `ico_conv_r1_main_engine` 中 `to_weight_t` 指令数；
   - total Compile/Link、Unroll/Inline、Array/Struct instructions。
4. 如果 design-size 明显下降，再继续推进 `temporal_conv1d_r1_engine` 的 temporal weight staging。
5. 如果 design-size 未下降或上升，则检查 helper 是否被 HLS 以内联方式重新展开，需要尝试 `INLINE off` 或将 staging buffer/loop 组织调整为更稳定的 engine 边界。

