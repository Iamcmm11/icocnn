# Layer2-5 C8/T6 参数化 Smoke 与首次 Synth 报告 12

日期：2026-06-02

## 1. 本轮目标

根据计划 11，本轮先推进主干网络 `hls_src/HLS/layer2-5`，目标是把原先固定的 C32/T52 R1 ConvIco kernel 参数化，并验证 C8/T6 窗口级 kernel 能走通：

- native baseline 不回归；
- C8/T6 native smoke 通过；
- C8/T6 HLS `csim` 通过；
- C8/T6 HLS `synth` 生成 top-level `csynth` 报告。

本轮仍使用 deterministic synthetic C8/T6 数据，只证明参数化、构建入口和 HLS 后端链路可达；真实 IFAN C8_R2 权重/golden 对齐放在下一轮。

## 2. 代码改动

已完成：

1. `ico_conv_layer2_5.hpp` 参数化以下宏，并保留默认行为：

```cpp
ICO_LAYER2_5_TIME_STEPS = 52
ICO_LAYER2_5_CIN        = 32
ICO_LAYER2_5_COUT       = 32
ICO_LAYER2_5_OC_PAR_FACTOR = 2
```

默认仍映射到原有：

```cpp
TIME_STEPS = 52
CIN        = 32
COUT       = 32
```

2. `test_ico_conv_layer2_5.cpp` 新增 C8/T6 synthetic smoke 路径：

- 默认 C32/T52 继续读取 `hls_testdata/layer2-5/layer*` 文件并做真实 reference 对齐；
- 非默认尺寸默认使用确定性 synthetic input/weight/kernel/reorder；
- 后续可通过 `ICO_LAYER2_5_FORCE_FILE_DATA` 改回文件数据模式。
- 后续真实 C8/T6 数据可通过第二个命令行参数或环境变量 `ICO_LAYER2_5_DATA_DIR` 指定目录。

3. `build_layer2_5.bat` 支持：

```bat
build_layer2_5.bat baseline
build_layer2_5.bat c8t6
```

其中 C8/T6 生成：

```text
test_ico_conv_layer2_5_c8t6.exe
```

4. `Makefile` 新增：

```bash
make c8t6
```

5. `run_hls.bat` 支持：

```bat
run_hls.bat csim c8t6
run_hls.bat synth c8t6
```

6. `run_hls.tcl` 支持 `ICO_HLS_CPPFLAGS`，并将 C8/T6 宏传入设计文件和 testbench。

## 3. Native 验证结果

### Baseline C32/T52

命令：

```bat
cd hls_src\HLS\layer2-5
build_layer2_5.bat baseline
test_ico_conv_layer2_5.exe 2
```

结果：

```text
Configured shape: T=52 CIN=32 COUT=32 RIN=6 ROUT=6
Layer2-5 Output: size=399360
Max Error: 7.62939e-006
RMSE: 6.37311e-007
PASS
```

结论：默认 C32/T52 行为未回归。

### C8/T6 synthetic

命令：

```bat
cd hls_src\HLS\layer2-5
build_layer2_5.bat c8t6
test_ico_conv_layer2_5_c8t6.exe
```

结果：

```text
Configured shape: T=6 CIN=8 COUT=8 RIN=6 ROUT=6
Data mode: deterministic synthetic smoke
Layer2-5 Output: size=11520
min=-0.0067004, max=0.0422683, mean=0.00347996
```

结论：C8/T6 参数化 native smoke 通过，输出 finite。

## 4. HLS 验证结果

### C8/T6 csim

命令：

```bat
cd hls_src\HLS\layer2-5
run_hls.bat csim c8t6
```

关键日志：

```text
CppFlags : -DICO_LAYER2_5_TIME_STEPS=6 -DICO_LAYER2_5_CIN=8 -DICO_LAYER2_5_COUT=8
Configured shape: T=6 CIN=8 COUT=8 RIN=6 ROUT=6
Data mode: deterministic synthetic smoke
CSim done with 0 errors.
```

结论：Vitis HLS `csim` 已确认吃到 C8/T6 宏。

### C8/T6 synth

命令：

```bat
cd hls_src\HLS\layer2-5
run_hls.bat synth c8t6
```

报告路径：

```text
hls_src/HLS/layer2-5/layer2_5_c8t6_hls_prj/sol1/syn/report/conv_ico_layer2_5_csynth.rpt
hls_src/HLS/layer2-5/layer2_5_c8t6_hls_prj/sol1/syn/report/conv_ico_layer2_5_csynth.xml
hls_src/hls_reports/layer2_5_c8t6_hls_prj_sol1_20260602_135322/
```

摘要：

| 指标 | C8/T6 synthetic |
|---|---:|
| Target clock | 5.00 ns |
| Estimated clock | 4.209 ns |
| Latency | 1,956,787 cycles |
| II | 1,956,788 |
| BRAM_18K | 26 / 890 |
| DSP | 72 / 840 |
| FF | 43,579 / 407,600 |
| LUT | 69,613 / 203,800 |

与此前 C32/T52 baseline 报告对比：

| 指标 | C32/T52 baseline | C8/T6 synthetic | 变化 |
|---|---:|---:|---:|
| Latency | 261,686,621 | 1,956,787 | 约 133.7x 下降 |
| BRAM_18K | 64 | 26 | 下降 |
| DSP | 72 | 72 | 基本不变 |
| FF | 43,639 | 43,579 | 基本不变 |
| LUT | 69,382 | 69,613 | 基本不变 |

## 5. 阶段判断

本轮已经证明：`layer2-5` 可以作为 C8/T6 主干复用 kernel 继续推进，且不需要在 `stage1_ifan_c8_r2` 下再 fork 一套 R1 ConvIco。

需要注意的是，本轮 C8/T6 是 synthetic smoke，资源只能说明参数化后的 kernel 结构可综合，不能直接当作真实 IFAN C8_R2 数值实现结果。

资源结果中有一个重要现象：

```text
T/C 缩小后，latency 和 BRAM 明显下降，但 DSP/LUT/FF 没有按比例下降。
```

这说明当前面积主要受以下结构影响：

- `OC_PAR_FACTOR=2` 的并行骨架；
- output tile / post-process tile / ri_partial 的数组分区；
- `pad_ico_quantized` 与极点处理；
- expanded kernel 动态索引路径；
- HLS 对接口和局部数组的保守资源估计。

因此，下一步真实 C8 对齐完成后，如果需要压资源，应优先做结构级 sweep：

- `ICO_LAYER2_5_OC_PAR_FACTOR=1` vs `2`；
- 降低 output/post-process tile 分区；
- 单独评估 `pad_ico_quantized` 和 post-process 对 LUT 的贡献；
- 把真实 C8 权重/索引接入后再决定是否进一步定点位宽 sweep。

## 6. 下一步

下一轮建议进入真实数据对齐：

1. 在 Torch debug/export 中补齐每个 C8_R2 fusion/final ConvIco 的单层节点：
   - ConvIco input；
   - ConvIco output，位于 ReLU/temporal Conv1d/LNorm 之前。
2. 导出到：

```text
hls_testdata/layer2-5_c8_t6/
  fusion0/
  fusion1/
  fusion2/
  fusion3/
  final/
```

3. 让 C++ testbench 支持 C8/T6 file-data 模式，读取真实：
   - `input_rearranged.txt`
   - `weight.txt`
   - `bias.txt`
   - `kernel_expansion_idx.txt`
   - `reorder_idx.txt`
   - `output.txt`
4. 运行方式建议：

```bat
set ICO_LAYER2_5_DATA_DIR=..\..\..\hls_testdata\layer2-5_c8_t6\fusion0
build_layer2_5.bat c8t6
test_ico_conv_layer2_5_c8t6.exe 2
```

或者：

```bat
test_ico_conv_layer2_5_c8t6.exe 2 ..\..\..\hls_testdata\layer2-5_c8_t6\fusion0
```

5. 先跑 native max error / RMSE，再跑 HLS `csim`。
6. 真实 C8 ConvIco 对齐后，再开始 temporal Conv1d/LNorm 和 FeatureMABA。
