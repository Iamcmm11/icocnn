# Temporal R1 真实对齐与独立 csim 报告 16

日期：2026-06-02

## 1. 与报告 15 的衔接

报告 15 的结论是：

1. `TemporalConv1d R1` 独立模块骨架已经建立；
2. `export_temporal_r1_testdata.py` 已经具备导出真实数据的基本能力；
3. 当前真正卡点不在 HLS temporal 结构，而在：

```text
temporal/LNorm 的真实 golden 导出映射是否自洽
```

因此，本轮的目标不是继续扩展新模块，而是先把 `temporal_r1` 的 replay 对齐问题彻底收口，再把这条独立 temporal 线跑到：

```text
真实数据导出
-> native 验证
-> HLS csim
```

## 2. 本轮目标

本轮重点完成四件事：

1. 修复 `export_temporal_r1_testdata.py` 的真实 replay 偏差；
2. 生成 `hls_testdata/temporal_r1_c8_t6/{fusion0..fusion3,final}`；
3. 验证独立 `temporal_r1` 模块在真实数据上的 native 对齐；
4. 至少补一轮代表性真实数据 `HLS csim`，确认 temporal 模块已经从“结构开发态”进入“定点误差评估态”。

## 3. replay 偏差的根因定位与修复

### 3.1 首次异常

报告 15 中，`export_temporal_r1_testdata.py` 首次导出时出现：

```text
fusion0 replay mismatch too large: 0.28642749786376953
```

随后修正 `fusion` 对应的 `norm_gamma/norm_beta` 槽位后，又发现：

```text
final replay mismatch too large: 0.17421817779541016
```

这说明：

- temporal 卷积公式本身基本正确；
- 真正偏差主要来自 `norm_gamma/norm_beta` 的槽位映射，而不是 `CausConv1d` 方向本身。

### 3.2 修复后的槽位映射

本轮最终确认：

```text
fusion0 -> norm slot 1
fusion1 -> norm slot 2
fusion2 -> norm slot 3
fusion3 -> norm slot 4
final   -> norm slot 5
```

而不是之前误用的：

```text
fusion0..3 -> 3..6
final      -> 7
```

### 3.3 修复结果

修复后重新运行：

```bat
python hls_src/HLS/stage1_ifan_c8_r2/export_temporal_r1_testdata.py
```

输出为：

```text
fusion0: replay_max=7.15256e-07 replay_rmse=1.06246e-07
fusion1: replay_max=7.15256e-07 replay_rmse=1.06521e-07
fusion2: replay_max=1.43051e-06 replay_rmse=1.29191e-07
fusion3: replay_max=9.53674e-07 replay_rmse=1.35881e-07
final:   replay_max=7.15256e-07 replay_rmse=1.35862e-07
```

这说明真实 temporal replay 已经完全闭合，误差已回到 `1e-6` 量级。

## 4. 新生成的真实 temporal 数据

本轮已成功生成：

```text
hls_testdata/temporal_r1_c8_t6/
  fusion0/
  fusion1/
  fusion2/
  fusion3/
  final/
  manifest.json
  export_summary.json
```

每个子目录包含：

- `input.{npy,txt}`
- `weight.{npy,txt}`
- `bias.{npy,txt}`
- `output.{npy,txt}`
- `manifest.json`

语义为：

- `input`：`ReLU(ConvIco output)`
- `weight/bias`：真实 `TemporalConv1d` 权重
- `output`：真实 `TemporalConv1d` 原始输出（`LNormIco` 之前）

因此，`temporal_r1_c8_t6` 已经成为与报告 13 中 `layer2-5_c8_t6` 对应的下一层真实 golden 数据集。

## 5. 独立 temporal_r1 native 验证结果

### 5.1 构建方式

```bat
cd hls_src\HLS\stage1_ifan_c8_r2
build.bat temporal
```

生成：

```text
test_ifan_temporal_r1.exe
```

### 5.2 代表性验证

#### fusion0

```bat
test_ifan_temporal_r1.exe ..\..\..\hls_testdata\temporal_r1_c8_t6\fusion0
```

结果：

```text
Max Error: 0
RMSE: 0
PASS
```

#### final

```bat
test_ifan_temporal_r1.exe ..\..\..\hls_testdata\temporal_r1_c8_t6\final
```

结果：

```text
Max Error: 0
RMSE: 0
PASS
```

### 5.3 判断

这说明：

1. `TemporalConv1d R1` 的独立模块实现已经与真实导出的 temporal golden 完全一致；
2. 从报告 13 的 `ConvIco` 独立对齐，到本轮 `TemporalConv1d` 独立对齐，主干后处理链已经向前推进了一大步；
3. 当前 temporal 模块已经不再是“代码骨架”，而是一个真实可验证的独立子模块。

## 6. 独立 temporal_r1 HLS csim 结果

### 6.1 HLS 入口补强

本轮进一步修改：

- `run_hls.bat`
- `run_hls.tcl`

新增：

```text
Module = temporal
Top    = ifan_temporal_r1_top
Project= stage1_ifan_c8_r2_temporal_r1_hls_prj
```

从而允许 `temporal_r1` 以独立 HLS top 进入 `csim`。

### 6.2 csim 编译问题与修复

首次 `csim` 失败不是算法问题，而是 Vitis 生成的 `csim.mk` 只把 testbench 编进了 `HLS_SOURCES`，导致：

```text
undefined symbol: ifan_temporal_r1_top(...)
```

为先打通数值链路，本轮采用了一个最小、可逆的 `csim` 修复：

- 仅在 `__HLS_CSIM__` 下，把 `ifan_temporal_r1.cpp` 内联进 `test_ifan_temporal_r1.cpp`
- 不影响本地 native 构建
- 先保证真实数据 `csim` 可运行

### 6.3 代表性 csim 结果

命令：

```bat
set IFAN_TEMPORAL_R1_DATA_DIR=..\..\..\hls_testdata\temporal_r1_c8_t6\fusion0
set IFAN_TEMPORAL_R1_MAX_ERR_TOL=0.002
set IFAN_TEMPORAL_R1_RMSE_TOL=0.001
run_hls.bat csim temporal
```

结果：

```text
Max Error: 0.00138947
RMSE: 0.000368823
PASS
CSim done with 0 errors.
```

### 6.4 解释

这与报告 13 中 `layer2-5` 的现象一致：

- 本地 native 路径：严格 float 对账，可做到 `0` 或 `1e-6`
- Vitis HLS `csim` 路径：启用 `ap_fixed` 类型，出现可观但稳定的定点误差

因此，`temporal_r1` 现在也已经进入：

```text
真实定点误差评估阶段
```

而不是仍停留在数据接口或 replay 错误阶段。

## 7. 当前阶段结论

截至本轮，可以把 `temporal_r1` 的状态总结为：

### 已完成

1. `TemporalConv1d R1` 独立模块骨架建立；
2. 真实 `temporal_r1_c8_t6/{fusion0..final}` 数据导出；
3. 真实 replay 完全闭合；
4. 独立 native 对齐通过；
5. 独立 HLS `csim` 跑通；
6. 获得 temporal 模块首个真实定点误差锚点。

### 新增关键结论

`TemporalConv1d R1` 现在已经和 `layer2-5 ConvIco` 一样，具备了：

- 真实数据输入
- 独立 testbench
- native 对齐
- 独立 `csim`

这意味着 IFAN C8_R2 主干的可复用链已经从：

```text
ConvIco only
```

推进到：

```text
ConvIco
-> TemporalConv1d
```

## 8. 后续建议

下一步建议优先做：

1. 把本轮结果写回主线阶段记录与 README；
2. 开始抽取 `LNormIco R1` 独立模块；
3. 在 `LNormIco R1` 也稳定后，再考虑：

```text
layer2-5 ConvIco
-> TemporalConv1d R1
-> LNormIco R1
```

三段组合链的统一调度与综合。

如果后续目标偏向资源评估，也可以在 `temporal_r1` 上先补：

- 独立 synth
- 资源/延迟 summary
- 与 `layer2-5` 的资源占比对照

本报告的核心意义是：报告 15 中的“temporal replay 未闭合”问题已经被解决，`temporal_r1` 已经从探索态转入可复用子模块态。
