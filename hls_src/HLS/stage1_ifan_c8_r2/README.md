# IFAN_C8_R2 Stage-1 分区 HLS 工程

本目录已经从“完整 Stage-1 纯 PL top”重构为 PS/PL 分工后的模块集合。

当前默认活跃 HLS 边界是双特征前端：

```text
frontend_dual_feature/ifan_dual_frontend_top
```

融合后的 R1 主干不再在这里复制一套切块代码。后续主干硬件工作直接沿用并参数化 `hls_src/HLS/layer2-5`。

## 目录边界

| 目录 | 定位 |
|---|---|
| `frontend_dual_feature/` | 当前默认 PHAT/LMS 双特征前端 top，输出 `[T, 8, 6, 5, 4, 8]` |
| `full_stage1_legacy/` | 旧完整 `ifan_stage1_top`，仅用于 native 回归和历史对比 |
| `feature_maba/` | pre-readout FeatureMABA 独立切片，后续 PL 候选 |
| `post_maba/` | channel readout、region max、CleanVertices、SoftArgMax 后处理切片 |
| `optimize/` | 阶段记录和执行计划 |

## 当前判断

`layer2-5` 已经是 baseline 中反复出现的 R1 ConvIco 共享硬件线：

```text
R1, H=2, W=4, Rin=Rout=6, Cin=Cout
```

C8_R2 的 4 个 fusion block 和 final block 使用同型 ConvIco。主要差异是 `C=8`、`T=6`，以及 ConvIco 外围还串了 temporal Conv1d/LNorm。因此下一阶段主干不应在本目录继续 fork，而应在 `layer2-5` 上做参数化复用。

## 构建

默认前端 smoke：

```bat
cd hls_src\HLS\stage1_ifan_c8_r2
build.bat
test_ifan_dual_frontend.exe
```

旧完整 Stage-1 回归：

```bat
build.bat full
test_ifan_stage1.exe
```

独立切片回归：

```bat
build.bat maba
test_feature_maba.exe

build.bat post
test_post_maba.exe
```

Vitis HLS 默认目标已切到 `ifan_dual_frontend_top`：

```bat
run_hls.bat csim
run_hls.bat synth
```

## 下一步

1. `frontend_dual_feature` 只作为双特征前端切片与 PS 侧暂存依据。
2. 在 `hls_src/HLS/layer2-5` 中参数化 `C=8,T=6`，覆盖 C8_R2 fusion/final ConvIco。
3. 单独推进 `feature_maba` 的 native/csim/synth，作为下一阶段 PL 创新模块资源证据。
4. `post_maba` 暂列为可选 PL head，等 MABA 资源闭合后再决定是否纳入。

详细计划见 `optimize/10_stage1_partition_refactor_plan.md`。
