# IFAN_C8_R2_MABA Pre-Readout Stage-1 HLS 工程

本目录是 `ifan_c8_r2_maba_pre_readout_best` 定型网络的第一阶段 HLS 工程入口。

第一阶段只覆盖主干网络到 MABA 之前的部分，目标是先把主干的数据边界、可复用功能块和顶层调度固定下来。MABA、channel readout、region max、CleanVertices、SoftArgMax 留到后续阶段。

## 1. 阶段边界

- 输入：算法侧已经生成的 PHAT/LMS 双特征图序列，形状为 `[2, T, 5, 4, 8]`。
- 输出：pre-MABA / pre-readout 特征张量，形状为 `[T, 8, 6, 5, 2, 4]`。
- 当前不包含：PHAT/LMS 特征生成、FeatureMABA、channel readout、region max、CleanVertices、SoftArgMax。

当前默认 `T=6`，由 `IFAN_STAGE1_T` 宏控制。后续导出不同长度的 golden data 时，可以在编译期修改该宏。

## 2. 可复用功能块

本工程不是继续复制 layer-specific 大函数，而是先拆出一组可复用硬件功能块：

| 功能块 | 作用 |
|---|---|
| `ico_conv_r2_stem_engine` | R2 分辨率 stem 卷积，`Cin=1, Rin=1 -> Cout=8, Rout=6` |
| `ico_conv_r2_main_engine` | R2 分辨率主干卷积，`Cin=8, Rin=6 -> Cout=8, Rout=6` |
| `pool_ico_r2_to_r1_engine` | pre-fusion pooling，`r=2 -> r=1` |
| `ico_conv_r1_main_engine` | R1 分辨率 fusion/final 卷积 |
| `temporal_conv1d_r1_engine` | 标准 causal Conv1d，`C=8, kernel=5` |
| `lnorm_ico_r1_engine` / `lnorm_ico_r2_engine` | 对每个空间位置的 `(C,R)` 做 LNormIco |
| elementwise engines | ReLU、Sigmoid、residual add、attention fusion |

这些模块后续可以继续做定点位宽 sweep、tile 调整和资源复用优化。

## 3. Stage-1 顶层调度

顶层函数是 `ifan_stage1_top`，表达当前 IFAN_C8_R2 主干的资源复用调度顺序：

```text
PHAT branch: stem + residual
LMS branch:  stem + residual
  |
shared attention，对 PHAT/LMS enhanced feature 分别调用
  |
attention fusion: direct + enhanced * sigmoid(attention)
  |
PHAT/LMS 两支相加
  |
PoolIco: r=2 -> r=1
  |
4 x (IcoConv -> ReLU -> TemporalConv1d -> LNorm -> ReLU)
  |
Final block: IcoConv -> ReLU -> TemporalConv1d -> LNorm
  |
Stage-1 output: [T, 8, 6, 5, 2, 4]
```

这个输出会作为后续 `FeatureMABATemporalRefiner` 的输入。

## 4. 构建与运行

Windows：

```bat
cd hls_src\HLS\stage1_ifan_c8_r2
build.bat
test_ifan_stage1.exe
```

Linux / MSYS：

```bash
cd hls_src/HLS/stage1_ifan_c8_r2
make clean && make run
```

Vitis HLS：

```bat
run_hls.bat csim
run_hls.bat synth
```

## 5. 当前验证状态

当前工程已经完成：

- Stage-1 HLS 输入/输出边界固定；
- 可复用功能块接口和张量形状固定；
- `ifan_stage1_top` 主干调用顺序固定；
- native `g++` smoke test 构建并通过；
- smoke test 输出形状已验证为 `[6, 8, 6, 5, 2, 4]`。

需要注意：

- 当前 smoke test 使用的是合成权重、合成 reorder 表和合成 kernel index，只用于验证调用链和有限值输出。
- Vitis HLS `csim` 已能完成编译并生成 `csim.exe`，但完整 debug-mode Stage-1 仿真较慢，后续建议先做模块级 HLS testbench 或临时减小 `IFAN_STAGE1_T`。
- 当前还没有完成 PyTorch checkpoint 的真实权重导出，也没有完成与 PyTorch `final_head_logits` 的 golden data 对齐。
- 当前还没有完成 `csynth` 资源闭合。

## 6. 下一步工作

后续建议按下面顺序推进：

1. 从 checkpoint 导出 `ifan_c8_r2_maba_pre_readout_best` 的真实权重，填充 `IfanStage1Weights`。
2. 导出真实 `reorder_r2_stem`、`reorder_r2_main`、`reorder_r1`、`kernel_idx_stem`、`kernel_idx_main`。
3. 导出 PyTorch Stage-1 golden data，对齐 `final_head_logits`，即 MABA 前的输出。
4. 先做模块级 C-sim 对齐，再做 `ifan_stage1_top` 整体对齐。
5. 在正确性稳定后，推进 `input_t / weight_t / act_t / acc_t` 定点位宽 sweep。
6. 最后跑 `csynth`，记录资源、latency、Estimated Clock 和关键 loop II。

## 7. 设计定位

这个工程的目标不是第一版就达到最低 latency，而是先形成一个清晰、可复用、可验证的 Stage-1 主干硬件框架。

后续硬件创新点可以围绕：

- IcoConv 功能块复用；
- PHAT/LMS 双分支分时调度；
- fusion blocks 的同构复用；
- 定点化和资源压缩；
- 后续 MABA 的 LightMamba 风格时序 scan 与 PoT 定点优化。

