# Temporal R1 独立复用推进状态报告 15

日期：2026-06-02

## 1. 与报告 13 的衔接

报告 13 已经完成了 `layer2-5` 这条共享 `ConvIco` 硬件线在 IFAN C8_R2 主干中的真实数据对齐，结论是：

1. `fusion0..fusion3/final` 五个真实 R1 `ConvIco` 子层已经全部与参数化后的 `layer2-5` 对齐；
2. 当前活跃主线中的 `ConvIco` 复用路径已经成立；
3. 后续主干继续推进时，新的主要边界不再是 `ConvIco`，而是其后的：
   - `TemporalConv1d`
   - `LNormIco`

因此，本轮工作不再扩展 `layer2-5` 本体，而是开始把 `ConvIco` 后面的时序后处理从旧 `full_stage1_legacy` 中抽出，尝试形成可独立复用的小模块。

## 2. 当前判断：为什么继续推进 temporal/LNorm

Torch 侧的 fusion/final block 并不是只有 `ConvIco`，而是：

```text
ConvIco -> ReLU -> TemporalConv1d -> LNormIco -> (optional ReLU)
```

具体体现在：

- `IFAN_Edge/ifan_edge/models/placeholders.py`
  - `FusionTemporalBlock`
  - `FinalFusionBlock`

而当前活跃 HLS 主线只完成了：

```text
ConvIco 部分（通过 layer2-5 C8/T6 参数化复用）
```

尚未在这条新主线上完成：

```text
TemporalConv1d R1
LNormIco R1
```

所以“继续推进”的含义不是回去重做完整大 top，而是继续把主干后半段拆成独立可复用小 IP。

换句话说，当前阶段继续推进的重点是：

1. `TemporalConv1d R1` 独立复用；
2. `LNormIco R1` 独立复用；
3. 最终与前面已经验证过的 `ConvIco` 形成：

```text
ConvIco IP + TemporalConv1d IP + LNormIco IP
```

## 3. 本轮已落地的代码

### 3.1 新增 temporal_r1 独立模块骨架

已新增目录：

```text
hls_src/HLS/stage1_ifan_c8_r2/temporal_r1/
```

其中已创建：

- `ifan_temporal_r1.hpp`
- `ifan_temporal_r1.cpp`
- `test_ifan_temporal_r1.cpp`

当前接口边界为：

```text
input  : [T=6, C=8, R=6, charts=5, H=2, W=4]
weight : [Cout=8, Cin=8, K=5]
bias   : [8]
output : [T=6, C=8, R=6, charts=5, H=2, W=4]
```

这一步的目标是先把 `TemporalConv1d R1` 单独拉成一个最小可编译、可验证、可综合的小模块。

### 3.2 复用 legacy temporal 实现思路

新模块当前不是从零重写算法，而是沿用旧 `full_stage1_legacy` 中已经存在的 temporal 核思路：

- `temporal_conv1d_r1_engine`
- temporal weight staging
- `OC_TILE` 级输出通道分块
- causal 窗口：
  ```text
  src_t = t - (K - 1) + k
  ```

因此，这一轮本质上是在做：

```text
legacy temporal engine -> 独立 temporal IP 的结构抽取
```

### 3.3 构建入口已接入

已修改：

- `hls_src/HLS/stage1_ifan_c8_r2/build.bat`
- `hls_src/HLS/stage1_ifan_c8_r2/Makefile`

新增本地构建目标：

```bat
build.bat temporal
```

以及对应 Makefile 目标：

```bash
make temporal
```

当前还没有把 `run_hls.*` 接成独立 temporal top 模式，因为真实 golden 还未最终钉死，先避免把错误 testdata 带入 HLS。

### 3.4 新增 temporal testdata 导出脚本

已新增：

- `hls_src/HLS/stage1_ifan_c8_r2/export_temporal_r1_testdata.py`

该脚本的设计意图是：

1. 复用报告 13 中已经对齐的 `layer2-5_c8_t6/{fusion0..final}` `ConvIco` 输出；
2. 再结合 `stage1_weights.npz` 中的：
   - `fusion_temporal_w`
   - `fusion_temporal_b`
   - `final_temporal_w`
   - `final_temporal_b`
   - `norm_gamma`
   - `norm_beta`
3. 生成独立的：

```text
hls_testdata/temporal_r1_c8_t6/
  fusion0/
  fusion1/
  fusion2/
  fusion3/
  final/
```

也就是说，这个脚本是报告 13 的 `ConvIco-only` 数据导出之后的下一层导出器。

## 4. 本轮最关键的新发现

### 4.1 Torch 的 LNormIco 真实语义已确认

本轮重新核对了 `icoCNN` 中 `LNormIco` 的真实定义，确认它不是简单的“按空间位置手写均值方差归一化”，而是：

```text
对每个 (chart, h, w) 位置上的 (C, R) 做 LayerNorm((C, R))
```

然后再做：

```text
output = norm(x) * weight[C,1] + bias[C,1]
```

这意味着后续 `LNormIco R1` 完全有条件作为独立小模块继续推进，并且它的行为边界已经明确。

### 4.2 temporal testdata replay 目前还未闭合

当前 `export_temporal_r1_testdata.py` 首次尝试导出时，`fusion0` 的 replay 出现了明显偏差：

```text
fusion0 replay mismatch too large: 0.28642749786376953
```

进一步排查后，至少可以确认两点：

1. 这不是 `LNormIco` 定义完全未知的问题，因为其真实 PyTorch 语义已被核对；
2. 也不是简单的 temporal kernel 正反方向问题，因为直接与 reverse 两种卷积方向试算后，仍不能对齐 `fusion_head_blocks_0`。

因此，当前 temporal 导出卡点更可能位于以下之一：

1. `ConvIco -> ReLU -> CausConv1d -> LNormIco -> ReLU` 中，导出时对中间节点的选取仍有偏差；
2. `fusion_temporal_w` / `fusion_temporal_b` 与 block 顺序或权重槽位映射还需要再核对；
3. `CausConv1d` 的 PyTorch 执行边界与我们当前 numpy replay 之间，还存在一层 padding / slicing / reshape 细节未对齐。

## 5. 当前阶段结论

本轮虽然还没有把 `TemporalConv1d R1` 的真实 golden 闭合，但已经完成了三个很重要的铺垫：

1. 明确了“报告 13 之后为什么要继续推进 temporal/LNorm”；
2. 把 `TemporalConv1d R1` 独立模块骨架、testbench、build 入口先搭了起来；
3. 把问题从“要不要继续做 1D”收缩成了“真实 temporal/LNorm golden 的导出映射还差最后一层钉死”。

因此，这一轮不是失败，而是把下一步工作的真正瓶颈找清楚了：

```text
当前卡点不在 HLS temporal 结构，而在 temporal/LNorm 的真实 golden 导出自洽性。
```

## 6. 对后续路线的影响

本轮之后，后续路线更清晰了：

### 不建议

1. 直接跳回完整 `full_stage1_legacy` 大 top 继续综合；
2. 在 temporal golden 未对齐前，就继续给新 `temporal_r1` 跑 HLS synth；
3. 在错误 replay 上继续叠加 `LNormIco` HLS 实现。

### 建议

1. 先修正 `export_temporal_r1_testdata.py` 的真实 replay 映射；
2. 让 `fusion0..final` 的 temporal 导出先全部闭合；
3. 再用该数据驱动：
   - `TemporalConv1d R1` native
   - `TemporalConv1d R1` csim
4. temporal 稳定后，再抽：
   - `LNormIco R1` 独立模块；
5. 最终形成主干可复用链：

```text
layer2-5 ConvIco
-> TemporalConv1d R1
-> LNormIco R1
```

## 7. 下一步建议

建议下一轮按以下顺序继续：

1. 在 Torch / export 脚本侧补 temporal block 的更细 debug 节点：
   - temporal input（即 ReLU(conv_output)）
   - temporal raw output（LNorm 前）
   - LNorm output（ReLU 前）
2. 用这些更细节点反推并修正 `export_temporal_r1_testdata.py`；
3. 当 `fusion0..final` replay 全部达到 `1e-5` 量级后：
   - 编译 `build.bat temporal`
   - 跑 `test_ifan_temporal_r1.exe`
   - 再把 `run_hls.*` 接成独立 temporal top；
4. temporal 稳定后，再进入 `LNormIco R1` 独立模块。

本报告的核心作用是把“为什么报告 13 之后要继续推进 1D/LNorm”与“当前具体卡点在哪”固化下来，避免后续工作又回到“大主干一起做”的老路线。
