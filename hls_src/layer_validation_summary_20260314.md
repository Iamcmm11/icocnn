# Layer0-Layer1-Layer2-5 仿真验证总结

> 2026-03-21 更新：本文档中的 `layer0`、`layer1`、`layer2-5` 验证结果已按当前重新生成的 testdata 重跑并更新。

## 文档说明

本文档汇总当前 `layer0`、`layer1` 和 `layer2-5` 的仿真验证情况，重点说明两部分内容：

1. 每一层目前验证了哪些内容。
2. 每一层当前得到的验证结果。

其中：

- `layer0` 的结构说明和历史验证过程，参考 [Layer0代码分类说明.md](G:/3DSLED/icocnn/Layer0代码分类说明.md)
- `layer1` 的中间层对比结果，采用本次重新导出的 Python/C 调试文件与 `compare_intermediate_layer1.py` 的输出结果
- `layer2-5` 采用共享 `ConvIco(r=1)` 验证块，验证脚本位于 [tools/layer2-5](G:/3DSLED/icocnn/tools/layer2-5)

---

## Layer0 验证情况

### 验证了什么

`layer0` 当前已经完成两类验证：

1. 完整流程验证
- 读取 `input_rearranged.txt`
- 读取 `weight.txt`、`bias.txt`
- 读取 `kernel_expansion_idx.txt`、`reorder_idx.txt`
- 运行 `conv_ico_layer0`
- 将 C 端输出与 Python 参考输出 `output_layer0.txt` 做逐点比较

2. 中间层对齐验证
- 输入张量一致性
- `PadIco` 后输出一致性
- `Reshape` 后输入一致性
- 用于确认 C 端与 Python 端在关键中间步骤上的实现一致

### 结构说明

`layer0` 当前实现已经具备以下特点：

1. 完成了 `SmoothVertices`、`PadIco`、`get_kernel`、主卷积与输出端顶点平滑等核心流程。
2. 已移除部分大中间缓冲，结构上更接近后续 HLS 集成版本。
3. 已作为后续层验证的基准层使用。

### 验证结果

基于当前重新生成的 `layer0` testdata，`layer0` 端到端验证结果为：

- Max Error：`7.15256e-07`
- RMSE：`6.49467e-08`
- 结论：`PASS`

中间层对齐方面，本次重跑后已确认以下项目通过：

- 输入层：`PASS`
- Padding 后：`PASS`
- Reshaped 输入：`PASS`

### 当前结论

`layer0` 已经完成完整验证，可以作为后续 `layer1`、`layer2` 等层继续扩展验证的参考基线。

---

## Layer1 验证情况

### 验证了什么

`layer1` 当前已经完成两类验证：

1. 完整流程验证
- 使用 `layer1` 对应测试数据
- 运行 `conv_ico_layer1`
- 将 C 端输出与 Python 参考输出 `output_layer1.txt` 做逐点比较

2. 中间层对齐验证
- `Frame0 Input`
- `After PadIco`
- `Frame0 Final Output`
- 通过 `compare_intermediate_layer1.py` 对 Python 与 C 端的中间层结果进行逐项比较

### 验证结果

#### 1. 完整流程验证结果

本次基于重新生成的 `layer1` testdata，`layer1` 端到端输出验证结果为：

- Max Error：`8.58307e-06`
- RMSE：`6.49275e-07`
- 结论：`PASS`

#### 2. 中间层对比结果

本次重新导出的 Python/C 中间层文件，经 `compare_intermediate_layer1.py` 对比如下：

- `Frame0 Input`
  - Max Error：`0.00000000e+00`
  - RMSE：`0.00000000e+00`
  - Mean Abs：`0.00000000e+00`
  - 结论：`PASS`

- `After PadIco`
  - Max Error：`4.80000000e-07`
  - RMSE：`3.67841395e-08`
  - Mean Abs：`6.97395833e-09`
  - 结论：`PASS`

- `Frame0 Final Output`
  - Max Error：`5.25000000e-06`
  - RMSE：`6.42740787e-07`
  - Mean Abs：`4.51994466e-07`
  - 结论：`PASS`

### 当前结论

`layer1` 已经完成：

1. 端到端输出验证
2. 关键中间层对齐验证
3. Python 与 C 端的一致性确认

因此，`layer1` 当前也已经具备继续向后续层扩展验证的条件。

---

## Layer2-5 验证情况

### 验证了什么

`layer2-5` 当前采用共享 `ConvIco(r=1)` 验证块，已经完成两类验证：

1. 完整流程验证
- 对 `layer2`、`layer3`、`layer4`、`layer5` 分别生成专属 testdata
- 使用统一的 `conv_ico_layer2_5`
- 将 C 端输出与各层 Python 参考输出 `output.txt` 做逐点比较

2. 中间层对齐验证
- `Frame0 Input`
- `After PadIco`
- `Frame0 Final Output`
- 通过 `compare_intermediate_layer2_5.py` 对每一层的 Python/C 中间层结果逐项比较

### 结构说明

`layer2-5` 当前验证实现具备以下特点：

1. 统一固定为 `Cin=Cout=32`、`Rin=Rout=6`、`H=2`、`W=4` 的共享卷积块。
2. 采用紧凑 7 邻域权重，通过 `kernel_expansion_idx` 在 MAC 过程中展开使用。
3. 保留 `PadIco + Conv + 输出顶点后处理` 的完整行为，以对齐 PyTorch `ConvIco(r=1)`。

### 验证结果

#### 1. 完整流程验证结果

本次基于重新生成的 `layer2-5` testdata，四层端到端输出验证结果如下：

- `layer2`
  - Max Error：`4.05312e-06`
  - RMSE：`3.93617e-07`
  - 结论：`PASS`

- `layer3`
  - Max Error：`3.33786e-06`
  - RMSE：`3.77224e-07`
  - 结论：`PASS`

- `layer4`
  - Max Error：`4.05312e-06`
  - RMSE：`4.01841e-07`
  - 结论：`PASS`

- `layer5`
  - Max Error：`3.33786e-06`
  - RMSE：`3.25302e-07`
  - 结论：`PASS`

#### 2. 中间层对比结果

本次重新导出的 Python/C 中间层文件，经 `compare_intermediate_layer2_5.py` 对比如下：

- `layer2`
  - `Frame0 Input`: Max Error `5.00000000e-06`, RMSE `9.96721808e-07`, Mean Abs `3.57694010e-07`, `PASS`
  - `After PadIco`: Max Error `5.00000000e-06`, RMSE `9.83987668e-07`, Mean Abs `4.04418637e-07`, `PASS`
  - `Frame0 Final Output`: Max Error `6.77000000e-06`, RMSE `1.92575291e-06`, Mean Abs `1.23465039e-06`, `PASS`

- `layer3`
  - `Frame0 Input`: Max Error `5.00000000e-06`, RMSE `9.37262428e-07`, Mean Abs `3.37140625e-07`, `PASS`
  - `After PadIco`: Max Error `5.00000000e-06`, RMSE `9.56868280e-07`, Mean Abs `3.55857465e-07`, `PASS`
  - `Frame0 Final Output`: Max Error `7.18000000e-06`, RMSE `1.86177701e-06`, Mean Abs `1.17315719e-06`, `PASS`

- `layer4`
  - `Frame0 Input`: Max Error `5.00000000e-06`, RMSE `1.07281488e-06`, Mean Abs `4.09910156e-07`, `PASS`
  - `After PadIco`: Max Error `4.99000000e-06`, RMSE `1.04023860e-06`, Mean Abs `4.33477865e-07`, `PASS`
  - `Frame0 Final Output`: Max Error `5.91000000e-06`, RMSE `2.21709636e-06`, Mean Abs `1.58174544e-06`, `PASS`

- `layer5`
  - `Frame0 Input`: Max Error `5.00000000e-06`, RMSE `1.07958236e-06`, Mean Abs `4.29705729e-07`, `PASS`
  - `After PadIco`: Max Error `5.00000000e-06`, RMSE `1.09157126e-06`, Mean Abs `4.46580382e-07`, `PASS`
  - `Frame0 Final Output`: Max Error `6.77000000e-06`, RMSE `1.95419042e-06`, Mean Abs `1.26882773e-06`, `PASS`

### 当前结论

`layer2-5` 已经完成：

1. 共享卷积核的四层端到端输出验证
2. 关键中间层对齐验证
3. Python 与 C 端的一致性确认

因此，`layer2-5` 当前已经具备继续推进 HLS 综合与资源评估的条件。

---

## 当前总体结论

截至目前：

- `layer0`：验证通过
- `layer1`：验证通过
- `layer2-5`：验证通过

其中：

- `layer0` 已完成完整流程验证，并已作为基础参考层稳定使用。
- `layer1` 已完成完整输出验证与中间层对齐验证，结果均为 `PASS`。
- `layer2-5` 已完成共享验证块的四层端到端验证与中间层对齐验证，结果均为 `PASS`。

这说明当前从 `layer0` 到 `layer5` 的 C 端实现与 Python 参考模型已经形成连续、可追踪、可扩展的验证链路，并且已基于当前最新生成的数据重新确认通过。

---

## 后续建议

下一步继续推进后续层时，建议保持当前同样的验证结构：

1. 先生成该层专属 testdata。
2. 先跑端到端输出比对。
3. 再补关键中间层对齐。
4. 每层单独保留验证脚本、调试脚本与结果文档。

这样后面即使网络更深，定位问题也会比较直接。

---

## 备注：面向 Layer2-5 的后续优化思路

以下内容属于后续优化参考，不作为当前 `layer0`、`layer1`、`layer2-5` 基础验证通过的前置条件。

建议在 `layer2-5` 的基础功能验证全部完成之后，再开始实际实现。

### 1. 总体目标

后续优化的重点，不再是继续为每一层分别写一份独立的卷积实现，而是考虑抽象出一套可复用的参数化卷积计算块，用于覆盖网络中重复出现的主干 block，尤其是 `layer2-5`。

从当前网络结构看，`layer2-5` 具有以下共同特征：

1. 都属于重复的空间卷积 block
2. 通道规模保持一致，基本都是 `32 -> 32`
3. 旋转维保持 `R = 6`
4. 空间尺寸在降采样后保持较小规模
5. 后面接的时域卷积结构也具有较强重复性

因此，`layer2-5` 比 `layer0` 更适合作为“统一参数化 HLS 计算核”复用的目标层。

### 2. 推荐的后续优化方向

建议的后续优化路线如下：

1. 保留 `layer0` 作为前端特化层
2. 保留 `layer1` 作为从特化实现过渡到参数化实现的第一层
3. 将 `layer2-5` 统一映射为一个可配置的 `ConvIco` 主干计算块

该统一计算块后续可逐步引入：

1. `IC/OC tiling`
2. 紧凑 7 邻域权重表示
3. tile 级局部 `psum` 累加
4. tile 级输入窗口缓存
5. `smooth + pad + conv` 的更细粒度流式融合

### 3. 为什么建议放在 Layer2-5 再做

原因主要有三点：

1. `layer0` 的输入规模小、功能特殊，更适合作为稳定的前端参考层，而不是通用块模板。
2. `layer1` 虽然已经可以开始体现 `IC/OC tiling`，但它更像是过渡层，主要用于验证这条路线是可行的。
3. `layer2-5` 结构重复、参数一致性高，更适合体现“设计一个通用块然后反复调用”的硬件复用思想。

因此，从工程效率和论文表达两方面看，把真正的统一参数化 block 放在 `layer2-5` 上最合适。

### 4. 后续真正实现时的建议顺序

当 `layer2-5` 全部完成基础验证之后，建议按以下顺序推进：

1. 先完成 `layer2` 的独立 C/HLS 版本，确认输入输出和中间层对齐
2. 再比对 `layer2-5` 的通道规模、空间尺寸、旋转维与数据流是否足够一致
3. 若一致性满足要求，则抽象出统一参数化 `ConvIco` 计算块
4. 先实现最保守的 tile 版统一卷积核，保证功能正确
5. 再逐步优化 `psum`、输入 tile buffer、输出写回和层间接口

### 5. 当前可作为后续参考的已有基础

目前已经完成、可直接作为后续参考的基础包括：

1. `layer0` 的部分融合数据流
2. `layer1` 去掉完整 `kernel` 展开缓存后的紧凑权重实现
3. `layer1` 去掉 `input_after_smooth` 显式缓存后的进一步部分融合版本
4. `layer1` 第一版 `IC/OC tiling` 的可综合验证结果
5. `layer2-5` 共享验证块的完整 Python/C 验证链路

这意味着，后续做 `layer2-5` 通用 block 时，不需要从零开始，而是可以直接复用当前已经验证过的这几条关键设计思路。
