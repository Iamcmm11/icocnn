# Stage-1 当前硬件实现网络架构框图说明 02

日期：2026-05-26  
对象：`hls_src/HLS/stage1_ifan_c8_r2`  
用途：用文字定义当前硬件实现网络架构图的内容、风格、布局、模块连接关系和问题标记，作为后续绘制论文图、汇报图或优化设计图的依据。

## 1. 图片类型

建议图片类型为：

```text
硬件实现数据流框图 / HLS top-level architecture diagram
```

这张图不是普通神经网络结构图，也不是 PyTorch module graph。它要表达的是当前 HLS C++ top 如何把模块串接起来、哪些模块已经在当前综合边界内、哪些模块只是 native 对齐但尚未进入当前 HLS top，以及为什么当前结构会导致综合前端膨胀。

图应采用矢量框图风格，适合放入论文、阶段汇报和优化记录文档。推荐后续用 draw.io、Visio、PowerPoint 或 Mermaid/Graphviz 先画草图，再整理成论文用矢量图。

## 2. 图片目的

这张图的目的有四个：

1. 固定当前 Stage-1 HLS top 的真实边界。  
   当前 `run_hls.tcl` 综合的是 `ifan_stage1_top`，不包含 FeatureMABA、channel readout、region max、CleanVertices、SoftArgMax。

2. 展示当前数据流顺序。  
   从 `[2,T,5,4,8]` PHAT/LMS 输入开始，经过双分支 frontend、shared attention、attention fusion、PoolIco、4 个 fusion block、final block，输出 `[T,8,6,5,2,4]` pre-MABA feature。

3. 标出当前结构性问题。  
   重点标记整网静态 top、多处同型 IcoConv 调用、全尺寸 static buffer、权重路径展开、缺少显式调度层这些问题。

4. 为下一阶段优化提供对照。  
   后续优化图应能和这张图对比，体现从“整网静态串接”转向“少量计算引擎 + 显式调度 FSM + tile buffer”的结构变化。

## 3. 整体风格

推荐整体风格：

- 横向主数据流，从左到右阅读。
- 上半部分显示当前 `ifan_stage1_top` 实际 HLS 综合边界。
- 下半部分或右侧灰色虚线区显示 native 已对齐但暂未进入当前 HLS top 的 MABA/post-MABA 后续模块。
- 使用颜色区分模块类型：
  - 蓝色：输入、输出和主要 feature tensor。
  - 绿色：可复用计算模块，如 IcoConv、TemporalConv、LNorm、PoolIco。
  - 黄色：权重、kernel index、reorder index 等参数/索引输入。
  - 红色：当前结构问题和综合瓶颈标记。
  - 灰色虚线：当前未纳入 HLS top 的后续模块。
- 图中应避免画成“每一层一个神经网络 block”的算法图，而要突出 HLS 里实际存在的 buffer、函数调用点和综合边界。

## 4. 画面主体

画面主体分成三个区域：

### 区域 A：当前 HLS top 综合边界

这个区域用一个大边框表示，标题建议写：

```text
Current HLS Top: ifan_stage1_top
Synth boundary of current run_hls.tcl
```

边框内放当前真实进入 `csynth` 的模块。这个区域是整张图最重要的主体。

### 区域 B：参数与索引输入

这个区域放在区域 A 的上方，使用黄色小框表示：

```text
IfanStage1Weights
reorder_r2_stem
reorder_r2_main
reorder_r1
kernel_idx_stem
kernel_idx_main
```

这些输入用细箭头连接到对应的 IcoConv、PoolIco、LNorm、TemporalConv 模块。重点要表现：权重和 index 不是单纯存储成本，它们在当前实现中进入了较深的循环和展开路径。

### 区域 C：当前 HLS top 之外的后续模块

这个区域建议放在图的右侧，用灰色虚线框表示，标题建议写：

```text
Native-aligned modules, not in current HLS top
```

包含：

```text
FeatureMABA
Channel Readout
Region Max
CleanVertices
SoftArgMax
Coords Output
```

这些模块 native C++ 已对齐，但当前不应画进 `ifan_stage1_top` 的实线综合边界里。它们可以用灰色虚线箭头接在 Stage-1 output 后面，表示后续阶段。

## 5. 布局结构

推荐主图使用三行结构。

第一行：参数和索引输入。

```text
IfanStage1Weights
kernel_idx_stem / kernel_idx_main
reorder_r2_stem / reorder_r2_main / reorder_r1
```

第二行：当前 HLS top 主数据流。

```text
Input [2,T,5,4,8]
  -> Extract PHAT/LMS
  -> PHAT Frontend Branch
  -> LMS Frontend Branch
  -> Shared Attention
  -> Attention Fuse
  -> Add PHAT/LMS
  -> PoolIco R2->R1
  -> Fusion Block x4
  -> Final Block
  -> Stage-1 Output [T,8,6,5,2,4]
```

第三行：问题标记和未进入 top 的后续模块。

```text
Problem Markers P1..P6
FeatureMABA -> Channel Readout -> Region Max -> CleanVertices -> SoftArgMax
```

视觉上建议让第二行占据最大面积；第三行的问题标记用红色 callout 指向对应模块。

## 6. 模块连接关系

当前 `ifan_stage1_top` 的连接关系如下。

### 6.1 输入拆分

```text
Input [2,T,5,4,8]
  -> extract_feature_channel(input, 0) -> phat_input
  -> extract_feature_channel(input, 1) -> lms_input
```

输入包含两个 feature channel：

- channel 0：PHAT
- channel 1：LMS

图中应画成一个输入框分成两条支路。

### 6.2 PHAT frontend branch

模块名称建议写：

```text
PHAT Frontend Branch
```

内部顺序：

```text
ico_conv_r2_stem_engine
-> relu_feature_r2
-> ico_conv_r2_main_engine (res0)
-> relu_feature_r2
-> ico_conv_r2_main_engine (res1)
-> lnorm_ico_r2_engine
-> residual_add_relu_r2
```

输出：

```text
phat_direct
phat_enhanced
```

图中应画出 `direct` 和 `enhanced` 两个输出，因为后续 attention fusion 会同时使用它们。

### 6.3 LMS frontend branch

模块名称建议写：

```text
LMS Frontend Branch
```

内部顺序与 PHAT branch 相同：

```text
ico_conv_r2_stem_engine
-> relu_feature_r2
-> ico_conv_r2_main_engine (res0)
-> relu_feature_r2
-> ico_conv_r2_main_engine (res1)
-> lnorm_ico_r2_engine
-> residual_add_relu_r2
```

输出：

```text
lms_direct
lms_enhanced
```

图中要强调 PHAT/LMS 逻辑同构，但当前 HLS top 中是两个调用路径，不等价于一个硬件单元自动复用。

### 6.4 Shared attention

模块名称建议写：

```text
Shared Attention Engine
```

实际调用两次：

```text
shared_attention_engine(phat_enhanced) -> phat_attention
shared_attention_engine(lms_enhanced)  -> lms_attention
```

内部顺序：

```text
lnorm_ico_r2_engine
-> ico_conv_r2_main_engine (attn0)
-> relu_feature_r2
-> ico_conv_r2_main_engine (attn1)
-> sigmoid_feature_r2
```

图中应画成一个“Shared Attention”大框，但在大框内或旁边标注：

```text
Called twice: PHAT path and LMS path
```

这是一个重要问题点：算法上权重共享，并不代表 HLS 自动生成单一 attention 硬件单元。

### 6.5 Attention fuse

模块名称建议写：

```text
Attention Fuse
```

连接关系：

```text
phat_direct + phat_enhanced * phat_attention -> phat_fused
lms_direct  + lms_enhanced  * lms_attention  -> lms_fused
```

对应函数：

```text
attention_fuse_r2
```

随后：

```text
add_feature_r2(phat_fused, lms_fused) -> fused_r2
```

图中建议把 `attention_fuse_r2` 画成两个并列小框，再用一个 `add_feature_r2` 合并框。

### 6.6 PoolIco R2 to R1

模块名称建议写：

```text
PoolIco R2->R1
```

连接关系：

```text
fused_r2
  -> pool_ico_r2_to_r1_engine
  -> fused_r1_a
```

输入 shape 语义从 R2 空间变为 R1 空间：

```text
[T,8,6,5,4,8] -> [T,8,6,5,2,4]
```

图中应明确标出 resolution change：

```text
R2 spatial: H=4,W=8
R1 spatial: H=2,W=4
```

### 6.7 Fusion Block x4

模块名称建议写：

```text
Fusion Block x4
```

内部顺序：

```text
ico_conv_r1_main_engine
-> relu_feature_r1
-> temporal_conv1d_r1_engine
-> lnorm_ico_r1_engine
-> relu_feature_r1
```

循环关系：

```text
for block = 0..3:
    fused_r1_a -> fusion_block_engine -> fused_r1_b
    if block < 3:
        fused_r1_b -> copy -> fused_r1_a
```

图中可以画成一个大框 `Fusion Block x4`，内部写 `same function, different weights`。同时用红色问题标记强调：当前 top 表达的是循环调用，但 HLS 仍需要处理整段 top 的数组和调用上下文，并未形成明确的全局调度器。

### 6.8 Final Block

模块名称建议写：

```text
Final Block
```

内部顺序：

```text
ico_conv_r1_main_engine
-> relu_feature_r1
-> temporal_conv1d_r1_engine
-> lnorm_ico_r1_engine
```

注意 final block 不再执行最后的 ReLU。

输出：

```text
Stage-1 Output [T,8,6,5,2,4]
```

这个输出是当前 HLS top 的最终边界，也是后续 FeatureMABA 的输入。

### 6.9 HLS top 之外的后续模块

这些模块建议画在灰色虚线区：

```text
Stage-1 Output
  - - -> FeatureMABA
  - - -> Channel Readout
  - - -> Region Max
  - - -> CleanVertices
  - - -> SoftArgMax
  - - -> Coords [T,3]
```

标注：

```text
Native aligned, not synthesized in current ifan_stage1_top
```

这可以避免读图者误以为当前资源评估已经包含 MABA 和 post-MABA。

## 7. 每个模块名称清单

图中建议使用以下模块名称。

当前 HLS top 内：

```text
Input PHAT/LMS Tensor
Extract PHAT/LMS
PHAT Frontend Branch
LMS Frontend Branch
IcoConv R2 Stem
IcoConv R2 Main
LNormIco R2
Residual Add + ReLU
Shared Attention Engine
Sigmoid Attention
Attention Fuse
PHAT/LMS Add
PoolIco R2->R1
Fusion Block x4
IcoConv R1 Main
TemporalConv1d R1
LNormIco R1
Final Block
Stage-1 Output
```

参数和索引：

```text
IfanStage1Weights
kernel_idx_stem
kernel_idx_main
reorder_r2_stem
reorder_r2_main
reorder_r1
```

当前 HLS top 外：

```text
FeatureMABA
Channel Readout
Region Max
CleanVertices
SoftArgMax
Coords Output
```

问题标记：

```text
P1: Whole-network static top
P2: Repeated IcoConv call sites
P3: Weight conversion/index selection in inner loops
P4: Full-tensor static buffers
P5: Missing explicit scheduler/FSM
P6: csynth stops before final resource report
```

## 8. 重点强调的内容

图中应重点强调以下信息：

1. 当前综合边界只到 Stage-1 output。  
   FeatureMABA 和 post-MABA 虽然 native 已对齐，但不属于当前 `ifan_stage1_top` 的 HLS 综合边界。

2. PHAT/LMS 双分支是同构路径。  
   但当前 C++ 调用结构没有把它们收敛成一个硬件单元分时复用。

3. Shared attention 是权重共享，不等价于硬件自动共享。  
   图中要标出它被 PHAT/LMS enhanced feature 分别调用。

4. IcoConv 是主要计算和展开压力来源。  
   尤其是 `ico_conv_r2_main_engine` 和 `ico_conv_r1_main_engine`。

5. 权重路径是局部高亮问题。  
   `to_weight_t`、kernel index 选择、rotated kernel 权重访问在当前实现中处于深层循环路径。

6. full-tensor static buffer 是结构性问题。  
   需要在图中看到 PHAT/LMS direct/enhanced/attention/fused 等中间张量，否则无法解释 Array/Struct 阶段膨胀。

7. 下一阶段优化方向不是继续接模块，而是建立调度层。  
   图中可以在右下角放一个优化目标 callout：

```text
Next target:
engine reuse + explicit scheduler + tile/ping-pong buffers
```

## 9. 当前结构问题标记

建议在图中用红色编号标记以下问题。

### P1：整网静态 top

标记位置：`Current HLS Top: ifan_stage1_top` 大边框右上角。

说明：

```text
Current top describes the whole Stage-1 network as one static design.
HLS sees a large combined IR before resource binding.
```

中文注释：

```text
整网静态串接，综合前端 IR 过大。
```

### P2：同型 IcoConv 多调用点展开

标记位置：PHAT/LMS frontend、Shared Attention、Fusion Block、Final Block 中所有 IcoConv 旁边。

说明：

```text
IcoConv functions are reused in C++ source, but each call site still expands in HLS analysis.
```

中文注释：

```text
软件函数复用不等于硬件单元复用。
```

### P3：权重转换和 kernel index 选择位于深层循环

标记位置：`IfanStage1Weights/kernel_idx_main -> IcoConv R2/R1 Main` 的箭头上。

说明：

```text
to_weight_t and kernel index selection appear repeatedly in inner-loop paths.
```

中文注释：

```text
权重路径在内层循环反复实例化。
```

### P4：全尺寸 static 中间 buffer

标记位置：`phat_direct`、`phat_enhanced`、`lms_direct`、`lms_enhanced`、`phat_attention`、`lms_attention`、`phat_fused`、`lms_fused`、`fused_r2`、`fused_r1_a/b` 附近。

说明：

```text
Multiple full-tensor static buffers increase Array/Struct pressure.
```

中文注释：

```text
全量中间缓存增加数组结构变换压力。
```

### P5：缺少显式调度 FSM

标记位置：当前 top 大边框底部，或 Fusion Block x4 附近。

说明：

```text
No explicit scheduler constrains one hardware engine to execute multiple blocks sequentially.
```

中文注释：

```text
缺少统一调度层，复用关系没有硬件化表达。
```

### P6：没有最终资源表

标记位置：图右上角或标题下方。

说明：

```text
Current csynth produced only csynth_design_size.rpt.
No final LUT/DSP/BRAM report yet.
```

中文注释：

```text
当前停在 design-size 阶段，尚无最终资源表。
```

## 10. 推荐图中文字版草图

下面是框图的文字草图，可直接作为后续绘图参考。

```text
                         [IfanStage1Weights]
      [kernel_idx_stem/main] [reorder_r2/r1 tables]
                  |              |              |
                  v              v              v
+--------------------------------------------------------------------+
| Current HLS Top: ifan_stage1_top                                   |
|                                                                    |
| [Input PHAT/LMS Tensor]                                            |
|        |                                                           |
|        v                                                           |
| [Extract PHAT/LMS]                                                 |
|     |              |                                               |
|     v              v                                               |
| [PHAT Frontend] [LMS Frontend]        P2: repeated IcoConv calls   |
|     | direct/enh  | direct/enh                                      |
|     v              v                                               |
| [Shared Attention for PHAT] [Shared Attention for LMS]             |
|     |              |                         P3: weight/index path   |
|     v              v                                               |
| [Attention Fuse PHAT] [Attention Fuse LMS]                         |
|     |              |                                               |
|     +-------> [PHAT/LMS Add]                                       |
|                    |                                               |
|                    v                                               |
|            [PoolIco R2->R1]                                        |
|                    |                                               |
|                    v                                               |
|            [Fusion Block x4]     P4: full-tensor static buffers    |
|                    |                                               |
|                    v                                               |
|              [Final Block]                                         |
|                    |                                               |
|                    v                                               |
|       [Stage-1 Output: T x 8 x 6 x 5 x 2 x 4]                      |
|                                                                    |
| P1: whole-network static top                                       |
| P5: no explicit scheduler/FSM                                      |
+--------------------------------------------------------------------+
                    |
                    | dashed, not in current HLS top
                    v
  - - - - - - - - - - - - - - - - - - - - - -
  | [FeatureMABA] -> [Channel Readout] -> [Region Max] |
  |      -> [CleanVertices] -> [SoftArgMax] -> [Coords]|
  | Native aligned, not synthesized in current top     |
  - - - - - - - - - - - - - - - - - - - - - -

P6: current csynth stops before final resource report.
```

## 11. 当前结论

当前硬件实现网络架构图应当表达一个核心判断：

```text
Stage-1 当前 C++ top 是数值正确的整网静态串接版本，
但还不是资源可控的硬件复用调度版本。
```

因此，图里不能只画“网络层顺序”，还必须画出：

- HLS top 综合边界
- 同型模块多调用点
- 权重和 index 进入深层计算路径
- full-tensor static buffer
- native 已对齐但尚未进入 top 的后续模块
- 当前缺少显式调度层的问题

这张图后续应作为 Stage-1 scheduler 设计图的对照基线。下一张优化图应展示如何将当前多个同型调用点收敛为少量 IcoConv/TemporalConv 引擎，并由 FSM 或调度表按 block 顺序分时执行。
