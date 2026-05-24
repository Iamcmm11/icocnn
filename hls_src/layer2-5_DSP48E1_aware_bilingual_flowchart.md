# Layer2-5 DSP48E1-Aware Bilingual Architecture Description

这份文档将原本的 `Mermaid` 框架图改写成纯文字版图文描述，便于在普通 Markdown 预览器中直接阅读，也更适合后续整理进论文、汇报或设计说明。

## One-Paragraph Figure Description

这张 `layer2-5` 的 `DSP48E1-aware` 框架图可以概括为：整个共享 `ConvIco` 卷积块被划分为 `S0` 到 `S4` 五个层次，其中 `S0 Host / Quant Control` 负责离线量化、打包和控制信息下发，`S1 Input Value Path` 将主机侧浮点输入转换为内部 `feat_t` 载荷，并完成 `PadIco` 所需的填充、重排和规则化缓存组织，`S2 Weight Value Path` 则将紧凑存储的量化权重结合 `kernel expansion index` 展开为可供局部卷积窗口直接消费的 `3x3` 权重载荷；在此基础上，`S3 DSP MAC Array` 作为核心计算层，只执行 `feat_t × wgt_t -> prod_t` 的窄位宽乘法以及后续局部部分和、`RI` 级归并和 `output tile` 缓冲，不再采用旧式的 `acc_t × acc_t` 宽乘法路径；与此同时，`scale_t / exp_t` 等缩放元数据不进入同拍主乘法，而是通过独立的 `Scale Path` 在外围缓存和旁路传输，最终在 `S4 Scale / Normalize / Output` 中与输出和统一结合，完成数值恢复、归一化、极点修正和输出后处理，最后才回到主机可见的浮点输出。换句话说，这张图强调的不是“把所有量化信息塞进 MAC”，而是把真正的热点 `DSP48E1` 主路径收敛为窄载荷乘加，把复杂的 scale 管理放到旁路与后处理阶段，从而为 `layer2-5` 的后续位宽扫描、block floating 扩展和 DSP 映射优化建立一个更清晰的结构基线。

## 1. Overall View

这套 `layer2-5` 共享卷积块的新框架，不再从旧的 `input_t / weight_t / act_t / acc_t` 命名出发，而是从 `DSP48E1` 真正关心的两类信息出发来组织系统：

- `Value Path`：真正进入乘法器和累加器的数据载荷，也就是 `feat_t / wgt_t / prod_t / psum_t / osum_t`
- `Scale Path`：用于恢复数值范围的辅助元数据，也就是 `scale_t / exp_t / block metadata`

整个结构分为五层：

1. `S0 Host / Quant Control`  
   `主机与量化控制`
2. `S1 Input Value Path`  
   `输入数值通路`
3. `S2 Weight Value Path`  
   `权重数值通路`
4. `S3 DSP MAC Array`  
   `DSP MAC 阵列`
5. `S4 Scale / Normalize / Output`  
   `缩放恢复、归一化与输出`

这五层的核心思想是：让主乘法路径只做窄位宽、高吞吐、易映射到 `DSP48E1` 的乘加工作；而 `scale`、`exponent`、分组元数据等信息不进入同拍主乘法，而是在外围缓存、旁路传输，并在输出恢复阶段统一使用。

## 2. Main Dataflow

如果把整张图用一条主线串起来，可以写成：

`Host Float Input`  
`主机浮点输入`  
-> `Input Quant Loader`  
`输入量化加载器`  
-> `Feature Payload Buffer`  
`特征载荷缓冲`  
-> `Pad / Reorder Engine`  
`填充与重排引擎`  
-> `Regularized Feature SRAM`  
`规则化特征 SRAM`  
-> `Spatial Window Buffer`  
`空间窗口缓冲`  
-> `Narrow Multiply PE`  
`窄乘法 PE`  
-> `Local Product / Psum`  
`局部乘积与部分和`  
-> `RI Partial Merge`  
`RI 局部归并`  
-> `Output Tile Buffer`  
`输出 Tile 缓冲`  
-> `Restore / Normalize Engine`  
`恢复与归一化引擎`  
-> `Pole Fix / Output Post`  
`极点修正与输出后处理`  
-> `Host Float Output`  
`主机浮点输出`

与这条主线并行的另一条通路是权重侧：

`Quantized Weight Store`  
`量化权重存储`  
-> `Tile-Local Weight Unpack`  
`Tile 局部权重解包`  
-> `3x3 Weight Payload Buffer`  
`3x3 权重载荷缓冲`  
-> `Narrow Multiply PE`  
`窄乘法 PE`

再与之并行的是 scale 元数据通路：

`Offline Quant / Pack Control`  
`离线量化与打包控制`  
-> `Scale Metadata Buffer`  
`缩放元数据缓冲`  
-> `Restore / Normalize Engine`  
`恢复与归一化引擎`

这里最重要的结构约束是：

- 数值主路径在 `Narrow Multiply PE` 中执行的是 `feat_t × wgt_t -> prod_t`
- `scale_t / exp_t` 不进入同拍乘法
- 数值恢复只在 `Restore / Normalize Engine` 统一发生一次，而不是每次乘法后都恢复

## 3. Layer-By-Layer Description

### S0 Host / Quant Control  
### 主机与量化控制

这一层不承担热点乘加，而是承担格式准备和控制下发。它的起点是：

- `Host Float Input`  
  `主机浮点输入`
- `Offline Quant / Pack Control`  
  `离线量化与打包控制`

其中，`Host Float Input` 保持当前工程里的 `float` 接口边界，主要用于兼容现有 testbench、Python/C 验证链路和 host 侧数据组织。`Offline Quant / Pack Control` 则代表软件侧或离线流程负责的工作，包括：

- 将浮点输入或权重转换成硬件内部使用的窄载荷格式
- 生成 group scale、shared exponent 或其他 block metadata
- 打包出适合 `DSP48E1` 主路径消费的载荷布局

这一层的设计原则是：复杂的量化参数推导尽量离线完成，硬件在线路径只接收已打包好的 payload 与 metadata。

### S1 Input Value Path  
### 输入数值通路

这一层负责把原始输入整理成主 MAC 阵列可稳定消费的规则化特征窗口。它由以下模块组成：

- `Input Quant Loader`  
  `输入量化加载器`
- `Feature Payload Buffer`  
  `特征载荷缓冲`
- `Pad / Reorder Engine`  
  `填充与重排引擎`
- `Regularized Feature SRAM`  
  `规则化特征 SRAM`
- `Spatial Window Buffer`  
  `空间窗口缓冲`

其处理逻辑可以概括为三步：

1. 先将 host 侧浮点输入装载并转换为内部使用的 `feat_t`
2. 再完成 `PadIco` 所需的几何填充、重排和规则化访问组织
3. 最后形成适合 `3x3` 窗口读取的稳定特征缓冲

也就是说，输入侧不再被看作“直接从大数组里临时取数”，而是被重构成“先规则化，再把已经整理好的特征窗口交给主 MAC”。这样做的目的，是把 `PadIco`、重排索引、窗口组织等复杂访存行为隔离在乘法热路径之外。

### S2 Weight Value Path  
### 权重数值通路

这一层负责把紧凑存储的卷积权重整理成 `DSP48E1` 可消费的 `3x3` 权重载荷。组成模块是：

- `Quantized Weight Store`  
  `量化权重存储`
- `Kernel Expansion Index`  
  `卷积核展开索引`
- `Tile-Local Weight Unpack`  
  `Tile 局部权重解包`
- `3x3 Weight Payload Buffer`  
  `3x3 权重载荷缓冲`

当前 `layer2-5` 的权重并不是直接按完整 `3x3` 卷积核存储，而是保留紧凑形式，再借助 `kernel_expansion_idx` 展开。因此这里的设计重点不是“重新发明权重表示”，而是把“紧凑权重 -> 索引展开 -> 本地 `3x3` kernel payload”这件事稳定下来，并让其服务于后续的窄位宽乘法器。

换句话说，这一层做的是“参数整理”和“主 MAC 前的最后一跳供数”，而不是在热点路径上临时解码复杂索引。

### S3 DSP MAC Array  
### DSP MAC 阵列

这一层是整套设计的核心，负责真正的主乘加。它由以下模块组成：

- `Narrow Multiply PE`  
  `窄乘法 PE`
- `Local Product / Psum`  
  `局部乘积与部分和`
- `RI Partial Merge`  
  `RI 局部归并`
- `Output Tile Buffer`  
  `输出 Tile 缓冲`

这里必须明确强调一条新的设计红线：

主乘法不是 `acc_t × acc_t`，而是 `feat_t × wgt_t -> prod_t`。

也就是说，输入和权重应以各自合适的“窄载荷”位宽进入 `DSP48E1`，先生成乘积 `prod_t`，再进入本地部分和 `psum_t`，最后归并为输出和 `osum_t`。这比“先把输入和权重都扩成最终累加位宽再相乘”的旧方式更符合 DSP 导向设计。

这层的计算组织可以分成三段理解：

1. `Narrow Multiply PE`  
   只负责紧凑位宽乘法，是最接近 `DSP48E1` 甜点区的操作单元
2. `Local Product / Psum`  
   接收乘积后做局部部分和累加，避免一开始就冲击全局输出缓冲
3. `RI Partial Merge` 和 `Output Tile Buffer`  
   将局部部分和进一步归并，形成更稳定、更可调度的输出 tile 边界

因此，`ri_partial` 和 `output_tile` 仍然有意义，但它们不再是图里的主命名中心，而是作为“局部部分和边界”和“输出 tile 边界”被重新解释。

### S4 Scale / Normalize / Output  
### 缩放恢复、归一化与输出

这一层负责把主乘加阶段生成的输出和，结合旁路 scale 信息恢复为 host 可理解的数值域，并完成输出收尾。由以下模块组成：

- `Scale Metadata Buffer`  
  `缩放元数据缓冲`
- `Restore / Normalize Engine`  
  `恢复与归一化引擎`
- `Pole Fix / Output Post`  
  `极点修正与输出后处理`
- `Host Float Output`  
  `主机浮点输出`

`Scale Metadata Buffer` 存放的不是参与同拍乘法的载荷，而是与当前输出块配套的 `scale_t / exp_t / block metadata`。这些信息在 `Restore / Normalize Engine` 中与 `osum_t` 结合，统一完成数值域恢复。

随后，`Pole Fix / Output Post` 继续承接当前实现中的极点修正、输出后处理、收尾平滑等逻辑。最后，数据才跨出硬件内部定点或 block-floating 域，重新回到 host 可见的 `float` 输出边界。

## 4. Code Mapping

虽然这份描述采用的是新的 DSP-aware 架构语言，但它仍然可以一一映射回当前代码路径：

- `Host Float Input -> Input Quant Loader`  
  对应当前的 `float host input -> stage_input_frame_quantized`
- `Pad / Reorder Engine`  
  对应当前的 `pad_ico_quantized`
- `Kernel Expansion Index + Tile-Local Weight Unpack + 3x3 Weight Payload Buffer`  
  对应当前的 `kernel expansion`
- `Narrow Multiply PE + Local Product / Psum + RI Partial Merge`  
  对应当前的 `3x3 MAC -> ri_partial`
- `Output Tile Buffer`  
  对应当前的 `output_tile` 架构边界
- `Pole Fix / Output Post`  
  对应当前的 `output_post` 架构边界
- `Host Float Output`  
  对应当前的 `float host output`

因此，这不是一张脱离现有实现的“新概念图”，而是一张把当前 `layer2-5` 数据流重新解释成 `DSP48E1` 导向结构的架构图文字说明。

## 5. Design Emphasis

如果把这张图的设计重点浓缩成几句话，可以写成下面这样：

- 第一，系统被明确拆成 `Value Path` 和 `Scale Path` 两条并行通路
- 第二，真正进入主乘法器的是 `feat_t` 和 `wgt_t`，而不是扩宽后的通用累加类型
- 第三，`DSP48E1` 只承担热点主乘加，复杂量化控制和 scale 管理尽量放到外围
- 第四，输出恢复只在归约完成后统一发生一次，不在每拍乘法后恢复
- 第五，当前阶段只讨论 `DSP-aware fixed / block-floating hybrid` 主线，不把 packed FP16 DSP trick 混入 HLS 主路径

## 6. Suggested Figure Caption

如果你后面要把它配成论文里的图注，可以直接用下面这段：

`Figure: DSP48E1-aware architecture description for the shared layer2-5 ConvIco block. The design separates the main value path from the scale-metadata path, keeps the hot MAC loop as narrow-payload multiply-and-accumulate, and postpones scale restoration until the output normalization stage.`

对应中文也可以写成：

`图：面向 DSP48E1 的 layer2-5 共享 ConvIco 卷积块架构描述。该设计将主数值通路与缩放元数据通路显式分离，使热点 MAC 路径仅承担窄载荷乘加，并将数值恢复延后到输出归一化阶段统一完成。`
