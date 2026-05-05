# Stage 03 Architecture Compare

## 目标

这份文档只做一件事：

- 用尽量简化的方式，把 `icoCNN baseline`、`论文 IFAN（按当前理解）`、`当前代码 IFAN` 三者按层对齐

说明：

- `论文 IFAN` 这一列，不是逐字抄论文源码，而是按你当前的结构理解、以及 Fig.6 图上的通道标注整理出来的“结构理解版”
- 当前采用的解释是：
  - `Fusion Feature` 之后的主干通道保持 `16`
  - 不把正文中的 `32 kernels` 直接解释成“特征通道升到 32”
- `当前代码 IFAN` 这一列，完全以当前源码为准

源码入口：

- 原始 `icoCNN`：[`acousticTrackingModels.py`](/home/cmm/icocnn/acousticTrackingModels.py#L19)
- 当前 `IFAN`：[`placeholders.py`](/home/cmm/icocnn/IFAN_Edge/ifan_edge/models/placeholders.py#L168)

当前对比配置：

- `icoCNN`: `r=2, C=32, Cin=1`
- `当前 IFAN`: `r=2, paper_dual_mainline, branch_channels=16, shared_attention, fusion_head=4+1, final_head_pooling=false`

---

## 1. 一眼结论

| 项目 | icoCNN baseline | 论文 IFAN（按当前理解） | 当前代码 IFAN |
| --- | --- | --- | --- |
| 输入 | 单特征 | 双特征 `PHAT + LMS` | 双特征 `PHAT + LMS` |
| 主体形式 | 单分支深时空网络 | 双分支 + residual + attention + 深 fusion head | 双分支 + residual + attention + 深 fusion head |
| 时间卷积 | 几乎每层都有 | 融合后连续多层都有 | 融合后连续多层都有 |
| 主体通道 | `32` | `16` 分支，Fusion Feature 后继续保持 `16` | `16` 分支，Fusion Feature 后继续保持 `16` |
| 融合后深度 | 深 | 深 | 深 |
| 参数量 | `290,017` | 按图示理解更像 `0.12M ~ 0.15M` | `125,440` |

最直接的判断：

- 旧版 IFAN 主干不算严格按论文复现
- 当前代码主线已经和这份文档里的“论文 IFAN（按当前理解）”主体对齐
- 当前剩余歧义主要集中在 `final_head_pooling` 位置，而不是主干主体拓扑

换句话说：

- 如果“论文 IFAN”以这份文档当前采用的解释为准，那么当前代码已经符合这版结构理解
- 当前没有看到新的主干结构偏差，剩下的是图示歧义和训练验证问题，不是新的 topology mismatch

---

## 2. 从头到尾逐层对照

这张表只保留“这一层做了什么”，不展开太多内部细枝末节。

| 层/阶段 | icoCNN baseline | 论文 IFAN（按当前理解） | 当前代码 IFAN |
| --- | --- | --- | --- |
| 0. 输入 | `SRP-PHAT` 单特征输入 | `PHAT + LMS` 双特征输入 | `PHAT + LMS` 双特征输入 |
| 1. 第一层空间卷积 | 1 次 `IcoConv`，`1 -> 32` | 两个前端各自先 1 次 `IcoConv`，`1 -> 16` | 两个前端各自先 1 次 `IcoConv`，`1 -> 16` |
| 2. 第一层后的分路 | 无 | 每个前端各自分成两路：直通特征、residual-enhanced 特征 | 有，当前代码已显式保留直通特征和 enhanced feature |
| 3. Residual Learning Module | 无单独模块，直接继续主干 | 每个前端各有 1 个 residual learning module，用来得到 enhanced feature | 有，每个前端 1 个 residual learning module |
| 4. Feature Attention Weight Module | 无 | enhanced feature 进入 attention module：`L-norm -> IcoConv -> ReLU -> IcoConv -> Sigmoid`，再和对应直通特征融合 | 有，且 PHAT/LMS 两支共享同一套 attention 权重 |
| 5. 前端内部融合 | 无 | `PHAT` 内部先融合一次，`LMS` 内部也先融合一次 | 有，当前代码已做 branch-local fusion |
| 6. 双路再次融合 | 无 | 两个前端各自得到的融合特征，再做一次融合 | 有，当前代码已做 second-stage fusion |
| 7. pooling 位置 | 网络中间按层节奏做 | 在得到 Fusion Feature 前做 pooling，输出 `B T 16 6 5 2 4` | 有，当前代码在 second-stage fusion 后做一次 `PoolIco(r=2)` |
| 8. Fusion Feature | 无单独概念 | 有，结构约 `B T 16 6 5 2 4` | 有，当前代码的 Fusion Feature 也是 `B T 16 6 5 2 4` |
| 9. Fusion head 第 1 组 | baseline 的 block 继续 `IcoConv + Conv1d + Norm + ReLU` | `IcoConv -> ReLU -> Conv1d -> L-norm -> ReLU` | 有 |
| 10. Fusion head 第 2 组 | 同上 | `IcoConv -> ReLU -> Conv1d -> L-norm -> ReLU` | 有 |
| 11. Fusion head 第 3 组 | 同上 | `IcoConv -> ReLU -> Conv1d -> L-norm -> ReLU` | 有 |
| 12. Fusion head 第 4 组 | 同上 | `IcoConv -> ReLU -> Conv1d -> L-norm -> ReLU` | 有 |
| 13. 最后输出前一组 | baseline 最后一层仍是时空耦合 block | `IcoConv -> ReLU -> Conv1d -> L-norm -> R-pooling / readout`，通道仍保持 `16` | `IcoConv -> ReLU -> Conv1d(16->16) -> LNorm -> channel readout -> R-pooling(max over R)`，并支持额外的 `final_head_pooling` 开关 |
| 14. 输出层 | `CleanVertices -> SoftArgMax` | `SoftArgMax` | `CleanVertices -> SoftArgMax` |

---

## 3. 只看“有没有”

如果你只想快速看出谁有谁没有，看这张表就够了。

| 结构项 | icoCNN baseline | 论文 IFAN（按当前理解） | 当前代码 IFAN |
| --- | --- | --- | --- |
| 单前端先做 1 次 `IcoConv` | 有 | 有 | 有 |
| 双前端输入 | 无 | 有 | 有 |
| 每个前端“直通 + residual”双路保留 | 无 | 有 | 有 |
| 每个前端内部先融合一次 | 无 | 有 | 有 |
| attention module | 无 | 有 | 有 |
| attention 后再和本前端直通特征融合 | 无 | 有 | 有 |
| 两路融合特征再次融合 | 无 | 有 | 有 |
| 融合后形成显式 `Fusion Feature` | 无 | 有 | 有 |
| 融合后连续 4 组 `IcoConv + Conv1d` | 无 | 有 | 有 |
| temporal conv 贯穿主干 | 有 | 有，Fusion Feature 后连续多层都有 | 有，Fusion Feature 后连续多层都有 |
| 主干主体通道 32 | 有 | 否，论文 IFAN 更像 16 通道分支 | 否，主体是 16 |

---

## 4. 代码里当前 IFAN 真正在干什么

如果只看当前代码，不看论文，它的 forward 可以压缩成下面这条线：

| 顺序 | 当前代码 IFAN |
| --- | --- |
| 1 | 输入 `PHAT + LMS` |
| 2 | `PHAT stem` |
| 3 | `PHAT residual learning module` |
| 4 | `PHAT attention weight module + branch-local fusion` |
| 5 | `LMS stem` |
| 6 | `LMS residual learning module` |
| 7 | `LMS attention weight module + branch-local fusion` |
| 8 | `second-stage feature fusion` |
| 9 | `PoolIco` |
| 10 | `4 x (IcoConv -> ReLU -> Conv1d -> LNorm -> ReLU)` |
| 11 | `final block: IcoConv -> ReLU -> Conv1d -> LNorm` |
| 12 | `optional final_head_pooling` |
| 13 | `channel_readout -> R-pooling(max over R) -> CleanVertices -> SoftArgMax` |

这里需要特别澄清：

1. 当前 IFAN 末端本来就已经有一层 **`R-pooling`**
   - 它对应的是对 `R=6` 个 orientation channels 做 `max`
   - 它不会降低 `icosahedral` 空间分辨率
2. `final_head_pooling` 是额外的 **icosahedral pooling**
   - 它会继续降低空间分辨率
   - 它不是 baseline 图里 `R-pooling` 的同义词

这条线和你描述的论文结构相比，目前剩下的不是新的主干偏差，而是两个显式歧义点：

1. `final_head_pooling` 的默认位置和是否开启仍是一个显式歧义点
2. 论文图和正文里的 `32 kernels` 仍需继续核实是否只代表卷积核数，而不是通道数

---

## 5. 参数量对比

| 模型 | 参数量 |
| --- | ---: |
| `icoCNN(r=2, C=32)` | `290,017` |
| `旧版 IFAN stage2/3 简化主干` | `76,433 ~ 86,657` |
| `当前重构后的论文版 IFAN` | `125,440` |

当前重构后论文版 IFAN 参数分解：

| 模块 | 参数量 |
| --- | ---: |
| `phat_stem` | `128` |
| `lms_stem` | `128` |
| `phat_residual` | `21,568` |
| `lms_residual` | `21,568` |
| `shared_attention` | `21,568` |
| `fusion_blocks` | `48,384` |
| `final_head` | `10,851` |

这说明当前 IFAN 的问题不只是“结构有偏差”，还有：

- 当前主干容量本身也显著小于 baseline

按“Fig.6 图上 Fusion Feature 后主干全程保持 16 通道”来反推，论文 IFAN 的合理参数量估计如下：

| 论文 IFAN 估算版本 | 假设 | 估算参数量 |
| --- | --- | ---: |
| `all16_shared_attention` | 两支 attention 共用一套权重，Fusion Feature 后全程 `16` 通道 | `125,440` |
| `all16_independent_attention` | 两支 attention 不共享权重，Fusion Feature 后全程 `16` 通道 | `145,793` |

所以更合理的量级是：

- `约 0.12M ~ 0.15M`

当前代码的新主线参数量 `125,440` 已经落进这个区间，而且现在和“最后一路始终保持 16 通道”的共享 attention 解释一致。

---

## 6. 最终判断

如果按你现在这段论文理解来问：

“我们当前 IFAN 的复现是正确的吗？”

更准确的回答是：

- `前端双特征方向`：大方向是对的
- `当前主干结构`：已经按当前论文理解落地

最关键的结构偏差是：

1. `final_head_pooling` 的位置仍然带有图示歧义
2. 论文图和正文中 `32 kernels` 的表述仍需继续核实是否只代表卷积核数，而不是通道数
3. 还需要训练验证这版结构是否真的比旧版结构更稳定

所以当前最稳妥的判断不是：

- “我们已经正确复现了 IFAN，只是训练不够”

而是：

- “我们当前代码主线已经符合这份文档中采用的论文 IFAN 结构解释，但仍有少量图示歧义和训练效果问题需要继续验证”

---

## 7. 这对后续意味着什么

后续如果你要决定“继续修前端，还是开始重估结构”，这张表给出的结论很直接：

- 前端现在已经证明有价值
- 当前主线结构已经完成一版重构，下一步重点应转向训练验证和剩余歧义核对

而且结构重估不该再是抽象地说“是不是不对”，而是直接围绕下面几个点：

| 优先级 | 结构点 | 原因 |
| --- | --- | --- |
| 1 | `final_head_pooling` 默认应不应该开启 | 图示与文字仍有歧义，而且它与末端已有的 `R-pooling` 不是同一种操作 |
| 2 | `32 kernels` 是否只代表卷积核数而非通道数 | 文字解释仍需继续核实 |
| 3 | 重构后新主线能否稳定优于旧版结构 | 需要训练验证 |
| 4 | LMS 速度如何在不改语义前提下继续下降 | 这是后续实验成本核心 |

如果你愿意，我下一步可以继续把这份文档再压缩成一张“结构偏差清单”，只保留：

- 论文该有
- 当前没有
- 风险高低

这样就能直接指导我们下一步到底改哪几层。  
