# DCASE Stage 2：Azimuth-Only / Folded-Azimuth 输出头计划

## 目标

在已验证的 `stereo frontend + folded-azimuth 训练目标` 基础上，将当前“3D 坐标输出头”改成真正的 `azimuth-only / folded-azimuth` 输出头，减少输出空间冗余，并重点改善：

- 后期收敛效率
- moving 子集建模能力
- `DOA error (deg)` 的最终下限

这一步继续保持 DCASE 独立分支，不影响冻结主线 `ifan_c8_r2_maba_pre_readout_best`。

## 为什么进入 Stage 2

Stage 1 已经证明：

- stereo frontend 对齐有效
- folded-azimuth 目标对齐有效
- `DOA error (deg)` 已从 zero-shot `44.9631` 降到 `37.4179`

但 Stage 1 同时暴露出两个问题：

- 后期继续加轮次仍有收益，但已明显进入边际递减区间
- `moving_single_source` 在 `10 -> 80` 轮之间没有继续变好，说明主要瓶颈可能不在训练轮次，而在输出建模方式

## Stage 2 核心改动

### 1. 输出头从 3D 坐标改为真正的方位角建模

当前方案：

- 模型输出 3D 坐标
- 训练目标是水平面 folded-azimuth 对应的 3D 单位向量

Stage 2 改成：

- 模型直接输出 folded azimuth
- 输出定义域固定在 DCASE folded-azimuth 口径

第一版建议采用：

- 连续回归头
- 直接预测一个标量方位角

不建议第一步就做角度分类头，因为：

- 回归头改动更小
- 更容易和当前 `DOA error (deg)` 指标直接对齐
- 更适合做与 Stage 1 的一变量对照

### 2. 目标与损失同步切换

Stage 2 训练目标：

- 直接使用 folded azimuth 标量

损失建议：

- 主损失使用 circular angle regression loss
- 不能直接对角度做普通 MSE，因为角度有周期性

推荐实现：

- 模型输出 `sin(phi)` 与 `cos(phi)` 两个值
- 训练目标也转成 `(sin, cos)`
- 用 2 维 MSE 训练
- 推理时再反解回 folded azimuth

这样做的优点：

- 保留角度周期性
- 训练稳定
- 改动比保留完整 3D 坐标更聚焦
- 不需要自己额外处理 `-90/90` 或 `-180/180` 附近的数值跳变

### 3. 前端和数据切分保持不变

Stage 2 不改以下内容：

- stereo proxy frontend
- `strict` speech-only 单声源训练集
- `dev-train-*` 训练 / 验证，`devtest_strict` 最终测试
- `K=4096`，`step=3072`
- `exclude_initial_windows = 5`

这样可以保证：

- Stage 1 与 Stage 2 的收益可直接比较
- 改善来源主要归因于输出层，而不是数据口径变化

### 4. 评估指标保持不变

Stage 2 仍然使用：

- 主指标：`DOA error (deg)`
- 辅助指标：`Folded Az RMSE`

不额外引入新的主指标，避免和 Stage 1 不可比。

## 代码层面改动范围

### 新增或修改的模块

- 新增 DCASE 专用 azimuth-only 模型封装
  - 可以在 `IFAN_Edge/ifan_edge/models/` 下新增轻量模块
  - 或在现有 DCASE 独立分支里新增一个包装器，复用 IFAN 主干并替换最后输出层

- 新增 DCASE azimuth-only 训练逻辑
  - 基于现有 `IFAN_Edge/ifan_edge/training/dcase_stereo.py`
  - 但只替换目标构造、损失和输出解析部分

- 新增独立配置文件
  - 建议命名：
    `IFAN_Edge/configs/dcase_stereo_azimuth_only_c8_r2_maba_pre_readout.toml`

- 新增独立训练脚本
  - 建议命名：
    `IFAN_Edge/scripts/train_dcase_stereo_ifan_azimuth_only.py`

### 不改的部分

- 不改冻结主线 Stage-3 代码路径
- 不删除 Stage 1 分支
- 不覆盖现有 `dcase_stereo_folded_azimuth_*` 训练结果

## 初始化策略

Stage 2 继续从 Stage 1 的稳定权重出发。

推荐初始化来源：

- `IFAN_Edge/outputs/dcase_stage3/dcase_stereo_ifan_run80_bg_20260609_131300/checkpoints/best_doa_error.pt`

原因：

- 它已经完成了 stereo frontend 和 folded-azimuth 口径对齐
- 比直接从冻结的 LOCATA 主线 checkpoint 起步更接近 Stage 2 任务

注意：

- 只加载可兼容部分
- 新输出头参数从头初始化

## 训练策略

第一轮 Stage 2 实验建议：

- 训练轮次：`30`
- 设备：`cuda`
- 其余 batch / lr 可先沿用 Stage 1 默认值

推荐比较对象：

- Stage 1 best 80 epoch：`DOA error = 37.4179`
- Stage 2 azimuth-only 30 epoch

如果 Stage 2 的 `30` 轮就能接近或超过 `37.4179`，说明输出头方向是对的。

## 验收标准

### 最低验收

- 训练闭环跑通
- 独立评估脚本能输出 Stage 2 checkpoint 的 DCASE 报告
- `DOA error (deg)` 不差于 Stage 1 早期结果

### 目标验收

- 整体 `DOA error (deg) < 37.4179`

### 附加观察点

- `moving_single_source` 是否优于 Stage 1 的 `29.8176`
- validation 曲线是否比 Stage 1 更快收敛

## 风险与应对

### 风险 1：回归头不稳定

应对：

- 不直接回归角度值
- 采用 `sin/cos` 双输出

### 风险 2：moving 样本过少，指标仍不稳定

应对：

- Stage 2 第一轮先不改数据口径
- 若 moving 仍无改善，再进入下一步做 subset 重加权或过采样

### 风险 3：与 Stage 1 不可比

应对：

- 固定前端
- 固定数据切分
- 固定评估指标
- 只改输出头和损失

## Stage 2 完成后的分叉

如果 Stage 2 成功：

- 把 `azimuth-only` 头定为 DCASE 主线
- 再考虑加轮次或做样本重加权

如果 Stage 2 效果不明显：

- 再考虑 moving 子集重加权 / 过采样
- 或回头验证是否需要更改前端表示方式，而不是只改输出头

## Stage 2 后续计划

Stage 2 30 轮结果已经明显超过目标验收后，后续不再优先围绕 3D 输出头或单纯长轮次推进。

下一阶段进入：

- `IFAN_Edge/docs/DCASE/Stage2_5_tail_edge_and_simulation计划.md`

核心方向：

- 在真实 DCASE stereo / folded-azimuth 口径下治理 tail / edge 大错
- 引入 channel-swap、边缘方位加权和 tail 诊断指标
- 另开 DCASE-conditioned synthetic benchmark，用仿真验证 IFAN 结构有效性
- 真实 DCASE 结果与 synthetic benchmark 结果分开汇报
