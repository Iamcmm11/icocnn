# DCASE Stage 2.5：Tail / Edge Robustness 与 DCASE-Conditioned 仿真计划

## 总目标

在 Stage 2 已经证明 `azimuth-only / folded-azimuth` 输出头有效的基础上，继续推进两条互补路线：

- 真实 DCASE 路线：贴合 DCASE stereo / folded-azimuth 口径，压低真实 devtest 上的尾部大错。
- 仿真验证路线：使用 DCASE metadata 分布构造可控声学仿真数据，验证 IFAN 结构在不同阵列和输出口径下的有效性。

这两条路线都允许使用合理的数据处理、增强和仿真手段，但结论必须分开表述：

- 真实 DCASE 结果只能来自 `locata_like_devtest_strict` 真音频评估。
- 仿真结果只能称为 `DCASE-conditioned synthetic benchmark`，不能等同于官方 DCASE leaderboard 结果。

## 当前依据

Stage 2 30 轮结果已经达到：

- overall `DOA error (deg) = 29.3776`
- static single-source `29.3799`
- moving single-source `29.2652`

相对 Stage 1 best 80 epoch：

- overall: `37.4179 -> 29.3776`
- static: `37.5780 -> 29.3799`
- moving: `29.8176 -> 29.2652`

但误差分布显示 Stage 2 仍有明显尾部问题：

- median 已降到约 `18.49 deg`
- p95 仍约 `100.21 deg`
- `>=90 deg` 大错主要集中在 `dev-test-tau` 的静态样本
- 弱区主要是 folded azimuth 边缘方位，尤其 `|azimuth| >= 60 deg`

因此，下一步不以简单加长 epoch 为主，而优先治理边缘方位和尾部大错。

## 坐标与口径原则

### 内部 IFAN / LOCATA / 仿真口径

主线内部使用球坐标 `(theta, phi)`：

- `theta = acos(z / r)`，从 `+Z` 轴向下量的极角
- `phi = atan2(y, x)`，XY 平面中从 `+X` 转向 `+Y`

内部方向示例：

- `phi = 0 deg`：`+X`
- `phi = 90 deg`：`+Y`，阵列前方
- `phi = -90 deg`：`-Y`，阵列后方
- `theta = 90 deg`：水平面

LOCATA 和主线仿真都使用这套完整 3D 表示。

### DCASE 口径

DCASE azimuth 的读法是：

- `0 deg`：前方，对应内部 `+Y`
- `+90 deg`：右侧，对应内部 `+X`
- `-90 deg`：左侧，对应内部 `-X`

代码中转换为：

```text
phi_internal = 90 deg - dcase_azimuth
```

DCASE2025 当前数据只提供 azimuth 和 distance，并且 azimuth 被折叠到前方视野。Stage 2 / Stage 2.5 的真实 DCASE 路线继续固定：

```text
theta = 90 deg
target = folded azimuth sin/cos
```

## Stage 2.5A：真实 DCASE Tail / Edge Robustness

### 目标

在不改变真实 DCASE 评估口径的前提下，降低 Stage 2 的尾部大错，重点改善：

- `p90` / `p95`
- `>=60 deg` 和 `>=90 deg` 大错 clip 数
- `|folded azimuth| >= 60 deg` 边缘方位分桶
- overall `DOA error (deg)` 不回退

### 初始化

继续从 Stage 2 best checkpoint 出发：

```text
IFAN_Edge/outputs/dcase_stage3/dcase_stereo_ifan_azimuth_only_stage2_azimuth_only_run30_bg_20260614_172251/checkpoints/best_doa_error.pt
```

建议第一轮使用短 fine-tune：

- epochs: `10` 或 `15`
- lr: `5e-6`，必要时对照 `1e-5`
- batch / micro-batch 沿用 Stage 2
- checkpoint 仍按 validation `DOA error (deg)` 选择

### 改动 1：Channel-Swap 数据增强

对 stereo 输入做左右声道交换，同时同步翻转标签：

```text
audio[:, left, :] <-> audio[:, right, :]
folded_azimuth -> -folded_azimuth
sin/cos target 对应重新计算
```

目的：

- 增强左右对称性
- 缓解边缘方位的符号翻转大错
- 不引入额外外部数据

第一版建议：

- 训练时以 `p=0.5` 随机启用
- 验证和测试绝不启用
- 在 summary 中记录 `channel_swap_augmentation = true`

### 改动 2：边缘方位加权

对 active window 的 loss 加权：

```text
base weight = 1.0
if |folded_azimuth| >= 60 deg: weight = edge_weight
```

第一组对照：

- `edge_weight = 1.5`
- `edge_weight = 2.0`

不建议第一轮超过 `2.0`，避免模型牺牲中心方位。

### 改动 3：尾部诊断指标

评估报告除现有 mean / median / std / min / max 外，新增：

- `p75`
- `p90`
- `p95`
- `count_ge_45_deg`
- `count_ge_60_deg`
- `count_ge_90_deg`
- by-angle-bin：
  - `[-90,-60)`
  - `[-60,-30)`
  - `[-30,0)`
  - `[0,30)`
  - `[30,60)`
  - `[60,90]`
- by-room / by-mix 诊断表，仅作为 devtest 分析，不作为训练选择依据

这样可以避免只看 overall mean，把尾部翻转问题藏起来。

### 实验矩阵

第一轮建议只做小矩阵：

| Run | Init | LR | Epochs | Channel Swap | Edge Weight | 目的 |
| --- | --- | ---: | ---: | --- | ---: | --- |
| 2.5A-baseline-ft | Stage 2 best | `5e-6` | `10` | off | `1.0` | 检查低 LR fine-tune 是否有自然收益 |
| 2.5A-swap | Stage 2 best | `5e-6` | `10` | on | `1.0` | 验证左右增强 |
| 2.5A-edge15 | Stage 2 best | `5e-6` | `10` | off | `1.5` | 验证边缘加权 |
| 2.5A-swap-edge15 | Stage 2 best | `5e-6` | `10` | on | `1.5` | 第一候选组合 |
| 2.5A-swap-edge20 | Stage 2 best | `5e-6` | `10` | on | `2.0` | 检查更强边缘权重 |

### 验收标准

最低验收：

- 训练和评估闭环跑通
- overall `DOA error (deg)` 不显著差于 `29.3776`
- 新增 tail 指标完整输出

目标验收：

- overall `DOA error (deg) < 29.3776`
- p95 低于 Stage 2
- `>=90 deg` 大错 clip 数下降
- `|azimuth| >= 60 deg` 两个边缘分桶至少一个明显改善，另一个不明显恶化

守门条件：

- 如果 overall 下降但 p95 / `>=90 deg` 明显恶化，不作为主线候选。
- 如果 edge bin 改善但中心 `[0,30)` / `[-30,0)` 明显恶化，需要降低 edge weight。

## Stage 2.5B：DCASE-Conditioned Stereo 仿真增强

### 目标

使用 DCASE metadata 的方位、距离、active duration 和 subset 分布，生成额外 stereo 仿真训练数据，验证仿真增强是否能提升真实 DCASE devtest。

### 数据原则

使用：

- DCASE train split metadata 分布
- clean speech source，例如 LibriSpeech
- stereo proxy array 或可控双耳阵列

不使用：

- devtest metadata 做训练条件
- DCASE devtest 音频做训练源
- 已经带空间混响的 DCASE stereo 音频作为 dry source 再卷积

### 仿真标签

对每个 synthetic clip：

```text
dcase_azimuth -> folded azimuth
phi_internal = 90 deg - folded_azimuth
theta = 90 deg
distance = metadata distance 或采样后的近似距离
```

对 static 样本：

- 方位和距离保持近似恒定

对 moving 样本：

- 从 metadata 插值方位 / 距离轨迹
- 若只有低频 frame 标签，则按当前窗口中心插值

### 训练策略

第一版不替代真实 DCASE 训练，只作为混合增强：

- real strict train: 保持全量
- synthetic train: 按比例混入
- synthetic:real 先试 `1:1`，再试 `2:1`
- validation 仍只用 real validation
- final test 仍只用 real devtest

### 验收标准

最低验收：

- synthetic 数据生成可复现
- 训练配置记录 synthetic 数据比例和生成参数
- real validation 可正常选择 checkpoint

目标验收：

- 真实 devtest overall 优于 Stage 2
- p95 / `>=90 deg` 大错减少
- 合成数据不导致真实 static 大样本退化

## Stage 2.5C：DCASE-Conditioned Benchmark2 仿真验证

### 目标

把 DCASE 的方位 / 距离 / 动静分布渲染到 LOCATA `benchmark2` 12 通道阵列上，用来验证主线 3D IFAN 结构在 DCASE-like 分布下是否有效。

这条路线回答的问题不是“DCASE stereo 分数能不能更高”，而是：

```text
在 DCASE-like 方位和距离分布下，如果阵列几何和训练目标回到 IFAN 主线熟悉的 benchmark2 / 3D DOA，模型是否仍然有效？
```

### 数据构造

使用：

- DCASE train metadata 分布
- clean speech source
- `benchmark2_array_setup`
- gpuRIR / 现有 RandomTrajectoryDataset 风格仿真
- 完整 3D target：`theta, phi -> 3D unit vector`

第一版 elevation 可固定为水平面：

```text
theta = 90 deg
phi = 90 deg - folded_azimuth
```

后续如需更贴近真实 3D，可加入小幅 elevation 扰动，但必须单独标记为 synthetic assumption。

### 对照对象

- 主线 frozen IFAN checkpoint zero-shot
- 在 DCASE-conditioned benchmark2 synthetic 上 fine-tune 后的 checkpoint
- Stage 2 azimuth-only 结果只作为任务口径参考，不直接比较绝对数值

### 输出结论边界

可以说：

- IFAN 主体结构在 DCASE-like 分布下有效
- benchmark2 阵列下模型能学习该方位分布
- synthetic 训练是否改善同分布 synthetic test

不能说：

- 这等价于真实 DCASE stereo 提升
- 这等价于官方 DCASE leaderboard 指标

## Stage 2.5D：Paired Synthetic 一致性验证（可选）

在同一个 synthetic trajectory 上同时渲染：

- stereo proxy 音频
- benchmark2 12 通道音频

并训练 / 评估：

- stereo azimuth-only head
- benchmark2 3D head

观察两者在同一轨迹上的预测是否一致：

- folded azimuth MAE
- internal `phi` error
- 边缘方位翻转率

这是结构诊断实验，不作为第一优先级。

## 推荐执行顺序

### 第一步：实现 Stage 2.5A

优先级最高，因为它直接服务真实 DCASE 结果：

1. 新增 tail 指标报告。
2. 新增 channel-swap 训练增强。
3. 新增 edge-weighted sin/cos loss。
4. 跑 `swap + edge_weight=1.5` 的 10 epoch 短训。
5. 与 Stage 2 best 在 real devtest 上比较。

### 第二步：实现 Stage 2.5B

如果 Stage 2.5A 仍有明显 tail 问题：

1. 生成 DCASE-conditioned stereo synthetic train。
2. real + synthetic 混合训练。
3. validation 只看 real validation。
4. devtest 只看 real devtest。

### 第三步：实现 Stage 2.5C

用于论文 / 报告中证明模型结构有效性：

1. 生成 DCASE-conditioned benchmark2 synthetic。
2. 用主线 3D IFAN 训练 / fine-tune。
3. 输出 synthetic benchmark 结果。
4. 与真实 DCASE 结果分开表述。

## 产物命名建议

配置文件：

- `IFAN_Edge/configs/dcase_stereo_azimuth_only_stage2_5_tail_edge.toml`
- `IFAN_Edge/configs/dcase_stereo_azimuth_only_stage2_5_synthetic_mix.toml`
- `IFAN_Edge/configs/dcase_conditioned_benchmark2_synthetic_ifan.toml`

脚本：

- 真实 DCASE tail 训练可继续复用：
  `IFAN_Edge/scripts/train_dcase_stereo_ifan_azimuth_only.py`
- synthetic 生成建议新增：
  `IFAN_Edge/scripts/generate_dcase_conditioned_synthetic.py`
- benchmark2 synthetic 训练如与主线差异明显，新增：
  `IFAN_Edge/scripts/train_dcase_conditioned_benchmark2_ifan.py`

输出后缀：

- `stage2_5_swap_edge15_ft10`
- `stage2_5_swap_edge20_ft10`
- `stage2_5_synthmix_1x`
- `dcase_conditioned_benchmark2_synth`

报告：

- 真实 DCASE 报告继续放：
  `IFAN_Edge/outputs/stage3/analysis`
- synthetic benchmark 报告建议单独标记：
  `IFAN_Edge/outputs/stage3/analysis/synthetic`

## 风险与应对

### 风险 1：真实 mean 下降但尾部恶化

应对：

- 把 p95 / `>=90 deg` 设为守门指标
- 不只按 overall mean 选主线

### 风险 2：edge weighting 牺牲中心方位

应对：

- 从 `1.5` 开始
- 检查 by-angle-bin
- 必要时改成平滑权重，而不是硬阈值

### 风险 3：synthetic-real gap

应对：

- synthetic 只作为增强，不替代 real validation
- final test 始终用 real devtest
- 报告中明确 synthetic assumption

### 风险 4：结论混淆

应对：

- 真实 DCASE 结果和 DCASE-conditioned synthetic benchmark 分表汇报
- synthetic 实验标题和 summary 中必须包含 `synthetic`
- 不把 synthetic 指标写成官方 DCASE 指标

## 阶段完成定义

Stage 2.5 完成时至少应产出：

- 一个真实 DCASE Stage 2.5 checkpoint
- 一份带 tail 指标的真实 devtest 报告
- 一份 Stage 2 vs Stage 2.5 对照表
- 若进入仿真路线，则额外产出 synthetic 数据生成配置和 synthetic benchmark 报告

最终推荐主线的判断标准：

- 若 Stage 2.5A 已降低真实 devtest tail 且 overall 不退化，先把它作为 DCASE 默认路线。
- 若 Stage 2.5B 能进一步提升真实 devtest，再把 synthetic mix 纳入后续训练策略。
- Stage 2.5C / 2.5D 作为模型有效性证据，不替代真实 DCASE 结果。
