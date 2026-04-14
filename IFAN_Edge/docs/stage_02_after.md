# 阶段 02 完成后

## 当前结论

阶段 2 当前口径固定为：

“`PHAT + LMS` 主线下的工程验证已完成，尚未进入正式训练验收。”

## 已完成内容

- 已实现 `IFANModelConfig` 与 `IFANModel` 的阶段 2 最小骨架。
- 已保留双分支输入结构：
  - `phat_in_channels = 1`
  - `aux_in_channels = 1`
- 已实现共享注意力融合与回归头。
- 已保留唯一默认配置：
  - `configs/stage2_default.toml`
- 已保留唯一主线工程验证脚本：
  - `scripts/check_stage2_forward.py`

## 当前工程验证范围

- shape 检查
- forward 检查
- backward 检查
- 一条 `Windowing -> DualFeatureIcoPreprocessor -> IFANModel` 的真实工程链路检查

## 当前验证口径

- 阶段 2 只验证 `PHAT + LMS` 主线。
- 输入张量口径固定为：
  - `Input shape: (1, 2, 3, 5, 4, 8)`
  - `PHAT shape: (1, 1, 3, 5, 4, 8)`
  - `LMS shape: (1, 1, 3, 5, 4, 8)`
  - `Output shape: (1, 3, 3)`
- dummy loss 可完成反向传播。
- 输出、loss 与梯度均要求为有限值，且至少一组模型参数梯度非空。

## 当前未做事项

- 未启动正式训练。
- 未进入训练验收。
- 未做完整消融或论文级对齐确认。

## 备注

- 阶段 2 当前只保留 `PHAT + LMS` 主线，不保留备选前端。
- 当前“完成”仅表示工程验证完成，不代表训练结果已经完成。
