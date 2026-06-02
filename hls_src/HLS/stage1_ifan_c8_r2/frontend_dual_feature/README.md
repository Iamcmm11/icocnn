# 双特征前端切片

ZYNQ 分工转向后的当前活跃 Stage-1 切片。

边界：

- 输入：PHAT/LMS maps，`[2, T, 5, 4, 8]`
- 输出：PoolIco 之前的 fused R2 feature，`[T, 8, 6, 5, 4, 8]`

本切片只包含：

1. PHAT branch stem + residual enhancement
2. LMS branch stem + residual enhancement
3. shared attention
4. branch-local fusion
5. PHAT/LMS feature addition

重复 R1 fusion-head ConvIco 主干不在这里复制，后续继续从 `../layer2-5` 硬件线推进。
