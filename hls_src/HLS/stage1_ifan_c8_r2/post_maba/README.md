# Post-MABA 切片

本目录保存 MABA 后处理与坐标头：

1. channel readout
2. optional final-pool copy boundary
3. region max over `R`
4. CleanVertices mask
5. SoftArgMax coordinates

该切片暂不并入默认 HLS top，后续根据 MABA 资源闭合结果决定放在 PS 还是 PL。
