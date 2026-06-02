# FeatureMABA 切片

从原 Stage-1 根目录中独立出来的 pre-readout FeatureMABA refiner。

输入和输出均为 pre-readout feature tensor：

```text
[T, 8, 6, 5, 2, 4]
```

该切片后续可作为独立 PL 创新候选，与双特征前端、`layer2-5` ConvIco 主干分开评估资源。
