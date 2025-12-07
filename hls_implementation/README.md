# Layer0 IcoConv C++ 验证框架

本目录包含 IcoTempCNN 第一层（IcoConv）的 **纯 C++ 实现**，用于在 Vivado HLS 综合之前验证算法逻辑的正确性。

## 🎯 验证流程说明

```
Step 1: 纯 C++ 顺序执行验证 (当前阶段)
  ├─ 用标准 C++ 编译器 (g++/MSVC) 编译
  ├─ 不添加任何 HLS pragma 指令
  ├─ 顺序执行，确保算法逻辑与 PyTorch 完全一致
  └─ 误差验证: max_error < 1e-4

Step 2: Vivado HLS 综合 (下一阶段)
  ├─ 导入验证通过的 C++ 代码
  ├─ 添加 #pragma HLS 优化指令
  ├─ C 仿真 → C/RTL 联合仿真
  └─ 生成 IP 核

Step 3: FPGA 板级部署 (最终阶段)
  └─ 集成到完整系统
```

## 目录结构

```
hls_implementation/
├── include/              # 头文件
│   ├── ico_types.h       # 类型定义和配置
│   ├── ico_conv_layer0.h # Layer0 接口
│   └── utils.h           # 工具函数
├── src/                  # 源文件
│   ├── ico_conv_layer0.cpp  # Layer0 实现
│   ├── utils.cpp            # 工具函数实现
│   ├── test_ico_conv.cpp    # 测试主程序
│   └── Makefile             # 编译脚本
├── build/                # 编译中间文件 (自动生成)
├── bin/                  # 可执行文件 (自动生成)
└── README.md             # 本文档
```

## Layer0 架构

### 输入/输出
- **输入**: `[batch, in_channels, time_frames, num_vertices]`
  - batch = 1
  - in_channels = 4
  - time_frames = 10
  - num_vertices = 42 (icosahedral r=2)

- **输出**: `[batch, out_channels, time_frames, num_vertices]`
  - out_channels = 32

### 关键参数
- **邻居数**: 7 (中心顶点 + 6个邻居)
- **权重**: `[32, 4, 7]` = 896 参数
- **偏置**: `[32]` = 32 参数

### 计算公式
```
对于每个顶点 v:
  output[b][out_c][t][v] = ReLU(
      sum_{in_c, neighbor_n} (
          input[b][in_c][t][neighbors[v][n]] * weight[out_c][in_c][n]
      ) + bias[out_c]
  )
```

## 使用步骤

### 前置准备: 准备测试数据

从服务器导出并上传以下数据文件到 `../hls_testdata/layer0/` 目录:

- **必须文件**:
  - `input.txt` - 输入特征 [1, 4, 10, 42] = 1,680 个值
  - `weight.txt` - 卷积权重 [32, 4, 7] = 896 个值
  - `bias.txt` - 偏置 [32] = 32 个值
  - `neighbors.txt` - 邻居索引 [42, 7] = 294 个整数
  - `output.txt` - PyTorch 参考输出 [1, 32, 10, 42] = 13,440 个值

详细数据格式请参考: [数据文件说明.md](./数据文件说明.md)

### 1. 编译 C++ 代码

```bash
cd hls_implementation/src
make clean
make
```

### 2. 运行验证

```bash
make run
```

或者直接运行:

```bash
cd hls_implementation/bin
./test_ico_conv
```

### 3. 查看结果

程序会输出:
- 数据加载状态
- 输入/输出统计信息
- **最大绝对误差**
- **相对误差 (RMSE)**
- 测试通过/失败状态

## 验证标准

- **误差阈值**: `1e-4`
- 如果 `max_error < 1e-4`,测试通过 ✓
- 否则显示不匹配的值并保存 `output_hls.txt`

## HLS 优化要点

当前实现是**功能验证版本**,后续 HLS 综合时可优化:

### 1. 循环流水线
```cpp
#pragma HLS PIPELINE II=1
```

### 2. 数据缓存
```cpp
#pragma HLS ARRAY_PARTITION variable=weight_cache cyclic factor=8
```

### 3. 并行计算
```cpp
#pragma HLS UNROLL factor=4  // 部分展开
```

### 4. 接口优化
```cpp
#pragma HLS INTERFACE m_axi port=input  bundle=gmem0
#pragma HLS INTERFACE m_axi port=output bundle=gmem1
```

## 资源估算 (Kintex-7)

### 当前版本 (未优化)
- **LUT**: ~5K (估计)
- **BRAM**: ~10 (权重缓存)
- **DSP**: ~64 (乘法器)
- **延迟**: ~100K cycles

### 优化后 (流水线 + 并行)
- **LUT**: ~15K
- **BRAM**: ~20
- **DSP**: ~128
- **吞吐量**: ~10 cycles/vertex (目标)

你的板子资源:
- LUT: 203K ✓ (充足)
- BRAM: 445 ✓ (充足)
- DSP: 840 ✓ (充足)

## 下一步计划

1. ✓ **功能验证** (当前阶段)
2. **HLS 综合** - 使用 Vivado HLS
3. **性能优化** - 添加 pragma 指令
4. **接口设计** - AXI Stream/MM
5. **多层集成** - 后续层级

## 文件说明

### `ico_types.h`
- 基本类型定义 (`data_t`, `index_t`)
- 配置参数宏
- 张量索引宏
- ReLU 激活函数

### `ico_conv_layer0.h/cpp`
- IcoConv 核心实现
- 数据加载函数
- 卷积计算函数

### `utils.h/cpp`
- 文件 I/O (txt 格式)
- 误差计算
- 统计信息

### `test_ico_conv.cpp`
- 主测试程序
- 数据加载
- 结果验证
- 错误报告

## 常见问题

### Q: 编译时找不到 math.h?
A: 确保安装了 C++ 编译器 (MinGW/MSVC)

### Q: 测试数据不存在?
A: 先运行 `generate_layer0_data.py`

### Q: 误差很大?
A: 检查数据路径、权重加载、索引计算

## 联系

如有问题请参考:
- Python 原始实现: `acousticTrackingModels.py`
- icoCNN 源码: `icoCNN-master/`
- 调试脚本: `inference_debug.py`
