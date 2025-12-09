# Layer0 IcoConv HLS验证步骤说明

## 重要发现：icoCNN的旋转等变性实现

icoCNN的输出包含**6个旋转方向**，但这并非6次独立卷积，而是通过**旋转等变性**实现的：

- 权重: `[32, 1, 1, 7]` - 只有7个权重值
- 通过 `kernel_expansion_idx` 将7个权重扩展成 6个旋转方向的 3×3 卷积核
- 每个旋转方向的卷积核是同一组权重的不同旋转映射

## 验证方案

### 方案A：简化验证（推荐用于快速测试）

**目标**：验证基础卷积逻辑的正确性
**实现**：只验证第一个旋转方向(r_out=0)的输出

**优点**：
- 实现简单，不需要 `kernel_expansion_idx` 和 `reorder_idx`
- 可以快速验证邻居关系和卷积计算是否正确
- 适合初期调试

**缺点**：
- 无法验证完整的旋转等变性
- 与真实FPGA实现有差距

### 方案B：完整验证（推荐用于最终部署）

**目标**：完整实现icoCNN的Layer0，包括旋转等变性
**实现**：使用 `hls_src/` 目录中已有的完整实现

**文件结构**：
```
hls_src/
├── ico_conv_layer0.hpp  - 完整的头文件定义
├── ico_conv_layer0.cpp  - 包含padding, conv2d, kernel expansion等
├── test_ico_conv.cpp    - 完整的测试程序
└── ...
```

**优点**：
- 完全匹配PyTorch的实现
- 包含所有优化(padding, reshape, 等)
- 可以直接用于HLS综合

**缺点**：
- 需要额外提取 `kernel_expansion_idx` 和 `reorder_idx` 数据
- 实现更复杂，调试难度更高

## 当前数据文件状态

你已经成功生成了以下数据：

```
hls_testdata/layer0/
├── input.npy / input.txt       - [1, 103, 1, 1, 5, 4, 8] → reshape to [1, 1, 1, 103, 160]
├── weight.npy / weight.txt     - [32, 1, 7]
├── bias.npy / bias.txt         - [32]
├── neighbors.npy / neighbors.txt - [160, 7]
└── output.npy / output.txt     - [1, 103, 32, 6, 5, 4, 8] → reshape to [1, 32, 6, 103, 160]
```

## 下一步建议

### 如果选择方案A（简化验证）：

1. **修改 `hls_implementation/` 代码**，让它只处理第一个旋转方向
   - 将 `OUT_ROTATIONS` 改为 1
   - 只验证 `output[:, :, 0, :, :]` 这部分

2. **修改Python数据保存代码**，只保存r_out=0的输出
   ```python
   # 在 move_layer0_io_data() 中添加
   output_data = np.load('hls_testdata/layer0/output.npy')
   output_r0 = output_data[:, :, :, 0, :, :, :]  # 只保留第一个旋转方向
   output_r0_reshaped = output_r0.transpose(0, 2, 1, 3, 4, 5).reshape(1, 32, 1, 103, 160)
   ```

3. **编译并运行C++验证**
   ```cmd
   cd hls_implementation/src
   make clean
   make
   make run
   ```

4. **检查误差**
   - 最大绝对误差应 < 1e-4
   - 如果误差大，检查邻居索引是否正确

### 如果选择方案B（完整验证）：

1. **使用 `hls_src/` 中的实现**（已经包含完整功能）

2. **提取额外的查找表数据**：
   - `kernel_expansion_idx` - [32, 6, 1, 1, 9, 4]
   - `reorder_idx` - [1, 5, 6, 10]

3. **修改 `inference_debug.py`** 添加这些查找表的提取

4. **编译并测试**
   ```cmd
   cd hls_src
   make clean
   make
   ./test_ico_conv
   ```

## 资源估算（完整版）

基于实际参数：
- **DSP48E**：约 840个（32输出通道 × 6方向 × 并行度）
- **BRAM**：
  - 输入缓存: 66KB (103×1×160×float32)
  - 权重: <1KB (32×1×7×float32)
  - 中间结果: ~300KB (padding后的数据)
  - 输出缓存: 12.6MB → **需要流式处理或分块**
- **LUT/FF**：控制逻辑和地址生成
- **计算量**：22M MAC操作

## Kintex-7 xc7k325t 资源
- DSP48E: 840 (100%利用率)
- BRAM 36Kb: 445块 (~2MB total)
- LUT: 203,800
- FF: 407,600

**关键瓶颈**: 输出BRAM不够，必须使用流式处理！

## 建议的验证流程

1. **第一阶段**：用方案A验证基本算法（当前阶段）
   - 验证邻居索引正确性
   - 验证卷积计算逻辑
   - 确认数据格式转换正确

2. **第二阶段**：用方案B验证完整实现
   - 添加padding和kernel expansion
   - 验证旋转等变性
   - 确认与PyTorch完全一致

3. **第三阶段**：HLS优化
   - 添加PIPELINE和DATAFLOW指令
   - 实现流式处理降低BRAM需求
   - 综合并查看资源使用

4. **第四阶段**：FPGA部署
   - 生成IP核
   - 集成到系统中
   - 板级验证

## 当前状态

你已经完成：
- ✅ 生成了Layer0的所有基础数据文件
- ✅ 创建了HLS C++验证框架
- ✅ 理解了数据结构和旋转等变性原理

下一步：
- ⏳ 决定使用简化验证还是完整验证
- ⏳ 编译并运行C++测试
- ⏳ 对比输出并修复误差

## 注意事项

1. **数据格式**：Python保存的数据需要reshape成C++期待的格式
2. **旋转方向**：简化版只验证r_out=0，完整版需要所有6个方向
3. **BRAM限制**：输出12.6MB远超2MB，必须流式处理
4. **邻居索引**：确保使用真实的六边形拓扑，不能简化

如有任何问题，请随时询问！
