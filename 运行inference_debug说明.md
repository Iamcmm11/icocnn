# 运行 inference_debug.py 生成 HLS 验证数据

## 📝 说明

我已经修改了你的 `inference_debug.py` 脚本,现在它可以**自动生成 HLS C++ 验证所需的全部数据**。

## ✨ 新增功能

### 1. 自动提取 Layer0 权重和邻居索引
- 提取第一个 IcoConv 层的权重 `[32, 4, 7]`
- 提取偏置 `[32]`
- 提取邻居索引表 `[42, 7]`

### 2. 保存完整的输入和输出
- Layer0 的输入 (完整数据)
- Layer0 的输出 (完整数据,用于对比验证)

### 3. 自动整理到 HLS 目录
- 所有数据自动保存到 `hls_testdata/layer0/`
- 格式完全符合 HLS C++ testbench 要求

## 🚀 使用步骤

### 步骤 1: 运行 debug 脚本

```bash
python inference_debug.py
```

### 步骤 2: 检查输出

脚本运行完成后,会在最后显示:

```
============================================================
HLS Verification Files Summary:
============================================================
  ✓ input.txt          -   XX.XX KB -   XXXX lines
  ✓ weight.txt         -   XX.XX KB -    897 lines
  ✓ bias.txt           -   XX.XX KB -     33 lines
  ✓ neighbors.txt      -   XX.XX KB -    295 lines
  ✓ output.txt         -   XX.XX KB -  XXXXX lines
============================================================
```

### 步骤 3: 验证文件完整性

检查 `hls_testdata/layer0/` 目录应该包含:

- ✅ `input.txt` - Layer0 输入 (应该有完整数据)
- ✅ `weight.txt` - 卷积权重 (897 行: 1 行注释 + 896 个值)
- ✅ `bias.txt` - 偏置 (33 行: 1 行注释 + 32 个值)
- ✅ `neighbors.txt` - 邻居索引 (295 行: 1 行注释 + 294 个整数)
- ✅ `output.txt` - PyTorch 参考输出 (应该有完整数据)

## 📊 预期输出维度

根据你的模型配置 (r=2, C=32):

| 文件 | 维度 | 元素数量 | 文件行数 |
|------|------|----------|----------|
| `input.txt` | `[1, 4, T, 42]` | 168×T | 1 + 168×T |
| `weight.txt` | `[32, 4, 7]` | 896 | 897 |
| `bias.txt` | `[32]` | 32 | 33 |
| `neighbors.txt` | `[42, 7]` | 294 | 295 |
| `output.txt` | `[1, 32, T, 42]` | 1344×T | 1 + 1344×T |

**注**: T 是时间帧数,默认是 103

## ⚠️ 常见问题

### Q1: 脚本运行失败,提示找不到模型权重?

**A**: 这是正常的! 脚本会使用**随机初始化的权重**继续运行。修改这一行:
```python
MODEL_PATH = 'models/your_model_weights.bin'  # 改为你实际的权重文件路径
```

如果没有训练好的权重,不用管它,**随机权重也可以验证算法逻辑正确性**!

### Q2: 看到 "Layer0_IcoConv_input.txt not found" 警告?

**A**: 这说明 Hook 没有正确捕获 Layer0 的输入。检查:
1. 模型初始化是否正确
2. 是否有其他错误导致推理没有执行

### Q3: 输入输出的维度不对?

**A**: 检查 `R_LEVEL` 和 `CHANNELS` 配置:
```python
R_LEVEL = 2     # 应该和你训练时一致
CHANNELS = 32   # 应该和你训练时一致
```

## 🎯 下一步

数据生成成功后,就可以运行 HLS C++ 验证了:

```bash
cd hls_implementation
编译并测试.bat
```

## 📂 生成的文件结构

```
icocnn/
├── debug_outputs/           # 所有层的 debug 输出
│   ├── Layer0_IcoConv_input.npy/txt
│   ├── Layer0_IcoConv_output.npy/txt
│   ├── Layer1_...
│   └── ...
└── hls_testdata/
    └── layer0/              # HLS 验证专用数据 ✓
        ├── input.txt/npy
        ├── weight.txt/npy
        ├── bias.txt/npy
        ├── neighbors.txt/npy
        └── output.txt/npy
```

## 💡 提示

1. **随机种子已固定** (`torch.manual_seed(42)`)，每次运行结果一致
2. **数据格式**: txt 文件每行一个数值,方便 C++ 读取
3. **浮点精度**: 保留 8 位小数,足够 HLS 验证

---

**准备好了吗? 运行 `python inference_debug.py` 开始吧!** 🚀
