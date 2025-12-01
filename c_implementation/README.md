# Cross3D Preprocessing - C Implementation

## 概述

本项目实现了Cross3D声源定位系统的预处理模块，采用C语言编写，适用于：
- **Zynq PS端 (ARM)**: 数据读取和控制
- **HLS验证**: FFT、GCC-PHAT、SRP-Map算法验证
- **PC端测试**: 功能验证和性能评估

## 项目结构

```
c_implementation/
├── include/                    # 头文件
│   ├── config.h               # 系统配置参数
│   ├── types.h                # 数据类型定义
│   ├── audio_reader.h         # 音频读取模块
│   ├── fft.h                  # FFT模块
│   ├── gcc_phat.h             # GCC-PHAT模块
│   ├── srp_map.h              # SRP-Map模块
│   └── test_data.h            # 测试数据生成模块
├── src/                        # 源文件
│   ├── main.c                 # 主程序
│   ├── audio_reader.c         # 音频读取实现
│   ├── fft.c                  # FFT实现
│   ├── gcc_phat.c             # GCC-PHAT实现
│   ├── srp_map.c              # SRP-Map实现
│   └── test_data.c            # 测试数据生成实现
├── output/                     # 输出文件目录
├── Makefile                    # Linux/Mac构建文件
├── build.bat                   # Windows构建脚本
└── README.md                   # 本文件
```

## 模块说明

### 1. 配置模块 (config.h)
定义系统参数：
- 采样率: 24000 Hz
- 通道数: 12
- FFT点数: 4096
- 帧长: 4096 samples
- 帧移: 1024 samples

### 2. 音频读取模块 (audio_reader)
- 从二进制文件读取多通道音频
- 分帧处理
- 汉宁窗加窗

### 3. FFT模块 (fft)
- Cooley-Tukey基2 FFT算法
- 支持正向FFT和逆向IFFT
- 预计算旋转因子优化

### 4. GCC-PHAT模块 (gcc_phat)
- 广义互相关-相位变换
- 计算66对麦克风的时延估计
- PHAT加权归一化

### 5. SRP-Map模块 (srp_map)
- 空间功率谱投影
- Tau Table预计算
- 三维空间网格映射

### 6. 测试数据模块 (test_data)
- 生成模拟多通道麦克风信号
- 支持设置声源角度
- 添加高斯白噪声
- 二进制/文本格式保存

## 数据文件格式

### 音频数据 (audio_data.bin)
```
Header (16 bytes):
  - magic: "AUD\0" (4 bytes)
  - num_channels: int32
  - num_samples: int32
  - sample_rate: int32
Data:
  - float32[num_channels][num_samples]
```

### FFT结果 (fft_result.bin)
```
Header (16 bytes):
  - magic: "FFT\0" (4 bytes)
  - num_channels: int32
  - num_bins: int32
  - reserved: int32
Data:
  - complex_t[num_channels][num_bins]
```

### GCC结果 (gcc_result.bin)
```
Header (16 bytes):
  - magic: "GCC\0" (4 bytes)
  - num_pairs: int32
  - gcc_length: int32
  - reserved: int32
Data:
  - float32[num_pairs][gcc_length]
```

### SRP结果 (srp_result.bin)
```
Header (16 bytes):
  - magic: "SRP\0" (4 bytes)
  - elevation_bins: int32
  - azimuth_bins: int32
  - range_bins: int32
Data:
  - float32[elevation_bins][azimuth_bins][range_bins]
```

### Tau Table (tau_table.bin)
```
Header (16 bytes):
  - magic: "TAU\0" (4 bytes)
  - num_pairs: int32
  - table_size: int32
  - reserved: int32
Data:
  - int32[num_pairs][table_size]
```

## 编译和运行

### Windows (使用GCC/MinGW)
```batch
# 方法1: 使用build.bat
build.bat

# 方法2: 使用make (需要安装make)
make

# 运行
bin\cross3d_preprocess.exe
```

### Linux/Mac
```bash
make
./bin/cross3d_preprocess
```

## HLS移植指南

将以下模块移植到HLS：

1. **FFT模块** (`fft.c`)
   - 可直接使用Xilinx FFT IP核替代
   - 或将`fft_forward()`函数转换为HLS

2. **GCC-PHAT模块** (`gcc_phat.c`)
   - `gcc_phat_compute_pair()`函数适合HLS流水线化
   - 复数运算可并行化

3. **SRP-Map模块** (`srp_map.c`)
   - `srp_map_compute()`函数高度并行
   - Tau Table存储在BRAM中

## 输出示例

```
╔══════════════════════════════════════════════════════════════╗
║           Cross3D Preprocessing - C Implementation           ║
╚══════════════════════════════════════════════════════════════╝

Configuration:
  Sample Rate:     24000 Hz
  Channels:        12
  Frame Length:    4096 samples
  FFT Size:        4096
  Mic Pairs:       66
  SRP Grid:        5 x 4 x 8

========== Step 1: Initialize Modules ==========
[INFO] FFT module initialized (N=4096)
[INFO] Generated 66 microphone pairs
[INFO] GCC-PHAT module initialized
[INFO] Computing Tau Table...
[INFO] SRP-Map module initialized

========== Step 2: Generate Test Audio ==========
[INFO] Simulated source at angle=0.79 rad (45.00 deg)
[INFO] Generated 12-channel audio, 108544 samples

...
```

## 作者

Cross3D C Implementation - 2024
