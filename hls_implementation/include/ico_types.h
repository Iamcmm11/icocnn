#ifndef ICO_TYPES_H
#define ICO_TYPES_H

#include <stdint.h>

// 基本类型定义
typedef float data_t;      // 数据类型 (float32)
typedef int index_t;       // 索引类型

// Layer0 配置参数 - 基于实际icoCNN数据
#define R_LEVEL 2
#define H (1 << R_LEVEL)     // 4 (2^r)
#define W (1 << (R_LEVEL+1)) // 8 (2^(r+1))
#define NUM_CHARTS 5         // icosahedral 20面体的5个chart
#define NUM_VERTICES (NUM_CHARTS * H * W)  // 160 = 5 * 4 * 8
#define TIME_FRAMES 103      // 时间帧数
#define IN_CHANNELS 1        // 输入通道数
#define IN_ROTATIONS 1       // 输入旋转方向数
#define OUT_CHANNELS 32      // 输出通道数
#define OUT_ROTATIONS 6      // 输出旋转方向数 (旋转等变性)
#define NUM_NEIGHBORS 7      // 每个顶点的邻居数 (包括自己)
#define BATCH_SIZE 1

// 数组大小定义
// 注意：实际PyTorch输出形状是 [B, T, C_out, R_out, Charts, H, W]
// 但为了HLS实现简单，我们重排为 [B, C_out, R_out, T, V]
#define INPUT_SIZE (BATCH_SIZE * IN_CHANNELS * IN_ROTATIONS * TIME_FRAMES * NUM_VERTICES)
#define OUTPUT_SIZE (BATCH_SIZE * OUT_CHANNELS * OUT_ROTATIONS * TIME_FRAMES * NUM_VERTICES)
#define WEIGHT_SIZE (OUT_CHANNELS * IN_CHANNELS * IN_ROTATIONS * NUM_NEIGHBORS)
#define BIAS_SIZE (OUT_CHANNELS)
#define NEIGHBORS_SIZE (NUM_VERTICES * NUM_NEIGHBORS)

// ReLU 激活函数
inline data_t relu(data_t x) {
    return (x > 0) ? x : 0;
}

// 张量索引宏
// input[b][c][r_in][t][v] -> index (r_in=1 固定)
#define INPUT_IDX(b, c, r_in, t, v) \
    ((b) * IN_CHANNELS * IN_ROTATIONS * TIME_FRAMES * NUM_VERTICES + \
     (c) * IN_ROTATIONS * TIME_FRAMES * NUM_VERTICES + \
     (r_in) * TIME_FRAMES * NUM_VERTICES + \
     (t) * NUM_VERTICES + \
     (v))

// output[b][c][r_out][t][v] -> index
#define OUTPUT_IDX(b, c, r_out, t, v) \
    ((b) * OUT_CHANNELS * OUT_ROTATIONS * TIME_FRAMES * NUM_VERTICES + \
     (c) * OUT_ROTATIONS * TIME_FRAMES * NUM_VERTICES + \
     (r_out) * TIME_FRAMES * NUM_VERTICES + \
     (t) * NUM_VERTICES + \
     (v))

// weight[out_c][in_c][r_in][neighbor] -> index
#define WEIGHT_IDX(out_c, in_c, r_in, n) \
    ((out_c) * IN_CHANNELS * IN_ROTATIONS * NUM_NEIGHBORS + \
     (in_c) * IN_ROTATIONS * NUM_NEIGHBORS + \
     (r_in) * NUM_NEIGHBORS + \
     (n))

// neighbors[v][n] -> index
#define NEIGHBOR_IDX(v, n) ((v) * NUM_NEIGHBORS + (n))

#endif // ICO_TYPES_H
