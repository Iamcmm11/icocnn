#ifndef ICO_TYPES_H
#define ICO_TYPES_H

#include <stdint.h>

// 基本类型定义
typedef float data_t;      // 数据类型 (float32)
typedef int index_t;       // 索引类型

// Layer0 配置参数
#define R_LEVEL 2
#define NUM_VERTICES 42      // 10*R^2 + 2 = 42
#define TIME_FRAMES 10       // 时间帧数
#define IN_CHANNELS 4        // 输入通道数
#define OUT_CHANNELS 32      // 输出通道数
#define NUM_NEIGHBORS 7      // 每个顶点的邻居数 (包括自己)
#define BATCH_SIZE 1

// 数组大小定义
#define INPUT_SIZE (BATCH_SIZE * IN_CHANNELS * TIME_FRAMES * NUM_VERTICES)
#define OUTPUT_SIZE (BATCH_SIZE * OUT_CHANNELS * TIME_FRAMES * NUM_VERTICES)
#define WEIGHT_SIZE (OUT_CHANNELS * IN_CHANNELS * NUM_NEIGHBORS)
#define BIAS_SIZE (OUT_CHANNELS)
#define NEIGHBORS_SIZE (NUM_VERTICES * NUM_NEIGHBORS)

// ReLU 激活函数
inline data_t relu(data_t x) {
    return (x > 0) ? x : 0;
}

// 张量索引宏 (NCHW 格式)
// input[b][c][t][v] -> index
#define INPUT_IDX(b, c, t, v) \
    ((b) * IN_CHANNELS * TIME_FRAMES * NUM_VERTICES + \
     (c) * TIME_FRAMES * NUM_VERTICES + \
     (t) * NUM_VERTICES + \
     (v))

// output[b][c][t][v] -> index
#define OUTPUT_IDX(b, c, t, v) \
    ((b) * OUT_CHANNELS * TIME_FRAMES * NUM_VERTICES + \
     (c) * TIME_FRAMES * NUM_VERTICES + \
     (t) * NUM_VERTICES + \
     (v))

// weight[out_c][in_c][neighbor] -> index
#define WEIGHT_IDX(out_c, in_c, n) \
    ((out_c) * IN_CHANNELS * NUM_NEIGHBORS + \
     (in_c) * NUM_NEIGHBORS + \
     (n))

// neighbors[v][n] -> index
#define NEIGHBOR_IDX(v, n) ((v) * NUM_NEIGHBORS + (n))

#endif // ICO_TYPES_H
