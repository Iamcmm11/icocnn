#include "ico_conv_layer0.h"
#include <string.h>

/**
 * @brief 单个像素的卷积计算
 * 
 * 对于一个顶点 v，计算输出通道 out_c 的值:
 * output[out_c][v] = sum_{in_c, neighbor_n} (
 *     input[in_c][neighbors[v][n]] * weight[out_c][in_c][n]
 * ) + bias[out_c]
 */
data_t compute_conv_pixel(
    const data_t input_tile[IN_CHANNELS][NUM_VERTICES],
    const data_t weight_tile[IN_CHANNELS][NUM_NEIGHBORS],
    const index_t* neighbor_list,
    int vertex_idx
) {
    data_t sum = 0.0f;
    
    // 遍历所有输入通道
    for (int in_c = 0; in_c < IN_CHANNELS; in_c++) {
        // 遍历邻居
        for (int n = 0; n < NUM_NEIGHBORS; n++) {
            int neighbor_v = neighbor_list[n];
            data_t in_val = input_tile[in_c][neighbor_v];
            data_t w_val = weight_tile[in_c][n];
            sum += in_val * w_val;
        }
    }
    
    return sum;
}

/**
 * @brief 加载输入 tile 到本地缓存
 */
void load_input_tile(
    const data_t* input,
    data_t tile[IN_CHANNELS][NUM_VERTICES],
    int batch_idx,
    int time_idx
) {
    for (int c = 0; c < IN_CHANNELS; c++) {
        for (int v = 0; v < NUM_VERTICES; v++) {
            int idx = INPUT_IDX(batch_idx, c, time_idx, v);
            tile[c][v] = input[idx];
        }
    }
}

/**
 * @brief 加载权重 tile 到本地缓存
 */
void load_weight_tile(
    const data_t* weight,
    data_t tile[OUT_CHANNELS][IN_CHANNELS][NUM_NEIGHBORS]
) {
    for (int out_c = 0; out_c < OUT_CHANNELS; out_c++) {
        for (int in_c = 0; in_c < IN_CHANNELS; in_c++) {
            for (int n = 0; n < NUM_NEIGHBORS; n++) {
                int idx = WEIGHT_IDX(out_c, in_c, n);
                tile[out_c][in_c][n] = weight[idx];
            }
        }
    }
}

/**
 * @brief Icosahedral 卷积层核心实现
 */
void ico_conv_layer0(
    const data_t input[INPUT_SIZE],
    const data_t weight[WEIGHT_SIZE],
    const data_t bias[BIAS_SIZE],
    const index_t neighbors[NEIGHBORS_SIZE],
    data_t output[OUTPUT_SIZE]
) {
    // 权重缓存 (HLS 会将其映射到 BRAM)
    static data_t weight_cache[OUT_CHANNELS][IN_CHANNELS][NUM_NEIGHBORS];
    
    // 加载权重到缓存
    load_weight_tile(weight, weight_cache);
    
    // 遍历 batch
    for (int b = 0; b < BATCH_SIZE; b++) {
        // 遍历时间帧
        for (int t = 0; t < TIME_FRAMES; t++) {
            // 输入 tile 缓存
            data_t input_tile[IN_CHANNELS][NUM_VERTICES];
            load_input_tile(input, input_tile, b, t);
            
            // 遍历输出通道
            for (int out_c = 0; out_c < OUT_CHANNELS; out_c++) {
                data_t bias_val = bias[out_c];
                
                // 遍历所有顶点
                for (int v = 0; v < NUM_VERTICES; v++) {
                    // 获取该顶点的邻居列表
                    index_t neighbor_list[NUM_NEIGHBORS];
                    for (int n = 0; n < NUM_NEIGHBORS; n++) {
                        neighbor_list[n] = neighbors[NEIGHBOR_IDX(v, n)];
                    }
                    
                    // 计算卷积
                    data_t conv_val = compute_conv_pixel(
                        input_tile,
                        weight_cache[out_c],
                        neighbor_list,
                        v
                    );
                    
                    // 加偏置 + ReLU
                    data_t out_val = relu(conv_val + bias_val);
                    
                    // 写回输出
                    int out_idx = OUTPUT_IDX(b, out_c, t, v);
                    output[out_idx] = out_val;
                }
            }
        }
    }
}
