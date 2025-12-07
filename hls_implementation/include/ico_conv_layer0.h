#ifndef ICO_CONV_LAYER0_H
#define ICO_CONV_LAYER0_H

#include "ico_types.h"

/**
 * @brief Icosahedral 卷积层 - Layer0
 * 
 * 输入: [batch, in_channels, time_frames, num_vertices]
 * 输出: [batch, out_channels, time_frames, num_vertices]
 * 
 * @param input      输入特征图
 * @param weight     卷积权重 [out_ch, in_ch, num_neighbors]
 * @param bias       偏置 [out_ch]
 * @param neighbors  邻居索引表 [num_vertices, num_neighbors]
 * @param output     输出特征图
 */
void ico_conv_layer0(
    const data_t input[INPUT_SIZE],
    const data_t weight[WEIGHT_SIZE],
    const data_t bias[BIAS_SIZE],
    const index_t neighbors[NEIGHBORS_SIZE],
    data_t output[OUTPUT_SIZE]
);

/**
 * @brief 加载数据到本地缓存 (为 HLS 优化)
 */
void load_input_tile(
    const data_t* input,
    data_t tile[IN_CHANNELS][NUM_VERTICES],
    int batch_idx,
    int time_idx
);

void load_weight_tile(
    const data_t* weight,
    data_t tile[OUT_CHANNELS][IN_CHANNELS][NUM_NEIGHBORS]
);

/**
 * @brief 单个输出通道的卷积计算
 */
data_t compute_conv_pixel(
    const data_t input_tile[IN_CHANNELS][NUM_VERTICES],
    const data_t weight_tile[IN_CHANNELS][NUM_NEIGHBORS],
    const index_t* neighbor_list,
    int vertex_idx
);

#endif // ICO_CONV_LAYER0_H
