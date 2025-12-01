/**
 * @file test_data.h
 * @brief 测试数据生成和保存模块头文件
 * @author Cross3D C Implementation
 * @date 2024
 * 
 * 该模块用于生成模拟的多通道麦克风信号，
 * 并提供数据保存和加载功能，便于HLS和ARM测试。
 */

#ifndef TEST_DATA_H
#define TEST_DATA_H

#include "types.h"
#include "config.h"

/*============================================================================
 * 数据文件格式定义
 *============================================================================*/
/*
 * 二进制文件格式 (.bin):
 * 
 * 1. 音频数据文件 (audio_data.bin):
 *    Header (16 bytes):
 *      - magic: 4 bytes = "AUD\0"
 *      - num_channels: 4 bytes (int32)
 *      - num_samples: 4 bytes (int32)
 *      - sample_rate: 4 bytes (int32)
 *    Data:
 *      - float32[num_channels][num_samples] (交织存储)
 * 
 * 2. FFT结果文件 (fft_result.bin):
 *    Header (16 bytes):
 *      - magic: 4 bytes = "FFT\0"
 *      - num_channels: 4 bytes (int32)
 *      - num_bins: 4 bytes (int32)
 *      - reserved: 4 bytes
 *    Data:
 *      - complex_t[num_channels][num_bins]
 * 
 * 3. GCC结果文件 (gcc_result.bin):
 *    Header (16 bytes):
 *      - magic: 4 bytes = "GCC\0"
 *      - num_pairs: 4 bytes (int32)
 *      - gcc_length: 4 bytes (int32)
 *      - reserved: 4 bytes
 *    Data:
 *      - float32[num_pairs][gcc_length]
 * 
 * 4. SRP结果文件 (srp_result.bin):
 *    Header (16 bytes):
 *      - magic: 4 bytes = "SRP\0"
 *      - elevation_bins: 4 bytes (int32)
 *      - azimuth_bins: 4 bytes (int32)
 *      - range_bins: 4 bytes (int32)
 *    Data:
 *      - float32[elevation_bins][azimuth_bins][range_bins]
 */

/*============================================================================
 * 文件头结构
 *============================================================================*/
typedef struct {
    char magic[4];
    int32_t param1;
    int32_t param2;
    int32_t param3;
} file_header_t;

/*============================================================================
 * 函数声明
 *============================================================================*/

/**
 * @brief 生成模拟的多通道麦克风信号
 * @param audio_data 输出音频数据
 * @param num_samples 每通道采样点数
 * @param source_angle 模拟声源角度 (弧度)
 * @return 状态码
 */
status_t test_data_generate_audio(float32_t** audio_data,
                                   int num_samples,
                                   float32_t source_angle);

/**
 * @brief 生成带有时延的正弦波信号
 * @param output 输出信号
 * @param num_samples 采样点数
 * @param frequency 频率 (Hz)
 * @param delay_samples 时延采样点数
 * @param amplitude 幅度
 */
void test_data_generate_sine_with_delay(float32_t* output,
                                         int num_samples,
                                         float32_t frequency,
                                         float32_t delay_samples,
                                         float32_t amplitude);

/**
 * @brief 添加高斯白噪声
 * @param data 输入/输出数据
 * @param num_samples 采样点数
 * @param snr_db 信噪比 (dB)
 */
void test_data_add_noise(float32_t* data, int num_samples, float32_t snr_db);

/**
 * @brief 保存音频数据到二进制文件
 * @param filename 文件路径
 * @param audio_data 音频数据
 * @param num_channels 通道数
 * @param num_samples 采样点数
 * @return 状态码
 */
status_t test_data_save_audio(const char* filename,
                               float32_t** audio_data,
                               int num_channels,
                               int num_samples);

/**
 * @brief 从二进制文件加载音频数据
 * @param filename 文件路径
 * @param audio_data 输出音频数据
 * @param num_channels 输出通道数
 * @param num_samples 输出采样点数
 * @return 状态码
 */
status_t test_data_load_audio(const char* filename,
                               float32_t*** audio_data,
                               int* num_channels,
                               int* num_samples);

/**
 * @brief 保存FFT结果到二进制文件
 * @param filename 文件路径
 * @param fft_result FFT结果
 * @return 状态码
 */
status_t test_data_save_fft(const char* filename,
                             const fft_result_t* fft_result);

/**
 * @brief 保存GCC结果到二进制文件
 * @param filename 文件路径
 * @param gcc_result GCC结果
 * @return 状态码
 */
status_t test_data_save_gcc(const char* filename,
                             const gcc_result_t* gcc_result);

/**
 * @brief 保存SRP结果到二进制文件
 * @param filename 文件路径
 * @param srp_result SRP结果
 * @return 状态码
 */
status_t test_data_save_srp(const char* filename,
                             const srp_map_t* srp_result);

/**
 * @brief 保存数据为文本格式（便于查看）
 * @param filename 文件路径
 * @param data 数据指针
 * @param rows 行数
 * @param cols 列数
 * @return 状态码
 */
status_t test_data_save_as_text(const char* filename,
                                 const float32_t* data,
                                 int rows,
                                 int cols);

/**
 * @brief 生成默认麦克风阵列位置（12元环形阵列）
 * @param positions 输出位置数组
 * @param radius 阵列半径 (m)
 */
void test_data_generate_mic_positions(mic_position_t* positions, 
                                       float32_t radius);

/**
 * @brief 分配音频数据内存
 * @param num_channels 通道数
 * @param num_samples 采样点数
 * @return 分配的内存指针
 */
float32_t** test_data_alloc_audio(int num_channels, int num_samples);

/**
 * @brief 释放音频数据内存
 * @param audio_data 音频数据指针
 * @param num_channels 通道数
 */
void test_data_free_audio(float32_t** audio_data, int num_channels);

#endif /* TEST_DATA_H */
