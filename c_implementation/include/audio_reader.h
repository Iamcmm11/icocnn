/**
 * @file audio_reader.h
 * @brief 音频数据读取模块头文件
 * @author Cross3D C Implementation
 * @date 2024
 * 
 * 该模块负责从文件读取多通道音频数据，
 * 并进行分帧和加窗处理。
 */

#ifndef AUDIO_READER_H
#define AUDIO_READER_H

#include "types.h"
#include "config.h"

/*============================================================================
 * 函数声明
 *============================================================================*/

/**
 * @brief 从二进制文件读取多通道音频数据
 * @param filename 文件路径
 * @param audio_data 输出音频数据 [NUM_CHANNELS][total_samples]
 * @param total_samples 总采样点数
 * @return 状态码
 */
status_t audio_read_from_file(const char* filename, 
                               float32_t** audio_data, 
                               int* total_samples);

/**
 * @brief 对音频数据进行分帧
 * @param audio_data 输入音频数据 [NUM_CHANNELS][total_samples]
 * @param total_samples 总采样点数
 * @param frame_index 帧索引
 * @param frame 输出帧数据
 * @return 状态码
 */
status_t audio_get_frame(float32_t** audio_data,
                          int total_samples,
                          int frame_index,
                          audio_frame_t* frame);

/**
 * @brief 对帧数据应用汉宁窗
 * @param frame 输入/输出帧数据
 * @return 状态码
 */
status_t audio_apply_hanning_window(audio_frame_t* frame);

/**
 * @brief 生成汉宁窗系数
 * @param window 输出窗函数系数
 * @param length 窗长度
 */
void audio_generate_hanning_window(float32_t* window, int length);

/**
 * @brief 释放音频数据内存
 * @param audio_data 音频数据指针
 */
void audio_free_data(float32_t** audio_data);

/**
 * @brief 打印音频帧信息（调试用）
 * @param frame 帧数据
 */
void audio_print_frame_info(const audio_frame_t* frame);

#endif /* AUDIO_READER_H */
