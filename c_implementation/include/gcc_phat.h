/**
 * @file gcc_phat.h
 * @brief GCC-PHAT模块头文件
 * @author Cross3D C Implementation
 * @date 2024
 * 
 * 该模块实现广义互相关-相位变换(GCC-PHAT)算法，
 * 用于计算麦克风对之间的时延估计。
 */

#ifndef GCC_PHAT_H
#define GCC_PHAT_H

#include "types.h"
#include "config.h"

/*============================================================================
 * 函数声明
 *============================================================================*/

/**
 * @brief 初始化GCC-PHAT模块
 * @return 状态码
 */
status_t gcc_phat_init(void);

/**
 * @brief 释放GCC-PHAT模块资源
 */
void gcc_phat_cleanup(void);

/**
 * @brief 计算所有麦克风对的GCC-PHAT
 * @param fft_result 输入FFT结果
 * @param gcc_result 输出GCC结果
 * @return 状态码
 */
status_t gcc_phat_compute_all(const fft_result_t* fft_result, 
                               gcc_result_t* gcc_result);

/**
 * @brief 计算单个麦克风对的GCC-PHAT
 * @param fft_ch1 通道1的FFT结果
 * @param fft_ch2 通道2的FFT结果
 * @param gcc_output 输出GCC结果
 * @param num_bins FFT频点数
 * @return 状态码
 */
status_t gcc_phat_compute_pair(const complex_t* fft_ch1,
                                const complex_t* fft_ch2,
                                float32_t* gcc_output,
                                int num_bins);

/**
 * @brief 获取麦克风对索引
 * @param pair_index 麦克风对索引 (0-65)
 * @param mic1 输出第一个麦克风索引
 * @param mic2 输出第二个麦克风索引
 */
void gcc_phat_get_mic_pair(int pair_index, int* mic1, int* mic2);

/**
 * @brief 初始化麦克风对索引表
 */
void gcc_phat_init_mic_pairs(void);

/**
 * @brief 打印GCC结果信息（调试用）
 * @param gcc_result GCC结果
 * @param pair_index 麦克风对索引
 * @param num_samples 打印的采样点数
 */
void gcc_phat_print_result(const gcc_result_t* gcc_result, 
                            int pair_index, 
                            int num_samples);

#endif /* GCC_PHAT_H */
