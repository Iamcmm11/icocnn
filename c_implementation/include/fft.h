/**
 * @file fft.h
 * @brief FFT模块头文件
 * @author Cross3D C Implementation
 * @date 2024
 * 
 * 该模块实现快速傅里叶变换(FFT)和逆变换(IFFT)，
 * 采用Cooley-Tukey算法。
 */

#ifndef FFT_H
#define FFT_H

#include "types.h"
#include "config.h"

/*============================================================================
 * 函数声明
 *============================================================================*/

/**
 * @brief 初始化FFT模块（预计算旋转因子）
 * @return 状态码
 */
status_t fft_init(void);

/**
 * @brief 释放FFT模块资源
 */
void fft_cleanup(void);

/**
 * @brief 对音频帧执行实数FFT
 * @param frame 输入音频帧
 * @param result 输出FFT结果
 * @return 状态码
 */
status_t fft_execute_real(const audio_frame_t* frame, fft_result_t* result);

/**
 * @brief 对单通道数据执行FFT
 * @param input 输入实数数据
 * @param output 输出复数数据
 * @param n FFT点数
 * @return 状态码
 */
status_t fft_forward(const float32_t* input, complex_t* output, int n);

/**
 * @brief 对复数数据执行IFFT
 * @param input 输入复数数据
 * @param output 输出复数数据
 * @param n IFFT点数
 * @return 状态码
 */
status_t fft_inverse(const complex_t* input, complex_t* output, int n);

/**
 * @brief 复数乘法
 * @param a 复数a
 * @param b 复数b
 * @return 乘积结果
 */
complex_t complex_multiply(complex_t a, complex_t b);

/**
 * @brief 复数共轭
 * @param a 输入复数
 * @return 共轭结果
 */
complex_t complex_conjugate(complex_t a);

/**
 * @brief 复数模值
 * @param a 输入复数
 * @return 模值
 */
float32_t complex_magnitude(complex_t a);

/**
 * @brief 打印FFT结果信息（调试用）
 * @param result FFT结果
 * @param channel 通道索引
 * @param num_bins 打印的频点数
 */
void fft_print_result(const fft_result_t* result, int channel, int num_bins);

#endif /* FFT_H */
