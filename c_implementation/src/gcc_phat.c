/**
 * @file gcc_phat.c
 * @brief GCC-PHAT模块实现
 * @author Cross3D C Implementation
 * @date 2024
 * 
 * 实现广义互相关-相位变换算法
 * GCC-PHAT(f) = IFFT{ X1(f) * conj(X2(f)) / |X1(f) * conj(X2(f))| }
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gcc_phat.h"
#include "fft.h"

/*============================================================================
 * 静态变量
 *============================================================================*/
static mic_pair_t g_mic_pairs[NUM_MIC_PAIRS];
static int g_gcc_initialized = 0;

/* 用于IFFT的临时缓冲区 */
static complex_t g_cross_spectrum[FFT_SIZE];
static complex_t g_ifft_result[FFT_SIZE];

/*============================================================================
 * 函数实现
 *============================================================================*/

void gcc_phat_init_mic_pairs(void)
{
    int pair_idx = 0;
    
    /* 生成所有麦克风对组合 C(12,2) = 66 */
    for (int i = 0; i < NUM_CHANNELS; i++) {
        for (int j = i + 1; j < NUM_CHANNELS; j++) {
            g_mic_pairs[pair_idx].mic1 = i;
            g_mic_pairs[pair_idx].mic2 = j;
            pair_idx++;
        }
    }
    
    printf("[INFO] Generated %d microphone pairs\n", pair_idx);
}

status_t gcc_phat_init(void)
{
    if (g_gcc_initialized) {
        return STATUS_OK;
    }
    
    /* 初始化麦克风对 */
    gcc_phat_init_mic_pairs();
    
    g_gcc_initialized = 1;
    printf("[INFO] GCC-PHAT module initialized\n");
    
    return STATUS_OK;
}

void gcc_phat_cleanup(void)
{
    g_gcc_initialized = 0;
}

void gcc_phat_get_mic_pair(int pair_index, int* mic1, int* mic2)
{
    if (pair_index >= 0 && pair_index < NUM_MIC_PAIRS) {
        *mic1 = g_mic_pairs[pair_index].mic1;
        *mic2 = g_mic_pairs[pair_index].mic2;
    } else {
        *mic1 = -1;
        *mic2 = -1;
    }
}

status_t gcc_phat_compute_pair(const complex_t* fft_ch1,
                                const complex_t* fft_ch2,
                                float32_t* gcc_output,
                                int num_bins)
{
    const float32_t epsilon = 1e-10f;  /* 防止除零 */
    
    /* 
     * 步骤1: 计算互功率谱 X1(f) * conj(X2(f))
     * 步骤2: PHAT加权 (归一化)
     */
    
    /* 清零交叉谱缓冲区 */
    memset(g_cross_spectrum, 0, FFT_SIZE * sizeof(complex_t));
    
    /* 计算正频率部分 */
    for (int bin = 0; bin < num_bins; bin++) {
        /* 互功率谱: X1 * conj(X2) */
        complex_t conj_ch2 = complex_conjugate(fft_ch2[bin]);
        complex_t cross = complex_multiply(fft_ch1[bin], conj_ch2);
        
        /* PHAT加权: 归一化 */
        float32_t magnitude = complex_magnitude(cross);
        if (magnitude > epsilon) {
            g_cross_spectrum[bin].real = cross.real / magnitude;
            g_cross_spectrum[bin].imag = cross.imag / magnitude;
        } else {
            g_cross_spectrum[bin].real = 0.0f;
            g_cross_spectrum[bin].imag = 0.0f;
        }
    }
    
    /* 利用共轭对称性填充负频率部分 */
    for (int bin = 1; bin < num_bins - 1; bin++) {
        g_cross_spectrum[FFT_SIZE - bin] = complex_conjugate(g_cross_spectrum[bin]);
    }
    
    /* 步骤3: IFFT得到GCC */
    status_t status = fft_inverse(g_cross_spectrum, g_ifft_result, FFT_SIZE);
    if (status != STATUS_OK) {
        return status;
    }
    
    /* 提取实部作为GCC结果，并进行fftshift */
    int half = FFT_SIZE / 2;
    for (int i = 0; i < FFT_SIZE; i++) {
        /* fftshift: 将零时延移到中心 */
        int shifted_idx = (i + half) % FFT_SIZE;
        gcc_output[shifted_idx] = g_ifft_result[i].real;
    }
    
    return STATUS_OK;
}

status_t gcc_phat_compute_all(const fft_result_t* fft_result, 
                               gcc_result_t* gcc_result)
{
    if (!g_gcc_initialized) {
        printf("[ERROR] GCC-PHAT module not initialized\n");
        return STATUS_ERROR_INVALID_PARAM;
    }
    
    /* 遍历所有麦克风对 */
    for (int pair = 0; pair < NUM_MIC_PAIRS; pair++) {
        int mic1 = g_mic_pairs[pair].mic1;
        int mic2 = g_mic_pairs[pair].mic2;
        
        status_t status = gcc_phat_compute_pair(
            fft_result->data[mic1],
            fft_result->data[mic2],
            gcc_result->data[pair],
            FFT_BINS
        );
        
        if (status != STATUS_OK) {
            printf("[ERROR] GCC-PHAT failed for pair %d (mic %d, %d)\n", 
                   pair, mic1, mic2);
            return status;
        }
    }
    
    return STATUS_OK;
}

void gcc_phat_print_result(const gcc_result_t* gcc_result, 
                            int pair_index, 
                            int num_samples)
{
    int mic1, mic2;
    gcc_phat_get_mic_pair(pair_index, &mic1, &mic2);
    
    printf("\n=== GCC-PHAT Result (Pair %d: Mic %d - Mic %d) ===\n", 
           pair_index, mic1, mic2);
    
    /* 打印中心区域（零时延附近） */
    int center = FFT_SIZE / 2;
    int half_range = num_samples / 2;
    
    printf("Sample\t\tTau\t\tValue\n");
    for (int i = -half_range; i <= half_range; i++) {
        int idx = center + i;
        if (idx >= 0 && idx < GCC_LENGTH) {
            printf("%d\t\t%d\t\t%.6f\n", idx, i, gcc_result->data[pair_index][idx]);
        }
    }
    
    /* 找到峰值 */
    float32_t max_val = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < GCC_LENGTH; i++) {
        if (gcc_result->data[pair_index][i] > max_val) {
            max_val = gcc_result->data[pair_index][i];
            max_idx = i;
        }
    }
    
    printf("\nPeak: index=%d, tau=%d samples, value=%.6f\n", 
           max_idx, max_idx - center, max_val);
    printf("================================================\n\n");
}
