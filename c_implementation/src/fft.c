/**
 * @file fft.c
 * @brief FFT模块实现
 * @author Cross3D C Implementation
 * @date 2024
 * 
 * 采用Cooley-Tukey基2 FFT算法实现
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fft.h"

/*============================================================================
 * 静态变量
 *============================================================================*/
static complex_t* g_twiddle_factors = NULL;  /* 旋转因子 */
static int* g_bit_reverse_table = NULL;       /* 位反转表 */
static int g_fft_initialized = 0;

/*============================================================================
 * 辅助函数
 *============================================================================*/

/**
 * @brief 计算位反转索引
 */
static int bit_reverse(int x, int log2n)
{
    int result = 0;
    for (int i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

/**
 * @brief 计算log2(n)
 */
static int log2_int(int n)
{
    int result = 0;
    while (n > 1) {
        n >>= 1;
        result++;
    }
    return result;
}

/*============================================================================
 * 复数运算函数
 *============================================================================*/

complex_t complex_multiply(complex_t a, complex_t b)
{
    complex_t result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

complex_t complex_conjugate(complex_t a)
{
    complex_t result;
    result.real = a.real;
    result.imag = -a.imag;
    return result;
}

float32_t complex_magnitude(complex_t a)
{
    return sqrtf(a.real * a.real + a.imag * a.imag);
}

static complex_t complex_add(complex_t a, complex_t b)
{
    complex_t result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

static complex_t complex_sub(complex_t a, complex_t b)
{
    complex_t result;
    result.real = a.real - b.real;
    result.imag = a.imag - b.imag;
    return result;
}

/*============================================================================
 * FFT核心函数
 *============================================================================*/

status_t fft_init(void)
{
    if (g_fft_initialized) {
        return STATUS_OK;
    }
    
    int log2n = log2_int(FFT_SIZE);
    
    /* 分配旋转因子内存 */
    g_twiddle_factors = (complex_t*)malloc(FFT_SIZE / 2 * sizeof(complex_t));
    if (g_twiddle_factors == NULL) {
        return STATUS_ERROR_MEMORY_ALLOC;
    }
    
    /* 预计算旋转因子 W_N^k = exp(-j * 2 * pi * k / N) */
    for (int k = 0; k < FFT_SIZE / 2; k++) {
        float32_t angle = -TWO_PI * k / FFT_SIZE;
        g_twiddle_factors[k].real = cosf(angle);
        g_twiddle_factors[k].imag = sinf(angle);
    }
    
    /* 分配位反转表内存 */
    g_bit_reverse_table = (int*)malloc(FFT_SIZE * sizeof(int));
    if (g_bit_reverse_table == NULL) {
        free(g_twiddle_factors);
        return STATUS_ERROR_MEMORY_ALLOC;
    }
    
    /* 预计算位反转索引 */
    for (int i = 0; i < FFT_SIZE; i++) {
        g_bit_reverse_table[i] = bit_reverse(i, log2n);
    }
    
    g_fft_initialized = 1;
    printf("[INFO] FFT module initialized (N=%d)\n", FFT_SIZE);
    
    return STATUS_OK;
}

void fft_cleanup(void)
{
    if (g_twiddle_factors != NULL) {
        free(g_twiddle_factors);
        g_twiddle_factors = NULL;
    }
    if (g_bit_reverse_table != NULL) {
        free(g_bit_reverse_table);
        g_bit_reverse_table = NULL;
    }
    g_fft_initialized = 0;
}

status_t fft_forward(const float32_t* input, complex_t* output, int n)
{
    if (!g_fft_initialized) {
        return STATUS_ERROR_FFT_FAILED;
    }
    
    int log2n = log2_int(n);
    
    /* 位反转重排 + 实数转复数 */
    for (int i = 0; i < n; i++) {
        int j = g_bit_reverse_table[i];
        output[j].real = input[i];
        output[j].imag = 0.0f;
    }
    
    /* Cooley-Tukey 蝶形运算 */
    for (int stage = 1; stage <= log2n; stage++) {
        int m = 1 << stage;          /* 当前阶段的蝶形大小 */
        int m2 = m >> 1;             /* 半蝶形大小 */
        int step = FFT_SIZE / m;     /* 旋转因子步长 */
        
        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < m2; j++) {
                complex_t w = g_twiddle_factors[j * step];
                complex_t t = complex_multiply(w, output[k + j + m2]);
                complex_t u = output[k + j];
                
                output[k + j] = complex_add(u, t);
                output[k + j + m2] = complex_sub(u, t);
            }
        }
    }
    
    return STATUS_OK;
}

status_t fft_inverse(const complex_t* input, complex_t* output, int n)
{
    if (!g_fft_initialized) {
        return STATUS_ERROR_FFT_FAILED;
    }
    
    int log2n = log2_int(n);
    
    /* 位反转重排 */
    for (int i = 0; i < n; i++) {
        int j = g_bit_reverse_table[i];
        output[j] = input[i];
    }
    
    /* IFFT: 使用共轭旋转因子 */
    for (int stage = 1; stage <= log2n; stage++) {
        int m = 1 << stage;
        int m2 = m >> 1;
        int step = FFT_SIZE / m;
        
        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < m2; j++) {
                /* IFFT使用共轭旋转因子 */
                complex_t w = complex_conjugate(g_twiddle_factors[j * step]);
                complex_t t = complex_multiply(w, output[k + j + m2]);
                complex_t u = output[k + j];
                
                output[k + j] = complex_add(u, t);
                output[k + j + m2] = complex_sub(u, t);
            }
        }
    }
    
    /* 归一化 */
    float32_t scale = 1.0f / n;
    for (int i = 0; i < n; i++) {
        output[i].real *= scale;
        output[i].imag *= scale;
    }
    
    return STATUS_OK;
}

status_t fft_execute_real(const audio_frame_t* frame, fft_result_t* result)
{
    static complex_t temp_buffer[FFT_SIZE];
    
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        /* 执行FFT */
        status_t status = fft_forward(frame->data[ch], temp_buffer, FFT_SIZE);
        if (status != STATUS_OK) {
            return status;
        }
        
        /* 只保留正频率部分 (0 到 N/2) */
        for (int bin = 0; bin < FFT_BINS; bin++) {
            result->data[ch][bin] = temp_buffer[bin];
        }
    }
    
    return STATUS_OK;
}

void fft_print_result(const fft_result_t* result, int channel, int num_bins)
{
    printf("\n=== FFT Result (Channel %d) ===\n", channel);
    printf("Bin\t\tReal\t\tImag\t\tMagnitude\n");
    
    for (int i = 0; i < num_bins && i < FFT_BINS; i++) {
        float32_t mag = complex_magnitude(result->data[channel][i]);
        printf("%d\t\t%.4f\t\t%.4f\t\t%.4f\n", 
               i, 
               result->data[channel][i].real,
               result->data[channel][i].imag,
               mag);
    }
    printf("==============================\n\n");
}
