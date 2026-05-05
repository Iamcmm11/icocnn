/**
 * @file test_data.c
 * @brief 测试数据生成和保存模块实现
 * @author Cross3D C Implementation
 * @date 2024
 * 
 * 生成模拟的多通道麦克风信号用于测试
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "test_data.h"

/*============================================================================
 * 辅助函数
 *============================================================================*/

/**
 * @brief 生成高斯随机数（Box-Muller变换）
 */
static float32_t randn(void)
{
    static int has_spare = 0;
    static float32_t spare;
    
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    
    float32_t u, v, s;
    do {
        u = (float32_t)rand() / RAND_MAX * 2.0f - 1.0f;
        v = (float32_t)rand() / RAND_MAX * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    
    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    has_spare = 1;
    
    return u * s;
}

/**
 * @brief 计算信号功率
 */
static float32_t compute_power(const float32_t* data, int num_samples)
{
    float32_t sum = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        sum += data[i] * data[i];
    }
    return sum / num_samples;
}

/*============================================================================
 * 函数实现
 *============================================================================*/

void test_data_generate_mic_positions(mic_position_t* positions, float32_t radius)
{
    /* 生成12元环形麦克风阵列 */
    for (int i = 0; i < NUM_CHANNELS; i++) {
        float32_t angle = TWO_PI * i / NUM_CHANNELS;
        positions[i].x = radius * cosf(angle);
        positions[i].y = radius * sinf(angle);
        positions[i].z = 0.0f;  /* 平面阵列 */
    }
    
    printf("[INFO] Generated %d-element circular array, radius=%.3f m\n", 
           NUM_CHANNELS, radius);
    
    /* 打印麦克风位置 */
    printf("Microphone positions:\n");
    for (int i = 0; i < NUM_CHANNELS; i++) {
        printf("  Mic %2d: (%.4f, %.4f, %.4f)\n", 
               i, positions[i].x, positions[i].y, positions[i].z);
    }
}

void test_data_generate_sine_with_delay(float32_t* output,
                                         int num_samples,
                                         float32_t frequency,
                                         float32_t delay_samples,
                                         float32_t amplitude)
{
    for (int i = 0; i < num_samples; i++) {
        float32_t t = (float32_t)(i - delay_samples) / SAMPLE_RATE;
        output[i] = amplitude * sinf(TWO_PI * frequency * t);
    }
}

void test_data_add_noise(float32_t* data, int num_samples, float32_t snr_db)
{
    /* 计算信号功率 */
    float32_t signal_power = compute_power(data, num_samples);
    
    /* 计算噪声功率 */
    float32_t noise_power = signal_power / powf(10.0f, snr_db / 10.0f);
    float32_t noise_std = sqrtf(noise_power);
    
    /* 添加噪声 */
    for (int i = 0; i < num_samples; i++) {
        data[i] += noise_std * randn();
    }
}

float32_t** test_data_alloc_audio(int num_channels, int num_samples)
{
    float32_t** audio_data = (float32_t**)malloc(num_channels * sizeof(float32_t*));
    if (audio_data == NULL) {
        return NULL;
    }
    
    for (int ch = 0; ch < num_channels; ch++) {
        audio_data[ch] = (float32_t*)calloc(num_samples, sizeof(float32_t));
        if (audio_data[ch] == NULL) {
            /* 释放已分配的内存 */
            for (int i = 0; i < ch; i++) {
                free(audio_data[i]);
            }
            free(audio_data);
            return NULL;
        }
    }
    
    return audio_data;
}

void test_data_free_audio(float32_t** audio_data, int num_channels)
{
    if (audio_data != NULL) {
        for (int ch = 0; ch < num_channels; ch++) {
            if (audio_data[ch] != NULL) {
                free(audio_data[ch]);
            }
        }
        free(audio_data);
    }
}

status_t test_data_generate_audio(float32_t** audio_data,
                                   int num_samples,
                                   float32_t source_angle)
{
    /* 初始化随机数生成器 */
    srand((unsigned int)time(NULL));
    
    /* 生成麦克风位置 */
    mic_position_t mic_positions[NUM_CHANNELS];
    float32_t array_radius = 0.05f;  /* 5cm半径 */
    test_data_generate_mic_positions(mic_positions, array_radius);
    
    /* 声源参数 */
    float32_t source_distance = 2.0f;  /* 2米距离 */
    float32_t source_x = source_distance * cosf(source_angle);
    float32_t source_y = source_distance * sinf(source_angle);
    float32_t source_z = 0.0f;
    
    printf("[INFO] Simulated source at angle=%.2f rad (%.2f deg), distance=%.2f m\n",
           source_angle, source_angle * 180.0f / PI, source_distance);
    printf("  Source position: (%.4f, %.4f, %.4f)\n", source_x, source_y, source_z);
    
    /* 信号参数 */
    float32_t frequency = 1000.0f;  /* 1kHz正弦波 */
    float32_t amplitude = 0.8f;
    float32_t snr_db = 20.0f;       /* 20dB信噪比 */
    
    /* 为每个麦克风生成带时延的信号 */
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        /* 计算声源到麦克风的距离 */
        float32_t dx = source_x - mic_positions[ch].x;
        float32_t dy = source_y - mic_positions[ch].y;
        float32_t dz = source_z - mic_positions[ch].z;
        float32_t distance = sqrtf(dx*dx + dy*dy + dz*dz);
        
        /* 计算时延（采样点数） */
        float32_t delay_seconds = distance / SPEED_OF_SOUND;
        float32_t delay_samples = delay_seconds * SAMPLE_RATE;
        
        printf("  Mic %2d: distance=%.4f m, delay=%.2f samples\n", 
               ch, distance, delay_samples);
        
        /* 生成带时延的正弦波 */
        test_data_generate_sine_with_delay(audio_data[ch], num_samples,
                                            frequency, delay_samples, amplitude);
        
        /* 添加噪声 */
        test_data_add_noise(audio_data[ch], num_samples, snr_db);
    }
    
    printf("[INFO] Generated %d-channel audio, %d samples, SNR=%.1f dB\n",
           NUM_CHANNELS, num_samples, snr_db);
    
    return STATUS_OK;
}

status_t test_data_save_audio(const char* filename,
                               float32_t** audio_data,
                               int num_channels,
                               int num_samples)
{
    FILE* fp = fopen(filename, "wb");
    if (fp == NULL) {
        printf("[ERROR] Cannot create file: %s\n", filename);
        return STATUS_ERROR_FILE_NOT_FOUND;
    }
    
    /* 写入文件头 */
    char magic[4] = "AUD";
    int32_t nc = num_channels;
    int32_t ns = num_samples;
    int32_t sr = SAMPLE_RATE;
    
    fwrite(magic, 1, 4, fp);
    fwrite(&nc, sizeof(int32_t), 1, fp);
    fwrite(&ns, sizeof(int32_t), 1, fp);
    fwrite(&sr, sizeof(int32_t), 1, fp);
    
    /* 写入音频数据（按通道存储） */
    for (int ch = 0; ch < num_channels; ch++) {
        fwrite(audio_data[ch], sizeof(float32_t), num_samples, fp);
    }
    
    fclose(fp);
    printf("[INFO] Audio saved to: %s (%d bytes)\n", 
           filename, 16 + num_channels * num_samples * sizeof(float32_t));
    
    return STATUS_OK;
}

status_t test_data_load_audio(const char* filename,
                               float32_t*** audio_data,
                               int* num_channels,
                               int* num_samples)
{
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("[ERROR] Cannot open file: %s\n", filename);
        return STATUS_ERROR_FILE_NOT_FOUND;
    }
    
    /* 读取文件头 */
    char magic[4];
    int32_t nc, ns, sr;
    
    fread(magic, 1, 4, fp);
    if (strncmp(magic, "AUD", 3) != 0) {
        printf("[ERROR] Invalid audio file format\n");
        fclose(fp);
        return STATUS_ERROR_INVALID_PARAM;
    }
    
    fread(&nc, sizeof(int32_t), 1, fp);
    fread(&ns, sizeof(int32_t), 1, fp);
    fread(&sr, sizeof(int32_t), 1, fp);
    
    /* 分配内存 */
    *audio_data = test_data_alloc_audio(nc, ns);
    if (*audio_data == NULL) {
        fclose(fp);
        return STATUS_ERROR_MEMORY_ALLOC;
    }
    
    /* 读取音频数据 */
    for (int ch = 0; ch < nc; ch++) {
        fread((*audio_data)[ch], sizeof(float32_t), ns, fp);
    }
    
    *num_channels = nc;
    *num_samples = ns;
    
    fclose(fp);
    printf("[INFO] Audio loaded: %d channels, %d samples, %d Hz\n", nc, ns, sr);
    
    return STATUS_OK;
}

status_t test_data_save_fft(const char* filename,
                             const fft_result_t* fft_result)
{
    FILE* fp = fopen(filename, "wb");
    if (fp == NULL) {
        printf("[ERROR] Cannot create file: %s\n", filename);
        return STATUS_ERROR_FILE_NOT_FOUND;
    }
    
    /* 写入文件头 */
    char magic[4] = "FFT";
    int32_t nc = NUM_CHANNELS;
    int32_t nb = FFT_BINS;
    int32_t reserved = 0;
    
    fwrite(magic, 1, 4, fp);
    fwrite(&nc, sizeof(int32_t), 1, fp);
    fwrite(&nb, sizeof(int32_t), 1, fp);
    fwrite(&reserved, sizeof(int32_t), 1, fp);
    
    /* 写入FFT数据 */
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        fwrite(fft_result->data[ch], sizeof(complex_t), FFT_BINS, fp);
    }
    
    fclose(fp);
    printf("[INFO] FFT result saved to: %s\n", filename);
    
    return STATUS_OK;
}

status_t test_data_save_gcc(const char* filename,
                             const gcc_result_t* gcc_result)
{
    FILE* fp = fopen(filename, "wb");
    if (fp == NULL) {
        printf("[ERROR] Cannot create file: %s\n", filename);
        return STATUS_ERROR_FILE_NOT_FOUND;
    }
    
    /* 写入文件头 */
    char magic[4] = "GCC";
    int32_t np = NUM_MIC_PAIRS;
    int32_t gl = GCC_LENGTH;
    int32_t reserved = 0;
    
    fwrite(magic, 1, 4, fp);
    fwrite(&np, sizeof(int32_t), 1, fp);
    fwrite(&gl, sizeof(int32_t), 1, fp);
    fwrite(&reserved, sizeof(int32_t), 1, fp);
    
    /* 写入GCC数据 */
    for (int pair = 0; pair < NUM_MIC_PAIRS; pair++) {
        fwrite(gcc_result->data[pair], sizeof(float32_t), GCC_LENGTH, fp);
    }
    
    fclose(fp);
    printf("[INFO] GCC result saved to: %s\n", filename);
    
    return STATUS_OK;
}

status_t test_data_save_srp(const char* filename,
                             const srp_map_t* srp_result)
{
    FILE* fp = fopen(filename, "wb");
    if (fp == NULL) {
        printf("[ERROR] Cannot create file: %s\n", filename);
        return STATUS_ERROR_FILE_NOT_FOUND;
    }
    
    /* 写入文件头 */
    char magic[4] = "SRP";
    int32_t eb = SRP_ELEVATION_BINS;
    int32_t ab = SRP_AZIMUTH_BINS;
    int32_t rb = SRP_RANGE_BINS;
    
    fwrite(magic, 1, 4, fp);
    fwrite(&eb, sizeof(int32_t), 1, fp);
    fwrite(&ab, sizeof(int32_t), 1, fp);
    fwrite(&rb, sizeof(int32_t), 1, fp);
    
    /* 写入SRP数据 */
    fwrite(srp_result->data, sizeof(float32_t), 
           SRP_ELEVATION_BINS * SRP_AZIMUTH_BINS * SRP_RANGE_BINS, fp);
    
    fclose(fp);
    printf("[INFO] SRP result saved to: %s\n", filename);
    
    return STATUS_OK;
}

status_t test_data_save_as_text(const char* filename,
                                 const float32_t* data,
                                 int rows,
                                 int cols)
{
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("[ERROR] Cannot create file: %s\n", filename);
        return STATUS_ERROR_FILE_NOT_FOUND;
    }
    
    /* 写入维度信息 */
    fprintf(fp, "# Rows: %d, Cols: %d\n", rows, cols);
    
    /* 写入数据 */
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            fprintf(fp, "%.8f", data[r * cols + c]);
            if (c < cols - 1) fprintf(fp, "\t");
        }
        fprintf(fp, "\n");
    }
    
    fclose(fp);
    printf("[INFO] Text data saved to: %s\n", filename);
    
    return STATUS_OK;
}
