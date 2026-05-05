/**
 * @file audio_reader.c
 * @brief 音频数据读取模块实现
 * @author Cross3D C Implementation
 * @date 2024
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "audio_reader.h"

/*============================================================================
 * 静态变量
 *============================================================================*/
static float32_t g_hanning_window[FRAME_LENGTH];
static int g_window_initialized = 0;

/*============================================================================
 * 函数实现
 *============================================================================*/

void audio_generate_hanning_window(float32_t* window, int length)
{
    for (int i = 0; i < length; i++) {
        window[i] = 0.5f * (1.0f - cosf(TWO_PI * i / (length - 1)));
    }
}

status_t audio_read_from_file(const char* filename, 
                               float32_t** audio_data, 
                               int* total_samples)
{
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("[ERROR] Cannot open file: %s\n", filename);
        return STATUS_ERROR_FILE_NOT_FOUND;
    }
    
    /* 读取文件头 */
    char magic[4];
    int32_t num_channels, num_samples, sample_rate;
    
    fread(magic, 1, 4, fp);
    if (strncmp(magic, "AUD", 3) != 0) {
        printf("[ERROR] Invalid audio file format\n");
        fclose(fp);
        return STATUS_ERROR_INVALID_PARAM;
    }
    
    fread(&num_channels, sizeof(int32_t), 1, fp);
    fread(&num_samples, sizeof(int32_t), 1, fp);
    fread(&sample_rate, sizeof(int32_t), 1, fp);
    
    if (num_channels != NUM_CHANNELS) {
        printf("[ERROR] Channel count mismatch: expected %d, got %d\n", 
               NUM_CHANNELS, num_channels);
        fclose(fp);
        return STATUS_ERROR_INVALID_PARAM;
    }
    
    /* 读取音频数据 */
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        fread(audio_data[ch], sizeof(float32_t), num_samples, fp);
    }
    
    *total_samples = num_samples;
    fclose(fp);
    
    printf("[INFO] Loaded audio: %d channels, %d samples, %d Hz\n",
           num_channels, num_samples, sample_rate);
    
    return STATUS_OK;
}

status_t audio_get_frame(float32_t** audio_data,
                          int total_samples,
                          int frame_index,
                          audio_frame_t* frame)
{
    int start_sample = frame_index * HOP_LENGTH;
    
    /* 检查边界 */
    if (start_sample + FRAME_LENGTH > total_samples) {
        printf("[ERROR] Frame index %d out of range\n", frame_index);
        return STATUS_ERROR_INVALID_PARAM;
    }
    
    /* 复制帧数据 */
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        memcpy(frame->data[ch], 
               &audio_data[ch][start_sample], 
               FRAME_LENGTH * sizeof(float32_t));
    }
    
    frame->frame_index = frame_index;
    
    return STATUS_OK;
}

status_t audio_apply_hanning_window(audio_frame_t* frame)
{
    /* 初始化窗函数（仅首次） */
    if (!g_window_initialized) {
        audio_generate_hanning_window(g_hanning_window, FRAME_LENGTH);
        g_window_initialized = 1;
    }
    
    /* 对每个通道应用窗函数 */
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        for (int i = 0; i < FRAME_LENGTH; i++) {
            frame->data[ch][i] *= g_hanning_window[i];
        }
    }
    
    return STATUS_OK;
}

void audio_free_data(float32_t** audio_data)
{
    if (audio_data != NULL) {
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            if (audio_data[ch] != NULL) {
                free(audio_data[ch]);
            }
        }
        free(audio_data);
    }
}

void audio_print_frame_info(const audio_frame_t* frame)
{
    printf("\n=== Audio Frame Info ===\n");
    printf("Frame Index: %d\n", frame->frame_index);
    
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        float32_t sum = 0.0f;
        float32_t max_val = 0.0f;
        
        for (int i = 0; i < FRAME_LENGTH; i++) {
            sum += frame->data[ch][i] * frame->data[ch][i];
            if (fabsf(frame->data[ch][i]) > max_val) {
                max_val = fabsf(frame->data[ch][i]);
            }
        }
        
        float32_t rms = sqrtf(sum / FRAME_LENGTH);
        printf("  Channel %2d: RMS=%.6f, Max=%.6f\n", ch, rms, max_val);
    }
    printf("========================\n\n");
}
