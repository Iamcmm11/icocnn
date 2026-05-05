/**
 * @file main.c
 * @brief 主程序 - Cross3D预处理测试
 * @author Cross3D C Implementation
 * @date 2024
 * 
 * 测试流程:
 * 1. 生成模拟的12通道麦克风信号
 * 2. 分帧和加窗
 * 3. FFT变换
 * 4. GCC-PHAT计算
 * 5. SRP-Map投影
 * 6. 保存所有中间结果
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config.h"
#include "types.h"
#include "audio_reader.h"
#include "fft.h"
#include "gcc_phat.h"
#include "srp_map.h"
#include "test_data.h"

/*============================================================================
 * 输出文件路径
 *============================================================================*/
#define OUTPUT_DIR              "output"
#define AUDIO_FILE              OUTPUT_DIR "/audio_data.bin"
#define FFT_FILE                OUTPUT_DIR "/fft_result.bin"
#define GCC_FILE                OUTPUT_DIR "/gcc_result.bin"
#define SRP_FILE                OUTPUT_DIR "/srp_result.bin"
#define TAU_TABLE_FILE          OUTPUT_DIR "/tau_table.bin"
#define AUDIO_TEXT_FILE         OUTPUT_DIR "/audio_data.txt"
#define FFT_TEXT_FILE           OUTPUT_DIR "/fft_result.txt"
#define GCC_TEXT_FILE           OUTPUT_DIR "/gcc_result.txt"
#define SRP_TEXT_FILE           OUTPUT_DIR "/srp_result.txt"

/*============================================================================
 * 函数声明
 *============================================================================*/
static void print_banner(void);
static void print_config(void);
static status_t create_output_dir(void);
static void print_processing_time(const char* stage, clock_t start, clock_t end);

/*============================================================================
 * 主函数
 *============================================================================*/
int main(int argc, char* argv[])
{
    status_t status;
    clock_t start_time, end_time, total_start;
    
    total_start = clock();
    
    print_banner();
    print_config();
    
    /* 创建输出目录 */
    status = create_output_dir();
    if (status != STATUS_OK) {
        printf("[WARNING] Could not create output directory\n");
    }
    
    /*========================================================================
     * 步骤1: 初始化各模块
     *========================================================================*/
    printf("\n========== Step 1: Initialize Modules ==========\n");
    
    start_time = clock();
    
    /* 初始化FFT模块 */
    status = fft_init();
    if (status != STATUS_OK) {
        printf("[ERROR] FFT initialization failed\n");
        return -1;
    }
    
    /* 初始化GCC-PHAT模块 */
    status = gcc_phat_init();
    if (status != STATUS_OK) {
        printf("[ERROR] GCC-PHAT initialization failed\n");
        fft_cleanup();
        return -1;
    }
    
    /* 生成麦克风位置并初始化SRP模块 */
    mic_position_t mic_positions[NUM_CHANNELS];
    test_data_generate_mic_positions(mic_positions, 0.05f);  /* 5cm半径 */
    
    status = srp_map_init(mic_positions);
    if (status != STATUS_OK) {
        printf("[ERROR] SRP-Map initialization failed\n");
        gcc_phat_cleanup();
        fft_cleanup();
        return -1;
    }
    
    end_time = clock();
    print_processing_time("Initialization", start_time, end_time);
    
    /*========================================================================
     * 步骤2: 生成测试音频数据
     *========================================================================*/
    printf("\n========== Step 2: Generate Test Audio ==========\n");
    
    start_time = clock();
    
    /* 计算所需采样点数 */
    int total_samples = FRAME_LENGTH + (NUM_FRAMES - 1) * HOP_LENGTH;
    printf("Total samples needed: %d (%.2f seconds)\n", 
           total_samples, (float)total_samples / SAMPLE_RATE);
    
    /* 分配音频内存 */
    float32_t** audio_data = test_data_alloc_audio(NUM_CHANNELS, total_samples);
    if (audio_data == NULL) {
        printf("[ERROR] Memory allocation failed\n");
        srp_map_cleanup();
        gcc_phat_cleanup();
        fft_cleanup();
        return -1;
    }
    
    /* 生成模拟音频（声源角度45度） */
    float32_t source_angle = PI / 4.0f;  /* 45度 */
    status = test_data_generate_audio(audio_data, total_samples, source_angle);
    if (status != STATUS_OK) {
        printf("[ERROR] Audio generation failed\n");
        test_data_free_audio(audio_data, NUM_CHANNELS);
        srp_map_cleanup();
        gcc_phat_cleanup();
        fft_cleanup();
        return -1;
    }
    
    /* 保存音频数据 */
    test_data_save_audio(AUDIO_FILE, audio_data, NUM_CHANNELS, total_samples);
    
    end_time = clock();
    print_processing_time("Audio Generation", start_time, end_time);
    
    /*========================================================================
     * 步骤3: 处理第一帧（演示完整流程）
     *========================================================================*/
    printf("\n========== Step 3: Process Frame 0 ==========\n");
    
    /* 分配结果内存 */
    audio_frame_t* frame = (audio_frame_t*)malloc(sizeof(audio_frame_t));
    fft_result_t* fft_result = (fft_result_t*)malloc(sizeof(fft_result_t));
    gcc_result_t* gcc_result = (gcc_result_t*)malloc(sizeof(gcc_result_t));
    srp_map_t* srp_result = (srp_map_t*)malloc(sizeof(srp_map_t));
    
    if (!frame || !fft_result || !gcc_result || !srp_result) {
        printf("[ERROR] Memory allocation failed for results\n");
        test_data_free_audio(audio_data, NUM_CHANNELS);
        srp_map_cleanup();
        gcc_phat_cleanup();
        fft_cleanup();
        return -1;
    }
    
    /* 3.1 获取帧数据 */
    printf("\n--- 3.1 Get Frame ---\n");
    start_time = clock();
    
    status = audio_get_frame(audio_data, total_samples, 0, frame);
    if (status != STATUS_OK) {
        printf("[ERROR] Get frame failed\n");
        goto cleanup;
    }
    
    end_time = clock();
    print_processing_time("Get Frame", start_time, end_time);
    
    /* 3.2 应用汉宁窗 */
    printf("\n--- 3.2 Apply Hanning Window ---\n");
    start_time = clock();
    
    status = audio_apply_hanning_window(frame);
    if (status != STATUS_OK) {
        printf("[ERROR] Windowing failed\n");
        goto cleanup;
    }
    
    end_time = clock();
    print_processing_time("Windowing", start_time, end_time);
    
#if DEBUG_PRINT
    audio_print_frame_info(frame);
#endif
    
    /* 3.3 FFT变换 */
    printf("\n--- 3.3 FFT Transform ---\n");
    start_time = clock();
    
    status = fft_execute_real(frame, fft_result);
    if (status != STATUS_OK) {
        printf("[ERROR] FFT failed\n");
        goto cleanup;
    }
    
    end_time = clock();
    print_processing_time("FFT", start_time, end_time);
    
#if DEBUG_PRINT
    fft_print_result(fft_result, 0, 10);  /* 打印通道0的前10个频点 */
#endif
    
    /* 保存FFT结果 */
#if SAVE_INTERMEDIATE
    test_data_save_fft(FFT_FILE, fft_result);
#endif
    
    /* 3.4 GCC-PHAT计算 */
    printf("\n--- 3.4 GCC-PHAT Calculation ---\n");
    start_time = clock();
    
    status = gcc_phat_compute_all(fft_result, gcc_result);
    if (status != STATUS_OK) {
        printf("[ERROR] GCC-PHAT failed\n");
        goto cleanup;
    }
    
    end_time = clock();
    print_processing_time("GCC-PHAT", start_time, end_time);
    
#if DEBUG_PRINT
    gcc_phat_print_result(gcc_result, 0, 20);  /* 打印第一对麦克风的结果 */
#endif
    
    /* 保存GCC结果 */
#if SAVE_INTERMEDIATE
    test_data_save_gcc(GCC_FILE, gcc_result);
#endif
    
    /* 3.5 SRP-Map投影 */
    printf("\n--- 3.5 SRP-Map Projection ---\n");
    start_time = clock();
    
    status = srp_map_compute(gcc_result, srp_result);
    if (status != STATUS_OK) {
        printf("[ERROR] SRP-Map failed\n");
        goto cleanup;
    }
    
    end_time = clock();
    print_processing_time("SRP-Map", start_time, end_time);
    
#if DEBUG_PRINT
    srp_map_print_result(srp_result);
#endif
    
    /* 保存SRP结果 */
#if SAVE_INTERMEDIATE
    test_data_save_srp(SRP_FILE, srp_result);
    srp_map_save_tau_table(TAU_TABLE_FILE);
#endif
    
    /*========================================================================
     * 步骤4: 保存文本格式结果（便于查看）
     *========================================================================*/
    printf("\n========== Step 4: Save Text Results ==========\n");
    
    /* 保存音频数据的一部分为文本 */
    test_data_save_as_text(AUDIO_TEXT_FILE, frame->data[0], 1, 100);
    
    /* 保存SRP结果为文本 */
    test_data_save_as_text(SRP_TEXT_FILE, (float32_t*)srp_result->data,
                           SRP_ELEVATION_BINS, SRP_AZIMUTH_BINS * SRP_RANGE_BINS);
    
    /*========================================================================
     * 步骤5: 处理所有帧（性能测试）
     *========================================================================*/
    printf("\n========== Step 5: Process All Frames ==========\n");
    
    start_time = clock();
    
    int processed_frames = 0;
    for (int f = 0; f < NUM_FRAMES; f++) {
        /* 获取帧 */
        status = audio_get_frame(audio_data, total_samples, f, frame);
        if (status != STATUS_OK) break;
        
        /* 加窗 */
        audio_apply_hanning_window(frame);
        
        /* FFT */
        fft_execute_real(frame, fft_result);
        
        /* GCC-PHAT */
        gcc_phat_compute_all(fft_result, gcc_result);
        
        /* SRP-Map */
        srp_map_compute(gcc_result, srp_result);
        
        processed_frames++;
        
        /* 进度显示 */
        if ((f + 1) % 20 == 0 || f == NUM_FRAMES - 1) {
            printf("  Processed %d/%d frames\n", f + 1, NUM_FRAMES);
        }
    }
    
    end_time = clock();
    
    float32_t total_time = (float32_t)(end_time - start_time) / CLOCKS_PER_SEC;
    float32_t fps = processed_frames / total_time;
    
    printf("\nAll Frames Processing:\n");
    printf("  Frames: %d\n", processed_frames);
    printf("  Total Time: %.3f seconds\n", total_time);
    printf("  FPS: %.2f frames/second\n", fps);
    printf("  Time per frame: %.3f ms\n", 1000.0f / fps);
    
    /*========================================================================
     * 完成
     *========================================================================*/
    printf("\n========== Processing Complete ==========\n");
    
    float32_t total_elapsed = (float32_t)(clock() - total_start) / CLOCKS_PER_SEC;
    printf("Total elapsed time: %.3f seconds\n", total_elapsed);
    
    printf("\nOutput files:\n");
    printf("  Audio data: %s\n", AUDIO_FILE);
    printf("  FFT result: %s\n", FFT_FILE);
    printf("  GCC result: %s\n", GCC_FILE);
    printf("  SRP result: %s\n", SRP_FILE);
    printf("  Tau table:  %s\n", TAU_TABLE_FILE);
    
    status = STATUS_OK;
    
cleanup:
    /* 释放内存 */
    free(frame);
    free(fft_result);
    free(gcc_result);
    free(srp_result);
    test_data_free_audio(audio_data, NUM_CHANNELS);
    
    /* 清理模块 */
    srp_map_cleanup();
    gcc_phat_cleanup();
    fft_cleanup();
    
    printf("\n[INFO] Cleanup complete\n");
    
    return (status == STATUS_OK) ? 0 : -1;
}

/*============================================================================
 * 辅助函数实现
 *============================================================================*/

static void print_banner(void)
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           Cross3D Preprocessing - C Implementation           ║\n");
    printf("║                                                              ║\n");
    printf("║  Modules: Audio Reader -> FFT -> GCC-PHAT -> SRP-Map         ║\n");
    printf("║  Target:  Zynq PS (ARM) / HLS Verification                   ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

static void print_config(void)
{
    printf("Configuration:\n");
    printf("  Sample Rate:     %d Hz\n", SAMPLE_RATE);
    printf("  Channels:        %d\n", NUM_CHANNELS);
    printf("  Frame Length:    %d samples\n", FRAME_LENGTH);
    printf("  Hop Length:      %d samples\n", HOP_LENGTH);
    printf("  FFT Size:        %d\n", FFT_SIZE);
    printf("  FFT Bins:        %d\n", FFT_BINS);
    printf("  Mic Pairs:       %d\n", NUM_MIC_PAIRS);
    printf("  SRP Grid:        %d x %d x %d\n", 
           SRP_ELEVATION_BINS, SRP_AZIMUTH_BINS, SRP_RANGE_BINS);
    printf("\n");
}

static status_t create_output_dir(void)
{
#ifdef _WIN32
    system("if not exist output mkdir output");
#else
    system("mkdir -p output");
#endif
    return STATUS_OK;
}

static void print_processing_time(const char* stage, clock_t start, clock_t end)
{
    float32_t elapsed = (float32_t)(end - start) / CLOCKS_PER_SEC * 1000.0f;
    printf("[TIME] %s: %.3f ms\n", stage, elapsed);
}
