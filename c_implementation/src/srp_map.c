/**
 * @file srp_map.c
 * @brief SRP-Map投影模块实现
 * @author Cross3D C Implementation
 * @date 2024
 * 
 * 实现SRP(Steered Response Power)空间功率谱投影
 * 将GCC结果映射到三维空间网格
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "srp_map.h"
#include "gcc_phat.h"

/*============================================================================
 * 静态变量
 *============================================================================*/
static tau_table_t g_tau_table;
static mic_position_t g_mic_positions[NUM_CHANNELS];
static int g_srp_initialized = 0;

/* 空间网格参数 */
static float32_t g_elevation_range[2] = {0.0f, PI};           /* 俯仰角范围 */
static float32_t g_azimuth_range[2] = {-PI, PI};              /* 方位角范围 */
static float32_t g_range_values[SRP_RANGE_BINS] = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f};

/*============================================================================
 * 辅助函数
 *============================================================================*/

/**
 * @brief 计算两点之间的欧氏距离
 */
static float32_t compute_distance(const mic_position_t* p1, 
                                   float32_t x, float32_t y, float32_t z)
{
    float32_t dx = p1->x - x;
    float32_t dy = p1->y - y;
    float32_t dz = p1->z - z;
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

/**
 * @brief 球坐标转笛卡尔坐标
 */
static void sph2cart(float32_t elevation, float32_t azimuth, float32_t range,
                      float32_t* x, float32_t* y, float32_t* z)
{
    *x = range * sinf(elevation) * cosf(azimuth);
    *y = range * sinf(elevation) * sinf(azimuth);
    *z = range * cosf(elevation);
}

/*============================================================================
 * 函数实现
 *============================================================================*/

int srp_map_compute_tau(const mic_position_t* mic1_pos,
                         const mic_position_t* mic2_pos,
                         float32_t elevation,
                         float32_t azimuth,
                         float32_t range)
{
    /* 计算声源位置（笛卡尔坐标） */
    float32_t src_x, src_y, src_z;
    sph2cart(elevation, azimuth, range, &src_x, &src_y, &src_z);
    
    /* 计算声源到两个麦克风的距离 */
    float32_t d1 = compute_distance(mic1_pos, src_x, src_y, src_z);
    float32_t d2 = compute_distance(mic2_pos, src_x, src_y, src_z);
    
    /* 计算时延（采样点数） */
    float32_t tau_seconds = (d1 - d2) / SPEED_OF_SOUND;
    int tau_samples = (int)roundf(tau_seconds * SAMPLE_RATE);
    
    return tau_samples;
}

status_t srp_map_compute_tau_table(const mic_position_t* mic_positions,
                                    tau_table_t* tau_table)
{
    float32_t elev_step = (g_elevation_range[1] - g_elevation_range[0]) / 
                          (SRP_ELEVATION_BINS - 1);
    float32_t azim_step = (g_azimuth_range[1] - g_azimuth_range[0]) / 
                          (SRP_AZIMUTH_BINS - 1);
    
    printf("[INFO] Computing Tau Table...\n");
    printf("  Elevation: %.2f to %.2f rad, %d bins\n", 
           g_elevation_range[0], g_elevation_range[1], SRP_ELEVATION_BINS);
    printf("  Azimuth: %.2f to %.2f rad, %d bins\n", 
           g_azimuth_range[0], g_azimuth_range[1], SRP_AZIMUTH_BINS);
    printf("  Range: %d bins\n", SRP_RANGE_BINS);
    
    /* 遍历所有麦克风对 */
    for (int pair = 0; pair < NUM_MIC_PAIRS; pair++) {
        int mic1, mic2;
        gcc_phat_get_mic_pair(pair, &mic1, &mic2);
        
        int table_idx = 0;
        
        /* 遍历所有空间网格点 */
        for (int e = 0; e < SRP_ELEVATION_BINS; e++) {
            float32_t elevation = g_elevation_range[0] + e * elev_step;
            
            for (int a = 0; a < SRP_AZIMUTH_BINS; a++) {
                float32_t azimuth = g_azimuth_range[0] + a * azim_step;
                
                for (int r = 0; r < SRP_RANGE_BINS; r++) {
                    float32_t range = g_range_values[r];
                    
                    /* 计算时延 */
                    int tau = srp_map_compute_tau(
                        &mic_positions[mic1],
                        &mic_positions[mic2],
                        elevation, azimuth, range
                    );
                    
                    /* 转换为GCC数组索引（考虑fftshift后零时延在中心） */
                    int gcc_idx = GCC_LENGTH / 2 + tau;
                    
                    /* 边界检查 */
                    if (gcc_idx < 0) gcc_idx = 0;
                    if (gcc_idx >= GCC_LENGTH) gcc_idx = GCC_LENGTH - 1;
                    
                    tau_table->tau_indices[pair][table_idx] = gcc_idx;
                    table_idx++;
                }
            }
        }
    }
    
    printf("[INFO] Tau Table computed: %d pairs x %d grid points\n", 
           NUM_MIC_PAIRS, TAU_TABLE_SIZE);
    
    return STATUS_OK;
}

status_t srp_map_init(const mic_position_t* mic_positions)
{
    if (g_srp_initialized) {
        return STATUS_OK;
    }
    
    /* 保存麦克风位置 */
    memcpy(g_mic_positions, mic_positions, 
           NUM_CHANNELS * sizeof(mic_position_t));
    
    /* 计算Tau Table */
    status_t status = srp_map_compute_tau_table(mic_positions, &g_tau_table);
    if (status != STATUS_OK) {
        return status;
    }
    
    g_srp_initialized = 1;
    printf("[INFO] SRP-Map module initialized\n");
    
    return STATUS_OK;
}

void srp_map_cleanup(void)
{
    g_srp_initialized = 0;
}

status_t srp_map_compute(const gcc_result_t* gcc_result, 
                          srp_map_t* srp_result)
{
    if (!g_srp_initialized) {
        printf("[ERROR] SRP-Map module not initialized\n");
        return STATUS_ERROR_INVALID_PARAM;
    }
    
    /* 清零输出 */
    memset(srp_result, 0, sizeof(srp_map_t));
    
    /* 遍历所有空间网格点 */
    for (int e = 0; e < SRP_ELEVATION_BINS; e++) {
        for (int a = 0; a < SRP_AZIMUTH_BINS; a++) {
            for (int r = 0; r < SRP_RANGE_BINS; r++) {
                float32_t sum = 0.0f;
                int grid_idx = e * SRP_AZIMUTH_BINS * SRP_RANGE_BINS + 
                               a * SRP_RANGE_BINS + r;
                
                /* 累加所有麦克风对的GCC值 */
                for (int pair = 0; pair < NUM_MIC_PAIRS; pair++) {
                    int gcc_idx = g_tau_table.tau_indices[pair][grid_idx];
                    sum += gcc_result->data[pair][gcc_idx];
                }
                
                srp_result->data[e][a][r] = sum;
            }
        }
    }
    
    return STATUS_OK;
}

const tau_table_t* srp_map_get_tau_table(void)
{
    return &g_tau_table;
}

status_t srp_map_save_tau_table(const char* filename)
{
    FILE* fp = fopen(filename, "wb");
    if (fp == NULL) {
        printf("[ERROR] Cannot create file: %s\n", filename);
        return STATUS_ERROR_FILE_NOT_FOUND;
    }
    
    /* 写入文件头 */
    char magic[4] = "TAU";
    int32_t num_pairs = NUM_MIC_PAIRS;
    int32_t table_size = TAU_TABLE_SIZE;
    int32_t reserved = 0;
    
    fwrite(magic, 1, 4, fp);
    fwrite(&num_pairs, sizeof(int32_t), 1, fp);
    fwrite(&table_size, sizeof(int32_t), 1, fp);
    fwrite(&reserved, sizeof(int32_t), 1, fp);
    
    /* 写入Tau Table数据 */
    for (int pair = 0; pair < NUM_MIC_PAIRS; pair++) {
        fwrite(g_tau_table.tau_indices[pair], sizeof(int), TAU_TABLE_SIZE, fp);
    }
    
    fclose(fp);
    printf("[INFO] Tau Table saved to: %s\n", filename);
    
    return STATUS_OK;
}

status_t srp_map_load_tau_table(const char* filename)
{
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("[ERROR] Cannot open file: %s\n", filename);
        return STATUS_ERROR_FILE_NOT_FOUND;
    }
    
    /* 读取文件头 */
    char magic[4];
    int32_t num_pairs, table_size, reserved;
    
    fread(magic, 1, 4, fp);
    if (strncmp(magic, "TAU", 3) != 0) {
        printf("[ERROR] Invalid Tau Table file format\n");
        fclose(fp);
        return STATUS_ERROR_INVALID_PARAM;
    }
    
    fread(&num_pairs, sizeof(int32_t), 1, fp);
    fread(&table_size, sizeof(int32_t), 1, fp);
    fread(&reserved, sizeof(int32_t), 1, fp);
    
    /* 读取Tau Table数据 */
    for (int pair = 0; pair < NUM_MIC_PAIRS; pair++) {
        fread(g_tau_table.tau_indices[pair], sizeof(int), TAU_TABLE_SIZE, fp);
    }
    
    fclose(fp);
    printf("[INFO] Tau Table loaded from: %s\n", filename);
    
    return STATUS_OK;
}

void srp_map_print_result(const srp_map_t* srp_result)
{
    printf("\n=== SRP-Map Result ===\n");
    printf("Shape: [%d, %d, %d] (elevation, azimuth, range)\n",
           SRP_ELEVATION_BINS, SRP_AZIMUTH_BINS, SRP_RANGE_BINS);
    
    /* 找到最大值位置 */
    float32_t max_val = -1e10f;
    int max_e = 0, max_a = 0, max_r = 0;
    
    for (int e = 0; e < SRP_ELEVATION_BINS; e++) {
        for (int a = 0; a < SRP_AZIMUTH_BINS; a++) {
            for (int r = 0; r < SRP_RANGE_BINS; r++) {
                if (srp_result->data[e][a][r] > max_val) {
                    max_val = srp_result->data[e][a][r];
                    max_e = e;
                    max_a = a;
                    max_r = r;
                }
            }
        }
    }
    
    float32_t elev_step = (g_elevation_range[1] - g_elevation_range[0]) / 
                          (SRP_ELEVATION_BINS - 1);
    float32_t azim_step = (g_azimuth_range[1] - g_azimuth_range[0]) / 
                          (SRP_AZIMUTH_BINS - 1);
    
    float32_t est_elevation = g_elevation_range[0] + max_e * elev_step;
    float32_t est_azimuth = g_azimuth_range[0] + max_a * azim_step;
    float32_t est_range = g_range_values[max_r];
    
    printf("\nPeak Location:\n");
    printf("  Grid Index: [%d, %d, %d]\n", max_e, max_a, max_r);
    printf("  Elevation: %.2f rad (%.2f deg)\n", 
           est_elevation, est_elevation * 180.0f / PI);
    printf("  Azimuth: %.2f rad (%.2f deg)\n", 
           est_azimuth, est_azimuth * 180.0f / PI);
    printf("  Range: %.2f m\n", est_range);
    printf("  Value: %.4f\n", max_val);
    
    /* 打印一个切片 */
    printf("\nSRP-Map slice at range index %d:\n", max_r);
    printf("Elev\\Azim\t");
    for (int a = 0; a < SRP_AZIMUTH_BINS; a++) {
        printf("%.1f\t", (g_azimuth_range[0] + a * azim_step) * 180.0f / PI);
    }
    printf("\n");
    
    for (int e = 0; e < SRP_ELEVATION_BINS; e++) {
        printf("%.1f\t\t", (g_elevation_range[0] + e * elev_step) * 180.0f / PI);
        for (int a = 0; a < SRP_AZIMUTH_BINS; a++) {
            printf("%.2f\t", srp_result->data[e][a][max_r]);
        }
        printf("\n");
    }
    
    printf("======================\n\n");
}
