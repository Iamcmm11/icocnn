/**
 * @file srp_map.h
 * @brief SRP-Map投影模块头文件
 * @author Cross3D C Implementation
 * @date 2024
 * 
 * 该模块实现SRP(Steered Response Power)空间功率谱投影，
 * 将GCC结果映射到三维空间网格。
 */

#ifndef SRP_MAP_H
#define SRP_MAP_H

#include "types.h"
#include "config.h"

/*============================================================================
 * 函数声明
 *============================================================================*/

/**
 * @brief 初始化SRP-Map模块
 * @param mic_positions 麦克风位置数组
 * @return 状态码
 */
status_t srp_map_init(const mic_position_t* mic_positions);

/**
 * @brief 释放SRP-Map模块资源
 */
void srp_map_cleanup(void);

/**
 * @brief 计算SRP-Map
 * @param gcc_result 输入GCC结果
 * @param srp_result 输出SRP-Map结果
 * @return 状态码
 */
status_t srp_map_compute(const gcc_result_t* gcc_result, 
                          srp_map_t* srp_result);

/**
 * @brief 预计算Tau Table
 * @param mic_positions 麦克风位置数组
 * @param tau_table 输出Tau表
 * @return 状态码
 */
status_t srp_map_compute_tau_table(const mic_position_t* mic_positions,
                                    tau_table_t* tau_table);

/**
 * @brief 计算两个麦克风之间的理论时延
 * @param mic1_pos 麦克风1位置
 * @param mic2_pos 麦克风2位置
 * @param source_pos 声源位置 (球坐标: elevation, azimuth, range)
 * @return 时延采样点数
 */
int srp_map_compute_tau(const mic_position_t* mic1_pos,
                         const mic_position_t* mic2_pos,
                         float32_t elevation,
                         float32_t azimuth,
                         float32_t range);

/**
 * @brief 获取Tau Table
 * @return Tau Table指针
 */
const tau_table_t* srp_map_get_tau_table(void);

/**
 * @brief 保存Tau Table到文件
 * @param filename 文件路径
 * @return 状态码
 */
status_t srp_map_save_tau_table(const char* filename);

/**
 * @brief 从文件加载Tau Table
 * @param filename 文件路径
 * @return 状态码
 */
status_t srp_map_load_tau_table(const char* filename);

/**
 * @brief 打印SRP-Map结果信息（调试用）
 * @param srp_result SRP-Map结果
 */
void srp_map_print_result(const srp_map_t* srp_result);

#endif /* SRP_MAP_H */
