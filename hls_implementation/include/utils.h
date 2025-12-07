#ifndef UTILS_H
#define UTILS_H

#include "ico_types.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief 从文本文件加载数据
 * 
 * @param filename 文件路径
 * @param data     输出数据指针
 * @param size     期望的数据大小
 * @return 实际读取的数据数量
 */
int load_data_from_txt(const char* filename, data_t* data, int size);

/**
 * @brief 从文本文件加载整数索引
 */
int load_indices_from_txt(const char* filename, index_t* data, int size);

/**
 * @brief 保存数据到文本文件
 */
int save_data_to_txt(const char* filename, const data_t* data, int size);

/**
 * @brief 计算两个数组的误差
 * 
 * @return 最大绝对误差
 */
data_t compute_max_error(const data_t* ref, const data_t* test, int size);

/**
 * @brief 计算相对误差
 */
data_t compute_relative_error(const data_t* ref, const data_t* test, int size);

/**
 * @brief 打印数组统计信息
 */
void print_array_stats(const char* name, const data_t* data, int size);

#endif // UTILS_H
