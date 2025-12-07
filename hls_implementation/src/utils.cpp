#include "utils.h"
#include <math.h>
#include <string.h>

/**
 * @brief 从文本文件加载浮点数据
 */
int load_data_from_txt(const char* filename, data_t* data, int size) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return -1;
    }
    
    // 跳过注释行
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] != '#') {
            fseek(fp, -(long)strlen(line), SEEK_CUR);
            break;
        }
    }
    
    // 读取数据
    int count = 0;
    while (count < size && fscanf(fp, "%f", &data[count]) == 1) {
        count++;
    }
    
    fclose(fp);
    
    printf("Loaded %d values from %s\n", count, filename);
    return count;
}

/**
 * @brief 从文本文件加载整数索引
 */
int load_indices_from_txt(const char* filename, index_t* data, int size) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return -1;
    }
    
    // 跳过注释行
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] != '#') {
            fseek(fp, -(long)strlen(line), SEEK_CUR);
            break;
        }
    }
    
    // 读取数据
    int count = 0;
    while (count < size && fscanf(fp, "%d", &data[count]) == 1) {
        count++;
    }
    
    fclose(fp);
    
    printf("Loaded %d indices from %s\n", count, filename);
    return count;
}

/**
 * @brief 保存数据到文本文件
 */
int save_data_to_txt(const char* filename, const data_t* data, int size) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot create file %s\n", filename);
        return -1;
    }
    
    fprintf(fp, "# Size: %d\n", size);
    for (int i = 0; i < size; i++) {
        fprintf(fp, "%.8f\n", data[i]);
    }
    
    fclose(fp);
    printf("Saved %d values to %s\n", size, filename);
    return 0;
}

/**
 * @brief 计算最大绝对误差
 */
data_t compute_max_error(const data_t* ref, const data_t* test, int size) {
    data_t max_err = 0.0f;
    for (int i = 0; i < size; i++) {
        data_t err = fabs(ref[i] - test[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

/**
 * @brief 计算相对误差 (RMSE)
 */
data_t compute_relative_error(const data_t* ref, const data_t* test, int size) {
    data_t sum_sq_err = 0.0f;
    data_t sum_sq_ref = 0.0f;
    
    for (int i = 0; i < size; i++) {
        data_t err = ref[i] - test[i];
        sum_sq_err += err * err;
        sum_sq_ref += ref[i] * ref[i];
    }
    
    if (sum_sq_ref < 1e-10f) {
        return 0.0f;
    }
    
    return sqrt(sum_sq_err / sum_sq_ref);
}

/**
 * @brief 打印数组统计信息
 */
void print_array_stats(const char* name, const data_t* data, int size) {
    data_t min_val = data[0];
    data_t max_val = data[0];
    data_t sum = 0.0f;
    
    for (int i = 0; i < size; i++) {
        data_t val = data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }
    
    data_t mean = sum / size;
    
    printf("%s: size=%d, min=%.4f, max=%.4f, mean=%.4f\n",
           name, size, min_val, max_val, mean);
}
