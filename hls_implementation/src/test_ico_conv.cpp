#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ico_conv_layer0.h"
#include "utils.h"

// 数据文件路径
#define DATA_DIR "../hls_testdata/layer0/"

int main() {
    printf("========================================\n");
    printf("  HLS Layer0 IcoConv 测试程序\n");
    printf("========================================\n\n");
    
    // 分配内存
    printf("[1/6] 分配内存...\n");
    data_t* input = (data_t*)malloc(INPUT_SIZE * sizeof(data_t));
    data_t* weight = (data_t*)malloc(WEIGHT_SIZE * sizeof(data_t));
    data_t* bias = (data_t*)malloc(BIAS_SIZE * sizeof(data_t));
    index_t* neighbors = (index_t*)malloc(NEIGHBORS_SIZE * sizeof(index_t));
    data_t* output = (data_t*)malloc(OUTPUT_SIZE * sizeof(data_t));
    data_t* output_ref = (data_t*)malloc(OUTPUT_SIZE * sizeof(data_t));
    
    if (!input || !weight || !bias || !neighbors || !output || !output_ref) {
        printf("错误: 内存分配失败!\n");
        return -1;
    }
    
    printf("  ✓ 输入: %d x %.2f KB\n", INPUT_SIZE, INPUT_SIZE * sizeof(data_t) / 1024.0);
    printf("  ✓ 权重: %d x %.2f KB\n", WEIGHT_SIZE, WEIGHT_SIZE * sizeof(data_t) / 1024.0);
    printf("  ✓ 输出: %d x %.2f KB\n", OUTPUT_SIZE, OUTPUT_SIZE * sizeof(data_t) / 1024.0);
    
    // 加载测试数据
    printf("\n[2/6] 加载测试数据...\n");
    
    char filepath[256];
    
    sprintf(filepath, "%s/input.txt", DATA_DIR);
    if (load_data_from_txt(filepath, input, INPUT_SIZE) != INPUT_SIZE) {
        printf("错误: 加载输入数据失败!\n");
        return -1;
    }
    
    sprintf(filepath, "%s/weight.txt", DATA_DIR);
    if (load_data_from_txt(filepath, weight, WEIGHT_SIZE) != WEIGHT_SIZE) {
        printf("错误: 加载权重数据失败!\n");
        return -1;
    }
    
    sprintf(filepath, "%s/bias.txt", DATA_DIR);
    if (load_data_from_txt(filepath, bias, BIAS_SIZE) != BIAS_SIZE) {
        printf("错误: 加载偏置数据失败!\n");
        return -1;
    }
    
    sprintf(filepath, "%s/neighbors.txt", DATA_DIR);
    if (load_indices_from_txt(filepath, neighbors, NEIGHBORS_SIZE) != NEIGHBORS_SIZE) {
        printf("错误: 加载邻居索引失败!\n");
        return -1;
    }
    
    sprintf(filepath, "%s/output.txt", DATA_DIR);
    if (load_data_from_txt(filepath, output_ref, OUTPUT_SIZE) != OUTPUT_SIZE) {
        printf("错误: 加载参考输出失败!\n");
        return -1;
    }
    
    // 打印数据统计
    printf("\n[3/6] 数据统计:\n");
    print_array_stats("  输入", input, INPUT_SIZE);
    print_array_stats("  权重", weight, WEIGHT_SIZE);
    print_array_stats("  偏置", bias, BIAS_SIZE);
    print_array_stats("  参考输出", output_ref, OUTPUT_SIZE);
    
    // 运行 HLS 实现
    printf("\n[4/6] 运行 HLS IcoConv Layer0...\n");
    
    // 初始化输出为0
    memset(output, 0, OUTPUT_SIZE * sizeof(data_t));
    
    // 执行卷积
    ico_conv_layer0(input, weight, bias, neighbors, output);
    
    printf("  ✓ 计算完成\n");
    
    // 验证结果
    printf("\n[5/6] 验证结果...\n");
    print_array_stats("  HLS输出", output, OUTPUT_SIZE);
    
    data_t max_err = compute_max_error(output_ref, output, OUTPUT_SIZE);
    data_t rel_err = compute_relative_error(output_ref, output, OUTPUT_SIZE);
    
    printf("\n  最大绝对误差: %.8f\n", max_err);
    printf("  相对误差 (RMSE): %.8f\n", rel_err);
    
    // 判断是否通过
    const data_t ERROR_THRESHOLD = 1e-4;
    
    if (max_err < ERROR_THRESHOLD) {
        printf("\n  ✓✓✓ 测试通过! ✓✓✓\n");
    } else {
        printf("\n  ✗✗✗ 测试失败! ✗✗✗\n");
        printf("  误差超过阈值 %.8f\n", ERROR_THRESHOLD);
        
        // 打印前10个不匹配的值
        printf("\n  前10个不匹配的值:\n");
        int mismatch_count = 0;
        for (int i = 0; i < OUTPUT_SIZE && mismatch_count < 10; i++) {
            data_t err = fabs(output_ref[i] - output[i]);
            if (err > ERROR_THRESHOLD) {
                printf("    [%d] ref=%.6f, hls=%.6f, err=%.6f\n",
                       i, output_ref[i], output[i], err);
                mismatch_count++;
            }
        }
    }
    
    // 保存输出
    printf("\n[6/6] 保存结果...\n");
    sprintf(filepath, "%s/output_hls.txt", DATA_DIR);
    save_data_to_txt(filepath, output, OUTPUT_SIZE);
    
    // 释放内存
    free(input);
    free(weight);
    free(bias);
    free(neighbors);
    free(output);
    free(output_ref);
    
    printf("\n========================================\n");
    printf("  测试完成!\n");
    printf("========================================\n");
    
    return (max_err < ERROR_THRESHOLD) ? 0 : 1;
}
