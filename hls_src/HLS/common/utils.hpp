#ifndef UTILS_HPP
#define UTILS_HPP

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

// 读取 .txt 文件中的浮点数据（一行一个值）
inline std::vector<float> read_txt_data(const std::string& filename) {
    std::vector<float> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return data;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // 跳过注释行
        if (line.empty() || line[0] == '/' || line[0] == '#') {
            continue;
        }
        
        try {
            float val = std::stof(line);
            data.push_back(val);
        } catch (...) {
            continue;
        }
    }
    
    file.close();
    return data;
}

// 读取整型数据
inline std::vector<int> read_txt_data_int(const std::string& filename) {
    std::vector<int> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return data;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '/' || line[0] == '#') {
            continue;
        }
        
        try {
            int val = std::stoi(line);
            data.push_back(val);
        } catch (...) {
            continue;
        }
    }
    
    file.close();
    return data;
}

// 计算两个数组的最大误差
inline float max_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        std::cerr << "Error: Size mismatch in max_error" << std::endl;
        return -1.0f;
    }
    
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float err = std::abs(a[i] - b[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

// 计算均方根误差
inline float rmse(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        std::cerr << "Error: Size mismatch in rmse" << std::endl;
        return -1.0f;
    }
    
    float sum_sq = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / a.size());
}

// 打印数组统计信息
inline void print_stats(const std::string& name, const std::vector<float>& data) {
    if (data.empty()) {
        std::cout << name << ": empty" << std::endl;
        return;
    }
    
    float min_val = data[0], max_val = data[0], sum = 0.0f;
    for (float val : data) {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }
    float mean = sum / data.size();
    
    std::cout << name << ": size=" << data.size() 
              << ", min=" << min_val << ", max=" << max_val 
              << ", mean=" << mean << std::endl;
}

#endif // UTILS_HPP
