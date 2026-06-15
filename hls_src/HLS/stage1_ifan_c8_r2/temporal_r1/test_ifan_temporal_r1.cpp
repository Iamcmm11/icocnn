#include "ifan_temporal_r1.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

static data_t g_input[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static data_t g_weight[IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_TEMPORAL_KERNEL];
static data_t g_bias[IFAN_BRANCH_CHANNELS];
static data_t g_output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static data_t g_ref_output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];

static double resolve_max_error_tolerance() {
    const char *env_tol = std::getenv("IFAN_TEMPORAL_R1_MAX_ERR_TOL");
    if (env_tol != NULL && env_tol[0] != '\0') {
        const double parsed = std::strtod(env_tol, NULL);
        if (parsed > 0.0) {
            return parsed;
        }
    }
    return 1.0e-5;
}

static double resolve_rmse_tolerance() {
    const char *env_tol = std::getenv("IFAN_TEMPORAL_R1_RMSE_TOL");
    if (env_tol != NULL && env_tol[0] != '\0') {
        const double parsed = std::strtod(env_tol, NULL);
        if (parsed > 0.0) {
            return parsed;
        }
    }
    return 1.0e-6;
}

template <typename T>
static bool read_flat_file(const std::string &path, T *dst, std::size_t count) {
    std::ifstream f(path.c_str());
    if (!f) {
        std::cerr << "Missing file: " << path << "\n";
        return false;
    }

    std::size_t index = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        while (iss && index < count) {
            T value;
            if (!(iss >> value)) {
                break;
            }
            dst[index++] = value;
        }
    }
    if (index != count) {
        std::cerr << "Invalid value count in " << path << ": " << index
                  << " expected " << count << "\n";
        return false;
    }
    return true;
}

static std::string normalize_dir(const std::string &path) {
    if (path.empty()) {
        return path;
    }
    const char last = path[path.size() - 1];
    if (last == '/' || last == '\\') {
        return path;
    }
    return path + "/";
}

static bool file_exists(const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static std::string resolve_existing_dir(const std::string &raw_path) {
    if (raw_path.empty()) {
        return "";
    }
    const std::string normalized = normalize_dir(raw_path);
    if (file_exists(normalized + "input.txt")) {
        return normalized;
    }
    std::string prefix = "";
    for (int depth = 0; depth < 8; depth++) {
        const std::string candidate = prefix + normalized;
        if (file_exists(candidate + "input.txt")) {
            return candidate;
        }
        prefix += "../";
    }
    return "";
}

static std::string resolve_data_dir(int argc, char **argv) {
    if (argc >= 2) {
        const std::string resolved = resolve_existing_dir(argv[1]);
        if (!resolved.empty()) {
            return resolved;
        }
        return normalize_dir(argv[1]);
    }

    const char *env_dir = std::getenv("IFAN_TEMPORAL_R1_DATA_DIR");
    if (env_dir != NULL && env_dir[0] != '\0') {
        const std::string resolved = resolve_existing_dir(env_dir);
        if (!resolved.empty()) {
            return resolved;
        }
        return normalize_dir(env_dir);
    }

    const std::string candidates[] = {
        "../../../hls_testdata/temporal_r1_c8_t6/fusion0/",
        "../../../../hls_testdata/temporal_r1_c8_t6/fusion0/",
        "hls_testdata/temporal_r1_c8_t6/fusion0/",
    };
    for (std::size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        if (file_exists(candidates[i] + "input.txt")) {
            return candidates[i];
        }
    }
    return "";
}

struct DiffStats {
    double max_abs;
    double rmse;
};

static DiffStats compare_flat(const data_t *actual, const data_t *expected, std::size_t count) {
    DiffStats stats = {0.0, 0.0};
    double sq = 0.0;
    for (std::size_t i = 0; i < count; i++) {
        const double err = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
        const double abs_err = std::fabs(err);
        if (abs_err > stats.max_abs) {
            stats.max_abs = abs_err;
        }
        sq += err * err;
    }
    stats.rmse = std::sqrt(sq / static_cast<double>(count));
    return stats;
}

int main(int argc, char **argv) {
    const std::string data_dir = resolve_data_dir(argc, argv);
    const double max_err_tol = resolve_max_error_tolerance();
    const double rmse_tol = resolve_rmse_tolerance();
    if (data_dir.empty()) {
        std::cerr << "Cannot resolve temporal_r1 test data directory\n";
        return 1;
    }

    const std::size_t input_count =
        IFAN_STAGE1_T * IFAN_BRANCH_CHANNELS * IFAN_R_FULL * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1;
    const std::size_t weight_count = IFAN_BRANCH_CHANNELS * IFAN_BRANCH_CHANNELS * IFAN_TEMPORAL_KERNEL;
    const std::size_t bias_count = IFAN_BRANCH_CHANNELS;

    bool ok = true;
    ok = ok && read_flat_file(data_dir + "input.txt", &g_input[0][0][0][0][0][0], input_count);
    ok = ok && read_flat_file(data_dir + "weight.txt", &g_weight[0][0][0], weight_count);
    ok = ok && read_flat_file(data_dir + "bias.txt", &g_bias[0], bias_count);
    ok = ok && read_flat_file(data_dir + "output.txt", &g_ref_output[0][0][0][0][0][0], input_count);
    if (!ok) {
        return 1;
    }

    std::cout << "=== IFAN Temporal R1 Verification ===\n";
    std::cout << "Data dir: " << data_dir << "\n";
    std::cout << "Shape   : T=" << IFAN_STAGE1_T
              << " C=" << IFAN_BRANCH_CHANNELS
              << " R=" << IFAN_R_FULL
              << " charts=" << IFAN_CHARTS
              << " H=" << IFAN_H_R1
              << " W=" << IFAN_W_R1
              << " K=" << IFAN_TEMPORAL_KERNEL << "\n";
    std::cout << "Tol     : max_err=" << max_err_tol << " rmse=" << rmse_tol << "\n";

    ifan_temporal_r1_top(g_input, g_weight, g_bias, g_output);

    const DiffStats stats = compare_flat(&g_output[0][0][0][0][0][0], &g_ref_output[0][0][0][0][0][0], input_count);
    std::cout << "Max Error: " << stats.max_abs << "\n";
    std::cout << "RMSE: " << stats.rmse << "\n";
    if (stats.max_abs <= max_err_tol && stats.rmse <= rmse_tol) {
        std::cout << "PASS\n";
        return 0;
    }

    std::cout << "FAIL\n";
    return 1;
}
