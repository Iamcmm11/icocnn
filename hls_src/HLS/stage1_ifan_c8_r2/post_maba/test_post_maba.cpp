#include "ifan_stage1_post.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

static data_t g_input[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static PostMabaWeights g_weights;

static data_t g_channel_readout_logits[IFAN_STAGE1_T][1][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static data_t g_post_final_pool_logits[IFAN_STAGE1_T][1][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static data_t g_region_max_logits[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static int g_region_argmax_idx[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static data_t g_softargmax_input[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static data_t g_softargmax_prob[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static data_t g_coords[IFAN_STAGE1_T][IFAN_COORD_DIMS];

static data_t g_ref_channel_readout_logits[IFAN_STAGE1_T][1][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static data_t g_ref_post_final_pool_logits[IFAN_STAGE1_T][1][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static data_t g_ref_region_max_logits[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static int g_ref_region_argmax_idx[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static data_t g_ref_softargmax_input[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static data_t g_ref_softargmax_prob[IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static data_t g_ref_coords[IFAN_STAGE1_T][IFAN_COORD_DIMS];

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

static bool print_check(const std::string &name, const data_t *actual, const data_t *expected, std::size_t count, double max_tol, double rmse_tol) {
    double max_abs = 0.0;
    double sq = 0.0;
    for (std::size_t i = 0; i < count; i++) {
        const double err = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
        const double abs_err = std::fabs(err);
        if (abs_err > max_abs) {
            max_abs = abs_err;
        }
        sq += err * err;
    }
    const double rmse = std::sqrt(sq / static_cast<double>(count));
    std::cout << name << ": max_abs=" << max_abs << " rmse=" << rmse << "\n";
    return max_abs <= max_tol && rmse <= rmse_tol;
}

static bool print_check_int(const std::string &name, const int *actual, const int *expected, std::size_t count) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < count; i++) {
        if (actual[i] != expected[i]) {
            mismatches++;
        }
    }
    std::cout << name << ": mismatches=" << mismatches << "\n";
    return mismatches == 0;
}

static bool load_data_from(const std::string &data_dir) {
    const std::string post_dir = data_dir + "/post_maba";
    bool ok = true;
    ok = ok && read_flat_file(post_dir + "/tensors/pre_readout_refined_logits.txt", &g_input[0][0][0][0][0][0],
                              IFAN_STAGE1_T * IFAN_MABA_CHANNELS * IFAN_R_FULL * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1);
    ok = ok && read_flat_file(post_dir + "/weights/channel_readout_weight.txt", &g_weights.channel_readout_weight[0],
                              IFAN_MABA_CHANNELS);
    ok = ok && read_flat_file(post_dir + "/weights/channel_readout_bias.txt", &g_weights.channel_readout_bias, 1);
    ok = ok && read_flat_file(post_dir + "/tensors/clean_vertices_mask.txt", &g_weights.clean_vertices_mask[0][0],
                              IFAN_H_R1 * IFAN_W_R1);
    ok = ok && read_flat_file(post_dir + "/tensors/softargmax_indexes.txt", &g_weights.softargmax_indexes[0][0][0][0],
                              IFAN_COORD_DIMS * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1);

    ok = ok && read_flat_file(post_dir + "/tensors/channel_readout_logits.txt", &g_ref_channel_readout_logits[0][0][0][0][0][0],
                              IFAN_STAGE1_T * 1 * IFAN_R_FULL * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1);
    ok = ok && read_flat_file(post_dir + "/tensors/post_final_pool_logits.txt", &g_ref_post_final_pool_logits[0][0][0][0][0][0],
                              IFAN_STAGE1_T * 1 * IFAN_R_FULL * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1);
    ok = ok && read_flat_file(post_dir + "/tensors/region_max_logits.txt", &g_ref_region_max_logits[0][0][0][0],
                              IFAN_STAGE1_T * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1);
    ok = ok && read_flat_file(post_dir + "/tensors/region_argmax_idx.txt", &g_ref_region_argmax_idx[0][0][0][0],
                              IFAN_STAGE1_T * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1);
    ok = ok && read_flat_file(post_dir + "/tensors/softargmax_input.txt", &g_ref_softargmax_input[0][0][0][0],
                              IFAN_STAGE1_T * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1);
    ok = ok && read_flat_file(post_dir + "/tensors/softargmax_prob.txt", &g_ref_softargmax_prob[0][0][0][0],
                              IFAN_STAGE1_T * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1);
    ok = ok && read_flat_file(post_dir + "/tensors/coords.txt", &g_ref_coords[0][0],
                              IFAN_STAGE1_T * IFAN_COORD_DIMS);
    if (ok) {
        std::cout << "Loaded post-MABA data: " << data_dir << "\n";
    }
    return ok;
}

static bool load_data() {
    const std::string candidates[] = {
        "../../../hls_testdata/stage1_ifan_c8_r2/scene_1_t6",
        "hls_testdata/stage1_ifan_c8_r2/scene_1_t6"
    };
    for (std::size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        if (load_data_from(candidates[i])) {
            return true;
        }
    }
    return false;
}

int main() {
    if (!load_data()) {
        return 1;
    }

    post_maba_engine(
        g_input,
        g_weights,
        g_channel_readout_logits,
        g_post_final_pool_logits,
        g_region_max_logits,
        g_region_argmax_idx,
        g_softargmax_input,
        g_softargmax_prob,
        g_coords
    );

    bool ok = true;
    ok = ok && print_check("channel_readout_logits", &g_channel_readout_logits[0][0][0][0][0][0], &g_ref_channel_readout_logits[0][0][0][0][0][0],
                           IFAN_STAGE1_T * IFAN_R_FULL * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1, 1.0e-5, 1.0e-6);
    ok = ok && print_check("post_final_pool_logits", &g_post_final_pool_logits[0][0][0][0][0][0], &g_ref_post_final_pool_logits[0][0][0][0][0][0],
                           IFAN_STAGE1_T * IFAN_R_FULL * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1, 1.0e-5, 1.0e-6);
    ok = ok && print_check("region_max_logits", &g_region_max_logits[0][0][0][0], &g_ref_region_max_logits[0][0][0][0],
                           IFAN_STAGE1_T * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1, 1.0e-5, 1.0e-6);
    ok = ok && print_check_int("region_argmax_idx", &g_region_argmax_idx[0][0][0][0], &g_ref_region_argmax_idx[0][0][0][0],
                               IFAN_STAGE1_T * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1);
    ok = ok && print_check("softargmax_input", &g_softargmax_input[0][0][0][0], &g_ref_softargmax_input[0][0][0][0],
                           IFAN_STAGE1_T * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1, 1.0e-5, 1.0e-6);
    ok = ok && print_check("softargmax_prob", &g_softargmax_prob[0][0][0][0], &g_ref_softargmax_prob[0][0][0][0],
                           IFAN_STAGE1_T * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1, 1.0e-5, 1.0e-6);
    ok = ok && print_check("coords", &g_coords[0][0], &g_ref_coords[0][0],
                           IFAN_STAGE1_T * IFAN_COORD_DIMS, 1.0e-5, 1.0e-6);

    if (!ok) {
        std::cerr << "FAIL: post-MABA mismatch\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
