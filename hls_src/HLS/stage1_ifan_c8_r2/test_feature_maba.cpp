#include "ifan_stage1_maba.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

static data_t g_input[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static FeatureMabaWeights g_weights;

static data_t g_input_positions[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_CHANNELS];
static data_t g_in_proj_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_dw_conv_input[IFAN_MABA_POSITIONS][IFAN_MABA_D_MODEL][IFAN_STAGE1_T];
static data_t g_dw_conv_input_padded[IFAN_MABA_POSITIONS][IFAN_MABA_D_MODEL][IFAN_STAGE1_T + IFAN_MABA_CONV_KERNEL - 1];
static data_t g_dw_conv_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_mix_pre_norm[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_mix_norm_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_state_input[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM * 2];
static data_t g_q[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM];
static data_t g_gate[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM];
static data_t g_alpha[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM];
static data_t g_state_sequence[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM];
static data_t g_state_back_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_refined_pre_dropout[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_delta_flat[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_CHANNELS];
static data_t g_delta[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static data_t g_output[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];

static data_t g_ref_input_positions[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_CHANNELS];
static data_t g_ref_in_proj_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_ref_dw_conv_input[IFAN_MABA_POSITIONS][IFAN_MABA_D_MODEL][IFAN_STAGE1_T];
static data_t g_ref_dw_conv_input_padded[IFAN_MABA_POSITIONS][IFAN_MABA_D_MODEL][IFAN_STAGE1_T + IFAN_MABA_CONV_KERNEL - 1];
static data_t g_ref_dw_conv_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_ref_mix_pre_norm[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_ref_mix_norm_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_ref_state_input[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM * 2];
static data_t g_ref_q[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM];
static data_t g_ref_gate[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM];
static data_t g_ref_alpha[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM];
static data_t g_ref_state_sequence[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_STATE_DIM];
static data_t g_ref_state_back_out[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_ref_refined_pre_dropout[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_D_MODEL];
static data_t g_ref_delta_flat[IFAN_MABA_POSITIONS][IFAN_STAGE1_T][IFAN_MABA_CHANNELS];
static data_t g_ref_delta[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];
static data_t g_ref_output[IFAN_STAGE1_T][IFAN_MABA_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1];

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

struct DiffStats {
    double max_abs;
    double rmse;
    std::size_t count;
};

static DiffStats compare_flat(const data_t *actual, const data_t *expected, std::size_t count) {
    DiffStats stats = {0.0, 0.0, count};
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

static bool print_check(const std::string &name, const data_t *actual, const data_t *expected, std::size_t count) {
    const DiffStats stats = compare_flat(actual, expected, count);
    std::cout << name << ": max_abs=" << stats.max_abs << " rmse=" << stats.rmse << "\n";
    return stats.max_abs <= 1.0e-5 && stats.rmse <= 1.0e-6;
}

static bool load_data_from(const std::string &data_dir) {
    const std::string maba_dir = data_dir + "/maba";
    bool ok = true;
    ok = ok && read_flat_file(data_dir + "/final_head_logits.txt", &g_input[0][0][0][0][0][0],
                              IFAN_STAGE1_T * IFAN_MABA_CHANNELS * IFAN_R_FULL * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1);

    ok = ok && read_flat_file(maba_dir + "/weights/in_proj_weight.txt", &g_weights.in_proj_weight[0][0],
                              IFAN_MABA_D_MODEL * IFAN_MABA_CHANNELS);
    ok = ok && read_flat_file(maba_dir + "/weights/in_proj_bias.txt", &g_weights.in_proj_bias[0],
                              IFAN_MABA_D_MODEL);
    ok = ok && read_flat_file(maba_dir + "/weights/dw_conv_weight.txt", &g_weights.dw_conv_weight[0][0][0],
                              IFAN_MABA_D_MODEL * 1 * IFAN_MABA_CONV_KERNEL);
    ok = ok && read_flat_file(maba_dir + "/weights/dw_conv_bias.txt", &g_weights.dw_conv_bias[0],
                              IFAN_MABA_D_MODEL);
    ok = ok && read_flat_file(maba_dir + "/weights/mix_norm_weight.txt", &g_weights.mix_norm_weight[0],
                              IFAN_MABA_D_MODEL);
    ok = ok && read_flat_file(maba_dir + "/weights/mix_norm_bias.txt", &g_weights.mix_norm_bias[0],
                              IFAN_MABA_D_MODEL);
    ok = ok && read_flat_file(maba_dir + "/weights/state_proj_weight.txt", &g_weights.state_proj_weight[0][0],
                              (IFAN_MABA_STATE_DIM * 2) * IFAN_MABA_D_MODEL);
    ok = ok && read_flat_file(maba_dir + "/weights/state_proj_bias.txt", &g_weights.state_proj_bias[0],
                              IFAN_MABA_STATE_DIM * 2);
    ok = ok && read_flat_file(maba_dir + "/weights/state_back_weight.txt", &g_weights.state_back_weight[0][0],
                              IFAN_MABA_D_MODEL * IFAN_MABA_STATE_DIM);
    ok = ok && read_flat_file(maba_dir + "/weights/state_back_bias.txt", &g_weights.state_back_bias[0],
                              IFAN_MABA_D_MODEL);
    ok = ok && read_flat_file(maba_dir + "/weights/out_proj_weight.txt", &g_weights.out_proj_weight[0][0],
                              IFAN_MABA_CHANNELS * IFAN_MABA_D_MODEL);
    ok = ok && read_flat_file(maba_dir + "/weights/out_proj_bias.txt", &g_weights.out_proj_bias[0],
                              IFAN_MABA_CHANNELS);

    ok = ok && read_flat_file(maba_dir + "/tensors/input_positions.txt", &g_ref_input_positions[0][0][0],
                              IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_CHANNELS);
    ok = ok && read_flat_file(maba_dir + "/tensors/in_proj_out.txt", &g_ref_in_proj_out[0][0][0],
                              IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_D_MODEL);
    ok = ok && read_flat_file(maba_dir + "/tensors/dw_conv_input.txt", &g_ref_dw_conv_input[0][0][0],
                              IFAN_MABA_POSITIONS * IFAN_MABA_D_MODEL * IFAN_STAGE1_T);
    ok = ok && read_flat_file(maba_dir + "/tensors/dw_conv_input_padded.txt", &g_ref_dw_conv_input_padded[0][0][0],
                              IFAN_MABA_POSITIONS * IFAN_MABA_D_MODEL * (IFAN_STAGE1_T + IFAN_MABA_CONV_KERNEL - 1));
    ok = ok && read_flat_file(maba_dir + "/tensors/dw_conv_out.txt", &g_ref_dw_conv_out[0][0][0],
                              IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_D_MODEL);
    ok = ok && read_flat_file(maba_dir + "/tensors/mix_pre_norm.txt", &g_ref_mix_pre_norm[0][0][0],
                              IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_D_MODEL);
    ok = ok && read_flat_file(maba_dir + "/tensors/mix_norm_out.txt", &g_ref_mix_norm_out[0][0][0],
                              IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_D_MODEL);
    ok = ok && read_flat_file(maba_dir + "/tensors/state_input.txt", &g_ref_state_input[0][0][0],
                              IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_STATE_DIM * 2);
    ok = ok && read_flat_file(maba_dir + "/tensors/q.txt", &g_ref_q[0][0][0],
                              IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_STATE_DIM);
    ok = ok && read_flat_file(maba_dir + "/tensors/gate.txt", &g_ref_gate[0][0][0],
                              IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_STATE_DIM);
    ok = ok && read_flat_file(maba_dir + "/tensors/alpha.txt", &g_ref_alpha[0][0][0],
                              IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_STATE_DIM);
    ok = ok && read_flat_file(maba_dir + "/tensors/state_sequence.txt", &g_ref_state_sequence[0][0][0],
                              IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_STATE_DIM);
    ok = ok && read_flat_file(maba_dir + "/tensors/state_back_out.txt", &g_ref_state_back_out[0][0][0],
                              IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_D_MODEL);
    ok = ok && read_flat_file(maba_dir + "/tensors/refined_pre_dropout.txt", &g_ref_refined_pre_dropout[0][0][0],
                              IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_D_MODEL);
    ok = ok && read_flat_file(maba_dir + "/tensors/delta_flat.txt", &g_ref_delta_flat[0][0][0],
                              IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_CHANNELS);
    ok = ok && read_flat_file(maba_dir + "/tensors/delta.txt", &g_ref_delta[0][0][0][0][0][0],
                              IFAN_STAGE1_T * IFAN_MABA_CHANNELS * IFAN_R_FULL * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1);
    ok = ok && read_flat_file(maba_dir + "/tensors/output.txt", &g_ref_output[0][0][0][0][0][0],
                              IFAN_STAGE1_T * IFAN_MABA_CHANNELS * IFAN_R_FULL * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1);
    if (ok) {
        std::cout << "Loaded FeatureMABA data: " << data_dir << "\n";
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

    feature_maba_engine(
        g_input,
        g_weights,
        g_input_positions,
        g_in_proj_out,
        g_dw_conv_input,
        g_dw_conv_input_padded,
        g_dw_conv_out,
        g_mix_pre_norm,
        g_mix_norm_out,
        g_state_input,
        g_q,
        g_gate,
        g_alpha,
        g_state_sequence,
        g_state_back_out,
        g_refined_pre_dropout,
        g_delta_flat,
        g_delta,
        g_output
    );

    bool ok = true;
    ok = ok && print_check("input_positions", &g_input_positions[0][0][0], &g_ref_input_positions[0][0][0],
                           IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_CHANNELS);
    ok = ok && print_check("in_proj_out", &g_in_proj_out[0][0][0], &g_ref_in_proj_out[0][0][0],
                           IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_D_MODEL);
    ok = ok && print_check("dw_conv_input", &g_dw_conv_input[0][0][0], &g_ref_dw_conv_input[0][0][0],
                           IFAN_MABA_POSITIONS * IFAN_MABA_D_MODEL * IFAN_STAGE1_T);
    ok = ok && print_check("dw_conv_input_padded", &g_dw_conv_input_padded[0][0][0], &g_ref_dw_conv_input_padded[0][0][0],
                           IFAN_MABA_POSITIONS * IFAN_MABA_D_MODEL * (IFAN_STAGE1_T + IFAN_MABA_CONV_KERNEL - 1));
    ok = ok && print_check("dw_conv_out", &g_dw_conv_out[0][0][0], &g_ref_dw_conv_out[0][0][0],
                           IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_D_MODEL);
    ok = ok && print_check("mix_pre_norm", &g_mix_pre_norm[0][0][0], &g_ref_mix_pre_norm[0][0][0],
                           IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_D_MODEL);
    ok = ok && print_check("mix_norm_out", &g_mix_norm_out[0][0][0], &g_ref_mix_norm_out[0][0][0],
                           IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_D_MODEL);
    ok = ok && print_check("state_input", &g_state_input[0][0][0], &g_ref_state_input[0][0][0],
                           IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_STATE_DIM * 2);
    ok = ok && print_check("q", &g_q[0][0][0], &g_ref_q[0][0][0],
                           IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_STATE_DIM);
    ok = ok && print_check("gate", &g_gate[0][0][0], &g_ref_gate[0][0][0],
                           IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_STATE_DIM);
    ok = ok && print_check("alpha", &g_alpha[0][0][0], &g_ref_alpha[0][0][0],
                           IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_STATE_DIM);
    ok = ok && print_check("state_sequence", &g_state_sequence[0][0][0], &g_ref_state_sequence[0][0][0],
                           IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_STATE_DIM);
    ok = ok && print_check("state_back_out", &g_state_back_out[0][0][0], &g_ref_state_back_out[0][0][0],
                           IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_D_MODEL);
    ok = ok && print_check("refined_pre_dropout", &g_refined_pre_dropout[0][0][0], &g_ref_refined_pre_dropout[0][0][0],
                           IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_D_MODEL);
    ok = ok && print_check("delta_flat", &g_delta_flat[0][0][0], &g_ref_delta_flat[0][0][0],
                           IFAN_MABA_POSITIONS * IFAN_STAGE1_T * IFAN_MABA_CHANNELS);
    ok = ok && print_check("delta", &g_delta[0][0][0][0][0][0], &g_ref_delta[0][0][0][0][0][0],
                           IFAN_STAGE1_T * IFAN_MABA_CHANNELS * IFAN_R_FULL * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1);
    ok = ok && print_check("output", &g_output[0][0][0][0][0][0], &g_ref_output[0][0][0][0][0][0],
                           IFAN_STAGE1_T * IFAN_MABA_CHANNELS * IFAN_R_FULL * IFAN_CHARTS * IFAN_H_R1 * IFAN_W_R1);

    if (!ok) {
        std::cerr << "FAIL: FeatureMABA frontend mismatch\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
