#ifndef IFAN_STAGE1_HPP
#define IFAN_STAGE1_HPP

#include <cmath>

#if defined(__has_include)
#if __has_include(<ap_fixed.h>)
#include <ap_fixed.h>
#define IFAN_STAGE1_HAS_AP_FIXED 1
#endif
#endif

#ifndef IFAN_STAGE1_HAS_AP_FIXED
#define IFAN_STAGE1_HAS_AP_FIXED 0
#endif

#ifndef IFAN_STAGE1_T
#define IFAN_STAGE1_T 6
#endif

#define IFAN_IN_CHANNELS 2
#define IFAN_BRANCH_CHANNELS 8
#define IFAN_R_FULL 6
#define IFAN_CHARTS 5
#define IFAN_H_R2 4
#define IFAN_W_R2 8
#define IFAN_H_R1 2
#define IFAN_W_R1 4
#define IFAN_KERNEL_NEIGHBORS 7
#define IFAN_KERNEL_H 3
#define IFAN_KERNEL_W 3
#define IFAN_TEMPORAL_KERNEL 5

#ifndef IFAN_STAGE1_INPUT_W
#define IFAN_STAGE1_INPUT_W 16
#endif
#ifndef IFAN_STAGE1_INPUT_I
#define IFAN_STAGE1_INPUT_I 4
#endif
#ifndef IFAN_STAGE1_WEIGHT_W
#define IFAN_STAGE1_WEIGHT_W 14
#endif
#ifndef IFAN_STAGE1_WEIGHT_I
#define IFAN_STAGE1_WEIGHT_I 3
#endif
#ifndef IFAN_STAGE1_ACT_W
#define IFAN_STAGE1_ACT_W 24
#endif
#ifndef IFAN_STAGE1_ACT_I
#define IFAN_STAGE1_ACT_I 8
#endif
#ifndef IFAN_STAGE1_ACC_W
#define IFAN_STAGE1_ACC_W 40
#endif
#ifndef IFAN_STAGE1_ACC_I
#define IFAN_STAGE1_ACC_I 18
#endif

typedef float data_t;

#if IFAN_STAGE1_HAS_AP_FIXED
typedef ap_fixed<IFAN_STAGE1_INPUT_W, IFAN_STAGE1_INPUT_I, AP_RND, AP_SAT> input_t;
typedef ap_fixed<IFAN_STAGE1_WEIGHT_W, IFAN_STAGE1_WEIGHT_I, AP_RND, AP_SAT> weight_t;
typedef ap_fixed<IFAN_STAGE1_ACT_W, IFAN_STAGE1_ACT_I, AP_RND, AP_SAT> act_t;
typedef ap_fixed<IFAN_STAGE1_ACC_W, IFAN_STAGE1_ACC_I, AP_RND, AP_SAT> acc_t;
#else
typedef float input_t;
typedef float weight_t;
typedef float act_t;
typedef float acc_t;
#endif

enum IcoResolution {
    ICO_R2 = 2,
    ICO_R1 = 1
};

struct IcoConvConfig {
    int cin;
    int cout;
    int rin;
    int rout;
    int height;
    int width;
    int padded_height;
    int padded_width;
};

struct IfanStage1Weights {
    data_t phat_stem_w[IFAN_BRANCH_CHANNELS][1][1][IFAN_KERNEL_NEIGHBORS];
    data_t phat_stem_b[IFAN_BRANCH_CHANNELS];
    data_t lms_stem_w[IFAN_BRANCH_CHANNELS][1][1][IFAN_KERNEL_NEIGHBORS];
    data_t lms_stem_b[IFAN_BRANCH_CHANNELS];

    data_t phat_res_w[2][IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_NEIGHBORS];
    data_t phat_res_b[2][IFAN_BRANCH_CHANNELS];
    data_t lms_res_w[2][IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_NEIGHBORS];
    data_t lms_res_b[2][IFAN_BRANCH_CHANNELS];

    data_t attn_w[2][IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_NEIGHBORS];
    data_t attn_b[2][IFAN_BRANCH_CHANNELS];

    data_t fusion_w[4][IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_NEIGHBORS];
    data_t fusion_b[4][IFAN_BRANCH_CHANNELS];
    data_t fusion_temporal_w[4][IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_TEMPORAL_KERNEL];
    data_t fusion_temporal_b[4][IFAN_BRANCH_CHANNELS];

    data_t final_w[IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_KERNEL_NEIGHBORS];
    data_t final_b[IFAN_BRANCH_CHANNELS];
    data_t final_temporal_w[IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_TEMPORAL_KERNEL];
    data_t final_temporal_b[IFAN_BRANCH_CHANNELS];

    data_t norm_gamma[16][IFAN_BRANCH_CHANNELS];
    data_t norm_beta[16][IFAN_BRANCH_CHANNELS];
};

void ifan_stage1_top(
    data_t input[IFAN_IN_CHANNELS][IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    const IfanStage1Weights &weights,
    const int reorder_r2_stem[1][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2],
    const int reorder_r2_main[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2],
    const int reorder_r1[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1 + 2][IFAN_W_R1 + 2],
    const int kernel_idx_stem[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][1][1][9][4],
    const int kernel_idx_main[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][9][4],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
);

#endif
