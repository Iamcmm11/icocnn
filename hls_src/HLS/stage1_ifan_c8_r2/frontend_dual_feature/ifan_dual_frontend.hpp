#ifndef IFAN_DUAL_FRONTEND_HPP
#define IFAN_DUAL_FRONTEND_HPP

#include "../full_stage1_legacy/ifan_stage1.hpp"

struct IfanDualFrontendWeights {
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

    data_t norm_gamma[3][IFAN_BRANCH_CHANNELS];
    data_t norm_beta[3][IFAN_BRANCH_CHANNELS];
};

void ifan_dual_frontend_top(
    data_t input[IFAN_IN_CHANNELS][IFAN_STAGE1_T][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2],
    const IfanDualFrontendWeights &weights,
    const int reorder_r2_stem[1][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2],
    const int reorder_r2_main[IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2 + 2][IFAN_W_R2 + 2],
    const int kernel_idx_stem[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][1][1][9][4],
    const int kernel_idx_main[IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][9][4],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R2][IFAN_W_R2]
);

#endif
