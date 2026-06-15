#ifndef IFAN_TEMPORAL_R1_HPP
#define IFAN_TEMPORAL_R1_HPP

#include "../full_stage1_legacy/ifan_stage1.hpp"

void ifan_temporal_r1_engine(
    data_t input[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    const data_t weight[IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_TEMPORAL_KERNEL],
    const data_t bias[IFAN_BRANCH_CHANNELS],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
);

void ifan_temporal_r1_top(
    data_t input[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1],
    const data_t weight[IFAN_BRANCH_CHANNELS][IFAN_BRANCH_CHANNELS][IFAN_TEMPORAL_KERNEL],
    const data_t bias[IFAN_BRANCH_CHANNELS],
    data_t output[IFAN_STAGE1_T][IFAN_BRANCH_CHANNELS][IFAN_R_FULL][IFAN_CHARTS][IFAN_H_R1][IFAN_W_R1]
);

#endif
