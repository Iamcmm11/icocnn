#pragma once
#include "ifan_stage1.hpp"
#include "ifan_stage1_maba.hpp"
#include "ifan_stage1_post.hpp"
void ifan_v14_top(
 data_t input[2][IFAN_STAGE1_T][5][4][8], const IfanStage1Weights &w,
 const int r0[1][5][6][10], const int r2[6][5][6][10], const int r1[6][5][4][6],
 const int k0[8][6][1][1][9][4], const int k1[8][6][8][6][9][4],
 const FeatureMabaWeights &mw, const PostMabaWeights &pw, data_t coords[IFAN_STAGE1_T][3]);
