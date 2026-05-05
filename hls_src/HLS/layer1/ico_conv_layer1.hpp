#ifndef ICO_CONV_LAYER1_HPP
#define ICO_CONV_LAYER1_HPP

#include <cmath>

#define R_LEVEL     2
#define H           4
#define W           8
#define CHARTS      5
#define TIME_STEPS  52

#define CIN         32
#define COUT        32
#define RIN         6
#define ROUT        6

#define H_PADDED    (H + 2)
#define W_PADDED    (W + 2)

#define KERNEL_H    3
#define KERNEL_W    3

#define IC_TILE     8
#define OC_TILE     4

#define OC_PAR_FACTOR 2

typedef float data_t;

void smooth_vertices(
    data_t input[CIN][RIN][CHARTS][H][W],
    data_t output[CIN][RIN][CHARTS][H][W]
);

void clean_vertices(
    data_t input[CHARTS][H][W],
    data_t output[CHARTS][H][W]
);

void pad_ico(
    data_t input[CIN][RIN][CHARTS][H][W],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    data_t output[CIN][RIN][CHARTS][H_PADDED][W_PADDED]
);

void conv_ico_layer1(
    data_t input[TIME_STEPS][CIN][RIN][CHARTS][H][W],
    const data_t weight[COUT][CIN][RIN][7],
    const data_t bias[COUT],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    data_t output[TIME_STEPS][COUT][ROUT][CHARTS][H][W]
);

void get_kernel(
    const data_t weight[COUT][CIN][RIN][7],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    data_t kernel[COUT][ROUT][CIN][RIN][KERNEL_H][KERNEL_W]
);

#endif
