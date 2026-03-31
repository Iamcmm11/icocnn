#ifndef ICO_CONV_LAYER2_5_HPP
#define ICO_CONV_LAYER2_5_HPP

#include <cmath>

#if defined(__has_include)
#if __has_include(<ap_fixed.h>)
#include <ap_fixed.h>
#define ICO_LAYER2_5_HAS_AP_FIXED 1
#endif
#endif

#ifndef ICO_LAYER2_5_HAS_AP_FIXED
#define ICO_LAYER2_5_HAS_AP_FIXED 0
#endif

#define R_LEVEL     1
#define H           2
#define W           4
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

#define OC_PAR_FACTOR 2
#define OC_TILE      OC_PAR_FACTOR

#ifndef ICO_LAYER2_5_INPUT_W
#define ICO_LAYER2_5_INPUT_W 16
#endif

#ifndef ICO_LAYER2_5_INPUT_I
#define ICO_LAYER2_5_INPUT_I 4
#endif

#ifndef ICO_LAYER2_5_WEIGHT_W
#define ICO_LAYER2_5_WEIGHT_W 14
#endif

#ifndef ICO_LAYER2_5_WEIGHT_I
#define ICO_LAYER2_5_WEIGHT_I 3
#endif

#ifndef ICO_LAYER2_5_ACT_W
#define ICO_LAYER2_5_ACT_W 24
#endif

#ifndef ICO_LAYER2_5_ACT_I
#define ICO_LAYER2_5_ACT_I 8
#endif

#ifndef ICO_LAYER2_5_ACC_W
#define ICO_LAYER2_5_ACC_W 40
#endif

#ifndef ICO_LAYER2_5_ACC_I
#define ICO_LAYER2_5_ACC_I 18
#endif

// Host-visible interface type kept as float to preserve the current test data
// format and native g++ verification flow.
typedef float data_t;

#if ICO_LAYER2_5_HAS_AP_FIXED
typedef ap_fixed<ICO_LAYER2_5_INPUT_W, ICO_LAYER2_5_INPUT_I, AP_RND, AP_SAT> input_t;
typedef ap_fixed<ICO_LAYER2_5_WEIGHT_W, ICO_LAYER2_5_WEIGHT_I, AP_RND, AP_SAT> weight_t;
// Keep input/weight compact by default, but make the internal accumulation
// widths tunable so we can sweep quantization without changing the algorithm.
typedef ap_fixed<ICO_LAYER2_5_ACT_W, ICO_LAYER2_5_ACT_I, AP_RND, AP_SAT> act_t;
typedef ap_fixed<ICO_LAYER2_5_ACC_W, ICO_LAYER2_5_ACC_I, AP_RND, AP_SAT> acc_t;
#else
typedef float input_t;
typedef float weight_t;
typedef float act_t;
typedef float acc_t;
#endif

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

void conv_ico_layer2_5(
    data_t input[TIME_STEPS][CIN][RIN][CHARTS][H][W],
    const data_t weight[COUT][CIN][RIN][7],
    const data_t bias[COUT],
    const int kernel_expansion_idx[COUT][ROUT][CIN][RIN][9][4],
    const int reorder_idx[RIN][CHARTS][H_PADDED][W_PADDED],
    data_t output[TIME_STEPS][COUT][ROUT][CHARTS][H][W]
);

#endif
