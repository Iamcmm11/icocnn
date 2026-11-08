#pragma once
// Explicit FP32 contract, independent of legacy ap_fixed auto-detection.
#ifndef MABA_T
#define MABA_T 6
#endif
#ifndef MABA_VARIANT
#define MABA_VARIANT 3
#endif
#ifndef MABA_TILE
#define MABA_TILE 16
#endif
#ifndef MABA_PAR
#define MABA_PAR 1
#endif
#ifndef MABA_FIFO
#define MABA_FIFO 8
#endif
#ifndef MABA_POSITIONS
#define MABA_POSITIONS 240
#endif
namespace maba_hw {
constexpr int C=8, D=16, S=8, P=MABA_POSITIONS, WCOUNT=864, COUNT=MABA_T*C*P;
struct Weights {
    float wi[D][C], bi[D], wc[D][3], bc[D], gamma[D], beta[D];
    float ws[2*S][D], bs[2*S], wb[D][S], bb[D], wo[C][D], bo[C];
    float wh[C][S], bf[C]; // offline folded projection; appended to original 792 floats
};
static_assert(sizeof(Weights)==WCOUNT*sizeof(float), "weight layout must be packed FP32");
static_assert(MABA_PAR==1 || MABA_PAR==2 || MABA_PAR==4, "unsupported PAR");
static_assert(MABA_TILE>0 && MABA_TILE<=P, "invalid tile");
static_assert(P>0 && P<=240 && MABA_T>0, "invalid validation shape");
static_assert(MABA_VARIANT>=0 && MABA_VARIANT<=4, "invalid variant");
}
// Flat external layout [T,C,R,chart,H,W]; weights layout documented by exporter.
extern "C" void maba_top(const float input[maba_hw::COUNT],
                         const float weights[maba_hw::WCOUNT],
                         float output[maba_hw::COUNT]);
