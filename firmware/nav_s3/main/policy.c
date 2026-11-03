#include "policy.h"
#include "policy_weights.h"
#include <math.h>
#include <string.h>

// policy_weights.h defines uint32_t bit-pattern arrays (endian-agnostic).
// On little-endian targets (ESP32 / x86) we reinterpret in place.
#define FP(bits) ((const float *)(bits))

void policy_forward(const float obs[8], float act[2]) {
    const float *mu = FP(obs_mu_bits);
    const float *sd = FP(obs_sd_bits);
    const float *W0 = FP(W0_bits); // [64][8]
    const float *b0 = FP(b0_bits);
    const float *W1 = FP(W1_bits); // [64][64]
    const float *b1 = FP(b1_bits);
    const float *W2 = FP(W2_bits); // [2][64]
    const float *b2 = FP(b2_bits);

    float x[8];
    for (int i = 0; i < 8; i++) x[i] = (obs[i] - mu[i]) / sd[i]; // 0/1 -> normalized

    float h0[64];
    for (int o = 0; o < 64; o++) {
        float s = b0[o];
        for (int i = 0; i < 8; i++) s += W0[o * 8 + i] * x[i];
        h0[o] = tanhf(s);
    }
    float h1[64];
    for (int o = 0; o < 64; o++) {
        float s = b1[o];
        for (int i = 0; i < 64; i++) s += W1[o * 64 + i] * h0[i];
        h1[o] = tanhf(s);
    }
    for (int o = 0; o < 2; o++) {
        float s = b2[o];
        for (int i = 0; i < 64; i++) s += W2[o * 64 + i] * h1[i];
        act[o] = s; // linear output
    }
}
