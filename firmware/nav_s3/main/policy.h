#pragma once
// M0 nav policy on ESP32-S3: obs(8) -> MLP -> act(2).
// Weights are the REAL trained dagger_r4 BC/DAgger policy (76% on R2 stress),
// stored as float32 bit patterns (see policy_weights.h). tanh on hidden layers,
// linear output. Architecture: 8 -> 64(tanh) -> 64(tanh) -> 2.
void policy_forward(const float obs[8], float act[2]);
