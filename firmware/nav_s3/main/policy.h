#pragma once
// M0 nav policy on ESP32-S3: obs(8) -> MLP -> act(2).
// Weights are the trained dagger_r4 policy, stored as float32 bit patterns
// (see policy_weights.h). seed=0 eval_all.py: 151/200 = 76% in the training
// domain, the same printed percent as vanilla BC backup (153/200 = 76%).
// Not a gen_test result. tanh on hidden layers,
// linear output. Architecture: 8 -> 64(tanh) -> 64(tanh) -> 2.
void policy_forward(const float obs[8], float act[2]);
