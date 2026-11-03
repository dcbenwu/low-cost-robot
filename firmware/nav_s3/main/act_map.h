#pragma once
/* Policy outputs are left and right wheel torque commands in about [-1, 1],
   matching DiffDriveGoalEnv's two motor actuators. They are not (v, w).
   Signed duty is what a later TB6612 direction + LEDC magnitude will take.
   The driver is not wired; callers may printf the duty.
*/
static inline float nav_clampf(float v, float lo, float hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

static inline void act_to_lr_duty(const float act[2], float duty_lr[2]) {
    duty_lr[0] = nav_clampf(act[0], -1.f, 1.f); /* left */
    duty_lr[1] = nav_clampf(act[1], -1.f, 1.f); /* right */
}
