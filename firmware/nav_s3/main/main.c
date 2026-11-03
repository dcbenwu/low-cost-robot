#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "policy.h"

// ---- Stub I/O (no hardware wired yet) ---------------------------------------
// 8-d observation for the M0 "reach/hold" nav task. Replace with real sensor
// reads (VL53L0X ToF + IMU + encoder) when the board is assembled.
static void read_obs(float obs[8]) {
    obs[0] = 1.00f; // position-ish (m)
    obs[1] = 0.80f;
    obs[2] = 0.00f;
    obs[3] = 0.50f;
    obs[4] = 0.00f;
    obs[5] = 0.20f;
    obs[6] = 1.20f;
    obs[7] = 1.50f;
}

// Map raw policy output (act[0]=longitudinal, act[1]=steering) to physical units.
// Replace the print with ledcWrite + TB6612 direction logic on real hardware.
#define MAX_SPEED 0.24f // m/s
#define MAX_STEER 0.45f // rad/s
static void apply_act(const float act[2]) {
    float v = MAX_SPEED * act[0];
    float w = MAX_STEER * act[1];
    printf("[act] v=%.3f m/s  w=%.3f rad/s\n", v, w);
}

void app_main(void) {
    printf("nav_s3 firmware STUB: obs(8) -> REAL dagger_r4 MLP(8-64-64-2) -> act(2)\n");
    float obs[8], act[2];
    while (1) {
        read_obs(obs);
        policy_forward(obs, act);
        printf("[policy] act[0]=%.4f act[1]=%.4f\n", act[0], act[1]);
        apply_act(act);
        vTaskDelay(pdMS_TO_TICKS(50)); // 20 Hz control
    }
}
