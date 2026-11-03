#include <stdio.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "act_map.h"
#include "host_link.h"
#include "obs_builder.h"
#include "policy.h"
#include "sensors.h"

/* Map policy left/right torque commands to signed PWM duty.
   TODO(driver): magnitude -> LEDC, sign -> TB6612 IN1/IN2. No pins are
   assigned; the board is not assembled. Printf is the stand-in.
*/
static void apply_act(const float act[2]) {
    float duty[2];
    act_to_lr_duty(act, duty);
    printf("[act] L=%+.3f R=%+.3f signed PWM duty "
           "(left/right torque cmd, STUB, no motor driver)\n",
           duty[0], duty[1]);
}

void app_main(void) {
    const float dt_s = 0.05f; /* 20 Hz, same tick as the previous stub loop */
    float obs[NAV_OBS_DIM];
    float act[2];

    printf("nav_s3: STUB sensors -> obs_builder -> dagger_r4 policy_forward -> L/R duty\n");
    sensors_init();
    host_link_init();

    while (1) {
        nav_pose_t pose;
        tof_reading_t tof;
        float goal_x, goal_y, wl, wr;
        const char *goal_src = 0;

        host_link_poll();
        sensors_update(dt_s);
        sensors_get_pose(&pose);
        sensors_get_wheel_rates(&wl, &wr);
        host_link_goal(&goal_x, &goal_y, &goal_src);
        obs_build(pose.x_m, pose.y_m, pose.yaw_rad, pose.vx_world_m_s, pose.vy_world_m_s,
                  goal_x, goal_y, wl, wr, obs);

        /* Read ToF so a future obstacle hook has a place to land. Not obs[0:2]. */
        sensors_read_tof(&tof);
        policy_forward(obs, act);

        printf("[obs] gx=%.4f gy=%.4f sin=%.4f cos=%.4f vx=%.4f vy=%.4f "
               "wL=%.4f wR=%.4f goal=%s\n",
               obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7],
               goal_src ? goal_src : "?");
        printf("[tof] STUB valid=%d front=%.2f left=%.2f right=%.2f "
               "(obstacle/debug only, not the goal)\n",
               tof.valid, tof.range_m[0], tof.range_m[1], tof.range_m[2]);
        printf("[policy] L=%.4f R=%.4f\n", act[0], act[1]);
        apply_act(act);
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}
