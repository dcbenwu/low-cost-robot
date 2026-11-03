#include "obs_builder.h"

#include <math.h>

void obs_goal_body(float goal_x_m, float goal_y_m, float pos_x_m, float pos_y_m,
                   float yaw_rad, float *gx_m, float *gy_m) {
    /* env_diffdrive._obs:
         dx, dy = goal - pos
         c, s = cos(-heading), sin(-heading)
         gx = c*dx - s*dy
         gy = s*dx + c*dy
    */
    float dx = goal_x_m - pos_x_m;
    float dy = goal_y_m - pos_y_m;
    float c = cosf(-yaw_rad);
    float s = sinf(-yaw_rad);
    *gx_m = c * dx - s * dy;
    *gy_m = s * dx + c * dy;
}

void obs_build(float pos_x_m, float pos_y_m, float yaw_rad, float vx_world_m_s,
               float vy_world_m_s, float goal_x_world_m, float goal_y_world_m,
               float wheel_l_rad_s, float wheel_r_rad_s, float obs[NAV_OBS_DIM]) {
    obs_goal_body(goal_x_world_m, goal_y_world_m, pos_x_m, pos_y_m, yaw_rad,
                  &obs[0], &obs[1]);
    obs[2] = sinf(yaw_rad);
    obs[3] = cosf(yaw_rad);
    obs[4] = vx_world_m_s;
    obs[5] = vy_world_m_s;
    obs[6] = wheel_l_rad_s;
    obs[7] = wheel_r_rad_s;
}
