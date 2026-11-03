#pragma once
/* Observation vector for the M0 diff-drive policy.

   Order and units match sim/m0_nav_demo/env_diffdrive.py DiffDriveGoalEnv._obs.
   Do not reorder: the embedded MLP was trained on this layout.

     0  gx   goal x in the body frame, meters
     1  gy   goal y in the body frame, meters
     2  sin  sin(heading), heading in radians
     3  cos  cos(heading)
     4  vx   body linear velocity, world-frame x, m/s
              (sim: free-joint qvel x, not body-frame)
     5  vy   body linear velocity, world-frame y, m/s
     6  wL   left wheel joint rate, rad/s
     7  wR   right wheel joint rate, rad/s

   gx, gy are a world-frame goal (host teleop or offboard beacon) rotated into
   the body frame with the same formula as the env. They are not a ToF range.
*/
#define NAV_OBS_DIM 8

void obs_goal_body(float goal_x_m, float goal_y_m, float pos_x_m, float pos_y_m,
                   float yaw_rad, float *gx_m, float *gy_m);

void obs_build(float pos_x_m, float pos_y_m, float yaw_rad, float vx_world_m_s,
               float vy_world_m_s, float goal_x_world_m, float goal_y_world_m,
               float wheel_l_rad_s, float wheel_r_rad_s, float obs[NAV_OBS_DIM]);
