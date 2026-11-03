#pragma once
/* STUB sensor facade for the unassembled ESP32-S3 base.

   Real drivers are not in this tree. Each read path is labeled STUB and can be
   fed from the host link (sim-bridge pose inject) or left on a fixed zero vector.

   VL53L0X ranges are obstacle/debug only. Nothing in this interface is a goal.
*/

typedef struct {
    float x_m;
    float y_m;
    float yaw_rad;       /* same heading obs_build puts through sin/cos */
    float vx_world_m_s;  /* world-frame, sim free-joint qvel x */
    float vy_world_m_s;  /* world-frame, sim free-joint qvel y */
} nav_pose_t;

typedef struct {
    float yaw_rad;
    int valid; /* STUB reports valid with the held yaw; there is no gyro yet */
} imu_reading_t;

typedef struct {
    float left_rad_s;
    float right_rad_s;
    int left_ticks;  /* STUB: always 0 until a quadrature driver exists */
    int right_ticks;
} encoder_reading_t;

typedef struct {
    float range_m[3]; /* front, left, right. Not a goal vector. */
    int valid;
} tof_reading_t;

void sensors_init(void);

/* STUB reads. */
void sensors_read_imu(imu_reading_t *out);
void sensors_read_encoders(encoder_reading_t *out);
void sensors_read_tof(tof_reading_t *out);

/* Integrate encoder rates into world v and xy, using IMU yaw as heading.
   Skipped while a host pose inject is held. */
void sensors_update(float dt_s);

void sensors_get_pose(nav_pose_t *out);
void sensors_get_wheel_rates(float *left_rad_s, float *right_rad_s);

/* Host / test injection. */
void sensors_inject_yaw(float yaw_rad);
void sensors_inject_wheel_rates(float left_rad_s, float right_rad_s);
void sensors_inject_pose(float x_m, float y_m, float yaw_rad, float vx_world_m_s,
                         float vy_world_m_s);
void sensors_inject_tof(float front_m, float left_m, float right_m);

/* Drop pose-hold and wheel-rate override. Last pose and yaw are kept. */
void sensors_release_host_inject(void);
