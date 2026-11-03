#include "sensors.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* robot.xml: wheel radius 0.05 m, wheels at y = ±0.11 m (track 0.22 m).
   Positive rate on both hinges is treated as +body-x. That sign is the
   kinematic assumption used for odometry assist; it is not a bench calibration
   (no drivetrain is wired). Host pose inject bypasses it for sim-bridge tests.
   Track width is unused until a wheel-yaw term is fused with the IMU.
*/
#define NAV_WHEEL_RADIUS_M 0.05f

static nav_pose_t s_pose;
static float s_yaw;
static float s_rate_l;
static float s_rate_r;
static float s_wheel_l;
static float s_wheel_r;
static int s_pose_hold;
static int s_rate_override;
static tof_reading_t s_tof;

void sensors_init(void) {
    memset(&s_pose, 0, sizeof s_pose);
    s_yaw = 0.f;
    s_rate_l = 0.f;
    s_rate_r = 0.f;
    s_wheel_l = 0.f;
    s_wheel_r = 0.f;
    s_pose_hold = 0;
    s_rate_override = 0;
    memset(&s_tof, 0, sizeof s_tof); /* valid=0: STUB ToF has no ranges */
    printf("[sensors] STUB imu/encoder/tof. ToF is obstacle/debug only, not a goal.\n");
    /* TODO(driver): MPU6050 on I2C. Complementary filter supplies yaw_rad.
       TODO(driver): quadrature decode on the two 12 V encoder motors.
         rad/s = (delta_ticks / ticks_per_rev) * 2*pi / dt
         ticks_per_rev is motor-specific and is not calibrated here.
       TODO(driver): VL53L0X x3, XSHUT address remap. Publish ranges only
         through sensors_read_tof. Do not write them into the goal or obs[0:2].
    */
}

void sensors_read_imu(imu_reading_t *out) {
    /* STUB: held yaw (default 0, or the last inject). No MPU6050 sample. */
    out->yaw_rad = s_yaw;
    out->valid = 1;
}

void sensors_read_encoders(encoder_reading_t *out) {
    /* STUB: injected rad/s, else the fixed zero vector. Ticks are not counted. */
    out->left_rad_s = s_rate_override ? s_rate_l : 0.f;
    out->right_rad_s = s_rate_override ? s_rate_r : 0.f;
    out->left_ticks = 0;
    out->right_ticks = 0;
}

void sensors_read_tof(tof_reading_t *out) {
    /* STUB. Ranges are never a goal. */
    *out = s_tof;
}

void sensors_update(float dt_s) {
    imu_reading_t imu;
    encoder_reading_t enc;
    sensors_read_imu(&imu);
    sensors_read_encoders(&enc);
    s_wheel_l = enc.left_rad_s;
    s_wheel_r = enc.right_rad_s;
    if (s_pose_hold) {
        return; /* sim-bridge owns x, y, yaw, and world velocity */
    }
    if (imu.valid) {
        s_pose.yaw_rad = imu.yaw_rad;
    }
    /* No sideslip: forward speed from the two wheel rates, then world frame
       with the IMU heading. This is the stand-in for sim free-joint qvel. */
    float v = NAV_WHEEL_RADIUS_M * 0.5f * (enc.left_rad_s + enc.right_rad_s);
    s_pose.vx_world_m_s = v * cosf(s_pose.yaw_rad);
    s_pose.vy_world_m_s = v * sinf(s_pose.yaw_rad);
    s_pose.x_m += s_pose.vx_world_m_s * dt_s;
    s_pose.y_m += s_pose.vy_world_m_s * dt_s;
}

void sensors_get_pose(nav_pose_t *out) { *out = s_pose; }

void sensors_get_wheel_rates(float *left_rad_s, float *right_rad_s) {
    *left_rad_s = s_wheel_l;
    *right_rad_s = s_wheel_r;
}

void sensors_inject_yaw(float yaw_rad) { s_yaw = yaw_rad; }

void sensors_inject_wheel_rates(float left_rad_s, float right_rad_s) {
    s_rate_l = left_rad_s;
    s_rate_r = right_rad_s;
    s_rate_override = 1;
}

void sensors_inject_pose(float x_m, float y_m, float yaw_rad, float vx_world_m_s,
                         float vy_world_m_s) {
    s_pose.x_m = x_m;
    s_pose.y_m = y_m;
    s_pose.yaw_rad = yaw_rad;
    s_pose.vx_world_m_s = vx_world_m_s;
    s_pose.vy_world_m_s = vy_world_m_s;
    s_yaw = yaw_rad;
    s_pose_hold = 1;
}

void sensors_inject_tof(float front_m, float left_m, float right_m) {
    s_tof.range_m[0] = front_m;
    s_tof.range_m[1] = left_m;
    s_tof.range_m[2] = right_m;
    s_tof.valid = 1;
}

void sensors_release_host_inject(void) {
    s_pose_hold = 0;
    s_rate_override = 0;
}
