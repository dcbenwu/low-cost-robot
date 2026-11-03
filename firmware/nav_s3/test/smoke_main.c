#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "act_map.h"
#include "host_link.h"
#include "obs_builder.h"
#include "policy.h"
#include "sensors.h"

static int g_errors;

static void expect_true(const char *name, int cond) {
    if (cond) {
        printf("CHECK %s PASS\n", name);
    } else {
        printf("CHECK %s FAIL\n", name);
        g_errors++;
    }
}

static int near(float a, float b, float tol) {
    float d = a - b;
    if (d < 0.f) {
        d = -d;
    }
    return d <= tol;
}

static void print_obs(const char *tag, const float obs[8]) {
    printf("%s", tag);
    for (int i = 0; i < 8; i++) {
        printf(" %.8f", obs[i]);
    }
    printf("\n");
}

static void build_from_sensors(float obs[8]) {
    nav_pose_t pose;
    float gx, gy, wl, wr;
    sensors_update(0.05f);
    sensors_get_pose(&pose);
    sensors_get_wheel_rates(&wl, &wr);
    host_link_goal(&gx, &gy, 0);
    obs_build(pose.x_m, pose.y_m, pose.yaw_rad, pose.vx_world_m_s, pose.vy_world_m_s, gx, gy,
              wl, wr, obs);
}

static void check_geometry(void) {
    float obs[8];
    sensors_init();
    host_link_init();
    build_from_sensors(obs);
    /* STUB pose at the origin, STUB world goal (1, 0) -> body goal (1, 0). */
    expect_true("obs_stub_goal", near(obs[0], 1.f, 1e-6f) && near(obs[1], 0.f, 1e-6f) &&
                                     near(obs[2], 0.f, 1e-6f) && near(obs[3], 1.f, 1e-6f) &&
                                     near(obs[4], 0.f, 1e-6f) && near(obs[6], 0.f, 1e-6f) &&
                                     near(obs[7], 0.f, 1e-6f));
    print_obs("OBS stub", obs);

    obs_build(0.f, 0.f, 1.57079637f, 0.f, 0.f, 1.f, 0.f, 0.25f, -0.5f, obs);
    expect_true("obs_yaw90", near(obs[0], 0.f, 1e-5f) && near(obs[1], -1.f, 1e-5f) &&
                                 near(obs[2], 1.f, 1e-5f) && near(obs[3], 0.f, 1e-5f) &&
                                 near(obs[6], 0.25f, 1e-6f) && near(obs[7], -0.5f, 1e-6f));
    print_obs("OBS yaw90", obs);
}

static void check_odometry(void) {
    nav_pose_t pose;
    sensors_init();
    host_link_init();
    sensors_inject_wheel_rates(1.f, 1.f);
    sensors_update(0.1f);
    sensors_get_pose(&pose);
    /* v = 0.05 * (1+1) / 2 = 0.05 m/s along +x at yaw 0; dt = 0.1 -> dx = 0.005. */
    expect_true("odom_forward", near(pose.vx_world_m_s, 0.05f, 1e-6f) &&
                                    near(pose.vy_world_m_s, 0.f, 1e-6f) &&
                                    near(pose.x_m, 0.005f, 1e-6f));
}

static void check_tof_is_not_goal(void) {
    float obs[8];
    unsigned char tof_pkt[16];
    tof_reading_t tof;
    const char *src = 0;
    float gx, gy;

    sensors_init();
    host_link_init();
    sensors_inject_tof(0.3f, 0.4f, 0.5f);
    sensors_read_tof(&tof);
    build_from_sensors(obs);
    expect_true("tof_not_in_obs", tof.valid == 1 && near(tof.range_m[0], 0.3f, 1e-6f) &&
                                      near(obs[0], 1.f, 1e-6f) && near(obs[1], 0.f, 1e-6f));

    memset(tof_pkt, 0, sizeof tof_pkt);
    tof_pkt[0] = 'L';
    tof_pkt[1] = 'C';
    tof_pkt[2] = 'R';
    tof_pkt[3] = '1';
    tof_pkt[4] = 1;
    tof_pkt[5] = (unsigned char)NAV_FLAG_GOAL_FROM_TOF;
    /* goal floats 5, 6 would be a different goal if the flag were honored */
    expect_true("tof_flag_rejected", host_link_ingest(tof_pkt, 16) != 0);
    host_link_goal(&gx, &gy, &src);
    expect_true("tof_flag_keeps_stub", src && strcmp(src, "STUB") == 0 && near(gx, 1.f, 1e-6f) &&
                                           near(gy, 0.f, 1e-6f));
    expect_true("short_packet_rejected", host_link_ingest(tof_pkt, 8) != 0);
    expect_true("bad_magic_rejected", host_link_ingest((const unsigned char *)"NOPE____________", 16) != 0);
}

static void check_host_goal_and_release(void) {
    float obs[8];
    nav_pose_t pose;
    unsigned char pkt[44];
    const char *src = 0;
    float gx, gy;
    float duty[2];
    float act[2];

    sensors_init();
    host_link_init();
    memset(pkt, 0, sizeof pkt);
    pkt[0] = 'L';
    pkt[1] = 'C';
    pkt[2] = 'R';
    pkt[3] = '1';
    pkt[4] = 1;
    pkt[5] = (unsigned char)NAV_FLAG_INJECT_POSE;
    /* little-endian float32 payload filled by the python runner via --packet.
       This in-tree packet uses a goal of (4, 6), pose (1, 2, yaw 0), vel (0.1, -0.2),
       wheels (0.5, -0.25). Bytes are written as raw LE floats below. */
    {
        float fields[9] = {4.f, 6.f, 1.f, 2.f, 0.f, 0.1f, -0.2f, 0.5f, -0.25f};
        unsigned char *dst = pkt + 8;
        for (int i = 0; i < 9; i++) {
            memcpy(dst + 4 * i, &fields[i], 4);
        }
    }
    expect_true("inject_accepted", host_link_ingest(pkt, 44) == 0);
    build_from_sensors(obs);
    expect_true("inject_obs", near(obs[0], 3.f, 1e-5f) && near(obs[1], 4.f, 1e-5f) &&
                                  near(obs[2], 0.f, 1e-6f) && near(obs[3], 1.f, 1e-6f) &&
                                  near(obs[4], 0.1f, 1e-6f) && near(obs[5], -0.2f, 1e-6f) &&
                                  near(obs[6], 0.5f, 1e-6f) && near(obs[7], -0.25f, 1e-6f));
    print_obs("OBS inject", obs);
    host_link_goal(&gx, &gy, &src);
    expect_true("goal_source_host", src && strcmp(src, "HOST") == 0);

    /* Goal-only packet drops the pose hold. Wheel rates return to the STUB zero,
       so the next odometry update reports zero world velocity and keeps position. */
    pkt[5] = 0;
    expect_true("goal_only_accepted", host_link_ingest(pkt, 16) == 0);
    sensors_update(0.1f);
    sensors_get_pose(&pose);
    expect_true("release_stops_inject_vel", near(pose.x_m, 1.f, 1e-5f) && near(pose.y_m, 2.f, 1e-5f) &&
                                                near(pose.vx_world_m_s, 0.f, 1e-6f));

    act[0] = 2.f;
    act[1] = -0.25f;
    act_to_lr_duty(act, duty);
    expect_true("duty_is_left_right", near(duty[0], 1.f, 0.f) && near(duty[1], -0.25f, 1e-6f));
    act[0] = 0.4f;
    act[1] = -0.2f;
    act_to_lr_duty(act, duty);
    expect_true("duty_passthrough", near(duty[0], 0.4f, 1e-6f) && near(duty[1], -0.2f, 1e-6f));
}

static void check_policy(void) {
    /* Verified reset(seed=0) sample (gx, gy are that draw; the rest is the settle). */
    static const float ref[8] = {-0.4063f, 1.0991f, -0.2376f, 0.9714f, 0.0417f, 0.f,
                                 -0.5898f, -0.5898f};
    static const float ident[8] = {1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f};
    static const float zero[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const float *cases[3] = {ref, ident, zero};
    for (int c = 0; c < 3; c++) {
        float act[2];
        policy_forward(cases[c], act);
        printf("POLICY");
        for (int i = 0; i < 8; i++) {
            printf(" %.8f", cases[c][i]);
        }
        printf(" %.8f %.8f\n", act[0], act[1]);
    }
}

static int ingest_file(const char *path) {
    FILE *f = fopen(path, "rb");
    unsigned char buf[128];
    size_t n;
    float obs[8];
    if (!f) {
        fprintf(stderr, "open %s failed\n", path);
        return 1;
    }
    n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    sensors_init();
    host_link_init();
    if (host_link_ingest(buf, (int)n) != 0) {
        fprintf(stderr, "packet rejected\n");
        return 1;
    }
    build_from_sensors(obs);
    print_obs("OBS packet", obs);
    return 0;
}

static int listen_once(void) {
    int i;
    float obs[8];
    sensors_init();
    host_link_init();
    if (!host_link_udp_ready()) {
        fprintf(stderr, "udp socket not bound\n");
        return 1;
    }
    printf("LISTEN %d\n", NAV_HOST_UDP_PORT);
    fflush(stdout);
    for (i = 0; i < 400; i++) {
        host_link_poll();
        if (!host_link_goal_is_stub()) {
            build_from_sensors(obs);
            print_obs("OBS udp", obs);
            return 0;
        }
        usleep(10000);
    }
    fprintf(stderr, "udp timeout\n");
    return 1;
}

int main(int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--listen") == 0) {
            return listen_once();
        }
    }
    check_geometry();
    check_odometry();
    check_tof_is_not_goal();
    check_host_goal_and_release();
    check_policy();
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--packet") == 0 && i + 1 < argc) {
            if (ingest_file(argv[++i]) != 0) {
                g_errors++;
            }
        }
    }
    printf("CHECKS %d\n", g_errors);
    return g_errors ? 1 : 0;
}
