#include "host_link.h"

#include "sensors.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(ESP_PLATFORM) || defined(NAV_POSIX_UDP)
#include <fcntl.h>
#include <unistd.h>
#if defined(ESP_PLATFORM)
#include "lwip/sockets.h"
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#endif
#endif

static float s_goal_x = NAV_STUB_GOAL_X;
static float s_goal_y = NAV_STUB_GOAL_Y;
static int s_from_host;

#if defined(ESP_PLATFORM) || defined(NAV_POSIX_UDP)
static int s_sock = -1;
#endif

static uint32_t rd_u32_le(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static float rd_f32_le(const unsigned char *p) {
    uint32_t u = rd_u32_le(p);
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

void host_link_init(void) {
    s_goal_x = NAV_STUB_GOAL_X;
    s_goal_y = NAV_STUB_GOAL_Y;
    s_from_host = 0;
    printf("[host_link] goal source STUB world (%.3f, %.3f) until a host/beacon "
           "packet. VL53L0X is not a goal source.\n",
           s_goal_x, s_goal_y);

#if defined(ESP_PLATFORM) || defined(NAV_POSIX_UDP)
    if (s_sock >= 0) {
        return;
    }
    s_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (s_sock < 0) {
        printf("[host_link] STUB socket unavailable (%d); staying on compile-time goal\n",
               errno);
        return;
    }
    int reuse = 1;
    setsockopt(s_sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof reuse);
    int flags = fcntl(s_sock, F_GETFL, 0);
    if (flags >= 0) {
        fcntl(s_sock, F_SETFL, flags | O_NONBLOCK);
    }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons(NAV_HOST_UDP_PORT);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(s_sock, (struct sockaddr *)&addr, sizeof addr) != 0) {
        printf("[host_link] UDP bind %d failed (%d); ingest() still works, "
               "goal stays STUB until a packet is parsed\n",
               NAV_HOST_UDP_PORT, errno);
        close(s_sock);
        s_sock = -1;
        return;
    }
    printf("[host_link] UDP :%d listening for LCR1 world-goal packets\n",
           NAV_HOST_UDP_PORT);
    /* TODO(net): ESP-IDF STA once an SSID is provisioned. This file does not
       start Wi-Fi. socket/bind still run so a later netif can be polled; if
       they fail, the compile-time STUB goal stays in force. */
#else
    printf("[host_link] no UDP socket in this build; host_link_ingest() parses "
           "LCR1 packets for tests\n");
#endif
}

int host_link_udp_ready(void) {
#if defined(ESP_PLATFORM) || defined(NAV_POSIX_UDP)
    return s_sock >= 0;
#else
    return 0;
#endif
}

void host_link_poll(void) {
#if defined(ESP_PLATFORM) || defined(NAV_POSIX_UDP)
    if (s_sock < 0) {
        return;
    }
    unsigned char buf[128];
    int budget = 8; /* keep the 20 Hz loop from draining a burst forever */
    for (;;) {
        struct sockaddr_in from;
        if (budget-- <= 0) {
            return;
        }
        socklen_t fromlen = sizeof from;
        int n = recvfrom(s_sock, buf, sizeof buf, 0, (struct sockaddr *)&from, &fromlen);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                return;
            }
            printf("[host_link] recv error %d\n", errno);
            return;
        }
        if (n == 0) {
            continue;
        }
        host_link_ingest(buf, n);
    }
#else
    /* Socket-less build. The control loop still runs on the STUB goal. */
#endif
}

int host_link_ingest(const unsigned char *data, int len) {
    if (data == 0 || len < NAV_LCR1_GOAL_BYTES) {
        return -1;
    }
    if (rd_u32_le(data) != NAV_LCR1_MAGIC || data[4] != NAV_LCR1_VERSION) {
        return -1;
    }
    unsigned flags = data[5];
    /* Refused, not implemented. ToF must not become the goal through this link. */
    if (flags & NAV_FLAG_GOAL_FROM_TOF) {
        printf("[host_link] rejected packet: goal-from-ToF is not a goal source\n");
        return -1;
    }
    if ((flags & NAV_FLAG_INJECT_POSE) && len < NAV_LCR1_POSE_BYTES) {
        return -1;
    }
    float gx = rd_f32_le(data + 8);
    float gy = rd_f32_le(data + 12);
    s_goal_x = gx;
    s_goal_y = gy;
    s_from_host = 1;
    if (flags & NAV_FLAG_INJECT_POSE) {
        float x = rd_f32_le(data + 16);
        float y = rd_f32_le(data + 20);
        float yaw = rd_f32_le(data + 24);
        float vx = rd_f32_le(data + 28);
        float vy = rd_f32_le(data + 32);
        float wl = rd_f32_le(data + 36);
        float wr = rd_f32_le(data + 40);
        sensors_inject_yaw(yaw);
        sensors_inject_wheel_rates(wl, wr);
        sensors_inject_pose(x, y, yaw, vx, vy);
        printf("[host_link] goal source HOST world (%.3f, %.3f) pose inject\n", gx, gy);
    } else {
        sensors_release_host_inject();
        printf("[host_link] goal source HOST world (%.3f, %.3f)\n", gx, gy);
    }
    return 0;
}

void host_link_goal(float *x_m, float *y_m, const char **source) {
    *x_m = s_goal_x;
    *y_m = s_goal_y;
    if (source) {
        *source = s_from_host ? "HOST" : "STUB";
    }
}

int host_link_goal_is_stub(void) { return s_from_host ? 0 : 1; }
