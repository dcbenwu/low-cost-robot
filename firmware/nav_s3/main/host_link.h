#pragma once
/* Goal link: world-frame goal x/y from the host teleop or an offboard beacon.

   The goal is never taken from the VL53L0X. A packet that sets
   NAV_FLAG_GOAL_FROM_TOF is rejected and leaves the previous goal in place.
   Until the first valid packet, the goal is the compile-time STUB
   (NAV_STUB_GOAL_X / NAV_STUB_GOAL_Y, default 1 m, 0 m).

   Wire format, little-endian (magic bytes are 'L','C','R','1'):

     offset  size  field
     0       u32   magic 0x3152434C
     4       u8    version 1
     5       u8    flags
     6       u16   reserved
     8       f32   goal_x  world meters
     12      f32   goal_y  world meters
     16      f32   pose_x  present only if NAV_FLAG_INJECT_POSE
     20      f32   pose_y
     24      f32   yaw radians (same heading as obs sin/cos)
     28      f32   vx world m/s (sim free-joint qvel x)
     32      f32   vy world m/s
     36      f32   left wheel rad/s
     40      f32   right wheel rad/s

   Goal-only packets are 16 bytes. Pose inject packets are 44 bytes.
   NAV_FLAG_INJECT_POSE is for sim-bridge tests, not for localization.
   UDP port is NAV_HOST_UDP_PORT (9000). The control task polls; it does not block.
*/
#define NAV_HOST_UDP_PORT 9000
#define NAV_LCR1_MAGIC 0x3152434Cu
#define NAV_LCR1_VERSION 1
#define NAV_FLAG_INJECT_POSE 0x01u
#define NAV_FLAG_GOAL_FROM_TOF 0x80u
#define NAV_LCR1_GOAL_BYTES 16
#define NAV_LCR1_POSE_BYTES 44

#ifndef NAV_STUB_GOAL_X
#define NAV_STUB_GOAL_X 1.0f
#endif
#ifndef NAV_STUB_GOAL_Y
#define NAV_STUB_GOAL_Y 0.0f
#endif

void host_link_init(void);
void host_link_poll(void);
int host_link_udp_ready(void);

/* 0 on accept, -1 on reject. Rejects do not change the goal. */
int host_link_ingest(const unsigned char *data, int len);

void host_link_goal(float *x_m, float *y_m, const char **source);
int host_link_goal_is_stub(void);
