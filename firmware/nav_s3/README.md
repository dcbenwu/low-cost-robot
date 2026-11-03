# nav_s3 observation pipeline

20 Hz loop on the ESP32-S3: read sensors, build the 8-d observation the MuJoCo policy was trained on, run the embedded `dagger_r4` MLP, and apply **left / right** wheel duty. `policy_weights.h` is unchanged.

The board is not assembled. IMU, encoders, ToF, Wi-Fi, and the motor driver are **STUB**s. The observation math and the host goal packet are real.

`release/nav_s3.bin` is the previous stub image (the old `v`/`w` mapping). This tree does not rebuild it.

## Observation

Order matches `DiffDriveGoalEnv._obs` in `sim/m0_nav_demo/env_diffdrive.py`. Do not reorder.

| Index | Field | Unit | On the robot |
| --- | --- | --- | --- |
| 0 | gx | m | goal in the body frame |
| 1 | gy | m | goal in the body frame |
| 2 | sin(heading) | — | IMU yaw |
| 3 | cos(heading) | — | IMU yaw |
| 4 | vx | m/s | world-frame linear velocity (sim: free-joint qvel x) |
| 5 | vy | m/s | world-frame linear velocity (sim: free-joint qvel y) |
| 6 | ωL | rad/s | left wheel joint rate |
| 7 | ωR | rad/s | right wheel joint rate |

Body-frame goal, same rotation as the env (`heading` is the yaw used in sin/cos):

```
dx, dy = goal_world - position_world
c, s   = cos(-heading), sin(-heading)
gx     = c*dx - s*dy
gy     = s*dx + c*dy
```

A verified `reset(seed=0)` sample. Indexes 2–7 are the settle state (they repeat). `gx, gy` are that run's goal draw: the env samples the goal with the global NumPy RNG, not from the VL53L0X.

```
[-0.4063, 1.0991, -0.2376, 0.9714, 0.0417, 0.0, -0.5898, -0.5898]
```

## Goal source

World-frame goal `x, y` comes from the **host teleop or an offboard beacon** (`host_link`, UDP port 9000, little-endian `LCR1` packet). Until the first valid packet the goal is the compile-time **STUB** `NAV_STUB_GOAL_X/Y` (default 1 m, 0 m).

**The VL53L0X is not a goal source.** `sensors_read_tof` is obstacle/debug only and is not an input to `obs_build`. A packet with `NAV_FLAG_GOAL_FROM_TOF` is rejected. There is no build flag that copies a range into `obs[0:2]`.

`NAV_FLAG_INJECT_POSE` (sim bridge only) also sends pose, world velocity, and wheel rates so the firmware rebuilds the env observation. It does not localize the goal.

## Action

`act[0]` and `act[1]` are **left and right wheel torque commands**, clipped to signed PWM duty in `[-1, 1]`. They are not longitudinal speed and steering rate. LEDC / TB6612 writes are still a printf (no pins assigned).

`policy_forward` is the existing graph: normalize, tanh, tanh, **linear** head. Sim rollout scripts apply one more tanh before `env.step`. This tree does not change that forward. On the published reset sample the linear head is about `-5.44, +5.47`, so the signed duty saturates at `-1, +1`.

## Loop

`app_main` every 50 ms:

1. `host_link_poll` — UDP goal, or keep the STUB
2. `sensors_update` — STUB IMU yaw, encoder rates, wheel-odometry world velocity
3. `obs_build`
4. `policy_forward`
5. `apply_act` — left/right duty

Encoder world velocity (when no pose inject is held) assumes no sideslip: `v = 0.05 * (ωL + ωR) / 2`, then `vx = v cos(yaw)`, `vy = v sin(yaw)`. Positive rate on both wheels is taken as +body-x. That sign is not bench-calibrated.

## Checks

```bash
python3 firmware/nav_s3/test/run_x86_smoke.py
python3 sim/m0_nav_demo/sim_obs_bridge.py --steps 20
python3 sim/m0_nav_demo/sim_obs_bridge.py --steps 5 --udp 127.0.0.1:9000
```

The bridge prints the env's 8-d obs, checks a Python copy of the body-frame formula against `_obs`, and can UDP the `LCR1` goal/pose packet. The x86 smoke compiles `policy.c` with the new modules, checks a known obs against the same numpy graph (linear head), and checks the packet parser.
