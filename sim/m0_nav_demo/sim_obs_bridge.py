#!/usr/bin/env python3
"""Print (and optionally UDP-send) the 8-d obs from DiffDriveGoalEnv.

The layout is DiffDriveGoalEnv._obs / firmware obs_builder, in this order:

    0 gx, 1 gy     goal in the body frame, meters
    2 sin(yaw), 3 cos(yaw)
    4 vx, 5 vy     world-frame free-joint qvel, m/s
    6 wL, 7 wR     wheel joint rates, rad/s

The world goal is an env target (on the robot: host teleop or an offboard
beacon). It is not a VL53L0X reading. With --udp HOST:PORT each step sends
the firmware LCR1 packet (world goal + pose inject) defined in
firmware/nav_s3/main/host_link.h. Stdout still carries the env obs.

A verified reset(seed=0) sample (gx, gy are that process's goal draw; indexes
2-7 are the deterministic settle state):

    [-0.4063, 1.0991, -0.2376, 0.9714, 0.0417, 0.0, -0.5898, -0.5898]
"""

from __future__ import annotations

import argparse
import math
import re
import socket
import struct
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HOST_LINK_H = ROOT / "firmware" / "nav_s3" / "main" / "host_link.h"

# Rounded sample from a local reset(seed=0). Not a ToF measurement.
REF_RESET = np.array(
    [-0.4063, 1.0991, -0.2376, 0.9714, 0.0417, 0.0, -0.5898, -0.5898],
    dtype=np.float64,
)


def load_link_constants(path: Path = HOST_LINK_H) -> dict:
    text = path.read_text()

    def define(name: str) -> str:
        match = re.search(rf"^#define\s+{name}\s+(\S+)", text, re.M)
        if not match:
            raise SystemExit(f"missing #define {name} in {path}")
        return match.group(1).rstrip("u").rstrip("f")

    return {
        "magic": int(define("NAV_LCR1_MAGIC"), 0),
        "version": int(define("NAV_LCR1_VERSION"), 0),
        "flag_inject": int(define("NAV_FLAG_INJECT_POSE"), 0),
        "flag_tof": int(define("NAV_FLAG_GOAL_FROM_TOF"), 0),
        "port": int(define("NAV_HOST_UDP_PORT"), 0),
        "goal_bytes": int(define("NAV_LCR1_GOAL_BYTES"), 0),
        "pose_bytes": int(define("NAV_LCR1_POSE_BYTES"), 0),
        "stub_x": float(define("NAV_STUB_GOAL_X")),
        "stub_y": float(define("NAV_STUB_GOAL_Y")),
    }


def build_obs(pos, yaw, vel, goal, wheels) -> np.ndarray:
    """Python copy of obs_builder.c / env_diffdrive._obs (float64, then float32)."""
    pos = np.asarray(pos, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    wheels = np.asarray(wheels, dtype=np.float64)
    dx, dy = float(goal[0] - pos[0]), float(goal[1] - pos[1])
    c, s = math.cos(-float(yaw)), math.sin(-float(yaw))
    gx = c * dx - s * dy
    gy = s * dx + c * dy
    return np.concatenate(
        [[gx, gy], [math.sin(float(yaw)), math.cos(float(yaw))], vel, wheels]
    ).astype(np.float32)


def world_delta_from_body(gx: float, gy: float, yaw: float) -> tuple[float, float]:
    """Inverse of the body-frame rotation in build_obs."""
    c, s = math.cos(yaw), math.sin(yaw)
    return c * gx - s * gy, s * gx + c * gy


def pack_lcr1(goal_xy, pose=None, wheels=(0.0, 0.0), flags: int = 0, constants: dict | None = None) -> bytes:
    c = constants or load_link_constants()
    if pose is not None:
        flags |= c["flag_inject"]
        x, y, yaw, vx, vy = pose
        wl, wr = wheels
        blob = struct.pack(
            "<IBBHfffffffff",
            c["magic"],
            c["version"],
            flags & 0xFF,
            0,
            float(goal_xy[0]),
            float(goal_xy[1]),
            float(x),
            float(y),
            float(yaw),
            float(vx),
            float(vy),
            float(wl),
            float(wr),
        )
    else:
        blob = struct.pack(
            "<IBBHff",
            c["magic"],
            c["version"],
            flags & 0xFF,
            0,
            float(goal_xy[0]),
            float(goal_xy[1]),
        )
    if pose is not None and len(blob) != c["pose_bytes"]:
        raise RuntimeError(f"pose packet is {len(blob)} bytes, header says {c['pose_bytes']}")
    if pose is None and len(blob) != c["goal_bytes"]:
        raise RuntimeError(f"goal packet is {len(blob)} bytes, header says {c['goal_bytes']}")
    return blob


def _fmt(obs: np.ndarray) -> str:
    return " ".join(f"{float(v):.8f}" for v in np.asarray(obs).reshape(-1))


def formula_self_check() -> None:
    """Cases that do not need MuJoCo. Same formulas as obs_builder.c."""
    ident = build_obs([0.0, 0.0], 0.0, [0.0, 0.0], [1.0, 0.0], [0.0, 0.0])
    if not np.allclose(ident, [1, 0, 0, 1, 0, 0, 0, 0], atol=1e-6):
        raise SystemExit(f"identity formula mismatch {ident}")

    yaw = math.pi / 2
    turned = build_obs([0.0, 0.0], yaw, [0.1, -0.2], [1.0, 0.0], [0.25, -0.5])
    expect = np.array([0.0, -1.0, 1.0, 0.0, 0.1, -0.2, 0.25, -0.5], dtype=np.float32)
    if not np.allclose(turned, expect, atol=1e-6):
        raise SystemExit(f"yaw90 formula mismatch {turned}")

    shifted = build_obs([1.0, 2.0], 0.0, [0.0, 0.0], [1.0, 2.0], [3.0, 4.0])
    if not np.allclose(shifted, [0, 0, 0, 1, 0, 0, 3, 4], atol=1e-6):
        raise SystemExit(f"shifted formula mismatch {shifted}")

    # Round-trip: body goal -> world delta -> body goal.
    dx, dy = world_delta_from_body(-0.4063, 1.0991, -0.24)
    back = build_obs([0.25, -0.5], -0.24, [0.0, 0.0], [0.25 + dx, -0.5 + dy], [0.0, 0.0])
    if not np.allclose(back[:2], [-0.4063, 1.0991], atol=1e-5):
        raise SystemExit(f"body/world round-trip mismatch {back[:2]}")
    print("FORMULA PASS")


def _state(env):
    from env_diffdrive import _quat_heading

    pos = env.data.xpos[env.body_id][:2].copy()
    yaw = _quat_heading(env.data.xquat[env.body_id])
    vel = env.data.qvel[env.body_vel_adr : env.body_vel_adr + 2].copy()
    wheels = np.array(
        [env.data.qvel[env.model.jnt_dofadr[j]] for j in env.wheel_ids],
        dtype=np.float64,
    )
    return pos, yaw, vel, wheels


def check_published_settle(env) -> None:
    """Indexes 2-7 of reset(seed=0) match the verified sample; gx, gy are a draw."""
    pos, yaw, vel, wheels = _state(env)
    tail = np.array(
        [math.sin(yaw), math.cos(yaw), vel[0], vel[1], wheels[0], wheels[1]],
        dtype=np.float64,
    )
    if not np.allclose(tail, REF_RESET[2:], atol=5e-4):
        raise SystemExit(f"seed=0 settle mismatch tail={tail} ref={REF_RESET[2:]}")
    dx, dy = world_delta_from_body(float(REF_RESET[0]), float(REF_RESET[1]), yaw)
    built = build_obs(pos, yaw, vel, pos + np.array([dx, dy]), wheels)
    if not np.allclose(built, REF_RESET.astype(np.float32), atol=5e-4):
        raise SystemExit(f"published sample is not the body-frame formula {built}")
    print("SETTLE PASS", _fmt(REF_RESET))


def pursuit(obs: np.ndarray) -> np.ndarray:
    """Short-horizon steering used only to move the episode. Not the MLP."""
    gx, gy = float(obs[0]), float(obs[1])
    ang = math.atan2(gy, gx)
    turn = max(-1.0, min(1.0, -0.8 * ang))
    fwd = 0.9 * max(0.0, math.cos(ang))
    return np.array(
        [max(-1.0, min(1.0, fwd + turn)), max(-1.0, min(1.0, fwd - turn))],
        dtype=np.float32,
    )


def run_env(steps: int, seed: int, udp: str | None, episode: bool) -> None:
    warnings.filterwarnings("ignore", module="gymnasium")
    try:
        from env_diffdrive import DiffDriveGoalEnv
    except ModuleNotFoundError as exc:
        print(f"SKIP_ENV {exc}")
        return

    # The env draws the goal from the global RNG (np.random), not self.np_random.
    np.random.seed(seed)
    env = DiffDriveGoalEnv()
    obs, _ = env.reset(seed=seed)
    check_published_settle(env)
    obs = env._obs()

    constants = load_link_constants()
    sock = None
    addr = None
    if udp:
        host, port_s = udp.rsplit(":", 1)
        addr = (host, int(port_s))
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    limit = 1000 if episode else steps
    mismatches = 0
    info: dict = {}
    for i in range(limit):
        pos, yaw, vel, wheels = _state(env)
        built = build_obs(pos, yaw, vel, env.goal, wheels)
        if not np.array_equal(built, obs):
            mismatches += 1
            if mismatches <= 3:
                print("PARITY_DIFF", _fmt(obs), "built", _fmt(built))
        if (not episode) or i < 3 or i + 1 == limit:
            print("ENV_OBS", _fmt(obs))
        if sock is not None and addr is not None:
            pkt = pack_lcr1(
                env.goal,
                (pos[0], pos[1], yaw, vel[0], vel[1]),
                (wheels[0], wheels[1]),
                constants=constants,
            )
            sock.sendto(pkt, addr)
        if episode and i + 1 >= limit:
            break
        if not episode and i + 1 >= steps:
            break
        nxt, _reward, term, trunc, info = env.step(pursuit(obs))
        obs = nxt
        if term or trunc:
            print("ENV_OBS", _fmt(obs))
            break
    env.close()
    if sock is not None:
        sock.close()
    if mismatches:
        raise SystemExit(f"obs_builder formula mismatched env._obs on {mismatches} steps")
    dist = float(np.hypot(obs[0], obs[1]))
    ended = " success=1" if info.get("success") else ""
    print(f"PARITY PASS goal_dist_m={dist:.4f}{ended}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20, help="control steps after reset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode", action="store_true", help="run until the env ends")
    parser.add_argument("--udp", default=None, help="host:port for LCR1 goal/pose packets")
    parser.add_argument("--formula-only", action="store_true")
    args = parser.parse_args()
    formula_self_check()
    if args.formula_only:
        return
    run_env(args.steps, args.seed, args.udp, args.episode)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
