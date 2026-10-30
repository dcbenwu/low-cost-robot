"""Proportional-controller probe (both turn signs tested).

Is the reach-goal task actually learnable/solvable with a *simple* controller?
If one sign reaches the goal at high rate → env is fine, the bottleneck is
reward + policy optimization (GPU training will pay off).
If BOTH signs fail at ~0 → the env/dynamics/reward are the real problem.

Control law (per step):
  ang   = wrap(atan2(gy, gx))            # goal in body frame
  T     = clip(sign * K * ang, -A, +A)   # differential (turn) torque
  fwd   = FWD * max(0, 1 - |ang|/1.2)    # cut forward when not aligned
         * 0.4  when close (< 0.45 m)    # ease in
  a_l   = clip(fwd + T, -1, 1)
  a_r   = clip(fwd - T, -1, 1)
"""
import statistics

import numpy as np

from env_diffdrive import DiffDriveGoalEnv


def wrapped(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_controller(e, sign, n=100, K=2.0, FWD=0.95, A=1.0, max_steps=1000):
    ok = 0
    dists, steps = [], []
    for _ in range(n):
        obs, _ = e.reset()
        done = False
        used = 0
        while not done:
            gx, gy = obs[0], obs[1]
            dist = float(np.hypot(gx, gy))
            ang = wrapped(float(np.arctan2(gy, gx)))
            T = float(np.clip(sign * K * ang, -A, A))
            fwd = FWD * max(0.0, 1.0 - abs(ang) / 1.2)
            if dist < 0.45:
                fwd *= 0.4
            a_l = float(np.clip(fwd + T, -1.0, 1.0))
            a_r = float(np.clip(fwd - T, -1.0, 1.0))
            obs, r, term, trunc, info = e.step(np.array([a_l, a_r]))
            used += 1
            done = (term or trunc) or used >= max_steps
        if term and info.get("success"):
            ok += 1
        dists.append(dist)
        steps.append(used)
    return ok, statistics.median(dists), statistics.median(steps), min(dists)


def main():
    e = DiffDriveGoalEnv()
    for s in (+1, -1):
        ok, md, ms, best = run_controller(e, s)
        print(f"sign={s:+d}: success={ok:3d}/100  "
              f"median_dist={md:.3f}m  median_steps={ms:4.0f}  best={best:.3f}m")


if __name__ == "__main__":
    main()
