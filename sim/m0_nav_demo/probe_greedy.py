"""Probe: deterministic greedy controller — is the task physically solvable?

Policy (no learning):
  ang_err = wrap(goal_dir_body - 0)
  if |ang_err| > 0.35:  turn  (differential torque in that direction)
  elif dist > 0.25:     forward
  else:                 brake (let it drift into the radius)
"""
import sys

import numpy as np

from env_diffdrive import DiffDriveGoalEnv, GOAL_RADIUS


def wrapped(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def main() -> None:
    e = DiffDriveGoalEnv()
    ok = 0
    dists = []
    steps_used = []
    for ep in range(100):
        obs, _ = e.reset()
        p0 = obs[3:5]
        done = False
        used = 0
        while not done:
            gx, gy = obs[0], obs[1]
            dist = float(np.hypot(gx, gy))
            ang = float(np.arctan2(gy, gx))
            if abs(ang) > 0.35:
                a = np.array([1.0, -1.0]) * np.sign(ang)
            elif dist > max(0.3, GOAL_RADIUS * 2):
                a = np.array([0.8, 0.8])
            else:
                a = np.array([0.2, 0.2])
            obs, r, term, trunc, info = e.step(a)
            done = term or trunc
            used += 1
        if term and info.get("success"):
            ok += 1
        dists.append(dist)
        steps_used.append(used)
    import statistics

    print("PROBE greedy controller (100 eps):")
    print(f"  success: {ok}/100")
    print(f"  median final dist: {statistics.median(dists):.3f} m")
    print(f"  median steps used: {statistics.median(steps_used):.0f} (budget 400)")
    print(f"  best dist: {min(dists):.3f} m")


if __name__ == "__main__":
    main()
