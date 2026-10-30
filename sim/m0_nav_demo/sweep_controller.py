"""Gain sweep for the diff-drive goal-reaching task.

Goal: find a simple, deterministic controller that reaches the goal at high
success rate so we (a) hand the boss a working robot controller and (b) use it
as a "good policy" reference to diagnose why PPO under-performs.

Controller families (sign = -1 confirmed correct in probe_proportional.py):
  L  (linear fwd cut) : fwd = FWD * max(0, 1 - |ang|/ANGCUT)
  C  (cosine arc)     : fwd = FWD * cos(ang)          (natural turning arc)
Both apply:
  T   = clip(-K * ang, -A, +A)         # turn authority
  near goal (dist<CLOSE): fwd *= EASE  # settle / avoid overshoot
  a_l = clip(fwd + T, -1, 1);  a_r = clip(fwd - T, -1, 1)

We grid over  K, A, FWD, ANGCUT, CLOSE, EASE  and keep the best few.
"""
import itertools
import statistics

import numpy as np

from env_diffdrive import DiffDriveGoalEnv


def wrapped(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def run(e, n=100, *, family="L", K=1.0, A=0.5, FWD=0.8, ANGCUT=1.0,
        CLOSE=0.35, EASE=0.3, max_steps=1000):
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
            T = float(np.clip(-K * ang, -A, A))
            if family == "L":
                fwd = FWD * max(0.0, 1.0 - abs(ang) / ANGCUT)
            else:  # cosine arc
                fwd = FWD * max(0.0, float(np.cos(ang)))
            if dist < CLOSE:
                fwd *= EASE
                T *= 0.6
            a_l = float(np.clip(fwd + T, -1.0, 1.0))
            a_r = float(np.clip(fwd - T, -1.0, 1.0))
            obs, _, term, trunc, info = e.step(np.array([a_l, a_r]))
            used += 1
            done = (term or trunc) or used >= max_steps
        if term and info.get("success"):
            ok += 1
        dists.append(dist)
        steps.append(used)
    return ok, statistics.median(dists), statistics.median(steps)


def main():
    e = DiffDriveGoalEnv()
    rows = []
    K_grid = [0.4, 0.8, 1.4, 2.2]
    A_grid = [0.3, 0.5, 0.8, 1.0]
    FWD_grid = [0.7, 0.9]
    for fam in ("L", "C"):
        for K, A, FWD in itertools.product(K_grid, A_grid, FWD_grid):
            angs = [1.0] if fam == "L" else [1.0]
            for angc in angs:
                for close, ease in [(0.3, 0.3), (0.45, 0.25), (0.6, 0.5)]:
                    n_ok, md, ms = run(e, K=K, A=A, FWD=FWD, ANGCUT=angc,
                                       CLOSE=close, EASE=ease, family=fam)
                    rows.append((n_ok, fam, K, A, FWD, angc, close, ease, md, ms))
    rows.sort(key=lambda r: (-r[0], r[8]))
    print("top 12 by success:")
    print(f"{'ok':>3} {'fam':>4} {'K':>4} {'A':>4} {'FWD':>4} {'cut':>4} "
          f"{'cl':>4} {'ease':>5} {'medD':>6} {'medS':>5}")
    for r in rows[:12]:
        n_ok, fam, K, A, FWD, angc, close, ease, md, ms = r
        print(f"{n_ok:>3} {fam:>4} {K:>4.1f} {A:>4.2f} {FWD:>4.2f} {angc:>4.1f} "
              f"{close:>4.2f} {ease:>5.2f} {md:>6.3f} {ms:>5.0f}")


if __name__ == "__main__":
    main()
