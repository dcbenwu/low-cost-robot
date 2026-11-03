#!/usr/bin/env python3
# 并行 controller 竞技场：teacher vs 改进弧跟踪 controller，同一批 episode 对比 success/median_steps
import sys, math
sys.path.insert(0, "/home/hock/lcr/m0_nav_demo")
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import env_diffdrive as E


def _run_episode(seed, ctl):
    env = E.DiffDriveGoalEnv()
    o, _ = env.reset(seed=seed)
    success = False
    while True:
        a = np.array(ctl(o), dtype=np.float32)
        o, _, t, tr, info = env.step(a)
        if t or tr:
            success = bool(info.get("success"))
            steps = env.steps
            dist = float(info.get("distance", -1))
            break
    env.close()
    return (1 if success else 0), steps, dist


def _batch(args):
    seeds, ctl = args
    out = [_run_episode(s, ctl) for s in seeds]
    ok = sum(x[0] for x in out)
    steps = np.array([x[1] for x in out], dtype=np.int64)
    return ok, len(out), int(np.median(steps)), steps.tolist()


def measure(ctl, N, nprocs=16):
    seeds = list(range(1000, 1000 + N))
    per = (N + nprocs - 1) // nprocs
    nchunk = max(1, (N + per - 1) // per)
    chunks = [seeds[i * per:(i + 1) * per] for i in range(nchunk)]
    with ProcessPoolExecutor(max_workers=nprocs) as ex:
        res = list(ex.map(_batch, [(c, ctl) for c in chunks]))
    ok = sum(r[0] for r in res)
    tot = sum(r[1] for r in res)
    all_steps = np.concatenate([np.array(r[3]) for r in res])
    med = int(np.median(all_steps))
    return ok, tot, med


def teacher(obs, FWD=0.9, NEAR=0.15):
    gx, gy = float(obs[0]), float(obs[1])
    dist, ang = math.hypot(gx, gy), math.atan2(gy, gx)
    fwd = FWD * max(0.0, math.cos(ang)) * min(1.0, dist)
    if dist < NEAR:
        fwd *= 0.5
    T = -0.8 * ang
    if dist < NEAR:
        T *= 0.6
    return [float(np.clip(fwd + T, -1, 1)), float(np.clip(fwd - T, -1, 1))]


def arc(obs, KF=1.0, KT=1.3, NEAR=0.18, FLOOR=0.30):
    gx, gy = float(obs[0]), float(obs[1])
    dist, ang = math.hypot(gx, gy), math.atan2(gy, gx)
    facing = (1.0 + math.cos(ang)) / 2.0          # 1 forward, 0 behind
    base = min(1.0, dist)
    fwd = KF * base * (FLOOR + (1 - FLOOR) * facing)
    if dist < NEAR:
        fwd *= (0.45 + 0.55 * (dist / NEAR))      # gentle taper, never a dead stall
    T = -KT * ang
    if dist < NEAR:
        T *= 0.85
    return [float(np.clip(fwd + T, -1, 1)), float(np.clip(fwd - T, -1, 1))]


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    for name, ctl in [("teacher", teacher), ("arc", arc)]:
        ok, tot, med = measure(ctl, N)
        print(f"{name}: success={ok}/{tot} ({100*ok/tot:.0f}%)  median_steps={med}")


if __name__ == "__main__":
    main()
