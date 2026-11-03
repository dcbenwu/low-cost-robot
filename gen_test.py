#!/usr/bin/env python3
# 泛化 stress test：目标距离推到训练分布(0.6-1.5m)之外，比较 vanilla BC vs DAgger 谁掉得少
# spawn 安全：worker 用模块级函数 + 显式传 policy/rng（子进程内才设 GOAL_RANGE）
import sys
sys.path.insert(0, "/home/hock/lcr/m0_nav_demo")
import numpy as np
import env_diffdrive as E
from concurrent.futures import ProcessPoolExecutor


def load(path):
    d = np.load(path, allow_pickle=True).item()
    for k in ("mu","sd","W0","b0","W1","b1","W2","b2"):
        d[k] = d[k].astype(np.float64)
    return d


def infer(d, obs):
    obs = np.asarray(obs, dtype=np.float64)
    x = (obs - d["mu"]) / d["sd"]
    h0 = np.tanh(d["W0"] @ x + d["b0"])
    h1 = np.tanh(d["W1"] @ h0 + d["b1"])
    return np.tanh(d["W2"] @ h1 + d["b2"])


def run(policy, seed):
    env = E.DiffDriveGoalEnv()
    o, _ = env.reset(seed=seed)
    while True:
        o, _, t, tr, info = env.step(np.array(infer(policy, o), dtype=np.float32))
        if t or tr:
            ok = bool(info.get("success"))
            env.close()
            return ok


def _worker(item):
    seeds, policy, rng = item
    E.GOAL_RANGE = tuple(rng)
    return sum(1 for s in seeds if run(policy, s)), len(seeds)


def evald(policy, rng, N=500, procs=16):
    seeds = list(range(5000, 5000 + N))
    per = (N + procs - 1) // procs
    nchunk = max(1, (N + per - 1) // per)
    chunks = [seeds[i*per:(i+1)*per] for i in range(nchunk)]
    args = [(c, policy, tuple(rng)) for c in chunks]
    with ProcessPoolExecutor(max_workers=procs) as ex:
        res = list(ex.map(_worker, args))
    ok = sum(r[0] for r in res); tot = sum(r[1] for r in res)
    return f"{ok}/{tot} ({100*ok/tot:.0f}%)"


if __name__ == "__main__":
    P = "/home/hock/lcr/m0_nav_demo/"
    vc = load(P + "bc_policy_vanillaBC_backup.npz.npy")
    dg = load(P + "bc_policy_dagger_r2.npy")
    print("目标距离分布 -> 成功率（N=500）。DAgger 学的是自己踩过的状态，越界应更抗")
    print(f"{'分布':30s} | {'vanilla BC':>12s} | {'DAgger-best':>12s}")
    vfull = evald(vc, (0.6, 1.5)); dfull = evald(dg, (0.6, 1.5))
    print(f"{'训练域 0.6-1.5m':30s} | {vfull:>12s} | {dfull:>12s}")
    for label, band in [("略越界 0.3-0.9m", (0.3, 0.9)),
                          ("宽越界 0.4-2.2m", (0.4, 2.2)),
                          ("难尾 1.5-2.0m 远超", (1.5, 2.0))]:
        v = evald(vc, band); d = evald(dg, band)
        print(f"{label:30s} | {v:>12s} | {d:>12s}")
    print("唯一变量：训练法（纯模仿 vs DAgger），同一 MLP、同一 env、同一 obs 归一化口径")
