"""Collect behavior-cloning data from the 78% deterministic controller.

Uses the best config from sweep_controller.py (family=C cosine arc, K=0.8,
A=1.0, FWD=0.9, CLOSE=0.6, EASE=0.5) to roll out episodes in the env and
record (obs, action) pairs.  This is the "teacher" for BC.
"""
import numpy as np

from env_diffdrive import DiffDriveGoalEnv


def wrapped(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def best_actions(obs):
    gx, gy = obs[0], obs[1]
    dist = float(np.hypot(gx, gy))
    ang = wrapped(float(np.arctan2(gy, gx)))
    T = float(np.clip(-0.8 * ang, -1.0, 1.0))
    fwd = 0.9 * max(0.0, float(np.cos(ang)))
    if dist < 0.6:
        fwd *= 0.5
        T *= 0.6
    a_l = float(np.clip(fwd + T, -1.0, 1.0))
    a_r = float(np.clip(fwd - T, -1.0, 1.0))
    return np.array([a_l, a_r], dtype=np.float32)


def main(n_ep=400, max_steps=1000):
    env = DiffDriveGoalEnv()
    obs_list, act_list, succ_list = [], [], []
    ok = 0
    for i in range(n_ep):
        obs, _ = env.reset()
        o, a = [], []
        s = False
        for _ in range(max_steps):
            act = best_actions(obs)
            o.append(obs)          # state
            a.append(act)          # action chosen at that state (1:1 pair)
            obs, r, term, trunc, info = env.step(act)
            if term and info.get("success"):
                s = True
                ok += 1
                break
            if term or trunc or obs is None:
                break
        obs_list.append(np.asarray(o, dtype=np.float32))
        act_list.append(np.asarray(a, dtype=np.float32))
        succ_list.append(1.0 if s else 0.0)
        if (i + 1) % 50 == 0:
            print(f"ep {i+1}/{n_ep}  running_success={ok}/{i+1}")
    obs_stack = np.concatenate(obs_list, axis=0)
    act_stack = np.concatenate(act_list, axis=0)
    succ = np.asarray(succ_list, dtype=np.float32)
    out = dict(obs=obs_stack, act=act_stack)
    np.savez("bc_data.npz", **out)
    print(f"saved bc_data.npz: {obs_stack.shape[0]} pairs, "
          f"success_episodes={ok}/{n_ep} (rate {ok/n_ep:.2f}), "
          f"success_frac_flags={float(succ.mean()):.2f} of episodes")
    print(f"obs sample: {obs_stack[0]}")
    print(f"act sample: {act_stack[0]}")


if __name__ == "__main__":
    main()
