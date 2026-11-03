"""dagger.py — DAgger (Dataset Aggregation) for the low-cost robot M0 task.

Attacks the root cause that keeps vanilla behavior cloning at its ~64% ceiling:
COMPOUNDING ERROR.  When a BC policy rolls out, small per-step drift pushes it
into states the teacher never visited during data collection, where it has no
supervision and fails.  DAgger fixes this by labeling the LEARNER'S OWN visited
states with the expert action and aggregating them into the training set over
rounds (Ross et al. 2011).

Expert = the sweep-found 78% analytic controller (collect_bc.best_actions), an
analytic oracle we can query at ANY state — exactly the property DAgger needs.

Loop (R rounds):
  1. Roll out the current policy in the env.
  2. At every visited state s, record (s, expert(s)).
  3. Aggregate: seed BC data + all DAgger-labeled pairs (retained across rounds).
  4. Retrain the MLP on the aggregate, save, evaluate in-env (deterministic).

Goal: climb from ~64% (vanilla BC) toward / above the teacher's ~78%.
Pure CPU (Mujoco sim + small MLP) — no GPU, safe alongside the miner.
"""
import os
import numpy as np
import torch
import torch.nn as nn

from env_diffdrive import DiffDriveGoalEnv

# --------------------------------------------------------------------------- expert
def _wrapped(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def expert(obs: np.ndarray) -> np.ndarray:
    """Analytic oracle, verbatim from collect_bc.py best_actions (best sweep)."""
    gx, gy = float(obs[0]), float(obs[1])
    dist = float(np.hypot(gx, gy))
    ang = _wrapped(float(np.arctan2(gy, gx)))
    T = float(np.clip(-0.8 * ang, -1.0, 1.0))
    fwd = 0.9 * max(0.0, float(np.cos(ang)))
    if dist < 0.6:
        fwd *= 0.5
        T *= 0.6
    return np.array([np.clip(fwd + T, -1, 1), np.clip(fwd - T, -1, 1)], dtype=np.float32)


# --------------------------------------------------------------------------- policy
class MlpPolicy(nn.Module):
    """Identical architecture to train_bc.py / eval_bc.py (8-64-64-2, tanh)."""

    def __init__(self, obs_dim=8, act_dim=2, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
            nn.Linear(hid, act_dim),
        )

    def forward(self, x):
        return torch.tanh(self.net(x))


def _act(m, obs, mu, sd) -> np.ndarray:
    xo = (np.asarray(obs, dtype=np.float32) - mu) / sd
    with torch.no_grad():
        a = m(torch.from_numpy(xo).unsqueeze(0)).numpy()[0]
    return np.clip(a, -1, 1).astype(np.float32)


def rollout_collect(m, mu, sd, env, n_ep, max_steps=1000, only_failed=False):
    """Drive the current policy; at each visited state record (state, expert(state)).

    DAgger's core step: labels are the expert's action AT THE POLICY'S OWN STATE
    (not the expert's trajectory) — this covers the off-distribution states the
    policy actually visits.  Returns (StackS, StackA, rollout_success_rate).
    """
    S, A = [], []
    ok = 0
    for i in range(n_ep):
        obs, _ = env.reset()
        o_ep, a_ep = [], []
        succ = False
        while True:
            act = _act(m, obs, mu, sd)
            o_ep.append(obs)
            a_ep.append(expert(obs))  # oracle label at the policy's state
            obs, _, term, trunc, info = env.step(act)
            if term and info.get("success"):
                succ = True
                break
            if term or trunc:
                break
        if succ:
            ok += 1
        if (not only_failed) or (not succ):
            S.append(np.asarray(o_ep, np.float32))
            A.append(np.asarray(a_ep, np.float32))
        if (i + 1) % 100 == 0:
            print(f"    rollout {i+1}/{n_ep}  running_success={ok}/{i+1}", flush=True)
    if S:
        return np.concatenate(S, 0), np.concatenate(A, 0), ok / n_ep
    return (np.zeros((0, 8), np.float32), np.zeros((0, 2), np.float32), ok / n_ep)


def retrain(seed_x, seed_y, dag_x, dag_y, epochs=150, bs=512, lr=3e-4):
    X = np.concatenate([seed_x, dag_x], 0).astype(np.float32)
    Y = np.concatenate([seed_y, dag_y], 0).astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    model = MlpPolicy()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt, Yt = torch.from_numpy(Xs.astype(np.float32)), torch.from_numpy(Y)
    n = len(Xt)
    model.train()
    for ep in range(1, epochs + 1):
        idx = torch.randperm(n)
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            loss = lossf(model(Xt[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return model, mu, sd, n


def eval_policy(m, mu, sd, env, n_ep=100):
    ok = 0
    dists = []
    for _ in range(n_ep):
        obs, _ = env.reset()
        done = False
        while not done:
            a = _act(m, obs, mu, sd)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        if term and info.get("success"):
            ok += 1
        dists.append(float(np.hypot(obs[0], obs[1])))
    return ok, float(np.median(dists))


def save_np(m, mu, sd, path):
    """Save in the EXACT format train_bc.py / eval_bc.py use (np.save of dict)."""
    np.save(path, dict(
        mu=mu, sd=sd,
        W0=m.net[0].weight.detach().numpy(), b0=m.net[0].bias.detach().numpy(),
        W1=m.net[2].weight.detach().numpy(), b1=m.net[2].bias.detach().numpy(),
        W2=m.net[4].weight.detach().numpy(), b2=m.net[4].bias.detach().numpy(),
    ), allow_pickle=True)


def load_np(path):
    d = np.load(path, allow_pickle=True).item()
    m = MlpPolicy()
    with torch.no_grad():
        m.net[0].weight.copy_(torch.as_tensor(d["W0"], dtype=torch.float32))
        m.net[0].bias.copy_(torch.as_tensor(d["b0"], dtype=torch.float32))
        m.net[2].weight.copy_(torch.as_tensor(d["W1"], dtype=torch.float32))
        m.net[2].bias.copy_(torch.as_tensor(d["b1"], dtype=torch.float32))
        m.net[4].weight.copy_(torch.as_tensor(d["W2"], dtype=torch.float32))
        m.net[4].bias.copy_(torch.as_tensor(d["b2"], dtype=torch.float32))
    return m, np.asarray(d["mu"], np.float32), np.asarray(d["sd"], np.float32)


def main(rounds: int = 5, roll_eps: int = 300, only_failed: bool = False,
         start: str = "bc_policy.npz.npy"):
    torch.manual_seed(0)
    seed = np.load("bc_data.npz")
    seed_x, seed_y = seed["obs"].astype(np.float32), seed["act"].astype(np.float32)
    m, mu, sd = load_np(start)
    env = DiffDriveGoalEnv()

    base_ok, base_med = eval_policy(m, mu, sd, env, 100)
    print(f"[baseline] {os.path.basename(start)}: success={base_ok}/100 "
          f"median={base_med:.3f}m   (seed {seed_x.shape[0]} pairs)", flush=True)

    dag_x, dag_y = [], []
    best = (-1, "")
    for r in range(1, rounds + 1):
        print(f"\n=== DAgger round {r}/{rounds} ===", flush=True)
        sx, sy, roll_sr = rollout_collect(m, mu, sd, env, roll_eps, only_failed=only_failed)
        dag_x.append(sx); dag_y.append(sy)
        AdagX = np.concatenate(dag_x, 0); AdagY = np.concatenate(dag_y, 0)
        print(f"    rollout success={roll_sr:.2f}  +{sx.shape[0]} labeled states "
              f"(agg {AdagX.shape[0]} + seed {seed_x.shape[0]})", flush=True)
        torch.manual_seed(r)
        m, mu, sd, n = retrain(seed_x, seed_y, AdagX, AdagY)
        ok, med = eval_policy(m, mu, sd, env, 100)
        out = f"bc_policy_dagger_r{r}.npy"
        save_np(m, mu, sd, out)
        print(f"    DAgger r{r}: success={ok}/100 median={med:.3f}m  (train n={n})", flush=True)
        if ok > best[0]:
            best = (ok, out)

    print(f"\n=== DAgger summary: baseline {base_ok}/100 -> "
          f"BEST {best[0]}/100 ({os.path.basename(best[1])}) ===", flush=True)
    return best


if __name__ == "__main__":
    import sys
    r = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    e = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    main(rounds=r, roll_eps=e)
