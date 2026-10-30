"""Diagnostic: re-evaluate the saved BC policy with EXACTLY the same code path
as train_bc.py's eval (same MlpPolicy forward, same loop), to get the TRUE current
baseline for the warm-start starting point."""
import numpy as np
import torch
import torch.nn as nn

from env_diffdrive import DiffDriveGoalEnv


class MlpPolicy(nn.Module):
    def __init__(self, obs_dim=8, act_dim=2, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
            nn.Linear(hid, act_dim),
        )

    def forward(self, x):
        return torch.tanh(self.net(x))


def main(n_ep=100):
    d = np.load("bc_policy.npz.npy", allow_pickle=True).item()
    mu, sd = d["mu"], d["sd"]
    m = MlpPolicy()
    with torch.no_grad():
        m.net[0].weight.copy_(torch.as_tensor(d["W0"], dtype=torch.float32))
        m.net[0].bias.copy_(torch.as_tensor(d["b0"], dtype=torch.float32))
        m.net[2].weight.copy_(torch.as_tensor(d["W1"], dtype=torch.float32))
        m.net[2].bias.copy_(torch.as_tensor(d["b1"], dtype=torch.float32))
        m.net[4].weight.copy_(torch.as_tensor(d["W2"], dtype=torch.float32))
        m.net[4].bias.copy_(torch.as_tensor(d["b2"], dtype=torch.float32))

    env = DiffDriveGoalEnv()
    ok = 0
    dists, steps = [], []
    for ep in range(n_ep):
        obs, _ = env.reset()
        done = False
        used = 0
        while not done:
            xo = (np.asarray(obs, dtype=np.float32) - mu) / sd
            with torch.no_grad():
                a = m(torch.from_numpy(xo).unsqueeze(0)).numpy()[0]
            a = np.clip(a, -1, 1).astype(np.float32)
            obs, _, term, trunc, info = env.step(a)
            used += 1
            done = (term or trunc) or used >= 1000
        if term and info.get("success"):
            ok += 1
        dists.append(float(np.hypot(obs[0], obs[1])))
        steps.append(used)
    print(f"BC policy re-eval ({n_ep} ep): success={ok}/{n_ep}   "
          f"median_dist={np.median(dists):.3f}m   best={min(dists):.3f}m   "
          f"median_steps={np.median(steps):.0f}")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
