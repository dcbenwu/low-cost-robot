"""Train a small MLP policy by behavior-cloning the 78% controller, then
evaluate it in the env (deterministic) to get an honest in-env success rate.

This gives us a real neural policy that "works" (~teacher level), which is the
deliverable.  It can later be polished by a short PPO warm-start (RL fine-tune)
if we want to beat the teacher.
"""
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
        return torch.tanh(self.net(x))  # actions in [-1, 1]


def main(epochs=200, bs=512, lr=3e-4):
    d = np.load("bc_data.npz")
    X, Y = d["obs"], d["act"]
    print(f"data: X={X.shape} Y={Y.shape}")
    # standardize obs (actions already in [-1,1])
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = ((X - mu) / sd).astype(np.float32)
    print(f"obs mu={np.round(mu,3)} sd={np.round(sd,3)}")

    dev = "cpu"
    model = MlpPolicy()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt = torch.from_numpy(Xs)
    Yt = torch.from_numpy(Y)
    n = len(Xt)
    model.train()
    for ep in range(1, epochs + 1):
        idx = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            pred = model(Xt[b])
            loss = lossf(pred, Yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if ep % 20 == 0 or ep == 1:
            print(f"ep {ep:3d}  loss={tot/(n/bs):.5f}")

    # ---- evaluate the NN policy in the env (deterministic) ----
    env = DiffDriveGoalEnv()
    ok = 0
    dists, steps = [], []
    for ep in range(100):
        obs, _ = env.reset()
        done = False
        used = 0
        while not done:
            xo = (np.asarray(obs, dtype=np.float32) - mu) / sd
            with torch.no_grad():
                a = model(torch.from_numpy(xo).unsqueeze(0)).numpy()[0]
            a = np.clip(a, -1, 1).astype(np.float32)
            obs, _, term, trunc, info = env.step(a)
            used += 1
            done = (term or trunc) or used >= 1000
        if term and info.get("success"):
            ok += 1
        dists.append(float(np.hypot(obs[0], obs[1])))
        steps.append(used)

    np.save("bc_policy.npz", dict(mu=mu, sd=sd,
            W0=model.net[0].weight.detach().numpy(),
            b0=model.net[0].bias.detach().numpy(),
            W1=model.net[2].weight.detach().numpy(),
            b1=model.net[2].bias.detach().numpy(),
            W2=model.net[4].weight.detach().numpy(),
            b2=model.net[4].bias.detach().numpy()))
    print(f"\n=== BC NN in-env eval (100 ep) ===")
    print(f"success={ok}/100   median_dist={np.median(dists):.3f}m   "
          f"median_steps={np.median(steps):.0f}   best={min(dists):.3f}m")


if __name__ == "__main__":
    main()
