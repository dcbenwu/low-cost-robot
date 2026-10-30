"""Option 1: warm-start PPO from the 79% BC policy.

Mean net  = BC MLP (8->64->64->2) with loaded weights; action = tanh(t) (squashed).
variance  = learnable log_std (init log(0.25)).
value net = small MLP, random init (value fn doesn't need BC init).

Standard PPO with GAE + clipped surrogate + tanh-squashed gaussian log-prob.
Eval with deterministic mean.  Goal: beat/hold the 79% BC teacher.
"""
import numpy as np
import torch
import torch.nn as nn

from env_diffdrive import DiffDriveGoalEnv


class MeanNet(nn.Module):
    def __init__(self, W0, b0, W1, b1, W2, b2, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32) + 1e-8)
        self.lin0 = nn.Linear(8, 64); self.lin1 = nn.Linear(64, 64); self.lin2 = nn.Linear(64, 2)
        with torch.no_grad():
            self.lin0.weight.copy_(torch.as_tensor(W0, dtype=torch.float32))
            self.lin0.bias.copy_(torch.as_tensor(b0, dtype=torch.float32))
            self.lin1.weight.copy_(torch.as_tensor(W1, dtype=torch.float32))
            self.lin1.bias.copy_(torch.as_tensor(b1, dtype=torch.float32))
            self.lin2.weight.copy_(torch.as_tensor(W2, dtype=torch.float32))
            self.lin2.bias.copy_(torch.as_tensor(b2, dtype=torch.float32))

    def forward(self, obs):
        h = torch.tanh(self.lin0(obs))
        h = torch.tanh(self.lin1(h))
        return self.lin2(h)  # t (pre-tanh)


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 64), nn.Tanh(),
                                nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def log_prob_squash(mean, log_std, n_act):
    # mean: (batch, n_act); log_std: (n_act,) -> std: (n_act,)
    std = torch.exp(log_std)
    z = torch.randn_like(mean)
    t = mean + std * z
    a = torch.tanh(t)
    gp = -0.5 * ((t - mean) ** 2 / std ** 2 + torch.as_tensor(np.log(2 * np.pi), dtype=torch.float32) + 2 * log_std)
    corr = torch.log(1 - a ** 2 + 1e-6)
    logp = (gp - corr).sum(-1)
    return a, logp, t


def obs_norm(o, mu, sd):
    return (o - mu) / sd


def main(total_steps=2_000_000, n_steps=2048, K=10, mb=256, lr=2e-4,
         gamma=0.99, lam=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.02, max_steps=1000):
    import os
    if os.path.exists("bc_policy.npz"):
        d = dict(np.load("bc_policy.npz", allow_pickle=True))
    else:
        d = np.load("bc_policy.npz.npy", allow_pickle=True).item()
    mu, sd = d["mu"], d["sd"]
    mu_t, sd_t = torch.as_tensor(mu, dtype=torch.float32), torch.as_tensor(sd, dtype=torch.float32)

    mean_net = MeanNet(d["W0"], d["b0"], d["W1"], d["b1"], d["W2"], d["b2"], mu, sd)
    # frozen BC anchor: keeps PPO near the BC prior (only move where reward improves)
    bc_anchor = nn.Sequential(
        nn.Linear(8, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 2))
    with torch.no_grad():
        bc_anchor[0].weight.copy_(torch.as_tensor(d["W0"], dtype=torch.float32))
        bc_anchor[0].bias.copy_(torch.as_tensor(d["b0"], dtype=torch.float32))
        bc_anchor[2].weight.copy_(torch.as_tensor(d["W1"], dtype=torch.float32))
        bc_anchor[2].bias.copy_(torch.as_tensor(d["b1"], dtype=torch.float32))
        bc_anchor[4].weight.copy_(torch.as_tensor(d["W2"], dtype=torch.float32))
        bc_anchor[4].bias.copy_(torch.as_tensor(d["b2"], dtype=torch.float32))
    for p in bc_anchor.parameters():
        p.requires_grad_(False)
    val_net = ValueNet()
    log_std = nn.Parameter(torch.full((2,), float(np.log(0.25))))

    params = list(mean_net.parameters()) + list(val_net.parameters()) + [log_std]
    opt = torch.optim.Adam(params, lr=lr)
    dev = "cpu"
    mean_net.to(dev); val_net.to(dev)

    env = DiffDriveGoalEnv()

    def sample_and_step(n):
        obs0, _ = env.reset()
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
        for _ in range(n):
            o = torch.as_tensor(obs_norm(obs0, mu, sd), dtype=torch.float32)
            with torch.no_grad():
                mean = mean_net(o.unsqueeze(0)).squeeze(0)
                a, logp, _ = log_prob_squash(mean, log_std, 2)
                v = val_net(o.unsqueeze(0)).squeeze(0)
            act = a.numpy().clip(-1, 1)
            obs0, r, term, trunc, info = env.step(act)
            obs_buf.append(o.numpy()); act_buf.append(act)
            logp_buf.append(logp.item()); rew_buf.append(r)
            done_buf.append(float(term or trunc) if (term or trunc) else 0.0)
            val_buf.append(v.item())
            if term or trunc:
                obs0, _ = env.reset()
        obs_buf = np.asarray(obs_buf)
        act_buf = np.asarray(act_buf[:n])
        logp_buf = np.asarray(logp_buf[:n]); rew_buf = np.asarray(rew_buf[:n])
        done_buf = np.asarray(done_buf[:n]); val_buf = np.asarray(val_buf[:n])
        lastobs = torch.as_tensor(obs_norm(obs0, mu, sd), dtype=torch.float32)
        return dict(obs=obs_buf, act=act_buf, logp=logp_buf, rew=rew_buf,
                    done=done_buf, val=val_buf, lastobs=lastobs)

    def eval(n_ep=100):
        ok = 0; dists = []
        for _ in range(n_ep):
            ob, _ = env.reset()
            for _ in range(max_steps):
                o = torch.as_tensor(obs_norm(ob, mu, sd), dtype=torch.float32)
                with torch.no_grad():
                    mean = mean_net(o.unsqueeze(0)).squeeze(0)
                    a = torch.tanh(mean).numpy()
                obs_, r, term, trunc, info = env.step(np.clip(a, -1, 1).astype(np.float32))
                if term and info.get("success"):
                    ok += 1
                    break
                if term or trunc or obs_ is None:
                    break
                ob = obs_
            dists.append(float(np.hypot(ob[0], ob[1])))
        return ok / n_ep, float(np.median(dists))

    # ---- baseline: deterministic BC mean (no std) ----
    base_sr, base_d = eval()
    print(f"[baseline=deterministic BC mean] success={base_sr:.2f}  median_dist={base_d:.3f}", flush=True)

    # ---- value warm-start: fit V to BC-policy returns so PPO advantages start small ----
    print("[value warm-start] rolling out BC policy to fit value net...", flush=True)
    vX, vR = [], []
    for _ in range(300):
        ob, _ = env.reset()
        obs_row, rew_row = [], []
        for _ in range(max_steps):
            o = torch.as_tensor(obs_norm(ob, mu, sd), dtype=torch.float32)
            with torch.no_grad():
                mn = mean_net(o.unsqueeze(0)).squeeze(0)
            act = np.clip(torch.tanh(mn).numpy(), -1, 1).astype(np.float32)
            ob2, r, term, trunc, info = env.step(act)
            obs_row.append(o.numpy()); rew_row.append(float(r if r is not None else 0.0))
            if term or trunc:
                break
            ob = ob2
        m = len(obs_row)
        v = 0.0
        for t in range(m - 1, -1, -1):
            v = rew_row[t] + gamma * v
            vR.append(v)
        vX.extend(obs_row)
    vXt = torch.as_tensor(np.asarray(vX, dtype=np.float32))
    vRt = torch.as_tensor(np.asarray(vR, dtype=np.float32))
    vopt = torch.optim.Adam(val_net.parameters(), lr=3e-4)
    vlossf = nn.MSELoss()
    for ep in range(40):
        idx = torch.randperm(len(vXt)); tot = 0.0
        for s in range(0, len(vXt), 512):
            b = idx[s:s + 512]
            pred = val_net(vXt[b]).squeeze(-1)
            loss = vlossf(pred, vRt[b])
            vopt.zero_grad(); loss.backward(); vopt.step()
            tot += loss.item()
        if ep % 10 == 0 or ep == 39:
            print(f"  val-ws ep {ep} loss={tot / (len(vXt) / 512):.4f}", flush=True)
    with torch.no_grad():
        vv = val_net(vXt[:128]).squeeze(-1)
        print(f"  val-ws done | mean_return~{vRt.mean():.2f}  V~[{vv.min():.2f},{vv.max():.2f}]", flush=True)

    g = 0
    step_idx = 0
    while step_idx < total_steps:
        d_ = sample_and_step(n_steps)
        with torch.no_grad():
            lastv = val_net(d_["lastobs"].unsqueeze(0)).squeeze(0)
        # GAE
        n = len(d_["obs"])
        adv = np.zeros(n)
        lastg = 0.0
        obs_t = torch.as_tensor(d_["obs"], dtype=torch.float32)
        for t in range(n - 1, -1, -1):
            nextv = lastv.item() if t == n - 1 else d_["val"][t + 1]
            nw = 0.0 if d_["done"][t] else 1.0
            delta = d_["rew"][t] + gamma * nextv * nw - d_["val"][t]
            lastg = delta + gamma * lam * nw * lastg
            adv[t] = lastg
        ret = adv + d_["val"]
        # init logprob/mean for stored acts
        with torch.no_grad():
            mean0 = mean_net(obs_t).squeeze(0) if False else mean_net(obs_t)
        # convert to tensors for update
        act_t = torch.as_tensor(d_["act"], dtype=torch.float32)
        oldlogp = torch.as_tensor(d_["logp"], dtype=torch.float32)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)
        obs_t = torch.as_tensor(d_["obs"], dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # PPO epochs
        idx = np.arange(n)
        for _ in range(K):
            np.random.shuffle(idx)
            for s in range(0, n, mb):
                b = idx[s:s + mb]
                mean = mean_net(obs_t[b])
                _, logp_new, _ = log_prob_squash(mean, log_std, 2)
                ratio = torch.exp(logp_new - oldlogp[b])
                surr1 = ratio * adv_t[b]
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[b]
                pol_loss = -torch.min(surr1, surr2).mean()
                v = val_net(obs_t[b])
                v_loss = ((v - ret_t[b]) ** 2).mean()
                # anchor toward BC mean (pre-tanh t): stay near the prior
                anc = bc_anchor(obs_t[b]).detach()
                bc_reg = ((mean - anc) ** 2).sum(-1).mean()
                # entropy ~ mean logp (encourage exploration)
                ent = logp_new.mean()
                loss = pol_loss + vf_coef * v_loss - ent_coef * ent + 0.5 * bc_reg
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()
        step_idx += n
        if step_idx % 256000 == 0 or step_idx == total_steps:
            sr, md = eval(20)
            print(f"steps={step_idx//1000}k  running_eval(20ep) success={sr:.2f} median_dist={md:.3f}",
                  flush=True)

    sr, md = eval()
    print(f"\n=== PPO warm-start final (100 ep) ===")
    print(f"success={sr:.2f}  median_dist={md:.3f}   (baseline BC={base_sr:.2f})")
    # save
    sd_save = dict(mu=mu, sd=sd,
                   W0=mean_net.lin0.weight.detach().numpy(), b0=mean_net.lin0.bias.detach().numpy(),
                   W1=mean_net.lin1.weight.detach().numpy(), b1=mean_net.lin1.bias.detach().numpy(),
                   W2=mean_net.lin2.weight.detach().numpy(), b2=mean_net.lin2.bias.detach().numpy(),
                   log_std=log_std.detach().numpy())
    np.savez("ppo_warm_policy.npz", **sd_save)
    print("saved ppo_warm_policy.npz")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    a = ap.parse_args()
    main(total_steps=a.steps)
