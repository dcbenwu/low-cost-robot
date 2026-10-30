"""M0: train a diff-drive goal-reaching policy with SB3 PPO (CPU), then evaluate.

Outputs (in ./out/):
  model.zip          - trained policy
  results.json       - training stats + evaluation metrics
  curve.csv          - per-epoch reward / episode length
  train.monitor      - SB3 per-episode training log

Usage:
  python train.py [--steps 300000] [--seed 0] [--n-eval 100]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)

    import stable_baselines3  # noqa: F401
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from env_diffdrive import DiffDriveGoalEnv

    # ------------------------------------------------------------- training
    env = Monitor(DiffDriveGoalEnv(), filename=str(out / "train"))
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        verbose=1,
    )
    model.learn(total_timesteps=args.steps, log_interval=4)

    # --------------------------------------------------------------- curve
    import numpy as np
    import csv as _csv

    mon = out / "train.monitor.csv"
    rewards: list[float] = []
    lengths: list[float] = []
    if mon.exists():
        with mon.open() as fh:
            lines = [l for l in fh if not l.lstrip().startswith("#")]
            rows = list(_csv.DictReader(lines))
            rewards = [float(r["r"]) for r in rows]
            lengths = [float(r["l"]) for r in rows]
    (out / "curve.csv").write_text(
        "episode,mean_reward,episode_length\n"
        + "".join(f"{i+1},{r:.4f},{l:.1f}\n" for i, (r, l) in enumerate(zip(rewards, lengths)))
    )

    # ---------------------------------------------------------------- eval
    eval_env = DiffDriveGoalEnv()
    successes = 0
    final_dists: list[float] = []
    for ep in range(args.n_eval):
        obs, _ = eval_env.reset()
        done = False
        last_dist = float("nan")
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = eval_env.step(act)
            last_dist = float(info.get("distance", float("nan")))
            if info.get("success"):
                successes += 1
        if not np.isnan(last_dist):
            final_dists.append(last_dist)
    eval_env.close()

    metrics = {
        "steps_trained": int(args.steps),
        "wall_time_s": round(time.time() - t0, 1),
        "eval_episodes": int(args.n_eval),
        "eval_success_episodes": successes,
        "eval_success_rate": round(successes / args.n_eval, 3),
        "eval_mean_final_distance_m": round(float(np.mean(final_dists)), 3) if final_dists else None,
        "torch": _torch_info(),
    }
    (out / "results.json").write_text(json.dumps(metrics, indent=2) + "\n")
    model.save(out / "model.zip")

    print("\n=== M0 RESULTS ===")
    print(json.dumps(metrics, indent=2))


def _torch_info() -> str:
    try:
        import torch

        return f"torch {torch.__version__} cuda={torch.cuda.is_available()}"
    except Exception as e:  # pragma: no cover
        return f"torch n/a ({e})"


if __name__ == "__main__":
    main()
