# M0 导航线 — seed=0 重跑（2026-11-03）

> 训练机目录 `/home/hock/lcr/m0_nav_demo`。解释器 `/home/hock/lcr/m0env/bin/python`。纯 CPU（`CUDA_VISIBLE_DEVICES=""`）。
> 脚本没有 `--seed` 命令行参数；种子写死在脚本里。
> 成功率 = 到达目标（`info["success"]`）的回合比例。表中的百分数是脚本按整数打印的结果（`:.0f`）；以分数为准。
> 下面两张表是这次核对过的重跑。文末 git 提交里的数字是历史锚点，**不是** seed=0。

## 种子协议

| 脚本 | N | 种子 | 权重 |
|---|---|---|---|
| `eval_all.py` | 200 | `env.reset(seed=10000+s)`，`s = 0..199` | 下表七个文件，逐个加载 |
| `gen_test.py` | 500 | `range(5000, 5000+N)` | BC = `bc_policy_vanillaBC_backup.npz.npy`；DAgger = `bc_policy_dagger_r2.npy` |

`gen_test.py` 把第二列的表头写成 `DAgger-best`。那只是脚本里写死的列名。这一列的权重是 `bc_policy_dagger_r2.npy`，不是「最好的一轮」。

## eval_all.py N=200

同一 env、同一 obs 归一化、同一 MLP（8→64→64→2）。只换权重文件。

| weight | success | rate |
|---|---|---|
| `bc_policy_vanillaBC_backup.npz.npy` | 153/200 | 76% |
| `bc_policy.npz.npy` | 146/200 | 73% |
| `bc_policy_dagger_r1.npy` | 148/200 | 74% |
| `bc_policy_dagger_r2.npy` | 146/200 | 73% |
| `bc_policy_dagger_r3.npy` | 144/200 | 72% |
| `bc_policy_dagger_r4.npy` | 151/200 | 76% |
| `bc_policy_dagger_r5.npy` | 148/200 | 74% |

r4（151/200 = 76%）是五轮 DAgger 里计数最高的一轮。它和 vanilla BC backup 的打印百分数同为 76%，计数是 backup 153、r4 151。`bc_policy.npz.npy` 与 r2 同为 146/200 = 73%。

## gen_test.py N=500

BC = `bc_policy_vanillaBC_backup.npz.npy`。DAgger = `bc_policy_dagger_r2.npy`。

| band | BC | DAgger(r2) |
|---|---|---|
| 0.6–1.5m（训练带） | 355/500 (71%) | 356/500 (71%) |
| 0.3–0.9m | 443/500 (89%) | 451/500 (90%) |
| 0.4–2.2m | 313/500 (63%) | 322/500 (64%) |
| 1.5–2.0m（难尾） | 252/500 (50%) | 242/500 (48%) |

## 命令

训练机上执行，不加 `--seed`：

```bash
CUDA_VISIBLE_DEVICES="" /home/hock/lcr/m0env/bin/python eval_all.py
CUDA_VISIBLE_DEVICES="" /home/hock/lcr/m0env/bin/python -u gen_test.py
```

仓库里的同名脚本在 `sim/m0_nav_demo/`。脚本用绝对路径加载 `/home/hock/lcr/m0_nav_demo` 下的权重。

## 结论

训练带打平：gen_test 0.6–1.5m 上 BC 355/500 = 71%，DAgger(r2) 356/500 = 71%。难尾 BC 略高：1.5–2.0m 上 BC 252/500 = 50%，DAgger(r2) 242/500 = 48%。另外两档的打印百分数只差 1 个点（89/90，63/64）。DAgger does not dominate。

eval_all 上 DAgger 五轮是 74% / 73% / 72% / 76% / 74%。最高的 DAgger 轮与 vanilla BC backup 打印同为 76%，计数更低。

PPO 只有历史提交里的 3/100，见下节，不进入这张 seed=0 表。

## 已被这次重跑取代的旧数字

不要再引用下面这些：

- 口头数字：BC 87 / DAgger 69，难尾 40/47。
- 上一版 `RESULTS.md` 与报告 v3 的 gen_test 表：训练带 78/75，难尾 38/50。
- 上一版 `RESULTS.md` 的 eval_all 计数（backup 143/200 = 72% 那一表，其中 r5 = 66%）。

它们都不是这次 seed=0 重跑。

## 历史锚点（git 提交，不是 seed=0）

这些数字来自当时的提交，env 版本与上表不同，不能和 seed=0 表比绝对值。

| 数字 | 提交 | 说明 |
|---|---|---|
| PPO warm-start **3/100** @ 300k CPU steps（提交说明中的墙钟 72.8s） | `a0ce2ff` | 仓库里仅有的 PPO 结果。不是 seed=0，也不是「RL > BC」。 |
| 旧 env BC **64/100**（median 0.120m） | `d2362a8` | env 时间预算/奖励修正前的 `eval_bc`。 |
| sweep teacher **78%** / BC NN **79%** | `ad692bd` | 另一版 env。不可与 seed=0 表对照。 |

## 相关脚本（`sim/m0_nav_demo/`）

- `eval_all.py` — 上表七个权重，N=200，`reset(seed=10000+s)`
- `gen_test.py` — N=500，种子 `range(5000, 5000+N)`；列名 `DAgger-best` 只是写死的表头
- `dagger.py` — 写出 `bc_policy_dagger_r1.npy` … `r5.npy`
- `train_bc.py` / `collect_bc.py` / `eval_bc.py`
- `train_ppo_warm.py` — 历史 3/100 的脚本（`a0ce2ff`）
- `sweep_controller.py` — 历史 teacher 78% 的脚本（`ad692bd`）
- `env_diffdrive.py`
