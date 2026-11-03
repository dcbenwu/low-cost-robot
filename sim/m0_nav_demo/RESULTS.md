# M0 导航线 — 实测结果（可复现）

> 生成: 2026-11-03 · 在 udoge (192.168.50.33) `/home/hock/lcr/m0_nav_demo/`
> 环境: Python 3.12 · numpy 2.5.4 · mujoco 3.17.0 · torch 2.15.0+cpu（venv `/home/hock/lcr/m0env`）
> 约束: 纯 CPU（`CUDA_VISIBLE_DEVICES=""`），不占 GPU
> 本文件只记录**能复现的数字**，每条都给出可打开的脚本 + 复现命令。
> 复现命令见末尾。所有成功率 = 「到达目标（body 系到目标距离 < 阈值）」占 rollout 集的比例。

---

## R1. 训练域成功率（eval_all.py, N=200/模型）

同一 env、同一 obs 归一化口径、同一 MLP(8→64→64→2)，只换训练法：

| 模型（权重文件） | 成功 | 成功率 |
|---|---|---|
| `bc_policy_vanillaBC_backup.npz.npy` | 143/200 | **72%** |
| `bc_policy.npz.npy`（当前 BC） | 147/200 | **74%** |
| `bc_policy_dagger_r1.npy` | 150/200 | **75%** |
| `bc_policy_dagger_r2.npy` | 143/200 | **72%** |
| `bc_policy_dagger_r3.npy` | 145/200 | **72%** |
| `bc_policy_dagger_r4.npy` | 153/200 | **76%** |
| `bc_policy_dagger_r5.npy` | 132/200 | **66%** |

**读数**: BC（72–74%）与 DAgger（66–76%）落在同一区间；DAgger 没有"碾压" BC，
最佳轮 r4=76% 只比当前 BC 74% 高 2pt，r5=66% 反而更低。
**结论要诚实**：在本任务 + 本预算下，DAgger ≈ BC，增益有限且不单调。

## R2. 泛化曲线（gen_test.py, N=500, vanilla BC vs DAgger-best/r2）

跨目标距离分布，测越界抗性：

| 距离档 | vanilla BC | DAgger-best |
|---|---|---|
| 训练域 0.6–1.5m | 388/500 · **78%** | 376/500 · 75% |
| 略越界 0.3–0.9m | 422/500 · 84% | 422/500 · 84% |
| 宽越界 0.4–2.2m | 343/500 · 69% | 328/500 · 66% |
| **难尾 1.5–2.0m** | 189/500 · 38% | 250/500 · **50%** |

**读数**: 训练域 / 宽越界 BC 略高；**唯一 DAgger 明显赢的是最难的越界尾段**（38%→50%）——
与 DAgger 论文（1011.0686）理论一致：吃 student 自己踩到的 OOD 状态，对最难的尾更抗。
这不是"DAgger 整体碾压"，是"DAgger 在尾段更稳"。

## R3. 历史真实锚点（git 提交号，可 `git show <hash>` 查）

| 数字 | 来源提交 | 说明 |
|---|---|---|
| sweep analytic 控制器 **78%** | `ad692bd` | 解析控制器上界（M0 参照） |
| BC NN 策略 **79%** | `ad692bd` | 同上 env 的 MLP（与 R1 74% 属不同 env/奖励版本，见 a0ce2ff→d2362a8 的演进） |
| eval_bc **64/100**，median 0.120m | `d2362a8` | 早期基线版环境下的 BC（时间预算/奖励修正前） |
| PPO warm-start **3/100** @ 300k CPU step, 72.8s | `a0ce2ff` | RL(PPO) 从 BC 初始化，本预算下未超 BC |

> ⚠️ 版本说明：`ad692bd`(79%)、`d2362a8`(64/100) 是 **env 时间预算/奖励修正前** 的旧口径；
> R1/R2 是**修正后 env**（`env_diffdrive.py` 现版本）的重跑。两者不可直接比同一绝对值，
> 各自只在同一个 env 内可比。报告引用时都会标注所属 env 版本。

## R4. 关于"RL > BC"的诚实结论

- **本任务未复现 RL > BC**：PPO warm-start（`train_ppo_warm.py`）300k CPU step → 3/100（`a0ce2ff`），
  低于 BC 基线。原因预算内探索不足（稀疏奖励 + 300k step 对 8-d obs 差速任务太少）。
- **交互式模仿（DAgger）≈ BC**，且在最难尾段更稳（R2: 38%→50%）。
- 因此"RL > BC"在本项目里是**待验证假设 / 未来工作**，不是"已复现结论"。
  要成立需要：更长训练 + 更好奖励整形 / 更大网络 + 抓取等更精细任务（见路线图 M1）。

## 复现命令（可直接执行）

```bash
# 登录 udoge
ssh hock@192.168.50.33
cd /home/hock/lcr/m0_nav_demo
PY=/home/hock/lcr/m0env/bin/python

# R1 训练域成功率（5 个 ckpt × N=200）
CUDA_VISIBLE_DEVICES="" $PY eval_all.py

# R2 泛化曲线（vanilla BC vs DAgger-best, N=500）
CUDA_VISIBLE_DEVICES="" $PY gen_test.py
```

## 相关脚本（均已 `git add` 进 `sim/m0_nav_demo/`）

- `collect_bc.py` — teacher 采样造 BC 数据（`bc_data.npz` 270k 对）
- `train_bc.py` — BC（MLP 8→64→64→2, tanh, MSE）
- `eval_bc.py` — 早期基线 eval（历史 64/100 的来源）
- `dagger.py` — DAgger 自举（5 轮 → `bc_policy_dagger_r1..r5.npy`）
- `gen_test.py` — 泛化 stress（N=500）
- `eval_all.py` — 5 ckpt 训练域成功率（本次新增）
- `ctl_test.py` — 控制器/teacher 行为测试
- `train_ppo_warm.py` — PPO warm-start（3/100 的来源）
- `sweep_controller.py` — analytic 控制器 sweep（78% 的来源）
- `env_diffdrive.py` — MuJoCo 差速底盘环境（8-d obs / 2-d act）
