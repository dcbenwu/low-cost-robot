# 低成本移动机器人 · 研究落地框架与 M0 实测报告

交付：**项目报告（v3 结构；M0 成功率已换成 seed=0 重跑）** · 2026-11-03
范围：**M0 差速导航线（BC / DAgger / PPO warm-start，仿真）**；**M2 硬件部署（修正版 BOM）**；后续里程碑为规划。

> **诚信声明（写在前面）**
> 本报告 **只引用能打开、能验的东西**：
> - 论文：3 篇，均已逐篇抓取 `arxiv.org/abs` 页面核对标题+作者；
> - seed=0 成功率：训练机 `/home/hock/lcr/m0_nav_demo` 的重跑，记在 `sim/m0_nav_demo/RESULTS.md`（无 `--seed` 参数，种子写死在脚本里）；
> - 历史数字：只挂在对应 git 提交上（`a0ce2ff` / `d2362a8` / `ad692bd`），**不**写成 seed=0；
> - **不再有**无法查证的名词、编造的成功率、或把旧表当成这次重跑。
> 凡未真正跑过的（3D 抓取、更长时间 RL），本报告一律标注"规划 / 待验证假设"，**不写成已完成结论**。

---

## 0. 执行摘要（只讲真话）

**这条线我们真实做出了什么：**
1. MuJoCo 上搭了一个 8 维观测 / 2 维动作的**差速底盘**环境（`env_diffdrive.py`）。
2. 用解析 teacher 采样（`collect_bc.py` → `bc_data.npz`），训了 BC（`train_bc.py`）。seed=0 `eval_all.py`（N=200，`reset(seed=10000+s)`）：`bc_policy_vanillaBC_backup.npz.npy` **153/200 = 76%**，`bc_policy.npz.npy` **146/200 = 73%**。
3. 实现了 DAgger 自举 5 轮（`dagger.py` → `bc_policy_dagger_r1.npy` … `r5.npy`）。同一套 seed=0 eval：**74% / 73% / 72% / 76% / 74%**（148、146、144、151、148，分母都是 200）。r4 是 DAgger 轮次里最高的，打印百分数与 vanilla backup 同为 76%，计数 151 < 153。
4. PPO warm-start 只有历史提交：**3/100** @ 300k CPU steps（`a0ce2ff`）。这不是 seed=0，也不写成「RL > BC」。
5. 泛化 stress（`gen_test.py`, N=500，种子 `range(5000, 5000+N)`；BC = vanilla backup，DAgger = r2）：训练带 **71% / 71%**（355/500 与 356/500）；难尾 **BC 252/500 = 50%**，**DAgger r2 242/500 = 48%**。训练带打平，难尾 BC 略高。DAgger does not dominate。脚本把列名写成 `DAgger-best`，那只是写死的表头。
6. 给出**电压配套一致**的 M2 硬件 BOM（`docs/bom.md`），基础版 ≈ ¥700–1,400。

**我们**没有**做出什么（诚实标注，不当成果）：**
- ❌ **没有 seed=0 的 "RL > BC"**：PPO 只有历史提交 3/100 @ 300k CPU steps（`a0ce2ff`），不能和 seed=0 的 BC/DAgger 表比。RL>BC 仍是**待验证假设 / 未来工作**。
- ❌ **没有 3D 抓取 / 机械臂**代码或结果（之前版本"抓取 RL 1990 > BC 1620"**已删，无来源**）。
- ❌ 没做**真机**（硬件 BOM 已修，未下单，未装配）。

**一句话**：一条**能从 teacher 到 BC/DAgger 策略、能在 CPU 跑通并能被真机 MLP 部署**的仿真实验线，数字全部可复现；RL>BC 与仿真→真机迁移都是**下一步要验证的假设**，不是已交付成果。

---

## 1. 立项与调研

### 1.1 一句话定位
用**可复现代码 + 真实可验证数字**，把"便宜移动底盘 + BC/DAgger 策略 + 真机部署"跑通；用实验**检验**（而非预设）模仿学习 vs 强化学习、仿真→真机的边界。

### 1.2 真实、可核的参考（每篇都抓了 abs 页验证）

| 论文（arXiv id） | 标题 · 作者 | 与本项目关系 |
|---|---|---|
| `1011.0686` | *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* · Ross, Gordon, Bagnell | DAgger 算法出处。本次 seed=0 重跑没有出现难尾上 DAgger 更高：难尾 BC 252/500 = 50%，DAgger r2 242/500 = 48%。 |
| `1707.06347` | *Proximal Policy Optimization Algorithms* · Schulman, Wolski, Dhariwal, Radford, Klimov | PPO warm-start（`train_ppo_warm.py`）用的算法。 |
| `1710.06537` | *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization* · Peng, Andrychowicz, Zaremba, Abbeel | 仿真→真机（M2/M3）方法论背景；本项目走"小 MLP 部署 + 轻量域随机化"而非重型域随机化。 |

> 环境用 MuJoCo（Todorov, Erel, Tassa 2012，物理引擎）。3 篇之外**不再**引用无法核对的编号或名词。

### 1.3 差异化（可验证的）
- **资产全进 git**：每个数字 → 一个 `sim/m0_nav_demo/*.py` + `RESULTS.md` + 一个提交号，`git show <hash>` 可查；
- **预算透明**：仿真全在 CPU 跑通，真机 BOM 基础版 ¥700–1,400（电压配套一致）；
- **诚实边界**：把"RL>BC / 仿真→真机"作为**待验证假设**，实验成能直接答"是否成立 + 差多少"。

---

## 2. 技术路线（M0 → M4）

| 阶段 | 目标 | 状态 |
|---|---|---|
| **M0** | 差速底盘 + BC→（DAgger）→真机可部署 MLP，**仿真内跑通并量化** | ✅ **已落地**（本报告主体） |
| **M2** | 真机硬件（修正版 BOM）+ 遥操/半自主 + 仿真策略上真机 | 📋 方案已定，**未下单未装配** |
| **M1** | Isaac Lab + 抓取（夹爪/吸盘）+ 换更会转的老师 → **真正测"RL≥BC"** | 📋 规划（需 Isaac Lab venv 重建） |
| **M3** | 真机自主定位（AprilTag/IR beacon + 里程计）→ 全自主导航 | 📋 规划 |
| **M4** | 抓取 + 真机 + 域随机化，产出可复现证据 | 📋 规划 |

> 排序逻辑：**先做实"策略从仿真搬到真实 MLP"这条最短链路**（M0/M2），再上更高成本 Isaac Lab 抓取线（M1/M4）。

---

## 3. M0 完整落地过程（每步都有脚本 + 数字）

固定条件：**纯 CPU**（`CUDA_VISIBLE_DEVICES=""`）；同一 env；同一 obs 归一化口径；同一 MLP。

### 3.1 环境 & 老师
- 环境：`env_diffdrive.py`（MuJoCo 差速底盘，8-d obs / 2-d act）。
- 老师：解析余弦弧线控制器，参数族 **C**（K=0.8, A=1.0, FWD=0.8, CLOSE=0.6, EASE=0.5），由 `sweep_controller.py` sweep 选出 → **解析控制器上界 78%**（提交 `ad692bd`，旧 env，不是 seed=0）。
- 已知**设计性盲区**：目标在车体侧后方（`|ang| > 90°`）时该老师 `fwd=max(0,cos(ang))→0` → 停死。
  **这是老师公式的覆盖盲区，不是物理上限**——换一个会原地转向/能处理侧后方的老师即可突破（M1）。

### 3.2 模仿学习（BC）
| 步骤 | 脚本 | 产出 |
|---|---|---|
| teacher 采样 | `collect_bc.py` | `bc_data.npz` |
| 训 BC | `train_bc.py`（MLP 8→64→64→2, tanh, MSE） | `bc_policy.npz.npy` |
| **结果（seed=0）** | `eval_all.py`（N=200，`reset(seed=10000+s)`） | vanilla backup **153/200 = 76%**；`bc_policy.npz.npy` **146/200 = 73%** |
| 历史基线 | `eval_bc.py`（旧 env，**不是** seed=0） | **64/100，median 0.120m**（提交 `d2362a8`） |
| 老师上界 | `sweep_controller.py`（旧 env，**不是** seed=0） | **78%**（`ad692bd`）；同期 BC NN 79% |

> 版本注：`ad692bd`（teacher 78% / BC NN 79%）和 `d2362a8`（64/100）是 **env 时间预算/奖励修正前** 的旧口径，不能和 seed=0 表比绝对值。seed=0 的 BC 数字只以上面 `eval_all.py` 这一行为准。

### 3.3 DAgger 自举（交互式模仿）
- 脚本 `dagger.py`，5 轮 → `bc_policy_dagger_r1.npy` … `r5.npy`。
- seed=0 `eval_all.py`（N=200，`reset(seed=10000+s)`）：

| weight | success | rate |
|---|---|---|
| `bc_policy_dagger_r1.npy` | 148/200 | 74% |
| `bc_policy_dagger_r2.npy` | 146/200 | 73% |
| `bc_policy_dagger_r3.npy` | 144/200 | 72% |
| `bc_policy_dagger_r4.npy` | 151/200 | 76% |
| `bc_policy_dagger_r5.npy` | 148/200 | 74% |

- r4 是 DAgger 轮次里计数最高的一轮（151/200 = 76%），与 vanilla BC backup（153/200 = 76%）打印百分数相同，计数更低。r5 是 148/200 = 74%，不是更低的另一套数字。DAgger does not dominate。

### 3.4 泛化 stress（`gen_test.py`, N=500）

种子 `range(5000, 5000+N)`。BC = `bc_policy_vanillaBC_backup.npz.npy`。DAgger = `bc_policy_dagger_r2.npy`。脚本列名 `DAgger-best` 只是写死的表头。

| band | BC | DAgger(r2) |
|---|---|---|
| 0.6–1.5m（训练带） | 355/500 (71%) | 356/500 (71%) |
| 0.3–0.9m | 443/500 (89%) | 451/500 (90%) |
| 0.4–2.2m | 313/500 (63%) | 322/500 (64%) |
| 1.5–2.0m（难尾） | 252/500 (50%) | 242/500 (48%) |

**结论**：训练带打平（71% / 71%）。难尾 BC 略高（50% / 48%）。另外两档打印百分数只差 1 个点。DAgger does not dominate。

### 3.5 强化学习（PPO warm-start）——只有历史 3/100
- 脚本 `train_ppo_warm.py`（value warm-start + BC 动作锚）。
- 历史提交 `a0ce2ff`：300k CPU step，提交说明中的墙钟 72.8s，eval **3/100**。这不是 seed=0 重跑，不能和上面的 BC/DAgger 表比绝对值。
- **诚实结论**：仓库里没有 seed=0 的 PPO 结果。仅有的 PPO 数字是这条历史 3/100，不构成「RL > BC」。

### 3.6 "RL > BC" 假设的处置
| 论断 | 状态 |
|---|---|
| "RL 显著 > BC" | ❌ **没有 seed=0 结果**。仅有历史 PPO 3/100（`a0ce2ff`）——列为**待验证假设 / 未来工作** |
| "DAgger ≥ BC，尾段更稳" | ❌ **这次 seed=0 不支持**。训练带 71% / 71%；难尾 BC 50%，DAgger r2 48% |
| 真正验证 RL≥BC | 需要 **M1 Isaac Lab 抓取线**：更长训练 + 更好奖励整形 + 抓取精细任务 |

---

## 4. M2 硬件（修正版 BOM，电压配套一致）

详见 `docs/bom.md`。要点（对照早期版本硬伤逐条修正）：

| 项 | 修正前（有硬伤） | 修正后 |
|---|---|---|
| 电机 | 24V 660 | **12V 660 编码器** |
| 电池 | 4S 14.8V | **3S 11.1V 2600mAh**（满电 12.6V） |
| 驱动板 | TB6612（未标上限） | **TB6612FNG**（4.5–15.8V / 1.2A/通道）；上 24V 电机则换 **BTS7960** |
| 测距 | VL53L0X"0.1–4m"+"同址" | **有效 ~2m**；3 片 **XSHUT 改 I2C 地址**（默认全 0x29 会冲突） |
| 术语 | 超声 | **ToF 激光**（VL53L0X 是飞行时间激光，非 HC-SR04 超声） |
| 自主 | 超声阵列自主定位（缺定位方案，不可行） | **遥操/半自主 + 目标信标**（AprilTag/IR beacon）；全自主定位推 M3 |
| 预算 | ¥1,000–1,700 | **≈ ¥700–1,400** |

**装配 + 策略上真机（4 步）**：
1. 底盘装 2×12V 660 + 万向轮，3S 电池仓居中；PDB 出 5V（主控）+ 12V（驱动）。
2. 3×VL53L0X 各用 XSHUT 引脚切 I2C 地址；MPU6050、编码 A/B（硬件捕获）接 ESP32-S3。
3. 遥操链路：手机 App/PC → UDP/TCP → ESP32-S3 → obs（IMU+yaw+编码器里程计+3路 ToF）→ 部署 BC/DAgger 8→64→64→2 MLP。
4. 部署：MLP → TFLite Micro(int8) 在 ESP32-S3；`esp-ledc` 20kHz 出两轮 PWM，编码器闭环。

> **MLP 部署可行性**：8→64→64→2 共 ~1.6 万参数，int8 后 < 20KB，ESP32-S3（8MB PSRAM / 16MB flash）**足够**——这条**可行、非猜测**。

**M0→M2 关键迁移假设**（诚实标注为**待验证**）：
1. obs 口径一致（IMU + yaw + 编码器里程计 + 3 距离）→ 训练/推理同构（需在固件里把仿真 obs 与传感器逐项对齐）。
2. 动作映射一致（两轮速度 → PWM）。
3. 域差异（打滑/摩擦/噪声）→ 必要时加轻量域随机化（1710.06537，本项目走轻量版）。

**未验证项**：无真实电机数据、无真机实测成功率——**不下"迁移会成功"的结论**。M2 的目标就是**实测这 1–2 项假设并拿到真机数字**。

---

## 5. 里程碑（诚实状态）

| 里程碑 | 状态 | 证据 |
|---|---|---|
| M0 环境 + teacher 采样 | ✅ | `env_diffdrive.py` / `collect_bc.py` / `bc_data.npz` |
| M0 BC（seed=0 eval_all） | ✅ | vanilla backup 153/200 = 76%；`bc_policy.npz.npy` 146/200 = 73% |
| M0 DAgger（seed=0 eval_all） | ✅ 五轮已跑，未高于 BC backup | r1–r5 = 74/73/72/76/74%（151/200 为 DAgger 轮次最高，计数低于 backup 的 153/200） |
| M0 PPO warm-start | 历史提交，**不是** seed=0 | `a0ce2ff`：3/100 @ 300k CPU steps |
| M0 泛化 stress（seed=0 gen_test） | ✅ | 训练带 71/71；难尾 BC 50%、DAgger r2 48% |
| 3D 抓取 / 机械臂 | ❌ **未做**（无代码无结果） | 规划于 M1/M4 |
| 真机 M2 硬件 + 部署 | 📋 方案已定（`docs/bom.md`），**未下单** | 待采购装配 |
| M3 自主定位 | 📋 规划 | 待 M2 后 |

---

## 6. 可复现资产（全部真实、全部能打开）

**仓库** `dcbenwu/low-cost-robot`（`sim/m0_nav_demo/`）：
- 代码：`env_diffdrive.py` · `collect_bc.py` · `train_bc.py` · `eval_bc.py` · `dagger.py` · `gen_test.py` · `eval_all.py` · `ctl_test.py` · `train_ppo_warm.py` · `sweep_controller.py` · `probe_*.py`
- 结果：`RESULTS.md`（seed=0 重跑；口头 87/69、难尾 40/47，以及 v3 的 78/75 与 38/50 已作废）
- BOM：`docs/bom.md`（修正版）

**关键 git 提交**（`git show <hash>` 可查）：
- `d1a1ebe` dagger.py/gen_test.py/eval_all.py/ctl_test.py 入仓 + RESULTS.md + BOM 修正
- `ad692bd` sweep 控制器 78% / BC NN 79%（旧 env 口径）
- `d2362a8` eval_bc 64/100, median 0.120m + PPO warm-start 脚本
- `a0ce2ff` 300k CPU run, eval 3/100（PPO，历史提交，不是 seed=0）

**服务器（udoge 192.168.50.33）** 工作目录 `/home/hock/lcr/m0_nav_demo/`、环境 `/home/hock/lcr/m0env`、模型 `bc_policy_*.npy` / `bc_policy_dagger_r1..r5.npy` / `ppo_warm_policy.npz`。

**复现命令**（与 `RESULTS.md` 相同；无 `--seed`）：
```bash
CUDA_VISIBLE_DEVICES="" /home/hock/lcr/m0env/bin/python eval_all.py
CUDA_VISIBLE_DEVICES="" /home/hock/lcr/m0env/bin/python -u gen_test.py
```
脚本在训练机目录 `/home/hock/lcr/m0_nav_demo` 用绝对路径读权重。`eval_all.py` 使用 `reset(seed=10000+s)`；`gen_test.py` 使用 `range(5000, 5000+N)`。

---

## 7. 诚实声明（哪些真 / 哪些未来）

**真（本报告只讲这些）**：
- M0 仿真线的 seed=0 数字来自训练机重跑，记在 `RESULTS.md`：eval_all 上 vanilla backup 153/200 = 76%，DAgger 五轮 74/73/72/76/74%；gen_test 训练带 71/71，难尾 BC 50%、DAgger r2 48%。训练带打平，难尾 BC 略高。DAgger does not dominate；
- PPO 只有历史 3/100（`a0ce2ff`），不是 seed=0；
- M2 硬件 BOM 电压配套一致、预算 ¥700–1,400，MLP 部署 ESP32-S3 可行；
- 3 篇引用论文逐篇核对过 abs 页标题/作者。

**未来 / 待验证（诚实标注，非成果）**：
- **RL > BC 没有 seed=0 结果**（仅历史 PPO 3/100，`a0ce2ff`）——待验证假设，需 M1（Isaac Lab 抓取线 + 更久训练 + 更好奖励）；
- **3D 抓取**无代码无结果，规划于 M1/M4；
- **真机**未下单未装配，M2 目标是拿到真机实测数字；
- "老师到顶"是**老师公式的盲区**（可换老师突破），**不是物理硬上限**；
- 仿真→真机迁移**未实测**，不下"会成功"的结论。

> 本报告不含任何无法打开验证的引用、编造的成功率、或与 git log 矛盾的绝对值。凡未跑过的，一律标注为规划/待验证假设。
