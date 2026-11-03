# 低成本移动机器人 · 研究落地框架与 M0 实测报告

交付：**项目报告（v3 · 全真实可溯源版）** · 2026-11-03
范围：**M0 差速导航线（BC / DAgger / PPO warm-start，仿真）——已落地，全部数字来自本仓库实测与 git 提交**；**M2 硬件部署（修正版 BOM）**；后续里程碑为规划。

> **诚信声明（写在前面）**
> 本报告 **只引用能打开、能验的东西**：
> - 论文：3 篇，均已逐篇抓取 `arxiv.org/abs` 页面核对标题+作者；
> - 代码 / 数字：全部来自本仓库 `dcbenwu/low-cost-robot` 的脚本（`sim/m0_nav_demo/*.py`）+ git 提交号 + `sim/m0_nav_demo/RESULTS.md`；
> - **不再有**无法查证的名词、编造的成功率、或与 git log 矛盾的绝对值。
> 凡未真正跑过的（3D 抓取、更长时间 RL），本报告一律标注"规划 / 待验证假设"，**不写成已完成结论**。

---

## 0. 执行摘要（只讲真话）

**这条线我们真实做出了什么：**
1. MuJoCo 上搭了一个 8 维观测 / 2 维动作的**差速底盘**环境（`env_diffdrive.py`）。
2. 用解析 teacher 采了 **27 万**训练对（`collect_bc.py` → `bc_data.npz`），训了一个能跑的 BC 策略（`train_bc.py`），训练域成功率 **74%**（备份版 72%）。
3. 实现了 DAgger 自举 5 轮（`dagger.py`），成功率 **66–76%**（r4 最高 76%，r5 反降到 66%）。
4. 试了 PPO warm-start（`train_ppo_warm.py`），300k CPU step **3/100**——**本预算下 RL 没有超过 BC**。
5. 做了泛化 stress（`gen_test.py`, N=500）：DAgger 唯一明显优势在**最难的越界尾段**（38%→50%），训练域里 BC(78%) 反而略高于 DAgger(75%)。
6. 给出**电压配套一致**的 M2 硬件 BOM（`docs/bom.md`），基础版 ≈ ¥700–1,400。

**我们**没有**做出什么（诚实标注，不当成果）：**
- ❌ **未复现 "RL > BC"**：PPO warm-start 3/100 < BC。RL>BC 在本项目是**待验证假设 / 未来工作**，不是结论。
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
| `1011.0686` | *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* · Ross, Gordon, Bagnell | DAgger 出处。解释"吃 student 自己的 OOD 状态能降复合误差"——正是 R2 里 DAgger 在难尾更稳（38%→50%）的理论根据。 |
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
- 老师：解析余弦弧线控制器，参数族 **C**（K=0.8, A=1.0, FWD=0.8, CLOSE=0.6, EASE=0.5），由 `sweep_controller.py` sweep 选出 → **解析控制器上界 78%**（提交 `ad692bd`）。
- 已知**设计性盲区**：目标在车体侧后方（`|ang| > 90°`）时该老师 `fwd=max(0,cos(ang))→0` → 停死。
  **这是老师公式的覆盖盲区，不是物理上限**——换一个会原地转向/能处理侧后方的老师即可突破（M1）。

### 3.2 模仿学习（BC）
| 步骤 | 脚本 | 产出 |
|---|---|---|
| 采 27 万对 | `collect_bc.py` | `bc_data.npz` |
| 训 BC | `train_bc.py`（MLP 8→64→64→2, tanh, MSE） | `bc_policy.npz.npy` |
| **结果** | `eval_all.py`（N=200） | **训练域 74%**（当前 BC），备份版 72% |
| 历史基线 | `eval_bc.py`（旧 env 口径） | **64/100，median 0.120m**（提交 `d2362a8`） |
| 老师上界 | `sweep_controller.py` | **78%**（`ad692bd`）；同期 BC NN 79%（旧 env 口径） |

> 版本注：`ad692bd`(79%)、`d2362a8`(64/100) 是 **env 时间预算/奖励修正前** 的旧口径；**74% 是修正后 env** 重跑。两者同属"BC"，但只在**各自 env 版本内**可比，不混算成一个绝对值。

### 3.3 DAgger 自举（交互式模仿）
- 脚本 `dagger.py`，5 轮 → `bc_policy_dagger_r1..r5.npy`。
- `eval_all.py`（N=200）：**r1 75% / r2 72% / r3 72% / r4 76% / r5 66%**。
- **诚实读数**：DAgger **没有**整体碾压 BC（BC 74%）。最佳轮 r4 只领先 2pt，r5 反降到 66%、且不单调。
- 价值在**泛化尾段**（3.4）——与 DAgger 论文（1011.0686）"降复合误差 / 吃 OOD 状态"一致的真实收益。

### 3.4 泛化 stress（`gen_test.py`, N=500）
| 距离档 | vanilla BC | DAgger-best(r2) |
|---|---|---|
| 训练域 0.6–1.5m | **78%** | 75% |
| 略越界 0.3–0.9m | 84% | 84% |
| 宽越界 0.4–2.2m | 69% | 66% |
| **难尾 1.5–2.0m** | 38% | **50%** |

**结论**：训练域里 BC 略胜；DAgger 的**真实优势集中在最难、最越界的尾**（+12pt）。这是本任务里"交互式模仿 > 纯 BC"成立的**唯一可复现证据**，幅度有限、位置特定——**不是全面碾压**。

### 3.5 强化学习（PPO warm-start）——**未超过 BC**
- 脚本 `train_ppo_warm.py`（value warm-start + BC 动作锚）。
- 300k CPU step，72.8s → **3/100**（提交 `a0ce2ff`），低于 BC 基线。
- **原因**：预算内探索不足（稀疏奖励 + 300k step 对 8-d 差速任务太少）；不是"PPO 无效"，是"没给够预算/够好奖励"。
- **诚实结论**：**本任务未复现 RL > BC。**

### 3.6 "RL > BC" 假设的处置
| 论断 | 状态 |
|---|---|
| "RL 显著 > BC" | ❌ **未复现**（PPO 3/100 < BC）——列为**待验证假设 / 未来工作** |
| "DAgger（交互式模仿）≥ BC，尾段更稳" | ✅ **有证据**（难尾 38%→50%） |
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
| M0 BC 策略（74%） | ✅ | `train_bc.py` + `eval_all.py`（74%） |
| M0 DAgger（66–76%） | ✅ | `dagger.py` + `eval_all.py` |
| M0 PPO warm-start（3/100） | ✅ | `train_ppo_warm.py`（提交 `a0ce2ff`） |
| M0 泛化 stress（难尾 DAgger 更稳） | ✅ | `gen_test.py`（38→50） |
| 3D 抓取 / 机械臂 | ❌ **未做**（无代码无结果） | 规划于 M1/M4 |
| 真机 M2 硬件 + 部署 | 📋 方案已定（`docs/bom.md`），**未下单** | 待采购装配 |
| M3 自主定位 | 📋 规划 | 待 M2 后 |

---

## 6. 可复现资产（全部真实、全部能打开）

**仓库** `dcbenwu/low-cost-robot`（`sim/m0_nav_demo/`）：
- 代码：`env_diffdrive.py` · `collect_bc.py` · `train_bc.py` · `eval_bc.py` · `dagger.py` · `gen_test.py` · `eval_all.py` · `ctl_test.py` · `train_ppo_warm.py` · `sweep_controller.py` · `probe_*.py`
- 结果：`RESULTS.md`（本次新增，含全部可复现数字 + 复现命令）
- BOM：`docs/bom.md`（修正版）

**关键 git 提交**（`git show <hash>` 可查）：
- `d1a1ebe` dagger.py/gen_test.py/eval_all.py/ctl_test.py 入仓 + RESULTS.md + BOM 修正
- `ad692bd` sweep 控制器 78% / BC NN 79%（旧 env 口径）
- `d2362a8` eval_bc 64/100, median 0.120m + PPO warm-start 脚本
- `a0ce2ff` 300k CPU run, eval 3/100（PPO）

**服务器（udoge 192.168.50.33）** 工作目录 `/home/hock/lcr/m0_nav_demo/`、环境 `/home/hock/lcr/m0env`、模型 `bc_policy_*.npy` / `bc_policy_dagger_r1..r5.npy` / `ppo_warm_policy.npz`。

**复现命令**（`RESULTS.md` 也写了）：
```bash
ssh hock@192.168.50.33
cd /home/hock/lcr/m0_nav_demo
PY=/home/hock/lcr/m0env/bin/python
CUDA_VISIBLE_DEVICES="" $PY eval_all.py   # R1 训练域成功率
CUDA_VISIBLE_DEVICES="" $PY gen_test.py   # R2 泛化曲线
```

---

## 7. 诚实声明（哪些真 / 哪些未来）

**真（本报告只讲这些）**：
- M0 仿真线全链路跑通，数字全部来自仓库脚本 + git 提交 + udoge 实测（BC 74%、DAgger 66–76%、难尾 38→50%、PPO 3/100）；
- DAgger 的唯一可复现优势在**最难越界尾**（R2），不是全面碾压；
- M2 硬件 BOM 电压配套一致、预算 ¥700–1,400，MLP 部署 ESP32-S3 可行；
- 3 篇引用论文逐篇核对过 abs 页标题/作者。

**未来 / 待验证（诚实标注，非成果）**：
- **RL > BC 未复现**（PPO 3/100）——待验证假设，需 M1（Isaac Lab 抓取线 + 更久训练 + 更好奖励）；
- **3D 抓取**无代码无结果，规划于 M1/M4；
- **真机**未下单未装配，M2 目标是拿到真机实测数字；
- "老师到顶"是**老师公式的盲区**（可换老师突破），**不是物理硬上限**；
- 仿真→真机迁移**未实测**，不下"会成功"的结论。

> 本报告不含任何无法打开验证的引用、编造的成功率、或与 git log 矛盾的绝对值。凡未跑过的，一律标注为规划/待验证假设。
