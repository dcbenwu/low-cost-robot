# 滚筒洗衣环境（基线）

把篓子里的脏衣服放进前置滚筒洗衣机，打开洗衣液瓶盖，把洗衣液倒进主洗盒，关抽屉和门，选一个合法洗涤程序，按启动，让滚筒转完规定圈数。

这是任务环境，不是已经训好的策略。没有加载 Franka lift 检查点（那次 Newton 评测是 0/100，不能当洗衣技能）。也没有布料仿真和流体仿真。

## 假设（画面和接口上都标了）

- 衣服是 4 个可抓的刚体盒子（shirt / pants / towel / sock），不是可变形布料。
- 洗衣液是瓶口姿态 + 一次 40 ml 的剂量记账，不是流体。瓶里初始 150 ml，主洗盒容量 60 ml。
- 门、抽屉、瓶盖、程序旋钮、启动键、门锁是有行程和速度上限的关节。CPU 后端用速度受限的位置伺服加上自由落体；MJCF 里的力矩上限留给 Newton。CPU 测试不是 MuJoCo / Newton 接触验证。
- 重力固定为 `(0, 0, -9.81)`。不加载 lift 任务在 ADR 难度 0 时写进去的零重力。
- `scripted_kinematic` 的标签是 `scripted environment validation / NOT learned robotic manipulation`。
- `physical` 的标签是 `physical actuator mode / NOT a learned policy`。

成功看状态，不看步骤计数器。衣服可以先装或先开抽屉，但启动时必须同时满足：四件衣服都在滚筒体积内、门关、抽屉关、主洗盒剂量在 30–60 ml、没有超量溢出、程序是 cotton / synthetic / delicate、瓶盖关、门锁上，并且滚筒再转满 2 圈。倒在地上、倒进关闭的抽屉、倒进柔顺剂盒、空瓶、门开着按启动、选 spin_only，都不算成功。运行中门锁住，门开不动。

## CPU 任务测试（云端和本机都能跑，不需要 GPU）

在仓库的 `sim/` 下：

```bash
cd sim
python -m unittest laundry_washer.tests.test_task_gates -v
```

这些测试走任务逻辑和 numpy 刚体积分。输出里的 `gpu_physics_validation` 是 false。它们不能代替 uDoge 上的 Newton 步长或画面。

脚本化全过程（CPU 软件光栅，不是 NewtonGL）：

```bash
cd sim
python -m laundry_washer.cli demo --mode scripted_kinematic --out /tmp/laundry_demo
python -m laundry_washer.cli reset --state /tmp/laundry_state.json
python -m laundry_washer.cli step --state /tmp/laundry_state.json --action open_door
python -m laundry_washer.cli diagnostics --state /tmp/laundry_state.json
python -m laundry_washer.cli config
```

`demo` 会在结束前把 `summary.json` 和 `frames/frame_*.png` 写好。帧来自当前刚体位姿的软件光栅，画面上有 “NOT LEARNED”“NOT DEFORMABLE CLOTH”“NOT FLUID”。

## uDoge 上的 GPU 路径（一环境，约 10.6 GB 空闲）

不要停 ComfyUI。不要改、不要升级 `/home/hock/IsaacLab`（develop `93d7bef`）。不要加载 Franka USD，不要开 Kit RTX 视口，不要用 64 环境的 lift 配置。

把本目录打成独立实验目录后（不需要克隆本仓库）：

```bash
export PYGLET_HEADLESS=1
export PYGLET_HEADLESS_DEVICE=0
export PYTHONPATH="/path/to/laundry_washer_experiment"
/home/hock/IsaacLab/env_isaaclab/bin/python -m laundry_washer.newton_run \
  --steps 180 --out /tmp/laundry_newton
```

要画面时再加 `--record`。DISPLAY 为空，所以必须先让 pyglet 走 headless，否则会在导入 viewer 时去连 X11。之前 NewtonGL v2 能录到动的帧；这次就绪检查没有再测 GL context。`--record` 如果建不出 context，进程以码 3 退出，`newton_diagnostics.json` 里写 `render_error`，不会补一张假图。

只看重力和步长、不录画面时不要加 `--record`。

### 支持的后端和相机（就这一套）

| 项 | 值 |
|---|---|
| 世界数 | 1 |
| 求解器 | `newton.solvers.SolverMuJoCo`（mujoco_warp 3.12，Newton 1.6，warp 1.17） |
| `use_mujoco_cpu` | false |
| CUDA graph | 关 |
| `njmax` / `nconmax` | 80 / 48 |
| `iterations` / `ls_iterations` | 20 / 8 |
| `dt` | 1/120 |
| 重力 | `(0, 0, -9.81)`，从模型读回，对不上就失败 |
| 画面 | `newton.viewer.ViewerGL(headless=True, width=640, height=360)` |
| 眼 / 看向 | `(1.35, -1.15, 0.92)` / `(0.28, 0.00, 0.45)`，Z 向上，pitch/yaw 为角度，与 Isaac Lab `93d7bef` 的 `_eye_lookat_to_pitch_yaw` 相同 |
| 纹理 | 无。只有程序生成的纯色几何，没有 HDR、没有穹顶长图 |
| 录像源别名 | `visualizer:newton`（`newton_gl` 的旧别名，这台机器上录出过真实帧） |

Isaac Lab 包装是可选的，不是默认。它会拉起 Isaac 应用，显存比直接 Newton 更大：

```bash
cd /home/hock/IsaacLab
source env_isaaclab/bin/activate
export PYGLET_HEADLESS=1
export PYTHONPATH="/path/to/laundry_washer_experiment"
python -m laundry_washer.isaac_run --headless --num-envs 1 --steps 120 --device cuda:0 --out /tmp/laundry_isaac
```

`--num-envs` 不是 1 会直接拒绝。配置是 `NewtonCfg(use_cuda_graph=False)` + `MJWarpSolverCfg` + `NewtonGLVisualizerCfg(streaming_view=False, enable_shadows=False, enable_sky=False)`。不读 lift 的 ADR。

## 观测、动作、奖励、终止

观测 43 维，顺序在 `spec.OBS_NAMES`：门/抽屉/盖/旋钮/按钮/锁/滚筒，瓶子位姿和余量，三个盒子和地面泼洒，四件衣服的位置和是否在筒内，夹爪，运行标志，圈数，程序是否合法，超量标志，以及 `gravity_z`。

动作 11 维，`spec.ACTION_NAMES`：四个机构目标、按钮、夹爪 xyz、抓取、倾倒触发、瓶子倾角。`env.step(action)` 走 4 个物理子步。`env.run("open_door")` 这类原语是给脚本和测试的，仍然受速度上限和关门挡板约束，不是把成功位写进去。

奖励：每 10 ms `-0.002`；一件衣服新进入滚筒 `+1`；合法倒入主洗盒 `+2`；洒出 `-1`；超量 `-2`；成功启动 `+5`；转完 `+10`；运行中硬开门 `-0.5`。

终止：`holds_success` 为真则 terminated；步数到达 `MAX_STEPS` 则 truncated。失败条件默认不截断回合，方便以后的策略自己改正。现在没有策略。

## 已知缺口

- 衣服会重叠，CPU 里没有衣服之间的碰撞。Newton MJCF 里它们是分开的自由刚体。
- CPU 关门约束是门洞平面，不是网格接触。接触求解只在 Newton/MuJoCo 加载 MJCF 时发生，云端没有跑过。
- 瓶身倾角在 CPU 任务里是一个铰链坐标；MJCF 里瓶子是自由刚体，盖子是铰链。GPU 脚本不宣称自己完成了倒液。
- 洗涤时滚筒转、衣服留在筒内，不模拟离心脱水。
- 机械臂是场景里的 4 自由度可达布局，不是 Franka，也不是那次失败的 lift 策略。
- 云端没有 NVIDIA GPU，没有 Isaac，没有 Newton。下面的 CPU 测试就是这里能做的全部检查。
