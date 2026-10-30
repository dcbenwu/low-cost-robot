# 低成本机器人项目（Low-Cost Robot）

> 一句话：**给机器人配一个 AI 大脑和一副 1–2k 元的身体**——仿真里用 RL 学会任务，最终把学出来的策略部署到便宜的 ESP32 级真机，可遥操作、可自主、可对话。

## 灵感来源

- [alexkoch.ai](https://www.alexkoch.ai/) —— Alexander Koch，Tau Robotics CEO/联合创始人（Berkeley，TUM CS MSc，Z Fellow）
- [tau-robotics.com](https://www.tau-robotics.com) —— 全球唯一的人形清洁机器人**服务**：两台人形机器人上门清洁 1–3 小时，真人远程监督 + AI 判断何时放权自主、何时接管；SF Bay Area + LA，邀请制，$30/hr；BBC/Reuters/AP/USA Today 均报道过
- Koch 本人的研究线（自述）：可扩展机器人学习、语言条件化 meta/分层 RL、模仿学习、**低成本机器人硬件**

本项目 = 该方向里"低成本硬件 × 机器人学习"的自建版：不做百万级人形，做千元级能跑的真机 + 可迁移的策略训练管线。

## 我们的现有资产（复用，不重造）

| 资产 | 位置 | 状态 |
|---|---|---|
| RTX 5090 (32GB) | udoge 192.168.50.33 | Isaac Lab + rsl_rl；lift_franka 已验收（2026-09-18，model_14999.pt）；**注意 .venv 后被清，需重建** |
| ESP-IDF / xiaozhi-esp32 编译链 | Mac | 已验证（见 xiaozhi-esp32-build skill） |
| Mac 常开 + LAN | 192.168.50.x | 远程操控/语音桥 |
| MuJoCo + Stable-Baselines3 | udoge 新 venv | M0 训练回路（本次跑通） |

## 里程碑

- **M0（今天）：训练回路跑通。** 自建 diff-drive 移动机器人 MuJoCo 环境 + SB3 PPO，目标 reach-go 任务，headless 训练出可验收的 reward 曲线 + 策略 rollout。✅/❌ 由指标说话
- **M1：主仿真栈恢复。** Isaac Lab `uv sync` 重建 + mobile_base/velocity env 上轮式/四足机器人训练（备选 LeRobot）
- **M2：真机 BOM 定稿 + 采购 + 组装。** ESP32 主控 + 差速底盘 + 电机驱动 + IMU + 超声/LDROB/LiDAR（**需老板拍板预算**）
- **M3：sim-to-real。** M0/M1 策略部署到真机；遥操作/自主切换（低成本版"真人监督 + AI 放权"模式，致敬 Tau 的业务形态）
- **M4（可选）：** xiaozhi 语音接入、基础 SLAM/建图、语言条件化任务（对齐 Koch 的研究方向）

## 预算

- M0/M1：**0 元**（自有算力）
- M2：约 **¥1,500–2,500** 一次性（底盘 ¥400–700 / 驱动板 ¥60–150 / 电机×2 ¥200–400 / 轮组+编码器 ¥150–300 / IMU+超声 ¥100–200 / LiDAR（可选）¥500–900 / 电池+杂项 ¥200）
- 合计预期 **< ¥3,000**，不含人形/四足（明确不做）

## 风险与对策

| 风险 | 对策 |
|---|---|
| Isaac .venv 被清，重建多 GB 且与矿工互斥 | M0 先用 MuJoCo 轻栈；重建时 pkill 矿工并播报、用毕请示拉回 |
| sim-to-real gap | 先验证速度跟踪/目标到达等低层任务，不做一步到位全自主 |
| 本机 GPU 三互斥（出片/矿工/对话） | 训练任务排空闲时段，跑前查 nvidia-smi 显存占用 |
| 硬件到货周期 | BOM 全部淘宝可买现货，预留 3–7 天物流 |

## 验收标准（每级都要真数字）

- **M0**：训练 reward 曲线存在且上升；测试集 goal-reach 成功率 ≥ 60%（100 episodes）；模型+曲线落盘
- **M1**：velocity/reach env 收敛，成功率和 episode 时长有记录
- **M2**：真机通电后 30s 内可遥操作响应（Mac → ESP32 双向）
- **M3**：训练好的策略上真机，velocity/goal 误差有阈值记录

## 不做（明确边界）

- 不做人形/四足（成本越线）
- 不做付费云/订阅服务
- 不碰付费模型/付费 API（成本纪律）
