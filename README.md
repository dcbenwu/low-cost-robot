# low-cost-robot

低成本机器人项目：仿真里用强化学习（RL）训练策略 → 部署到 ESP32 级千元级真机。

**立项背景 / 里程碑 / 预算 / 验收：[PROJECT.md](PROJECT.md)**

## 目录

```
low-cost-robot/
├── PROJECT.md            # 立项章程：背景、里程碑、预算、风险、验收
├── sim/
│   └── m0_nav_demo/      # M0：MuJoCo diff-drive + SB3 PPO（已跑通，见 README）
│       ├── robot.xml
│       ├── env_diffdrive.py
│       └── train.py
├── robot/                # M2：真机（ESP32 固件 / 驱动 / 遥操作），待采购
├── deploy/               # M3：sim-to-real 部署桥，待建
└── docs/
    └── bom.md            # 真机 BOM（预估）
```

## 快速上手

```bash
# M0 训练（udoge 上）
ssh hock@192.168.50.33
cd /home/hock/lcr/m0_nav_demo
/home/hock/lcr/m0env/bin/python train.py --steps 300000
```
