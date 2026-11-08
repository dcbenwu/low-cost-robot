# low-cost-robot

低成本机器人项目：仿真里用强化学习（RL）训练策略，再部署到真机。长期目标是家务人形；当前 M0–M2 的身体是一辆两轮差速小车（两驱 + 万向轮，三脚着地，不是平衡车）。

**立项背景 / 长期愿景 / 里程碑 / 预算 / 验收：[PROJECT.md](PROJECT.md)**

## 目录

```
low-cost-robot/
├── PROJECT.md            # 立项章程：愿景、里程碑、预算、风险、验收
├── sim/
│   ├── m0_nav_demo/      # M0：MuJoCo diff-drive + SB3 PPO（已跑通，见 README）
│   │   ├── robot.xml
│   │   ├── env_diffdrive.py
│   │   └── train.py
│   └── laundry_washer/   # 滚筒洗衣任务环境（刚体衣物代理，无训练好的策略）
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

洗衣任务环境在 `sim/laundry_washer/`。CPU 任务测试不碰 GPU；uDoge 上的一环境 Newton 路径见该目录 README。没有加载 Franka lift 检查点。
