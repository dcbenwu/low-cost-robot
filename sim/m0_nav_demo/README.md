# M0 — 训练回路 demo（diff-drive goal-reaching）

最简闭环：**一个差速移动机器人（MuJoCo）→ SB3 PPO → 学会到达随机目标点 → 指标落盘**。
本目录是整条 sim-to-real 管线的最小可运行版本，后续 M1（Isaac Lab）复用同样的"定义环境 → 训练 → 评估 → 验收指标"结构。

## 跑法（udoge，已备好 m0env）

```bash
ssh hock@192.168.50.33
cd /home/hock/lcr/m0_nav_demo
/home/hock/lcr/m0env/bin/python train.py --steps 300000 --seed 0
```

## 验收（见 PROJECT.md）

- 训练曲线 `out/curve.csv` 存在且 reward 上升趋势
- 100 episodes 确定性评估成功率 ≥ 60%
- `out/results.json` 指标落盘

## 文件

| 文件 | 说明 |
|---|---|
| `robot.xml` | MuJoCo 差速底盘（盒体 + 前万向球 + 左右驱动轮，2 力矩执行器） |
| `env_diffdrive.py` | gymnasium 环境：obs=8（body 系目标/朝向/速度/轮速），action=2 归一化力矩，reward=-dist+success 10，4s 超时 |
| `train.py` | PPO 训练 + 确定性评估 + 指标落盘 |
