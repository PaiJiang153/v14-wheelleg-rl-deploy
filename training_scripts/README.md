# 训练脚本（可选，给需要复现/继续训练的人）

这些脚本来自训练工作区，依赖 mjlab/MuJoCo Warp 训练环境与 v14 完整闭链模型：

- `gen_passive_rise_states.py`：离线生成「动作全 0 被动抬升」轨迹上的 78 个姿态快照
  （起身课程的接手点）。
- `collect_teacher_rollouts.py`：在完整闭链 MuJoCo 模型里采教师策略轨迹
  （旧 v14_flat 策略 9 条 episode / 10,800 步，obs25+125 历史+动作）。
- `bc_pretrain.py`：把教师轨迹蒸馏进 `125→256³→6` MLP（BC），产出可被 PPO 续训的
  rsl-rl checkpoint。

训练主工程、环境配置与复现命令见 `reports/TRAINING_REPORT.md`。
