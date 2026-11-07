# v14 完整闭链模型强化学习训练报告

生成：2026-11-07（Asia/Shanghai）。工作区：`/home/paimiomio/rl`。
对照对象：旧方案（`plane_v14`，Fudan `wheel_legged_gym` + Isaac Gym，**后链串联等效简化模型**）。

> 一句话：本轮不再用「后链串联等效」简化训练模型，改成在 **v14 完整闭链模型**上训练
> （15 hinge + freejoint、4 条 `<connect>`、含底部导轮/滚轮支撑板几何、14.189 kg），
> 用 mjlab / MuJoCo Warp 在 RTX 5060 上训练；同时发现并修掉了一个会让新策略在部署侧
> 完全失效的 **观测历史布局** 口径问题。

---

## 1. 结论摘要

（待正式训练结束后补充最终数字）

| 项 | 结果 |
| --- | --- |
| 训练资产 | v14 **完整闭链** MJCF（不是简化串联模型） |
| 训练后端 | mjlab 1.6.0 / MuJoCo Warp 3.11.0 / torch 2.11+cu128，RTX 5060 Laptop 8G |
| 模型结构自检 | `neq=4`、`nu=6`、`nbody=16`、总质量 14.1891110559 kg、训练/部署 MJCF 除 ctrlrange 外逐字节一致 |
| 运动学校验 | v14 闭链 torch 批量版 vs `v14_leg_model.py` 参考：21/21 项 PASS（残差/雅可比/站姿/批量/CUDA） |
| 训练规模 | 1024 envs × 3000 iter（策略 100 Hz，dt=0.005 × decimation 2），约 1 h 10 min |
| 训练结果（末段） | reward 41.4、ep_len 1974/2000、track_lin 0.91、track_yaw 0.95、height 0.95、stand_height 0.178、零指令漂移 −0.003 m/s |
| sim2sim（完整闭链 MJCF，40/40/20） | 站/走/转/高度 **4/4 PASS**；高度 0.1877/0.18，倾角 ≤1.04°，双轮接地 100%，非轮蹭地 0 步 |
| sim2sim（±3.14 N·m 参考口径） | 4/4 FAIL：任何策略都站不住（静力需 ~17 N·m 后髋 / ~7 N·m 前髋） |
| 关键修复 | mjlab 历史布局 = term-major，部署 spec = frame-major；已在导出侧用 W1 列置换精确修正（否则策略直接翻车） |
| 交付 | `policy/v14_full_closed_deploy.onnx`（部署/sim2sim 用）+ `policy/v14_full_closed_raw.onnx`（原生口径） |
| 起身任务（趴卧→站走） | **已实现**：`policy/v14_rise_deploy.onnx`（sha256 `a120eb39…`）从趴卧姿自主起身 **0.50 s**、保持占比 **0.95**，起身后站/走/转/高度 **4/4 PASS**（另 3 个扰动种子也 PASS）；方法=教师轨迹蒸馏（BC）+ PPO 精调 + 被动抬升快照课程，见 §6 |
| 起身任务算力 | 教师数据采集 10.8k 步 + BC 80 epoch（~2 min）+ PPO 600 iter @1024 env（~15 min） |

---

## 2. 旧方案为什么不行（量化证据）

旧 `plane_v14` 训练模型由 `build_train_urdf.py` 从 v14 完整 URDF **裁剪**而来：

| 项 | 旧训练模型（简化） | v14 完整模型（本轮） |
| --- | --- | --- |
| link 数 / 关节数 | 7 link / 6 revolute | 15 link / 14 hinge + freejoint |
| 前链闭链 | **删除**（前链 4 关节 + 2 销全删） | 保留，4 条 `<connect>` |
| 底部导轮/滚轮支撑板 | 不在（固定装饰件被删） | 在（base_link 碰撞体） |
| 总质量 | 13.584338 kg | **14.1891110559 kg**（+0.604773 kg） |
| 几何 | 后链串联等效（虚拟膝映射回真机） | 真机闭链（训练即部署同一套物理） |
| 训练框架 | Isaac Gym + Fudan wheel_legged_gym | mjlab / MuJoCo Warp |

把旧策略放到**完整闭链模型**上评估（`sim2sim/out/v14_flat/`，评估器已用
`v14_flat.onnx` 逐项复现原 TRAINING_REPORT：站立 0.1767 m / 倾角 0.8° / 零指令漂移
−0.081 / 高度 0.20 指令 0.1940 / ±3.14 时塌到 0.0469，全部吻合）：

| 口径 | 站立高度 | 前进 0.5 | 转向 0.3 | 高度 0.20 | 判定 |
| --- | --- | --- | --- | --- | --- |
| 训练口径 40/40/20 N·m | 0.1767 m、倾角 0.80° | 0.653 m/s | 0.247 rad/s | 0.1940 m | 4/4 PASS |
| 部署 MJCF 口径 ±3.14 N·m | 0.047 m @0.19 s | 塌 | 塌 | 塌 | 4/4 FAIL |

**两个硬结论**：

1. **简化模型训练出的策略在完整闭链模型上「训练口径」下仍然能站**（0.1767 m），
   说明旧结果不是完全不能用；但 **±3.14 N·m 的部署 MJCF 口径下 0.19 s 就塌**——
   站立静力需求约 9~10 N·m/虚拟关节，±3.14 物理上不够（髋力矩需求 >99% 采样超限）。
   这是旧策略实车不行的**首要原因**，且与模型简化与否无关：训练力矩口径 ≠ 实车能力。
2. 旧策略稳态动作是 `[0.12, 1.22, −0.14, −0.12, −1.24, 0.14]`——**膝动作 ~±1.2**。
   旧 v12 训练代码里有「物理动作硬裁 ±1」的安全钳位，若照搬会把 v14 的站立动作直接
   截死；本工程改为可配置 `action_clip=3.0`（部署运行时只按 CLIP_ACTIONS=100 裁剪）。

> 关于「完整模型」本身的价值：旧策略能迁移到完整模型，但用的仍是「简化模型训练的
> 网络 + 部署侧闭链映射」，训练时从未见过完整闭链的接触/约束/导轮几何。本轮把训练
> 与部署统一到同一份 MJCF，目标是消除这层 sim2sim gap，而不是靠后处理补偿。

---

## 3. 本轮做了什么

### 3.1 资产：完整闭链 MJCF（训练 = 部署）

- `model/rl_chassis_v14/mjcf/rl_chassis_v14.xml`：v14 完整模型（部署/实机口径 `±3.14`）。
- `model/rl_chassis_v14/mjcf/rl_chassis_v14_train.xml`：自动生成副本，**只改 6 个电机
  ctrlrange**（前髋 40 / 后髋 40 / 轮 20 N·m）；`mjcf.compare_geometry` 证明两者除
  ctrlrange 外逐字节一致。
- 结构自检（`scripts/check_train_mjcf.json`）：`neq=4`（`L_loop1/L_loop2/R_loop3/R_loop4`）、
  `nu=6`、`nbody=16`、`nq=21`、`nv=20`、总质量 14.1891110559 kg、两轮 geom 是解析圆柱。

### 3.2 运动学：v14 闭链 torch 批量实现

`src/mjlab_wheelleg_v14/leg_torch.py`（纯 torch，CPU/CUDA、float32/float64、任意 batch）
提供解析 `residual` / `constraint_jacobian` / `shank_jacobian` / `solve_virtual_batch` /
`theta0` / `virtual_leg` / `ik_virtual`，几何全部从 v14 MJCF 的 `<body>/<site pos>` 推导。

校验（`scripts/check_leg_torch_v14.json`，21/21 PASS）：

- 串联常数 vs 参考 `v14_leg_model.py`：最大偏差 5.6e-17
- 闭链残差 vs 参考：2.2e-16；雅可比 vs 参考：1.9e-7（门限 1e-4）
- 站姿：`CLOSED_DEFAULT` 残差 6.2e-17、base 0.18 m 时轮底 −1.2e-7 m（正好触地）
- `solve_virtual_batch`（reset 用）：1.9e-16，batch 1/7/1024/4096 全过；CUDA vs CPU 1.7e-15
- 性能（CUDA float32, n=8192）：`shank_jacobian` ≈0.8 ms、`solve_virtual_batch` ≈5 ms

### 3.3 训练环境（mjlab ManagerBasedRlEnv）

- 观测 25 维（`ang_vel×0.25 / gravity / commands×[2,0.25,5] / 虚拟髋膝角偏差 4 /
  6 关节速度×0.05 / 上一动作 6`）× 5 帧历史；动作 6 维（虚拟髋/膝位置 +0.5·a、轮速 10·a）。
- 闭链力矩映射：虚拟 PD(kp=20/kd=1，轮 kd=0.2) → 解析雅可比虚功映射到真机
  (后髋, 前髋曲柄) → 15 ms 一阶执行器滞后 → 6 个 motor。
- 站姿 0.18 m（`L0=0.23508 m`）；reset 用闭链一致解（前链 4 关节牛顿投影，
  避免 `<connect>` 开局拉拽）；底盘质量随机化 ±2 kg、轮摩擦 0.6~1.4。
- 终止：超时 20 s、倾角超限 1 s、`base_too_low <0.06 m` 0.3 s、底盘/腿/导轮支撑板
  持续 >30 N 接触 0.5 s。
- 课程：stage 0 全站定 → 站稳（近期均值 ep_len ≥ 8 s）后 stage 1 放开速度/偏航指令。
- 执行器口径：`--actuator-profile train`（默认，40/40/20）/ `actual`（±3.14，参考口径，压力测试）。

相对 v12 闭链训练基线（v12 mjlab 平地任务）按 v14 实测证据改了 4 处：

| 项 | v12 基线 | v14 本轮 | 依据 |
| --- | --- | --- | --- |
| 物理动作硬裁 | ±1 | **±3** | v14 站姿静力需动作 ~0.97、旧策略实测膝动作 ±1.22；±1 会顶死 |
| 动作探索 init_std | 0.5 | **1.0** | 同上，起步需要覆盖 ~1 量级动作才能保持站姿 |
| LR 调度 | adaptive | **fixed 1e-3** | v14 属重模型，adaptive 实测被 KL 踩到 1e-5 卡死（v12 solid 同结论） |
| 欠高惩罚 | 无（仅 solid 变体） | **height_under −5** | 14.19 kg + 深蹲站姿从零训练收敛到「低伏解」（stand_height 0.07、height_err 0.11） |

### 3.4 关键修复：观测历史布局（term-major vs frame-major）

- mjlab 的 `history_length=5` 把历史按 **term-major** 展开：
  `[ang_vel×5帧, gravity×5帧, ..., applied_action×5帧]`。
- 部署 spec（`deploy_policy_v14`）要求 **frame-major**：
  `[frame0(25维), frame1(25维), ..., frame4(25维)]`。
- 直接拿 mjlab 原生 ONNX 跑 sim2sim：策略输出 ±5 动作、0.5 s 弹到 0.55 m 后翻车
  （iter 200 checkpoint 实测）。
- 修复：MLP 第一层是 Linear，输入重排 = 重排 W1 的列，数学上精确等价。
  `export_deploy_onnx.py` 导出部署口径 ONNX（`deploy(x_frame_major) == raw(x_term_major)`，
  数值校验 9.5e-7）。同一 checkpoint 修复后 sim2sim：**高度 0.1869、倾角 14.4°、PASS**。

---

## 4. 训练结果（平地任务：站定 / 走行 / 转向 / 高度跟踪）

run：`logs/rsl_rl/wheelleg_v14_closed/2026-11-07_18-08-19_formal_1024x3000`，
1024 envs × 3000 iter，1024 env 下 1.1~2.2 s/iter（约 4~5 万 env-step/s），总耗时 1:09:43。

![训练进度（仪表盘静态图）](../../dashboard/out/progress_flat_formal.png)

| 指标（末段 20 点均值） | 起步（~iter 20） | iter 1000 | iter 2000 | 末段（iter ~3000） |
| --- | --- | --- | --- | --- |
| mean reward | −0.68 | ~30 | ~40 | **41.36** |
| 平均 episode 长度（满 2000 = 20 s） | 20 | ~1900 | ~1950 | **1974**（98.7%） |
| 线速度跟踪（1.0=完美） | 0.009 | — | — | **0.9105** |
| 偏航跟踪 | 0.006 | — | — | **0.9539** |
| 高度跟踪 | 0.007 | — | — | **0.9529** |
| 高度误差（m） | 0.040 | — | — | **0.0136** |
| 站定高度（指令 0.18，m） | 0.148 | — | — | **0.1778** |
| 站定漂移（m/s） | −0.0887 | — | — | **−0.0032** |
| base 塌陷终止（次/轮） | 0 | 13 峰值 | ~2 | **0.05** |
| 底盘蹭地终止 / 倾角终止 | 0 / 0 | — | — | **0 / 0.02** |

训练过程：stage 0（只站定）约在 iter 250 达标（近期平均 ep_len ≥ 800）进入 stage 1
（站定 50% + 速度/偏航指令），之后指标快速拉起并在 1000 iter 后进入平台，后续在打磨精度。
早期（iter 20~100）出现「低伏稳定解」（stand_height 0.07~0.10 m、height_err 0.11 m），
`height_under` 欠高惩罚 + 固定 LR(1e-3) + 更大探索 std 后压回，见 §3.3。

训练截图证据：`/home/paimiomio/rl/dashboard/out/progress_flat_formal.png`（仪表盘 3×4 面板静态图）。

---

## 5. sim2sim 验收（完整闭链 MJCF）

评估器：`sim2sim/run_matrix.py`（复用 `reference/mujoco_deploy/onnx_mj_v14.py` 的 obs/action/PD/
闭链力矩映射），策略 ONNX 用 **部署口径**（frame-major 历史，W1 置换）。
每条工况 10 s，模型 = `model/rl_chassis_v14/mjcf/rl_chassis_v14_scene.xml`（完整闭链）。

### 5.1 训练/部署力矩口径（train，40/40/20 N·m）—— 4/4 PASS

| 工况 | 高度/指令 (m) | 高度误差 | vx/指令 (m/s) | 误差 | wz/指令 (rad/s) | 误差 | 倾角均值/最大 | 轮接地 | 闭链 max | 非轮蹭地 | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stand 0.18 | 0.1877/0.18 | +0.0077 | −0.0174/0 | −0.0174 | −0.0006/0 | −0.0006 | 0.63/1.04° | 2.00 | 0.62 mm | 0 | **PASS** |
| fwd 0.5 | 0.1906/0.18 | +0.0106 | 0.3835/0.5 | −0.1165 | −0.0193/0 | −0.0193 | 0.46/1.23° | 2.00 | 0.62 mm | 0 | **PASS** |
| yaw 0.3 | 0.1875/0.18 | +0.0075 | 0.0088/0 | +0.0088 | 0.2774/0.3 | −0.0226 | 0.85/0.85° | 2.00 | 0.67 mm | 0 | **PASS** |
| height 0.20 | 0.2042/0.20 | +0.0042 | −0.0186/0 | −0.0186 | 0.0013/0 | +0.0013 | 0.90/0.99° | 2.00 | 0.72 mm | 0 | **PASS** |

曲线：`sim2sim/out/v14_full_closed/curves_*.png`、总览 `overview.png`、汇总 `summary.md/json`。
ONNX sha256 `f59facc5866834c9e52cf616391389c5bc5ebb90e09d37920211a18ed117eccf`。

### 5.2 ±3.14 N·m 参考口径 —— 4/4 FAIL（模型力矩上限问题，不是策略问题）

| 工况 | 高度 | 倾角 | 髋饱和 | 需求超限 | 判定 |
| --- | --- | --- | --- | --- | --- |
| 站/走/转/高度 | 0.038 m | 5~6° | 100% | 100% | FAIL: fell（~0.2 s） |

**这一口径下任何策略都站不住**：站立静力虚拟髋/膝力矩 ~9–10 N·m，经 d≈±0.75 映射后
后髋电机需 **~17 N·m**、前髋曲柄需 **~7 N·m**，远超 ±3.14。用户已确认 ±3.14 不是实车硬限
（DM 电机 TMAX=54 N·m），因此交付口径应为 **40/40/20**，并同步修改部署 MJCF 的
`ctrlrange` 与固件输出限幅（见 §7）。

### 5.3 与旧「简化模型」策略在同一完整闭链模型上的对比（train 口径）

| 指标 | 旧策略（简化模型训练，v14_flat） | 新策略（完整闭链训练） | 变化 |
| --- | --- | --- | --- |
| 站定高度（指令 0.18） | 0.1767（−3.3 mm） | 0.1877（+7.7 mm） | 新策略略高 |
| 零指令漂移 vx | **−0.0806 m/s** | **−0.0174 m/s** | 改善 4.6× |
| 前进 0.5 m/s | 0.6529（+30.6%，WARN） | 0.3835（−23.3%，PASS） | 不再超速、过检 |
| 转向 0.3 rad/s | 0.2472（−17.6%） | 0.2774（−7.5%） | 误差减半 |
| 高度 0.20 m | 0.1940（−6.0 mm） | 0.2042（+4.2 mm） | 更准 |
| 倾角（站定） | 0.80° | 0.63° | 更稳 |
| 非轮蹭地 / 闭链残差 | 0 步 / 0.59 mm | 0 步 / 0.62 mm | 持平 |

> 注意：两者不是同分布对比（旧策略是 Isaac Gym 简化模型 2048 env × 3000 iter 训出；
> 新策略是本工程完整闭链 1024 env × 3000 iter）。但**在同一完整闭链模型上评测**，
> 新策略在零指令漂移、前进速度过冲、转向误差上都有实质改善，同时第一次做到了
> 「训练资产 = 部署资产」。旧策略前进更快但超出跟踪判据（WARN），新策略保守但过检。

---

## 6. 起身任务（趴卧姿 → 起身 → 站走）：已实现

### 6.1 最终结果（完整闭链 MJCF，40/40/20 N·m）

策略：`policy/v14_rise_deploy.onnx`（sha256 `a120eb39b60e12c348695de3ba0749d9f923f8ef4b7b2f12a14c4c4d89a52035`），
`sim2sim --init prone` 结果（`sim2sim/out/v14_rise/summary.md`）：

| 工况（从趴卧开始） | 起身时间 | 最高/末态高度 | 保持占比 | 判定 |
| --- | --- | --- | --- | --- |
| 站定 0.18 | 0.50 s | 0.210 / 0.190 m | 0.95 | **PASS** |
| 前进 0.5 m/s | 0.50 s | 0.196 / 0.189 m | 0.95 | **PASS** |
| 转向 0.3 rad/s | 0.50 s | 0.200 / 0.190 m | 0.95 | **PASS** |
| 高度 0.20 m | 0.52 s | 0.227 / 0.202 m | 0.95 | **PASS** |

扰动鲁棒性（`--init-jitter`，base ±2 cm / yaw ±0.3 rad / 关节 ±0.05 rad，3 个种子）：
**3/3 PASS**，起身 0.47~0.48 s、末态高度 0.190 m、保持 0.95。

录像：`sim2sim/render/rise_final/`（`stand/forward/turn/combo` 均从趴卧起步；
`all_scenarios_1x.mp4` 为拼接版；`rise_sequence.png` 为起身分帧图）。

### 6.2 怎么训出来的（三次失败后的有效路径）

失败尝试（前 3 轮）见 `logs/rsl_rl/wheelleg_v14_rise/`：
1. 趴卧 50% + 站姿 50% 直接热启动 → 趴卧 episode 全部在宽限期结束判死，策略不动；
2. 加蹲姿混合 + 提高熵 → 前 300 iter 指标好，iter~3400 后 policy std 爆炸（0.2→1.4）崩溃；
3. 分阶段课程 + 稳定 LR 3e-4 → 蹲姿起身 99% 学会，但趴卧起身始终 0；sim2sim 里策略在
   趴卧态输出对称大动作把车「按住」。

**根因**：趴卧态对「只在站姿分布训练过」的网络是严重 OOD；纯 PPO 很难从零探索出
「先让默认站姿 PD 把车抬起来（~1 s 内 0.034→0.12 m、前倾 25°）→ 再接手维稳」这一步。

**有效方案 = 教师蒸馏 + 快照课程 + PPO 精调**：
1. **教师**：旧 `v14_flat.onnx` 在完整闭链模型上实测能从趴卧起身并保持（见
   `sim2sim/out/v14_flat/`）；用 `sim2sim/collect_teacher_rollouts.py` 在完整模型里
   采 9 条 episode / 10,800 步（趴卧/站姿起步 + 站/走/转/高度指令），记录
   obs25 + 125 维 frame-major 历史 + 动作。
2. **BC**：`scripts/bc_pretrain.py` 把历史转成 mjlab 的 term-major 布局，以平地策略
   权重为初值监督训练 `125→256³→6` MLP，80 epoch 后 MSE **3.3e-4 / MAE 0.0068**
   → 直接 sim2sim 从趴卧：4/4 rise PASS（0.53~0.65 s）。
3. **快照课程**：`scripts/gen_passive_rise_states.py` 离线生成「动作全 0 被动抬升」
   轨迹上的 78 个姿态快照（`model/rl_chassis_v14/states/passive_rise_qpos.npz`）；
   `ResetMixedStart` 按 25% 概率直接从这些快照 reset（训练策略接手被动抬升后的姿态）。
4. **PPO 精调**：`--task rise`，1024 env × 600 iter，LR 3e-4 fixed、entropy 0.004、
   `height_progress` 稠密塑形，从 BC checkpoint 续训 → `rise_frac_post` 从 0.08 升到
   **0.986**（能从被动抬升快照接手的比例），最终策略保持 BC 的趴卧起身能力并更稳。

### 6.3 复现命令

```bash
cd /home/paimiomio/rl
# 1) 生成被动抬升快照；2) 采教师轨迹；3) BC 热启动
train/mjlab_v14/.venv/bin/python train/mjlab_v14/scripts/gen_passive_rise_states.py
/home/paimiomio/轮腿sim2real/.venv-policy/bin/python sim2sim/collect_teacher_rollouts.py \
    --teacher sources/KaiWuRoboMaster-wxk-s-branch/rl_chassis_v14/rl/mujoco/v14_flat.onnx --n 9
cd train/mjlab_v14
.venv/bin/python scripts/bc_pretrain.py \
    --data ../../model/rl_chassis_v14/states/teacher_rollouts.npz \
    --base-checkpoint logs/rsl_rl/wheelleg_v14_closed/2026-11-07_18-08-19_formal_1024x3000/model_2999.pt
# 4) PPO 精调（rise 任务，热启动 = BC）
.venv/bin/python train.py --task rise --num-envs 1024 --max-iterations 600 \
    --run-name rise_bc_ft --resume --load-run warmstart_bc
# 5) 导出 + 从趴卧验收
.venv/bin/python export_deploy_onnx.py \
    --checkpoint logs/rsl_rl/wheelleg_v14_rise/<run>/model_3598.pt \
    --out ../../policy/v14_rise_deploy.onnx
cd /home/paimiomio/rl
/home/paimiomio/轮腿sim2real/.venv-policy/bin/python sim2sim/run_matrix.py \
    --onnx policy/v14_rise_deploy.onnx --label v14_rise --init prone --profiles train
```

### 6.4 剩余边界

- 力矩口径仍是 40/40/20；±3.14 N·m 下同样站不住（与 §8 一致）。
- 起身依赖「默认站姿 PD 先把车抬起来」的物理特性（部署 runtime 里同样是这个 PD 目标，
  所以可复现）；但在实车（摩擦/时延/气弹簧拆除后动力学）上需要重新验证。
- 教师蒸馏只覆盖教师见过的指令范围（|vx|≤0.4、|wz|≤0.25、高度 0.16~0.20）；
  PPO 精调把 `rise_frac_post` 拉到 0.986，但更大指令范围还需继续训练。

## 7. 与旧方案的量化对比

见 §2（旧方案为什么不行）与 §5.3（同一完整闭链模型上的新旧策略对比）。


---

## 8. 已知边界与后续建议

1. **力矩口径是实车部署的第一瓶颈（用户已确认可提高）**：v14 部署 MJCF 的
   `ctrlrange=±3.14 N·m` 对 14.189 kg 的深蹲站姿物理上不够（静力需求 ~9–10 N·m/虚拟关节，
   映射后后髋电机 ~17 N·m）；当前固件输出上限（1 N·m/1000 counts 量级）更低。
   用户已确认 ±3.14 不是实车硬限（DM 电机 TMAX=54 N·m 已读回）→ 部署侧需要：
   - 把 v14 部署 MJCF 的 6 个 motor `ctrlrange` 改为训练口径 **40/40/20**（或按电机能力取更高），
   - 同步提高固件 MIT/电流指令的输出限幅与 PID 保护阈值，
   - 之后用 `--actuator-profile train` 的 sim2sim 结果作为实车预期（已 4/4 PASS）。
2. **H723 板载推理结构不匹配**：旧板载 C 代码是 `encoder(125→3)+actor(28→6)` 双输入结构；
   本工程导出的是单输入 125→256³→6 MLP。上板需要重新生成 C 权重并改网络结构
   （`v14_deploy/gen_policy_v14_mcu_weights.py` 需适配）。
3. **底部导轮/滚轮支撑板**没有独立关节（在 base_link 碰撞体里），当前计入「蹭地」惩罚
   与终止。若实车设计上它们允许在斜坡/台阶承重，需要按几何位置把对应凸块从
   `BAD_CONTACT_GEOMS` 排除。
4. **0.18 m 是深蹲站姿**（腿可达约 0.46 m），静力力矩大；如果实车目标高度可以更高，
   提高站姿会显著降低力矩需求，部署可行性更好。
5. 质量/质心、轮径、摩擦、电机效率仍是 CAD/参考值，不是实车称重。

---

## 9. 复现命令

```bash
cd /home/paimiomio/rl/train/mjlab_v14
P=.venv/bin/python

# 资产与结构自检
$P scripts/make_train_mjcf.py

# 运动学校验（21 项）
$P scripts/check_leg_torch_v14.py

# 环境/GPU 冒烟
$P scripts/smoke_env.py --num-envs 64 --steps 100

# 训练（完整闭链模型；train=40/40/20 口径，actual=±3.14 口径）
$P train.py --actuator-profile train --num-envs 1024 --max-iterations 3000 \
    --run-name v14_full_closed

# 导出
$P export_onnx.py --checkpoint logs/rsl_rl/wheelleg_v14_closed/<run>/model_3000.pt \
    --out ../policy/v14_raw.onnx            # mjlab 原生（term-major 历史，训练/内部用）
$P export_deploy_onnx.py --checkpoint logs/rsl_rl/wheelleg_v14_closed/<run>/model_3000.pt \
    --out ../policy/v14_deploy.onnx          # 部署/sim2sim 用（frame-major，W1 置换）

# sim2sim（完整 v14 闭链 MJCF，train + actual 两口径）
/home/paimiomio/轮腿sim2real/.venv-policy/bin/python ../../sim2sim/run_matrix.py \
    --onnx ../policy/v14_deploy.onnx --label v14_full_closed
```

## 10. 证据索引

| 内容 | 路径 |
| --- | --- |
| 训练栈安装/GPU 验证 | `env/ENV_REPORT.md`、`env/logs/` |
| 完整模型结构自检 | `train/mjlab_v14/scripts/check_train_mjcf.json` |
| 运动学 21 项校验 | `train/mjlab_v14/scripts/check_leg_torch_v14.json` |
| 训练日志/ckpt | `train/mjlab_v14/logs/rsl_rl/wheelleg_v14_closed/<run>/` |
| 训练可视化 | `dashboard/`（实时 `127.0.0.1:8787`） |
| 训练成果仿真录像 | `sim2sim/render/full_closed/`（5 场景 mp4/gif/静帧；旧策略对照 `sim2sim/render/old_simplified/`） |
| sim2sim 结果 | `sim2sim/out/<label>/summary.{json,md}` |
| 旧策略完整模型评估（复现原报告） | `sim2sim/out/v14_flat/`、`sim2sim/out/v14_domain_robustness/` |
