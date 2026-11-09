# v14 轮腿策略部署交接包（完整闭链模型训练成果）

> 给接手部署的人 / AI：**先按 §2 跑通自检，再读 §3 接口合同和 §4 部署要点**。
> 所有路径都相对本包根目录；本包自包含（策略 + 模型 + 参考运行时 + 证据 + 录像）。

## 1. 这是什么

- 机器人：**v14 轮腿**（15 个 link、每腿 4 条前链关节 + 4 条 `<connect>` 闭链约束、
  6 个电机、总质量 **14.189111 kg**、含底部导轮/滚轮支撑板几何）。
- 训练方式：**mjlab / MuJoCo Warp GPU**，直接在**完整闭链模型**上训练（不是旧的
  「后链串联等效」简化模型）；训练资产与部署 MuJoCo 模型是同一份 MJCF。
- 交付两个策略（都是从同一完整模型训练体系导出）：

| 策略 | 文件 | 能力 |
| --- | --- | --- |
| **RISE-ANNEAL（v1.3，DR 退火折中档，推荐）** | `policy/v14_rise_anneal_deploy.onnx` | **趴卧起身 + 宽指令跟踪（标称 \|vx 误差\| 0.0396，接近 v1.1）+ 保留 DR 组合波动减半** |
| RISE-DR（v1.2，鲁棒档） | `policy/v14_rise_dr_deploy.onnx` | 趴卧起身 + 宽指令跟踪 + 强域随机化训练（组合扰动下速度误差标准差 -73%，代价：fwd 1.5 实测 1.19 m/s） |
| **RISE-WIDE（v1.1，性能档，推荐标称性能）** | `policy/v14_rise_wide_deploy.onnx` | **趴卧姿起身 + 保持 + vx ±1.5 m/s / wz ±1.2 rad/s / 高度 0.14–0.21 m 跟踪** |
| RISE（v1.0 保守档） | `policy/v14_rise_deploy.onnx` | 趴卧姿起身 + 站/走/转/高度（宽指令跟踪误差略大） |
| FLAT | `policy/v14_full_closed_deploy.onnx` | 站/走/转/高度跟踪（站姿起步；不保证趴卧起身） |

- RISE-ANNEAL（v1.3，折中档）：强 DR→弱 DR→无 DR 三段退火，标称 |vx 误差| **0.0396**
  （v1.2 0.0626 / v1.1 0.0349），DR 组合速度误差标准差 **0.0247**（v1.1 0.0499 / v1.2 0.0135）；
  代价：kp×0.55 单项扰动下不能起身（v1.2 可以）。详见 `reports/anneal_cmd_matrix.md`。
- RISE-DR（v1.2，鲁棒档）：训练期强域随机化（质量/质心/摩擦/关节/电机/kp/kd/滞后/延迟/推力/
  编码器偏置/趴卧姿态扰动）。验收：DR 分布内随机组合 12/12 起身成功，速度误差标准差
  **0.050→0.0135（-73%）**、最大误差 -54%；单项扰动与推力恢复 PASS。详见 `reports/robust_cmd_matrix.md`。
- RISE-WIDE（v1.1，性能档）从趴卧姿：**22 条指令全部 PASS**，平均 |vx 误差| 0.035、
  |wz 误差| 0.020、|高度误差| 0.003 m，起身 0.42 s；vx ±1.5、wz ±1.2、高度 0.14–0.21 全覆盖。
- RISE（v1.0）从趴卧姿：起身 0.50 s、末态 0.19 m、保持 0.95，4/4 PASS + 3 扰动种子（见 §5）。
- 训练全过程、失败尝试、诊断与复现命令：`reports/TRAINING_REPORT.md`。

## 2. 自检（拿到包先跑这三条）

依赖：Python ≥3.10，`numpy`、`mujoco>=3.2,<4`、`onnxruntime`（`runtime/requirements.txt`）。

```bash
# ① 文件哈希 + ONNX 数值测试向量 + MuJoCo 模型结构（应输出 PASS）
python runtime/verify_package.py

# ② 从趴卧姿闭环回放 10 s（应看到 h≈0.19 m、倾斜≈1°、只有轮子接地）
python runtime/sim2sim_replay.py --onnx policy/v14_rise_deploy.onnx \
    --no-viewer --duration 10 --init prone

# ③ 从站姿跑前进指令（应看到 vx 跟随 +0.4 m/s 左右、高度稳定）
python runtime/sim2sim_replay.py --onnx policy/v14_rise_deploy.onnx \
    --no-viewer --duration 8 --init stand --cmd-lin-vel 0.4
```

② 的期望末态（实测）：`h≈0.1903 m`、`倾斜≈1.3°`、`接地=轮->`（只有两个主动轮接地）、
全程无摔倒。`runtime/verify_package.py` 会校验 `SHA256SUMS.txt` 里每个文件、
5 组 ONNX 数值测试向量（阈值 1e-4）、以及 MJCF 结构（nq=21/nv=20/neq=4/nu=6/质量）。

## 3. 接口摘要（完整合同见 `INTERFACE.md`）

- **ONNX**：单输入 `obs` **(N,125) float32** → 输出 `actions` **(N,6) float32**。
  125 = **25 维观测 × 5 帧历史，frame-major，帧旧→新**（最新帧在最后）；
  reset 时用当前帧填满 5 帧。**注意**：mjlab 原生是 term-major，本 ONNX 已做
  （等价的）第一层权重列置换，**专门吃 frame-major 历史**——不要自己再转置。
- **观测 25 维**（顺序固定）：
  `角速度×0.25(3)` + `投影重力(3)` + `指令×[2.0,0.25,5.0](3)` +
  `[q_rear_L,q_shank_L,q_rear_R,q_shank_R] − 默认站姿(4)` +
  `6 关节速度×0.05` + `上一帧策略输出(6)`。
- **动作 6 维**：`[左髋(虚拟), 左膝(虚拟), 左轮, 右髋(虚拟), 右膝(虚拟), 右轮]`。
  解码：腿目标 = 默认虚拟角 + 0.5×动作，虚拟 PD `kp=20/kd=1`；轮 `kd=0.2`、
  目标轮速 `= 10×动作`；再按闭链雅可比（虚功原理）映射到真机
  `(后髋, 前髋曲柄, 轮)` 6 个电机力矩；限幅 **±40 / ±40 / ±20 N·m**。
- **控制频率**：策略 100 Hz（10 ms），物理 200 Hz（dt=0.005，decimation=2）。
- **策略输入的是「模型口径」关节角**：`q_rear/q_front` 由编码器换算（含零位/方向），
  `q_shank` 由闭链逆解 `v14_leg_model.V14Leg.solve_loop(q_front,q_rear)` 得到
  （`runtime/v14_leg_model.py`，随包交付）。硬件侧的编码器零位/符号见 `INTERFACE.md` §5。

## 4. 部署要点（必读）

1. **力矩口径必须是 40/40/20 N·m**。v14 MJCF 原始 `ctrlrange=±3.14` 物理上不够
   （站立静力需后髋约 17 N·m）——**任何策略在 ±3.14 下都会 0.2 s 内塌掉**。
   请同步修改部署 MJCF 的 `ctrlrange` 与固件输出限幅。
2. **H723 现有板载 C 推理不兼容本策略**：旧 `policy_v14.c` 是
   `encoder(125→3)+actor(28→6)` 的双输入结构；本策略是单输入
   `125→256→256→256→6`（ELU）MLP。上板需要：
   - 用本包 ONNX 重新生成 C 权重/常量（按 `INTERFACE.md` §2 的层结构），或
   - 改板载推理代码适配新结构；先用 `evidence/test_vectors.json` 在 PC 端逐维对齐
     （与 ONNX 输出误差 <1e-4），再上板。
3. **传感器**：IMU 需给机体系角速度与投影重力（直立时 `(0,0,-1)`）；轮速/关节角速度
   口径见 `INTERFACE.md`。策略没有机体线速度输入，靠 5 帧历史自估。
4. **安全**：先在仿真/吊起状态验证观测符号与力矩方向；限幅从低逐步放开；
   实车摩擦、时延、气弹簧已拆除后的动力学都未验证（见 §6）。

## 5. 已验证结果（完整闭链 MuJoCo，40/40/20 N·m）

从趴卧（MJCF home keyframe）开始，每工况 10 s：

| 工况 | 起身 | 起身时间 | 末态高度 | 保持占比 | 判定 |
| --- | --- | --- | --- | --- | --- |
| 站定 0.18 m | 成功 | 0.50 s | 0.1903 m | 0.95 | PASS |
| 前进 0.5 m/s | 成功 | 0.50 s | 0.1888 m | 0.95 | PASS |
| 转向 0.3 rad/s | 成功 | 0.50 s | 0.1903 m | 0.95 | PASS |
| 高度 0.20 m | 成功 | 0.52 s | 0.2018 m | 0.95 | PASS |

扰动鲁棒性（趴卧初始位姿 base ±2 cm / yaw ±0.3 rad / 关节 ±0.05 rad，3 个种子）：3/3 PASS。
报告的原始数据：`reports/rise_from_prone_sim2sim_summary.md/json`；
FLAT 策略的站/走/转/高度验收：`reports/flat_sim2sim_summary.md`。
录像与分帧图：`media/`（`rise_sequence.png` 是起身分帧，`all_scenarios_1x.mp4` 是 4 场景拼接）。

## 5.1 变高度功能（height 0.14–0.21 m，无需新网络）

策略把 height 指令编码在观测里（25 维第 8 维 ×5.0）。变高度 = 运行时平滑修改 height：
用 `runtime/policy_runtime.py` 的 `HeightCommand`（目标高度 + 斜坡限速 + 范围裁剪，默认
0.12 m/s、0.14–0.21 m）：

```python
from policy_runtime import HeightCommand
hc = HeightCommand(initial=0.18, rate=0.12)   # rate 建议 0.10~0.15 m/s
hc.set_target(0.21)                          # 例如按键“升高”
# 每个 10 ms 控制周期：
cmd_h = hc.step(dt=0.01)                     # 作为观测 commands[2] 使用
```

实测（v1.3，完整闭链 sim2sim）：站立 0.14→0.21 m 平均误差 **6.0 mm**、过渡 **0.20–0.37 s**；
边走边变高（0.15–0.20 m）平均误差 6.0 mm；走+转+变高平均误差 4.3 mm。
录像/曲线见 `media/height_demo/`，明细 `reports/height_change.md` 与
`reports/height_{stand,walk,mixed}.json`。

## 5.2 过地形 / 台阶能力现状（v1.4，阶段性）

- **轻粗糙地形（±2 cm）可用**：v1.3 策略从站姿以 0.4 m/s 指令可前进 3.97 m（体速 0.36 m/s、
  倾角 ≤4°）；地形训练后 ±3 cm 粗糙地形更平顺（倾角 22.5°→3.7°）。
- **台阶 / 跳台阶尚未学会**：台阶垂直面反力把腿 PD 压弯（base 下沉到 0.14–0.15 m），
  盲走策略缺少台阶距离信息；高度指令上限 0.21 m 也限制跳跃冲量。
- 已交付训练/验收设施：`--task terrain`（台阶/粗糙/障碍 + 地形课程 + 爬升奖励）、
  `sim2sim/terrain_scenes.py` + `sim2sim/terrain_eval.py`、`render_v14.py --xml`、
  `height_demo.py --xml/--wave`。
- 部署建议：目前仍用 v1.3 `v14_rise_anneal_deploy.onnx`（平坦/轻粗糙）；台阶场景等后续
  「跳跃课程 + 高度指令扩展到 0.30 m」版本。详见 `reports/terrain_status.md`。

## 6. 已知限制（不要外推）

- **只在 MuJoCo 完整闭链模型上验证**，没有实车数据；实车必须重新验证。
- **±3.14 N·m 口径不可用**（见 §4.1）。
- v1.1（推荐）训练/验证范围：`|vx| ≤ 1.5 m/s`、`|wz| ≤ 1.2 rad/s`、高度 0.14–0.21 m；
  实测平均误差 0.035 m/s / 0.020 rad/s / 0.003 m。高度 0.21 因腿长饱和在约 0.196 m。
- v1.0 保守档的训练/蒸馏范围较窄（|vx|≤0.4 教师数据 + 后续 PPO），宽指令下误差更大。
- 起身过程依赖「默认站姿 PD 先把车抬起来」的物理特性（部署 runtime 的 PD 默认目标
  就是站姿，所以可复现）；该特性在实车上受摩擦/时延影响。
- 底部导轮/滚轮支撑板是 base_link 的固定几何（无独立关节），训练时按「蹭地」惩罚处理；
  若实车设计上允许它们在斜坡承重，需要重新评估。
- 质量/质心、轮径、摩擦、电机效率仍是 CAD/参考值，不是实车称重/标定。

## 7. 文件清单

```
README.md                  ← 本文件
INTERFACE.md               ← 接口/时序/硬件映射的精确合同
SHA256SUMS.txt             ← 全部文件哈希（verify_package.py 会校验）
policy/
  v14_rise_anneal_deploy.onnx ← v1.3 DR 退火折中档（标称接近 v1.1，DR 波动减半）
  v14_rise_dr_deploy.onnx   ← v1.2 鲁棒档（强 DR 训练，组合扰动误差 -73%）
  v14_rise_wide_deploy.onnx ← v1.1 性能档：趴卧起身 + vx ±1.5 / wz ±1.2 / 高度 0.14–0.21
  v14_rise_deploy.onnx      ← v1.0 保守档：趴卧起身 + 站/走/转/高度
  v14_full_closed_deploy.onnx
runtime/
  policy_runtime.py        ← 部署参考数学：观测组装 + ONNX 推理 + 动作解码 + 闭链力矩映射
  v14_leg_model.py         ← v14 闭链运动学（solve_loop / jacobian / torque_map）
  v14_leg.py               ← 另一份单腿运动学参考（sim2sim_replay 用）
  sim2sim_replay.py        ← 已验收的闭环回放器（支持 --init prone/stand）
  verify_package.py        ← 自检脚本
  requirements.txt
model/
  mjcf/                    ← rl_chassis_v14.xml / scene / weld（完整闭链模型，含地面场景）
  meshes/                  ← 视觉+碰撞网格（与训练/验收完全一致）
  urdf/rl_chassis_v14_std.urdf
  config/
evidence/
  test_vectors.json        ← 5 组 (125 维历史 → 6 维动作) 数值测试向量
  provenance.json          ← 策略 sha256、来源 checkpoint、训练方法、验证结论
  rise_env.yaml / rise_agent.yaml   ← 训练环境/PPO 配置存档
  source_checkpoint_sha256.txt
reports/
  TRAINING_REPORT.md       ← 完整训练报告（含失败尝试、诊断、复现命令）
  robust_cmd_matrix.md / robust_v1*.json（v1.2 鲁棒性验收）
  terrain_status.md / terrain_easy_summary.json（v1.4 过地形现状）
  wide_cmd_matrix.md / wide_compare.json / wide_summary.json
  wide_jitter_seed{1,7,42}_summary.json
  rise_from_prone_sim2sim_summary.md/json
  flat_sim2sim_summary.md/json
media/
  rise_sequence.png / rise_preview.gif / all_scenarios_1x.mp4 / montage.png / montage_flat.png
  height_demo/（变高度三模式录像/曲线）  terrain_demo/（±2 cm 粗糙地形通过录像）
```

## 8. 如果接手方要移植到嵌入式

1. 用 `runtime/verify_package.py` 确认 PC 端 ONNX 数值正常。
2. 按 `INTERFACE.md` §2 实现：观测 25 维 → 5 帧历史 → 3 层 MLP（256³，ELU）→ 6 维动作。
3. 用 `evidence/test_vectors.json` 逐维对齐 C 实现（与 PC ONNX 误差 <1e-4）。
4. 实现动作解码 + 闭链力矩映射（`runtime/policy_runtime.py` 的公式可直接照抄），
   用 `runtime/v14_leg_model.py` 生成/校验常数，或移植其解析解。
5. 上板前用 `--dump`（`runtime/sim2sim_replay.py` 支持）导出 PC 仿真每步
   obs/动作/关节/力矩 CSV，与 MCU 逐维对照。
