# 接口合同（精确版）

本文是部署时必须逐项对齐的接口；任何一项口径不同都会导致「仿真能站、实车乱走」。
配套参考实现：`runtime/policy_runtime.py`（数学逐项对照训练/回放）。

## 1. ONNX 策略

| 项 | 值 |
| --- | --- |
| 文件 | `policy/v14_rise_anneal_deploy.onnx`（v1.3 DR 退火折中档）/ `policy/v14_rise_dr_deploy.onnx`（v1.2 鲁棒档）/ `policy/v14_rise_wide_deploy.onnx`（v1.1 宽指令）/ `policy/v14_rise_deploy.onnx`（v1.0 保守档）/ `policy/v14_full_closed_deploy.onnx`（平地） |
| sha256 | 见 `SHA256SUMS.txt` / `evidence/provenance.json` |
| 输入 | 名 `obs`，形状 `(batch, 125)`，dtype **float32**；**只有这一个输入** |
| 输出 | 名 `actions`，形状 `(batch, 6)`，dtype float32 |
| 网络 | MLP：125 → 256 → 256 → 256 → 6，隐藏层 **ELU**，无 obs normalization |
| 输出含义 | 原始动作（训练时在解码前硬裁到 ±3；部署侧建议同样裁剪，见 §4） |

**历史布局**：125 = 5 帧 × 25 维，**frame-major、帧旧→新**：
`[frame0(25), frame1(25), frame2(25), frame3(25), frame4(25)]`，其中 frame4 是最新一帧。
reset（上电/跌倒/切换）时把 5 帧全部填成当前帧（`reset_fill=current`）。
> mjlab 训练内部是 term-major；本 ONNX 的第一层权重已按置换矩阵重排，
> 所以**输入必须是 frame-major**。自己去转置反而会错。

## 2. 单帧 25 维观测（顺序固定，不可交换）

| 序号 | 名称 | 维度 | 计算 | 备注 |
| --- | --- | --- | --- | --- |
| 0–2 | `ang_vel` | 3 | 机体坐标系角速度 × **0.25** | rad/s，IMU gyro（body 系） |
| 3–5 | `gravity` | 3 | 机体系投影重力 | 直立 = `(0,0,-1)`，需归一化 |
| 6–8 | `commands` | 3 | `[vx_cmd, wz_cmd, height_cmd] × [2.0, 0.25, 5.0]` | m/s, rad/s, m |
| 9–12 | `hip_knee_pos` | 4 | `[q_rear_L, q_shank_L, q_rear_R, q_shank_R] − default_virtual` | 弧度（模型正方向） |
| 13–18 | `joint_vel` | 6 | `[qd_rear_L, qd_shank_L, qd_wheel_L, qd_rear_R, qd_shank_R, qd_wheel_R] × 0.05` | rad/s |
| 19–24 | `last_action` | 6 | 上一策略步的**策略原始输出**（不是延迟后的） | 无缩放 |

- `default_virtual = [−0.865168, −0.774701, +0.865169, +0.774700]`
  （左 rear, 左 shank, 右 rear, 右 shank；即 base 0.18 m 站姿）。
- 指令范围（超范围会被内部裁剪）：`vx∈[−1.2,1.2]`、`wz∈[−1.0,1.0]`、`height∈[0.14,0.20]`
  （v1.1 训练/验证覆盖 vx ±1.5、wz ±1.2、高度 0.14–0.21，运行时裁剪可放宽）。
  默认站定指令 `(0, 0, 0.18)`。
- 整帧最后再 `clip(-100,100)`（正常不会触发）。
- `q_shank` 在真机侧由 `v14_leg_model.V14Leg.solve_loop(q_front, q_rear)` 逆解得到；
  `qd_shank` 可由雅可比组合 `d_rear*qd_rear + d_front*qd_front`（也可直接差分）。

## 3. 动作解码（6 维 → 6 个电机力矩）

动作顺序：`[左髋(虚拟), 左膝(虚拟), 左轮, 右髋(虚拟), 右膝(虚拟), 右轮]`。

```
# 每侧 side ∈ {left, right}，i=0/1
a = clip(action, -3, 3)                       # 训练 action_clip=3
target_hip   = default_virtual[side][0] + 0.5 * a[3i]
target_knee  = default_virtual[side][1] + 0.5 * a[3i+1]
tau_hip  = 20 * (target_hip  - q_rear)  - 1.0 * qd_rear
tau_knee = 20 * (target_knee - q_shank) - 1.0 * qd_shank
tau_wheel= 0.2 * (10 * a[3i+2] - qd_wheel)
tau_hip  = clip(tau_hip,  ±40)
tau_knee = clip(tau_knee, ±40)
tau_wheel= clip(tau_wheel,±20)

# 闭链虚功映射（∂q_shank/∂(q_rear,q_front) = (d_rear,d_front)）
tau_rear_motor  = tau_hip + tau_knee * d_rear
tau_front_motor = tau_knee * d_front
# wheel 力矩直接为 tau_wheel

# 最终 6 电机力矩（顺序 [rear_L, front_L, wheel_L, rear_R, front_R, wheel_R]）
tau = clip(tau, ±40/±40/±20 按列)
# 执行器一阶滞后（训练用 15 ms；实机有自身闭环时可选）：
#   tau_lag += (tau - tau_lag) * min(dt / 0.015, 1)
```

- `(d_rear, d_front)`：`v14_leg_model.V14Leg.jacobian(q_front, q_rear)` 的返回值
  `(dq_shank/dq_rear, dq_shank/dq_front)`；随 robot 姿态变化，约在 ±0.75~1.7 之间。
- 时序：策略 **100 Hz**；物理/PD **200 Hz**（每策略周期 2 个子步，dt=0.005）。
  训练时 PD 与执行器滞后都按子步计算、动作有 1 个子步延迟；实机按自身控制周期实现即可。
- `kp=20`、`kd=1`、`wheel_kd=0.2` 是策略训练时使用的**部署口径**，不能随意改。

### 3.1 变高度功能

height 指令 `h_cmd ∈ [0.14, 0.21]` 就是「轮腿高度」控制量：修改它即可让策略蹲下/站起，
不需要新网络。**必须斜坡限速**（建议 0.10–0.15 m/s），阶跃指令会造成冲击。
参考实现见 `runtime/policy_runtime.py::HeightCommand`（目标 + 限速 + 裁剪）：

```python
hc = HeightCommand(initial=0.18, rate=0.12)
hc.set_target(0.21)                 # 升高
h_cmd = hc.step(dt=0.01)            # 每 10 ms 取一次，写进观测 commands[2]
```

实测（v1.3，站立）：0.14↔0.21 m 平均误差 6.0 mm、过渡 0.20–0.37 s；边走边变高平均误差 6.0 mm。
注意 0.14 m 为训练最低（实际 ~0.152 m），0.21 m 接近腿长上限（实际 ~0.207–0.211 m）。

## 4. 力矩限幅与安全

| 项 | 值 |
| --- | --- |
| 训练/验收力矩口径 | 前髋 ±40、后髋 ±40、轮 ±20 N·m |
| v14 原始 MJCF ctrlrange | ±3.14（**不够**，不要用于本策略） |
| 站立静力需求（实测） | 虚拟髋/膝 ≈9–10 N·m → 后髋电机 ≈17 N·m、前髋曲柄 ≈7 N·m |
| 动作裁剪建议 | `clip(action, -3, 3)`（与训练一致；训练用 action_clip=3） |

## 5. 硬件映射（来自同项目前序实车记录，部署前需复核）

| 关节 | 总线/ID | 备注 |
| --- | --- | --- |
| 左前髋（曲柄） | CAN1 DM1 | 模型 `Left_front_joint` |
| 左后髋（大腿） | CAN1 DM2 | 模型 `Left_rear_joint` |
| 右后髋（大腿） | CAN1 DM3 | 模型 `Right_rear_joint` |
| 右前髋（曲柄） | CAN1 DM4 | 模型 `Right_front_joint` |
| 左轮 | CAN2 C620 ID1（0x201） | 模型 `Left_wheel_joint` |
| 右轮 | CAN2 C620 ID2（0x202） | 模型 `Right_wheel_joint` |

- 轮速换算：`左 = −RPM×2π/(60×17)`，`右 = +RPM×2π/(60×17)`（17:1 减速箱）。
- 轮速符号：STATUS 前进口径 → 模型轮速 `[+1, −1]`；模型轮力矩 → C620 counts `[−1, −1]`。
- 力矩换算 3051 counts/N·m 是历史占位值，不是标定。
- IMU：body = `(−sensorY, +sensorX, +sensorZ)`。
- 关节零位/方向：见 `reports/TRAINING_REPORT.md` 与项目 `AGENTS.md`；
  **换零位会改变观测口径，必须重新标定并重放验证**。

## 6. 上板/移植建议顺序

1. PC 端 `runtime/verify_package.py` → PASS。
2. C/嵌入式实现观测 25 维 + 历史 125 → 用 `evidence/test_vectors.json` 对齐
   （5 组向量，误差 <1e-4 视为一致）。
3. 实现 MLP（125-256-256-256-6，ELU）→ 再对齐动作输出。
4. 实现动作解码 + 闭链力矩映射（公式见 §3），用 `runtime/policy_runtime.py` 的
   `decode_to_motor_torques` 做数值对照。
5. 台架（吊起/垫起）验证方向与限幅 → 落地低速验证 → 再放开。
6. 注意现有 H723 固件 `policy_v14.c` 是「encoder+actor 双输入」结构，**不能直接装
   本策略**；需按本文件生成新权重/改网络。

## 7. 数值测试向量格式（`evidence/test_vectors.json`）

```json
{
  "onnx": "policy/v14_rise_deploy.onnx",
  "onnx_sha256": "...",
  "layout": "frame-major oldest-first, 5x25 float32; reset_fill=current",
  "vectors": [
    {"description": "prone start",
     "obs_history_frame_major": [125 个 float],
     "expected_action": [6 个 float]}
  ]
}
```

两个 rise 策略（v1.1 宽指令 / v1.0 保守档）输入输出结构完全一致、可互换；各自的测试向量见
`evidence/test_vectors.json`（v1.0）与 `evidence/test_vectors_wide.json`（v1.1）。
`expected_action` 是 onnxruntime CPU 对该历史的输出；移植后用同一 125 维历史跑自己的
实现，`max|Δ| < 1e-4` 即认为前向数值一致（浮点实现差异通常 <1e-5）。
