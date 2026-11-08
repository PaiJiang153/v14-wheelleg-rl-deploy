#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v14 轮腿策略（完整闭链模型训练）部署参考运行时 —— 单文件、无本工作区依赖。

策略：policy/v14_rise_deploy.onnx（从趴卧姿起身 + 站/走/转/高度跟踪）
接口：单输入 obs (N,125) float32 -> 输出 actions (N,6) float32
      **125 = 25 维观测 × 5 帧历史，frame-major，帧旧→新（最新一帧在最后）**
动作：6 维 [左髋(虚拟), 左膝(虚拟), 左轮, 右髋, 右膝, 右轮]
      → 虚拟 PD → 闭链虚功映射 → 真机 6 个电机（后髋/前髋曲柄/轮，左右各 3）

本文件给出「从传感器/编码器到 6 个电机力矩」的完整参考实现，硬件移植时按需裁剪：
  1) `V14PolicyRuntime.assemble_obs(...)`  组单帧 25 维观测
  2) `ObsHistory.push(frame)`              维护 5 帧历史（reset 用当前帧填满）
  3) `V14PolicyRuntime.act(history)`       ONNX 推理（输入已裁剪 ±clip_actions=100）
  4) `V14PolicyRuntime.decode_to_motor_torques(...)`
       虚拟 PD（kp=20/kd=1，轮 kd=0.2）→ 闭链雅可比虚功映射 → (后髋, 前髋, 轮) 力矩
  5) 执行器一阶滞后 15 ms（实机已有自己的滞后/滤波时可省略）

硬件侧只需要：
  * 编码器角 q_front / q_rear（弧度，模型正方向）与角速度；
  * 用 `v14_leg_model.V14Leg.solve_loop(q_front,q_rear)` 求被动关节（得到 q_shank）；
  * IMU 陀螺/重力；轮速；上一步策略输出。

安全提示：
  * 本策略训练/验证的力矩口径为 后髋/前髋 ±40 N·m、轮 ±20 N·m；**不要**在
    ±3.14 N·m 下运行（静力需 ~17 N·m，任何策略都会塌）。
  * 部署 MJCF `ctrlrange` 与固件输出限幅要同步改成 40/40/20 一档。
  * 动作在解码前硬裁到 ±3（训练时 action_clip=3）；力矩另有 ±40/±20 限幅。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

# --- 与训练配置逐项一致的常量 ------------------------------------------------
HISTORY_LEN = 5
OBS_FRAME_DIM = 25
OBS_FLAT_DIM = OBS_FRAME_DIM * HISTORY_LEN

# 虚拟站姿默认角（(rear, shank)，与前链闭链解对应的站姿 0.18 m）
DEFAULT_VIRTUAL = {"left": (-0.865168, -0.774701), "right": (0.865169, 0.774700)}

KP = 20.0          # 虚拟腿 PD 位置增益（训练/部署一致）
KD = 1.0           # 虚拟腿 PD 速度增益
WHEEL_KD = 0.2     # 轮速增益
POS_ACTION_SCALE = 0.5
VEL_ACTION_SCALE = 10.0
ACTION_CLIP = 3.0                 # 训练 action_clip；解码前硬裁
TAU_LIMIT = (40.0, 40.0, 20.0)    # (后髋, 前髋曲柄, 轮) N·m，训练/部署口径
ACTUATOR_TAU = 0.015              # 一阶执行器滞后 s（实机可省略/替换）
CLIP_OBS = 100.0
CMD_RANGES = {"lin_vel": (-1.2, 1.2), "yaw_vel": (-1.0, 1.0),
              "height": (0.14, 0.20)}
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_CMD = (2.0, 0.25, 5.0)
OBS_SCALE_DOF_VEL = 0.05
DEFAULT_CMD = (0.0, 0.0, 0.18)

# 关节顺序（模型口径；实现时按名字对齐）
LEG_JOINT_ORDER = ("front", "child1", "child2", "child3", "rear", "shank")


def _leg_model():
    """延迟导入 v14_leg_model（它需要 mujoco 才能从 MJCF 推导几何）。"""
    import v14_leg_model
    return v14_leg_model


class ObsHistory:
    """5 帧 × 25 维历史，frame-major、帧旧→新；reset 时用当前帧填满。"""

    def __init__(self):
        self._frames = None

    def reset(self, current_frame: np.ndarray | None = None):
        self._frames = None
        if current_frame is not None:
            self.push(np.asarray(current_frame, dtype=np.float64))

    def push(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame, dtype=np.float64).reshape(OBS_FRAME_DIM)
        if self._frames is None:
            self._frames = [frame.copy() for _ in range(HISTORY_LEN)]
        else:
            self._frames = self._frames[1:] + [frame.copy()]

    def flatten(self) -> np.ndarray:
        if self._frames is None:
            raise RuntimeError("ObsHistory 还没有数据：先 push 一帧")
        return np.concatenate(self._frames, axis=0).astype(np.float32)


class V14PolicyRuntime:
    """策略推理 + 观测组装 + 动作解码 + 闭链力矩映射。"""

    def __init__(self, onnx_path: str | Path | None = None,
                 scene_xml: str | Path | None = None):
        import onnxruntime as ort

        onnx_path = Path(onnx_path) if onnx_path else (
            PKG_ROOT / "policy" / "v14_rise_deploy.onnx")
        scene_xml = Path(scene_xml) if scene_xml else (
            PKG_ROOT / "model" / "mjcf" / "rl_chassis_v14_scene.xml")
        self.onnx_path = onnx_path
        self.scene_xml = scene_xml
        self.session = ort.InferenceSession(str(onnx_path),
                                            providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape[-1] != OBS_FLAT_DIM:
            raise ValueError(
                f"ONNX 输入应为 (batch,{OBS_FLAT_DIM})，实际 {[i.shape for i in inputs]}")
        self.input_name = inputs[0].name

        _m = _leg_model()
        self.model, self.leg_left, self.leg_right = _m.load(str(scene_xml))
        self.legs = {"left": self.leg_left, "right": self.leg_right}
        self._tau_lag = np.zeros(6, dtype=np.float64)
        self.last_action = np.zeros(6, dtype=np.float64)

    # ------------------------------------------------------------------ 观测
    @staticmethod
    def assemble_obs(gyro_body, gravity_body, cmd, q_virtual, qd6, last_action):
        """组单帧 25 维观测（顺序必须与训练一致，勿改）。

        gyro_body      : (3,) 机体坐标系角速度 [rad/s]
        gravity_body   : (3,) 机体坐标系投影重力，直立 ≈ (0,0,-1)
        cmd            : (3,) [vx_cmd, wz_cmd, height_cmd]（m/s, rad/s, m）
        q_virtual      : (4,) [q_rear_L, q_shank_L, q_rear_R, q_shank_R]（模型弧度）
        qd6            : (6,) [qd_rear_L, qd_shank_L, qd_wheel_L,
                               qd_rear_R, qd_shank_R, qd_wheel_R]
        last_action    : (6,) 上一策略步原始输出（不是 delayed/applied）
        """
        gyro = np.asarray(gyro_body, dtype=np.float64).reshape(3) * OBS_SCALE_ANG_VEL
        grav = np.asarray(gravity_body, dtype=np.float64).reshape(3)
        c = np.asarray(cmd, dtype=np.float64).reshape(3).copy()
        c[0] = float(np.clip(c[0], *CMD_RANGES["lin_vel"]))
        c[1] = float(np.clip(c[1], *CMD_RANGES["yaw_vel"]))
        c[2] = float(np.clip(c[2], *CMD_RANGES["height"]))
        cmd_scaled = c * np.asarray(OBS_SCALE_CMD)
        q = np.asarray(q_virtual, dtype=np.float64).reshape(4)
        default = np.array([DEFAULT_VIRTUAL["left"][0], DEFAULT_VIRTUAL["left"][1],
                            DEFAULT_VIRTUAL["right"][0], DEFAULT_VIRTUAL["right"][1]])
        hip_knee = q - default
        vel = np.asarray(qd6, dtype=np.float64).reshape(6) * OBS_SCALE_DOF_VEL
        act = np.asarray(last_action, dtype=np.float64).reshape(6)
        frame = np.concatenate([gyro, grav, cmd_scaled, hip_knee, vel, act], axis=0)
        assert frame.shape == (OBS_FRAME_DIM,), frame.shape
        return np.clip(frame, -CLIP_OBS, CLIP_OBS)

    # ------------------------------------------------------------------ 推理
    def act(self, history_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """输入 125 维历史；返回 (raw_action, clipped_action)，各 (6,)。"""
        hist = np.asarray(history_flat, dtype=np.float32).reshape(1, OBS_FLAT_DIM)
        out = self.session.run(None, {self.input_name: hist})[0]
        raw = np.asarray(out, dtype=np.float64).reshape(6)
        clipped = np.clip(raw, -ACTION_CLIP, ACTION_CLIP)
        self.last_action = raw.copy()
        return raw, clipped

    # -------------------------------------------------------------- 动作解码
    def decode_to_motor_torques(self, action, q_by_side, qd_by_side,
                                jac_by_side, dt: float = 0.01,
                                use_lag: bool = True,
                                clip_action: bool = True) -> np.ndarray:
        """动作 → 真机 6 电机力矩 [rear_L, front_L, wheel_L, rear_R, front_R, wheel_R]。

        q_by_side  : {"left": (q_front,q_rear,q_shank), "right": (...)} 模型弧度
                     （q_shank 可以是仿真直接量到的，也可以由 solve_shank() 逆解）
        qd_by_side : {"left": (qd_rear,qd_shank,qd_wheel), ...}
        jac_by_side: {"left": (d_rear,d_front), ...} 闭链雅可比 ∂q_shank/∂(q_rear,q_front)
                     可直接用 v14_leg_model.V14Leg.jacobian(q_front,q_rear) 得到。
        """
        a = np.asarray(action, dtype=np.float64).reshape(6)
        if clip_action:
            a = np.clip(a, -ACTION_CLIP, ACTION_CLIP)
        tau = np.zeros(6, dtype=np.float64)
        for i, side in enumerate(("left", "right")):
            q_rear = float(q_by_side[side][1])
            q_shank = float(q_by_side[side][2])
            qd_rear, qd_shank, qd_wheel = [float(x) for x in qd_by_side[side]]
            d_rear, d_front = jac_by_side[side]

            target_hip = DEFAULT_VIRTUAL[side][0] + POS_ACTION_SCALE * a[3 * i]
            target_knee = DEFAULT_VIRTUAL[side][1] + POS_ACTION_SCALE * a[3 * i + 1]
            tau_hip = KP * (target_hip - q_rear) - KD * qd_rear
            tau_knee = KP * (target_knee - q_shank) - KD * qd_shank
            tau_hip = float(np.clip(tau_hip, -TAU_LIMIT[0], TAU_LIMIT[0]))
            tau_knee = float(np.clip(tau_knee, -TAU_LIMIT[1], TAU_LIMIT[1]))
            tau_wheel = WHEEL_KD * (VEL_ACTION_SCALE * a[3 * i + 2] - qd_wheel)
            tau_wheel = float(np.clip(tau_wheel, -TAU_LIMIT[2], TAU_LIMIT[2]))

            # 虚功原理：虚拟 (hip, knee) 力矩 -> 真机 (后髋, 前髋曲柄)
            tau[3 * i] = tau_hip + tau_knee * d_rear
            tau[3 * i + 1] = tau_knee * d_front
            tau[3 * i + 2] = tau_wheel
        tau = np.clip(tau, [-TAU_LIMIT[0], -TAU_LIMIT[1], -TAU_LIMIT[2],
                            -TAU_LIMIT[0], -TAU_LIMIT[1], -TAU_LIMIT[2]],
                      [TAU_LIMIT[0], TAU_LIMIT[1], TAU_LIMIT[2],
                       TAU_LIMIT[0], TAU_LIMIT[1], TAU_LIMIT[2]])
        if use_lag:
            alpha = min(1.0, max(dt, 1e-6) / ACTUATOR_TAU)
            self._tau_lag += (tau - self._tau_lag) * alpha
            tau = self._tau_lag.copy()
        return tau

    # ------------------------------------------------------- 闭链状态辅助函数
    def solve_shank(self, side: str, q_front: float, q_rear: float) -> float:
        """由测量到的 (曲柄, 大腿) 求被动关节（含 shank）——部署侧 q_shank 来源。"""
        q, residual, _ = self.legs[side].solve_loop(q_front, q_rear)
        if residual > 1e-6:
            raise RuntimeError(f"{side}: 闭链逆解残差 {residual:.2e} 过大")
        return float(q["shank"])

    def virtual_state(self, side: str, q_front: float, q_rear: float,
                      qd_front: float, qd_rear: float) -> tuple[float, float]:
        """返回 (q_shank, qd_shank)：qd_shank 由虚拟腿雅可比得到。"""
        q, residual, _ = self.legs[side].solve_loop(q_front, q_rear)
        if residual > 1e-6:
            raise RuntimeError(f"{side}: 闭链逆解残差 {residual:.2e} 过大")
        d_rear, d_front = self.legs[side].jacobian(q_front, q_rear)
        qd_shank = d_rear * qd_rear + d_front * qd_front
        return float(q["shank"]), float(qd_shank)

    def jacobians(self, q_by_side) -> dict:
        return {side: self.legs[side].jacobian(*q_by_side[side])
                for side in ("left", "right")}


def default_stand_history(runtime: V14PolicyRuntime) -> ObsHistory:
    """构造「站立、零指令」的初始历史（仿真/自检用）。"""
    q = np.array([DEFAULT_VIRTUAL["left"][0], DEFAULT_VIRTUAL["left"][1],
                  DEFAULT_VIRTUAL["right"][0], DEFAULT_VIRTUAL["right"][1]])
    frame = runtime.assemble_obs(
        gyro_body=[0, 0, 0], gravity_body=[0, 0, -1], cmd=DEFAULT_CMD,
        q_virtual=q, qd6=np.zeros(6), last_action=np.zeros(6))
    hist = ObsHistory()
    hist.reset(frame)
    return hist


class HeightCommand:
    """实机/运行时「变高度」指令：目标高度 + 斜坡限速 + 范围裁剪。

    策略本身已把 height 指令做进观测（25 维第 8 维），所以变高度不需要新网络，
    只需要按部署节奏平滑修改指令（阶跃会造成冲击）。建议上限 0.10~0.15 m/s。

        hc = HeightCommand(initial=0.18, rate=0.12)
        hc.set_target(0.21)
        while True:                       # 每 10 ms 调一次
            cmd = [vx, wz, hc.step(dt=0.01)]
            ...
    """

    def __init__(self, initial: float = 0.18, rate: float = 0.12,
                 lo: float = 0.14, hi: float = 0.21):
        self.lo = float(lo)
        self.hi = float(hi)
        self.rate = float(rate)
        self.height = float(np.clip(initial, self.lo, self.hi))
        self.target = self.height

    def set_target(self, target: float) -> None:
        self.target = float(np.clip(target, self.lo, self.hi))

    def step(self, dt: float = 0.01) -> float:
        max_step = self.rate * max(float(dt), 1e-6)
        delta = float(np.clip(self.target - self.height, -max_step, max_step))
        self.height = float(np.clip(self.height + delta, self.lo, self.hi))
        return self.height


if __name__ == "__main__":
    rt = V14PolicyRuntime()
    h = default_stand_history(rt)
    action = rt.act(h.flatten())[1]
    print("onnx:", rt.onnx_path)
    print("stand action:", np.round(action, 4).tolist())
    print("OK")
