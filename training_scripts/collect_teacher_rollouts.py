#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采集「教师策略」在完整 v14 闭链 MuJoCo 里的闭环轨迹（用于 BC 热启动）。

教师 = 旧 v14_flat 策略（在简化模型上训练，但在完整闭链模型上实测能完成
「趴卧→起身→保持→走/转」，见 sim2sim/out/v14_flat/ 与 v14_rise_smoke）。

记录每个策略步的：
  obs25   : 当前 25 维观测（部署口径，已缩放）
  hist_fm : 125 维 frame-major 历史（旧→新，与部署 spec 一致）
  act     : 教师输出动作（6 维，未裁剪）
训练我们的 mjlab 策略（term-major 历史）时，用 M @ hist_fm 转成 term-major。

用法：
  python sim2sim/collect_teacher_rollouts.py --teacher <v14_flat.onnx> \
      --out model/rl_chassis_v14/states/teacher_rollouts.npz --n 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RL_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import eval_v14 as ev  # noqa: E402

OUT_DEFAULT = RL_ROOT / "model/rl_chassis_v14/states/teacher_rollouts.npz"

# 训练/采集用的指令组合（起身后跟踪）
CASE_CMDS = [
    (0.0, 0.0, 0.18),    # 站定
    (0.4, 0.0, 0.18),    # 前进
    (0.0, 0.25, 0.18),   # 转向
    (0.0, 0.0, 0.20),    # 调高
    (0.3, 0.15, 0.16),   # 走+转+略低
]


class Recorder(ev.ClosedLoopEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_list: list[np.ndarray] = []
        self.hist_list: list[np.ndarray] = []
        self.act_list: list[np.ndarray] = []
        self._hist_buf: list[np.ndarray] = []

    def _policy(self, obs, hist):  # noqa: ANN001
        act = super()._policy(obs, hist)
        # 复算 frame-major 历史（首次用当前帧填满，之后旧→新）
        if not self._hist_buf:
            self._hist_buf = [np.asarray(obs, dtype=np.float32)] * 5
        else:
            self._hist_buf = self._hist_buf[1:] + [np.asarray(obs, dtype=np.float32)]
        self.obs_list.append(np.asarray(obs, dtype=np.float32))
        self.hist_list.append(np.concatenate(self._hist_buf, axis=0))
        self.act_list.append(np.asarray(act, dtype=np.float32))
        return act


def run_episode(teacher: str, cmd, init: str, n_episode: int,
                duration: float, seed: int) -> tuple:
    ref = ev.load_reference_engine(ev.DEFAULT_REFERENCE, ev.DEFAULT_LEG_TOOLS)
    args = ev.EngineArgs(
        xml=str(ev.DEFAULT_XML), onnx=str(Path(teacher).resolve()),
        actuator_profile="training", duration=float(duration),
        cmd_lin_vel=float(cmd[0]), cmd_yaw_vel=float(cmd[1]),
        cmd_height=float(cmd[2]), init=init, init_seed=int(seed),
        init_jitter=0.0,
    )
    rec = Recorder(args, ev.EngineArgs, ref, label=f"teacher{n_episode}")
    rec.evaluate()
    print(f"[collect] ep{n_episode}: init={init} cmd={cmd} steps={len(rec.obs_list)}")
    return (np.asarray(rec.obs_list), np.asarray(rec.hist_list),
            np.asarray(rec.act_list))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--n", type=int, default=12, help="episode 数")
    ap.add_argument("--duration", type=float, default=14.0)
    args = ap.parse_args()

    obs_all, hist_all, act_all = [], [], []
    for i in range(args.n):
        cmd = CASE_CMDS[i % len(CASE_CMDS)]
        init = "prone"            # 一半从趴卧起步（教师会先起身）
        if i % 3 == 2:
            init = "stand"
        o, h, a = run_episode(args.teacher, cmd, init, i, args.duration,
                              seed=i)
        obs_all.append(o); hist_all.append(h); act_all.append(a)
    obs = np.concatenate(obs_all, axis=0)
    hist = np.concatenate(hist_all, axis=0)
    act = np.concatenate(act_all, axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, obs=obs, hist_fm=hist, act=act)
    print(f"[collect] saved {obs.shape[0]} steps -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
