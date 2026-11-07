#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成「趴卧 → 被动 PD 抬升」过程中的姿态快照（起身课程用 reset 状态）。

动作全 0 时，默认站姿 PD 会把车从趴卧（base 0.034 m）抬到 ~0.12 m、前倾 ~25°，
然后撑不住回落。策略要学的「接手点」正是这条轨迹上的状态。离线把轨迹上的
qpos 快照存下来，训练时按概率从这些状态 reset（而不是从理想化的斜蹲姿 reset），
策略才能真正学会接力站稳。

输出：model/rl_chassis_v14/states/passive_rise_qpos.npz
  qpos: (K, 21) float64  —— 快照（base pos/quat + 14 关节）
  time: (K,)    float64  —— 对应仿真时间
  height: (K,)  float64  —— 快照时 base 高度
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
RL_ROOT = HERE.parents[3]          # .../rl
sys.path.insert(0, str(RL_ROOT / "sim2sim"))
import eval_v14 as ev  # noqa: E402

OUT = RL_ROOT / "model" / "rl_chassis_v14" / "states" / "passive_rise_qpos.npz"


class Recorder(ev.ClosedLoopEvaluator):
    def __init__(self, *args, record_every: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self._every = int(record_every)
        self._n = 0
        self.qpos: list[np.ndarray] = []
        self.times: list[float] = []
        self.heights: list[float] = []

    def _record_step(self, *args, **kwargs):  # noqa: D401
        super()._record_step(*args, **kwargs)
        self._n += 1
        if self._n % self._every:
            return
        d = self.sim.d
        self.qpos.append(np.array(d.qpos, dtype=np.float64))
        self.times.append(float(d.time))
        self.heights.append(float(d.qpos[2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--record-every", type=int, default=5)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    ref = ev.load_reference_engine(ev.DEFAULT_REFERENCE, ev.DEFAULT_LEG_TOOLS)
    engine_args = ev.EngineArgs(
        xml=str(ev.DEFAULT_XML), onnx=None, actuator_profile="training",
        duration=float(args.duration), cmd_lin_vel=0.0, cmd_yaw_vel=0.0,
        cmd_height=0.18, init="prone", zero_action=True,
    )
    recorder = Recorder(engine_args, ev.EngineArgs, ref, label="passive_rise",
                        record_every=args.record_every)
    recorder.evaluate()
    qpos = np.asarray(recorder.qpos)
    times = np.asarray(recorder.times)
    heights = np.asarray(recorder.heights)
    # 只保留真正在「抬升过程」中的快照（高度 0.05~0.135，避免回落/趴地）
    keep = (heights >= 0.05) & (heights <= 0.135)
    qpos, times, heights = qpos[keep], times[keep], heights[keep]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, qpos=qpos, time=times, height=heights)
    print(f"kept {len(qpos)} snapshots -> {args.out}")
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        i = min(len(qpos) - 1, int(frac * (len(qpos) - 1)))
        print(f"  t={times[i]:.2f}s z={heights[i]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
