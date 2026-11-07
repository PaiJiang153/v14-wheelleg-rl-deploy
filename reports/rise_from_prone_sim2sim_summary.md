# v14 sim2sim 评估汇总：`v14_rise`

- ONNX: `policy/v14_rise_deploy.onnx`
- ONNX sha256: `a120eb39b60e12c348695de3ba0749d9f923f8ef4b7b2f12a14c4c4d89a52035`
- 模型: `/home/paimiomio/rl/model/rl_chassis_v14/mjcf/rl_chassis_v14_scene.xml`（nq=21 nv=20 neq=4 nu=6，质量 14.189111 kg）
- 初始位姿: **prone**（home keyframe，jitter scale=0.0，seed=0）
- 时长: 10.0 s / 工况；执行器口径: train
- 生成时间: 2026-11-07T23:18:29+0800

## 1. 趴姿起身（--init prone，训练口径 train，hip 40 / wheel 20 N·m）

> rise 成功判据：基座高度 ≥ 指令 − 3 cm 且倾角 < 20°，持续 ≥ 0.3 s。判定以 `rise_success` 为主（PASS/FAIL），其余 checks 表示起身后的姿态/闭链/接触健康度。

> 倾角为整段统计（含趴姿→起身瞬态，故 max 偏大）；rise 成功后姿态见 `posture_ok`（final 高度/倾角）。

| 工况 | rise 成功 | rise 时间 (s) | max_height (m) | final_height (m) | 站姿保持占比 | 倾角均值/最大 (°) | 闭链 max (mm) | 非轮-地面 全程/起身后 (步) | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| stand_h018 | 是 | 0.50 | 0.2099 | 0.1903 | 0.95 | 1.32/18.63 | 0.7262 | 16/0 | PASS |
| fwd_0p5 | 是 | 0.50 | 0.1959 | 0.1888 | 0.95 | 1.18/17.30 | 0.7871 | 14/0 | PASS |
| yaw_0p3 | 是 | 0.50 | 0.2003 | 0.1903 | 0.95 | 1.21/18.41 | 0.8506 | 20/0 | PASS |
| height_0p20 | 是 | 0.52 | 0.2272 | 0.2018 | 0.95 | 1.32/20.00 | 0.6760 | 15/0 | PASS |

## 2. 判定口径

- `rise_success`：duration 内基座高度 ≥ 指令−3 cm 且倾角 < 20° 持续 ≥ 0.3 s；`rise_time_s` 为该持续段起始时刻（首次真正站住）。
- 判定：未站起 → `FAIL: did not rise`；站起后又倒 → `FAIL: fell after rise`；站起但闭链/接触等 check 失败 → `WARN`；全部通过 → `PASS`。
- 摔倒（prone）：只在起身成功之后再次出现 base 高度 < 0.08 m 或倾角 > 60° 才记 fall；初始趴姿本身不算摔倒。
- `stand_hold_fraction`：整个 run 中满足站姿判据（高度 ≥ 指令−3 cm 且倾角 < 20°）的采样占比。
- 闭链正常：4 条 connect 残差 max ≤ 5 mm 且前后半段平均漂移 ≤ 1 mm。
- 接触异常：非轮子 geom 接触地面的采样占比 > 1% 时 `contact_clean=false`。

