# 给接手 AI 的任务说明（可直接复制给下一个 AI）

你现在收到的是一个**完整的 v14 轮腿强化学习策略部署交接包**。请按顺序做：

1. **先读** `README_ZH.md`（总览、自检、部署要点）和 `INTERFACE.md`（精确接口合同）。
2. **先自检**（需要 numpy/mujoco/onnxruntime）：
   ```bash
   python runtime/verify_package.py
   python runtime/sim2sim_replay.py --onnx policy/v14_rise_deploy.onnx \
       --no-viewer --duration 10 --init prone
   ```
   期望：verify 输出 PASS；回放从趴卧起身，末态高度 ≈0.19 m、倾斜 ≈1°、只有两轮接地。
   如果复现不了，先检查 Python 依赖与 ONNX sha256（`SHA256SUMS.txt`），不要改策略。
3. **明确目标再动手**：主交付策略 `policy/v14_rise_deploy.onnx` 的能力是
   「趴卧姿起身 → 平衡 → 站/走/转/高度跟踪」。它**不是**旧 v14 的双输入
   encoder+actor 结构，而是单输入 `125→256³→6`（ELU）MLP，输入是 **frame-major
   5×25 历史**。不要把它和旧 `policy_v14.c` 的 weights 混用。
4. **移植/部署前必须知道**：
   - 力矩口径 **40/40/20 N·m**；±3.14 口径下站不住（静力需后髋约 17 N·m）。
   - 观测 25 维的顺序/缩放、动作解码、闭链雅可比映射，逐项按 `INTERFACE.md` §2/§3
     实现，不要自创口径。
   - 右腿被动关节与轮关节是 v14 的新命名（`Right_front_child1/2/3_joint`、
     `Left_wheel_joint`/`Right_wheel_joint`），与 v12 不同。
   - C 实现对齐用 `evidence/test_vectors.json`（5 组历史→动作，误差 <1e-4）。
5. **不要做的事**：不要假设实车已标定（质量/质心/轮径/力矩换算都是参考值）；
   不要在未验证方向/限幅前直接给大扭矩；不要把训练报告里 ±3.14 口径的结论当实车可用。
6. 需要更深的背景（训练方法、失败的 3 次尝试、诊断、复现命令）时读
   `reports/TRAINING_REPORT.md`；策略来源/哈希/训练配置见 `evidence/provenance.json`。

交付物清单见 `README_ZH.md` §7；所有文件哈希在 `SHA256SUMS.txt`。
