#!/usr/bin/env bash
# 一键自检：文件哈希 + ONNX 测试向量 + 模型结构 + 从趴卧起身回放 + 站姿前进回放
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=${PYTHON:-python3}

echo "================ 1) package verify ================"
$PY runtime/verify_package.py

echo "================ 2) rise from prone (10 s) ========"
$PY runtime/sim2sim_replay.py --onnx policy/v14_rise_deploy.onnx \
    --no-viewer --duration 10 --init prone | grep -E "== 结束|高度 均值|前进速度|倾斜" | tail -5

echo "================ 3) walk from stand (8 s) ========="
$PY runtime/sim2sim_replay.py --onnx policy/v14_rise_deploy.onnx \
    --no-viewer --duration 8 --init stand --cmd-lin-vel 0.4 | grep -E "== 结束|高度 均值|前进速度|倾斜" | tail -5

echo "================ ALL DONE ================"
