#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交付包自检：文件哈希、ONNX 数值测试向量、MuJoCo 模型结构。

用法：python runtime/verify_package.py
退出码 0 = 全部通过。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    ok = True
    print("== 1) 文件哈希 ==")
    sums = PKG / "SHA256SUMS.txt"
    for line in sums.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        f = PKG / rel
        if not f.is_file():
            print(f"  [FAIL] 缺文件: {rel}")
            ok = False
            continue
        got = sha256(f)
        status = "OK" if got == digest else "MISMATCH"
        if status != "OK":
            ok = False
        print(f"  [{status}] {rel}")

    print("== 2) ONNX 测试向量 ==")
    import onnxruntime as ort
    tv_files = [p for p in ("test_vectors.json", "test_vectors_wide.json",
                            "test_vectors_dr.json",
                            "test_vectors_anneal.json")
                if (PKG / "evidence" / p).is_file()]
    worst = 0.0
    for tv_name in tv_files:
        tv = json.loads((PKG / "evidence" / tv_name).read_text(encoding="utf-8"))
        session = ort.InferenceSession(str(PKG / tv["onnx"]),
                                       providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0].name
        print(f"  -- {tv['onnx']}")
        for i, vec in enumerate(tv["vectors"]):
            hist = np.asarray(vec["obs_history_frame_major"], dtype=np.float32)[None, :]
            got = session.run(None, {inp: hist})[0][0]
            exp = np.asarray(vec["expected_action"], dtype=np.float64)
            err = float(np.abs(got - exp).max())
            worst = max(worst, err)
            print(f"  [{'OK' if err < 1e-4 else 'FAIL'}] vector{i}: max|Δ|={err:.2e}")
            ok = ok and err < 1e-4
    print(f"  worst max|Δ| = {worst:.2e}（阈值 1e-4）")

    print("== 3) MuJoCo 模型结构 ==")
    import mujoco
    model = mujoco.MjModel.from_xml_path(
        str(PKG / "model/mjcf/rl_chassis_v14.xml"))
    mass = float(model.body_mass.sum())
    checks = {
        "nq": (int(model.nq), 21),
        "nv": (int(model.nv), 20),
        "neq(闭链约束)": (int(model.neq), 4),
        "nu(电机)": (int(model.nu), 6),
        "总质量 kg": (round(mass, 7), 14.1891111),
    }
    for name, (got, exp) in checks.items():
        good = (abs(got - exp) < 1e-6) if isinstance(exp, float) else (got == exp)
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] {name} = {got}（期望 {exp}）")

    print("\n结果:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
