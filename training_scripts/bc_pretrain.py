#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""行为克隆（BC）热启动：把教师轨迹蒸馏进 mjlab 策略网络。

输入：sim2sim/collect_teacher_rollouts.py 生成的 npz
  obs(25) / hist_fm(125, frame-major) / act(6)
网络：125（term-major 历史）→256³ ELU→6，与训练/导出同构。
初始权重 = 平地策略 checkpoint（domain warm start），在教师数据上监督微调。

产物：rsl-rl 兼容 checkpoint（actor 换成 BC 权重，critic/optimizer 沿用基座），
可直接 `train.py --task rise --resume --load-run warmstart_bc`。

用法：
  .venv/bin/python scripts/bc_pretrain.py \
      --data ../../model/rl_chassis_v14/states/teacher_rollouts.npz \
      --base-checkpoint logs/rsl_rl/wheelleg_v14_closed/2026-11-07_18-08-19_formal_1024x3000/model_2999.pt \
      --std 0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

HISTORY = 5
TERM_DIMS = [3, 3, 3, 4, 6, 6]
OBS_FRAME = sum(TERM_DIMS)
OBS_FLAT = OBS_FRAME * HISTORY
TERM_OFFSETS = [sum(TERM_DIMS[:i]) for i in range(len(TERM_DIMS))]


def frame_to_term_matrix() -> torch.Tensor:
  m = torch.zeros(OBS_FLAT, OBS_FLAT)
  for f in range(HISTORY):
    for t, d in enumerate(TERM_DIMS):
      for k in range(d):
        fm = f * OBS_FRAME + TERM_OFFSETS[t] + k
        tm = TERM_OFFSETS[t] * HISTORY + f * d + k
        m[tm, fm] = 1.0
  return m


class PolicyMLP(nn.Module):
  def __init__(self, num_obs: int = OBS_FLAT, num_actions: int = 6,
               hidden: tuple[int, ...] = (256, 256, 256)):
    super().__init__()
    dims = [num_obs, *hidden, num_actions]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
      layers.append(nn.Linear(dims[i], dims[i + 1]))
      if i < len(dims) - 2:
        layers.append(nn.ELU())
    self.mlp = nn.Sequential(*layers)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.mlp(obs)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", type=Path, required=True)
  ap.add_argument("--base-checkpoint", type=Path, required=True)
  ap.add_argument("--out-dir", type=Path,
                  default=Path("logs/rsl_rl/wheelleg_v14_rise/warmstart_bc"))
  ap.add_argument("--epochs", type=int, default=80)
  ap.add_argument("--batch", type=int, default=512)
  ap.add_argument("--lr", type=float, default=1e-3)
  ap.add_argument("--std", type=float, default=0.3)
  ap.add_argument("--device", default="cuda:0")
  args = ap.parse_args()

  data = np.load(str(args.data))
  obs25 = data["obs"].astype(np.float32)
  hist_fm = data["hist_fm"].astype(np.float32)
  act = data["act"].astype(np.float32)
  m = frame_to_term_matrix().numpy()
  hist_tm = hist_fm @ m.T
  print(f"[bc] data: {obs25.shape[0]} steps, hist_fm {hist_fm.shape}")

  dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
  X = torch.from_numpy(hist_tm).to(dev)
  Y = torch.from_numpy(act).to(dev)

  policy = PolicyMLP().to(dev)
  ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
  actor = ckpt["actor_state_dict"]
  with torch.no_grad():
    for i, mod in enumerate(policy.mlp):
      if isinstance(mod, nn.Linear):
        mod.weight.copy_(actor[f"mlp.{i}.weight"].float())
        mod.bias.copy_(actor[f"mlp.{i}.bias"].float())
  print("[bc] actor 初值 = 平地策略 checkpoint")

  opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
  loss_fn = nn.MSELoss()
  n = X.shape[0]
  for epoch in range(args.epochs):
    perm = torch.randperm(n, device=dev)
    total = 0.0
    for i in range(0, n, args.batch):
      idx = perm[i:i + args.batch]
      pred = policy(X[idx])
      loss = loss_fn(pred, Y[idx])
      opt.zero_grad()
      loss.backward()
      opt.step()
      total += float(loss) * len(idx)
    if (epoch + 1) % 10 == 0 or epoch == 0:
      print(f"[bc] epoch {epoch + 1}/{args.epochs} mse={total / n:.6f}")

  with torch.no_grad():
    pred = policy(X)
    mse = float(((pred - Y) ** 2).mean())
    mae = float((pred - Y).abs().mean())
  print(f"[bc] final mse={mse:.6f} mae={mae:.4f}")

  # 写回 rsl-rl checkpoint（保留 critic / optimizer；std 按 --std 抬高便于 PPO 探索）
  state = {}
  for i, mod in enumerate(policy.mlp):
    if isinstance(mod, nn.Linear):
      state[f"mlp.{i}.weight"] = mod.weight.detach().cpu()
      state[f"mlp.{i}.bias"] = mod.bias.detach().cpu()
  state["distribution.std_param"] = torch.full((6,), float(args.std))
  ckpt["actor_state_dict"] = state
  ckpt["iter"] = 2999
  args.out_dir.mkdir(parents=True, exist_ok=True)
  out = args.out_dir / "model_2999.pt"
  torch.save(ckpt, out)
  print(f"[bc] saved checkpoint -> {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
