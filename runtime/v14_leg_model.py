#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v14 单腿闭链运动学模型（主机离线参考实现）。

模型来源（唯一几何真值，不手抄几何）：
  model_updates/KaiWuRoboMaster-wxk-s-branch/rl_chassis_v14/mjcf/rl_chassis_v14_scene.xml
  robot XML sha256 = 5412ae3e...（见 v14_deploy/README.md 与 validation/v14_kinematics_check.json）

v14 与旧 v12 的差异（本模块不继承旧参数）：
  * 主动髋关节零姿态改变：q=0 时短杆 φ≈-34.52°、长杆 φ≈-118.12°（v12 约 +90°/-90.13°）；
  * 右前被动关节改名 Right_front_child2_joint / Right_front_child3_joint，
    轮关节改名 Left_wheel_joint / Right_wheel_joint；
  * 左 dφ/dq=-1、右 dφ/dq=+1（与 v12 相同，但零姿态不同）；
  * 短/长杆投影长度约 0.100 / 0.215 m（与 v12 接近，但连接位置/零位不同）。

每条腿两条链 + 两条闭合销：
  后链（串联）: base -> rear_Link(大腿) -> rear_child1(小腿/shank) -> wheel
  前链（曲柄）: base -> front_Link(曲柄) -> child1 -> child2 -> child3
  pin1: front_child2 <-> rear_Link
  pin2: front_child3 <-> rear_child1

坐标约定：机体系 x 前 / y 左 / z 上；φ = atan2(link_x, -link_z)，竖直向下=0、
向车头为正（与 calibration/model_leg_reference.json 一致）。

本模块只做运动学，不接触硬件。旧 v12 的 encoder offset / 被动分支种子 /
几何生成表没有被复用。
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

try:  # MuJoCo 只在主机使用；C 侧参数由另一个生成器导出
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise SystemExit('v14_leg_model requires mujoco (use .venv-policy/bin/python)') from exc

# 本文件随策略包交付：模型在包内 model/mjcf/ 下（与交付模型逐字节一致，sha256 自检）
PKG_ROOT = Path(__file__).resolve().parents[1]
WS = PKG_ROOT
MODEL_DIR = PKG_ROOT / 'model' / 'mjcf'
SCENE = MODEL_DIR / 'rl_chassis_v14_scene.xml'
ROBOT_XML = MODEL_DIR / 'rl_chassis_v14.xml'
SCENE_SHA256 = '4c8ff317487a88baa2b945e9756559e590248ccfa8476ab6244f44d4e24bfbd4'
ROBOT_SHA256 = '5412ae3e9bf9bfdc857650bdc58d0229b15fbdc279166a524d6bbc4296e13b8c'
EXPECTED_TOTAL_MASS_KG = 14.1891110559
EXPECTED_NU = 6

JOINT_ORDER = ('front', 'child1', 'child2', 'child3', 'rear', 'shank')
FREE_ORDER = ('child1', 'child2', 'child3', 'shank')

# CAN1 位置 -> (side, role)；AGENTS.md 已确认，不是本模块重新推断的。
CAN_SIDE_ROLE = {
    1: ('left', 'front'),
    2: ('left', 'rear'),
    3: ('right', 'rear'),
    4: ('right', 'front'),
}


def wrap_pi(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


class V14Leg:
    """一条 v14 腿的闭链运动学。side 取 'left' / 'right'。"""

    _NAMES = {
        'left': dict(
            front='Left_front_joint',
            child1='Left_front_child1_joint',
            child2='Left_front_child2_joint',
            child3='Left_front_child3_joint',
            rear='Left_rear_joint',
            shank='Left_rear_child1_joint',
            wheel='Left_wheel_joint',
            pin1_a='pin1_A',
            pin1_b='pin1_B',
            pin2_a='pin2_A',
            pin2_b='pin2_B',
            wheel_body='Left_wheel_link',
        ),
        'right': dict(
            front='Right_front_joint',
            child1='Right_front_child1_joint',
            child2='Right_front_child2_joint',
            child3='Right_front_child3_joint',
            rear='Right_rear_joint',
            shank='Right_rear_child1_joint',
            wheel='Right_wheel_joint',
            pin1_a='pin3_A',
            pin1_b='pin3_B',
            pin2_a='pin4_A',
            pin2_b='pin4_B',
            wheel_body='Right_wheel_Link',
        ),
    }

    def __init__(self, model, side, base_body='base_link'):
        self.m = model
        self.side = side
        self._data = mujoco.MjData(model)  # 复用，避免每次分配
        names = self._NAMES[side]
        self.names = dict(names)
        self.ids = {
            key: self._jid(names[key])
            for key in ('front', 'child1', 'child2', 'child3', 'rear', 'shank', 'wheel')
        }
        self.qadr = {key: int(model.jnt_qposadr[jid]) for key, jid in self.ids.items()}
        self.dofadr = {key: int(model.jnt_dofadr[jid]) for key, jid in self.ids.items()}
        self.site_ids = {key: self._sid(names[key])
                         for key in ('pin1_a', 'pin1_b', 'pin2_a', 'pin2_b')}
        self.wheel_body_id = int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, names['wheel_body']))
        self.base_body_id = int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, base_body))
        # 真机每腿 3 个电机 = front/rear/wheel；child1..3 + shank 是闭链被动关节。
        self.active = ('front', 'rear', 'wheel')
        self.free = FREE_ORDER
        self._wheel_geometry()
        self._serial_constants()
        self._driven_constants()

    # ---------------- 基础 ----------------
    def _jid(self, name):
        jid = int(mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name))
        if jid < 0:
            raise KeyError('%s: joint %r not found in v14 model' % (self.side, name))
        return jid

    def _sid(self, name):
        sid = int(mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, name))
        if sid < 0:
            raise KeyError('%s: site %r not found in v14 model' % (self.side, name))
        return sid

    def _wheel_geometry(self):
        for geom in range(self.m.ngeom):
            if int(self.m.geom_bodyid[geom]) != self.wheel_body_id:
                continue
            if int(self.m.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                self.wheel_radius = float(self.m.geom_size[geom][0])
                self.wheel_half_width = float(self.m.geom_size[geom][1])
                return
        raise RuntimeError('%s: wheel cylinder geom not found' % self.side)

    def _fk(self, **q):
        """把关节角写进 qpos 后做纯运动学，返回 hip/knee/wheel 位置（世界系）。"""
        data = self._data
        for key, value in q.items():
            data.qpos[self.qadr[key]] = float(value)
        mujoco.mj_kinematics(self.m, data)
        hip = data.xpos[int(self.m.jnt_bodyid[self.ids['rear']])].copy()
        knee = data.xpos[int(self.m.jnt_bodyid[self.ids['shank']])].copy()
        wheel = data.xpos[self.wheel_body_id].copy()
        return hip, knee, wheel

    def base_rotation(self):
        return self._data.xmat[self.base_body_id].reshape(3, 3)

    def wheel_relative(self, hip, wheel):
        """轮心相对髋位置，base 系。"""
        rotation = self.base_rotation()
        return rotation.T @ (wheel - hip)

    # ---------------- 后链等效串联参数 ----------------
    def _serial_constants(self):
        hip0, knee0, wheel0 = self._fk(rear=0.0, shank=0.0)

        def angle(u, v):
            return float(np.arctan2(v[2] - u[2], v[0] - u[0]))

        self.l1 = float(np.hypot(knee0[0] - hip0[0], knee0[2] - hip0[2]))
        self.l2 = float(np.hypot(wheel0[0] - knee0[0], wheel0[2] - knee0[2]))
        self.a1 = angle(hip0, knee0)
        self.a2 = angle(knee0, wheel0) - self.a1
        self.hip_base = hip0.copy()
        hip, knee, _ = self._fk(rear=0.1, shank=0.0)
        self.sgn_rear = float(np.sign(wrap_pi(angle(hip, knee) - self.a1) / 0.1))
        hip, knee, wheel = self._fk(rear=0.0, shank=0.1)
        self.sgn_shank = float(np.sign(
            wrap_pi(angle(knee, wheel) - angle(knee0, wheel0)) / 0.1))

    # ---------------- 驱动杆 φ(q) ----------------
    def _driven_phi(self, slot, q):
        """给定驱动关节角，返回被驱动杆的 φ（base 系），以及投影长度。"""
        joint_name = self.names[slot]
        endpoint_name = (self.names['child1'] if slot == 'front'
                         else self.names['shank'])
        joint = self._jid(joint_name)
        endpoint = self._jid(endpoint_name)
        data = self._data
        data.qpos[:] = self.m.qpos0
        data.qpos[self.qadr[slot]] = float(q)
        mujoco.mj_kinematics(self.m, data)
        rotation = self.base_rotation()
        vector = rotation.T @ (data.xanchor[endpoint] - data.xanchor[joint])
        phi = math.atan2(vector[0], -vector[2])
        return phi, float(np.hypot(vector[0], vector[2]))

    def _driven_constants(self):
        self.driven = {}
        for slot in ('front', 'rear'):
            phi0, length = self._driven_phi(slot, 0.0)
            shifted, _ = self._driven_phi(slot, 0.001)
            slope = wrap_pi(shifted - phi0) / 0.001
            sign = int(round(slope))
            if sign not in (-1, 1) or abs(slope - sign) > 1e-5:
                raise RuntimeError('%s: unexpected dphi/dq for %s: %r'
                                   % (self.side, slot, slope))
            self.driven[slot] = {
                'phi_at_model_q0_rad': float(phi0),
                'phi_at_model_q0_deg': math.degrees(phi0),
                'dphi_dq': sign,
                'projected_link_length_m': length,
            }
            # φ(q) 必须严格线性（单转动副），否则模型/命名有误
            for q in (-0.2, 0.2):
                actual, _ = self._driven_phi(slot, q)
                if abs(wrap_pi(actual - (phi0 + sign * q))) > 1e-5:
                    raise RuntimeError('%s: phi(q) not linear for %s' % (self.side, slot))

    def driven_phi(self, slot, q):
        """φ(q) = φ0 + dφdq*q（线性，atan2 展开到 ±π 之外时用 wrap_pi 比较）。"""
        item = self.driven[slot]
        return item['phi_at_model_q0_rad'] + item['dphi_dq'] * float(q)

    def model_q_from_phi(self, slot, phi_measured_rad):
        """由记录/实测的被驱动杆角 φ 反解模型关节角 q_model。"""
        item = self.driven[slot]
        return wrap_pi((float(phi_measured_rad) - item['phi_at_model_q0_rad'])
                       / item['dphi_dq'])

    # ---------------- 虚拟腿 L0 / theta0 ----------------
    def virtual_leg(self, q_rear, q_shank, offset=0.0, convention='down'):
        """虚拟腿：L0=髋到轮心距离，theta0=腿倾角。

        等价于 theta1 = a1 + sgn_rear*q_rear, theta2 = a2 + sgn_shank*q_shank，
        end = offset + l1*e^{i theta1} + l2*e^{i(theta1+theta2)}。
        convention='down'（本仓库采用）: theta0 = atan2(ex, -ey)，竖直向下=0、轮在前为正。
        """
        t1 = self.a1 + self.sgn_rear * float(q_rear)
        t2 = self.a2 + self.sgn_shank * float(q_shank)
        ex = offset + self.l1 * np.cos(t1) + self.l2 * np.cos(t1 + t2)
        ey = self.l1 * np.sin(t1) + self.l2 * np.sin(t1 + t2)
        if convention == 'fudan':
            theta = float(np.arctan2(ey, ex) - np.pi / 2)
        else:
            theta = float(np.arctan2(ex, -ey))
        return float(np.hypot(ex, ey)), theta

    def ik_virtual(self, L0, theta0, hint=(0.0, 0.0)):
        """(L0, theta0) -> (q_rear, q_shank)，两个膝弯分支取靠近 hint 的。"""
        ex, ey = L0 * np.sin(theta0), -L0 * np.cos(theta0)
        c2 = (L0 * L0 - self.l1 ** 2 - self.l2 ** 2) / (2.0 * self.l1 * self.l2)
        if abs(c2) > 1.0:
            raise ValueError('L0=%.4f outside reachable [%.4f, %.4f]'
                             % (L0, abs(self.l1 - self.l2), self.l1 + self.l2))
        best = None
        for t2 in (np.arccos(c2), -np.arccos(c2)):
            t1 = np.arctan2(ey, ex) - np.arctan2(
                self.l2 * np.sin(t2), self.l1 + self.l2 * np.cos(t2))
            q_rear = wrap_pi(t1 - self.a1) / self.sgn_rear
            q_shank = wrap_pi(t2 - self.a2) / self.sgn_shank
            distance = abs(q_rear - hint[0]) + abs(q_shank - hint[1])
            if best is None or distance < best[0]:
                best = (distance, float(q_rear), float(q_shank))
        return best[1], best[2]

    # ---------------- 闭链 ----------------
    def _pin_residual(self, q):
        data = self._data
        for key in JOINT_ORDER:
            data.qpos[self.qadr[key]] = float(q[key])
        mujoco.mj_kinematics(self.m, data)
        sites = self.site_ids
        p1a, p1b = data.site_xpos[sites['pin1_a']], data.site_xpos[sites['pin1_b']]
        p2a, p2b = data.site_xpos[sites['pin2_a']], data.site_xpos[sites['pin2_b']]
        return np.array([p1a[0] - p1b[0], p1a[2] - p1b[2],
                         p2a[0] - p2b[0], p2a[2] - p2b[2]])

    def _solve(self, fixed, unknowns, seed=None, tol=1e-13, iters=60, max_step=0.35):
        """固定 fixed 关节，Newton 解 unknowns 关节，返回 (q dict, 残差, x)。"""
        q = dict.fromkeys(JOINT_ORDER + ('wheel',), 0.0)
        q.update({key: float(value) for key, value in fixed.items()})
        x = np.zeros(len(unknowns)) if seed is None else np.asarray(
            seed, dtype=float).reshape(-1).copy()
        residual = float('inf')
        for _ in range(iters):
            for index, key in enumerate(unknowns):
                q[key] = float(x[index])
            r = self._pin_residual(q)
            residual = float(np.linalg.norm(r))
            if residual < tol:
                break
            jac = np.zeros((4, 4))
            eps = 1e-7
            for column in range(4):
                trial = x.copy()
                trial[column] += eps
                for index, key in enumerate(unknowns):
                    q[key] = float(trial[index])
                jac[:, column] = (self._pin_residual(q) - r) / eps
            try:
                dx = np.linalg.solve(jac, -r)
            except np.linalg.LinAlgError:
                dx = -np.linalg.lstsq(jac, r, rcond=None)[0]
            if not np.all(np.isfinite(dx)):
                break
            step = float(np.max(np.abs(dx)))
            if step > max_step:
                dx *= max_step / step
            x = x + dx
            if float(np.linalg.norm(dx)) < 1e-15:
                break
        for index, key in enumerate(unknowns):
            q[key] = float(x[index])
        return q, float(np.linalg.norm(self._pin_residual(q))), x

    def solve_loop(self, q_front, q_rear, seed=None, **kwargs):
        """真机驱动输入 (曲柄, 大腿) -> 全部关节角。返回 (q, 残差, x4)。"""
        return self._solve({'front': float(q_front), 'rear': float(q_rear)},
                           FREE_ORDER, seed, **kwargs)

    def solve_virtual(self, q_rear, q_shank, seed=None, **kwargs):
        """虚拟腿姿态 (大腿, 膝) -> 全部关节角（含驱动曲柄角）。"""
        return self._solve({'rear': float(q_rear), 'shank': float(q_shank)},
                           ('front', 'child1', 'child2', 'child3'), seed, **kwargs)

    def shank_from_inputs(self, q_front, q_rear, seed=None):
        q, residual, x4 = self.solve_loop(q_front, q_rear, seed=seed)
        return q['shank'], q, residual, x4

    def solutions(self, q_front, q_rear, n_start=12, seed=None, seed_range=(-math.pi, math.pi)):
        """枚举同一输入下的多个装配根，返回 [(残差, x4, q dict)]，按残差排序。

        默认随机起点覆盖 (-pi, pi)：v14 被动关节范围 ±2*pi，起点只取 (-1,1) 会
        漏掉 |q|>1 的装配根（v14 实测会漏 2 个，只剩 2~3 个）。
        """
        starts = [np.zeros(4), np.zeros(4)]
        if seed is not None:
            starts[0] = np.asarray(seed, dtype=float).reshape(4)
        rng = np.random.default_rng(0)
        low, high = float(seed_range[0]), float(seed_range[1])
        starts += [rng.uniform(low, high, 4) for _ in range(n_start)]
        found = []
        for start in starts:
            q, residual, x4 = self.solve_loop(q_front, q_rear, seed=start)
            if residual > 1e-9:
                continue
            # 铰链角加 2*pi 是同一物理构型：先折算到 (-pi, pi] 再判重，
            # 否则会把 x4 相差 2*pi 的同一构型当成不同装配根。
            canonical = (np.asarray(x4, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi
            if not any(np.max(np.abs(canonical - item[1])) < 1e-5 for item in found):
                found.append((residual, canonical.copy(), q))
        found.sort(key=lambda item: item[0])
        return found

    def jacobian(self, q_front, q_rear, seed=None, eps=1e-6):
        """FD: (dq_shank/dq_rear, dq_shank/dq_front)，仅用于交叉验证。"""
        def shank(front, rear):
            return self.shank_from_inputs(front, rear, seed=seed)[0]

        d_rear = (shank(q_front, q_rear + eps) - shank(q_front, q_rear - eps)) / (2 * eps)
        d_front = (shank(q_front + eps, q_rear) - shank(q_front - eps, q_rear)) / (2 * eps)
        return d_rear, d_front

    def torque_map(self, tau_hip, tau_knee, q_front, q_rear, seed=None):
        """虚拟 (髋, 膝) 力矩 -> 真机 (后髋, 曲柄) 力矩（虚功法）。"""
        d_rear, d_front = self.jacobian(q_front, q_rear, seed=seed)
        return {'rear': tau_hip + tau_knee * d_rear,
                'front': tau_knee * d_front}


def model_info(model=None):
    """v14 模型自检信息 + 来源哈希（不是实车测量）。"""
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(SCENE))
    return {
        'scene': str(SCENE.relative_to(WS)),
        'robot_xml': str(ROBOT_XML.relative_to(WS)),
        'scene_sha256': sha256_file(SCENE),
        'robot_sha256': sha256_file(ROBOT_XML),
        'expected_scene_sha256': SCENE_SHA256,
        'expected_robot_sha256': ROBOT_SHA256,
        'sha_match': (sha256_file(SCENE) == SCENE_SHA256
                      and sha256_file(ROBOT_XML) == ROBOT_SHA256),
        'total_mass_kg': float(model.body_mass.sum()),
        'expected_total_mass_kg': EXPECTED_TOTAL_MASS_KG,
        'base_mass_kg': float(model.body_mass[
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link'))]),
        'nq': int(model.nq),
        'nu': int(model.nu),
        'joint_names': [model.joint(i).name for i in range(model.njnt)],
        'actuator_names': [model.actuator(i).name for i in range(model.nu)],
        'actuator_ctrlrange': [model.actuator_ctrlrange[i].tolist()
                               for i in range(model.nu)],
        'keyframe_names': [model.key(i).name for i in range(model.nkey)],
        'note': ('ctrlrange ±3.14 的语义未确认（README 称对齐 COD ±3.14 N·m，'
                 '但不作为本模块的力矩/位置限值使用）'),
    }


def load(scene=None, base_body='base_link', verify=True):
    """便捷入口：返回 (model, left_leg, right_leg)。"""
    path = Path(scene) if scene is not None else SCENE
    model = mujoco.MjModel.from_xml_path(str(path))
    legs = (V14Leg(model, 'left', base_body), V14Leg(model, 'right', base_body))
    if verify:
        info = model_info(model)
        if not info['sha_match']:
            raise RuntimeError('v14 model hash mismatch; refusing to derive geometry')
        if abs(info['total_mass_kg'] - EXPECTED_TOTAL_MASS_KG) > 1e-6:
            raise RuntimeError('unexpected v14 total mass %r' % info['total_mass_kg'])
        if info['nu'] != EXPECTED_NU:
            raise RuntimeError('unexpected v14 nu %r' % info['nu'])
    return model, legs[0], legs[1]


if __name__ == '__main__':
    model, left, right = load()
    print('model:', str(SCENE.relative_to(WS)))
    print('mass %.10f kg, nu=%d, sha_match=%s'
          % (model.body_mass.sum(), model.nu, model_info(model)['sha_match']))
    for leg in (left, right):
        print('[%s] l1=%.6f l2=%.6f a1=%.3f deg a2=%.3f deg sgn_rear=%+.0f '
              'sgn_shank=%+.0f wheel_r=%.5f'
              % (leg.side, leg.l1, leg.l2, math.degrees(leg.a1),
                 math.degrees(leg.a2), leg.sgn_rear, leg.sgn_shank, leg.wheel_radius))
        for slot in ('front', 'rear'):
            item = leg.driven[slot]
            print('      %-5s phi0=%+9.5f deg  dphi/dq=%+d  length=%.9f m'
                  % (slot, item['phi_at_model_q0_deg'], item['dphi_dq'],
                     item['projected_link_length_m']))
        L0, theta = leg.virtual_leg(0.0, 0.0)
        print('      q=0: L0=%.6f m theta0=%.3f deg' % (L0, math.degrees(theta)))
