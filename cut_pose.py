"""根据果梗三点与机械臂当前姿态计算剪切目标位姿。

移植自 Vector/calculate_target_pose.py（2026-09），纯 numpy 无其他依赖。

约定:
1. 所有输入点均在机械臂基坐标系, 单位米
2. 工具 +X 轴: 从 P_up 指向 P_down（沿果梗方向）
3. 工具 +Z 轴: 垂直于果梗, 并尽量接近当前工具 +Z 方向（剪刀刃口朝向）
4. RPY 顺序: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
5. RPY 输入输出单位均为度

返回位姿: [P_cut.x, P_cut.y, P_cut.z, roll, pitch, yaw]

注意: 交换 P_up / P_down 会反转工具 X 轴, 目标姿态可能相差约 180°。
"""

import math
from typing import Iterable, List

import numpy as np


EPSILON = 1e-9


def _as_vector3(value: Iterable[float], name: str) -> np.ndarray:
    """Convert an input to a finite NumPy vector with exactly three values."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values; got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains NaN or infinity")
    return vector


def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < EPSILON:
        raise ValueError(f"{name} has zero or near-zero length")
    return vector / length


def rpy_degrees_to_rotation_matrix(rpy_degrees: Iterable[float]) -> np.ndarray:
    """Convert degree RPY to Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll_deg, pitch_deg, yaw_deg = _as_vector3(rpy_degrees, "current_rpy_degrees")
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])

    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_r, -sin_r],
            [0.0, sin_r, cos_r],
        ],
        dtype=np.float64,
    )

    rotation_y = np.array(
        [
            [cos_p, 0.0, sin_p],
            [0.0, 1.0, 0.0],
            [-sin_p, 0.0, cos_p],
        ],
        dtype=np.float64,
    )

    rotation_z = np.array(
        [
            [cos_y, -sin_y, 0.0],
            [sin_y, cos_y, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return rotation_z @ rotation_y @ rotation_x


def _wrap_degrees(angle_degrees: np.ndarray) -> np.ndarray:
    """Wrap angles to [-180, 180)."""
    return (angle_degrees + 180.0) % 360.0 - 180.0


def rotation_matrix_to_rpy_degrees(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to degree RPY using the ZYX convention.

    This is the inverse of:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    At gimbal lock, yaw is fixed to zero and roll absorbs the remaining
    equivalent rotation. The represented physical orientation remains valid.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")

    horizontal = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    singular = horizontal < EPSILON

    if not singular:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], horizontal)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        # At pitch = +/-90 degrees, infinitely many roll/yaw pairs represent
        # the same orientation. Choose yaw = 0 deterministically.
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], horizontal)
        yaw = 0.0

    return _wrap_degrees(np.rad2deg([roll, pitch, yaw]))


def real_to_calc(p):
    """真实系 -> 函数假设系"""
    x, y, z = p
    return (-x, -z, -y)

def calc_to_real(p):
    """函数假设系 -> 真实系"""
    x, y, z = p
    return (-x, -z, -y)   # 同一个公式

def calculate_target_pose(
    p_up: Iterable[float],
    p_cut: Iterable[float],
    p_down: Iterable[float],
    current_rpy_degrees: Iterable[float],
) -> List[float]:
    """Return [x, y, z, roll, pitch, yaw] for the cutting target.

    真实系 -> 函数假设系的轴映射：
        真实 X -> 函数 -X
        真实 Y -> 函数 -Z
        真实 Z -> 函数 -Y

    位置返回真实系的 p_cut；roll 在函数假设系的 XY 平面内拟合得到。
    """
    upper = _as_vector3(p_up, "p_up")
    cutting = _as_vector3(p_cut, "p_cut")
    lower = _as_vector3(p_down, "p_down")

    # ── 轴重映射：真实系 -> 计算系 ──
    def to_calc(p):
        x, y, z = p
        return (-x, -z, -y)

    upper_c = to_calc(upper)
    cutting_c = to_calc(cutting)
    lower_c = to_calc(lower)

    # 在计算系的 XY 平面上做最小二乘线性拟合 y = k * x + b
    xs = np.array([upper_c[0], cutting_c[0], lower_c[0]], dtype=np.float64)
    ys = np.array([upper_c[1], cutting_c[1], lower_c[1]], dtype=np.float64)

    x_mean = float(np.mean(xs))
    y_mean = float(np.mean(ys))
    dx = xs - x_mean
    dy = ys - y_mean
    denom = float(np.dot(dx, dx))

    if denom < EPSILON:
        target_roll = 0.0
    else:
        k = float(np.dot(dx, dy) / denom)
        if abs(k) > 100.0:
            target_roll = 0.0
        elif k < 0.0:
            target_roll = 90.0 + math.degrees(math.atan(k))
        else:
            target_roll = -90.0 + math.degrees(math.atan(k))

    # 位置用真实系的 cutting
    pose = np.concatenate(
        (cutting, np.array([target_roll, 0.0, 0.0], dtype=np.float64))
    )
    return [float(v) for v in pose]