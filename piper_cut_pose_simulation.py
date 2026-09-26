#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PiPER 四类果梗剪切姿态模拟计算版

功能：
1. 不连接 MoveIt；
2. 不驱动机械臂；
3. 不执行 U -> C -> D -> E；
4. 直接根据预定义 U、D、C、E 四个点计算最终剪切位姿；
5. 输出 FINAL [x, y, z, roll, pitch, yaw]；
6. 同时输出对应四元数，方便后续 ROS2 / MoveIt 使用。

坐标约定：
- 所有 XYZ 均在 base_link 坐标系下，单位 m；
- RPY 单位 degree；
- R = Rz(yaw) @ Ry(pitch) @ Rx(roll)；
- link6 +Y = D -> U，即果梗方向；
- link6 +Z = E -> C 投影到垂直果梗平面后的剪刀向前方向；
- link6 +X = Y x Z，按右手系确定。
"""

import math
from typing import Iterable, Sequence, Tuple


# ============================================================
# 固定工具补偿
# ============================================================

TOOL_Z_OFFSET_DEG = -90.0


# ============================================================
# 数学工具
# ============================================================

def _vector3(value: Iterable[float], name: str) -> Tuple[float, float, float]:
    try:
        result = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是三个有限数值") from exc

    if len(result) != 3 or not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} 必须是三个有限数值")

    return result


def _subtract(a: Sequence[float], b: Sequence[float]):
    return tuple(ai - bi for ai, bi in zip(a, b))


def _unit(v: Sequence[float], name: str):
    length = math.hypot(*v)

    if not math.isfinite(length) or length < 1e-12:
        raise ValueError(f"{name} 长度为零或数值异常")

    return tuple(a / length for a in v)


def _dot(a: Sequence[float], b: Sequence[float]):
    return sum(ai * bi for ai, bi in zip(a, b))


def _cross(a: Sequence[float], b: Sequence[float]):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def rpy_to_quaternion(roll: float, pitch: float, yaw: float):
    """
    输入：
        roll, pitch, yaw：弧度
    返回：
        qx, qy, qz, qw
    """
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw


# ============================================================
# U/D/C/E -> 最终剪切姿态
# ============================================================

def calculate_cut_rotation(U, D, C, E):
    """
    根据 U、D、C、E 计算 link6 目标旋转矩阵。

    轴映射：
        link6 +Y = D -> U
        link6 +Z = 剪刀向前方向
        link6 +X = Y x Z
    """
    upper, lower, cut, current = (
        _vector3(value, name)
        for value, name in (
            (U, "U"),
            (D, "D"),
            (C, "C"),
            (E, "E"),
        )
    )

    # D -> U：果梗方向
    stem_dir = _unit(_subtract(upper, lower), "U-D")

    # E -> C：剪刀接近方向
    approach = _unit(_subtract(cut, current), "C-E")

    # 将接近方向投影到垂直果梗方向的平面
    parallel = _dot(approach, stem_dir)
    projected = tuple(
        a - parallel * s
        for a, s in zip(approach, stem_dir)
    )

    if math.hypot(*projected) < 1e-10:
        raise ValueError(
            "E->C 与果梗方向几乎平行，投影后方向接近零，无法确定剪刀向前方向"
        )

    tool_y = stem_dir
    tool_z = _unit(projected, "剪刀向前方向")
    tool_x = _unit(_cross(tool_y, tool_z), "工具 X 轴")
    tool_z = _unit(_cross(tool_x, tool_y), "工具 Z 轴")

    R = [
        [tool_x[i], tool_y[i], tool_z[i]]
        for i in range(3)
    ]

    # 固定安装补偿：绕工具自身 Z 轴旋转
    theta = math.radians(TOOL_Z_OFFSET_DEG)
    c = math.cos(theta)
    s = math.sin(theta)

    R_offset = [
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ]

    R_corrected = [
        [
            sum(R[i][k] * R_offset[k][j] for k in range(3))
            for j in range(3)
        ]
        for i in range(3)
    ]

    return R_corrected


def _rotation_to_rpy_degrees(r):
    """
    Rotation Matrix -> RPY(deg)

    使用：
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    horizontal = math.hypot(r[0][0], r[1][0])
    pitch = math.atan2(-r[2][0], horizontal)

    if horizontal > 1e-12:
        roll = math.atan2(r[2][1], r[2][2])
        yaw = math.atan2(r[1][0], r[0][0])
    else:
        roll = math.atan2(-r[1][2], r[1][1])
        yaw = 0.0

    return [
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    ]


def calculate_cut_pose(U, D, C, E):
    """
    输入：
        U, D, C, E：四个 XYZ 点

    输出：
        [x, y, z, roll, pitch, yaw]
    """
    cut = _vector3(C, "C")
    rotation = calculate_cut_rotation(U, D, cut, E)

    return list(cut) + _rotation_to_rpy_degrees(rotation)


# ============================================================
# 模拟测试数据
# ============================================================
# ============================================================
# 模拟测试数据
# ============================================================
# ============================================================
# 模拟测试数据
# ============================================================

SAFE_RPY_DEFAULT = (-85.6, -2.47, -175.5)

COMMON_C = (-0.113, -0.334, 0.211)
COMMON_E = (-0.113, -0.324, 0.211)

# 对角线偏移量（Y-Z 平面内各偏移 7 cm，总长约 19.8 cm）
DIAG_DY = 0.07
DIAG_DZ = 0.07

TEST_CASES = (
    {
        "name": "01 竖直果梗",
        "U": (-0.113, -0.334, 0.201),
        "C": COMMON_C,
        "D": (-0.113, -0.334, 0.221),
        "E": COMMON_E,
    },
    {
        "name": "02 水平果梗",
        "U": (-0.113, -0.334, 0.211),
        "C": COMMON_C,
        "D": (-0.113, -0.324, 0.211),
        "E": COMMON_E,
    },
    {
        "name": "03 左上→右下",
        "U": (COMMON_C[0], COMMON_C[1] - DIAG_DY, COMMON_C[2] + DIAG_DZ),
        "C": COMMON_C,
        "D": (COMMON_C[0], COMMON_C[1] + DIAG_DY, COMMON_C[2] - DIAG_DZ),
        "E": COMMON_E,
    },
    {
        "name": "04 右上→左下",
        "U": (COMMON_C[0], COMMON_C[1] + DIAG_DY, COMMON_C[2] + DIAG_DZ),
        "C": COMMON_C,
        "D": (COMMON_C[0], COMMON_C[1] - DIAG_DY, COMMON_C[2] - DIAG_DZ),
        "E": COMMON_E,
    },
)


# ============================================================
# 输出
# ============================================================

def print_case_result(case):
    U = case["U"]
    D = case["D"]
    C = case["C"]
    E = case["E"]

    print("\n" + "=" * 72)
    print(f"模拟计算：{case['name']}")
    print("=" * 72)

    print(f"U = {list(U)}")
    print(f"C = {list(C)}")
    print(f"D = {list(D)}")
    print(f"E = {list(E)}")
    print(f"local-Z 工具补偿 = {TOOL_Z_OFFSET_DEG:+.1f}°")

    try:
        target = calculate_cut_pose(U, D, C, E)
    except ValueError as exc:
        print(f"❌ 最终姿态计算失败：{exc}")
        return False

    target_xyz = target[:3]
    target_rpy = target[3:]

    roll = math.radians(target_rpy[0])
    pitch = math.radians(target_rpy[1])
    yaw = math.radians(target_rpy[2])

    qx, qy, qz, qw = rpy_to_quaternion(roll, pitch, yaw)

    print("\n计算结果：")
    print(
        "FINAL XYZ = "
        f"[{target_xyz[0]:.6f}, "
        f"{target_xyz[1]:.6f}, "
        f"{target_xyz[2]:.6f}] m"
    )

    print(
        "FINAL RPY = "
        f"[{target_rpy[0]:.3f}, "
        f"{target_rpy[1]:.3f}, "
        f"{target_rpy[2]:.3f}] deg"
    )

    print(
        "FINAL quaternion = "
        f"[{qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f}]"
    )

    return True


def run_all_cases():
    print("\n" + "#" * 72)
    print("PiPER 四类果梗剪切姿态模拟计算")
    print("#" * 72)
    print("本程序只做数学计算，不连接 MoveIt，也不会驱动机械臂。")

    success_count = 0

    for case in TEST_CASES:
        if print_case_result(case):
            success_count += 1

    print("\n" + "#" * 72)
    print(f"模拟计算结束：{success_count}/{len(TEST_CASES)} 组计算成功")
    print("#" * 72)


def main():
    run_all_cases()


if __name__ == "__main__":
    main()
