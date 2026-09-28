#!/usr/bin/env python3
"""由 U、D、C、E 计算剪刀目标 [x, y, z, roll, pitch, yaw]。

四个点均使用机械臂基坐标系，长度单位一致，输出 XYZ 保持相同单位。
输出 RPY 单位为度，约定 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。
R 的列是工具坐标轴在基坐标系中的表达。

工具坐标系：TCP 位于实际剪切位置；+X 指向刀口前方；+Z 是切割
平面法向（current_z 提供时保持当前 TCP Z 轴方向）；+Y 按右手规则确定。
这是该工具坐标系的目标姿态；若控制器使用法兰坐标系或其他工具轴，
需先做安装变换，不能直接将此输出作为法兰位姿。四个位置本身无法推断
安装变换。

Python:
    # 现场版 (保持当前 TCP Z 轴方向):
    pose = calculate_cut_pose(U, D, C, E, current_z)
    # 旧四点法 (Z 取果梗方向 D→U):
    pose = calculate_cut_pose(U, D, C, E)
CLI:
    python3 src/grape_stem_3d/calculate_cut_pose.py --u 0 0 1 --d 0 0 -1 \
        --c 0 0 0 --e -1 0 0 --z 0 0 1
"""

import argparse
import json
import math


def _vector3(value, name):
    try:
        result = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是三个有限数值") from exc
    if len(result) != 3 or not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} 必须是三个有限数值")
    return result


def _subtract(a, b):
    return tuple(ai - bi for ai, bi in zip(a, b))


def _unit(v, name):
    length = math.hypot(*v)
    if not math.isfinite(length) or length == 0:
        raise ValueError(f"{name} 长度为零或数值超出范围")
    return tuple(a / length for a in v)


def _cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])


def calculate_cut_rotation(U, D, C, E, current_z=None):
    """返回 3×3 目标工具旋转矩阵（列向量 = 工具 X/Y/Z 轴）。

    current_z 提供（现场 5 参版，与 Box 一致）:
        - Z 轴保持当前机械臂 TCP 的 Z 轴方向；
        - -Y 为夹爪朝向果梗的方向，并保证其垂直于果梗；
        - X 由右手系确定（X = Y × Z）。
    current_z 为 None（旧四点法，兼容旧测试/旧调用）:
        - Z 取果梗方向 D→U；
        - X 取接近方向在垂直果梗平面内的投影；
        - Y = Z × X。

    参数:
        U, D : 果梗上/下点，D -> U 为果梗方向
        C    : 剪切点
        E    : 当前末端位置
        current_z : 当前 TCP 的 Z 轴方向（3 维向量，可选）

    返回:
        3×3 嵌套列表，列为工具坐标系的 X/Y/Z 轴。
    """
    upper, lower, cut, current = (
        _vector3(value, name)
        for value, name in ((U, "U"), (D, "D"), (C, "C"), (E, "E"))
    )
    stem = _unit(_subtract(upper, lower), "U-D")

    if current_z is None:
        # ── 旧四点法: 无当前 Z, 工具 Z 取果梗方向 ──
        approach = _unit(_subtract(cut, current), "C-E")
        # 将接近方向投影到垂直于果梗的平面。
        parallel = sum(a*b for a, b in zip(approach, stem))
        projected = tuple(a - parallel*b for a, b in zip(approach, stem))
        if math.hypot(*projected) < 1e-10:
            raise ValueError("接近方向与果梗平行，无法由这四个点确定刀口朝向")
        x = _unit(projected, "刀口朝向")
        y = _unit(_cross(stem, x), "工具 Y 轴")
        x = _unit(_cross(y, stem), "工具 X 轴")
        return [[x[i], y[i], stem[i]] for i in range(3)]

    # ── 现场 5 参版: 保持当前机械臂 TCP 的 Z 轴方向 ──
    z = _unit(_vector3(current_z, "current_z"), "current_z")
    # 当前末端 -> 剪切点，用于确定夹爪朝向的正负
    approach = _unit(_subtract(cut, current), "C-E")
    # Y 轴必须同时垂直于:
    #   1. TCP Z 轴
    #   2. 果梗方向
    y_candidate = _cross(z, stem)
    if math.hypot(*y_candidate) < 1e-10:
        # 当前 Z 与果梗近似平行，单靠 z 和 stem 无法确定 Y，
        # 此时使用接近方向在垂直 z 平面内的投影来定 -Y。
        parallel = sum(a * b for a, b in zip(approach, z))
        projected = tuple(
            a - parallel * b
            for a, b in zip(approach, z)
        )
        if math.hypot(*projected) < 1e-10:
            raise ValueError(
                "当前Z轴与果梗方向近似平行，且接近方向无法确定工具Y轴"
            )
        # -Y 朝向剪切点
        minus_y = _unit(projected, "夹爪切入方向")
        y = tuple(-v for v in minus_y)
    else:
        y = _unit(y_candidate, "工具Y轴")
        # ±Y 都满足垂直条件，选择让 -Y 更接近
        # “当前末端 -> 剪切点”的那个方向。
        minus_y = tuple(-v for v in y)
        if sum(a * b for a, b in zip(minus_y, approach)) < 0.0:
            y = tuple(-v for v in y)
    # 右手坐标系：X = Y × Z
    x = _unit(_cross(y, z), "工具X轴")
    return [
        [x[i], y[i], z[i]]
        for i in range(3)
    ]


def _rotation_to_rpy_degrees(r):
    horizontal = math.hypot(r[0][0], r[1][0])
    pitch = math.atan2(-r[2][0], horizontal)
    if horizontal > 1e-12:
        roll = math.atan2(r[2][1], r[2][2])
        yaw = math.atan2(r[1][0], r[0][0])
    else:
        # 欧拉角奇异点：固定 yaw=0，仍表示同一个物理朝向。
        roll = math.atan2(-r[1][2], r[1][1])
        yaw = 0.0
    return [math.degrees(a) for a in (roll, pitch, yaw)]


def calculate_cut_pose(U, D, C, E, current_z=None):
    """由 U/D/C/E（及可选当前 TCP Z 轴）计算切割位姿。

    参数:
        U, D : 果梗上/下点，D -> U 为果梗方向
        C    : 剪切点
        E    : 当前末端位置
        current_z : 当前 TCP 的 Z 轴方向（3 维向量，可选;
                    提供则保持该方向, 与现场 Box 版一致）

    返回:
        [x, y, z, roll, pitch, yaw]
        其中 (x, y, z) 为剪切点 C 的位置，
        (roll, pitch, yaw) 为由 calculate_cut_rotation 得到的姿态角（度）。
    """
    cut = _vector3(C, "C")

    rotation = calculate_cut_rotation(
        U,
        D,
        cut,
        E,
        current_z,
    )

    return list(cut) + _rotation_to_rpy_degrees(rotation)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, description in (("u", "上点"), ("d", "下点"),
                              ("c", "剪切点"), ("e", "当前末端位置")):
        parser.add_argument(f"--{name}", nargs=3, type=float, required=True,
                            metavar=("X", "Y", "Z"), help=description)
    parser.add_argument("--z", nargs=3, type=float, default=None,
                        metavar=("X", "Y", "Z"),
                        help="可选: 当前 TCP Z 轴方向 (提供则保持该方向, 现场版)")
    args = parser.parse_args()
    try:
        pose = calculate_cut_pose(args.u, args.d, args.c, args.e, args.z)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(pose, allow_nan=False))


if __name__ == "__main__":
    main()
