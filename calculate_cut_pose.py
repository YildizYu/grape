#!/usr/bin/env python3
"""由 U、D、C、E 计算剪刀目标 [x, y, z, roll, pitch, yaw]。

四个点均使用机械臂基坐标系，长度单位一致，输出 XYZ 保持相同单位。
输出 RPY 单位为度，约定 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。
R 的列是工具坐标轴在基坐标系中的表达。

工具坐标系：TCP 位于实际剪切位置；+X 指向刀口前方；+Z 是切割
平面法向，与 D -> U 同向；+Y 按右手规则确定。这是该工具坐标系的
目标姿态；若控制器使用法兰坐标系或其他工具轴，需先做安装变换，
不能直接将此输出作为法兰位姿。四个位置本身无法推断安装变换。

Python:
    pose = calculate_cut_pose(U, D, C, E)
CLI:
    python3 vector/calculate_cut_pose.py --u 0 0 1 --d 0 0 -1 \
        --c 0 0 0 --e -1 0 0
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


def calculate_cut_rotation(U, D, C, E):
    """返回 3×3 目标工具旋转矩阵（嵌套列表），用于检查或其他姿态格式。"""
    upper, lower, cut, current = (
        _vector3(value, name)
        for value, name in ((U, "U"), (D, "D"), (C, "C"), (E, "E"))
    )
    # 切割平面法向：从下点指向上点。
    z = _unit(_subtract(upper, lower), "U-D")
    approach = _unit(_subtract(cut, current), "C-E")
    # 将接近方向投影到垂直于果梗的平面。
    parallel = sum(a*b for a, b in zip(approach, z))
    projected = tuple(a - parallel*b for a, b in zip(approach, z))
    if math.hypot(*projected) < 1e-10:
        raise ValueError("接近方向与果梗平行，无法由这四个点确定刀口朝向")
    x = _unit(projected, "刀口朝向")
    y = _unit(_cross(z, x), "工具 Y 轴")
    x = _unit(_cross(y, z), "工具 X 轴")
    return [[x[i], y[i], z[i]] for i in range(3)]


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


def calculate_cut_pose(U, D, C, E):
    """输入四个 [x,y,z]，输出 [x,y,z,roll,pitch,yaw]，角度为度。

    U: 果梗上点；D: 果梗下点；C: 剪切点；E: 当前末端位置。
    XYZ 直接取 C；当前 RPY 不参与计算。工具轴定义见文件开头。
    无效坐标或不能确定方向时抛出 ValueError。
    """
    cut = _vector3(C, "C")
    rotation = calculate_cut_rotation(U, D, cut, E)
    return list(cut) + _rotation_to_rpy_degrees(rotation)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, description in (("u", "上点"), ("d", "下点"),
                              ("c", "剪切点"), ("e", "当前末端位置")):
        parser.add_argument(f"--{name}", nargs=3, type=float, required=True,
                            metavar=("X", "Y", "Z"), help=description)
    args = parser.parse_args()
    try:
        pose = calculate_cut_pose(args.u, args.d, args.c, args.e)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(pose, allow_nan=False))


if __name__ == "__main__":
    main()
