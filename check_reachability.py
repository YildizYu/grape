#!/usr/bin/env python3
"""机械臂可达性检查 — 调 DOBOT 逆解服务, 完全不动机械臂。

用法 (坐标 = d 键打印的 base 坐标, 米; 姿态 = 动态剪切姿态 rx/ry/rz, 度):
    python3 scripts/check_reachability.py --x -0.30 --y -0.50 --z 0.30
    python3 scripts/check_reachability.py --x -0.43 --y -0.96 --z 0.18 \
        --rx 136 --ry 52 --rz 44

判据:
    res=0 且 robot_return={j1,j2,...,j6} → 该点有逆解 (可达)
    res<0 → 不可达/求解失败 (如 -20000 指令执行失败: 先清机械臂错误再试)
    ※ DOBOT 逆解有解 ≠ MoveIt 路径可达, 但直驱链 (MovL) 以 DOBOT 逆解为准

提示: 目标距基座原点距离也会打印, ≥0.9m 会被 pick_flow 工作半径门禁拒绝。
"""

import argparse
import math
import sys

import numpy as np

try:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from dobot_msgs_v4.srv import InverseKin, RobotMode
except ImportError as e:
    sys.exit(
        f"缺少 ROS2 依赖: {e}\n"
        f"请先 source 工作区环境: source install/setup.bash"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Check DOBOT reachability via InverseKin")
    parser.add_argument("--x", type=float, required=True, help="base 系 x (米)")
    parser.add_argument("--y", type=float, required=True, help="base 系 y (米)")
    parser.add_argument("--z", type=float, required=True, help="base 系 z (米)")
    parser.add_argument("--rx", type=float, default=0.0, help="rx (度)")
    parser.add_argument("--ry", type=float, default=0.0, help="ry (度)")
    parser.add_argument("--rz", type=float, default=0.0, help="rz (度)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 距基座原点距离 (与 pick_flow 门禁一致)
    dist_m = float(np.linalg.norm([args.x, args.y, args.z]))
    print(f"目标 base=({args.x:.3f}, {args.y:.3f}, {args.z:.3f}) m | "
          f"姿态 rx={args.rx:.1f} ry={args.ry:.1f} rz={args.rz:.1f} deg")
    print(f"距基座原点: {dist_m*1000:.1f} mm "
          f"({'OK' if dist_m <= 0.9 else '超出 0.9m 工作半径门禁!'})")

    rclpy.init(args=None)
    node = Node("check_reachability")

    mode_client = node.create_client(RobotMode, "/dobot_bringup_ros2/srv/RobotMode")
    ik_client = node.create_client(InverseKin, "/dobot_bringup_ros2/srv/InverseKin")
    if not ik_client.wait_for_service(timeout_sec=5.0):
        print("ERROR: InverseKin 服务不可用 (bringup 未启动?)")
        return 1

    # 机械臂模式检查 (9=ERROR 时逆解会被拒)
    try:
        if mode_client.wait_for_service(timeout_sec=3.0):
            mode_resp = mode_client.call(RobotMode.Request())
            print(f"当前 RobotMode: {mode_resp.robot_return} "
                  f"({'ERROR! 先清错再测' if mode_resp.robot_return.strip('{}') == '9' else 'OK'})")
    except Exception as e:
        print(f"RobotMode 查询失败: {e!r}")

    req = InverseKin.Request()
    req.x = args.x * 1000.0   # m → mm
    req.y = args.y * 1000.0
    req.z = args.z * 1000.0
    req.rx = args.rx
    req.ry = args.ry
    req.rz = args.rz
    req.use_joint_near = "0"
    req.joint_near = ""
    req.user = "0"
    req.tool = "0"

    future = ik_client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    if not future.done():
        print("InverseKin 服务调用超时 (机械臂命令通道无响应, 先清错)")
        return 2
    resp = future.result()
    if resp is None:
        print("InverseKin 服务调用失败 (无结果)")
        return 2

    joints = resp.robot_return.strip("{}")
    if resp.res == 0 and joints:
        print(f"✅ 可达: 逆解成功, 关节角 = {{{joints}}}")
        return 0
    print(f"❌ 不可达/求解失败: res={resp.res} robot_return='{resp.robot_return}'")
    print("   常见原因: 目标超工作半径 / 姿态超出关节极限 / 机械臂在 ERROR 态")
    return 1


if __name__ == "__main__":
    sys.exit(main())
