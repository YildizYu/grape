#!/usr/bin/env python3
"""手动发布目标 PoseStamped — 不启动视觉, 单独测 MoveIt 链路。

用法:
    python3 scripts/test_target_pose_pub.py \
        --x 0.42 --y -0.13 --z 0.51 \
        --qx 0.0 --qy 0.0 --qz 0.0 --qw 1.0 \
        [--topic /grape_harvest/target_pose] [--frame base_link] [--times 1]

默认发布一次即退出; --times N 每隔 --interval-s 秒发一次。
"""

import argparse
import sys
import time

try:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
except ImportError as e:
    sys.exit(
        f"缺少 ROS2 依赖: {e}\n"
        f"请先 source 工作区环境: source install/setup.bash"
    )

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Manual PoseStamped publisher")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--qx", type=float, default=0.0)
    parser.add_argument("--qy", type=float, default=0.0)
    parser.add_argument("--qz", type=float, default=0.0)
    parser.add_argument("--qw", type=float, default=1.0)
    parser.add_argument("--topic", type=str, default="/grape_harvest/target_pose")
    parser.add_argument("--frame", type=str, default="dummy_link",
                        help="MoveIt planning frame (运行时实测 = dummy_link)")
    parser.add_argument("--times", type=int, default=1, help="发布次数 (默认 1)")
    parser.add_argument("--interval-s", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()

    # 校验: XYZ finite, 四元数 finite + 归一化
    quat = np.array([args.qx, args.qy, args.qz, args.qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not np.all(np.isfinite([args.x, args.y, args.z])) or not np.all(np.isfinite(quat)):
        print("ERROR: 输入含 NaN/Inf, 拒绝发布")
        return 1
    if abs(norm - 1.0) > 1e-6:
        if norm < 1e-12:
            print("ERROR: 四元数模长为 0, 拒绝发布")
            return 1
        print(f"WARNING: 四元数模长 {norm:.6f} ≠ 1, 已归一化")
        quat /= norm

    rclpy.init(args=None)
    node = Node("test_target_pose_pub")
    pub = node.create_publisher(PoseStamped, args.topic, 10)

    # 等订阅者出现 (最多 3s), 避免首条消息丢失
    deadline = time.time() + 3.0
    while time.time() < deadline and pub.get_subscription_count() == 0:
        rclpy.spin_once(node, timeout_sec=0.1)

    for i in range(args.times):
        msg = PoseStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = args.frame
        msg.pose.position.x = args.x
        msg.pose.position.y = args.y
        msg.pose.position.z = args.z
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        pub.publish(msg)
        print(f"已发布 [{i+1}/{args.times}] → {args.topic} (frame={args.frame}) | "
              f"Position: {args.x:.4f} {args.y:.4f} {args.z:.4f} | "
              f"Quaternion: {quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f} "
              f"(订阅者 {pub.get_subscription_count()})")
        if i < args.times - 1:
            time.sleep(args.interval_s)

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
