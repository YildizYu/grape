#!/usr/bin/env python3
"""监听 /grape_harvest/camera_target — 用户按 d 键后实时打印三个输入与输出。

打印内容:
  [CAMERA] 相机三维坐标 (视觉检测结果)
  [ROBOT]  当前机械臂 TCP 位姿 (订阅)
  [HAND-EYE] 手眼标定平移 (加载结果)
  [BASE]   计算得到的目标 base 坐标
  [CHECK]  距基座距离 + 工作半径判定

用法:
    python3 scripts/watch_camera_target.py [--timeout-s 300]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PointStamped
    from dobot_msgs_v4.msg import ToolVectorActual
except ImportError as e:
    sys.exit(f"缺少 ROS2 依赖: {e}\n请先: source install/setup.bash")

from grape_stem_3d.handeye_transform import (
    load_handeye_matrix,
    tool_vector_to_T_base_tool,
)
from grape_stem_3d.target_pose_bridge import build_target_pose


class CameraTargetWatcher(Node):
    def __init__(self, handeye_path: Path):
        super().__init__("camera_target_watcher")
        self._T_tool_camera = load_handeye_matrix(str(handeye_path))
        self._T_base_tool = None
        self._pose_time = 0.0
        self._got_target = False

        self.create_subscription(
            PointStamped, "/grape_harvest/camera_target", self._on_target, 10)
        self.create_subscription(
            ToolVectorActual, "/dobot_msgs_v4/msg/ToolVectorActual",
            self._on_tool_vector, 10)

    def _on_tool_vector(self, msg):
        self._T_base_tool = tool_vector_to_T_base_tool(
            msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz)
        self._pose_time = time.time()

    def _on_target(self, msg: PointStamped):
        self._got_target = True
        camera_xyz = (msg.point.x, msg.point.y, msg.point.z)
        print("=" * 64)
        print(f"[CAMERA] 相机三维坐标 (检测结果): "
              f"({camera_xyz[0]:.4f}, {camera_xyz[1]:.4f}, {camera_xyz[2]:.4f}) m "
              f"| frame={msg.header.frame_id}")

        if self._T_base_tool is None or time.time() - self._pose_time > 2.0:
            print("[ROBOT]  ❌ 无新鲜机械臂位姿 (2s 内无 ToolVectorActual)")
            return

        print(f"[ROBOT]  当前机械臂 TCP: "
              f"pos=({self._T_base_tool[0,3]*1000:.1f}, "
              f"{self._T_base_tool[1,3]*1000:.1f}, "
              f"{self._T_base_tool[2,3]*1000:.1f}) mm "
              f"(位姿新鲜 {time.time()-self._pose_time:.2f}s)")
        print(f"[HAND-EYE] 手眼平移: {self._T_tool_camera[:3,3]} m "
              f"(模长 {np.linalg.norm(self._T_tool_camera[:3,3])*1000:.1f} mm)")

        result = build_target_pose(
            camera_xyz, self._T_tool_camera, self._T_base_tool,
            fixed_rpy_deg=(0.0, 0.0, 0.0), workspace_max_radius_m=1.2)
        if not result["valid"]:
            print(f"[BASE]   ❌ {result['reason']}")
            return
        bx, by, bz = result["base_xyz"]
        dist = float(np.linalg.norm(result["base_xyz"]))
        print(f"[BASE]   目标 base 坐标: ({bx:.4f}, {by:.4f}, {bz:.4f}) m")
        print(f"[CHECK]  距基座 {dist*1000:.1f} mm "
              f"{'| ✅ 0.9m 工作半径内' if dist <= 0.9 else '| ⚠ 超出 0.9m, 够不到'}")
        if abs(bz) > 0.5:
            print(f"[CHECK]  ⚠ z={bz:.3f} m 偏高, 接近 CR5 高度上限")
        print("=" * 64, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-s", type=float, default=300.0)
    args = parser.parse_args()

    handeye_path = Path("/home/nvidia/GrapeHarvestControl/handeye_1/handeye_result.yaml")
    rclpy.init()
    node = CameraTargetWatcher(handeye_path)
    print(f"监听 /grape_harvest/camera_target, 超时 {args.timeout_s:.0f}s ... "
          "现在去主程序按 d 键", flush=True)

    deadline = time.time() + args.timeout_s
    try:
        while time.time() < deadline and not node._got_target:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if not node._got_target:
            print("超时: 未收到相机目标 (主程序跑了吗? 按 d 了吗?)")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
