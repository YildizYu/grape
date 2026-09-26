#!/usr/bin/env python3
"""目标位姿转换发布节点 (分层: 视觉 → 变换 → PoseStamped → MoveIt)。

职责 (本节点不驱动机械臂, 只发布目标):
  1. 订阅 /grape_harvest/camera_target  (geometry_msgs/PointStamped,
     相机系目标 XYZ, 米, 由 run_grasp_pipeline 检测成功时发布)
  2. 订阅 /dobot_msgs_v4/msg/ToolVectorActual (机械臂当前 TCP 位姿,
     x,y,z mm + rx,ry,rz 度, Rz·Ry·Rx 固定轴欧拉角)
  3. 读取 handeye_result.yaml (T_tool_camera, camera→tool TCP, 米)
  4. 计算 P_base = T_base_tool @ T_tool_camera @ P_camera
  5. 固定采摘姿态 RPY (config target_pose.fixed_rpy_deg) → 四元数
     Position comes from vision; orientation uses predefined harvesting orientation.
  6. 安全检查后发布 /grape_harvest/target_pose
     (geometry_msgs/PoseStamped, header.frame_id = base_link = MoveIt planning frame)

运行方式:
    python3 scripts/target_pose_transformer.py --config configs/fusion_pipeline.yaml
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
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from geometry_msgs.msg import PointStamped, PoseStamped
    from dobot_msgs_v4.msg import ToolVectorActual
except ImportError as e:
    sys.exit(
        f"缺少 ROS2 依赖: {e}\n"
        f"请先 source 工作区环境: source install/setup.bash"
    )

from grape_stem_3d.handeye_transform_new import (
    load_handeye_matrix,
    tool_vector_to_T_base_tool,
)
from grape_stem_3d.target_pose_bridge_new import (
    BASE_FRAME,
    CAMERA_FRAME,
    build_target_pose,
    rotation_matrix_to_quat,
)


class TargetPoseTransformer(Node):
    """订阅相机目标 + 机械臂位姿, 发布 base 系目标 PoseStamped。"""

    def __init__(self, config: dict):
        super().__init__("target_pose_transformer")

        tp_cfg = config.get("target_pose", {})
        robot_cfg = config.get("robot", {})
        self._camera_topic = tp_cfg.get(
            "camera_target_topic", "/grape_harvest/camera_target")
        self._target_topic = tp_cfg.get(
            "target_pose_topic", "/grape_harvest/target_pose")
        self._pose_topic = robot_cfg.get(
            "pose_topic", "/dobot_msgs_v4/msg/ToolVectorActual")
        self._frame_id = tp_cfg.get("frame_id", BASE_FRAME)
        self._fixed_rpy_deg = tuple(float(v) for v in tp_cfg.get(
            "fixed_rpy_deg", [0.0, 0.0, 0.0]))
        self._pose_max_age_s = float(tp_cfg.get("pose_max_age_s", 2.0))
        self._workspace_radius_m = tp_cfg.get("workspace_max_radius_m")
        if self._workspace_radius_m is not None:
            self._workspace_radius_m = float(self._workspace_radius_m)

        # 手眼矩阵 (默认新标定: calibration/handeye_result.yaml, 55 组 TSAI tool 链)
        handeye_path = tp_cfg.get("handeye_yaml") or robot_cfg.get(
            "handeye_yaml", "../dobot_demo/handeye_result.yaml")
        handeye_path = Path(handeye_path)
        if not handeye_path.is_absolute():
            handeye_path = _PROJECT_ROOT / handeye_path
        self._T_tool_camera = load_handeye_matrix(str(handeye_path))

        # ── 打印手眼标定结果 ──
        self.get_logger().info(f"[HAND-EYE] 标定文件: {handeye_path}")
        self.get_logger().info("[HAND-EYE] 手眼标定矩阵 T_tool_camera (camera → tool TCP):")
        for row in self._T_tool_camera:
            self.get_logger().info(
                f"    [{row[0]: .6f} {row[1]: .6f} {row[2]: .6f} {row[3]: .6f}]")
        translation_m = self._T_tool_camera[:3, 3]
        quat = rotation_matrix_to_quat(self._T_tool_camera[:3, :3])
        self.get_logger().info(
            f"[HAND-EYE] translation_m=({translation_m[0]:.6f}, "
            f"{translation_m[1]:.6f}, {translation_m[2]:.6f})")
        self.get_logger().info(
            f"[HAND-EYE] quaternion_xyzw=({quat[0]:.6f}, {quat[1]:.6f}, "
            f"{quat[2]:.6f}, {quat[3]:.6f})")

        # 机械臂当前位姿缓存
        self._T_base_tool = None
        self._pose_time = 0.0
        self._last_robot_log = None  # 位姿日志节流 (1Hz)

        self._camera_sub = self.create_subscription(
            PointStamped, self._camera_topic, self._on_camera_target, 10)
        self._pose_sub = self.create_subscription(
            ToolVectorActual, self._pose_topic, self._on_tool_vector, 10)
        self._target_pub = self.create_publisher(PoseStamped, self._target_topic, 10)

        self.get_logger().info(
            f"订阅相机目标: {self._camera_topic} | 机器人位姿: {self._pose_topic}")
        self.get_logger().info(
            f"发布目标: {self._target_topic} (frame={self._frame_id}, "
            f"fixed_rpy_deg={self._fixed_rpy_deg})")

    def _on_tool_vector(self, msg: ToolVectorActual):
        self._T_base_tool = tool_vector_to_T_base_tool(
            msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz)
        self._pose_time = time.time()

        # 订阅结果打印 (节流 1Hz, 避免话题高频刷屏)
        if self._last_robot_log is None or self._pose_time - self._last_robot_log >= 1.0:
            self._last_robot_log = self._pose_time
            self.get_logger().info(
                f"[ROBOT] 当前机械臂 TCP 位姿: "
                f"pos=({msg.x:.1f}, {msg.y:.1f}, {msg.z:.1f}) mm | "
                f"rpy=({msg.rx:.2f}, {msg.ry:.2f}, {msg.rz:.2f}) deg")

    def _on_camera_target(self, msg: PointStamped):
        # 相机帧检查: 手眼标定定义在 camera_color_optical_frame
        if msg.header.frame_id and msg.header.frame_id != CAMERA_FRAME:
            self.get_logger().warn(
                f"[CAMERA] 目标 frame={msg.header.frame_id} 与手眼标定帧 "
                f"{CAMERA_FRAME} 不一致, 忽略")
            return

        camera_xyz = (msg.point.x, msg.point.y, msg.point.z)
        self.get_logger().info(
            f"[CAMERA] Target XYZ: {camera_xyz[0]:.4f} {camera_xyz[1]:.4f} "
            f"{camera_xyz[2]:.4f} (m, {CAMERA_FRAME})")

        if self._T_base_tool is None:
            self.get_logger().warn("[ROBOT] 尚未收到机械臂位姿, 无法转换")
            return
        if time.time() - self._pose_time > self._pose_max_age_s:
            self.get_logger().warn(
                f"[ROBOT] 位姿超龄 {time.time() - self._pose_time:.1f}s "
                f"> {self._pose_max_age_s}s, 忽略")
            return

        self.get_logger().info(
            f"[ROBOT] Current Tool Pose: "
            f"pos={self._T_base_tool[:3, 3] * 1000.0} mm "
            f"(T_base_tool fresh {time.time() - self._pose_time:.2f}s)")

        result = build_target_pose(
            camera_xyz,
            self._T_tool_camera,
            self._T_base_tool,
            self._fixed_rpy_deg,
            workspace_max_radius_m=self._workspace_radius_m,
        )

        if not result["valid"]:
            self.get_logger().warn(
                f"[SAFETY] 目标无效, 不发布: {result['reason']}")
            return

        bx, by, bz = result["base_xyz"]
        qx, qy, qz, qw = result["quat"]
        self.get_logger().info(f"[BASE] Target XYZ: {bx:.4f} {by:.4f} {bz:.4f} (m)")

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self._frame_id
        pose_msg.pose.position.x = bx
        pose_msg.pose.position.y = by
        pose_msg.pose.position.z = bz
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        self._target_pub.publish(pose_msg)
        self.get_logger().info(
            f"[MOVEIT TARGET] Position: {bx:.4f} {by:.4f} {bz:.4f} | "
            f"Quaternion: {qx:.5f} {qy:.5f} {qz:.5f} {qw:.5f} "
            f"→ {self._target_topic}")


def parse_args():
    parser = argparse.ArgumentParser(description="Target pose transformer node")
    parser.add_argument(
        "--config", type=str, default="configs/fusion_pipeline.yaml",
        help="Path to fusion pipeline config (default: configs/fusion_pipeline.yaml)")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path

    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rclpy.init(args=None)
    node = TargetPoseTransformer(config)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C / SIGTERM 正常退出
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
