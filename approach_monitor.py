#!/usr/bin/env python3
"""接近目标监控节点: 实时距离监测 + 阈值触发关夹爪 (夹爪 = 剪刀手 合剪)。

数据流 (只读监测 + 触发, 不直接发运动指令):
    /grape_harvest/target_pose (PoseStamped, base_link)   ← target_pose_transformer
    /dobot_msgs_v4/msg/ToolVectorActual (mm/度)           ← dobot_bringup
    → 10Hz timer: 距离计算 → 状态机 → 连续确认 → 关夹爪 (一次性)

夹爪接口:
    gripper.dry_run=true  (默认): 只打印 [GRIPPER] MOCK CLOSE, 不动硬件
    gripper.dry_run=false: 复用 DobotClient 剪刀手 Modbus 链,
        close = 合剪 (0x03 → 轮询 0x02), 与 pick_flow 同一接口

安全: 关夹爪前 RobotMode 必须 = 5 (ENABLE); 9/10/11 (ERROR/PAUSE/COLLISION)
或 STOP 一律禁止, 进入 ERROR 状态。所有回调非阻塞, 距离检查在 timer 内。

运行方式:
    python3 scripts/approach_monitor.py --config configs/fusion_pipeline.yaml
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

try:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from dobot_msgs_v4.msg import ToolVectorActual
except ImportError as e:
    sys.exit(
        f"缺少 ROS2 依赖: {e}\n"
        f"请先 source 工作区环境: source install/setup.bash"
    )

from grape_stem_3d.approach_monitor_logic import (
    S_APPROACHING,
    S_COMPLETED,
    S_GRIPPING,
    S_MOVING,
    ApproachConfig,
    ApproachMonitor,
    compute_distance,
)
from grape_stem_3d.handeye_transform import tool_vector_to_T_base_tool


class ApproachMonitorNode(Node):
    """订阅目标 + 当前 TCP, timer 驱动接近监控状态机。"""

    def __init__(self, config: dict):
        super().__init__("approach_monitor")

        tp_cfg = config.get("target_pose", {})
        g_cfg = config.get("gripper", {})
        robot_cfg = config.get("robot", {})

        self._target_topic = tp_cfg.get(
            "target_pose_topic", "/grape_harvest/target_pose")
        self._pose_topic = robot_cfg.get(
            "pose_topic", "/dobot_msgs_v4/msg/ToolVectorActual")
        timer_hz = float(g_cfg.get("monitor_rate_hz", 10.0))

        # 夹爪配置
        self._dry_run = bool(g_cfg.get("dry_run", True))
        self._cut_timeout_s = float(g_cfg.get("cut_timeout_s", 15.0))
        self._mon_cfg = ApproachConfig(
            close_distance_threshold_m=float(g_cfg.get(
                "close_distance_threshold_m", 0.03)),
            approach_confirm_count=int(g_cfg.get("approach_confirm_count", 3)),
            dry_run=self._dry_run,
            max_distance_m=None,
        )
        # 运动检测阈值: 相邻周期位姿变化超过 1mm 视为机械臂已开始运动
        self._move_eps_m = float(g_cfg.get("move_detect_eps_m", 0.001))

        # 真实夹爪接口 (复用 pick_flow 同款 DobotClient 剪刀链)
        self._dobot = None
        self._scissors_ready = False
        if not self._dry_run:
            from grape_stem_3d.dobot_client import DobotClient
            self._dobot = DobotClient(self, robot_cfg, logger=self.get_logger().info)
            self._scissors_ready = self._dobot.initialize_scissors()
            self.get_logger().info(
                f"[GRIPPER] 剪刀链初始化: {'OK' if self._scissors_ready else '失败'}")

        self._monitor = ApproachMonitor(
            self._mon_cfg,
            logger=self.get_logger().info,
        )

        self._prev_xyz = None  # 运动检测用
        self._last_log_distance = None  # 距离日志节流
        self._log_hz_counter = 0
        # 夹爪动作在独立线程执行 (Modbus 等待最长 15s, 不能阻塞 timer);
        # 结果由 timer 线程回收, 保证状态机只被 timer 线程推进
        self._close_in_progress = False
        self._close_result = None

        self._target_sub = self.create_subscription(
            PoseStamped, self._target_topic, self._on_target, 10)
        self._pose_sub = self.create_subscription(
            ToolVectorActual, self._pose_topic, self._on_tool_vector, 10)
        self._timer = self.create_timer(1.0 / timer_hz, self._on_timer)

        self.get_logger().info(
            f"订阅目标: {self._target_topic} | 位姿: {self._pose_topic} | "
            f"{timer_hz}Hz | 阈值 {self._mon_cfg.close_distance_threshold_m*1000:.0f}mm "
            f"x{self._mon_cfg.approach_confirm_count} | "
            f"{'DRY-RUN' if self._dry_run else '实机'}")

    # ── 订阅回调 (只存数据, 不阻塞) ──────────────
    def _on_target(self, msg: PoseStamped):
        quat = msg.pose.orientation
        target_id = str(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)
        if self._monitor.new_target(
                target_id,
                (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
                (quat.x, quat.y, quat.z, quat.w)):
            self.get_logger().info(
                f"[TARGET] Position: {msg.pose.position.x:.4f} "
                f"{msg.pose.position.y:.4f} {msg.pose.position.z:.4f} (m) | "
                f"Quaternion: {quat.x:.4f} {quat.y:.4f} {quat.z:.4f} {quat.w:.4f}")
        else:
            self.get_logger().warn(
                f"[TARGET] 目标被拒绝: {self._monitor.snapshot().last_reason}")

    def _on_tool_vector(self, msg: ToolVectorActual):
        T = tool_vector_to_T_base_tool(msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz)
        self._monitor.update_current_xyz(tuple(T[:3, 3]))

    # ── 关夹爪执行 (一次性触发; dry-run 不碰硬件) ──
    def _do_close_gripper(self) -> bool:
        if self._dry_run:
            self.get_logger().info("[GRIPPER] MOCK CLOSE (dry-run, 不控制硬件)")
            return True
        if self._dobot is None or not self._scissors_ready:
            self.get_logger().error("[GRIPPER] 剪刀链未就绪, 无法合剪")
            return False
        # 安全复查: RobotMode 必须 5 (ENABLE)
        try:
            mode = self._dobot.robot_mode()
        except Exception as e:
            self.get_logger().error(f"[GRIPPER] RobotMode 查询失败: {e!r}")
            return False
        if mode != 5:
            self.get_logger().error(f"[GRIPPER] RobotMode={mode} ≠ 5, 拒绝合剪")
            return False
        result = self._dobot.scissors_cut(self._cut_timeout_s, threading.Event())
        self.get_logger().info(f"[GRIPPER] 合剪结果: {result}")
        return result == "done"

    # ── timer 主逻辑 ──────────────────────────────
    def _on_timer(self):
        self._log_hz_counter += 1
        st = self._monitor.snapshot()

        # PLANNING → MOVING: 位姿开始变化 (机械臂已动) 视为规划执行成功
        if st.state in ("TARGET_RECEIVED", "PLANNING") and st.current_xyz is not None:
            if self._prev_xyz is not None:
                moved = float(np.linalg.norm(
                    np.asarray(st.current_xyz) - np.asarray(self._prev_xyz)))
                if moved > self._move_eps_m:
                    self._monitor.notify_moving()
                    self.get_logger().info(
                        f"[APPROACH] 检测到机械臂运动 (Δ{moved*1000:.1f}mm) → MOVING")
            if st.state == "TARGET_RECEIVED":
                self._monitor.notify_planning()

        action = self._monitor.tick()
        st = self._monitor.snapshot()
        if st.current_xyz is not None:
            self._prev_xyz = st.current_xyz

        # 动作处理: 夹爪动作放独立线程, 结果由 timer 回收 (不阻塞回调)
        if action == "close_gripper" and not self._close_in_progress:
            self.get_logger().info(
                f"[GRIPPER] Threshold: {self._mon_cfg.close_distance_threshold_m*1000:.0f}mm | "
                f"Confirm: {st.confirm_count}/{self._mon_cfg.approach_confirm_count}")
            self.get_logger().info("[GRIPPER] Closing gripper...")
            self._close_in_progress = True

            def _close_task():
                self._close_result = self._do_close_gripper()

            threading.Thread(target=_close_task, daemon=True).start()

        if self._close_in_progress and self._close_result is not None:
            self._monitor.on_gripper_closed(self._close_result)
            self._close_in_progress = False
            self._close_result = None

        # 节流日志: 距离每变化 >5mm 或每秒最多一次
        throttle = (self._log_hz_counter % 10 == 0)
        if st.last_distance_m is not None and (
                throttle
                or self._last_log_distance is None
                or abs(st.last_distance_m - self._last_log_distance) > 0.005):
            if st.state in (S_MOVING, S_APPROACHING):
                self.get_logger().info(
                    f"[APPROACH] Target: {st.target_xyz[0]:.3f} "
                    f"{st.target_xyz[1]:.3f} {st.target_xyz[2]:.3f} | "
                    f"Current: {st.current_xyz[0]:.3f} {st.current_xyz[1]:.3f} "
                    f"{st.current_xyz[2]:.3f} | Distance: {st.last_distance_m*1000:.1f}mm "
                    f"| state={st.state}")
            self._last_log_distance = st.last_distance_m

        if action == "target_reached" and st.state == S_APPROACHING:
            self.get_logger().info(
                f"[GRIPPER] Confirm: {st.confirm_count}/"
                f"{self._mon_cfg.approach_confirm_count} (阈值 "
                f"{self._mon_cfg.close_distance_threshold_m*1000:.0f}mm)")
        if st.state == S_COMPLETED:
            self.get_logger().info("[APPROACH] Harvest Complete")


def parse_args():
    parser = argparse.ArgumentParser(description="Approach monitor node")
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
    node = ApproachMonitorNode(config)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
