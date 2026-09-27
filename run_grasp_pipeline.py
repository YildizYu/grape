#!/usr/bin/env python3
"""葡萄果梗检测 → DOBOT 抓取 全流程集成节点。

功能:
  1. 连接 ZED X Mini 相机，运行葡萄/果梗检测 pipeline
  2. 通过 DobotClient 订阅 /dobot_msgs_v4/msg/ToolVectorActual 获取
     工具点 TCP 位姿，用新标定 T_tool_camera 做手眼变换
  3. PLC 下发 START 后由 PickFlowController 执行。运动后端由
     robot.motion.send_via_topic 选择:
       send_via_topic=true (现场配置, 机械臂由 MoveIt 侧执行):
         检测 → 发布 /target_pose → 延时 close_delay_s (默认 2.5s)
         → 剪刀合剪 → [place_position_mm 已示教] 发布放果点
         → 延时 open_delay_s → 开剪放果 → DONE
       send_via_topic=false (本进程直接调 DOBOT MovL):
         检测 → MovL 接近 → MovL 抓取 → 剪刀合剪 → MovL 放果
         → 开剪 → MovL 退回 → DONE
     合剪失败/ABORT 时都会尽力开剪保安全 (剪刀不夹持果梗滞留)。
  4. PLC TCP 通信 (55 AA 协议 v2.1, 见 PLC_TCP_PROTOCOL.md):
       PLC 下发 START 触发采摘, ABORT 时 Box 立即 Stop 急停机械臂;
       相机连续读失败上报 FAIL_CAMERA;
       累计 5 次 DONE 或连续 10 次 FAIL_NO_TARGET → 发 0x83 DRIVE
       通知 PLC 驱动底盘换位 (发出后暂停计数/自动扫描, 等下一次 START)
  5. 上位机 (HMI) TCP 服务 (55 AA 同帧协议, Box 作服务器监听 hmi.port):
       上位机连接后收到周期状态帧, 可下发 START/ABORT (与 PLC 等效)
  6. 运动完成判据: RobotMode 轮询 == 5 (ENABLE 空闲)

运行方式:
  cd /home/user/DOBOT_6Axis_ROS2_V4-main/grape_stem_3d_zed_deploy
  . ../install/setup.bash          # 或从工作区根 source install/setup.bash

  # 完整实时模式 (PLC 通信由 configs/fusion_pipeline.yaml 的 plc.enabled 控制)
  python scripts/run_grasp_pipeline.py --config configs/fusion_pipeline.yaml

  # 仅检测不驱动（安全模式，不调用任何机器人服务）
  python scripts/run_grasp_pipeline.py \
      --config configs/fusion_pipeline.yaml --detect-only

键盘控制 (PLC 流程执行中自动忽略 d/g):
  d — 检测当前帧中的果梗，计算 base_link 坐标
  g — 手动执行一次完整采摘流程（等同一次 PLC START）
  a — 手动中止（等同一次 PLC ABORT, 立即急停机械臂）
  s — 保存当前帧
  q — 安全退出
自动采摘 (auto_pick.enabled=true 时与按键并存, 按键不被删除):
  收到 PLC START 后进入采摘会话, 空闲时每 auto_pick.interval_s 检测一次,
  目标距 base 原点 ≤ auto_pick.max_base_radius_m (机械臂最大工作位点, 宽松)
  即自动启动采摘流程。开机未收到 START 前与 Drive 发出后 (底盘换位中)
  一律待命不扫描 (drive_gate.can_pick 门禁); 连续 no_target 扫描计数
  达阈值同样触发 Drive (流程 §5.6 场景 B)。无 PLC 通道时按键/单机模式
  保持旧行为 (门禁恒开)。
"""

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# 确保 src/ 在 path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


# ── ROS2 imports ──────────────────────────────────────
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor


# ── 状态常量 (UI 显示) ────────────────────────────────
STATE_IDLE = "IDLE"
STATE_PICKING = "PICKING"      # PLC/手动 流程执行中
STATE_DETECTED = "DETECTED"    # 手动 'd' 检测成功
STATE_MOVING = "MOVING"        # 兼容旧 overlay 颜色表


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grape Stem Detection + DOBOT Grasp Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="configs/fusion_pipeline.yaml",
        help="Path to fusion pipeline config",
    )
    parser.add_argument(
        "--handeye-yaml", type=str, default=None,
        help="Path to handeye_result.yaml (默认取 robot.handeye_yaml 配置)",
    )
    parser.add_argument(
        "--grapes-weights", type=str, default=None,
        help="Override grape detector weights",
    )
    parser.add_argument(
        "--stem-weights", type=str, default=None,
        help="Override stem detector weights",
    )
    parser.add_argument(
        "--sam-checkpoint", type=str, default=None,
        help="Override SAM checkpoint path",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (e.g. '0', 'cpu')",
    )
    parser.add_argument(
        "--grapes-conf", type=float, default=None,
        help="Grape confidence threshold",
    )
    parser.add_argument(
        "--stem-conf", type=float, default=None,
        help="Stem confidence threshold",
    )
    parser.add_argument(
        "--vision-chain", type=str, default=None,
        choices=["sam3_keypoint", "stem_yolo"],
        help="视觉链: sam3_keypoint(正式链) / stem_yolo(仅历史兼容)",
    )
    parser.add_argument(
        "--detect-only", action="store_true",
        help="Detection only mode — do NOT call any robot service (safe test mode)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Max frames (0 = unlimited)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory for saved frames",
    )
    return parser.parse_args()


class GraspPipelineNode(Node):
    """ROS2 节点：仅负责葡萄/果梗检测与结果缓存（不再直接驱动机械臂）。"""

    def __init__(
        self,
        pipeline,
        camera,
        T_tool_camera: np.ndarray,
        handeye_chain: str = "tool",
        output_dir: Optional[Path] = None,
        gripper_offset_m=None,
        tool_mount_rpy_deg=None,
    ):
        super().__init__("grasp_pipeline_node")

        self._pipeline = pipeline
        self._camera = camera
        self._T_tool_camera = T_tool_camera
        self._handeye_chain = handeye_chain
        self._output_dir = output_dir

        # 机械臂末端(Link6) → 夹爪工具中心点 的平移偏移 (Link6 系, 米)
        self._gripper_offset_m = np.array(
            gripper_offset_m if gripper_offset_m else [0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        if np.any(self._gripper_offset_m):
            self.get_logger().info(
                f"夹爪偏移补偿: gripper_offset_m="
                f"({self._gripper_offset_m[0]}, {self._gripper_offset_m[1]}, "
                f"{self._gripper_offset_m[2]}) (Link6 系)")

        # 剪刀工具系相对 Link6 的固定安装旋转 (Link6 系 Rz·Ry·Rx 固定轴, 度)
        # 视觉/算法给出的目标姿态是工具系 (TCP 在剪切点) 的姿态, 发布前
        # 换算成 Link6 姿态: R_link6 = R_target @ R_mount^T
        self._tool_mount_rpy_deg = np.array(
            tool_mount_rpy_deg if tool_mount_rpy_deg else [0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        if np.any(self._tool_mount_rpy_deg):
            self.get_logger().info(
                f"工具安装旋转补偿: tool_mount_rpy_deg="
                f"({self._tool_mount_rpy_deg[0]}, {self._tool_mount_rpy_deg[1]}, "
                f"{self._tool_mount_rpy_deg[2]}) (Link6 系, 度)")

        # 检测结果缓存
        self._last_detection: Optional[Dict[str, Any]] = None
        self._selected_target_base_xyz: Optional[Tuple[float, float, float]] = None

        # /target_pose 发布器 (send_via_topic 模式: 位姿发 MoveIt 侧执行)
        self._target_pose_pub = self.create_publisher(
            PoseStamped, "/target_pose", 10
        )

        # 状态机 (仅 UI 显示)
        self._state = STATE_IDLE

        self.get_logger().info("GraspPipelineNode 已启动")
        self.get_logger().info(f"  手眼矩阵: {T_tool_camera.shape} (chain={handeye_chain})")
        self.get_logger().info("  运动执行: PickFlowController (MovL + RobotMode 判定)")

        if self._output_dir:
            (self._output_dir / "images").mkdir(parents=True, exist_ok=True)
            (self._output_dir / "json").mkdir(parents=True, exist_ok=True)

    # ── 检测 ────────────────────────────────────────
    def detect(self, rgbd_frame) -> Optional[Dict[str, Any]]:
        """运行检测 pipeline，返回帧结果（不含手眼变换）。"""
        frame_result = self._pipeline.process_frame(
            rgbd_frame,
            image_name=f"frame_{rgbd_frame.frame_id:06d}",
        )
        self._last_detection = frame_result
        return frame_result

    # ── /target_pose 发布 (MoveIt 侧 grape_arm_control 消费) ──
    def publish_target_pose(self, base_xyz, rpy_deg) -> None:
        """base 系 (x,y,z)米 + Rz·Ry·Rx 度 → PoseStamped 发布到 /target_pose。

        与 README_ZH 中 ros2 topic pub --once /target_pose 指令等价:
          header.frame_id = base_link (MoveIt planning frame, 运行时实测)
          orientation = rpy 换算的四元数
        """
        from grape_stem_3d.cut_pose import rpy_degrees_to_rotation_matrix
        from grape_stem_3d.target_pose_bridge_new import rotation_matrix_to_quat

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        # 工具安装补偿: 视觉/算法目标位姿为剪刀工具系 (TCP 在剪切点),
        # 发布的是 Link6 位姿:
        #   R_link6 = R_target @ R_mount^T
        #   P_link6 = P_target - R_link6 @ gripper_offset (偏移定义在 Link6 系)
        R_target = rpy_degrees_to_rotation_matrix(rpy_deg)
        R_link6 = R_target
        px, py, pz = float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2])
        if np.any(self._tool_mount_rpy_deg) or np.any(self._gripper_offset_m):
            R_mount = rpy_degrees_to_rotation_matrix(self._tool_mount_rpy_deg)
            R_link6 = R_target @ R_mount.T
            link6_pos = (
                np.array([px, py, pz]) - R_link6 @ self._gripper_offset_m
            )
            px, py, pz = (
                float(link6_pos[0]), float(link6_pos[1]), float(link6_pos[2])
            )
        qx, qy, qz, qw = rotation_matrix_to_quat(R_link6)
        msg.pose.position.x = px
        msg.pose.position.y = py
        msg.pose.position.z = pz
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self._target_pose_pub.publish(msg)
        self.get_logger().info(
            f"已发布 /target_pose: pos=({base_xyz[0]:.4f},{base_xyz[1]:.4f},"
            f"{base_xyz[2]:.4f}) quat=({qx:.4f},{qy:.4f},{qz:.4f},{qw:.4f})"
        )

    def publish_home_pose(self, xyz, quat_xyzw) -> None:
        """回初始位置: 发布配置的 home 位姿 (x,y,z 米 + qx,qy,qz,qw) 到 /target_pose。"""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(xyz[0])
        msg.pose.position.y = float(xyz[1])
        msg.pose.position.z = float(xyz[2])
        qx, qy, qz, qw = (float(v) for v in quat_xyzw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self._target_pose_pub.publish(msg)
        self.get_logger().info(
            f"已发布回初始位置 /target_pose: pos=({msg.pose.position.x:.4f},"
            f"{msg.pose.position.y:.4f},{msg.pose.position.z:.4f}) "
            f"quat=({qx:.4f},{qy:.4f},{qz:.4f},{qw:.4f})"
        )

    # ── 状态查询 (UI) ────────────────────────────────
    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, value: str):
        self._state = value

    @property
    def selected_target_base_xyz(self) -> Optional[Tuple[float, float, float]]:
        return self._selected_target_base_xyz

    @selected_target_base_xyz.setter
    def selected_target_base_xyz(self, value):
        self._selected_target_base_xyz = value

    @property
    def last_detection(self):
        return self._last_detection


# ── 检测结果分类 (pick_flow 适配) ─────────────────────
def classify_detection(
    frame_result: Dict,
) -> Tuple[str, Optional[Tuple[float, float, float]], Optional[Dict]]:
    """把检测帧结果归类为 pick_flow 期望的 (status, camera_xyz, cut_info)。

    status ∈ {DETECT_OK, DETECT_NO_TARGET, DETECT_INVALID_DEPTH, DETECT_ERROR}
    camera_xyz: 剪切点 (相机系, 米)
    cut_info: 果梗三点 (相机系, 米) — 新链 keypoint 提供, 无则 None:
        {"p_up_camera": (x,y,z), "p_cut_camera": (x,y,z), "p_down_camera": (x,y,z)}
    """
    from grape_stem_3d.pick_flow import (
        DETECT_ERROR,
        DETECT_INVALID_DEPTH,
        DETECT_NO_TARGET,
        DETECT_OK,
    )
    from grape_stem_3d.realtime_pipeline import STATUS_SUCCESS
    from grape_stem_3d.types import (
        STATUS_GRAPES_NOT_DETECTED,
        STATUS_INVALID_DEPTH,
        STATUS_STEM_NOT_DETECTED,
    )

    best_grape = None
    best_conf = -1.0

    for grape in frame_result.get("grapes", []):
        if grape.get("status") != STATUS_SUCCESS:
            continue
        xyz = grape.get("peduncle_centroid_camera_xyz")
        if xyz is None or xyz.get("x") is None:
            continue
        # 正式链用 SAM3 掩膜骨架路径长度排序，旧链用果梗置信度。
        conf = grape.get("keypoint_path_length_224")
        if conf is None:
            conf = grape.get("stem_confidence", 0)
        if conf > best_conf:
            best_conf = conf
            best_grape = grape

    if best_grape is not None:
        xyz = best_grape["peduncle_centroid_camera_xyz"]
        cut_info = None
        direction = best_grape.get("stem_direction_camera_xyz")
        if direction:
            p1 = direction.get("point_1")
            p3 = direction.get("point_3")
            if (p1 and p1.get("x") is not None
                    and p3 and p3.get("x") is not None
                    and xyz.get("x") is not None):
                cut_info = {
                    "p_up_camera": (p1["x"], p1["y"], p1["z"]),
                    "p_cut_camera": (xyz["x"], xyz["y"], xyz["z"]),
                    "p_down_camera": (p3["x"], p3["y"], p3["z"]),
                }
        return (DETECT_OK, (xyz["x"], xyz["y"], xyz["z"]), cut_info)

    # 无有效目标 → 按帧状态归类失败原因
    status = frame_result.get("status")
    if status == STATUS_INVALID_DEPTH:
        return (DETECT_INVALID_DEPTH, None, None)
    if status in (STATUS_GRAPES_NOT_DETECTED, STATUS_STEM_NOT_DETECTED,
                  "stem_detection_error", "roi_empty"):
        return (DETECT_NO_TARGET, None, None)
    if status == "partial_failure":
        # 部分成功部分失败, 通常含无效深度样本
        return (DETECT_INVALID_DEPTH, None, None)
    return (DETECT_ERROR, None, None)


# ── 可视化 overlay ────────────────────────────────────
def draw_grasp_overlay(
    color_bgr: np.ndarray,
    frame_result: Dict,
    ros_node: GraspPipelineNode,
    robot,
    controller,
    fps: float = 0.0,
) -> np.ndarray:
    """在帧上绘制检测结果和机器人/流程状态。"""
    from grape_stem_3d.realtime_visualizer import draw_overlay

    output = draw_overlay(color_bgr, frame_result, fps=fps)
    h, w = output.shape[:2]

    # 机器人 TCP 位姿
    y = 55
    if robot is not None:
        T = robot.get_T_base_tool(max_age_s=5.0)
        if T is not None:
            t = T[:3, 3]
            cv2.putText(output, f"RobotTCP: ({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})m",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y += 20

    state = ros_node.state
    state_colors = {
        STATE_IDLE: (200, 200, 200),
        STATE_PICKING: (0, 165, 255),
        STATE_DETECTED: (0, 255, 255),
        STATE_MOVING: (255, 165, 0),
    }
    color = state_colors.get(state, (255, 255, 255))
    cv2.putText(output, f"State: {state}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    y += 22

    target = ros_node.selected_target_base_xyz
    if target is None and controller is not None:
        target = controller.last_target_base
    if target is not None:
        cv2.putText(output,
                    f"Target(base): ({target[0]:.3f},{target[1]:.3f},{target[2]:.3f})m",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y += 20

    # 控制提示
    cv2.putText(output, "d:detect g:pick a:abort s:save q:quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return output


# ── 构建 pipeline ─────────────────────────────────────
def build_pipeline(args, config: dict):
    """构建检测组件。"""
    from grape_stem.grape_detector import GrapeDetector
    from grape_stem.stem_detector import StemDetector
    from grape_stem.stem_selector import StemSelector
    from grape_stem_3d.sam3_runtime import build_sam3_predictor
    from grape_stem_3d.realtime_pipeline import RealtimeGrapeStemPipeline

    models_cfg = config.get("models", {})
    grapes_cfg = config.get("grapes_detector", {})
    stem_cfg = config.get("stem_detector", {})
    sel_cfg = config.get("stem_selection", {})
    sam_cfg = config.get("sam", {})
    device = args.device or models_cfg.get("device", "0")

    grapes_path = _resolve_path(args.grapes_weights or models_cfg.get("grapes_weights", "weights/grapes/best.pt"))
    stem_path = _resolve_path(args.stem_weights or models_cfg.get("stem_weights", "weights/stem/best.pt"))

    grape_detector = GrapeDetector(
        weights_path=grapes_path,
        confidence=args.grapes_conf or grapes_cfg.get("confidence", 0.25),
        iou=grapes_cfg.get("iou", 0.50),
        imgsz=grapes_cfg.get("imgsz", 640),
        device=device,
    )
    print(f"Grape detector: {grapes_path}")

    stem_detector = StemDetector(
        weights_path=stem_path,
        confidence=args.stem_conf or stem_cfg.get("confidence", 0.15),
        iou=stem_cfg.get("iou", 0.50),
        imgsz=stem_cfg.get("imgsz", 960),
        device=device,
    )
    print(f"Stem detector: {stem_path}")

    stem_selector = StemSelector(
        confidence_weight=sel_cfg.get("confidence_weight", 0.60),
        position_weight=sel_cfg.get("position_weight", 0.40),
    )

    sam_segmenter = None

    # ── 视觉链选择 ────────────────────────────────
    vision_cfg = config.setdefault("vision", {})
    chain = args.vision_chain or vision_cfg.get("chain", "sam3_keypoint")
    vision_cfg["chain"] = chain  # 回写, RealtimeGrapeStemPipeline 读此键

    keypoint_predictor = None
    if chain == "sam3_keypoint":
        sam_segmenter, sam_checkpoint_path, sam_worker_python = build_sam3_predictor(
            _PROJECT_ROOT, config, checkpoint_override=args.sam_checkpoint,
            device_override=device,
        )
        print(f"SAM3 checkpoint: {sam_checkpoint_path}")
        print(f"SAM3 worker Python: {sam_worker_python}")
    print(f"Vision chain: {chain}")

    pipeline = RealtimeGrapeStemPipeline(
        grape_detector=grape_detector,
        stem_detector=stem_detector,
        stem_selector=stem_selector,
        intrinsics=None,
        sam_segmenter=sam_segmenter,
        keypoint_predictor=keypoint_predictor,
        config=config,
    )
    print("Pipeline ready.")
    return pipeline


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


# ── 保存帧 ───────────────────────────────────────────
def _save_frame(rgbd_frame, frame_result, session_dir: Path, config: dict):
    import json

    frame_id = rgbd_frame.frame_id
    output_cfg = config.get("output", {})

    if output_cfg.get("save_rgb", True):
        rgb_path = session_dir / "images" / f"frame_{frame_id:06d}.png"
        cv2.imwrite(str(rgb_path), rgbd_frame.color_bgr)

    if output_cfg.get("save_depth", True) and rgbd_frame.depth_data is not None:
        depth_path = session_dir / "depth" / f"frame_{frame_id:06d}.npy"
        np.save(str(depth_path), rgbd_frame.depth_data)

    if output_cfg.get("save_mask", True):
        mask_dir = session_dir / "masks"
        mask_dir.mkdir(exist_ok=True)
        for grape in frame_result.get("grapes", []):
            mask = grape.get("_segmentation_mask_full")
            if isinstance(mask, np.ndarray) and mask.size > 0:
                grape_id = int(grape.get("grape_id", 0))
                cv2.imwrite(
                    str(mask_dir / f"frame_{frame_id:06d}_grape_{grape_id:02d}_sam3_mask.png"),
                    mask,
                )

    if output_cfg.get("save_json", True):
        json_path = session_dir / "json" / f"frame_{frame_id:06d}.json"

        def serialize(obj):
            if isinstance(obj, dict):
                return {
                    k: serialize(v) for k, v in obj.items()
                    if not str(k).startswith("_")
                }
            elif isinstance(obj, (list, tuple)):
                return [serialize(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            return obj

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serialize(frame_result), f, indent=2, ensure_ascii=False, default=str)


# ── 主循环 ───────────────────────────────────────────
def main():
    args = parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path

    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"Config: {config_path}")

    robot_cfg = config.get("robot", {})

    # 加载手眼矩阵（默认取 robot.handeye_yaml 配置 → 55 组 TSAI tool 链）
    from grape_stem_3d.handeye_transform import load_handeye_matrix_with_chain

    handeye_path = args.handeye_yaml or robot_cfg.get(
        "handeye_yaml", "/home/nvidia/GrapeHarvestControl/handeye_1/handeye_result.yaml"
    )
    handeye_path = Path(handeye_path)
    if not handeye_path.is_absolute():
        handeye_path = _PROJECT_ROOT / handeye_path
    T_tool_camera, chain = load_handeye_matrix_with_chain(str(handeye_path))
    print(f"Handeye matrix: {handeye_path} (chain={chain})")
    print(f"  Translation: ({T_tool_camera[0,3]:.4f}, {T_tool_camera[1,3]:.4f}, {T_tool_camera[2,3]:.4f}) m")
    if chain != "tool":
        print("  WARNING: 加载的是旧 link6 链标定, 建议改用 handeye_1/handeye_result.yaml (tool 链)")

    # ── 先加载模型，降低 ZED 缓冲与 SAM3 加载峰值 ─────
    pipeline = build_pipeline(args, config)
    pipeline.warmup()

    # ── 启动相机 (Orbbec Gemini 336) ─────
    from grape_stem_3d.camera_adapter import CameraAdapter

    camera_cfg = config.get("camera", {})
    camera = CameraAdapter(
        width=camera_cfg.get("color_width", 640),
        height=camera_cfg.get("color_height", 480),
        fps=camera_cfg.get("color_fps", 30),
        align_to=camera_cfg.get("align_to", "color"),
        depth_width=camera_cfg.get("depth_width", 640),
        depth_height=camera_cfg.get("depth_height", 480),
        depth_fps=camera_cfg.get("depth_fps", 30),
        alignment_mode=camera_cfg.get("alignment_mode", "hardware"),
        serial_number=camera_cfg.get("serial_number"),
    )

    try:
        intrinsics = camera.start()
    except ImportError as e:
        pipeline.close()
        print(f"ERROR: pyzed 未安装: {e}")
        print("请安装与本机 ZED SDK 匹配的 pyzed。")
        sys.exit(1)
    except Exception as e:
        pipeline.close()
        print(f"ERROR: 相机启动失败: {e}")
        sys.exit(1)

    pipeline.intrinsics = intrinsics

    # ── 初始化 ROS2 ──────────────────────────────
    rclpy.init(args=None)

    # 输出目录
    output_root = Path(args.output_dir or config.get("output", {}).get("root", "outputs"))
    if not output_root.is_absolute():
        output_root = _PROJECT_ROOT / output_root
    session_dir = output_root / time.strftime("session_%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "images").mkdir(exist_ok=True)
    (session_dir / "depth").mkdir(exist_ok=True)
    (session_dir / "json").mkdir(exist_ok=True)

    # 创建 ROS2 节点
    ros_node = GraspPipelineNode(
        pipeline=pipeline,
        camera=camera,
        T_tool_camera=T_tool_camera,
        handeye_chain=chain,
        output_dir=session_dir,
        gripper_offset_m=config.get("target_pose", {}).get(
            "gripper_offset_m", [0.0, 0.0, 0.0]
        ),
        tool_mount_rpy_deg=config.get("target_pose", {}).get(
            "tool_mount_rpy_deg", [0.0, 0.0, 0.0]
        ),
    )

    executor = MultiThreadedExecutor()
    executor.add_node(ros_node)

    # ── 相机目标发布 (分层: 视觉 → target_pose_transformer → MoveIt) ──
    from geometry_msgs.msg import PointStamped

    tp_cfg = config.get("target_pose", {})
    camera_target_pub = ros_node.create_publisher(
        PointStamped,
        tp_cfg.get("camera_target_topic", "/grape_harvest/camera_target"),
        10,
    )

    def publish_camera_target(cam_xyz):
        """检测成功后发布相机系目标 XYZ (米, camera_color_optical_frame)。"""
        msg = PointStamped()
        msg.header.stamp = ros_node.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_optical_frame"
        msg.point.x = float(cam_xyz[0])
        msg.point.y = float(cam_xyz[1])
        msg.point.z = float(cam_xyz[2])
        camera_target_pub.publish(msg)

    # ROS2 spin 线程
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # ── 机械臂客户端 (DOBOT) ──────────────────────
    from grape_stem_3d.dobot_client import DobotClient

    detect_only = args.detect_only or not robot_cfg.get("enabled", True)
    robot = None
    if not detect_only:
        robot = DobotClient(ros_node, robot_cfg)
        if robot.wait_for_services():
            robot.initialize()
        # else:
        #     注释掉 detect-only 自动降级 (2026-09-20): 服务不可用时不再静默
        #     切换到"仅检测"模式, 保留 robot 对象让后续流程以明确错误暴露问题
        #     print("WARNING: 机械臂服务不可用 → 降级为 detect-only 模式")
        #     robot = None
        #     detect_only = True
    print(f"机械臂驱动: {'已连接' if robot is not None else '禁用 (detect-only)'}")

    # ── 自动采摘配置 (仅检测+驱动模式生效, 与 d/g 按键并存) ──
    auto_cfg = config.get("auto_pick", {})
    auto_enabled = bool(auto_cfg.get("enabled", False)) and not detect_only
    auto_interval_s = float(auto_cfg.get("interval_s", 1.5))
    auto_max_radius_m = float(auto_cfg.get("max_base_radius_m", 1.2))
    auto_home_radius_m = float(auto_cfg.get("home_radius_m", 0.05))
    # 扫描门禁: 仅当机械臂回到 home 附近才扫描 (home 位姿取 robot.motion 示教值)
    _m_home = config.get("robot", {}).get("motion", {})
    auto_home_xyz = (
        [float(v) for v in _m_home["home_pose_xyz_m"]]
        if _m_home.get("home_pose_xyz_m") else None
    )

    # 剪刀手时序提示 (现场核对: 合剪延时 + 固定放果点是否已示教)
    if robot is not None:
        _mcfg = robot_cfg.get("motion", {})
        _scfg = robot_cfg.get("scissors", {})
        if _mcfg.get("send_via_topic", False):
            _tol_mm = float(_scfg.get("open_arrive_tolerance_m", 0.01)) * 1000.0
            _timeout_s = float(_scfg.get("open_arrive_timeout_s", 10.0))
            _tail = (
                f"固定放果点已配置 → 剪切点到位 (TCP+剪刀手长度, 距放果点 ≤{_tol_mm:.0f}mm) "
                f"后开剪, 超时 {_timeout_s:.0f}s 仍开剪"
                if _mcfg.get("place_position_mm")
                else "固定放果点未示教 (motion.place_position_mm=null) → 合剪后不开剪"
            )
            print(f"剪刀时序: 发布目标位姿后 "
                  f"{float(_scfg.get('close_delay_s', 2.5)):.1f}s 合剪 | {_tail}")

    # ── 检测适配器 ────────────────────────────────
    def detect_fn(rgbd_frame):
        frame_result = ros_node.detect(rgbd_frame)
        if frame_result is None:
            return ("error", None)
        status, cam_xyz, cut_info = classify_detection(frame_result)
        if status == "ok" and cam_xyz is not None:
            publish_camera_target(cam_xyz)  # 供 target_pose_transformer 消费
        return (status, cam_xyz, cut_info)

    inference_lock = threading.Lock()

    # ── PLC 通信 (可选, 55 AA TCP 客户端) ──────────
    from grape_stem_3d import plc_comm as plc
    from grape_stem_3d.drive_gate import DriveGateStatusSink

    plc_cfg = config.get("plc", {})
    plc_client = None
    plc_flags = {"start": False, "abort": False}  # PLC/HMI 命令线程写入, 主循环消费

    class _NullStatusSink:
        """状态接收方全部未启用时的空实现（手动 g/a 键走同一流程控制器）。"""
        def send_status(self, state):
            pass

    if plc_cfg.get("enabled", False):
        plc_client = plc.PlcClient(
            host=plc_cfg.get("host", "192.168.0.11"),
            port=int(plc_cfg.get("port", 20001)),
            on_command=None,  # 回调在 controller 创建后设置
            reconnect_interval_s=plc_cfg.get("reconnect_interval_s", 3.0),
            status_interval_s=plc_cfg.get("status_interval_s", 1.0),
        )
        print(f"[PLC] 通信将启用: {plc_cfg.get('host')}:{plc_cfg.get('port')}")
    else:
        print("[PLC] 通信未启用 (plc.enabled=false)，按键模式")

    # ── 上位机 (HMI) 服务 (可选, 55 AA TCP 服务器) ──
    from grape_stem_3d.hmi_server import HmiServer

    hmi_cfg = config.get("hmi", {})
    hmi_server = None
    if hmi_cfg.get("enabled", False):
        hmi_server = HmiServer(
            host=hmi_cfg.get("host", "0.0.0.0"),
            port=int(hmi_cfg.get("port", 5000)),
            on_command=None,  # 回调在 controller 创建后设置
            status_interval_s=hmi_cfg.get("status_interval_s", 1.0),
        )
        print(f"[HMI] 服务将启用: {hmi_cfg.get('host')}:{hmi_cfg.get('port')} (Box 作服务器)")
    else:
        print("[HMI] 服务未启用 (hmi.enabled=false)")

    # ── 流程控制器 ────────────────────────────────
    from grape_stem_3d.pick_flow import PickFlowController

    class _StatusFanout:
        """把流程状态帧同时转发给 PLC 客户端与 HMI 服务。"""
        def __init__(self, sinks):
            self._sinks = list(sinks)

        def send_status(self, state):
            for sink in self._sinks:
                sink.send_status(state)

    status_sinks = [s for s in (plc_client, hmi_server) if s is not None]
    status_sink = _StatusFanout(status_sinks) if status_sinks else _NullStatusSink()

    # ── Drive 信号 (0x83) 计数门: DONE×N / 无果×N → 通知 PLC 换位 ──
    # 客户端取 PLC 优先; HMI-only 部署时 DRIVE 帧上报给上位机
    drive_cfg = plc_cfg.get("drive", {})
    drive_gate = DriveGateStatusSink(status_sink, plc_client or hmi_server, drive_cfg)
    if drive_cfg.get("enabled", False):
        print(
            f"[DRIVE] 启用: 累计 DONE ×{drive_cfg.get('done_threshold', 5)} "
            f"或连续无果 ×{drive_cfg.get('no_target_threshold', 10)} → 发 0x83"
        )

    controller = PickFlowController(
        plc_client=drive_gate,
        robot=robot,
        detect_fn=detect_fn,
        T_tool_camera=T_tool_camera,
        cfg=config,
        inference_lock=inference_lock,
        detect_only=detect_only,
        logger=print,
        publish_target_pose=ros_node.publish_target_pose,
        publish_home_pose=ros_node.publish_home_pose,
    )

    # ── 检测+冻结 (d 键 / 自动采摘共用同一路径) ──────────
    def _detect_and_freeze(rgbd_frame):
        """对当前帧跑一次检测并冻结结果。

        返回 (frame_result, status, cam_xyz, base_xyz):
          frame_result: 检测帧结果 (同时更新外层变量供 overlay/保存帧使用)
          status:       classify_detection 状态 ("ok" / DETECT_* )
          cam_xyz:      剪切点 (相机系, 米)
          base_xyz:     剪切点 (base 系, 米), 无机械臂位姿时为 None
        """
        nonlocal frame_result
        with inference_lock:
            frame_result = ros_node.detect(rgbd_frame)
        status, cam_xyz, cut_info = classify_detection(frame_result)
        base_xyz = None
        if status == "ok":
            publish_camera_target(cam_xyz)  # 供 target_pose_transformer 消费
            ros_node.state = STATE_DETECTED
            # 用当前 TCP 位姿换算 base 坐标供 UI 显示 / 自动采摘范围判断
            ros_node.selected_target_base_xyz = None
            T_base_tool = None
            if robot is not None:
                T_base_tool = robot.get_T_base_tool(max_age_s=2.0)
                if T_base_tool is not None:
                    from grape_stem_3d.handeye_transform import camera_to_base_link
                    base_xyz = camera_to_base_link(
                        cam_xyz, T_tool_camera, T_base_tool)
                    ros_node.selected_target_base_xyz = base_xyz
            # 冻结检测结果: PICK 直接用冻结的坐标与位姿, 不再重新取帧检测
            controller.set_last_detection(cam_xyz, cut_info, T_base_tool)
        else:
            ros_node.state = STATE_IDLE
            ros_node.selected_target_base_xyz = None
        return frame_result, status, cam_xyz, base_xyz

    # 命令回调工厂（PLC/HMI 通道共用同一套处理逻辑, ACK 回到各自通道;
    # 回调运行在通信线程内: 只 ACK + 置标志, 重活由主循环执行）
    def _make_command_handler(sender):
        def on_command(cmd, data):
            if cmd == plc.CMD_PLC_START:
                # 主循环尚未消费上一个 START 时, 新 START 按忙处理
                if controller.is_busy or plc_flags["start"]:
                    sender.send_ack(cmd, plc.ACK_BUSY)
                else:
                    sender.send_ack(cmd, plc.ACK_OK)
                    # START 被接受 → 解除 Drive gate, 开始新点位计数
                    drive_gate.on_plc_start()
                    plc_flags["start"] = True
            elif cmd == plc.CMD_PLC_ABORT:
                sender.send_ack(cmd, plc.ACK_OK)
                plc_flags["abort"] = True
            else:
                sender.send_ack(cmd, plc.ACK_UNKNOWN_CMD)

        return on_command

    if plc_client is not None:
        plc_client.set_on_command(_make_command_handler(plc_client))
        plc_client.start()
        print(f"[PLC] 通信已启用: {plc_cfg.get('host')}:{plc_cfg.get('port')}")
    if hmi_server is not None:
        hmi_server.set_on_command(_make_command_handler(hmi_server))
        hmi_server.start()
        if hmi_server.wait_ready(2.0):
            print(f"[HMI] 服务已就绪: {hmi_cfg.get('host')}:{hmi_server.bound_port}")
        else:
            print(f"[HMI] 警告: 监听失败 (端口 {hmi_cfg.get('port')} 可能被占用)")

    # ── 显示窗口 ──────────────────────────────────
    window_name = "Grape Stem Grasp Pipeline"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    frame_count = 0
    saved_count = 0
    cam_fail_cnt = 0
    cam_fail_reported = False
    read_fail_threshold = int(camera_cfg.get("read_fail_threshold", 10))
    read_fail_sleep_s = float(camera_cfg.get("read_fail_sleep_s", 0.1))
    fps = 0.0
    last_time = time.time()
    fps_alpha = 0.1

    print("\n" + "=" * 50)
    print("Grasp Pipeline Ready.")
    print("  d — 检测果梗，计算 base_link 坐标")
    print("  u — 回初始位置（发布 /target_pose 回家点）")
    print("  g — 手动执行一次完整采摘流程")
    print("  a — 手动中止（立即急停机械臂）")
    print("  s — 保存当前帧")
    print("  q — 退出")
    print(f"  模式: {'仅检测(安全)' if detect_only else '检测+驱动'}")
    if auto_enabled:
        print(f"  AUTO — 仅在初始位置附近扫描 (每 {auto_interval_s:.1f}s), 目标距 base ≤ {auto_max_radius_m}m 自动采摘 (d/g 键仍可用)")
        if plc_client is not None:
            print("  AUTO — 需收到 PLC START 才开始扫描; Drive 发出后暂停, 等下一次 START")
    if plc_client is not None:
        print(f"  PLC: {plc_cfg.get('host')}:{plc_cfg.get('port')} (START/ABORT 触发流程)")
    if hmi_server is not None:
        hmi_port = hmi_server.bound_port or hmi_cfg.get("port")
        print(f"  HMI: {hmi_cfg.get('host')}:{hmi_port} (上位机连入收状态/发 START/ABORT)")
    print("=" * 50 + "\n")

    frame_result = None
    last_auto_check_t = 0.0
    last_auto_status = None

    try:
        while rclpy.ok():
            # ── PLC/HMI 请求处理（置顶: 相机故障时 ABORT/START 依然可消费）──
            # 【高-1】修复: 消费逻辑不依赖本帧读相机成功, 相机死时 ABORT 仍生效
            if plc_flags["abort"]:
                plc_flags["abort"] = False
                busy_before = controller.is_busy
                controller.abort()  # 内含立即 Stop 急停
                if busy_before:
                    print("  [PLC] 收到 ABORT → 急停机械臂")
                else:
                    # 空闲期 ABORT: 同样回 ABORTED, 避免 PLC 2s 看门狗误报
                    controller.report_aborted()
                    print("  [PLC] 收到 ABORT (空闲) → 上报 ABORTED")
            if plc_flags["start"]:
                plc_flags["start"] = False
                print("  [PLC] 收到 START，开始自动采摘流程")
                if controller.start_pick():
                    ros_node.state = STATE_PICKING
                else:
                    print("  [PLC] START 被拒（流程忙），ACK 已回 BUSY")

            rgbd_frame = camera.read()
            if rgbd_frame is None:
                # 【中-4】修复: 连续读失败计数 → 上报 FAIL_CAMERA + 睡眠防空转
                cam_fail_cnt += 1
                if cam_fail_cnt >= read_fail_threshold:
                    if controller.is_busy:
                        controller.notify_camera_failed()
                    elif not cam_fail_reported:
                        controller.report_camera_failed()
                        cam_fail_reported = True
                        print("  [相机] 连续读帧失败 → 已上报 FAIL_CAMERA (0x06)")
                time.sleep(read_fail_sleep_s)
                continue
            cam_fail_cnt = 0
            cam_fail_reported = False

            frame_count += 1
            rgbd_frame.frame_id = frame_count

            # 帧槽: worker 流程取用（未在流程中时只是覆盖式缓存, 开销≈0）
            controller.submit_frame(rgbd_frame)

            # 可视化
            if frame_result is not None:
                display = draw_grasp_overlay(
                    rgbd_frame.color_bgr, frame_result, ros_node, robot,
                    controller, fps=fps,
                )
            else:
                display = rgbd_frame.color_bgr.copy()

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            busy = controller.is_busy

            if key == ord("q"):
                break
            elif key == ord("d"):
                if busy:
                    print(f"  自动流程执行中，忽略按键")
                else:
                    frame_result, status, cam_xyz, base_xyz = _detect_and_freeze(rgbd_frame)
                    if status == "ok":
                        print(f"  检测成功 | camera=({cam_xyz[0]:.3f},{cam_xyz[1]:.3f},{cam_xyz[2]:.3f})m"
                              f"{'' if base_xyz is None else ' | base=('+format(base_xyz[0],'.3f')+','+format(base_xyz[1],'.3f')+','+format(base_xyz[2],'.3f')+')m'}")
                    else:
                        print(f"  检测失败: {status}")
            elif key in (ord("u"), ord("U")):
                if busy:
                    print("  流程执行中，忽略按键")
                else:
                    mcfg = config.get("robot", {}).get("motion", {})
                    home_xyz = mcfg.get("home_pose_xyz_m")
                    home_quat = mcfg.get("home_quat_xyzw")
                    if home_xyz and home_quat:
                        ros_node.publish_home_pose(home_xyz, home_quat)
                        print("  已发布回初始位置 /target_pose (u)")
                    else:
                        print("  未配置 home_pose_xyz_m / home_quat_xyzw, 忽略")
            elif key == ord("g"):
                if busy:
                    print("  流程执行中，忽略按键")
                else:
                    if controller.start_pick():
                        ros_node.state = STATE_PICKING
                        print("  手动采摘流程已启动 (g)")
                    else:
                        print("  流程启动失败（忙）")
            elif key == ord("a"):
                controller.abort()
                print("  手动中止已下发 (a): 急停机械臂")
            elif key == ord("s"):
                if frame_result is not None:
                    _save_frame(rgbd_frame, frame_result, session_dir, config)
                    saved_count += 1
                    print(f"  已保存帧 #{frame_count} ({saved_count} 总计)")

            # ── 自动采摘: START 会话内 + 初始位置附近扫描 (每 auto_interval_s
            #    检测一次); 目标在工作位点内即自动启动, 流程结束自动回 home ──
            #    门禁 drive_gate.can_pick: 开机待命 / Drive 后底盘换位中不扫描
            if auto_enabled and not controller.is_busy and drive_gate.can_pick:
                t_auto = time.time()
                if t_auto - last_auto_check_t >= auto_interval_s:
                    last_auto_check_t = t_auto
                    # 位姿门禁: 距 home 超过 home_radius_m 不扫描, 避免
                    # 放果点/中途位置误触发 (回到 home 后扫描自动恢复)
                    at_home = True
                    if robot is not None and auto_home_xyz is not None:
                        T_cur = robot.get_T_base_tool(max_age_s=2.0)
                        if T_cur is None:
                            at_home = False
                        else:
                            d_home = float(np.linalg.norm(
                                np.asarray(T_cur[:3, 3], dtype=float)
                                - np.asarray(auto_home_xyz, dtype=float)))
                            at_home = d_home <= auto_home_radius_m
                    if not at_home:
                        if last_auto_status != "not_home":
                            last_auto_status = "not_home"
                            print(f"  [AUTO] 机械臂不在初始位置, 暂停扫描 "
                                  f"(回到 home ±{auto_home_radius_m:.2f}m 后自动恢复)")
                    else:
                        frame_result, status, cam_xyz, base_xyz = _detect_and_freeze(rgbd_frame)
                        # Drive 计数: 扫描结果喂给 drive gate —
                        # 检测到目标清零, 连续 no_target 达阈值发 Drive (流程 §5.6 场景 B)
                        if status == "ok":
                            drive_gate.note_scan_result(True)
                        elif status == "no_target":
                            drive_gate.note_scan_result(False)
                        r = float(np.linalg.norm(base_xyz)) if base_xyz is not None else None
                        if status == "ok" and r is not None and r <= auto_max_radius_m:
                            if controller.start_pick():
                                ros_node.state = STATE_PICKING
                                print(f"  [AUTO] 目标距 base {r:.2f}m ≤ {auto_max_radius_m}m → 自动采摘启动")
                            else:
                                print("  [AUTO] 启动失败（忙）")
                        elif last_auto_status != status:
                            # 状态变化才打印, 避免刷屏
                            last_auto_status = status
                            if status != "ok":
                                print(f"  [AUTO] 检测失败: {status}")
                            elif r is None:
                                print("  [AUTO] 无机械臂位姿, 无法判断范围, 跳过")
                            else:
                                print(f"  [AUTO] 目标距 base {r:.2f}m > {auto_max_radius_m}m, 不触发")
            elif drive_gate.driving and last_auto_status != "driving":
                last_auto_status = "driving"
                print("[AUTO] Drive 已发送, 自动扫描暂停, 等待 PLC START")
            elif (not controller.is_busy and not drive_gate.started
                  and last_auto_status != "waiting_start"):
                last_auto_status = "waiting_start"
                print("[AUTO] 未收到 PLC START, 待命中 (收到 START 后自动扫描)")

            # UI 状态同步
            if not controller.is_busy and ros_node.state == STATE_PICKING:
                ros_node.state = STATE_IDLE

            # FPS 计算
            now = time.time()
            dt = now - last_time
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = fps_alpha * instant_fps + (1 - fps_alpha) * fps
            last_time = now

            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"达到最大帧数 ({args.max_frames})，退出")
                break

    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        if plc_client is not None:
            plc_client.stop()
        if hmi_server is not None:
            hmi_server.stop()
        camera.stop()
        pipeline.close()
        cv2.destroyAllWindows()
        executor.shutdown()
        rclpy.shutdown()
        print(f"\n处理了 {frame_count} 帧，保存了 {saved_count} 帧")
        print(f"Session: {session_dir}")


if __name__ == "__main__":
    main()
