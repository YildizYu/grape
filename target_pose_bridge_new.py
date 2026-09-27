"""MoveIt 目标位姿桥接逻辑（纯 numpy, 无 ROS 依赖, 便于单元测试）。

分层数据流:
    视觉检测 (相机系 XYZ, camera_color_optical_frame, 米)
      ↓
    T_base_tool @ T_tool_camera  (手眼变换 + 机器人当前 TCP 位姿)
      ↓
    目标 Base XYZ (base_link, 米)
      ↓
    固定采摘姿态 RPY (base 系, 度, Rz·Ry·Rx 固定轴 — 与 DOBOT 约定一致)
      ↓
    四元数 + 安全检查
      ↓
    geometry_msgs/PoseStamped → /grape_harvest/target_pose → MoveIt

姿态来源声明:
    Position comes from vision;
    orientation uses predefined harvesting orientation.
    (动态三点姿态由 pick_flow/cut_pose 另行计算, 本桥接只做固定姿态,
     避免在无三点信息时伪造视觉姿态。)
"""

from typing import Dict, Optional, Tuple

import numpy as np

from grape_stem_3d.cut_pose import rpy_degrees_to_rotation_matrix
from grape_stem_3d.handeye_transform_new import (
    camera_to_base_link,
    tool_vector_to_T_base_tool,
)

CAMERA_FRAME = "camera_color_optical_frame"
# MoveIt planning frame (2026-09-19 运行时实测 = base_link;
# C++ 侧 grape_arm_control 校验 frame 必须与 planning frame 一致)
BASE_FRAME = "base_link"


def rotation_matrix_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """3×3 旋转矩阵 → 四元数 (qx, qy, qz, qw), 与 handeye_transform 同算法。"""
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return (float(qx), float(qy), float(qz), float(qw))


def quat_norm(q: Tuple[float, float, float, float]) -> float:
    qx, qy, qz, qw = q
    return float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))


def build_target_pose(
    camera_xyz: Tuple[float, float, float],
    T_tool_camera: np.ndarray,
    T_base_tool: np.ndarray,
    fixed_rpy_deg: Tuple[float, float, float],
    workspace_max_radius_m: Optional[float] = None,
) -> Dict:
    """相机系目标点 → MoveIt 目标位姿 dict, 含全部安全检查。

    Args:
        camera_xyz: (x, y, z) camera_color_optical_frame, 单位米
        T_tool_camera: 4×4 手眼矩阵 (camera → tool TCP), 来自 handeye_result.yaml
        T_base_tool: 4×4 机器人当前 TCP 位姿 (tool → base), 来自 ToolVectorActual
        fixed_rpy_deg: (rx, ry, rz) 固定采摘姿态, base 系, 度
        workspace_max_radius_m: 目标离 base 原点最大距离 (米), None 不检查

    Returns:
        {"valid": bool, "reason": str, "base_xyz": (x,y,z) 米,
         "quat": (qx,qy,qz,qw), "target_rpy_deg": (rx,ry,rz)}
        valid=False 时 base_xyz/quat 为 None, 调用方禁止发布。
    """
    # 输入有限性检查 (视觉深度异常 → NaN/Inf)
    if not np.all(np.isfinite(camera_xyz)):
        return {"valid": False, "reason": "camera_xyz 含 NaN/Inf",
                "base_xyz": None, "quat": None, "target_rpy_deg": None}

    for name, T in (("T_tool_camera", T_tool_camera), ("T_base_tool", T_base_tool)):
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            return {"valid": False, "reason": f"{name} 非法 (shape/NaN)",
                    "base_xyz": None, "quat": None, "target_rpy_deg": None}

    if not np.all(np.isfinite(fixed_rpy_deg)) or len(fixed_rpy_deg) != 3:
        return {"valid": False, "reason": "fixed_rpy_deg 非法",
                "base_xyz": None, "quat": None, "target_rpy_deg": None}

    base_xyz = camera_to_base_link(camera_xyz, T_tool_camera, T_base_tool)

    if not np.all(np.isfinite(base_xyz)):
        return {"valid": False, "reason": "base_xyz 含 NaN/Inf",
                "base_xyz": None, "quat": None, "target_rpy_deg": None}

    if (workspace_max_radius_m is not None
            and float(np.linalg.norm(base_xyz)) > float(workspace_max_radius_m)):
        return {"valid": False,
                "reason": f"目标超出工作半径 {workspace_max_radius_m}m",
                "base_xyz": None, "quat": None, "target_rpy_deg": None}

    # 固定姿态 RPY (度) → 旋转矩阵 → 四元数 (Rz·Ry·Rx 固定轴, 与 DOBOT 一致)
    R = rpy_degrees_to_rotation_matrix(fixed_rpy_deg)
    quat = rotation_matrix_to_quat(R)

    if abs(quat_norm(quat) - 1.0) > 1e-6:
        return {"valid": False, "reason": "quaternion 模长异常",
                "base_xyz": None, "quat": None, "target_rpy_deg": None}

    return {
        "valid": True,
        "reason": "ok",
        "base_xyz": (float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2])),
        "quat": quat,
        "target_rpy_deg": tuple(float(v) for v in fixed_rpy_deg),
    }


def tool_vector_to_T(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg) -> np.ndarray:
    """ToolVectorActual 数值 → T_base_tool (m/rad), 桥接封装 (mm→m)。"""
    return tool_vector_to_T_base_tool(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg)
