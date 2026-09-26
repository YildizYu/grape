"""手眼标定坐标变换模块。

将 camera_color_optical_frame 中的三维坐标转换到 base_link。

坐标变换链（tool 链，当前 DOBOT CR5 + ZED 部署）:
  P_camera  →  T_tool_camera        → P_tool    (手眼标定, 相机→工具点TCP)
  P_tool    →  T_base_tool          → P_base    (机器人正运动学, 工具点TCP位姿)

即: P_base = T_base_tool @ T_tool_camera @ P_camera

其中 T_base_tool 由 /dobot_msgs_v4/msg/ToolVectorActual 话题提供
(x, y, z 单位 mm, rx, ry, rz 单位度, Rz·Ry·Rx 固定轴欧拉角),
T_tool_camera 来自 dobot_demo/handeye_result.yaml (55 组 TSAI 标定)。

历史说明（link6 链）: 旧版本使用
  P_base = T_base_link6 @ T_tool_link6 @ T_link6_camera @ P_camera
并叠加一个静态 TCP 偏移常量 T_TOOL_LINK6_DEFAULT。该常量与注释/文档
自相矛盾（实际平移 -0.02/-0.17 m vs 注释 +0.01 m），已删除——新的
tool 链中 T_base_tool 本身就是工具点 TCP 位姿，不再需要任何静态偏移。
load_handeye_matrix 仍兼容旧标定文件键名 T_link6_camera_color_optical_frame。
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml


# 默认手眼标定结果路径 → 新标定 (2026-09-06, 46 组 TSAI tool 链)
DEFAULT_HANDEYE_YAML = str(
    Path(__file__).resolve().parent.parent.parent.parent
    / "dobot_demo/handeye_result.yaml"
)

# 新旧标定文件支持的键名（按优先级）
_HANDEYE_KEYS = (
    ("T_tool_camera", "tool"),                                # 新: tool(TCP) 链
    ("T_link6_camera_color_optical_frame", "link6"),          # 旧: link6 链
)


def load_handeye_matrix(
    yaml_path: Optional[str] = None,
) -> np.ndarray:
    """加载手眼标定矩阵（兼容新旧键名）。

    Args:
        yaml_path: handeye_result.yaml 路径，默认 calibration/handeye_result.yaml

    Returns:
        4×4 numpy 数组（T_tool_camera 或 T_link6_camera_color_optical_frame）
    """
    T, _chain = load_handeye_matrix_with_chain(yaml_path)
    return T


def load_handeye_matrix_with_chain(
    yaml_path: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """加载手眼标定矩阵，并返回坐标系链类型。

    依次在 selected_solution 与顶层两级查找 T_tool_camera（tool 链）
    和 T_link6_camera_color_optical_frame（link6 链）。

    Args:
        yaml_path: handeye_result.yaml 路径，默认 calibration/handeye_result.yaml

    Returns:
        (T, chain): T 为 4×4 numpy 数组; chain ∈ {"tool", "link6"}

    Raises:
        KeyError: 两个键名都不存在
        ValueError: 矩阵 shape 不是 (4,4)
    """
    path = Path(yaml_path or DEFAULT_HANDEYE_YAML)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    selected = data.get("selected_solution", data)

    for key, chain in _HANDEYE_KEYS:
        matrix_list = selected.get(key, data.get(key))
        if matrix_list is None:
            continue
        T = np.asarray(matrix_list, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(
                f"手眼矩阵 '{key}' shape 应为 (4,4)，实际为 {T.shape}"
            )
        return T, chain

    raise KeyError(
        f"未在 {path} 中找到手眼矩阵，支持键名: "
        f"'T_tool_camera' (tool 链) 或 'T_link6_camera_color_optical_frame' (link6 链)"
    )


def euler_to_rot(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """固定轴欧拉角（度）→ 3×3 旋转矩阵。

    惯例: R = Rz(rz) @ Ry(ry) @ Rx(rx)，绕基座固定轴（外旋 XYZ），
    与 DOBOT ToolVectorActual 的 rx/ry/rz 一致。
    参考: dobot_demo/handeye_math.py 的 euler_to_rot（此处独立实现，
    避免本包依赖 dobot_demo 包）。

    Args:
        rx_deg, ry_deg, rz_deg: 绕 X/Y/Z 固定轴旋转角，单位度

    Returns:
        3×3 numpy 旋转矩阵
    """
    rx, ry, rz = map(np.radians, (rx_deg, ry_deg, rz_deg))

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ])
    Ry = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ])
    Rz = np.array([
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ])

    return Rz @ Ry @ Rx


def tool_vector_to_T_base_tool(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> np.ndarray:
    """ToolVectorActual 话题数据 → T_base_tool 4×4 矩阵。

    Args:
        x_mm, y_mm, z_mm: 工具点 TCP 在基坐标系下的位置，单位 mm
        rx_deg, ry_deg, rz_deg: 工具点姿态，Rz·Ry·Rx 固定轴欧拉角，单位度

    Returns:
        4×4 齐次变换矩阵 T_base_tool（平移单位米）
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_to_rot(rx_deg, ry_deg, rz_deg)
    T[:3, 3] = [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0]
    return T


def pose_to_matrix(
    position: Tuple[float, float, float],
    quaternion_xyzw: Tuple[float, float, float, float],
) -> np.ndarray:
    """将位置 + 四元数 (x,y,z,w) 转换为 4×4 齐次变换矩阵。

    Args:
        position: (x, y, z) 平移
        quaternion_xyzw: (qx, qy, qz, qw) 四元数

    Returns:
        4×4 变换矩阵
    """
    qx, qy, qz, qw = quaternion_xyzw
    x, y, z = position

    # 四元数 → 旋转矩阵
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[0, 1] = 2 * (qx * qy - qz * qw)
    R[0, 2] = 2 * (qx * qz + qy * qw)
    R[1, 0] = 2 * (qx * qy + qz * qw)
    R[1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[1, 2] = 2 * (qy * qz - qx * qw)
    R[2, 0] = 2 * (qx * qz - qy * qw)
    R[2, 1] = 2 * (qy * qz + qx * qw)
    R[2, 2] = 1 - 2 * (qx**2 + qy**2)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T


def ros_pose_to_matrix(pose) -> np.ndarray:
    """将 ROS2 geometry_msgs/Pose 转换为 4×4 齐次变换矩阵。

    Args:
        pose: geometry_msgs.msg.Pose 对象（有 .position 和 .orientation）

    Returns:
        4×4 变换矩阵
    """
    return pose_to_matrix(
        position=(pose.position.x, pose.position.y, pose.position.z),
        quaternion_xyzw=(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ),
    )


def camera_to_base_link(
    camera_xyz: Tuple[float, float, float],
    T_camera_tool_or_link6: np.ndarray,
    T_base_tool_or_link6: np.ndarray,
) -> Tuple[float, float, float]:
    """将 camera_color_optical_frame 坐标转换到 base_link。

    公式: P_base = T_base_tool @ T_tool_camera @ P_camera（tool 链）
          P_base = T_base_link6 @ T_link6_camera @ P_camera（旧 link6 链）

    本函数对两种链通用：第二个参数传手眼矩阵（camera → tool/link6），
    第三个参数传机器人位姿（tool/link6 → base_link）。

    Args:
        camera_xyz: (x, y, z) 在 camera_color_optical_frame 中，单位米
        T_camera_tool_or_link6: 4×4 手眼矩阵
        T_base_tool_or_link6: 4×4 机器人末端位姿（工具点 TCP / link6）

    Returns:
        (x, y, z) 在 base_link 中，单位米
    """
    P_camera = np.array(
        [camera_xyz[0], camera_xyz[1], camera_xyz[2], 1.0],
        dtype=np.float64,
    )

    P_base = T_base_tool_or_link6 @ T_camera_tool_or_link6 @ P_camera

    return (float(P_base[0]), float(P_base[1]), float(P_base[2]))


def matrix_to_ros_pose(T: np.ndarray):
    """将 4×4 变换矩阵转为 ROS2 Pose 对象。

    Args:
        T: 4×4 变换矩阵

    Returns:
        geometry_msgs.msg.Pose
    """
    from geometry_msgs.msg import Pose

    pose = Pose()
    pose.position.x = float(T[0, 3])
    pose.position.y = float(T[1, 3])
    pose.position.z = float(T[2, 3])

    # 旋转矩阵 → 四元数
    R = T[:3, :3]
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

    pose.orientation.x = float(qx)
    pose.orientation.y = float(qy)
    pose.orientation.z = float(qz)
    pose.orientation.w = float(qw)

    return pose
