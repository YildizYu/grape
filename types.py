"""共享数据类型定义。"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CameraIntrinsics:
    """相机内参。

    Attributes:
        fx, fy: 焦距 (像素)
        cx, cy: 主点 (像素)
        width, height: 图像分辨率
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480


@dataclass
class Point3D:
    """三维坐标点。

    Attributes:
        x, y, z: 坐标值
        unit: 单位
        frame: 坐标系名称
    """

    x: float
    y: float
    z: float
    unit: str = "m"
    frame: str = "camera_color_optical_frame"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "unit": self.unit,
            "frame": self.frame,
        }

    @classmethod
    def none_dict(cls) -> Dict[str, Optional[Any]]:
        return {"x": None, "y": None, "z": None, "unit": "m", "frame": None}


@dataclass
class DepthSample:
    """深度采样结果。

    Attributes:
        depth_m: 深度值 (米)，None 表示无效
        pixel_xy: 实际采样的像素坐标 (整数)
        valid_count: 邻域内有效深度像素数
        method: 采样方法描述
    """

    depth_m: Optional[float]
    pixel_xy: Tuple[int, int]
    valid_count: int
    method: str


@dataclass
class RGBDFrame:
    """RGB-D 同步帧。

    Attributes:
        color_bgr: 彩色图 (H, W, 3) uint8 BGR
        depth_data: 深度数据 — Gemini 336 为 uint16 原始深度计数
        depth_frame: 相机 SDK 深度帧对象（离线帧可为 None）
        color_frame: 相机 SDK 彩色帧对象（离线帧可为 None）
        intrinsics: 相机内参
        timestamp: 捕获时间戳
        frame_id: 帧序号
    """

    color_bgr: np.ndarray
    depth_data: np.ndarray
    depth_frame: Any = None
    color_frame: Any = None
    intrinsics: Optional[CameraIntrinsics] = None
    depth_scale: float = 0.001  # uint16 → 米的转换系数，由相机 SDK 提供
    timestamp: float = 0.0
    frame_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# 失败状态常量
STATUS_SUCCESS = "success"
STATUS_GRAPES_NOT_DETECTED = "grapes_not_detected"
STATUS_STEM_NOT_DETECTED = "stem_not_detected"
STATUS_SAM_CHECKPOINT_MISSING = "sam_checkpoint_missing"
STATUS_SAM_FAILED = "sam_failed"
STATUS_EMPTY_MASK = "empty_mask"
STATUS_CENTROID_INVALID = "centroid_invalid"
STATUS_INVALID_DEPTH = "invalid_depth"
STATUS_DEPROJECTION_FAILED = "deprojection_failed"
STATUS_CAMERA_START_FAILED = "camera_start_failed"
STATUS_FRAME_TIMEOUT = "frame_timeout"
STATUS_RGB_DEPTH_NOT_ALIGNED = "rgb_depth_not_aligned"
STATUS_SAM_NOT_AVAILABLE = "sam_not_available"
