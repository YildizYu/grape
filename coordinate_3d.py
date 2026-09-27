"""像素→相机三维坐标转换。

使用 MergeExp (realsense_click_xyz.py) 的标准针孔相机模型反投影公式:
  X = (u - cx) * Z / fx
  Y = (v - cy) * Z / fy
  Z = depth

坐标系: camera_color_optical_frame
  X: 右 (+X = right)
  Y: 下 (+Y = down)
  Z: 前 (+Z = forward)
单位: 米 (m)

来源: MergeExp/robot/realsense/realsense_click_xyz.py:16-35
"""

from typing import Optional

from .types import CameraIntrinsics, Point3D


def pixel_to_camera_xyz(
    intrinsics: CameraIntrinsics,
    u: float,
    v: float,
    depth_m: float,
) -> Point3D:
    """将像素坐标 + 深度转换为相机坐标系三维坐标。

    与 MergeExp pixel_to_camera_xyz() 公式完全一致。

    Args:
        intrinsics: 相机内参 (fx, fy, cx, cy)
        u: 像素列坐标 (column, x)
        v: 像素行坐标 (row, y)
        depth_m: 深度值 (米)

    Returns:
        Point3D: 相机坐标系三维点
    """
    X = (u - intrinsics.cx) * depth_m / intrinsics.fx
    Y = (v - intrinsics.cy) * depth_m / intrinsics.fy
    Z = depth_m
    return Point3D(x=X, y=Y, z=Z, unit="m", frame="camera_color_optical_frame")


def intrinsics_from_rs(rs_intrinsics) -> CameraIntrinsics:
    """从 pyrealsense2 Intrinsics 对象创建 CameraIntrinsics。

    Args:
        rs_intrinsics: pyrealsense2 intrinsics 对象
            (有 .fx, .fy, .ppx, .ppy, .width, .height 属性)

    Returns:
        CameraIntrinsics: 内部内参表示
    """
    return CameraIntrinsics(
        fx=rs_intrinsics.fx,
        fy=rs_intrinsics.fy,
        cx=rs_intrinsics.ppx,
        cy=rs_intrinsics.ppy,
        width=rs_intrinsics.width,
        height=rs_intrinsics.height,
    )


def intrinsics_from_zed(left_cam) -> CameraIntrinsics:
    """从 pyzed CalibrationParameters.left_cam 对象创建 CameraIntrinsics。

    Args:
        left_cam: pyzed 左目标定参数对象
            (有 .fx, .fy, .cx, .cy 属性和 .image_size.width/.height)

    Returns:
        CameraIntrinsics: 内部内参表示
    """
    return CameraIntrinsics(
        fx=left_cam.fx,
        fy=left_cam.fy,
        cx=left_cam.cx,
        cy=left_cam.cy,
        width=left_cam.image_size.width,
        height=left_cam.image_size.height,
    )


def intrinsics_from_matrix(
    k: "np.ndarray",
    width: int,
    height: int,
) -> CameraIntrinsics:
    """从 3×3 相机内参矩阵创建 CameraIntrinsics。

    矩阵格式:
        [fx,  0, cx]
        [ 0, fy, cy]
        [ 0,  0,  1]

    Args:
        k: 3×3 numpy 数组
        width: 图像宽度
        height: 图像高度

    Returns:
        CameraIntrinsics
    """
    import numpy as np

    k = np.asarray(k).reshape(3, 3)
    return CameraIntrinsics(
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        width=width,
        height=height,
    )
