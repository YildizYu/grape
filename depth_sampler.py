"""鲁棒深度采样模块。

基于 MergeExp click_point_base_xyz.py get_depth_m() 的 7×7 patch 中位数方法，
改编为纯 numpy 实现（去 ROS2 依赖），并增加掩膜约束和最近有效像素搜索。

核心逻辑来源: MergeExp/robot/handeye_calibration/click_point_base_xyz.py:506-615
"""

from typing import Optional, Tuple

import numpy as np

from .types import DepthSample

# 默认参数（与 MergeExp 一致）
_DEFAULT_MIN_DEPTH_M = 0.05
_DEFAULT_MAX_DEPTH_M = 5.0
_DEFAULT_PATCH_RADIUS = 3  # 7×7 patch
_DEFAULT_TOLERANCE_ABS_M = 0.02
_DEFAULT_TOLERANCE_RELATIVE = 0.03
_DEFAULT_MIN_CENTER_NEIGHBORS = 3
_DEFAULT_SEARCH_RADIUS = 12
_DEFAULT_MIN_VALID_PIXELS = 3


def sample_depth(
    depth_data: np.ndarray,
    u: float,
    v: float,
    depth_scale: float = 0.001,
    mask: Optional[np.ndarray] = None,
    use_mask_only: bool = True,
    depth_min_m: float = _DEFAULT_MIN_DEPTH_M,
    depth_max_m: float = _DEFAULT_MAX_DEPTH_M,
    patch_radius: int = _DEFAULT_PATCH_RADIUS,
    tolerance_abs_m: float = _DEFAULT_TOLERANCE_ABS_M,
    tolerance_relative: float = _DEFAULT_TOLERANCE_RELATIVE,
    min_center_neighbors: int = _DEFAULT_MIN_CENTER_NEIGHBORS,
    search_radius: int = _DEFAULT_SEARCH_RADIUS,
    estimator: str = "median",
) -> DepthSample:
    """对给定像素坐标进行鲁棒深度采样。

    策略（按优先级）:
    1. 浮点中心四舍五入 → (ui, vi)
    2. 检查 (ui, vi) 是否在掩膜内（如提供）且深度有效
    3. 若无效: 在掩膜内搜索距离最近的有效像素
    4. 收集邻域 patch 内的有效深度值
    5. 计算中位数深度（中心参照容忍度）
    6. 所有值无效时返回 None

    Args:
        depth_data: 深度数据 (uint16 毫米 或 float32 米)
        u: 像素列坐标 (浮点)
        v: 像素行坐标 (浮点)
        depth_scale: uint16 到米的转换系数 (默认 0.001 mm→m)
        mask: 二值掩膜 (uint8, 0/255)，前景=255。为 None 则不使用掩膜约束
        use_mask_only: True 时只收集掩膜内像素的深度
        depth_min_m: 最小有效深度 (米)
        depth_max_m: 最大有效深度 (米)
        patch_radius: 中值滤波半径 (默认 3 → 7×7)
        tolerance_abs_m: 中心参照容忍度绝对值 (米)
        tolerance_relative: 中心参照容忍度相对系数
        min_center_neighbors: 中心参照最少近邻数
        search_radius: 最近有效像素搜索半径
        estimator: 估计器 ("median" / "mean")

    Returns:
        DepthSample: 深度采样结果
    """
    h, w = depth_data.shape[:2]

    # 1. 浮点中心四舍五入
    ui = int(round(u))
    vi = int(round(v))

    # 2. 转为米的浮点数组
    if depth_data.dtype == np.uint16:
        depth_m = depth_data.astype(np.float32) * depth_scale
    else:
        depth_m = depth_data.astype(np.float32)

    # 创建有效深度 mask
    valid_depth = np.isfinite(depth_m) & (depth_m >= depth_min_m) & (depth_m <= depth_max_m)

    # 组合掩膜: mask 前景 + 有效深度
    if mask is not None and use_mask_only:
        mask_fg = mask > 0
        combined_mask = valid_depth & mask_fg
    else:
        combined_mask = valid_depth

    # 3. 中心像素检查
    center_valid = False
    if 0 <= ui < w and 0 <= vi < h:
        if combined_mask[vi, ui]:
            center_valid = True
            sample_pixel = (ui, vi)
        else:
            # 尝试找掩膜内最近有效像素
            nearest = _find_nearest_valid_pixel(combined_mask, ui, vi, search_radius, w, h)
            if nearest is not None:
                ui, vi = nearest
                center_valid = True
                sample_pixel = (ui, vi)
    else:
        # 中心超出边界，找最近有效像素
        ui = max(0, min(w - 1, ui))
        vi = max(0, min(h - 1, vi))
        nearest = _find_nearest_valid_pixel(combined_mask, ui, vi, search_radius, w, h)
        if nearest is not None:
            ui, vi = nearest
            center_valid = True
            sample_pixel = (ui, vi)

    if not center_valid:
        return DepthSample(
            depth_m=None,
            pixel_xy=(int(round(u)), int(round(v))),
            valid_count=0,
            method="no_valid_pixel_found",
        )

    # 4. 提取 patch 内的深度值
    patch = _extract_patch(depth_m, combined_mask, ui, vi, patch_radius, w, h)
    if patch.size < 1:
        return DepthSample(
            depth_m=None,
            pixel_xy=sample_pixel,
            valid_count=0,
            method="patch_empty",
        )

    # 5. 中心参照容忍度过滤（与 MergeExp 逻辑一致）
    center_val = depth_m[vi, ui]
    tolerance = max(tolerance_abs_m, center_val * tolerance_relative)
    close_to_center = patch[np.abs(patch - center_val) <= tolerance]

    if close_to_center.size >= min_center_neighbors:
        if estimator == "median":
            result_m = float(np.median(close_to_center))
        else:
            result_m = float(np.mean(close_to_center))
        method = "center_median" if estimator == "median" else "center_mean"
    elif patch.size >= 1:
        # 退化为全 patch 估计
        if estimator == "median":
            result_m = float(np.median(patch))
        else:
            result_m = float(np.mean(patch))
        method = "patch_median" if estimator == "median" else "patch_mean"
    else:
        return DepthSample(
            depth_m=None,
            pixel_xy=sample_pixel,
            valid_count=0,
            method="no_valid_depth_in_patch",
        )

    return DepthSample(
        depth_m=result_m,
        pixel_xy=sample_pixel,
        valid_count=len(patch),
        method=method,
    )


def _extract_patch(
    depth_m: np.ndarray,
    combined_mask: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    w: int,
    h: int,
) -> np.ndarray:
    """从深度图中提取 patch 内的有效深度值。"""
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius + 1)
    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius + 1)

    region_mask = combined_mask[y1:y2, x1:x2]
    region_depth = depth_m[y1:y2, x1:x2]

    return region_depth[region_mask]


def _find_nearest_valid_pixel(
    combined_mask: np.ndarray,
    start_u: int,
    start_v: int,
    search_radius: int,
    w: int,
    h: int,
) -> Optional[Tuple[int, int]]:
    """在搜索半径内找到距离 (start_u, start_v) 最近的有效像素。

    使用螺旋搜索（从小到大半径依次扫描每个环上的像素）。
    """
    # 先检查中心（已被调用方检查过，但防御性编程）
    if 0 <= start_u < w and 0 <= start_v < h and combined_mask[start_v, start_u]:
        return (start_u, start_v)

    for r in range(1, search_radius + 1):
        # 检查边长为 2r+1 的正方形环
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                # 跳过内层（已被之前半径检查过）
                if max(abs(dx), abs(dy)) != r:
                    continue
                nx = start_u + dx
                ny = start_v + dy
                if 0 <= nx < w and 0 <= ny < h and combined_mask[ny, nx]:
                    return (nx, ny)

    return None
