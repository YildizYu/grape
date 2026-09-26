"""实时可视化模块。

在相机实时帧上绘制检测结果 overlay。
颜色约定:
  绿色       — 葡萄串检测框
  蓝色       — 葡萄扩展 ROI
  红色       — 果梗 YOLO 框
  半透明紫色 — SAM 果梗掩膜
  黄色十字   — 果梗二维几何中心
  白色圆点   — 实际深度采样像素
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── BGR 颜色常量 ───────────────────────────────────────
COLOR_GRAPE = (0, 255, 0)         # 绿色
COLOR_ROI = (255, 0, 0)            # 蓝色
COLOR_STEM = (0, 0, 255)          # 红色
COLOR_CENTROID = (0, 255, 255)    # 黄色
COLOR_DEPTH_SAMPLE = (255, 255, 255)  # 白色
COLOR_MASK = (255, 0, 255)        # 紫色
COLOR_FPS = (255, 255, 255)       # 白色
COLOR_STATUS_OK = (0, 255, 0)     # 绿色
COLOR_STATUS_FAIL = (0, 0, 255)   # 红色

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICKNESS = 1
LINE_THICKNESS = 2


def draw_overlay(
    color_bgr: np.ndarray,
    frame_result: Dict[str, Any],
    fps: Optional[float] = None,
    show_depth: bool = False,
    depth_data: Optional[np.ndarray] = None,
    depth_min_m: float = 0.05,
    depth_max_m: float = 5.0,
) -> np.ndarray:
    """在帧上绘制所有检测和三维结果。

    Args:
        color_bgr: 原始彩色帧
        frame_result: RealtimeGrapeStemPipeline.process_frame() 的输出
        fps: 当前 FPS（可选）
        show_depth: 是否显示深度伪彩色叠加
        depth_data: 深度数组（仅 show_depth=True 时需要）
        depth_min_m: 深度伪彩色最小值 (米)
        depth_max_m: 深度伪彩色最大值 (米)

    Returns:
        带 overlay 的 BGR 图像
    """
    output = color_bgr.copy()
    h, w = output.shape[:2]

    for grape in frame_result.get("grapes", []):
        grape_id = grape.get("grape_id", "?")

        # 测试模式透传的全帧二值掩膜，仅用于显示和保存。
        mask = grape.get("_segmentation_mask_full")
        if isinstance(mask, np.ndarray) and mask.shape == output.shape[:2]:
            mask_overlay = output.copy()
            mask_overlay[mask > 0] = COLOR_MASK
            output = cv2.addWeighted(output, 0.65, mask_overlay, 0.35, 0)

        # ── 葡萄检测框 (绿色) ─────────────
        grape_bbox = grape.get("grape_bbox_xyxy")
        if grape_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in grape_bbox]
            cv2.rectangle(output, (x1, y1), (x2, y2), COLOR_GRAPE, LINE_THICKNESS)
            label = f"G{grape_id}: {grape.get('grape_confidence', 0):.2f}"
            _put_text(output, label, (x1, y1 - 5), COLOR_GRAPE)

        # ── ROI 框 (蓝色) ─────────────────
        roi_bbox = grape.get("roi_bbox_xyxy")
        if roi_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in roi_bbox]
            cv2.rectangle(output, (x1, y1), (x2, y2), COLOR_ROI, 1)

        # ── 果梗框 (红色) ─────────────────
        stem_bbox = grape.get("stem_bbox_global_xyxy")
        if stem_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in stem_bbox]
            cv2.rectangle(output, (x1, y1), (x2, y2), COLOR_STEM, LINE_THICKNESS)
            cv2.putText(output, f"stem {grape.get('stem_confidence', 0):.2f}",
                        (x1, y1 - 5), FONT, FONT_SCALE, COLOR_STEM, FONT_THICKNESS)

        # ── 二维中心 (黄色十字) ─────────────
        centroid_xy = grape.get("peduncle_mask_centroid_xy")
        target_type = grape.get("target_type")
        if centroid_xy is not None:
            cx, cy = int(round(centroid_xy[0])), int(round(centroid_xy[1]))
            # 十字线
            cross_size = 12
            cv2.line(output, (cx - cross_size, cy), (cx + cross_size, cy), COLOR_CENTROID, 2)
            cv2.line(output, (cx, cy - cross_size), (cx, cy + cross_size), COLOR_CENTROID, 2)
            # 中心标记
            marker = "M" if target_type == "mask_centroid" else "B"
            cv2.putText(output, marker, (cx + 8, cy - 8), FONT, 0.5, COLOR_CENTROID, 1)

        # ── 果梗三点 (新链 keypoint: 方向线 + P1/CUT/P3) ──
        keypoints = grape.get("keypoints_global_xy")
        if keypoints:
            p1 = keypoints.get("point_1")
            cut = keypoints.get("cut_point")
            p3 = keypoints.get("point_3")
            if p1 is not None and p3 is not None:
                cv2.line(output, (int(p1[0]), int(p1[1])),
                         (int(p3[0]), int(p3[1])), COLOR_CENTROID, 2)
            for xy, color, label in (
                (p1, (255, 0, 0), "P1"),
                (cut, (0, 0, 255), "CUT"),
                (p3, (0, 255, 255), "P3"),
            ):
                if xy is None:
                    continue
                px, py = int(xy[0]), int(xy[1])
                cv2.circle(output, (px, py), 5, (255, 255, 255), -1)
                cv2.circle(output, (px, py), 3, color, -1)
                cv2.putText(output, label, (px + 7, py - 7), FONT, 0.45, color, 1)

        # ── 深度采样点 (白色圆点) ────────────
        depth_pixel = grape.get("depth_sample_pixel_xy")
        if depth_pixel is not None:
            dx, dy = depth_pixel
            cv2.circle(output, (dx, dy), 5, COLOR_DEPTH_SAMPLE, 1)
            cv2.circle(output, (dx, dy), 2, COLOR_DEPTH_SAMPLE, -1)

        # ── 文字信息 ──────────────────────
        info_lines = _build_info_lines(grape, grape_id)
        _draw_info_box(output, info_lines, grape_id, grape.get("status"))

    # ── 帧级状态 ──────────────────────────────
    status = frame_result.get("status", "unknown")
    n_grapes = len(frame_result.get("grapes", []))
    status_color = COLOR_STATUS_OK if status == "success" else COLOR_STATUS_FAIL

    # 顶部状态栏
    status_text = f"Status: {status} | Grapes: {n_grapes}"
    if fps is not None:
        status_text += f" | FPS: {fps:.1f}"
    cv2.putText(output, status_text, (10, 25), FONT, 0.6, status_color, 2)

    # 深度伪彩色叠加
    if show_depth and depth_data is not None:
        output = _blend_depth_overlay(output, depth_data, depth_min_m, depth_max_m)

    return output


def _build_info_lines(grape: Dict, grape_id: int) -> List[str]:
    """构建葡萄信息文本行。"""
    lines = [
        f"Grape {grape_id}",
    ]

    centroid = grape.get("peduncle_mask_centroid_xy")
    if centroid is not None:
        lines.append(f"2D: ({centroid[0]:.0f}, {centroid[1]:.0f})")

    depth_m = grape.get("depth_value")
    if depth_m is not None:
        lines.append(f"Z: {depth_m:.3f}m")

    xyz = grape.get("peduncle_centroid_camera_xyz")
    if xyz and xyz.get("x") is not None:
        lines.append(f"3D: ({xyz['x']:.3f}, {xyz['y']:.3f}, {xyz['z']:.3f})m")

    target_type = grape.get("target_type", "")
    if target_type == "stem_bbox_center":
        lines.append("(bbox center)")
    elif target_type == "mask_centroid":
        lines.append("(mask centroid)")

    status = grape.get("status")
    if status and status != "success":
        lines.append(f"!{status}")

    return lines


def _draw_info_box(
    img: np.ndarray,
    lines: List[str],
    grape_id: int,
    status: str,
) -> None:
    """在图像右上角绘制信息框。"""
    h, w = img.shape[:2]
    x_start = w - 260
    y_offset = 80 + grape_id * 120

    color = COLOR_STATUS_FAIL if status != "success" else COLOR_GRAPE

    for i, line in enumerate(lines):
        y = y_offset + i * 22
        if y > h - 20:
            break
        if y < 30:
            continue
        text_color = color if i == 0 else (200, 200, 200)
        cv2.putText(img, line, (x_start, y), FONT, 0.42, text_color, 1)


def _blend_depth_overlay(
    color_bgr: np.ndarray,
    depth_data: np.ndarray,
    min_m: float,
    max_m: float,
    alpha: float = 0.3,
) -> np.ndarray:
    """叠加深度伪彩色图。"""
    if depth_data is None or depth_data.size == 0:
        return color_bgr

    # 转为米
    if depth_data.dtype == np.uint16:
        depth_m = depth_data.astype(np.float32) * 0.001
    else:
        depth_m = depth_data.astype(np.float32)

    # 裁剪并归一化
    depth_m = np.clip(depth_m, min_m, max_m)
    depth_norm = ((depth_m - min_m) / max(max_m - min_m, 1e-6) * 255).astype(np.uint8)

    # 伪彩色 (JET)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    # 混合
    if color_bgr.shape[:2] != depth_color.shape[:2]:
        depth_color = cv2.resize(depth_color, (color_bgr.shape[1], color_bgr.shape[0]))

    blended = cv2.addWeighted(color_bgr, 1 - alpha, depth_color, alpha, 0)
    return blended


def _put_text(img, text, pos, color, scale=FONT_SCALE, thickness=FONT_THICKNESS):
    """放置文本（带黑色轮廓）。"""
    cv2.putText(img, text, pos, FONT, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, FONT, scale, color, thickness)

