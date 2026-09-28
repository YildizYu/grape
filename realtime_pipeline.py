"""实时葡萄果梗三维定位 Pipeline 编排器。

集成：
- 正式链：葡萄 YOLO → 果梗 YOLO → SAM3 → mask 骨架 P1/CUT/P3
- 本模块的深度采样和三维坐标转换
- 结果收集和状态管理
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np

from .types import (
    CameraIntrinsics,
    DepthSample,
    Point3D,
    RGBDFrame,
    STATUS_SUCCESS,
    STATUS_GRAPES_NOT_DETECTED,
    STATUS_STEM_NOT_DETECTED,
    STATUS_INVALID_DEPTH,
    STATUS_SAM_NOT_AVAILABLE,
)
from .coordinate_3d import pixel_to_camera_xyz
from .depth_sampler import sample_depth, DepthSample
from .vision_chain_common import expanded_grape_roi, postprocess_segmentation_mask


def largest_mask_component(mask, min_area: int = 0):
    """返回掩膜最大连通域 (bool 数组); 无有效域返回 None。

    用途: 串顶兜底剪前滤掉 SAM3 误分割的叶片小碎片。
    """
    import cv2

    if mask is None:
        return None
    binary = np.asarray(mask) > 0
    if not np.any(binary):
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    best = None
    best_area = int(min_area)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= best_area:
            best_area = area
            best = labels == index
    return best


def bunch_top_cut_point(mask, keep_ratio: float, band_half_height_px: int = 3):
    """葡萄串掩膜顶部兜底剪点 (ROI-local 像素)。

    剪点 y = 掩膜顶缘 + keep_ratio × 掩膜高度 (保留顶部几个葡萄);
    剪点 x = 剪点水平带内掩膜像素的 x 中位数 (穗轴位置)。

    Returns: (cut_x, cut_y) float; 掩膜无效返回 None。
    """
    if mask is None:
        return None
    binary = np.asarray(mask) > 0
    if not np.any(binary):
        return None
    ys, xs = np.nonzero(binary)
    top_y, bottom_y = int(ys.min()), int(ys.max())
    ratio = min(max(float(keep_ratio), 0.0), 1.0)
    cut_y = top_y + ratio * (bottom_y - top_y)
    cut_yi = int(round(cut_y))
    y0 = max(0, cut_yi - int(band_half_height_px))
    y1 = min(binary.shape[0], cut_yi + int(band_half_height_px) + 1)
    band_xs = np.nonzero(binary[y0:y1, :])[1]
    if len(band_xs) == 0:
        return None
    return float(np.median(band_xs)), float(cut_y)


class RealtimeGrapeStemPipeline:
    """实时三维定位 Pipeline。

    启动时加载所有模型一次；每帧调用 process_frame()。
    """

    def __init__(
        self,
        grape_detector,
        stem_detector,
        stem_selector,
        intrinsics: Optional[CameraIntrinsics] = None,
        sam_segmenter=None,
        keypoint_predictor=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            grape_detector: GrapeDetector 实例
            stem_detector: StemDetector 实例
            stem_selector: StemSelector 实例
            intrinsics: 相机内参（如果无相机则为 None）
            sam_segmenter: SAM3 worker 客户端（正式链必需）
            keypoint_predictor: 历史关键点 worker（仅兼容旧链；正式 SAM3 链不使用）
            config: 融合 pipeline 配置字典
        """
        self.grape_detector = grape_detector
        self.stem_detector = stem_detector
        self.stem_selector = stem_selector
        self.intrinsics = intrinsics
        self.sam_segmenter = sam_segmenter
        self.keypoint_predictor = keypoint_predictor
        self.config = config or {}

        # 从配置提取参数
        self._roi_config = self.config.get("roi", {})
        self._depth_config = self.config.get("depth", {})
        self._runtime_config = self.config.get("runtime", {})
        self._detect_every_n = self._runtime_config.get("detect_every_n_frames", 1)
        # 正式视觉链是 sam3_keypoint。旧值仅保留给历史测试，不作为默认路径。
        self._chain = self.config.get("vision", {}).get("chain", "sam3_keypoint")
        self._keypoint_config = self.config.get(
            "mask_postprocess", self.config.get("keypoint", {})
        )
        # 串顶兜底剪: 果梗链路失败时用葡萄掩膜顶部区域估算剪切点
        self._bunch_cfg = self.config.get("bunch_top_fallback", {})
        self._bunch_enabled = bool(self._bunch_cfg.get("enabled", False))

        self._frame_count = 0
        self._last_grape_results = []  # 缓存上一次检测结果（用于跳帧模式）

    def warmup(self):
        """CUDA 预热 — 运行一次空推理。"""
        print("Warming up models...")
        # SAM3 has the highest startup memory peak. Load it before YOLO and
        # before callers open native-resolution camera buffers.
        if self._chain == "sam3_keypoint" and self.sam_segmenter is not None:
            self.sam_segmenter.warmup()
            print("  SAM3 worker warmed up.")

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            _ = self.grape_detector.detect(dummy)
            print("  Grape detector warmed up.")
        except Exception as e:
            print(f"  Grape detector warmup: {e}")

        # 不执行 stem warmup — 需要真实 ROI 输入。

    def process_frame(
        self,
        rgbd_frame: RGBDFrame,
        image_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """处理一帧 RGB-D 数据。

        Args:
            rgbd_frame: RGBDFrame 同步帧
            image_name: 帧名称（用于日志和保存）

        Returns:
            完整帧结果字典
        """
        self._frame_count += 1
        color_bgr = rgbd_frame.color_bgr
        h, w = color_bgr.shape[:2]

        # 更新内参（如果 camera adapter 提供）
        if rgbd_frame.intrinsics is not None:
            self.intrinsics = rgbd_frame.intrinsics

        # 帧级结果
        frame_result = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(rgbd_frame.timestamp)),
            "frame_id": rgbd_frame.frame_id,
            "image_width": w,
            "image_height": h,
            "status": STATUS_SUCCESS,
            "camera": {
                "model": rgbd_frame.metadata.get("camera_model", "ZED X Mini"),
                "serial_number": rgbd_frame.metadata.get("camera_serial"),
                "alignment": rgbd_frame.metadata.get("alignment"),
                "fx": self.intrinsics.fx if self.intrinsics else None,
                "fy": self.intrinsics.fy if self.intrinsics else None,
                "cx": self.intrinsics.cx if self.intrinsics else None,
                "cy": self.intrinsics.cy if self.intrinsics else None,
                "frame": "camera_color_optical_frame",
                "units": "m",
            },
            "grapes": [],
        }

        # ── 1. 葡萄检测 ──────────────────────────────
        try:
            grape_detections = self.grape_detector.detect(color_bgr)
        except Exception as e:
            frame_result["status"] = "grape_detection_error"
            frame_result["error"] = str(e)
            return frame_result

        if not grape_detections:
            frame_result["status"] = STATUS_GRAPES_NOT_DETECTED
            return frame_result

        # ── 2. 逐葡萄处理 ────────────────────────────
        # 新链 (keypoint): 先批量推理全部裁剪图, 避免逐串事务开销
        crop_packets = None
        if self._chain == "keypoint" and self.keypoint_predictor is not None:
            crop_packets = self._prepare_keypoint_batch(
                grape_detections, color_bgr, w, h
            )

        for gi, grape_det in enumerate(grape_detections):
            if crop_packets is not None:
                packet = crop_packets.get(gi)
                crop_bbox = packet[0] if packet is not None else None
                crop_result = packet[1] if packet is not None else None
                grape_result = self._process_single_grape_keypoint(
                    grape_det=grape_det,
                    grape_id=gi,
                    crop_bbox=crop_bbox,
                    crop_result=crop_result,
                    rgbd_frame=rgbd_frame,
                    img_w=w,
                    img_h=h,
                )
            else:
                grape_result = self._process_single_grape(
                    grape_det=grape_det,
                    grape_id=gi,
                    color_bgr=color_bgr,
                    rgbd_frame=rgbd_frame,
                    img_w=w,
                    img_h=h,
                )
            frame_result["grapes"].append(grape_result)

        # 帧级状态
        if not frame_result["grapes"]:
            frame_result["status"] = STATUS_GRAPES_NOT_DETECTED
        elif all(g["status"] != STATUS_SUCCESS for g in frame_result["grapes"]):
            # 全部失败
            statuses = set(g["status"] for g in frame_result["grapes"])
            frame_result["status"] = list(statuses)[0] if len(statuses) == 1 else "partial_failure"

        return frame_result

    def _process_single_grape(
        self,
        grape_det: Dict,
        grape_id: int,
        color_bgr: np.ndarray,
        rgbd_frame: RGBDFrame,
        img_w: int,
        img_h: int,
    ) -> Dict[str, Any]:
        """处理单个葡萄串：ROI → 果梗 YOLO → 分割及后处理。"""
        grape_bbox = grape_det["bbox_xyxy"]  # (x1, y1, x2, y2)
        grape_conf = grape_det["confidence"]
        gx1, gy1, gx2, gy2 = grape_bbox

        # 基础结果
        result = {
            "grape_id": grape_id,
            "grape_bbox_xyxy": grape_bbox,
            "grape_confidence": grape_conf,
            "status": STATUS_SUCCESS,
            "roi_bbox_xyxy": None,
            "stem_bbox_roi_xyxy": None,
            "stem_bbox_global_xyxy": None,
            "stem_confidence": None,
            "peduncle_mask_centroid_roi_xy": None,
            "peduncle_mask_centroid_xy": None,
            "target_type": None,  # "mask_centroid" | "stem_bbox_center" | None
            "depth_sample_pixel_xy": None,
            "depth_value": None,
            "depth_sampling_method": None,
            "peduncle_centroid_camera_xyz": None,
            "coordinate_unit": "m",
            "coordinate_frame": "camera_color_optical_frame",
        }

        # ── ROI 扩展 ──────────────────────────────────
        roi_bbox = expanded_grape_roi(
            grape_bbox, img_w, img_h, self._roi_config
        )
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_bbox
        result["roi_bbox_xyxy"] = roi_bbox

        # 裁剪 ROI 图像
        roi_image = color_bgr[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi_image.size == 0:
            result["status"] = "roi_empty"
            return result

        # ── 果梗检测 ──────────────────────────────
        try:
            stem_detections = self.stem_detector.detect(roi_image)
        except Exception as e:
            result["status"] = "stem_detection_error"
            result["error"] = str(e)
            return result

        # ── 果梗选择 ──────────────────────────────
        # StemSelector 期望 ROI-local 坐标的 grape_bbox
        grape_local = (gx1 - roi_x1, gy1 - roi_y1, gx2 - roi_x1, gy2 - roi_y1)
        selection = self.stem_selector.select(stem_detections, grape_local, roi_bbox)

        if selection["status"] != "stem_detected":
            # 串顶兜底剪: 果梗未检出时用葡萄掩膜顶部区域估算剪切点
            if self._bunch_top_fallback(
                grape_result=result,
                crop_bbox=roi_bbox,
                grape_local_bbox=grape_local,
                rgbd_frame=rgbd_frame,
                img_w=img_w,
                img_h=img_h,
            ):
                return result
            result["status"] = STATUS_STEM_NOT_DETECTED
            return result

        stem_bbox_local = selection["best_stem_bbox"]
        stem_conf = selection["best_confidence"]
        result["stem_bbox_roi_xyxy"] = stem_bbox_local

        # 局部 → 全局坐标
        slx1, sly1, slx2, sly2 = stem_bbox_local
        stem_global = (roi_x1 + slx1, roi_y1 + sly1, roi_x1 + slx2, roi_y1 + sly2)
        result["stem_bbox_global_xyxy"] = stem_global
        result["stem_confidence"] = stem_conf

        if self._chain == "sam3_keypoint":
            return self._process_sam3_keypoint(
                grape_det=grape_det,
                grape_id=grape_id,
                roi_bbox=roi_bbox,
                roi_image=roi_image,
                stem_bbox_local=stem_bbox_local,
                stem_bbox_global=stem_global,
                stem_confidence=stem_conf,
                rgbd_frame=rgbd_frame,
                img_w=img_w,
                img_h=img_h,
            )

        # ── SAM 分割 (可选) ────────────────────────
        mask = None
        centroid_roi_xy = None
        target_type = "stem_bbox_center"  # 默认 fallback

        if self.sam_segmenter is not None:
            try:
                sam_result = self.sam_segmenter.segment(roi_image, stem_bbox_local)
                if sam_result["status"] == "success" and sam_result["mask"] is not None:
                    mask = sam_result["mask"]
            except Exception:
                pass  # SAM 失败优雅降级

        if mask is not None and mask.sum() > 0:
            # 用 SAM 掩膜计算质心
            from grape_stem.mask_utils import compute_centroid

            centroid_roi_xy = compute_centroid(mask)
            if centroid_roi_xy is not None:
                target_type = "mask_centroid"
                # 将 ROI-local mask 嵌入到全帧 mask 中，
                # 否则 sample_depth() 中 mask.shape != depth_data.shape 会崩溃
                full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                full_mask[roi_y1:roi_y2, roi_x1:roi_x2] = mask
                mask = full_mask  # 替换为全帧尺寸的 mask

        # 如果 SAM 掩膜质心不可用，用 stem bbox 中心
        if centroid_roi_xy is None:
            c_local_x = (slx1 + slx2) / 2
            c_local_y = (sly1 + sly2) / 2
            centroid_roi_xy = (c_local_x, c_local_y)

        local_cx, local_cy = centroid_roi_xy
        result["peduncle_mask_centroid_roi_xy"] = (local_cx, local_cy)

        # 转为全局像素坐标
        global_cx = roi_x1 + local_cx
        global_cy = roi_y1 + local_cy
        result["peduncle_mask_centroid_xy"] = (global_cx, global_cy)
        result["target_type"] = target_type

        # ── 深度采样 ──────────────────────────────
        depth_data = rgbd_frame.depth_data
        if depth_data is not None and self.intrinsics is not None:
            depth_sample = sample_depth(
                depth_data=depth_data,
                u=global_cx,
                v=global_cy,
                depth_scale=rgbd_frame.depth_scale,
                mask=mask if target_type == "mask_centroid" else None,
                use_mask_only=self._depth_config.get("use_mask_only", True),
                depth_min_m=self._depth_config.get("min_m", 0.05),
                depth_max_m=self._depth_config.get("max_m", 5.0),
                patch_radius=self._depth_config.get("patch_radius", 3),
                tolerance_abs_m=self._depth_config.get("tolerance_abs_m", 0.02),
                tolerance_relative=self._depth_config.get("tolerance_relative", 0.03),
                min_center_neighbors=self._depth_config.get("min_center_neighbors", 3),
                search_radius=self._depth_config.get("search_radius", 12),
                estimator=self._depth_config.get("estimator", "median"),
            )

            result["depth_sample_pixel_xy"] = depth_sample.pixel_xy
            result["depth_value"] = depth_sample.depth_m
            result["depth_sampling_method"] = depth_sample.method

            # ── 三维坐标 ──────────────────────────
            if depth_sample.depth_m is not None:
                # 使用浮点质心坐标（global_cx, global_cy）做反投影，
                # 而非深度采样的舍入整数像素，保留亚像素精度。
                # 深度值仍从最近整数像素读取（硬件限制），
                # 但反投影公式使用更精确的浮点质心位置。
                point3d = pixel_to_camera_xyz(
                    self.intrinsics,
                    global_cx,
                    global_cy,
                    depth_sample.depth_m,
                )
                result["peduncle_centroid_camera_xyz"] = point3d.to_dict()
            else:
                result["peduncle_centroid_camera_xyz"] = None
                result["status"] = STATUS_INVALID_DEPTH
        else:
            result["status"] = STATUS_INVALID_DEPTH

        return result

    def _process_sam3_keypoint(
        self,
        grape_det,
        grape_id,
        roi_bbox,
        roi_image,
        stem_bbox_local,
        stem_bbox_global,
        stem_confidence,
        rgbd_frame,
        img_w,
        img_h,
    ):
        """SAM3 mask → shared skeleton post-process → P1/CUT/P3 → 3D."""
        if self.sam_segmenter is None:
            crop_result = {"error": "SAM3 production worker is unavailable"}
            sam_result = {"status": "sam_not_available"}
        else:
            try:
                sam_result = self.sam_segmenter.segment(roi_image, stem_bbox_local)
            except Exception as exc:
                sam_result = {
                    "status": "sam_exception",
                    "mask": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            crop_result = postprocess_segmentation_mask(
                sam_result.get("mask"),
                source_width=roi_image.shape[1],
                source_height=roi_image.shape[0],
                stem_bbox_xyxy=stem_bbox_local,
                config=self.config,
            )
            if sam_result.get("status") != "success":
                crop_result["error"] = (
                    f"SAM3 {sam_result.get('status')}: "
                    f"{sam_result.get('error') or 'no usable mask'}"
                )

        result = self._process_single_grape_keypoint(
            grape_det=grape_det,
            grape_id=grape_id,
            crop_bbox=roi_bbox,
            crop_result=crop_result,
            rgbd_frame=rgbd_frame,
            img_w=img_w,
            img_h=img_h,
        )
        result["stem_bbox_roi_xyxy"] = stem_bbox_local
        result["stem_bbox_global_xyxy"] = stem_bbox_global
        result["stem_confidence"] = stem_confidence
        result["segmentation_model"] = "sam3"
        result["sam_status"] = sam_result.get("status")
        result["sam_error"] = sam_result.get("error")
        result["sam_score"] = sam_result.get("score")
        return result

    # ── 历史关键点链（正式 sam3_keypoint 不进入此分支） ────
    def _keypoint_crop(self, grape_bbox, img_w, img_h, color_bgr):
        """按历史 keypoint 链配置裁剪葡萄图。

        Returns: (crop_bbox, crop_image) 或 (None, None)（裁剪为空）
        """
        gx1, gy1, gx2, gy2 = grape_bbox
        gw = gx2 - gx1
        gh = gy2 - gy1
        el = self._keypoint_config.get("crop_expand_left", 0.05)
        er = self._keypoint_config.get("crop_expand_right", 0.05)
        et = self._keypoint_config.get("crop_expand_top", 0.30)
        eb = self._keypoint_config.get("crop_expand_bottom", 0.10)

        roi_x1 = max(0, int(gx1 - el * gw))
        roi_y1 = max(0, int(gy1 - et * gh))
        roi_x2 = min(img_w, int(gx2 + er * gw))
        roi_y2 = min(img_h, int(gy2 + eb * gh))
        crop_bbox = (roi_x1, roi_y1, roi_x2, roi_y2)

        crop_image = color_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
        if crop_image.size == 0:
            return None, None
        return crop_bbox, crop_image

    def _prepare_keypoint_batch(self, grape_detections, color_bgr, img_w, img_h):
        """批量推理本帧全部葡萄裁剪图的果梗三点。

        Returns: {grape_id: (crop_bbox, crop_result)}; 推理异常返回 {}。
        """
        entries = []  # (grape_id, crop_bbox, crop_image)
        for gi, grape_det in enumerate(grape_detections):
            crop_bbox, crop_image = self._keypoint_crop(
                grape_det["bbox_xyxy"], img_w, img_h, color_bgr
            )
            if crop_image is not None:
                entries.append((gi, crop_bbox, crop_image))
        if not entries:
            return {}
        crops = [entry[2] for entry in entries]
        try:
            results = self.keypoint_predictor.predict_batch(crops)
        except Exception as e:
            print(f"[PIPELINE] keypoint 批量推理异常: {e!r}")
            return {}
        return {
            gi: (crop_bbox, crop_result)
            for (gi, crop_bbox, _), crop_result in zip(entries, results)
        }

    def _sample_depth_at(self, u: float, v: float, rgbd_frame) -> DepthSample:
        """keypoint 链的深度采样封装（无掩膜）。"""
        return sample_depth(
            depth_data=rgbd_frame.depth_data,
            u=u,
            v=v,
            depth_scale=rgbd_frame.depth_scale,
            mask=None,
            use_mask_only=False,
            depth_min_m=self._depth_config.get("min_m", 0.05),
            depth_max_m=self._depth_config.get("max_m", 5.0),
            patch_radius=self._depth_config.get("patch_radius", 3),
            tolerance_abs_m=self._depth_config.get("tolerance_abs_m", 0.02),
            tolerance_relative=self._depth_config.get("tolerance_relative", 0.03),
            min_center_neighbors=self._depth_config.get("min_center_neighbors", 3),
            search_radius=self._depth_config.get("search_radius", 12),
            estimator=self._depth_config.get("estimator", "median"),
        )

    def _bunch_top_fallback(
        self,
        grape_result: Dict,
        crop_bbox,
        grape_local_bbox,
        rgbd_frame: RGBDFrame,
        img_w: int,
        img_h: int,
    ) -> bool:
        """串顶兜底剪: 果梗链路失败时用葡萄掩膜顶部区域估算剪切点。

        成功则填充 grape_result (status=success, target_type=bunch_top_fallback,
        fallback=True, 无方向点 → 下游回退固定姿态) 并返回 True;
        失败保持调用方原状态返回 False。
        """
        if not self._bunch_enabled or self.sam_segmenter is None:
            return False
        if crop_bbox is None or grape_local_bbox is None:
            return False
        if rgbd_frame.depth_data is None or self.intrinsics is None:
            return False
        cx1, cy1, cx2, cy2 = (int(v) for v in crop_bbox)
        if cx2 <= cx1 or cy2 <= cy1:
            return False
        roi_image = rgbd_frame.color_bgr[cy1:cy2, cx1:cx2]
        if roi_image.size == 0:
            return False

        try:
            sam_result = self.sam_segmenter.segment(roi_image, grape_local_bbox)
        except Exception as exc:
            print(f"[PIPELINE] 串顶兜底剪: SAM3 分割异常 {exc!r}")
            return False
        if sam_result.get("status") != "success" or sam_result.get("mask") is None:
            return False

        component = largest_mask_component(
            sam_result["mask"], int(self._bunch_cfg.get("min_mask_area", 20))
        )
        if component is None:
            return False

        keep_ratio = float(self._bunch_cfg.get("keep_ratio", 0.15))
        band_half = int(self._bunch_cfg.get("band_half_height_px", 3))
        point = bunch_top_cut_point(component, keep_ratio, band_half)
        if point is None:
            return False
        local_x, local_y = point

        # ROI-local → 全局像素
        global_x = cx1 + local_x
        global_y = cy1 + local_y
        cut_xi, cut_yi = int(round(global_x)), int(round(global_y))

        # 剪点水平带深度中值: 带内是果粒表面深度, 轴心在后方 → axis_depth_offset_m 补偿
        full_mask = np.zeros((img_h, img_w), dtype=bool)
        full_mask[cy1:cy2, cx1:cx2] = component
        y0 = max(0, cut_yi - band_half)
        y1 = min(img_h, cut_yi + band_half + 1)
        band_vals = rgbd_frame.depth_data[y0:y1, :][full_mask[y0:y1, :]]
        band_vals = band_vals[np.isfinite(band_vals)]
        depth_min = float(self._depth_config.get("min_m", 0.15))
        depth_max = float(self._depth_config.get("max_m", 8.0))
        band_vals = band_vals[(band_vals >= depth_min) & (band_vals <= depth_max)]
        if band_vals.size == 0:
            return False
        depth_m = float(np.median(band_vals)) + float(
            self._bunch_cfg.get("axis_depth_offset_m", 0.01)
        )

        point3d = pixel_to_camera_xyz(self.intrinsics, global_x, global_y, depth_m)
        grape_result["peduncle_centroid_camera_xyz"] = point3d.to_dict()
        grape_result["peduncle_mask_centroid_xy"] = (float(global_x), float(global_y))
        grape_result["target_type"] = "bunch_top_fallback"
        grape_result["fallback"] = True
        grape_result["status"] = STATUS_SUCCESS
        grape_result["depth_sample_pixel_xy"] = (cut_xi, cut_yi)
        grape_result["depth_value"] = depth_m
        grape_result["depth_sampling_method"] = "bunch_top_band_median"
        grape_result["error"] = None
        print(
            f"[PIPELINE] 串顶兜底剪: grape_id={grape_result.get('grape_id')} "
            f"cut_pixel=({cut_xi},{cut_yi}) 深度={depth_m:.3f}m (补偿后) → 固定姿态"
        )
        return True

    def _process_single_grape_keypoint(
        self,
        grape_det: Dict,
        grape_id: int,
        crop_bbox,
        crop_result,
        rgbd_frame: RGBDFrame,
        img_w: int,
        img_h: int,
    ) -> Dict[str, Any]:
        """公共三点结果 → 全局像素 → 三点深度 → 相机 3D。"""
        grape_bbox = grape_det["bbox_xyxy"]
        result = {
            "grape_id": grape_id,
            "grape_bbox_xyxy": grape_bbox,
            "grape_confidence": grape_det["confidence"],
            "status": STATUS_SUCCESS,
            "roi_bbox_xyxy": crop_bbox,
            "stem_bbox_roi_xyxy": None,
            "stem_bbox_global_xyxy": None,
            "stem_confidence": None,
            "peduncle_mask_centroid_roi_xy": None,
            "peduncle_mask_centroid_xy": None,
            "target_type": "keypoint_cut",
            "keypoints_roi_xy": None,
            "keypoints_global_xy": None,
            "keypoint_path_length_224": None,
            "depth_sample_pixel_xy": None,
            "depth_value": None,
            "depth_sampling_method": None,
            "peduncle_centroid_camera_xyz": None,
            "stem_direction_camera_xyz": None,
            "coordinate_unit": "m",
            "coordinate_frame": "camera_color_optical_frame",
        }

        # 测试模式可要求 worker 返回 224x224 二值掩膜。这里只做坐标还原和
        # 证据透传，不参与三点选择、深度采样或下游运动计算。
        mask_224 = crop_result.get("mask_224") if isinstance(crop_result, dict) else None
        if (
            isinstance(mask_224, np.ndarray)
            and mask_224.size > 0
            and crop_bbox is not None
        ):
            roi_w = max(1, int(crop_bbox[2] - crop_bbox[0]))
            roi_h = max(1, int(crop_bbox[3] - crop_bbox[1]))
            import cv2

            mask_roi = cv2.resize(
                mask_224, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST
            )
            full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            full_mask[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]] = mask_roi
            result["_segmentation_mask_full"] = full_mask

        # 分割或公共骨架后处理未给出有效三点 → 禁止输出运动目标。
        if crop_result is None or "cut_point" not in crop_result:
            # 串顶兜底剪: SAM3 掩膜/骨架失败时用葡萄掩膜顶部区域估算剪切点
            grape_local = None
            if crop_bbox is not None:
                gx1, gy1, gx2, gy2 = grape_bbox
                grape_local = (
                    gx1 - crop_bbox[0],
                    gy1 - crop_bbox[1],
                    gx2 - crop_bbox[0],
                    gy2 - crop_bbox[1],
                )
            if self._bunch_top_fallback(
                grape_result=result,
                crop_bbox=crop_bbox,
                grape_local_bbox=grape_local,
                rgbd_frame=rgbd_frame,
                img_w=img_w,
                img_h=img_h,
            ):
                return result
            result["status"] = STATUS_STEM_NOT_DETECTED
            if crop_result and "error" in crop_result:
                result["error"] = crop_result["error"]
            return result

        result["keypoints_roi_xy"] = {
            "point_1": tuple(crop_result["point_1"]),
            "cut_point": tuple(crop_result["cut_point"]),
            "point_3": tuple(crop_result["point_3"]),
        }
        result["keypoint_path_length_224"] = crop_result.get("path_length_224")

        # ROI 局部 → 全局像素
        roi_x1, roi_y1 = crop_bbox[0], crop_bbox[1]
        kp_local = result["keypoints_roi_xy"]
        kp_global = {
            key: (roi_x1 + xy[0], roi_y1 + xy[1]) for key, xy in kp_local.items()
        }
        result["keypoints_global_xy"] = kp_global
        cut_global = kp_global["cut_point"]
        result["peduncle_mask_centroid_xy"] = cut_global  # 兼容下游/可视化

        # ── 三点深度采样 + 反投影 ───────────────────
        depth_data = rgbd_frame.depth_data
        if depth_data is None or self.intrinsics is None:
            result["status"] = STATUS_INVALID_DEPTH
            return result

        fill_with_cut = self._keypoint_config.get("fill_invalid_with_cut", True)
        cut_depth = self._sample_depth_at(cut_global[0], cut_global[1], rgbd_frame)
        p1_depth = self._sample_depth_at(
            kp_global["point_1"][0], kp_global["point_1"][1], rgbd_frame
        )
        p3_depth = self._sample_depth_at(
            kp_global["point_3"][0], kp_global["point_3"][1], rgbd_frame
        )

        result["depth_sample_pixel_xy"] = cut_depth.pixel_xy
        result["depth_value"] = cut_depth.depth_m
        result["depth_sampling_method"] = cut_depth.method

        if cut_depth.depth_m is None:
            result["status"] = STATUS_INVALID_DEPTH
            return result

        # 果梗细、深度噪声大: P1/P3 深度无效时用 CUT 深度回填
        # (方向点深度只影响姿态, 回填保守且可配置)
        if fill_with_cut:
            p1_m = p1_depth.depth_m if p1_depth.depth_m is not None else cut_depth.depth_m
            p3_m = p3_depth.depth_m if p3_depth.depth_m is not None else cut_depth.depth_m
        else:
            p1_m, p3_m = p1_depth.depth_m, p3_depth.depth_m

        cut3d = pixel_to_camera_xyz(
            self.intrinsics, cut_global[0], cut_global[1], cut_depth.depth_m
        )
        result["peduncle_centroid_camera_xyz"] = cut3d.to_dict()

        if p1_m is not None and p3_m is not None:
            p1_3d = pixel_to_camera_xyz(
                self.intrinsics, kp_global["point_1"][0], kp_global["point_1"][1], p1_m
            )
            p3_3d = pixel_to_camera_xyz(
                self.intrinsics, kp_global["point_3"][0], kp_global["point_3"][1], p3_m
            )
            result["stem_direction_camera_xyz"] = {
                "point_1": p1_3d.to_dict(),
                "point_3": p3_3d.to_dict(),
            }
        # 方向点无深度: stem_direction 为 None, 下游回退固定姿态

        return result

    def close(self) -> None:
        """释放资源。"""
        for component in (self.sam_segmenter, self.keypoint_predictor):
            close = getattr(component, "close", None)
            if callable(close):
                close()
