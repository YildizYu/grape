#!/usr/bin/env python3
"""Gemini 336 fair comparison: shared 2-stage YOLO, SAM3 vs UNet mask."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grape_stem_3d.vision_chain_common import (  # noqa: E402
    expanded_grape_roi,
    postprocess_segmentation_mask,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare SAM3 and UNet with identical YOLO and post-processing"
    )
    parser.add_argument("--config", default="configs/fusion_pipeline.yaml")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--auto-save-limit", type=int, default=5)
    parser.add_argument("--save-gap-frames", type=int, default=20)
    parser.add_argument(
        "--output-dir", default="reports/sam3_unet_fair_comparison"
    )
    parser.add_argument(
        "--display", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def resolve(value) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def full_mask(mask_224, roi_bbox, image_shape):
    if not isinstance(mask_224, np.ndarray) or mask_224.size == 0:
        return None
    x1, y1, x2, y2 = roi_bbox
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = cv2.resize(
        mask_224, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
    )
    return mask


def sample_three_points(post, roi_bbox, frame, depth_cfg):
    from grape_stem_3d.coordinate_3d import pixel_to_camera_xyz
    from grape_stem_3d.depth_sampler import sample_depth

    if post.get("cut_point") is None:
        return None, None, None
    x1, y1, _, _ = roi_bbox
    local = {name: post[name] for name in ("point_1", "cut_point", "point_3")}
    points = {
        name: (x1 + int(point[0]), y1 + int(point[1]))
        for name, point in local.items()
    }
    samples = {}
    for name, (u, v) in points.items():
        samples[name] = sample_depth(
            depth_data=frame.depth_data,
            u=u,
            v=v,
            depth_scale=frame.depth_scale,
            mask=None,
            use_mask_only=False,
            depth_min_m=float(depth_cfg.get("min_m", 0.05)),
            depth_max_m=float(depth_cfg.get("max_m", 5.0)),
            patch_radius=int(depth_cfg.get("patch_radius", 3)),
            tolerance_abs_m=float(depth_cfg.get("tolerance_abs_m", 0.02)),
            tolerance_relative=float(depth_cfg.get("tolerance_relative", 0.03)),
            min_center_neighbors=int(depth_cfg.get("min_center_neighbors", 3)),
            search_radius=int(depth_cfg.get("search_radius", 12)),
            estimator=str(depth_cfg.get("estimator", "median")),
        )
    cut_depth = samples["cut_point"].depth_m
    if cut_depth is None:
        return points, samples, None
    fill = bool(depth_cfg.get("fill_invalid_with_cut", True))
    xyz = {}
    for name, (u, v) in points.items():
        depth = samples[name].depth_m
        if depth is None and fill and name != "cut_point":
            depth = cut_depth
        xyz[name] = (
            pixel_to_camera_xyz(frame.intrinsics, u, v, depth).to_dict()
            if depth is not None and frame.intrinsics is not None
            else None
        )
    return points, samples, xyz


def process_method(name, mask, roi_bbox, stem_bbox, frame, config, inference_ms):
    roi_width = int(roi_bbox[2] - roi_bbox[0])
    roi_height = int(roi_bbox[3] - roi_bbox[1])
    started = time.perf_counter()
    post = postprocess_segmentation_mask(
        mask,
        source_width=roi_width,
        source_height=roi_height,
        stem_bbox_xyxy=stem_bbox,
        config=config,
    )
    post_ms = (time.perf_counter() - started) * 1000.0
    mask_global = full_mask(post.get("mask_224"), roi_bbox, frame.color_bgr.shape)
    if mask_global is None:
        # Keep a paired, explicitly black mask as evidence when a backend did
        # not produce a usable mask; result.json records the failure reason.
        mask_global = np.zeros(frame.color_bgr.shape[:2], dtype=np.uint8)
    depth_cfg = dict(config.get("depth", {}))
    depth_cfg["fill_invalid_with_cut"] = config.get("mask_postprocess", config.get("keypoint", {})).get(
        "fill_invalid_with_cut", True
    )
    points, samples, xyz = sample_three_points(post, roi_bbox, frame, depth_cfg)
    return {
        "model": name,
        "status": "success" if points is not None else "postprocess_failed",
        "error": post.get("error"),
        "inference_ms": inference_ms,
        "shared_postprocess_ms": round(post_ms, 3),
        "path_length_224": post.get("path_length_224"),
        "keypoints_global_xy": points,
        "keypoints_camera_xyz": xyz,
        "depth_m": {
            key: value.depth_m for key, value in (samples or {}).items()
        },
        "_mask_full": mask_global,
    }


def overlay(base, detections, method_name, color):
    image = base.copy()
    for detection in detections:
        gx1, gy1, gx2, gy2 = map(int, detection["grape_bbox_xyxy"])
        sx1, sy1, sx2, sy2 = map(int, detection["stem_bbox_global_xyxy"])
        cv2.rectangle(image, (gx1, gy1), (gx2, gy2), (0, 200, 0), 2)
        cv2.rectangle(image, (sx1, sy1), (sx2, sy2), (0, 220, 255), 2)
        method = detection["methods"][method_name]
        mask = method.get("_mask_full")
        if isinstance(mask, np.ndarray) and np.any(mask):
            layer = np.zeros_like(image)
            layer[mask > 0] = color
            image = cv2.addWeighted(image, 1.0, layer, 0.42, 0)
        points = method.get("keypoints_global_xy") or {}
        p1, cut, p3 = points.get("point_1"), points.get("cut_point"), points.get("point_3")
        if p1 and cut and p3:
            cv2.line(image, tuple(p1), tuple(p3), (255, 255, 255), 2)
            for point, label, point_color in (
                (p1, "P1", (255, 0, 0)),
                (cut, "CUT", (0, 0, 255)),
                (p3, "P3", (0, 255, 255)),
            ):
                cv2.circle(image, tuple(point), 5, point_color, -1)
                cv2.putText(image, label, (point[0] + 5, point[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, point_color, 1, cv2.LINE_AA)
    cv2.putText(image, method_name.upper(), (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def save_bundle(session_dir, index, frame_id, frame, views, result):
    bundle = session_dir / f"pair_{index:02d}_frame_{frame_id:06d}"
    bundle.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(bundle / "color.png"), frame.color_bgr)
    cv2.imwrite(str(bundle / "comparison.png"), views["comparison"])
    cv2.imwrite(str(bundle / "sam3_annotated.png"), views["sam3"])
    cv2.imwrite(str(bundle / "unet_annotated.png"), views["unet"])
    np.save(str(bundle / "depth_raw.npy"), frame.depth_data)
    for det_index, detection in enumerate(result["detections"]):
        for method_name, method in detection["methods"].items():
            mask = method.get("_mask_full")
            if isinstance(mask, np.ndarray):
                cv2.imwrite(str(bundle / f"grape_{det_index:02d}_{method_name}_mask.png"), mask)
    (bundle / "result.json").write_text(
        json.dumps(serializable(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved paired evidence: {bundle}")


def main() -> int:
    args = parse_args()
    if args.max_frames <= 0 or args.auto_save_limit < 0 or args.save_gap_frames < 0:
        print("ERROR: frame/save arguments are invalid", file=sys.stderr)
        return 2
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    models = config.get("models", {})
    grape_cfg = config.get("grapes_detector", {})
    stem_cfg = config.get("stem_detector", {})
    sel_cfg = config.get("stem_selection", {})
    kp_cfg = config.get("mask_postprocess", config.get("keypoint", {}))
    unet_cfg = config.get("keypoint", {})
    sam_cfg = config.get("sam", {})
    compare_cfg = config.get("comparison", config.get("sam3_runtime", {}))

    required = {
        "grape YOLO": resolve(models.get("grapes_weights", "")),
        "stem YOLO": resolve(models.get("stem_weights", "")),
        "UNet": resolve(unet_cfg.get("unet_weights", "")),
        "SAM3": resolve(models.get("sam_checkpoint", "")),
        "UNet worker": resolve(unet_cfg.get("worker_script", "")),
        "UNet Python": resolve(unet_cfg.get("worker_python", "")),
        "SAM3 worker": resolve(compare_cfg.get("sam3_worker_script", compare_cfg.get("worker_script", ""))),
        "SAM3 Python": resolve(compare_cfg.get("sam3_worker_python", compare_cfg.get("worker_python", ""))),
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing files:\n  " + "\n  ".join(missing))

    from grape_stem.grape_detector import GrapeDetector
    from grape_stem.stem_detector import StemDetector
    from grape_stem.stem_selector import StemSelector
    from grape_stem_3d.camera_adapter import CameraAdapter
    from grape_stem_3d.keypoint_predictor import KeypointPredictor
    from grape_stem_3d.sam3_mask_predictor import Sam3MaskPredictor

    device = str(models.get("device", "0"))
    grape_detector = GrapeDetector(required["grape YOLO"], float(grape_cfg.get("confidence", .25)),
                                   float(grape_cfg.get("iou", .5)), int(grape_cfg.get("imgsz", 640)), device)
    stem_detector = StemDetector(required["stem YOLO"], float(stem_cfg.get("confidence", .15)),
                                 float(stem_cfg.get("iou", .5)), int(stem_cfg.get("imgsz", 960)), device)
    selector = StemSelector(float(sel_cfg.get("confidence_weight", .6)),
                            float(sel_cfg.get("position_weight", .4)))
    unet = KeypointPredictor(
        str(required["UNet Python"]), str(required["UNet worker"]), str(required["UNet"]),
        threshold=float(unet_cfg.get("threshold", .80)),
        pose_offset=float(kp_cfg.get("pose_offset", 12)),
        min_component_area=int(kp_cfg.get("min_component_area", 12)),
        min_path_length=float(kp_cfg.get("min_path_length", 20)),
        ready_timeout_s=float(unet_cfg.get("ready_timeout_s", 180)),
        job_timeout_s=float(unet_cfg.get("job_timeout_s", 60)),
        return_mask=True, mask_only=True,
    )
    sam3 = Sam3MaskPredictor(
        str(required["SAM3 Python"]), str(required["SAM3 worker"]), str(required["SAM3"]),
        device, float(sam_cfg.get("box_padding", .15)),
        float(compare_cfg.get("sam3_ready_timeout_s", 300)),
        float(compare_cfg.get("sam3_job_timeout_s", 120)),
    )
    camera_cfg = config.get("camera", {})
    camera = CameraAdapter(
        width=int(camera_cfg.get("color_width", 640)), height=int(camera_cfg.get("color_height", 480)),
        fps=int(camera_cfg.get("color_fps", 30)), align_to=camera_cfg.get("align_to", "color"),
        depth_width=int(camera_cfg.get("depth_width", 640)), depth_height=int(camera_cfg.get("depth_height", 480)),
        depth_fps=int(camera_cfg.get("depth_fps", 30)), alignment_mode=camera_cfg.get("alignment_mode", "hardware"),
        serial_number=camera_cfg.get("serial_number"),
    )
    session_dir = resolve(args.output_dir) / time.strftime("session_%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "test": "fair SAM3 vs UNet: identical two-stage YOLO and post-processing",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": {"machine": platform.machine(), "python": sys.version},
        "config": str(config_path), "frames": 0, "stem_yolo_frames": 0,
        "sam3_successes": 0, "unet_successes": 0, "saved_pairs": 0,
        "segmentation_batches": 0, "sam3_batch_ms_total": 0.0,
        "unet_batch_ms_total": 0.0,
    }
    last_save_frame = -args.save_gap_frames
    last_views = last_result = last_frame = None
    print("Loading models. SAM3 can take several minutes on first load...")
    try:
        # Force both workers to load before measurements begin.
        unet.predict_batch([np.zeros((224, 224, 3), np.uint8)])
        sam3._start()
        camera.start()
        print("Ready: both branches share grape YOLO, stem YOLO and mask post-processing.")
        for frame_id in range(1, args.max_frames + 1):
            frame = camera.read()
            if frame is None:
                continue
            frame.frame_id = frame_id
            image = frame.color_bgr
            image_h, image_w = image.shape[:2]
            grape_detections = grape_detector.detect(image)
            entries = []
            for grape in grape_detections:
                roi_bbox = expanded_grape_roi(grape["bbox_xyxy"], image_w, image_h, config.get("roi", {}))
                rx1, ry1, rx2, ry2 = roi_bbox
                roi = image[ry1:ry2, rx1:rx2]
                if roi.size == 0:
                    continue
                stems = stem_detector.detect(roi)
                gx1, gy1, gx2, gy2 = grape["bbox_xyxy"]
                grape_local = (gx1-rx1, gy1-ry1, gx2-rx1, gy2-ry1)
                selected = selector.select(stems, grape_local, roi_bbox)
                if selected.get("status") != "stem_detected":
                    continue
                entries.append((grape, roi_bbox, roi, selected["best_stem_bbox"], selected["best_confidence"]))

            result = {"frame_id": frame_id, "detections": []}
            if entries:
                summary["stem_yolo_frames"] += 1
                rois = [entry[2] for entry in entries]
                stem_boxes = [entry[3] for entry in entries]
                # Alternate execution order to avoid systematically favoring the
                # model that always runs first or second on the shared GPU.
                batch_timings = {}
                if frame_id % 2:
                    started = time.perf_counter()
                    sam_results = sam3.predict_batch(rois, stem_boxes)
                    batch_timings["sam3_ms"] = (time.perf_counter() - started) * 1000.0
                    started = time.perf_counter()
                    unet_results = unet.predict_batch(rois)
                    batch_timings["unet_ms"] = (time.perf_counter() - started) * 1000.0
                    batch_timings["execution_order"] = ["sam3", "unet"]
                else:
                    started = time.perf_counter()
                    unet_results = unet.predict_batch(rois)
                    batch_timings["unet_ms"] = (time.perf_counter() - started) * 1000.0
                    started = time.perf_counter()
                    sam_results = sam3.predict_batch(rois, stem_boxes)
                    batch_timings["sam3_ms"] = (time.perf_counter() - started) * 1000.0
                    batch_timings["execution_order"] = ["unet", "sam3"]
                batch_timings["sam3_ms"] = round(batch_timings["sam3_ms"], 3)
                batch_timings["unet_ms"] = round(batch_timings["unet_ms"], 3)
                result["segmentation_batch_timing"] = batch_timings
                summary["segmentation_batches"] += 1
                summary["sam3_batch_ms_total"] += batch_timings["sam3_ms"]
                summary["unet_batch_ms_total"] += batch_timings["unet_ms"]
                for entry, sam_raw, unet_raw in zip(entries, sam_results, unet_results):
                    grape, roi_bbox, roi, stem_bbox, stem_conf = entry
                    sam_method = process_method(
                        "sam3", sam_raw.get("mask"), roi_bbox, stem_bbox, frame, config,
                        sam_raw.get("inference_ms"),
                    )
                    unet_method = process_method(
                        "unet", (unet_raw or {}).get("mask_224"), roi_bbox, stem_bbox, frame, config,
                        (unet_raw or {}).get("inference_ms"),
                    )
                    if sam_method["status"] == "success": summary["sam3_successes"] += 1
                    if unet_method["status"] == "success": summary["unet_successes"] += 1
                    sx1, sy1, sx2, sy2 = stem_bbox
                    result["detections"].append({
                        "grape_bbox_xyxy": grape["bbox_xyxy"],
                        "grape_confidence": grape["confidence"],
                        "roi_bbox_xyxy": roi_bbox,
                        "stem_bbox_roi_xyxy": stem_bbox,
                        "stem_bbox_global_xyxy": (roi_bbox[0]+sx1, roi_bbox[1]+sy1, roi_bbox[0]+sx2, roi_bbox[1]+sy2),
                        "stem_confidence": stem_conf,
                        "sam3_segmentation_status": sam_raw.get("status"),
                        "sam3_segmentation_error": sam_raw.get("error"),
                        "methods": {"sam3": sam_method, "unet": unet_method},
                    })
            sam_view = overlay(image, result["detections"], "sam3", (180, 40, 180))
            unet_view = overlay(image, result["detections"], "unet", (255, 100, 0))
            views = {"sam3": sam_view, "unet": unet_view,
                     "comparison": np.hstack((sam_view, unet_view))}
            last_views, last_result, last_frame = views, result, frame
            summary["frames"] += 1
            below_limit = (
                args.auto_save_limit == 0
                or summary["saved_pairs"] < args.auto_save_limit
            )
            if entries and below_limit and frame_id-last_save_frame >= args.save_gap_frames:
                summary["saved_pairs"] += 1
                save_bundle(session_dir, summary["saved_pairs"], frame_id, frame, views, result)
                last_save_frame = frame_id
            if args.display:
                cv2.imshow("SAM3 vs UNet fair comparison", views["comparison"])
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s") and last_result is not None:
                    summary["saved_pairs"] += 1
                    save_bundle(session_dir, summary["saved_pairs"], frame_id, frame, views, result)
            if frame_id == 1 or frame_id % 30 == 0:
                print(
                    f"Frame {frame_id}: grapes={len(grape_detections)} "
                    f"selected_stems={len(entries)} saved={summary['saved_pairs']}"
                )
    finally:
        camera.stop()
        unet.close()
        sam3.close()
        cv2.destroyAllWindows()
        batches = summary["segmentation_batches"]
        summary["sam3_average_batch_ms"] = (
            round(summary["sam3_batch_ms_total"] / batches, 3) if batches else None
        )
        summary["unet_average_batch_ms"] = (
            round(summary["unet_batch_ms_total"] / batches, 3) if batches else None
        )
        (session_dir / "summary.json").write_text(
            json.dumps(serializable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Session: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
