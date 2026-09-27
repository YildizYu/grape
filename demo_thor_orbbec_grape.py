#!/usr/bin/env python3
"""Live Thor + Orbbec Gemini 336 + grape YOLO acceptance demo.

This entry point deliberately stops at grape detection and camera-frame XYZ.
SAM, stem localization, hand-eye calibration, and robot control are outside the
hardware-migration acceptance scope.
"""

import argparse
import json
import os
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grape_stem.grape_detector import GrapeDetector
from grape_stem_3d.camera_adapter import CameraAdapter
from grape_stem_3d.coordinate_3d import pixel_to_camera_xyz
from grape_stem_3d.depth_sampler import sample_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemini 336 live RGB-D and Thor CUDA grape detection demo"
    )
    parser.add_argument("--config", default="configs/fusion_pipeline.yaml")
    parser.add_argument("--weights", default=None, help="Override grape YOLO weights")
    parser.add_argument("--device", default=None, help="Ultralytics device, default from config")
    parser.add_argument("--frames", type=int, default=120, help="Maximum live frames")
    parser.add_argument(
        "--min-frames", type=int, default=30,
        help="Minimum processed frames required for acceptance",
    )
    parser.add_argument(
        "--display", action=argparse.BooleanOptionalAction, default=False,
        help="Show the live window; use --display for the group-meeting demo",
    )
    parser.add_argument(
        "--warmup", action=argparse.BooleanOptionalAction, default=True,
        help="Warm up YOLO on CUDA before timed live inference",
    )
    parser.add_argument(
        "--require-grape", action="store_true",
        help="Fail acceptance unless at least one grape is detected",
    )
    parser.add_argument(
        "--output-dir", default="reports/thor_orbbec_grape_demo",
        help="Directory for JSON and image evidence",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def orbbec_usb_speed_mbps() -> Optional[float]:
    """Read the negotiated Gemini 336 USB speed from sysfs."""
    for vendor_path in Path("/sys/bus/usb/devices").glob("*/idVendor"):
        try:
            if vendor_path.read_text().strip().lower() != "2bc5":
                continue
            if vendor_path.with_name("idProduct").read_text().strip().lower() != "0803":
                continue
            return float(vendor_path.with_name("speed").read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def required_check_names(require_grape: bool) -> List[str]:
    names = [
        "aarch64_platform",
        "cuda_available",
        "gpu_is_nvidia_thor",
        "usb3_link",
        "device_is_gemini_336",
        "rgbd_frames_processed",
        "rgb_depth_aligned",
        "d2c_alignment_enabled",
        "valid_depth_available",
        "grape_model_on_cuda",
        "live_frame_inference",
    ]
    if require_grape:
        names.extend(("grape_detected", "grape_depth_and_xyz"))
    return names


def evaluate_acceptance(checks: Dict[str, bool], require_grape: bool) -> bool:
    return all(bool(checks.get(name, False)) for name in required_check_names(require_grape))


def detection_with_3d(
    detection: Dict[str, Any],
    depth_data: np.ndarray,
    depth_scale: float,
    intrinsics,
    depth_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach robust center depth and camera optical-frame XYZ to one detection."""
    x1, y1, x2, y2 = detection["bbox_xyxy"]
    center_u = float((x1 + x2) / 2.0)
    center_v = float((y1 + y2) / 2.0)
    depth = sample_depth(
        depth_data=depth_data,
        u=center_u,
        v=center_v,
        depth_scale=depth_scale,
        mask=None,
        use_mask_only=False,
        depth_min_m=depth_cfg.get("min_m", 0.05),
        depth_max_m=depth_cfg.get("max_m", 5.0),
        patch_radius=depth_cfg.get("patch_radius", 3),
        tolerance_abs_m=depth_cfg.get("tolerance_abs_m", 0.02),
        tolerance_relative=depth_cfg.get("tolerance_relative", 0.03),
        min_center_neighbors=depth_cfg.get("min_center_neighbors", 3),
        search_radius=depth_cfg.get("search_radius", 12),
        estimator=depth_cfg.get("estimator", "median"),
    )
    xyz = None
    if depth.depth_m is not None:
        xyz = pixel_to_camera_xyz(intrinsics, center_u, center_v, depth.depth_m)

    return {
        "class_id": int(detection["class_id"]),
        "class_name": str(detection["class_name"]),
        "confidence": float(detection["confidence"]),
        "bbox_xyxy": [float(value) for value in detection["bbox_xyxy"]],
        "bbox_center_pixel": [center_u, center_v],
        "sampled_pixel": list(depth.pixel_xy),
        "depth_m": depth.depth_m,
        "depth_sample_method": depth.method,
        "depth_valid_count": depth.valid_count,
        "camera_xyz_m": xyz.to_dict() if xyz is not None else None,
    }


def draw_demo_frame(
    color_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    status_lines: List[str],
) -> np.ndarray:
    """Draw detections and compact acceptance status without changing the input."""
    canvas = color_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in detection["bbox_xyxy"]]
        height, width = canvas.shape[:2]
        x1, x2 = np.clip([x1, x2], 0, max(width - 1, 0))
        y1, y2 = np.clip([y1, y2], 0, max(height - 1, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 220, 60), 2)

        center_u, center_v = [int(round(v)) for v in detection["bbox_center_pixel"]]
        cv2.drawMarker(
            canvas, (center_u, center_v), (0, 255, 255),
            markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2,
        )
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        if detection["depth_m"] is not None:
            xyz = detection["camera_xyz_m"]
            label += f"  Z={detection['depth_m']:.3f}m"
            coordinate = f"camera XYZ=({xyz['x']:.3f}, {xyz['y']:.3f}, {xyz['z']:.3f})m"
        else:
            coordinate = "camera XYZ=invalid depth"
        label_y = max(22, y1 - 7)
        cv2.putText(canvas, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (30, 220, 60), 2, cv2.LINE_AA)
        coordinate_y = min(height - 8, max(label_y + 19, y1 + 17))
        cv2.putText(canvas, coordinate, (x1, coordinate_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

    line_height = 22
    panel_height = 12 + line_height * len(status_lines)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)
    for index, line in enumerate(status_lines):
        cv2.putText(
            canvas, line, (10, 21 + index * line_height), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return canvas


def _json_detection_score(detections: List[Dict[str, Any]]) -> Tuple[int, float]:
    return (
        len(detections),
        max((float(item["confidence"]) for item in detections), default=0.0),
    )


def main() -> int:
    args = parse_args()
    if args.frames <= 0 or args.min_frames <= 0:
        raise SystemExit("--frames and --min-frames must both be positive")

    config_path = resolve_project_path(args.config)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output_dir / ".ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    camera_cfg = config.get("camera", {})
    models_cfg = config.get("models", {})
    detector_cfg = config.get("grapes_detector", {})
    depth_cfg = config.get("depth", {})
    weights_path = resolve_project_path(
        args.weights or models_cfg.get("grapes_weights", "weights/grapes/best.pt")
    )
    device = args.device or models_cfg.get("device", "0")
    usb_speed = orbbec_usb_speed_mbps()
    required_frames = min(args.frames, args.min_frames)

    report: Dict[str, Any] = {
        "test": "Thor + Orbbec Gemini 336 + live grape YOLO acceptance",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": "strict_grape_required" if args.require_grape else "infrastructure",
        "config": str(config_path),
        "weights": str(weights_path),
        "requested_frames": args.frames,
        "minimum_required_frames": required_frames,
        "usb_speed_mbps": usb_speed,
        "checks": {},
        "required_checks": required_check_names(args.require_grape),
        "passed": False,
    }
    checks = report["checks"]
    checks["aarch64_platform"] = platform.machine() == "aarch64"
    checks["usb3_link"] = usb_speed is not None and usb_speed >= 5000
    camera = None
    best_score = (-1, -1.0)
    best_color = None
    best_depth = None
    best_annotated = None
    best_detections: List[Dict[str, Any]] = []
    last_annotated = None
    inference_times_ms: List[float] = []
    valid_depth_ratios: List[float] = []
    frames_processed = 0
    total_detections = 0
    detections_with_xyz = 0
    frames_with_grapes = 0
    started_at = None

    try:
        import torch
        import ultralytics

        checks["cuda_available"] = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if checks["cuda_available"] else None
        checks["gpu_is_nvidia_thor"] = bool(gpu_name and "thor" in gpu_name.lower())
        report["runtime"] = {
            "python": sys.version,
            "machine": platform.machine(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pyorbbecsdk2": version("pyorbbecsdk2"),
            "cuda_device": gpu_name,
            "inference_device_requested": str(device),
        }
        if not checks["cuda_available"]:
            raise RuntimeError("CUDA is unavailable; the demo must run on the Thor GPU")

        print("[1/4] Thor CUDA: PASS -", gpu_name)
        print("[2/4] Loading grape YOLO:", weights_path)
        detector = GrapeDetector(
            weights_path=weights_path,
            confidence=detector_cfg.get("confidence", 0.25),
            iou=detector_cfg.get("iou", 0.50),
            imgsz=detector_cfg.get("imgsz", 640),
            device=str(device),
        )
        if args.warmup:
            detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
            torch.cuda.synchronize()

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
        intrinsics = camera.start()
        checks["device_is_gemini_336"] = "336" in camera.device_model
        report["camera"] = {
            "model": camera.device_model,
            "serial_number": camera.device_serial,
            "alignment_mode": camera.active_alignment_mode,
            "intrinsics": {
                "fx": intrinsics.fx, "fy": intrinsics.fy,
                "cx": intrinsics.cx, "cy": intrinsics.cy,
                "width": intrinsics.width, "height": intrinsics.height,
            },
        }
        print(f"[3/4] Gemini 336 RGB-D: PASS - USB {usb_speed:g}M, "
              f"D2C {camera.active_alignment_mode}")
        print("[4/4] Live YOLO started. Press q to finish, s to save a snapshot.")

        if args.display:
            cv2.namedWindow("Thor + Gemini 336 Grape Acceptance", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Thor + Gemini 336 Grape Acceptance", 960, 720)

        started_at = time.monotonic()
        attempts = 0
        max_attempts = max(args.frames * 4, args.frames + 30)
        while frames_processed < args.frames and attempts < max_attempts:
            attempts += 1
            frame = camera.read()
            if frame is None:
                continue

            depth_m = frame.depth_data.astype(np.float32) * frame.depth_scale
            valid = np.isfinite(depth_m) & (
                depth_m >= depth_cfg.get("min_m", 0.05)
            ) & (depth_m <= depth_cfg.get("max_m", 5.0))
            valid_depth_ratios.append(float(valid.mean()))

            torch.cuda.synchronize()
            inference_start = time.perf_counter()
            raw_detections = detector.detect(frame.color_bgr)
            torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            inference_times_ms.append(inference_ms)
            frames_processed += 1

            detections = [
                detection_with_3d(
                    item, frame.depth_data, frame.depth_scale, intrinsics, depth_cfg
                )
                for item in raw_detections
            ]
            total_detections += len(detections)
            detections_with_xyz += sum(
                item["camera_xyz_m"] is not None for item in detections
            )
            if detections:
                frames_with_grapes += 1

            elapsed = time.monotonic() - started_at
            pipeline_fps = frames_processed / elapsed if elapsed > 0 else 0.0
            status_lines = [
                f"Thor CUDA | USB {usb_speed:g}M | Gemini 336 D2C {camera.active_alignment_mode}",
                f"frame {frames_processed}/{args.frames} | pipeline {pipeline_fps:.1f} FPS | infer {inference_ms:.1f} ms",
                f"grapes this frame {len(detections)} | detections total {total_detections}",
            ]
            annotated = draw_demo_frame(frame.color_bgr, detections, status_lines)
            last_annotated = annotated
            score = _json_detection_score(detections)
            if score > best_score:
                best_score = score
                best_color = frame.color_bgr.copy()
                best_depth = frame.depth_data.copy()
                best_annotated = annotated.copy()
                best_detections = detections

            if frames_processed == 1 or frames_processed % 10 == 0:
                print(
                    f"  frame={frames_processed:03d} infer={inference_ms:6.1f}ms "
                    f"grapes={len(detections)} valid_depth={valid_depth_ratios[-1]:.1%}"
                )

            if args.display:
                cv2.imshow("Thor + Gemini 336 Grape Acceptance", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    snapshot = output_dir / f"snapshot_{frames_processed:06d}.png"
                    cv2.imwrite(str(snapshot), annotated)
                    print("  Saved:", snapshot)
                elif key == ord("q"):
                    break

        elapsed = time.monotonic() - started_at
        model_device = None
        try:
            model_device = str(detector._model.predictor.device)
        except AttributeError:
            pass
        checks.update({
            "rgbd_frames_processed": frames_processed >= required_frames,
            "rgb_depth_aligned": bool(
                best_color is not None and best_depth is not None
                and best_color.shape[:2] == best_depth.shape
            ),
            "d2c_alignment_enabled": camera.active_alignment_mode in {"hardware", "software"},
            "valid_depth_available": bool(
                valid_depth_ratios and np.median(valid_depth_ratios) >= 0.01
            ),
            "grape_model_on_cuda": bool(model_device and model_device.startswith("cuda")),
            "live_frame_inference": len(inference_times_ms) == frames_processed and frames_processed > 0,
            "grape_detected": total_detections > 0,
            "grape_depth_and_xyz": detections_with_xyz > 0,
        })
        report.update({
            "frames_processed": frames_processed,
            "frames_with_grapes": frames_with_grapes,
            "total_grape_detections": total_detections,
            "grape_detections_with_xyz": detections_with_xyz,
            "pipeline_fps": round(frames_processed / elapsed, 2) if elapsed > 0 else 0.0,
            "inference_ms_median": round(float(np.median(inference_times_ms)), 2),
            "inference_ms_p95": round(float(np.percentile(inference_times_ms, 95)), 2),
            "valid_depth_ratio_median": float(np.median(valid_depth_ratios)),
            "model_parameter_device": model_device,
            "best_frame_detections": best_detections,
        })
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if camera is not None:
            camera.stop()
        cv2.destroyAllWindows()

        if best_annotated is not None:
            cv2.imwrite(str(output_dir / "best_annotated.png"), best_annotated)
            cv2.imwrite(str(output_dir / "best_color.png"), best_color)
            np.save(str(output_dir / "best_depth_raw.npy"), best_depth)
        if last_annotated is not None:
            cv2.imwrite(str(output_dir / "last_annotated.png"), last_annotated)

        for name in required_check_names(args.require_grape):
            checks.setdefault(name, False)
        report["passed"] = evaluate_acceptance(checks, args.require_grape)
        report["evidence"] = [
            name for name in (
                "report.json", "best_annotated.png", "best_color.png",
                "best_depth_raw.npy", "last_annotated.png",
            ) if name == "report.json" or (output_dir / name).exists()
        ]
        report_path = output_dir / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print("\n" + "=" * 66)
    print("ACCEPTANCE:", "PASS" if report["passed"] else "FAIL")
    print("MODE:", report["mode"])
    print("REPORT:", output_dir / "report.json")
    if not args.require_grape:
        print("NOTE: use --require-grape with a grape in view for strict acceptance")
    print("=" * 66)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
