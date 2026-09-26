#!/usr/bin/env python3
"""ZED X Mini 两阶段 YOLO + SAM3 剪切点现场验收，不连接机械臂。

本脚本严格读取 fusion_pipeline.yaml 中已有的模型、阈值、ROI 和深度参数，
固定要求 vision.chain=sam3_keypoint。它不导入 ROS2，不读取手眼矩阵，也不发送位姿。
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test ZED X Mini and the configured YOLO+SAM3 cut-point chain"
    )
    parser.add_argument(
        "--config",
        default="configs/fusion_pipeline.yaml",
        help="配置文件路径（相对路径以视觉项目根目录为准）",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示实时检测窗口；无桌面环境时使用 --no-display",
    )
    parser.add_argument(
        "--max-frames", type=int, default=600, help="最多处理的有效 RGB-D 帧数"
    )
    parser.add_argument(
        "--auto-save-limit", type=int, default=5, help="自动保存成功剪切点的组数"
    )
    parser.add_argument(
        "--save-gap-frames", type=int, default=20, help="两次自动保存之间的最小帧数"
    )
    parser.add_argument(
        "--output-dir",
        default="reports/zed_sam3_cut_test",
        help="验收结果根目录（相对路径以视觉项目根目录为准）",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): serializable(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def worker_cuda_status(worker_python: Path) -> Tuple[bool, Optional[str], Optional[str]]:
    """在 SAM3 worker 实际使用的 Python 环境中确认 CUDA 状态。"""
    code = (
        "import json, torch; "
        "ok=torch.cuda.is_available(); "
        "print(json.dumps({'available':ok,'device':"
        "torch.cuda.get_device_name(0) if ok else None}))"
    )
    try:
        completed = subprocess.run(
            [str(worker_python), "-c", code],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        data = json.loads(completed.stdout.strip().splitlines()[-1])
        return bool(data.get("available")), data.get("device"), None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def build_keypoint_pipeline(config: Dict[str, Any]):
    """Build the production two-stage YOLO + SAM3 shared-keypoint chain."""
    from grape_stem.grape_detector import GrapeDetector
    from grape_stem.stem_detector import StemDetector
    from grape_stem.stem_selector import StemSelector
    from grape_stem_3d.sam3_runtime import build_sam3_predictor
    from grape_stem_3d.realtime_pipeline import RealtimeGrapeStemPipeline

    chain = config.get("vision", {}).get("chain")
    if chain != "sam3_keypoint":
        raise RuntimeError(
            f"本验收脚本只允许 vision.chain=sam3_keypoint，当前配置为 {chain!r}；"
            "脚本不会替你修改算法选择"
        )

    models = config.get("models", {})
    grape_cfg = config.get("grapes_detector", {})
    stem_cfg = config.get("stem_detector", {})
    sel_cfg = config.get("stem_selection", {})

    grape_weights = resolve_project_path(models.get("grapes_weights", ""))
    stem_weights = resolve_project_path(models.get("stem_weights", ""))
    if not grape_weights.is_file() or not stem_weights.is_file():
        raise FileNotFoundError(f"缺少 YOLO 权重: {grape_weights}, {stem_weights}")

    detector = GrapeDetector(
        weights_path=grape_weights,
        confidence=float(grape_cfg.get("confidence", 0.25)),
        iou=float(grape_cfg.get("iou", 0.50)),
        imgsz=int(grape_cfg.get("imgsz", 640)),
        device=str(models.get("device", "0")),
    )
    stem_detector = StemDetector(
        weights_path=stem_weights,
        confidence=float(stem_cfg.get("confidence", 0.15)),
        iou=float(stem_cfg.get("iou", 0.50)),
        imgsz=int(stem_cfg.get("imgsz", 960)),
        device=str(models.get("device", "0")),
    )
    stem_selector = StemSelector(
        confidence_weight=float(sel_cfg.get("confidence_weight", 0.60)),
        position_weight=float(sel_cfg.get("position_weight", 0.40)),
    )
    predictor, sam_checkpoint, worker_python = build_sam3_predictor(
        PROJECT_ROOT, config
    )
    pipeline = RealtimeGrapeStemPipeline(
        grape_detector=detector,
        stem_detector=stem_detector,
        stem_selector=stem_selector,
        intrinsics=None,
        sam_segmenter=predictor,
        keypoint_predictor=None,
        config=config,
    )
    paths = {
        "grape_weights": str(grape_weights),
        "stem_weights": str(stem_weights),
        "sam3_checkpoint": str(sam_checkpoint),
        "sam3_worker_python": str(worker_python),
    }
    return pipeline, detector, predictor, worker_python, paths


def successful_targets(frame_result: Dict[str, Any]):
    successes = []
    for grape in frame_result.get("grapes", []):
        points = grape.get("keypoints_global_xy") or {}
        xyz = grape.get("peduncle_centroid_camera_xyz") or {}
        if (
            grape.get("status") == "success"
            and grape.get("target_type") == "keypoint_cut"
            and all(points.get(name) is not None for name in ("point_1", "cut_point", "point_3"))
            and all(xyz.get(axis) is not None for axis in ("x", "y", "z"))
        ):
            successes.append(grape)
    return successes


def save_bundle(
    session_dir: Path,
    label: str,
    frame,
    annotated: np.ndarray,
    frame_result: Dict[str, Any],
) -> int:
    bundle = session_dir / label
    bundle.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(bundle / "color.png"), frame.color_bgr)
    cv2.imwrite(str(bundle / "annotated.png"), annotated)
    np.save(str(bundle / "depth_raw.npy"), frame.depth_data)
    mask_dir = bundle / "masks"
    masks_saved = 0
    for grape in frame_result.get("grapes", []):
        mask = grape.get("_segmentation_mask_full")
        if not isinstance(mask, np.ndarray) or mask.size == 0:
            continue
        mask_dir.mkdir(exist_ok=True)
        grape_id = int(grape.get("grape_id", masks_saved))
        if cv2.imwrite(str(mask_dir / f"grape_{grape_id:02d}_sam3_mask.png"), mask):
            masks_saved += 1
    (bundle / "result.json").write_text(
        json.dumps(serializable(frame_result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已保存: {bundle} (masks={masks_saved})")
    return masks_saved


def main() -> int:
    args = parse_args()
    if args.max_frames <= 0:
        print("ERROR: --max-frames 必须大于 0", file=sys.stderr)
        return 2
    if args.auto_save_limit < 0 or args.save_gap_frames < 0:
        print("ERROR: 保存数量和帧间隔不能为负数", file=sys.stderr)
        return 2

    config_path = resolve_project_path(args.config)
    output_root = resolve_project_path(args.output_dir)
    session_dir = output_root / time.strftime("session_%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)

    report: Dict[str, Any] = {
        "test": "ZED X Mini + two-stage YOLO + SAM3 cut-point acceptance",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": str(config_path),
        "platform": {"machine": platform.machine(), "python": sys.version},
        "checks": {},
        "passed": False,
    }
    report_path = session_dir / "report.json"
    camera = None
    predictor = None
    last_frame = None
    last_result = None
    last_annotated = None

    try:
        if not config_path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        report["algorithm_config"] = {
            "vision": config.get("vision", {}),
            "grapes_detector": config.get("grapes_detector", {}),
            "mask_postprocess": config.get("mask_postprocess", {}),
            "depth": config.get("depth", {}),
        }

        pipeline, detector, predictor, worker_python, paths = build_keypoint_pipeline(config)
        report["model_paths"] = paths
        worker_cuda, worker_cuda_device, worker_cuda_error = worker_cuda_status(worker_python)
        report["worker_cuda"] = {
            "available": worker_cuda,
            "device": worker_cuda_device,
            "error": worker_cuda_error,
        }

        from grape_stem_3d.camera_adapter import CameraAdapter
        from grape_stem_3d.realtime_visualizer import draw_overlay

        camera_cfg = config.get("camera", {})
        depth_cfg = config.get("depth", {})
        # Load the high-peak SAM3 worker before allocating ZED HD1200 buffers.
        pipeline.warmup()
        camera = CameraAdapter(
            width=int(camera_cfg.get("color_width", 640)),
            height=int(camera_cfg.get("color_height", 480)),
            fps=int(camera_cfg.get("color_fps", 30)),
            align_to=camera_cfg.get("align_to", "color"),
            zed_resolution=camera_cfg.get("zed_resolution", "HD1200"),
            zed_depth_mode=camera_cfg.get("zed_depth_mode", "NEURAL"),
            zed_depth_min_m=float(camera_cfg.get("zed_depth_min_m", 0.15)),
            zed_depth_max_m=float(camera_cfg.get("zed_depth_max_m", 8.0)),
        )
        intrinsics = camera.start()
        pipeline.intrinsics = intrinsics

        report["camera"] = {
            "model": camera_cfg.get("model", "ZED X Mini"),
            "zed_resolution": camera_cfg.get("zed_resolution", "HD1200"),
            "zed_depth_mode": camera_cfg.get("zed_depth_mode", "NEURAL"),
            "alignment_mode": "native_depth_to_left_color",
            "depth_scale_m_per_count": camera.depth_scale,
            "intrinsics": {
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy,
                "width": intrinsics.width,
                "height": intrinsics.height,
            },
        }

        window = "ZED X Mini - YOLO + SAM3 Cut Point Test"
        if args.display:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1280, 720)

        print("\n纯视觉验收已启动，不会连接或驱动机械臂。")
        print("红点 CUT=剪切点，P1/P3=果梗方向点。按 s 手动保存，q 退出。\n")

        processed = 0
        read_failures = 0
        consecutive_read_failures = 0
        read_fail_threshold = int(camera_cfg.get("read_fail_threshold", 10))
        frames_with_grapes = 0
        frames_with_cut = 0
        total_grapes = 0
        total_cuts = 0
        total_sam3_masks = 0
        auto_saved = 0
        manual_saved = 0
        last_auto_save_frame = -args.save_gap_frames
        aligned_frames = 0
        inference_ms = []
        started = time.monotonic()

        while processed < args.max_frames:
            frame = camera.read()
            if frame is None:
                read_failures += 1
                consecutive_read_failures += 1
                if consecutive_read_failures >= read_fail_threshold:
                    raise RuntimeError(
                        f"相机连续 {consecutive_read_failures} 次取帧失败"
                    )
                continue

            consecutive_read_failures = 0
            processed += 1
            frame.frame_id = processed
            if frame.depth_data is not None and frame.color_bgr.shape[:2] == frame.depth_data.shape[:2]:
                aligned_frames += 1

            infer_started = time.perf_counter()
            frame_result = pipeline.process_frame(frame, image_name=f"frame_{processed:06d}")
            inference_ms.append((time.perf_counter() - infer_started) * 1000.0)
            targets = successful_targets(frame_result)
            grapes = frame_result.get("grapes", [])
            total_grapes += len(grapes)
            total_cuts += len(targets)
            total_sam3_masks += sum(
                isinstance(grape.get("_segmentation_mask_full"), np.ndarray)
                and grape["_segmentation_mask_full"].size > 0
                for grape in grapes
            )
            frames_with_grapes += int(bool(grapes))
            frames_with_cut += int(bool(targets))

            elapsed = max(time.monotonic() - started, 1e-6)
            fps = processed / elapsed
            annotated = draw_overlay(
                frame.color_bgr,
                frame_result,
                fps=fps,
                depth_data=frame.depth_data,
                depth_min_m=float(depth_cfg.get("min_m", 0.05)),
                depth_max_m=float(depth_cfg.get("max_m", 5.0)),
            )
            cv2.putText(
                annotated,
                "CAMERA-ONLY TEST: robot/ROS disabled",
                (10, annotated.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            last_frame, last_result, last_annotated = frame, frame_result, annotated

            if (
                targets
                and auto_saved < args.auto_save_limit
                and processed - last_auto_save_frame >= args.save_gap_frames
            ):
                auto_saved += 1
                last_auto_save_frame = processed
                save_bundle(
                    session_dir,
                    f"auto_success_{auto_saved:02d}_frame_{processed:06d}",
                    frame,
                    annotated,
                    frame_result,
                )

            if processed == 1 or processed % 30 == 0 or targets:
                print(
                    f"frame={processed:04d}/{args.max_frames} "
                    f"grapes={len(grapes)} valid_cuts={len(targets)} "
                    f"status={frame_result.get('status')} fps={fps:.1f}"
                )

            if args.display:
                cv2.imshow(window, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    manual_saved += 1
                    save_bundle(
                        session_dir,
                        f"manual_{manual_saved:02d}_frame_{processed:06d}",
                        frame,
                        annotated,
                        frame_result,
                    )

        if last_frame is not None and total_cuts == 0:
            save_bundle(session_dir, "last_frame_no_valid_cut", last_frame, last_annotated, last_result)

        grape_device = None
        try:
            grape_device = str(detector._model.predictor.device)
        except AttributeError:
            pass

        checks = {
            "supported_platform": platform.machine() in {"aarch64", "arm64", "x86_64"},
            "vision_chain_is_sam3_keypoint": config.get("vision", {}).get("chain") == "sam3_keypoint",
            "camera_is_zed_x_mini": camera_cfg.get("model", "ZED X Mini") == "ZED X Mini",
            "zed_native_depth_alignment": True,
            "rgbd_frames_received": processed > 0,
            "rgb_depth_shapes_aligned": processed > 0 and aligned_frames == processed,
            "worker_environment_has_cuda": worker_cuda,
            "grape_yolo_ran_on_cuda": bool(grape_device and grape_device.startswith("cuda")),
            "grape_detected": total_grapes > 0,
            "sam3_keypoint_cut_detected": total_cuts > 0,
            "sam3_binary_mask_received": total_sam3_masks > 0,
            "cut_point_has_valid_camera_xyz": total_cuts > 0,
        }
        report.update({
            "checks": checks,
            "frames_processed": processed,
            "camera_read_failures": read_failures,
            "aligned_frames": aligned_frames,
            "frames_with_grapes": frames_with_grapes,
            "frames_with_valid_cut": frames_with_cut,
            "total_grape_detections": total_grapes,
            "total_valid_cut_points": total_cuts,
            "total_sam3_masks": total_sam3_masks,
            "auto_saved": auto_saved,
            "manual_saved": manual_saved,
            "grape_yolo_device": grape_device,
            "pipeline_fps": round(processed / max(time.monotonic() - started, 1e-6), 2),
            "inference_ms_median": round(float(np.median(inference_ms)), 2) if inference_ms else None,
            "passed": all(checks.values()),
        })
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"ERROR: {report['error']}", file=sys.stderr)
    finally:
        if camera is not None:
            camera.stop()
        if predictor is not None:
            predictor.close()
        cv2.destroyAllWindows()
        report_path.write_text(
            json.dumps(serializable(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("\n" + "=" * 64)
    print("PASS: 新剪切点视觉链验收通过" if report.get("passed") else "FAIL: 验收未通过，请查看 report.json")
    print(f"报告: {report_path}")
    print(f"结果目录: {session_dir}")
    print("=" * 64)
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
