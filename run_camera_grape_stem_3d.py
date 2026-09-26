#!/usr/bin/env python3
"""实时葡萄果梗三维定位 — 主入口脚本。

运行方式:
    cd /home/user/grape/grape_stem_3d_zed_deploy
    python scripts/run_camera_grape_stem_3d.py [--config configs/fusion_pipeline.yaml]

键盘控制:
    q — 安全退出
    s — 保存当前帧 (RGB + 深度 + JSON)
    p — 暂停 / 继续
    d — 切换深度伪彩色叠加
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# 确保 src/ 在 path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time Grape Stem 3D Localization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fusion_pipeline.yaml",
        help="Path to fusion pipeline config (default: configs/fusion_pipeline.yaml)",
    )
    parser.add_argument(
        "--grapes-weights",
        type=str,
        default=None,
        help="Override grape detector weights path",
    )
    parser.add_argument(
        "--stem-weights",
        type=str,
        default=None,
        help="Override stem detector weights path",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default=None,
        help="Override SAM checkpoint path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (e.g. '0', 'cpu')",
    )
    parser.add_argument(
        "--grapes-conf",
        type=float,
        default=None,
        help="Grape detection confidence threshold",
    )
    parser.add_argument(
        "--stem-conf",
        type=float,
        default=None,
        help="Stem detection confidence threshold",
    )
    parser.add_argument(
        "--vision-chain",
        type=str,
        default=None,
        choices=["sam3_keypoint", "stem_yolo"],
        help="视觉链: sam3_keypoint(正式链) / stem_yolo(仅历史兼容)",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable display window (use --no-display for SSH/headless tests)",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable save-on-keypress",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames to process (0 = unlimited)",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warmup models before starting",
    )
    parser.add_argument(
        "--offline",
        type=str,
        default=None,
        help="Offline mode: path to saved frame .npz file or directory",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """加载融合 pipeline 配置。"""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def build_pipeline(args, config: dict):
    """构建所有组件并返回 (pipeline, camera, visualizer_config)。"""
    from grape_stem.grape_detector import GrapeDetector
    from grape_stem.stem_detector import StemDetector
    from grape_stem.stem_selector import StemSelector
    from grape_stem_3d.sam3_runtime import build_sam3_predictor
    from grape_stem_3d.realtime_pipeline import RealtimeGrapeStemPipeline

    models_cfg = config.get("models", {})
    grapes_cfg = config.get("grapes_detector", {})
    stem_cfg = config.get("stem_detector", {})
    sel_cfg = config.get("stem_selection", {})
    sam_cfg = config.get("sam", {})
    device = args.device or models_cfg.get("device", "0")

    # ── 葡萄检测器 ──────────────────────────
    grapes_path = Path(args.grapes_weights or models_cfg.get("grapes_weights", "weights/grapes/best.pt"))
    if not grapes_path.is_absolute():
        grapes_path = _PROJECT_ROOT / grapes_path

    grape_detector = GrapeDetector(
        weights_path=grapes_path,
        confidence=args.grapes_conf or grapes_cfg.get("confidence", 0.25),
        iou=grapes_cfg.get("iou", 0.50),
        imgsz=grapes_cfg.get("imgsz", 640),
        device=device,
    )
    print(f"Grape detector: {grapes_path}")

    # ── 果梗检测器 ──────────────────────────
    stem_path = Path(args.stem_weights or models_cfg.get("stem_weights", "weights/stem/best.pt"))
    if not stem_path.is_absolute():
        stem_path = _PROJECT_ROOT / stem_path

    stem_detector = StemDetector(
        weights_path=stem_path,
        confidence=args.stem_conf or stem_cfg.get("confidence", 0.15),
        iou=stem_cfg.get("iou", 0.50),
        imgsz=stem_cfg.get("imgsz", 960),
        device=device,
    )
    print(f"Stem detector: {stem_path}")

    # ── 果梗选择器 ──────────────────────────
    stem_selector = StemSelector(
        confidence_weight=sel_cfg.get("confidence_weight", 0.60),
        position_weight=sel_cfg.get("position_weight", 0.40),
    )

    sam_segmenter = None

    # ── 视觉链选择 ────────────────────────────────
    vision_cfg = config.setdefault("vision", {})
    chain = getattr(args, "vision_chain", None) or vision_cfg.get("chain", "sam3_keypoint")
    vision_cfg["chain"] = chain  # 回写, RealtimeGrapeStemPipeline 读此键

    keypoint_predictor = None
    if chain == "sam3_keypoint":
        sam_segmenter, sam_checkpoint_path, sam_worker_python = build_sam3_predictor(
            _PROJECT_ROOT, config, checkpoint_override=args.sam_checkpoint,
            device_override=device,
        )
        print(f"SAM3 checkpoint: {sam_checkpoint_path}")
        print(f"SAM3 worker Python: {sam_worker_python}")
    print(f"Vision chain: {chain}")

    # ── Pipeline ────────────────────────────
    pipeline = RealtimeGrapeStemPipeline(
        grape_detector=grape_detector,
        stem_detector=stem_detector,
        stem_selector=stem_selector,
        intrinsics=None,  # 由 camera adapter 设置
        sam_segmenter=sam_segmenter,
        keypoint_predictor=keypoint_predictor,
        config=config,
    )
    print("Pipeline ready.")

    return pipeline, None, config  # camera 在实时模式中创建


def run_realtime(args, config: dict):
    """实时相机模式。"""
    from grape_stem_3d.camera_adapter import CameraAdapter
    from grape_stem_3d.realtime_visualizer import draw_overlay

    camera_cfg = config.get("camera", {})
    depth_cfg = config.get("depth", {})
    runtime_cfg = config.get("runtime", {})
    output_cfg = config.get("output", {})

    # ── 构建 pipeline ──────────────────────
    pipeline, _, _ = build_pipeline(args, config)
    if args.warmup:
        pipeline.warmup()

    # ── 启动相机 (ZED X Mini) ─────────────
    camera = CameraAdapter(
        width=camera_cfg.get("color_width", 640),
        height=camera_cfg.get("color_height", 480),
        fps=camera_cfg.get("color_fps", 30),
        align_to=camera_cfg.get("align_to", "color"),
        zed_resolution=camera_cfg.get("zed_resolution", "HD1200"),
        zed_depth_mode=camera_cfg.get("zed_depth_mode", "NEURAL"),
        zed_depth_min_m=camera_cfg.get("zed_depth_min_m", 0.15),
        zed_depth_max_m=camera_cfg.get("zed_depth_max_m", 8.0),
    )

    try:
        intrinsics = camera.start()
        pipeline.intrinsics = intrinsics
    except ImportError as e:
        pipeline.close()
        print(f"ERROR: pyzed not installed. {e}")
        print("Install the ZED SDK and its matching pyzed package.")
        print("Or use --offline mode with saved frames.")
        sys.exit(1)
    except Exception as e:
        pipeline.close()
        print(f"ERROR: Camera start failed: {e}")
        sys.exit(1)

    # ── 显示窗口 ──────────────────────────
    window_name = "Grape Stem 3D Localization"
    if args.display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    # ── 输出目录 ──────────────────────────
    output_root = Path(args.output_dir or output_cfg.get("root", "outputs"))
    if not output_root.is_absolute():
        output_root = _PROJECT_ROOT / output_root
    session_dir = output_root / time.strftime("session_%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "images").mkdir(exist_ok=True)
    (session_dir / "depth").mkdir(exist_ok=True)
    (session_dir / "json").mkdir(exist_ok=True)
    (session_dir / "csv").mkdir(exist_ok=True)

    # ── 状态变量 ──────────────────────────
    paused = False
    show_depth = False
    frame_count = 0
    saved_count = 0
    fps = 0.0
    last_time = time.time()
    fps_alpha = 0.1  # EMA 平滑

    print("\n" + "=" * 50)
    print("Ready. Controls:")
    print("  q — quit")
    print("  s — save current frame")
    print("  p — pause/resume")
    print("  d — toggle depth overlay")
    print("=" * 50 + "\n")

    try:
        while True:
            if not paused:
                rgbd_frame = camera.read()
                if rgbd_frame is None:
                    continue

                frame_count += 1
                rgbd_frame.frame_id = frame_count

                # 处理帧
                frame_result = pipeline.process_frame(
                    rgbd_frame,
                    image_name=f"frame_{frame_count:06d}",
                )

                # 可视化
                if args.display:
                    display = draw_overlay(
                        color_bgr=rgbd_frame.color_bgr,
                        frame_result=frame_result,
                        fps=fps,
                        show_depth=show_depth,
                        depth_data=rgbd_frame.depth_data,
                        depth_min_m=depth_cfg.get("min_m", 0.05),
                        depth_max_m=depth_cfg.get("max_m", 5.0),
                    )
                    cv2.imshow(window_name, display)
            else:
                # 暂停时仍然消耗按键
                pass

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                if paused or frame_result is None:
                    continue
                saved_count += 1
                _save_frame(rgbd_frame, frame_result, session_dir, config)
                print(f"  Saved frame #{frame_count} ({saved_count} total)")
            elif key == ord("p"):
                paused = not paused
                print(f"  {'Paused' if paused else 'Resumed'}")
            elif key == ord("d"):
                show_depth = not show_depth
                print(f"  Depth overlay: {'ON' if show_depth else 'OFF'}")

            # FPS 计算
            now = time.time()
            dt = now - last_time
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = fps_alpha * instant_fps + (1 - fps_alpha) * fps
            last_time = now

            # 帧数限制
            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"Reached max frames ({args.max_frames}). Exiting.")
                break

    finally:
        camera.stop()
        pipeline.close()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames, saved {saved_count}.")
        print(f"Session: {session_dir}")


def _save_frame(rgbd_frame, frame_result, session_dir: Path, config: dict):
    """保存当前帧的所有数据。"""
    import json

    frame_id = rgbd_frame.frame_id
    output_cfg = config.get("output", {})

    # RGB
    if output_cfg.get("save_rgb", True):
        rgb_path = session_dir / "images" / f"frame_{frame_id:06d}.png"
        cv2.imwrite(str(rgb_path), rgbd_frame.color_bgr)

    # 深度 (numpy)
    if output_cfg.get("save_depth", True) and rgbd_frame.depth_data is not None:
        depth_path = session_dir / "depth" / f"frame_{frame_id:06d}.npy"
        np.save(str(depth_path), rgbd_frame.depth_data)

    if output_cfg.get("save_mask", True):
        mask_dir = session_dir / "masks"
        mask_dir.mkdir(exist_ok=True)
        for grape in frame_result.get("grapes", []):
            mask = grape.get("_segmentation_mask_full")
            if isinstance(mask, np.ndarray) and mask.size > 0:
                grape_id = int(grape.get("grape_id", 0))
                cv2.imwrite(
                    str(mask_dir / f"frame_{frame_id:06d}_grape_{grape_id:02d}_sam3_mask.png"),
                    mask,
                )

    # JSON
    if output_cfg.get("save_json", True):
        json_path = session_dir / "json" / f"frame_{frame_id:06d}.json"
        # 转换 tuple → list 以序列化
        serializable = _make_json_serializable(frame_result)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)


def _make_json_serializable(obj):
    """递归转换对象为 JSON 可序列化格式。"""
    if isinstance(obj, dict):
        return {
            k: _make_json_serializable(v)
            for k, v in obj.items()
            if not str(k).startswith("_")
        }
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def run_offline(source_path: str, args, config: dict):
    """离线模式：读取保存的帧文件运行 pipeline。"""
    from grape_stem_3d.types import RGBDFrame, CameraIntrinsics

    source = Path(source_path)
    if not source.exists():
        print(f"ERROR: Source not found: {source}")
        sys.exit(1)

    pipeline, _, _ = build_pipeline(args, config)
    if args.warmup:
        pipeline.warmup()

    if source.is_dir():
        npz_files = sorted(source.glob("*.npz"))
        print(f"Found {len(npz_files)} .npz files in {source}")
    else:
        npz_files = [source]

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        color_bgr = data.get("color_bgr")
        depth_data = data.get("depth_data")

        if color_bgr is None:
            print(f"  SKIP {npz_path.name}: no color_bgr")
            continue

        # 重建 CameraIntrinsics（如果有）
        intrinsics = None
        if "fx" in data:
            intrinsics = CameraIntrinsics(
                fx=float(data["fx"]),
                fy=float(data["fy"]),
                cx=float(data["cx"]),
                cy=float(data["cy"]),
            )
        pipeline.intrinsics = intrinsics

        rgbd_frame = RGBDFrame(
            color_bgr=color_bgr,
            depth_data=depth_data if depth_data is not None else np.zeros((480, 640), dtype=np.uint16),
            intrinsics=intrinsics,
            frame_id=int(npz_path.stem.split("_")[-1]) if npz_path.stem.split("_")[-1].isdigit() else 0,
        )

        frame_result = pipeline.process_frame(rgbd_frame, image_name=npz_path.stem)
        print(f"\n{'='*40}")
        print(f"Frame: {npz_path.name}")
        print(f"Status: {frame_result['status']}")
        for grape in frame_result.get("grapes", []):
            xyz = grape.get("peduncle_centroid_camera_xyz")
            # 【修复】果梗未检出行 centroid_xy 为 None, 必须做 None 保护
            # (参照 test_saved_rgbd.py 的 "N/A" 写法, 否则 None[0] TypeError)
            c2d = grape.get("peduncle_mask_centroid_xy")
            c2d_str = f"({c2d[0]:.0f},{c2d[1]:.0f})" if c2d else "N/A"
            print(f"  Grape {grape.get('grape_id')}: "
                  f"2D={c2d_str}"
                  f" | depth={grape.get('depth_value', 'N/A')}"
                  f" | 3D={xyz}")


def main():
    args = parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    print(f"Config: {config_path}")

    if args.offline:
        run_offline(args.offline, args, config)
    else:
        run_realtime(args, config)


if __name__ == "__main__":
    main()
