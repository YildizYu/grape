#!/usr/bin/env python3
"""离线测试模式。

读取保存的 RGB-D 帧，运行完整 pipeline 验证。
在没有相机硬件时测试所有检测和坐标转换逻辑。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Offline test with saved RGB-D frames")
    parser.add_argument("--source", type=str, required=True,
                        help="Directory of .npz files or single .npz file")
    parser.add_argument("--config", type=str, default="configs/fusion_pipeline.yaml")
    parser.add_argument("--grapes-weights", type=str, default=None)
    parser.add_argument("--stem-weights", type=str, default=None)
    parser.add_argument("--sam-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-visualization", action="store_true", help="Save annotated images")
    return parser.parse_args()


def main():
    args = parse_args()

    from grape_stem.config import load_yaml
    from grape_stem.grape_detector import GrapeDetector
    from grape_stem.stem_detector import StemDetector
    from grape_stem.stem_selector import StemSelector
    from grape_stem_3d.realtime_pipeline import RealtimeGrapeStemPipeline
    from grape_stem_3d.types import RGBDFrame, CameraIntrinsics
    from grape_stem_3d.realtime_visualizer import draw_overlay

    # 配置
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path
    config = load_yaml(config_path)

    models_cfg = config.get("models", {})
    depth_cfg = config.get("depth", {})
    device = args.device or models_cfg.get("device", "0")

    # 模型
    grapes_path = args.grapes_weights or models_cfg.get("grapes_weights", "weights/grapes/best.pt")
    stem_path = args.stem_weights or models_cfg.get("stem_weights", "weights/stem/best.pt")
    if not Path(grapes_path).is_absolute():
        grapes_path = Path(__file__).parent.parent / grapes_path
    if not Path(stem_path).is_absolute():
        stem_path = Path(__file__).parent.parent / stem_path

    grape_detector = GrapeDetector(weights_path=Path(grapes_path), device=device)
    stem_detector = StemDetector(weights_path=Path(stem_path), imgsz=960, device=device)
    stem_selector = StemSelector()

    pipeline = RealtimeGrapeStemPipeline(
        grape_detector=grape_detector,
        stem_detector=stem_detector,
        stem_selector=stem_selector,
        config=config,
    )
    pipeline.warmup()

    # 源文件
    source = Path(args.source)
    if not source.exists():
        print(f"ERROR: Source not found: {source}")
        sys.exit(1)

    npz_files = sorted(source.glob("*.npz")) if source.is_dir() else [source]
    print(f"Processing {len(npz_files)} frames...\n")

    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for i, npz_path in enumerate(npz_files):
        data = np.load(npz_path, allow_pickle=True)
        color_bgr = data.get("color_bgr")
        depth_data = data.get("depth_data")

        if color_bgr is None:
            print(f"  Skip {npz_path.name}: no color_bgr")
            continue

        # 重建内参
        intrinsics = None
        if "fx" in data.files:
            intrinsics = CameraIntrinsics(
                fx=float(data["fx"]),
                fy=float(data["fy"]),
                cx=float(data["cx"]),
                cy=float(data["cy"]),
                width=int(data.get("width", color_bgr.shape[1])),
                height=int(data.get("height", color_bgr.shape[0])),
            )
            pipeline.intrinsics = intrinsics

        if depth_data is None:
            depth_data = np.zeros((color_bgr.shape[0], color_bgr.shape[1]), dtype=np.uint16)

        frame = RGBDFrame(
            color_bgr=color_bgr,
            depth_data=depth_data,
            intrinsics=intrinsics,
            frame_id=i,
        )

        frame_result = pipeline.process_frame(frame, image_name=npz_path.stem)
        all_results.append(frame_result)

        # 输出
        status = frame_result["status"]
        n_grapes = len(frame_result.get("grapes", []))
        print(f"[{i+1}/{len(npz_files)}] {npz_path.name} — {status} — {n_grapes} grapes")

        for grape in frame_result.get("grapes", []):
            gid = grape["grape_id"]
            centroid = grape.get("peduncle_mask_centroid_xy")
            depth = grape.get("depth_value")
            xyz = grape.get("peduncle_centroid_camera_xyz")
            c2d = f"({centroid[0]:.0f},{centroid[1]:.0f})" if centroid else "N/A"
            d_str = f"{depth:.3f}m" if depth else "N/A"
            xyz_str = f"({xyz['x']:.3f},{xyz['y']:.3f},{xyz['z']:.3f})m" if xyz else "N/A"
            print(f"    Grape {gid}: 2D={c2d} Z={d_str} XYZ={xyz_str} [{grape['status']}]")

        # 可视化
        if args.save_visualization and output_dir:
            annotated = draw_overlay(
                color_bgr, frame_result,
                depth_min_m=depth_cfg.get("min_m", 0.05),
                depth_max_m=depth_cfg.get("max_m", 5.0),
            )
            out_path = output_dir / f"{npz_path.stem}_annotated.png"
            import cv2
            cv2.imwrite(str(out_path), annotated)

    # 保存所有结果
    if output_dir:
        results_path = output_dir / "offline_results.json"
        # 序列化
        def serialize(obj):
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            return obj

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(serialize(all_results), f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to {results_path}")

    # 总结
    success_count = sum(1 for r in all_results if r["status"] == "success")
    total_grapes = sum(len(r.get("grapes", [])) for r in all_results)
    grapes_with_3d = sum(
        sum(1 for g in r.get("grapes", [])
            if g.get("peduncle_centroid_camera_xyz") is not None)
        for r in all_results
    )
    print(f"\n{'='*50}")
    print(f"Total frames: {len(all_results)}")
    print(f"Success: {success_count}")
    print(f"Total grapes: {total_grapes}")
    print(f"Grapes with 3D: {grapes_with_3d}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
