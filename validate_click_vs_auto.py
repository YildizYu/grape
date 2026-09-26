#!/usr/bin/env python3
"""自动检测 vs 鼠标点击 — 三维坐标对比验证。

同时运行自动 pipeline 检测和鼠标点击，
比较自动质心和手动点击的三维坐标。
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np


clicked_pixel = None
clicked_depth = None
clicked_xyz = None


def on_mouse(event, x, y, flags, param):
    global clicked_pixel, clicked_depth, clicked_xyz
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixel = (x, y)
        clicked_depth = None
        clicked_xyz = None


def parse_args():
    parser = argparse.ArgumentParser(description="Validate auto vs manual click 3D")
    parser.add_argument("--config", type=str, default="configs/fusion_pipeline.yaml")
    parser.add_argument("--report", type=str, default="reports/click_auto_validation.csv")
    parser.add_argument("--grapes-weights", type=str, default=None)
    parser.add_argument("--stem-weights", type=str, default=None)
    parser.add_argument("--sam-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=50, help="Max comparison samples to collect")
    return parser.parse_args()


def main():
    global clicked_pixel, clicked_depth, clicked_xyz

    args = parse_args()

    # 加载配置
    from grape_stem.config import load_yaml
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path
    config = load_yaml(config_path)

    # 导入 pipeline
    from grape_stem.grape_detector import GrapeDetector
    from grape_stem.stem_detector import StemDetector
    from grape_stem.stem_selector import StemSelector
    from grape_stem_3d.camera_adapter import CameraAdapter
    from grape_stem_3d.realtime_pipeline import RealtimeGrapeStemPipeline
    from grape_stem_3d.coordinate_3d import pixel_to_camera_xyz
    from grape_stem_3d.depth_sampler import sample_depth
    from grape_stem_3d.realtime_visualizer import draw_overlay

    models_cfg = config.get("models", {})
    device = args.device or models_cfg.get("device", "0")

    # 探测器
    grapes_path = Path(args.grapes_weights or models_cfg.get("grapes_weights", "weights/grapes/best.pt"))
    if not grapes_path.is_absolute():
        grapes_path = Path(__file__).parent.parent / grapes_path

    stem_path = Path(args.stem_weights or models_cfg.get("stem_weights", "weights/stem/best.pt"))
    if not stem_path.is_absolute():
        stem_path = Path(__file__).parent.parent / stem_path

    grape_detector = GrapeDetector(weights_path=grapes_path, confidence=0.25, device=device)
    stem_detector = StemDetector(weights_path=stem_path, confidence=0.15, imgsz=960, device=device)
    stem_selector = StemSelector()

    pipeline = RealtimeGrapeStemPipeline(
        grape_detector=grape_detector,
        stem_detector=stem_detector,
        stem_selector=stem_selector,
        config=config,
    )

    # 相机
    camera = CameraAdapter()

    try:
        intrinsics = camera.start()
        pipeline.intrinsics = intrinsics
    except ImportError:
        print("ERROR: pyzed not installed.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    pipeline.warmup()

    window_name = "Click vs Auto Validation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, on_mouse)

    report_path = Path(__file__).resolve().parent.parent / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    comparisons = []
    print(f"\nCollecting up to {args.max_samples} click-vs-auto comparisons.")
    print("Click a pixel to record. Auto detection runs each frame.")
    print("Press 'q' to quit and save report.\n")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            # 自动 pipeline
            frame_result = pipeline.process_frame(frame)

            # 可视化
            display = draw_overlay(frame.color_bgr, frame_result)
            h, w = display.shape[:2]

            # 处理手动点击
            if clicked_pixel is not None:
                u, v = clicked_pixel
                if 0 <= u < w and 0 <= v < h:
                    # 使用与 pipeline 一致的鲁棒深度采样（7×7 patch 中位数）
                    manual_depth_sample = sample_depth(
                        depth_data=frame.depth_data,
                        u=float(u),
                        v=float(v),
                        depth_scale=camera.depth_scale,
                    )
                    manual_depth = manual_depth_sample.depth_m

                    if manual_depth is not None and intrinsics is not None:
                        manual_xyz = pixel_to_camera_xyz(intrinsics, u, v, manual_depth)
                    else:
                        manual_xyz = None

                    # 找到最近的自动检测果梗
                    for grape in frame_result.get("grapes", []):
                        auto_centroid = grape.get("peduncle_mask_centroid_xy")
                        auto_xyz = grape.get("peduncle_centroid_camera_xyz")
                        auto_depth = grape.get("depth_value")

                        if auto_centroid is None:
                            continue

                        # 像素距离
                        pix_dist = np.sqrt((auto_centroid[0] - u) ** 2 + (auto_centroid[1] - v) ** 2)

                        # 只记录距离 < 50 像素的
                        if pix_dist < 50 and auto_xyz and manual_xyz:
                            auto_xyz_arr = np.array([auto_xyz["x"], auto_xyz["y"], auto_xyz["z"]])
                            manual_xyz_arr = np.array([manual_xyz.x, manual_xyz.y, manual_xyz.z])
                            euclid_dist = np.sqrt(np.sum((auto_xyz_arr - manual_xyz_arr) ** 2))

                            comparison = {
                                "auto_pixel": f"({auto_centroid[0]:.1f},{auto_centroid[1]:.1f})",
                                "click_pixel": f"({u},{v})",
                                "pixel_distance": f"{pix_dist:.1f}",
                                "auto_xyz": f"({auto_xyz['x']:.4f},{auto_xyz['y']:.4f},{auto_xyz['z']:.4f})",
                                "click_xyz": f"({manual_xyz.x:.4f},{manual_xyz.y:.4f},{manual_xyz.z:.4f})",
                                "euclidean_distance_m": f"{euclid_dist:.4f}",
                            }
                            comparisons.append(comparison)
                            print(f"  #{len(comparisons)}: pixel_dist={pix_dist:.1f}px, "
                                  f"3D_dist={euclid_dist:.4f}m | "
                                  f"auto_z={auto_xyz['z']:.3f}m, click_z={manual_xyz.z:.3f}m")

                            # 标记
                            cv2.circle(display, (u, v), 8, (0, 255, 0), 2)
                            cv2.putText(display, f"d={euclid_dist:.3f}m", (u + 15, v),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                    # 显示手动点击信息
                    depth_text = (f"Click: ({u},{v}) Z={manual_depth:.3f}m"
                                  if manual_depth is not None else f"Click: ({u},{v}) Z=N/A")
                    cv2.putText(display, depth_text,
                                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    if manual_xyz:
                        cv2.putText(display, f"XYZ: ({manual_xyz.x:.3f},{manual_xyz.y:.3f},{manual_xyz.z:.3f})m",
                                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    clicked_pixel = None

            cv2.putText(display, f"Comparisons: {len(comparisons)}/{args.max_samples}",
                        (w - 300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or len(comparisons) >= args.max_samples:
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    # 保存报告
    if comparisons:
        fieldnames = ["auto_pixel", "click_pixel", "pixel_distance",
                      "auto_xyz", "click_xyz", "euclidean_distance_m"]
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparisons)

        print(f"\nSaved {len(comparisons)} comparisons to {report_path}")

        # 统计
        euclidean_dists = [float(c["euclidean_distance_m"]) for c in comparisons]
        print(f"  Mean 3D distance: {np.mean(euclidean_dists):.4f}m")
        print(f"  Median 3D distance: {np.median(euclidean_dists):.4f}m")
        print(f"  Max 3D distance: {np.max(euclidean_dists):.4f}m")
    else:
        print("\nNo comparisons collected.")


if __name__ == "__main__":
    main()
