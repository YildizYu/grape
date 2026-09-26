#!/usr/bin/env python3
"""RGB-D 对齐验证工具。

显示彩色图、深度伪彩色图、叠加图，以及对齐信息。
点击任意点查看深度值和三维坐标。
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Validate RGB-D alignment")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save snapshot")
    return parser.parse_args()


clicked_point = None
clicked_depth = None
clicked_xyz = None


def on_mouse(event, x, y, flags, param):
    global clicked_point, clicked_depth, clicked_xyz
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        clicked_depth = None
        clicked_xyz = None


def main():
    global clicked_point, clicked_depth, clicked_xyz

    args = parse_args()

    try:
        from grape_stem_3d.camera_adapter import CameraAdapter
        from grape_stem_3d.coordinate_3d import pixel_to_camera_xyz
        from grape_stem_3d.depth_sampler import sample_depth
    except ImportError as e:
        print(f"Import error: {e}")
        sys.exit(1)

    camera = CameraAdapter(
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    try:
        intrinsics = camera.start()
    except ImportError:
        print("ERROR: pyzed not installed. Install with: pip install pyzed")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    window_name = "RGB-D Alignment Validation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 480)
    cv2.setMouseCallback(window_name, on_mouse)

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Save directory: {save_dir}")

    print("\nClick any pixel to see depth + 3D coordinates.")
    print("Press 's' to save snapshot, 'q' to quit.\n")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            color = frame.color_bgr.copy()
            depth_data = frame.depth_data

            # 深度伪彩色
            if depth_data is not None:
                if depth_data.dtype == np.uint16:
                    depth_m = depth_data.astype(np.float32) * camera.depth_scale
                else:
                    depth_m = depth_data.astype(np.float32)

                valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 5.0)
                depth_m_disp = np.clip(depth_m, 0.05, 5.0)
                depth_norm = ((depth_m_disp - 0.05) / 4.95 * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

                # 叠加 (半透明)
                overlay = cv2.addWeighted(color, 0.5, depth_color, 0.5, 0)
            else:
                overlay = color

            # 显示信息
            h, w = color.shape[:2]
            info_lines = [
                f"Resolution: {w}x{h}",
                f"Align: Depth-to-Color",
                f"fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f}",
                f"cx={intrinsics.cx:.1f} cy={intrinsics.cy:.1f}",
                f"Depth scale: {camera.depth_scale:.6f}",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(color, line, (10, 25 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 点击处理
            if clicked_point is not None:
                u, v = clicked_point
                if 0 <= u < w and 0 <= v < h:
                    # 使用与 pipeline 一致的鲁棒深度采样（7×7 patch 中位数）
                    depth_sample = sample_depth(
                        depth_data=depth_data,
                        u=float(u),
                        v=float(v),
                        depth_scale=camera.depth_scale,
                    )
                    clicked_depth = depth_sample.depth_m

                    if clicked_depth is not None and intrinsics is not None:
                        p3d = pixel_to_camera_xyz(intrinsics, u, v, clicked_depth)
                        clicked_xyz = (p3d.x, p3d.y, p3d.z)
                    else:
                        clicked_xyz = None

                    # 画十字
                    cv2.drawMarker(color, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                    cv2.circle(overlay, (u, v), 5, (0, 255, 255), 1)
                    depth_text = (f"({u},{v}) Z={clicked_depth:.3f}m"
                                  if clicked_depth is not None else f"({u},{v}) Z=N/A")
                    cv2.putText(color, depth_text,
                                (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    if clicked_xyz:
                        cv2.putText(color, f"XYZ=({clicked_xyz[0]:.3f},{clicked_xyz[1]:.3f},{clicked_xyz[2]:.3f})m",
                                    (u + 10, v + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                    clicked_point = None

            # 并排显示
            combined = np.hstack([color, overlay])
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                if save_dir:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(str(save_dir / f"rgb_{ts}.png"), color)
                    cv2.imwrite(str(save_dir / f"overlay_{ts}.png"), overlay)
                    if depth_data is not None:
                        np.save(str(save_dir / f"depth_{ts}.npy"), depth_data)
                    if intrinsics:
                        np.savez(str(save_dir / f"intrinsics_{ts}.npz"),
                                 fx=intrinsics.fx, fy=intrinsics.fy,
                                 cx=intrinsics.cx, cy=intrinsics.cy)
                    print(f"Saved snapshot: {ts}")
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
