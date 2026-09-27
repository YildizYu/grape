#!/usr/bin/env python3
"""Headless Gemini 336 acceptance test with machine-readable evidence."""

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grape_stem_3d.camera_adapter import CameraAdapter


def orbbec_usb_speed_mbps():
    """Return the negotiated USB speed for VID 2bc5, without libusb access."""
    for vendor_path in Path("/sys/bus/usb/devices").glob("*/idVendor"):
        try:
            if vendor_path.read_text().strip().lower() != "2bc5":
                continue
            product_path = vendor_path.with_name("idProduct")
            if product_path.read_text().strip().lower() != "0803":
                continue
            return float(vendor_path.with_name("speed").read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Verify Orbbec Gemini 336 RGB-D capture")
    parser.add_argument("--config", default="configs/fusion_pipeline.yaml")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--min-fps", type=float, default=10.0)
    parser.add_argument("--min-valid-ratio", type=float, default=0.01)
    parser.add_argument("--output-dir", default="reports/orbbec_acceptance")
    return parser.parse_args()


def depth_preview(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 5.0)
    normalized = np.zeros(depth_m.shape, dtype=np.uint8)
    if valid.any():
        near, far = np.percentile(depth_m[valid], [2, 98])
        if far <= near:
            far = near + 0.001
        normalized[valid] = np.clip(
            (depth_m[valid] - near) / (far - near) * 255, 0, 255
        ).astype(np.uint8)
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        camera_cfg = yaml.safe_load(handle).get("camera", {})

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"

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

    report = {
        "test": "Orbbec Gemini 336 RGB-D acceptance",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sdk_version": version("pyorbbecsdk2"),
        "config": str(config_path),
        "requested_frames": args.frames,
        "checks": {},
        "passed": False,
    }
    usb_speed = orbbec_usb_speed_mbps()
    report["usb_speed_mbps"] = usb_speed

    frames = []
    try:
        intrinsics = camera.start()
        started_at = time.monotonic()
        max_attempts = max(args.frames * 3, args.frames + 10)
        for _ in range(max_attempts):
            frame = camera.read()
            if frame is not None:
                frames.append(frame)
            if len(frames) >= args.frames:
                break

        elapsed = time.monotonic() - started_at
        if not frames:
            raise RuntimeError("No complete RGB-D frame was received")

        last = frames[-1]
        depth_m = last.depth_data.astype(np.float32) * last.depth_scale
        valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 5.0)
        valid_ratios = []
        for frame in frames:
            metres = frame.depth_data.astype(np.float32) * frame.depth_scale
            valid_ratios.append(float(
                (np.isfinite(metres) & (metres > 0.05) & (metres < 5.0)).mean()
            ))

        preview = depth_preview(depth_m)
        overlay = cv2.addWeighted(last.color_bgr, 0.6, preview, 0.4, 0)
        cv2.imwrite(str(output_dir / "color.png"), last.color_bgr)
        cv2.imwrite(str(output_dir / "depth_preview.png"), preview)
        cv2.imwrite(str(output_dir / "alignment_overlay.png"), overlay)
        np.save(str(output_dir / "depth_raw.npy"), last.depth_data)

        observed_fps = len(frames) / elapsed if elapsed else 0.0
        checks = {
            "usb3_link": usb_speed is not None and usb_speed >= 5000,
            "device_is_gemini_336": "336" in camera.device_model,
            "received_all_frames": len(frames) == args.frames,
            "color_is_bgr_uint8": (
                last.color_bgr.dtype == np.uint8
                and last.color_bgr.ndim == 3
                and last.color_bgr.shape[2] == 3
            ),
            "depth_is_uint16": last.depth_data.dtype == np.uint16,
            "depth_aligned_to_color_shape": (
                last.depth_data.shape == last.color_bgr.shape[:2]
            ),
            "intrinsics_match_color": (
                intrinsics.width == last.color_bgr.shape[1]
                and intrinsics.height == last.color_bgr.shape[0]
                and intrinsics.fx > 0
                and intrinsics.fy > 0
            ),
            "depth_scale_is_m_per_count": 0 < last.depth_scale < 0.01,
            "valid_depth_ratio": float(np.median(valid_ratios)) >= args.min_valid_ratio,
            "d2c_alignment_enabled": last.metadata.get("alignment", "").startswith(
                "depth_to_color_"
            ),
            "capture_fps": observed_fps >= args.min_fps,
        }
        report.update({
            "device": {
                "model": camera.device_model,
                "serial_number": camera.device_serial,
                "alignment_mode": camera.active_alignment_mode,
            },
            "received_frames": len(frames),
            "observed_fps": round(observed_fps, 2),
            "color_shape": list(last.color_bgr.shape),
            "depth_shape": list(last.depth_data.shape),
            "depth_scale_m_per_count": last.depth_scale,
            "valid_depth_ratio_median": float(np.median(valid_ratios)),
            "valid_depth_median_m": float(np.median(depth_m[valid])) if valid.any() else None,
            "intrinsics": {
                "fx": intrinsics.fx, "fy": intrinsics.fy,
                "cx": intrinsics.cx, "cy": intrinsics.cy,
                "width": intrinsics.width, "height": intrinsics.height,
            },
            "checks": checks,
            "passed": all(checks.values()),
            "evidence": [
                "color.png", "depth_raw.npy", "depth_preview.png",
                "alignment_overlay.png", "report.json",
            ],
        })
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        camera.stop()
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Acceptance report: {report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
