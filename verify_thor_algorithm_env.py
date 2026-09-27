#!/usr/bin/env python3
"""Verify the Thor CUDA environment and existing YOLO weights without changing them."""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "reports" / "thor_algorithm_env.json"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    report = {
        "test": "Thor CUDA and YOLO environment acceptance",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "machine": platform.machine(),
        "checks": {},
        "models": {},
        "passed": False,
    }
    checks = report["checks"]
    checks["aarch64_platform"] = platform.machine() == "aarch64"

    try:
        import cv2
        import torch
        import ultralytics
        from ultralytics import YOLO

        report["versions"] = {
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        }
        checks["cuda_available"] = torch.cuda.is_available()
        report["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )

        for name, relative_path, image_size in (
            ("grapes", "weights/grapes/best.pt", 640),
            ("stem", "weights/stem/best.pt", 960),
        ):
            path = PROJECT_ROOT / relative_path
            model_result = {"path": str(path), "exists": path.is_file()}
            try:
                model = YOLO(str(path))
                source = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                predictions = model.predict(
                    source=source,
                    imgsz=image_size,
                    device=0,
                    verbose=False,
                )
                predictor_device = str(model.predictor.device)
                model_result.update({
                    "loaded": True,
                    "inference_device": predictor_device,
                    "cuda_inference": (
                        bool(predictions) and predictor_device.startswith("cuda")
                    ),
                    "result_count": len(predictions),
                })
            except Exception as exc:
                model_result.update({
                    "loaded": False,
                    "cuda_inference": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            report["models"][name] = model_result
            checks[f"{name}_model_cuda_inference"] = model_result["cuda_inference"]

        from grape_stem_3d.stem_keypoint_infer_torch import (
            load_model,
            torch_predict,
        )

        unet_path = PROJECT_ROOT / "weights" / "unet_mobilenetv2.onnx"
        unet_result = {"path": str(unet_path), "exists": unet_path.is_file()}
        try:
            unet = load_model(unet_path)
            first_parameter = next(unet.parameters())
            output = torch_predict(
                unet, np.zeros((224, 224, 3), dtype=np.uint8)
            )
            unet_result.update({
                "loaded": True,
                "device": str(first_parameter.device),
                "cuda_inference": bool(first_parameter.is_cuda),
                "output_shape": list(output.shape),
                "output_finite": bool(np.isfinite(output).all()),
            })
        except Exception as exc:
            unet_result.update({
                "loaded": False,
                "cuda_inference": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
        report["models"]["keypoint_unet"] = unet_result
        checks["keypoint_unet_cuda_inference"] = bool(
            unet_result["cuda_inference"] and unet_result.get("output_finite")
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        checks["cuda_available"] = False

    report["passed"] = all(checks.values())
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Environment report: {REPORT_PATH}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
