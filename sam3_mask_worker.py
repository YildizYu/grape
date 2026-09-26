#!/usr/bin/env python3
"""Persistent SAM3 mask worker used by the production and comparison pipelines."""

import argparse
import base64
import contextlib
import json
import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--box-padding", type=float, default=0.15)
    parser.add_argument(
        "--sam3-dir",
        default=str(PROJECT_ROOT / "third_party" / "sam3"),
        help="SAM3 代码目录 (默认 third_party/sam3; 优化/微调版由主程序传 ../sam3)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # SAM3 代码目录必须在导入 grape_stem.sam3_segmenter 之前加入 sys.path
    sys.path.insert(0, str(args.sam3_dir))
    from grape_stem.sam3_segmenter import SAMSegmenter

    segmenter = SAMSegmenter(
        sam_type="sam3",
        checkpoint_path=Path(args.checkpoint),
        box_padding=args.box_padding,
        device=args.device,
    )
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            segmenter._load_model()
    except Exception as exc:
        print(json.dumps({"fatal": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 2
    print(f"READY sam3 load_s={time.perf_counter() - started:.3f}", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            job_id = request.get("job_id")
            packets = request.get("packets") or []
            results = []
            for packet in packets:
                image = cv2.imread(str(packet.get("path")), cv2.IMREAD_COLOR)
                if image is None:
                    results.append({"status": "unreadable_image"})
                    continue
                infer_started = time.perf_counter()
                with contextlib.redirect_stdout(sys.stderr):
                    result = segmenter.segment(image, packet.get("stem_bbox"))
                elapsed_ms = (time.perf_counter() - infer_started) * 1000.0
                response = {
                    "status": result.get("status", "unknown"),
                    "error": result.get("error"),
                    "score": result.get("score"),
                    "inference_ms": round(elapsed_ms, 3),
                }
                mask = result.get("mask")
                if mask is not None:
                    ok, encoded = cv2.imencode(".png", mask)
                    if ok:
                        response["mask_png_base64"] = base64.b64encode(
                            encoded
                        ).decode("ascii")
                results.append(response)
            print(json.dumps({"job_id": job_id, "results": results}), flush=True)
        except Exception as exc:
            print(
                json.dumps({
                    "job_id": request.get("job_id") if "request" in locals() else None,
                    "error": f"{type(exc).__name__}: {exc}",
                }),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

