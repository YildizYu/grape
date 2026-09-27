"""Client for the persistent SAM3 mask worker used by the production pipeline."""

from __future__ import annotations

import base64
import json
import os
import select
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class Sam3MaskPredictor:
    def __init__(
        self,
        worker_python: str,
        worker_script: str,
        checkpoint: str,
        device: str,
        box_padding: float,
        ready_timeout_s: float = 240.0,
        job_timeout_s: float = 120.0,
        logger=print,
    ):
        self._cmd = [
            str(worker_python), str(worker_script),
            "--checkpoint", str(checkpoint),
            "--device", str(device),
            "--box-padding", str(box_padding),
        ]
        self._ready_timeout_s = float(ready_timeout_s)
        self._job_timeout_s = float(job_timeout_s)
        self._log = logger
        self._proc = None
        self._lock = threading.Lock()
        self._job_seq = 0

    @staticmethod
    def _read_line(proc, timeout_s):
        ready, _, _ = select.select([proc.stdout], [], [], timeout_s)
        if not ready:
            return None
        line = proc.stdout.readline()
        return line.rstrip("\n") if line else None

    def _start(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        self.close()
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        deadline = time.time() + self._ready_timeout_s
        while time.time() < deadline:
            line = self._read_line(self._proc, deadline - time.time())
            if line is None:
                break
            if line.startswith("READY"):
                self._log(f"[SAM3] worker ready: {line}")
                return
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("fatal"):
                raise RuntimeError(payload["fatal"])
        self.close()
        raise RuntimeError("SAM3 worker did not become ready before timeout")

    def predict_batch(self, images, stem_bboxes):
        if len(images) != len(stem_bboxes):
            raise ValueError("images and stem_bboxes must have equal length")
        if not images:
            return []
        with self._lock:
            self._start()
            tmpdir = tempfile.mkdtemp(prefix="sam3_compare_")
            try:
                packets = []
                for index, (image, bbox) in enumerate(zip(images, stem_bboxes)):
                    path = Path(tmpdir) / f"{index:04d}.png"
                    if not cv2.imwrite(str(path), image):
                        raise RuntimeError(f"failed to write temporary image: {path}")
                    packets.append({
                        "path": str(path),
                        "stem_bbox": [float(value) for value in bbox],
                    })
                self._job_seq += 1
                request = {"job_id": self._job_seq, "packets": packets}
                self._proc.stdin.write(json.dumps(request) + "\n")
                self._proc.stdin.flush()
                line = self._read_line(self._proc, self._job_timeout_s)
                if line is None:
                    raise RuntimeError("SAM3 worker response timeout")
                response = json.loads(line)
                if response.get("job_id") != self._job_seq:
                    raise RuntimeError("SAM3 worker job_id mismatch")
                if response.get("error"):
                    raise RuntimeError(response["error"])
                results = response.get("results") or []
                for result in results:
                    encoded = result.pop("mask_png_base64", None)
                    result["mask"] = None
                    if encoded:
                        data = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
                        result["mask"] = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                return results
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    def segment(self, roi_image, stem_bbox_xyxy):
        """Production-compatible single ROI segmentation interface."""
        results = self.predict_batch([roi_image], [stem_bbox_xyxy])
        if not results:
            return {"status": "sam_failed", "mask": None, "error": "empty worker response"}
        return results[0]

    def warmup(self):
        """Load SAM3 without running a synthetic segmentation prompt."""
        with self._lock:
            self._start()

    def close(self):
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass
