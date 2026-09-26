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
from typing import Optional

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
        sam3_source_dir: Optional[str] = None,
        ready_timeout_s: float = 240.0,
        job_timeout_s: float = 120.0,
        logger=print,
    ):
        worker_script_path = Path(worker_script).resolve()
        package_dir = worker_script_path.parent
        source_dir = package_dir.parent
        project_root = source_dir.parent
        # SAM3 代码目录: 默认 third_party/sam3 (原始版), 可由配置指向
        # 优化/微调后的目录 (fusion_pipeline.yaml 的 sam3_runtime.sam3_source_dir)
        if sam3_source_dir:
            sam3_dir = Path(sam3_source_dir)
            if not sam3_dir.is_absolute():
                sam3_dir = project_root / sam3_dir
            sam3_dir = sam3_dir.resolve()
        else:
            sam3_dir = (project_root / "third_party" / "sam3").resolve()
        worker_module = f"{package_dir.name}.{worker_script_path.stem}"
        self._cmd = [
            str(worker_python), "-m", worker_module,
            "--checkpoint", str(checkpoint),
            "--device", str(device),
            "--box-padding", str(box_padding),
            "--sam3-dir", str(sam3_dir),
        ]
        self._cwd = str(project_root)
        self._env = os.environ.copy()
        self._env.setdefault("SAM3_LOW_MEMORY", "1")
        self._env.setdefault("HF_HUB_OFFLINE", "1")
        self._env.setdefault("TRANSFORMERS_OFFLINE", "1")
        python_paths = [str(source_dir), str(sam3_dir)]
        inherited_pythonpath = self._env.get("PYTHONPATH")
        if inherited_pythonpath:
            python_paths.append(inherited_pythonpath)
        self._env["PYTHONPATH"] = os.pathsep.join(python_paths)
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
            cwd=self._cwd,
            env=self._env,
        )
        deadline = time.time() + self._ready_timeout_s
        while time.time() < deadline:
            line = self._read_line(self._proc, deadline - time.time())
            if line is None:
                return_code = self._proc.poll()
                if return_code is not None:
                    self.close()
                    raise RuntimeError(
                        f"SAM3 worker exited before ready (exit code {return_code})"
                    )
                break
            if line.startswith("READY"):
                self._log(f"[SAM3] worker ready: {line}")
                return
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("fatal"):
                message = payload["fatal"]
                self.close()
                raise RuntimeError(message)
        self.close()
        raise RuntimeError(
            f"SAM3 worker did not become ready within {self._ready_timeout_s:.0f}s"
        )

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
