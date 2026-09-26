"""UNet 果梗三点预测客户端（管线侧, 无 TensorFlow 依赖）。

通过常驻子进程 worker（stem_keypoint_infer.py, 运行于 conda grape-stem-poc
环境）批量推理葡萄裁剪图的果梗三点, 协议为 stdin/stdout JSON-lines。

要点:
- worker 进程只启动一次, UNet 模型只加载一次（约十几秒）;
  崩溃/超时后自动重启（最多连续 2 次, 之后该次调用返回 None）
- 线程安全: 同一时刻只允许一笔批处理事务
- 本模块绝不 import tensorflow（TF 只存在于 worker 的 conda 环境）

用法:
    predictor = KeypointPredictor(
        worker_python="/home/user/miniconda3/envs/grape-stem-poc/bin/python3",
        worker_script=".../stem_keypoint_infer.py",
        model_path="/home/user/Grape_zed/Stem_Dataset/UNET_mobilenetv2.h5",
        threshold=0.93,
    )
    results = predictor.predict_batch([roi_bgr_1, roi_bgr_2])  # 与输入同序
    # results[i] = {"point_1":[x,y],"cut_point":[x,y],"point_3":[x,y],
    #               "path_length_224": float} 或 None 或 {"error": ...}
    predictor.close()
"""

import json
import os
import select
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

MAX_CONSECUTIVE_RESTARTS = 2


class KeypointPredictor:
    def __init__(
        self,
        worker_python: str,
        worker_script: str,
        model_path: str,
        threshold: float = 0.93,
        pose_offset: float = 12.0,
        min_component_area: int = 12,
        min_path_length: float = 20.0,
        ready_timeout_s: float = 180.0,
        job_timeout_s: float = 60.0,
        worker_env: Optional[dict] = None,
        logger: Callable[[str], None] = print,
    ):
        self._worker_python = worker_python
        self._worker_script = str(Path(worker_script))
        self._model_path = str(Path(model_path))
        self._threshold = threshold
        self._pose_offset = pose_offset
        self._min_component_area = min_component_area
        self._min_path_length = min_path_length
        self._ready_timeout_s = ready_timeout_s
        self._job_timeout_s = job_timeout_s
        self._worker_env = worker_env  # 额外环境变量（测试注入用）
        self._log = logger

        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._job_seq = 0

    # ── worker 生命周期 ───────────────────────────────
    def _spawn(self) -> Optional[subprocess.Popen]:
        cmd = [
            self._worker_python,
            self._worker_script,
            "--model", self._model_path,
            "--threshold", str(self._threshold),
            "--pose-offset", str(self._pose_offset),
            "--min-component-area", str(self._min_component_area),
            "--min-path-length", str(self._min_path_length),
        ]
        env = os.environ.copy()
        if self._worker_env:
            env.update(self._worker_env)
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=env,
            )
        except OSError as e:
            self._log(f"[KPT] worker 启动失败: {e}")
            return None
        self._log(f"[KPT] worker 已启动: {' '.join(cmd)}")
        return proc

    @staticmethod
    def _read_line(proc: subprocess.Popen, timeout_s: float) -> Optional[str]:
        """带超时读一行; 返回 None 表示超时/EOF/异常。"""
        deadline = time.time() + timeout_s
        line = ""
        while time.time() < deadline:
            remaining = deadline - time.time()
            try:
                ready, _, _ = select.select([proc.stdout], [], [], remaining)
            except (OSError, ValueError):
                return None
            if not ready:
                return None  # 超时
            chunk = proc.stdout.readline()
            if chunk == "":  # EOF
                return None
            line += chunk
            if line.endswith("\n"):
                return line.rstrip("\n")
        return None

    def _ensure_worker(self) -> bool:
        """确保 worker 存活且已输出 READY。"""
        if self._proc is not None and self._proc.poll() is None:
            return True
        self._kill()
        proc = self._spawn()
        if proc is None:
            return False
        # 等待 READY（跳过非 READY 行, 如 fatal JSON 报错）
        deadline = time.time() + self._ready_timeout_s
        while time.time() < deadline:
            line = self._read_line(proc, deadline - time.time())
            if line is None:
                self._log("[KPT] worker 等待 READY 超时")
                self._kill_proc(proc)
                return False
            if line.startswith("READY"):
                self._log(f"[KPT] worker 就绪: {line}")
                self._proc = proc
                return True
            self._log(f"[KPT] worker 启动输出: {line}")
        self._kill_proc(proc)
        return False

    @staticmethod
    def _kill_proc(proc: subprocess.Popen):
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _kill(self):
        self._kill_proc(self._proc)
        self._proc = None

    # ── 对外接口 ──────────────────────────────────────
    def predict_batch(self, images: List[np.ndarray]) -> List[Optional[dict]]:
        """批量推理裁剪图, 返回与输入同序的结果列表。

        images: BGR 裁剪图 (H,W,3) uint8
        结果: {"point_1": [x,y], "cut_point": [x,y], "point_3": [x,y],
               "path_length_224": float} / None(无有效果梗) / {"error": str}
        worker 连续崩溃时整批返回 None。
        """
        count = len(images)
        if count == 0:
            return []
        with self._lock:
            restarts = 0
            while restarts <= MAX_CONSECUTIVE_RESTARTS:
                if not self._ensure_worker():
                    return [None] * count
                result = self._transact(images)
                if result is not None:
                    return result
                # worker 超时/崩溃: 重启后重试一次批处理
                restarts += 1
                self._log(f"[KPT] worker 事务失败, 第 {restarts} 次重启")
                self._kill()
            self._log("[KPT] worker 连续失败, 本批返回 None")
            return [None] * count

    def _transact(self, images: List[np.ndarray]) -> Optional[List[Optional[dict]]]:
        """一次批处理事务; 失败返回 None（worker 需要重启）。"""
        tmpdir = tempfile.mkdtemp(prefix="kpt_")
        paths = []
        try:
            for i, image in enumerate(images):
                path = os.path.join(tmpdir, f"{i:04d}.png")
                if not cv2.imwrite(path, image):
                    self._log(f"[KPT] 写入临时图失败: {path}")
                    return None
                paths.append(path)

            self._job_seq += 1
            request = {"job_id": self._job_seq, "paths": paths}
            try:
                self._proc.stdin.write(json.dumps(request) + "\n")
                self._proc.stdin.flush()
            except (OSError, ValueError) as e:
                self._log(f"[KPT] 发送任务失败: {e}")
                return None

            line = self._read_line(self._proc, self._job_timeout_s)
            if line is None:
                self._log("[KPT] worker 应答超时")
                return None
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                self._log(f"[KPT] worker 应答非 JSON: {line[:120]}")
                return None
            if response.get("job_id") != self._job_seq:
                self._log(f"[KPT] job_id 不匹配: {response.get('job_id')}")
                return None
            results = response.get("results") or {}
            return [results.get(path) for path in paths]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def close(self):
        """结束 worker 进程。"""
        with self._lock:
            self._kill()
            self._log("[KPT] worker 已关闭")
