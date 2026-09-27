"""Drive 信号 (0x83) 计数门 — 纯逻辑, 无 rclpy 依赖, 可单测。

包装 status sink: 计数 Box 上报的 DONE / FAIL_NO_TARGET, 达到阈值时向
PLC 发送 Drive (0x83) 事件帧, 通知其本点位采摘结束、可驱动底盘换位。

计数规则 (见 PLC_TCP_PROTOCOL.md §5.6):
  - DONE(0x02):           done_cnt += 1, miss_cnt = 0;
                          done_cnt >= done_threshold → 发 Drive
  - FAIL_NO_TARGET(0x03): miss_cnt += 1;
                          miss_cnt >= no_target_threshold → 发 Drive
  - 其他状态码:            不计数、不重置
  - 发 Drive 后:           两计数器清零, gate 置位 (暂停计数), 会话结束
  - on_plc_start():       gate 解除 + 会话开始 (PLC 已驱动底盘到位, 新点位)
  - gate 生效期间:         send_status 只转发不计数

START 会话门禁 (v3 新增, 对应流程文档"接收 START 后进入采摘状态"):
  - 开机未收到 START 时 _started=False, can_pick=False — 自动扫描/采摘
    一律待命 (无 PLC 通道的按键/单机模式除外, can_pick 恒 True)
  - 收到 START (ACK_OK) → _started=True; 发 Drive → _started=False
  - 自动扫描结果计数 note_scan_result(): 连续 no_target 达阈值同样
    触发 Drive (流程 §5.6 场景 B "连续 10 次未检测到有效果梗目标")

Drive 重发机制:
  发出 Drive 后若一直未收到下一次 START, 内部重发线程每隔
  retry_interval_s (默认 2 秒) 补发一次 Drive, 直到 on_plc_start()
  解除 gate。重复帧无副作用, PLC 按同一流程处理。

线程模型: send_status 可能被 pick_flow worker 线程与主循环并发调用
(相机故障上报等), on_plc_start 由 PLC/HMI 通信线程调用, 内部用
Condition 保护; 重发线程为 daemon, 进程退出即结束。
"""

import threading
from typing import Callable

from grape_stem_3d import plc_comm as plc


class DriveGateStatusSink:
    """Drive 计数包装层: 转发状态的同时计数 DONE / FAIL_NO_TARGET。"""

    def __init__(self, sink, plc_client, drive_cfg: dict, logger: Callable[[str], None] = print):
        cfg = drive_cfg or {}
        self._sink = sink
        self._plc_client = plc_client
        self._enabled = bool(cfg.get("enabled", False))
        self._done_threshold = int(cfg.get("done_threshold", 5))
        self._no_target_threshold = int(cfg.get("no_target_threshold", 10))
        # 未收到 START 时 Drive 重发间隔 (下限 0.1s, 防止误配 0 导致空转)
        self._retry_interval_s = max(
            float(cfg.get("retry_interval_s", 2.0)), 0.1
        )
        self._log = logger

        self._lock = threading.Condition()
        self._done_cnt = 0
        self._miss_cnt = 0
        self._gate = False  # True = Drive 已发, 暂停计数, 等下一次 START
        self._started = False  # START 会话: 收到 START 后 True, 发 Drive 后 False

        # Drive 重发线程 (daemon, 常驻): gate 生效期间每 retry_interval_s
        # 补发一次 0x83, on_plc_start() 解除 gate 后回到休眠。
        self._retry_thread = threading.Thread(
            target=self._retry_loop, daemon=True, name="DriveRetry"
        )
        self._retry_thread.start()

    # ── 状态接收 ────────────────────────────────────
    def send_status(self, state: int) -> None:
        """转发状态帧, 并计数 DONE / FAIL_NO_TARGET (接口同 PlcClient)。"""
        self._sink.send_status(state)
        with self._lock:
            if not self._enabled or self._gate:
                return
            if state == plc.STATE_DONE:
                self._done_cnt += 1
                self._miss_cnt = 0
                if self._done_cnt >= self._done_threshold:
                    self._log(
                        f"[DRIVE] 累计 DONE ×{self._done_cnt} ≥ "
                        f"{self._done_threshold} → 发送 Drive (0x83)"
                    )
                    self._fire_drive()
            elif state == plc.STATE_FAIL_NO_TARGET:
                self._miss_cnt += 1
                if self._miss_cnt >= self._no_target_threshold:
                    self._log(
                        f"[DRIVE] 连续 FAIL_NO_TARGET ×{self._miss_cnt} ≥ "
                        f"{self._no_target_threshold} → 发送 Drive (0x83)"
                    )
                    self._fire_drive()

    def _fire_drive(self) -> None:
        """发 Drive 事件帧, 清零计数器并置 gate (调用方须持有 _lock)。

        gate 置位会唤醒重发线程: 未收到 START 前每 retry_interval_s
        补发一次 Drive。会话同步结束 (can_pick → False, 待下一次 START)。
        """
        if self._plc_client is not None:
            self._plc_client.send_drive()
        else:
            self._log("[DRIVE] PLC 未启用, 0x83 未发送 (仅记录)")
        self._done_cnt = 0
        self._miss_cnt = 0
        self._gate = True
        self._started = False
        self._lock.notify_all()

    # ── START 恢复 ──────────────────────────────────
    def on_plc_start(self) -> None:
        """PLC START 被接受 (ACK_OK) 时调用: 解除 gate, 停止 Drive 重发。

        同时开启本点位采摘会话 (_started=True): 自动扫描/采摘
        只有在会话内才允许 (对应流程文档"接收 START 后进入采摘状态")。
        """
        with self._lock:
            if self._gate:
                self._log("[DRIVE] 收到 START, 恢复计数 (新点位), 停止补发")
            self._gate = False
            self._started = True
            self._lock.notify_all()

    # ── 查询 ────────────────────────────────────────
    @property
    def driving(self) -> bool:
        """gate 生效 (Drive 已发、等待 PLC START) 期间为 True。

        run_grasp_pipeline.py 的 auto_pick 扫描据此暂停,
        防止底盘移动中机械臂误触发采摘。
        """
        with self._lock:
            return self._gate

    @property
    def started(self) -> bool:
        """START 会话是否有效 (收到过 START 且本点位未结束)。"""
        with self._lock:
            return self._started

    @property
    def can_pick(self) -> bool:
        """自动扫描/采摘门禁。

        - 无 PLC 通道 (按键/单机模式): 恒 True, 保持旧行为
        - 有 PLC 通道: 要求 START 会话有效且 Drive gate 未生效
          (开机待命 / Drive 后等待下一次 START 期间均不允许自动扫描)
        """
        if self._plc_client is None:
            return True
        with self._lock:
            return self._started and not self._gate

    # ── 自动扫描计数 ────────────────────────────────
    def note_scan_result(self, ok: bool) -> None:
        """自动扫描结果计数 (流程 §5.6 场景 B: 连续 10 次无果 → Drive)。

        ok=True  (检测到有效目标): 无果连续计数清零;
        ok=False (未检测到目标):    miss_cnt += 1, 达阈值发 Drive。

        与 send_status 计数共用同一对计数器 (DONE 同样清零 miss_cnt)。
        仅在 START 会话内、gate 未生效、drive 启用时计数。
        """
        if self._plc_client is None:
            return
        with self._lock:
            if not self._enabled or self._gate or not self._started:
                return
            if ok:
                self._miss_cnt = 0
                return
            self._miss_cnt += 1
            if self._miss_cnt >= self._no_target_threshold:
                self._log(
                    f"[DRIVE] 自动扫描连续无果 ×{self._miss_cnt} ≥ "
                    f"{self._no_target_threshold} → 发送 Drive (0x83)"
                )
                self._fire_drive()

    # ── Drive 重发线程 ──────────────────────────────
    def _retry_loop(self) -> None:
        """gate 生效期间每 retry_interval_s 补发一次 Drive (0x83)。

        未连接/写入失败时 send_drive() 会置 PlcClient 的 pending 标志,
        由重连补发兜底; 收到 START (gate 解除) 立即停止。
        """
        while True:
            with self._lock:
                while not self._gate:
                    self._lock.wait()  # 休眠: 等 gate 置位
                # gate 已置位: 等一个重发间隔, 或被 START 唤醒
                self._lock.wait(self._retry_interval_s)
                if not self._gate:
                    continue  # START 已到, 回到休眠
                if self._plc_client is not None:
                    self._plc_client.send_drive()
                    self._log("[DRIVE] 未收到 START, 补发 Drive (0x83)")
