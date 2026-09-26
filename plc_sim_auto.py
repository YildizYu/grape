"""PLC 模拟器自动模式 — 纯逻辑, 无 socket。

模拟真实 PLC 的控制循环 (对齐 PLC_TCP_PROTOCOL.md §6):
- 已连接且收到 Box IDLE → 自动发 START (0x01), 进入等待终态
- DONE 后隔 retry_gap_s 自动再发; 非 FAIL_CUT 失败码按配置自动重试
- FAIL_CUT (0x0A) → 停机等人工确认 (按键 1 手动发 START 恢复)
- 发 START 后 result_timeout_s 内无终态 → 自动发 ABORT (0x02)
- 发 ABORT 后 abort_watchdog_s 内无 ABORTED → 告警一次
- 连续 watchdog_s 无任何帧 → 看门狗告警一次 (来帧自动恢复, 不刷屏)
- max_cycles 上限兜底 (0=不限, 防随机故障+自动重试死循环)

线程模型: 全部方法由外部单线程依次调用 (脚本侧经队列串行化),
tick 由脚本按 0.5s 周期驱动; 内部 RLock 仅作兜底。
"""

import threading
import time
from typing import Callable, Optional

from grape_stem_3d import plc_comm as plc

# ── 自动机状态 ──────────────────────────────────────────
S_DISABLED = "disabled"        # 自动模式关闭
S_ARMED = "armed"              # 等待 Box IDLE 即发 START
S_AWAIT_RESULT = "await_result"  # 已发 START, 等待终态
S_WAIT_RETRY = "wait_retry"    # 终态后冷却, 到期再武装
S_HALT_FAIL_CUT = "halt_fail_cut"  # FAIL_CUT 停机, 等人工
S_HALT_MANUAL = "halt_manual"  # 失败码不重试 (--no-retry-fail), 等人工
S_AWAIT_ABORT = "await_abort"  # 已发 ABORT, 等待 ABORTED


class AutoPlc:
    """PLC 自动模式状态机 (纯逻辑, 可注入时钟)。"""

    def __init__(
        self,
        send_fn: Callable[[int, bytes], None],
        logger: Callable[[str], None] = print,
        *,
        result_timeout_s: float = 60.0,
        abort_watchdog_s: float = 2.0,
        watchdog_s: float = 3.0,
        retry_gap_s: float = 1.0,
        retry_fail: bool = True,
        retry_after_abort: bool = True,
        max_cycles: int = 0,
        clock: Callable[[], float] = time.time,
    ):
        self._send = send_fn
        self._log = logger
        self._result_timeout_s = result_timeout_s
        self._abort_watchdog_s = abort_watchdog_s
        self._watchdog_s = watchdog_s
        self._retry_gap_s = retry_gap_s
        self._retry_fail = retry_fail
        self._retry_after_abort = retry_after_abort
        self._max_cycles = max_cycles
        self._clock = clock

        self._lock = threading.RLock()
        self._state = S_DISABLED
        self._connected = False
        self._last_state: Optional[int] = None  # Box 最近上报的状态码
        self._last_rx: Optional[float] = None   # 最近收帧时刻
        self._t0: Optional[float] = None        # START 发出时刻
        self._abort_t0: Optional[float] = None  # ABORT 发出时刻
        self._retry_since: Optional[float] = None  # 进入 wait_retry 时刻
        self._cycles = 0
        self._wd_alarmed = False
        self._abort_alarmed = False

    # ── 对外接口 ────────────────────────────────────────
    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def cycles(self) -> int:
        with self._lock:
            return self._cycles

    @property
    def last_box_state(self) -> Optional[int]:
        with self._lock:
            return self._last_state

    def set_enabled(self, on: bool) -> None:
        """自动模式开关。"""
        with self._lock:
            if on and self._state == S_DISABLED:
                self._state = S_ARMED
                self._log("[自动PLC] 自动模式: 开 (等待 Box IDLE 后自动发 START)")
            elif not on and self._state != S_DISABLED:
                self._state = S_DISABLED
                self._t0 = self._abort_t0 = self._retry_since = None
                self._log("[自动PLC] 自动模式: 关")

    # ── 事件入口 (脚本侧单线程依次调用) ─────────────────
    def on_connected(self, now: Optional[float] = None) -> None:
        now = self._clock() if now is None else now
        with self._lock:
            self._connected = True
            self._log("[自动PLC] Box 已连接 (等 IDLE 触发 START)")

    def on_disconnected(self, now: Optional[float] = None) -> None:
        now = self._clock() if now is None else now
        with self._lock:
            self._connected = False
            # 断线清计时, 除停机态外回到武装态
            self._t0 = self._abort_t0 = self._retry_since = None
            if self._state not in (S_DISABLED, S_HALT_FAIL_CUT, S_HALT_MANUAL):
                self._state = S_ARMED
            self._log("[自动PLC] Box 断线 (等待重连...)")

    def on_frame(self, cmd: int, data: bytes, now: Optional[float] = None) -> None:
        """收到 Box 帧: STATUS/ACK (任何帧均刷新看门狗)。"""
        now = self._clock() if now is None else now
        with self._lock:
            self._last_rx = now
            if self._wd_alarmed:
                self._wd_alarmed = False
                self._log("[自动PLC] 链路恢复 (收到帧)")
            if cmd == plc.CMD_BOX_STATUS and data:
                self._last_state = data[0]
                self._on_status(data[0], now)

    def on_manual_start(self, now: Optional[float] = None) -> None:
        """按键 1 手动发 START: 让自动机接管本轮 (FAIL_CUT 恢复入口)。"""
        now = self._clock() if now is None else now
        with self._lock:
            if self._state == S_DISABLED:
                return  # 自动关时不接管 (纯手动发帧)
            self._send(plc.CMD_PLC_START)
            self._t0 = now
            self._abort_t0 = None
            self._retry_since = None
            self._cycles += 1
            self._state = S_AWAIT_RESULT
            self._log(f"[自动PLC] 手动 START (第 {self._cycles} 轮, 自动机接管)")

    def tick(self, now: Optional[float] = None) -> None:
        """周期驱动: 看门狗 / 结果超时 / ABORT 看门狗 / 重试冷却。"""
        now = self._clock() if now is None else now
        with self._lock:
            if self._state == S_DISABLED:
                return
            self._check_watchdog(now)
            if self._state == S_AWAIT_RESULT and self._t0 is not None:
                if now - self._t0 > self._result_timeout_s:
                    self._send(plc.CMD_PLC_ABORT)
                    self._abort_t0 = now
                    self._state = S_AWAIT_ABORT
                    self._abort_alarmed = False
                    self._log(f"[自动PLC] {self._result_timeout_s:.0f}s 无终态, 自动发 ABORT {plc.build_frame(plc.CMD_PLC_ABORT).hex()}")
            elif self._state == S_AWAIT_ABORT and self._abort_t0 is not None:
                if (
                    now - self._abort_t0 > self._abort_watchdog_s
                    and not self._abort_alarmed
                ):
                    self._abort_alarmed = True
                    self._log(f"[自动PLC] ABORT 看门狗告警: {self._abort_watchdog_s:.0f}s 内未收到 ABORTED")
            elif self._state == S_WAIT_RETRY and self._retry_since is not None:
                if now - self._retry_since > self._retry_gap_s:
                    self._arm_or_start(now)

    # ── 内部 ────────────────────────────────────────────
    def _on_status(self, code: int, now: float) -> None:
        """状态帧驱动 (调用方已持锁)。"""
        if code == plc.STATE_IDLE:
            if self._state == S_ARMED:
                self._start(now)
            # 其他状态下的 IDLE (心跳) 不动作
        elif code == plc.STATE_DONE:
            if self._state == S_AWAIT_RESULT:
                self._to_retry(now)
        elif code == plc.STATE_FAIL_CUT:
            if self._state == S_AWAIT_RESULT:
                self._state = S_HALT_FAIL_CUT
                self._t0 = None
                self._log("[自动PLC] 收到 FAIL_CUT: 停机等人工确认剪刀/果藤 (按 1 手动恢复)")
        elif code in (plc.STATE_FAIL_NO_TARGET, plc.STATE_FAIL_INVALID_DEPTH,
                      plc.STATE_FAIL_NO_ROBOT_POSE, plc.STATE_FAIL_CAMERA,
                      plc.STATE_FAIL_GRASP_SEND, plc.STATE_FAIL_ROBOT_MOTION,
                      plc.STATE_FAIL_UNKNOWN):
            if self._state == S_AWAIT_RESULT:
                if self._retry_fail:
                    self._to_retry(now)
                else:
                    self._state = S_HALT_MANUAL
                    self._t0 = None
                    self._log(f"[自动PLC] 收到 {plc.state_name(code)}: 不自动重试, 等人工 (按 1 恢复)")
        elif code == plc.STATE_ABORTED:
            if self._state == S_AWAIT_ABORT:
                self._abort_t0 = None
                self._to_retry(now)
        # PICKING 心跳不迁移

    def _start(self, now: float) -> None:
        """发 START 并进入等待终态 (调用方已持锁, 已确认连接)。"""
        if not self._connected:
            return
        self._send(plc.CMD_PLC_START)
        self._t0 = now
        self._state = S_AWAIT_RESULT
        self._cycles += 1
        self._log(f"[自动PLC] 自动发 START (第 {self._cycles} 轮) {plc.build_frame(plc.CMD_PLC_START).hex()}")
        if self._max_cycles and self._cycles >= self._max_cycles:
            self.set_enabled(False)
            self._log(f"[自动PLC] 已达 --max-cycles {self._max_cycles} 上限, 自动模式关闭")

    def _to_retry(self, now: float) -> None:
        """终态 (DONE/失败/ABORTED) 后进入重试冷却 (调用方已持锁)。"""
        self._t0 = None
        self._abort_t0 = None
        self._retry_since = now
        self._state = S_WAIT_RETRY
        self._log(f"[自动PLC] 终态已收, {self._retry_gap_s:.1f}s 后重试")

    def _arm_or_start(self, now: float) -> None:
        """冷却到期: Box 已知 IDLE 则直接发 START, 否则武装等 IDLE。"""
        self._retry_since = None
        if self._connected and self._last_state == plc.STATE_IDLE:
            self._start(now)
        else:
            self._state = S_ARMED

    def _check_watchdog(self, now: float) -> None:
        if not self._connected:
            return
        if self._last_rx is not None and now - self._last_rx > self._watchdog_s:
            if not self._wd_alarmed:
                self._wd_alarmed = True
                self._log(f"[自动PLC] 看门狗告警: {self._watchdog_s:.0f}s 未收到 Box 任何帧")
