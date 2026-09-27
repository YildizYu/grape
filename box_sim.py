"""ZED Box 模拟器状态机 — 纯逻辑, 与 PlcClient 解耦。

复现真实 Box 的协议行为 (见 PLC_TCP_PROTOCOL.md):
- START → ACK(成功) → PICKING → 六阶段推进 (检测/接近/抓取/合剪/放果/退回)
  → DONE → IDLE
- 忙时 START → ACK(忙), PlcClient 心跳自动重发 PICKING
- ABORT 任意时刻生效 → ACK(成功) → ≤0.5s 内 ABORTED → IDLE
- 未知命令 → ACK(未知命令)
- 故障注入: 指定阶段返回 FAIL 码 (one-shot), 或按概率随机注入

线程模型 (对齐 pick_flow.py): on_command 在 PlcClient 线程内回调, 只做
ACK/置位/spawn worker, 绝不阻塞; 采摘流程在单飞 worker 线程执行;
阶段推进经 sleep_fn 按 abort_poll_s (0.05s) 分片, 保证 ABORT 上报延迟
≤0.1s (协议要求 ≤0.5s)。终态补发由 PlcClient.send_status 负责, 本类
只按序调用, 不重复实现。
"""

import random
import threading
import time
from typing import Callable, Dict, Optional, Set, Tuple

from grape_stem_3d import frame_desc
from grape_stem_3d import plc_comm as plc

# ── 采摘阶段 ────────────────────────────────────────────
# 顺序与 PLC_TCP_PROTOCOL.md §5.1 一致
STAGE_NAMES = ("detect", "approach", "grasp", "cut", "place", "retreat")

# 各阶段默认时长 (秒), 合计 19s ∈ 协议 10~30s
DEFAULT_STAGE_DURATIONS: Dict[str, float] = {
    "detect": 2.0,
    "approach": 4.0,
    "grasp": 4.0,
    "cut": 3.0,
    "place": 4.0,
    "retreat": 2.0,
}

# 各阶段合法 FAIL 码池 (随机注入用, 与协议 §5 的失败归类一致)
FAULT_POOLS: Dict[str, Tuple[int, ...]] = {
    "detect": (
        plc.STATE_FAIL_NO_TARGET,
        plc.STATE_FAIL_INVALID_DEPTH,
        plc.STATE_FAIL_CAMERA,
    ),
    "approach": (
        plc.STATE_FAIL_NO_ROBOT_POSE,
        plc.STATE_FAIL_GRASP_SEND,
        plc.STATE_FAIL_ROBOT_MOTION,
    ),
    "grasp": (
        plc.STATE_FAIL_NO_ROBOT_POSE,
        plc.STATE_FAIL_GRASP_SEND,
        plc.STATE_FAIL_ROBOT_MOTION,
    ),
    "cut": (plc.STATE_FAIL_CUT,),
    "place": (plc.STATE_FAIL_ROBOT_MOTION, plc.STATE_FAIL_UNKNOWN),
    "retreat": (plc.STATE_FAIL_ROBOT_MOTION, plc.STATE_FAIL_UNKNOWN),
}


class BoxSim:
    """ZED Box 模拟器状态机。

    plc_client 只需具备 send_status / send_ack / send_frame 三个方法
    (真实 PlcClient 或测试 Fake 均可, duck-typed)。
    """

    def __init__(
        self,
        plc_client,
        *,
        stage_durations: Optional[Dict[str, float]] = None,
        terminal_idle_gap_s: float = 0.2,
        abort_poll_s: float = 0.05,
        rng: Optional[random.Random] = None,
        fault_prob: float = 0.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        logger: Callable[[str], None] = print,
    ):
        self._plc = plc_client
        self._durations = dict(DEFAULT_STAGE_DURATIONS)
        if stage_durations:
            self._durations.update(stage_durations)
        self._terminal_idle_gap_s = terminal_idle_gap_s
        self._abort_poll_s = abort_poll_s
        self._rng = rng if rng is not None else random.Random()
        self._fault_prob = fault_prob
        self._sleep_fn = sleep_fn
        self._clock = clock
        self._log = logger

        self._busy_lock = threading.Lock()
        self._busy = False
        self._abort_event = threading.Event()
        self._stop_event = threading.Event()
        self._current_stage: Optional[str] = None

        self._fault_lock = threading.Lock()
        self._pending_fault: Optional[Tuple[str, int]] = None  # one-shot 注入
        self._run_fault: Optional[Tuple[str, int]] = None     # 本轮随机注入
        self._random_fault = fault_prob > 0.0

        self._threads_lock = threading.Lock()
        self._threads: Set[threading.Thread] = set()

    # ── 供 PlcClient 注册的回调 (在其线程内调用, 必须非阻塞) ──
    def on_command(self, cmd: int, data: bytes) -> None:
        """PlcClient 命令回调: 只做 ACK/置位/spawn, 不阻塞。"""
        if cmd == plc.CMD_PLC_START:
            self.start_pick()
        elif cmd == plc.CMD_PLC_ABORT:
            self._send_ack(cmd, plc.ACK_OK)
            if self.is_busy:
                self._log("流程中收到 ABORT: 通知 worker 中止...")
                self._abort_event.set()
            else:
                # 空闲期 ABORT: 同步置忙 (防与 START 状态交错) 后走短序列
                self._log("空闲期收到 ABORT: 上报 ABORTED → IDLE")
                with self._busy_lock:
                    self._busy = True
                self._spawn(self._run_abort_idle)
        else:
            self._send_ack(cmd, plc.ACK_UNKNOWN_CMD)

    # ── 对外接口 (键盘线程, 线程安全) ───────────────────
    @property
    def is_busy(self) -> bool:
        """流程是否执行中 (加锁布尔, 对齐 pick_flow.is_busy)。"""
        with self._busy_lock:
            return self._busy

    @property
    def current_stage(self) -> Optional[str]:
        """当前阶段名 (空闲时为 None, UI 显示用)。"""
        return self._current_stage

    @property
    def stage_durations(self) -> Dict[str, float]:
        """各阶段时长配置 (副本, UI 显示用)。"""
        return dict(self._durations)

    @property
    def pending_fault(self) -> Optional[Tuple[str, int]]:
        """当前预设的 one-shot 故障 (阶段, FAIL码), 无则 None。"""
        with self._fault_lock:
            return self._pending_fault

    @property
    def random_fault_enabled(self) -> bool:
        """随机故障注入是否开启。"""
        with self._fault_lock:
            return self._random_fault

    def start_pick(self) -> bool:
        """注入一次 START (与真实收到 START 走同一路径)。忙时回 ACK(忙)。"""
        with self._busy_lock:
            if self._busy:
                self._send_ack(plc.CMD_PLC_START, plc.ACK_BUSY)
                return False
            self._busy = True
        self._send_ack(plc.CMD_PLC_START, plc.ACK_OK)
        self._spawn(self._run_flow)
        return True

    def set_fault(self, stage: str, code: int) -> None:
        """预设下轮故障注入: 流程走到 stage 时上报 FAIL 码 (one-shot)。

        Raises:
            ValueError: 阶段名或状态码非法
        """
        if stage not in STAGE_NAMES:
            raise ValueError(f"未知阶段 {stage!r}, 可选: {', '.join(STAGE_NAMES)}")
        if not 0 <= code <= 0xFF:
            raise ValueError(f"状态码越界: {code:#x}")
        with self._fault_lock:
            self._pending_fault = (stage, code)

    def set_random_fault(self, enabled: bool, prob: Optional[float] = None) -> None:
        """随机故障注入开关: 每轮按概率从阶段合法码池抽取。"""
        with self._fault_lock:
            self._random_fault = bool(enabled)
            if prob is not None:
                self._fault_prob = prob

    def manual_status(self, state_code: int) -> None:
        """手动上报任意状态码 (绕过状态机, 调试用)。"""
        self._send_status(state_code)

    def manual_frame(self, cmd: int, data: bytes = b"") -> None:
        """手动发送任意帧 (自动计算 CRC, 调试用)。"""
        self._log(f"→ 手动帧 cmd={cmd:#04x} data={data.hex() or '-'}")
        self._plc.send_frame(cmd, data)

    def stop(self) -> None:
        """停止: 置停止事件解除 dwell 阻塞 (不置 abort — 退出非真中止,
        不应上报 ABORTED), join 所有 worker。"""
        self._stop_event.set()
        with self._threads_lock:
            threads = list(self._threads)
        for t in threads:
            t.join(timeout=2.0)

    # ── 线程管理 ────────────────────────────────────────
    def _spawn(self, target: Callable[[], None]) -> None:
        """spawn daemon worker 并登记 (禁用 threading.Timer: 其默认
        daemon=False 会卡进程退出)。"""
        t = threading.Thread(target=self._wrap, args=(target,), daemon=True)
        with self._threads_lock:
            self._threads.add(t)
        t.start()

    def _wrap(self, target: Callable[[], None]) -> None:
        try:
            target()
        finally:
            with self._threads_lock:
                self._threads.discard(threading.current_thread())

    # ── 采摘流程 (worker 线程) ──────────────────────────
    def _run_flow(self) -> None:
        """一轮采摘流程: PICKING → 逐阶段推进 → DONE/FAIL → IDLE。"""
        self._abort_event.clear()
        # 随机注入在本轮开始时一次性抽取 (确定性测试友好)
        with self._fault_lock:
            if self._random_fault and self._rng.random() < self._fault_prob:
                stage = self._rng.choice(STAGE_NAMES)
                code = self._rng.choice(FAULT_POOLS[stage])
                self._run_fault = (stage, code)
                self._log(f"随机注入本轮: 阶段 {stage} → {plc.state_name(code)}")
            else:
                self._run_fault = None

        self._send_status(plc.STATE_PICKING)
        try:
            for stage in STAGE_NAMES:
                if self._abort_event.is_set():
                    return self._finish(plc.STATE_ABORTED)
                inject = self._take_fault(stage)
                if inject is not None:
                    self._log(f"阶段 {stage} 注入故障: {plc.state_name(inject)}")
                    return self._finish(inject)
                self._current_stage = stage
                self._log(f"阶段: {stage} ({self._durations[stage]:.1f}s)")
                if self._dwell(self._durations[stage]):
                    if self._stop_event.is_set():
                        return  # 程序退出, 不再上报 (真中止才报 ABORTED)
                    return self._finish(plc.STATE_ABORTED)
            self._finish(plc.STATE_DONE)
        except Exception as e:
            self._log(f"流程异常: {e!r}")
            self._finish(plc.STATE_FAIL_UNKNOWN)
        finally:
            self._current_stage = None
            with self._busy_lock:
                self._busy = False

    def _run_abort_idle(self) -> None:
        """空闲期 ABORT 的短命 worker: ABORTED → 间隔 → IDLE。"""
        try:
            self._send_status(plc.STATE_ABORTED)
            self._dwell(self._terminal_idle_gap_s)
            self._send_status(plc.STATE_IDLE)
        finally:
            with self._busy_lock:
                self._busy = False

    def _take_fault(self, stage: str) -> Optional[int]:
        """该阶段是否命中注入故障 (one-shot 优先于随机), 命中返回 FAIL 码。"""
        with self._fault_lock:
            if self._pending_fault is not None and self._pending_fault[0] == stage:
                code = self._pending_fault[1]
                self._pending_fault = None
                return code
            if self._run_fault is not None and self._run_fault[0] == stage:
                code = self._run_fault[1]
                self._run_fault = None
                return code
        return None

    def _finish(self, terminal_code: int) -> None:
        """终态上报序列: FAIL/DONE/ABORTED → 间隔 → IDLE。

        终态经 PlcClient.send_status 自动记录, 断线重连时由其补发。
        """
        self._send_status(terminal_code)
        self._dwell(self._terminal_idle_gap_s)
        self._send_status(plc.STATE_IDLE)

    def _dwell(self, seconds: float) -> bool:
        """分片可中断睡眠; True = 被 ABORT/停止 打断。"""
        deadline = self._clock() + seconds
        while not self._abort_event.is_set() and not self._stop_event.is_set():
            remain = deadline - self._clock()
            if remain <= 0:
                return False
            self._sleep_fn(min(remain, self._abort_poll_s))
        return True

    # ── 发送包装 (附带日志) ─────────────────────────────
    def _send_status(self, state: int) -> None:
        self._log(
            f"→ {frame_desc.describe_frame(plc.CMD_BOX_STATUS, bytes([state]))[2:]}"
        )
        self._plc.send_status(state)

    def _send_ack(self, cmd: int, result: int) -> None:
        self._log(
            f"→ {frame_desc.describe_frame(plc.CMD_BOX_ACK, bytes([cmd, result]))[2:]}"
        )
        self._plc.send_ack(cmd, result)
