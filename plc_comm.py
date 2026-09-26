"""PLC ↔ ZED Box 55 AA TCP 通信模块。

协议规格见项目根目录 PLC_TCP_PROTOCOL.md，要点：
- Box 作 TCP 客户端，主动连接 PLC 服务器，断线自动重连
- 帧结构: 55 AA | LEN(2,大端) | CMD(1) | DATA(LEN) | CRC16(2,低字节在前)
- CRC16-Modbus，计算范围为 CMD 到 DATA 末尾（与剪刀手 RS485 协议一致）
- Box 周期上报状态帧 0x81（兼作心跳），状态变化时立即上报
- PLC 命令: 0x01 START 开始采摘 / 0x02 ABORT 中止

线程模型: PlcClient 为独立守护线程（与 run_grasp_pipeline.py 中 ROS2
spin 线程的模式一致），收到的命令通过 on_command 回调抛给主流程处理，
主流程通过 send_status() 上报状态（线程安全）。
"""

import select
import socket
import struct
import threading
import time
from typing import Callable, List, Optional, Tuple

# ── 帧常量 ────────────────────────────────────────────
FRAME_HEAD = b"\x55\xaa"
HEAD_SIZE = 2
LEN_SIZE = 2
CRC_SIZE = 2
MIN_FRAME_SIZE = HEAD_SIZE + LEN_SIZE + 1 + CRC_SIZE  # 7 字节 (LEN=0 时)
MAX_DATA_LEN = 255

# ── 命令码: PLC → Box ─────────────────────────────────
CMD_PLC_START = 0x01  # 开始采摘
CMD_PLC_ABORT = 0x02  # 中止当前流程

# ── 命令码: Box → PLC ─────────────────────────────────
CMD_BOX_STATUS = 0x81  # 状态上报 (DATA: 1 字节状态码)
CMD_BOX_ACK = 0x82     # 命令应答 (DATA: [被应答CMD, 结果])
CMD_BOX_DRIVE = 0x83   # 本点位采摘结束, 通知 PLC 驱动底盘至下一点位 (DATA 空)

# ── ACK 结果码 ────────────────────────────────────────
ACK_OK = 0x00          # 成功
ACK_BUSY = 0x01        # 忙（流程执行中）
ACK_UNKNOWN_CMD = 0x02  # 未知命令

# ── Box 状态码 (CMD_BOX_STATUS 的 DATA) ───────────────
STATE_IDLE = 0x00          # 空闲待命
STATE_PICKING = 0x01       # 采摘流程执行中
STATE_DONE = 0x02          # 一次采摘完成
STATE_FAIL_NO_TARGET = 0x03  # 未检测到葡萄/果梗
STATE_FAIL_INVALID_DEPTH = 0x04  # 目标点深度无效
STATE_FAIL_NO_ROBOT_POSE = 0x05  # 未收到机械臂位姿
STATE_FAIL_CAMERA = 0x06   # 相机读取/启动失败
STATE_FAIL_GRASP_SEND = 0x07  # 抓取命令发送失败
STATE_FAIL_ROBOT_MOTION = 0x08  # 机械臂运动异常（超时）
STATE_ABORTED = 0x09       # 流程被中止
STATE_FAIL_CUT = 0x0A      # 剪枝失败（合剪/开剪超时或剪刀响应丢失）
STATE_FAIL_UNKNOWN = 0xFF  # 其他未归类错误

# 终态事件（断线重连时需补发一次, 见 PlcClient.send_status）
_TERMINAL_STATES = (
    STATE_DONE,
    STATE_FAIL_NO_TARGET,
    STATE_FAIL_INVALID_DEPTH,
    STATE_FAIL_NO_ROBOT_POSE,
    STATE_FAIL_CAMERA,
    STATE_FAIL_GRASP_SEND,
    STATE_FAIL_ROBOT_MOTION,
    STATE_ABORTED,
    STATE_FAIL_CUT,
    STATE_FAIL_UNKNOWN,
)

# 发送侧命令码名称 (log_sent 日志用; Box 不会发 START/ABORT, 仅备查)
_SENT_CMD_NAMES = {
    CMD_PLC_START: "START",
    CMD_PLC_ABORT: "ABORT",
    CMD_BOX_STATUS: "STATUS",
    CMD_BOX_ACK: "ACK",
    CMD_BOX_DRIVE: "DRIVE",
}

# 状态码 → 名称（日志/调试用）
STATE_NAMES = {
    STATE_IDLE: "IDLE",
    STATE_PICKING: "PICKING",
    STATE_DONE: "DONE",
    STATE_FAIL_NO_TARGET: "FAIL_NO_TARGET",
    STATE_FAIL_INVALID_DEPTH: "FAIL_INVALID_DEPTH",
    STATE_FAIL_NO_ROBOT_POSE: "FAIL_NO_ROBOT_POSE",
    STATE_FAIL_CAMERA: "FAIL_CAMERA",
    STATE_FAIL_GRASP_SEND: "FAIL_GRASP_SEND",
    STATE_FAIL_ROBOT_MOTION: "FAIL_ROBOT_MOTION",
    STATE_ABORTED: "ABORTED",
    STATE_FAIL_CUT: "FAIL_CUT",
    STATE_FAIL_UNKNOWN: "FAIL_UNKNOWN",
}


def state_name(state: int) -> str:
    """状态码的可读名称。"""
    return STATE_NAMES.get(state, f"UNKNOWN_{state:#x}")


def is_terminal_state(state: int) -> bool:
    """是否为终态事件 (DONE/FAIL_*/ABORTED, 断线/新连接时需补发)。"""
    return state in _TERMINAL_STATES


# ── CRC16-Modbus ──────────────────────────────────────
def crc16_modbus(data: bytes) -> int:
    """CRC16-Modbus (poly 0xA001 反射, 初值 0xFFFF)。

    与剪刀手 RS485 MODBUS-RTU 协议同款校验。
    校验向量: b"\\x01\\x06\\x00\\x03\\x00\\x01" → 0x0AB8
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


# ── 帧编解码 ──────────────────────────────────────────
def build_frame(cmd: int, data: bytes = b"") -> bytes:
    """构造完整帧: 55 AA | LEN | CMD | DATA | CRC16。

    Args:
        cmd: 命令码 (0~255)
        data: 数据段 (≤255 字节)

    Raises:
        ValueError: data 超过 255 字节
    """
    if len(data) > MAX_DATA_LEN:
        raise ValueError(f"data too long: {len(data)} > {MAX_DATA_LEN}")
    header = FRAME_HEAD + struct.pack(">H", len(data)) + bytes([cmd])
    crc = crc16_modbus(bytes([cmd]) + data)  # CRC 覆盖范围: CMD + DATA
    return header + data + struct.pack("<H", crc)


def parse_frame(payload: bytes) -> Tuple[int, bytes]:
    """校验并解析一帧的 CMD+DATA+CRC 部分（不含帧头）。

    Args:
        payload: LEN + CMD + DATA + CRC，即帧头之后的所有字节

    Returns:
        (cmd, data)

    Raises:
        ValueError: LEN 不匹配或 CRC 校验失败
    """
    if len(payload) < LEN_SIZE + 1 + CRC_SIZE:
        raise ValueError(f"frame too short: {len(payload)} bytes")
    (data_len,) = struct.unpack(">H", payload[:LEN_SIZE])
    expected = LEN_SIZE + 1 + data_len + CRC_SIZE
    if len(payload) != expected:
        raise ValueError(
            f"length mismatch: LEN={data_len}, got {len(payload)} bytes, "
            f"expected {expected}"
        )
    body = payload[LEN_SIZE : LEN_SIZE + 1 + data_len]  # CMD + DATA
    (crc_recv,) = struct.unpack("<H", payload[-CRC_SIZE:])
    if crc16_modbus(body) != crc_recv:
        raise ValueError(
            f"CRC mismatch: computed {crc16_modbus(body):#06x}, "
            f"received {crc_recv:#06x}"
        )
    cmd = body[0]
    return cmd, body[1:]


class FrameParser:
    """流式帧解析器 — 处理 TCP 粘包/半包/干扰字节。

    用法:
        parser = FrameParser()
        frames = parser.feed(chunk)  # [(cmd, data), ...]
    """

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> List[Tuple[int, bytes]]:
        """喂入新字节，返回本次完整解析出的帧列表 [(cmd, data), ...]。

        - 半包: 缓冲不足时等待下次 feed
        - 干扰字节: 帧头前的孤立字节直接丢弃
        - 坏帧: LEN 超限或 CRC 错误时丢弃 1 字节后重新同步（逐字节
          重搜帧头，可容忍帧头错位/干扰 55 AA）
        """
        self._buffer.extend(chunk)
        frames: List[Tuple[int, bytes]] = []

        while True:
            # 搜索帧头
            head_pos = self._buffer.find(FRAME_HEAD)
            if head_pos < 0:
                # 保留末尾 1 字节 (可能是 0x55 的前半)
                keep = 1 if self._buffer and self._buffer[-1] == 0x55 else 0
                del self._buffer[: len(self._buffer) - keep]
                return frames
            if head_pos > 0:
                del self._buffer[:head_pos]  # 丢弃干扰字节

            # 帧头后的最小长度: LEN(2) + CMD(1) + CRC(2)
            if len(self._buffer) < HEAD_SIZE + LEN_SIZE + 1 + CRC_SIZE:
                return frames  # 半包, 等待更多数据
            (data_len,) = struct.unpack(">H", self._buffer[HEAD_SIZE:HEAD_SIZE + LEN_SIZE])
            if data_len > MAX_DATA_LEN:
                # LEN 非法 (帧头错位), 丢弃 1 字节后重新同步
                del self._buffer[:1]
                continue
            total = HEAD_SIZE + LEN_SIZE + 1 + data_len + CRC_SIZE
            if len(self._buffer) < total:
                return frames  # 半包, 等待更多数据

            frame_bytes = bytes(self._buffer[:total])
            try:
                cmd, data = parse_frame(frame_bytes[HEAD_SIZE:])
            except ValueError:
                # CRC 错误: 候选帧头为误判, 只丢弃 1 字节 (0x55) 后
                # 重新同步 — 真实帧头可能在误判帧的剩余字节中, 逐字节
                # 重搜可恢复 (数据区内的 55 AA 会被 CRC 校验拦住)
                del self._buffer[:1]
                continue
            del self._buffer[:total]
            frames.append((cmd, data))

    def reset(self):
        """清空缓冲（断线重连时调用）。"""
        self._buffer.clear()


# ── TCP 客户端 ────────────────────────────────────────
class PlcClient(threading.Thread):
    """Box 侧 PLC TCP 客户端（守护线程）。

    - 启动后循环: 连接 → 上报当前状态 → 收帧分发/周期心跳 → 断线重连
    - send_status() / send_frame() 线程安全
    - 收到 PLC 命令通过 on_command(cmd, data) 回调（在 PlcClient 线程内
      调用，回调应只做轻量操作，如设置标志位）

    Args:
        host: PLC 服务器 IP
        port: PLC 服务器端口
        on_command: 命令回调
        reconnect_interval_s: 断线重连间隔（默认 3.0 秒）
        status_interval_s: 心跳/状态周期上报间隔（默认 1.0 秒）
        logger: 日志函数（默认 print）
        log_sent: True 时打印发出的每一帧（链路调试用; 默认 False,
            避免 Box 主程序被心跳发送日志刷屏）
    """

    def __init__(
        self,
        host: str,
        port: int,
        on_command: Optional[Callable[[int, bytes], None]] = None,
        reconnect_interval_s: float = 3.0,
        status_interval_s: float = 1.0,
        logger: Callable[[str], None] = print,
        log_sent: bool = False,
    ):
        super().__init__(daemon=True, name="PlcClient")
        self._host = host
        self._port = port
        self._on_command = on_command
        self._reconnect_interval_s = reconnect_interval_s
        self._status_interval_s = status_interval_s
        self._log = logger
        self._log_sent = log_sent

        self._stop_event = threading.Event()
        self._sock: Optional[socket.socket] = None
        self._send_lock = threading.Lock()
        self._current_state = STATE_IDLE
        self._connected = False
        self._last_terminal_event: Optional[int] = None
        self._pending_drive = False  # 断线期间未送达的 DRIVE, 重连后补发

    # ── 对外接口（线程安全） ───────────────────────
    def set_on_command(self, on_command: Optional[Callable[[int, bytes], None]]) -> None:
        """设置命令回调（可在 start() 之前调用）。"""
        self._on_command = on_command

    def send_status(self, state: int):
        """设置当前状态并立即上报（线程安全）。

        终态事件（DONE/FAIL_*/ABORTED）会被记录, 断线重连成功后
        先补发一次终态再发当前状态——防止断线窗口内 PLC 漏掉
        采摘结果帧, 避免 PLC 等到 60s 超时才 ABORT。
        """
        self._current_state = state
        if is_terminal_state(state):
            self._last_terminal_event = state
        self.send_frame(CMD_BOX_STATUS, bytes([state]))

    def send_ack(self, cmd: int, result: int):
        """应答 PLC 命令（线程安全）。"""
        self.send_frame(CMD_BOX_ACK, bytes([cmd, result]))

    def send_drive(self) -> bool:
        """发送 0x83 DRIVE 事件帧（线程安全）。

        未送达 (未连接/写入失败) 时置 pending 标志, 重连成功后补发一次:
        DRIVE 决定底盘是否换位, 丢失会导致 PLC 一直等待, 必须可靠送达。
        """
        sent = self.send_frame(CMD_BOX_DRIVE)
        if not sent:
            self._pending_drive = True
        return sent

    def send_frame(self, cmd: int, data: bytes = b"") -> bool:
        """发送一帧（线程安全，未连接时静默丢弃）。

        Returns:
            True = 已写入 socket (送达对端 TCP 缓冲);
            False = 未连接或写入失败 (调用方可据此做 pending 补发)
        """
        frame = build_frame(cmd, data)
        with self._send_lock:
            sock = self._sock
        if sock is None:
            return False
        try:
            with self._send_lock:
                sock.sendall(frame)
            if self._log_sent:
                name = _SENT_CMD_NAMES.get(cmd, f"0x{cmd:02x}")
                desc = ""
                if cmd == CMD_BOX_STATUS and data:
                    desc = f" → 状态: {state_name(data[0])}"
                self._log(
                    f"[PLC] → {name} cmd={cmd:#04x} "
                    f"data={data.hex() or '-'}{desc}"
                )
            return True
        except OSError:
            # 连接已断, 主循环会在下次重连后补发状态
            return False

    @property
    def connected(self) -> bool:
        """是否已连接 PLC。"""
        return self._connected

    # ── 线程主体 ────────────────────────────────────
    def run(self):
        while not self._stop_event.is_set():
            try:
                self._log(f"[PLC] 连接 {self._host}:{self._port} ...")
                self._sock = socket.create_connection(
                    (self._host, self._port), timeout=3.0
                )
                self._connected = True
                self._log(f"[PLC] 已连接 {self._host}:{self._port}")

                # 重连后先补发断线期间可能丢失的终态事件, 再上报当前状态
                if self._last_terminal_event is not None:
                    self.send_frame(
                        CMD_BOX_STATUS, bytes([self._last_terminal_event])
                    )
                    self._last_terminal_event = None
                if self._pending_drive:
                    # 断线窗口内的 DRIVE 事件补发 (换位信号丢失会导致底盘停摆)
                    if self.send_frame(CMD_BOX_DRIVE):
                        self._pending_drive = False
                        self._log("[PLC] 已补发 DRIVE (0x83)")
                self.send_frame(CMD_BOX_STATUS, bytes([self._current_state]))

                self._recv_loop()
            except (OSError, socket.timeout) as e:
                self._log(f"[PLC] 连接失败: {e}，{self._reconnect_interval_s:.0f}s 后重连")
            finally:
                self._close_sock()

            self._wait_stop(self._reconnect_interval_s)

        self._log("[PLC] 客户端已退出")

    def _recv_loop(self):
        """收帧 + 周期心跳循环，直到断线或停止。"""
        parser = FrameParser()
        last_status_time = time.time()
        self._sock.settimeout(0.5)  # 半秒超时以便周期心跳/退出检查

        while not self._stop_event.is_set():
            # 周期心跳: 重发当前状态
            now = time.time()
            if now - last_status_time >= self._status_interval_s:
                self.send_frame(CMD_BOX_STATUS, bytes([self._current_state]))
                last_status_time = now

            # 收数据
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                continue
            except OSError as e:
                if self._stop_event.is_set():
                    return  # stop() 关闭 socket 导致的正常退出, 不打错误
                self._log(f"[PLC] 接收错误: {e}")
                return

            if not chunk:
                self._log("[PLC] PLC 服务器关闭了连接")
                return

            for cmd, data in parser.feed(chunk):
                # 注意: 命令码 0x01/0x02 与状态码 0x01/0x02 重号,
                # 收帧日志必须用命令码名称表 (不能用 STATE_NAMES)
                rx_name = {
                    CMD_PLC_START: "START",
                    CMD_PLC_ABORT: "ABORT",
                }.get(cmd, f"0x{cmd:02x}")
                self._log(
                    f"[PLC] ← {rx_name} cmd={cmd:#04x} data={data.hex() or '-'}"
                )
                if self._on_command is not None:
                    try:
                        self._on_command(cmd, data)
                    except Exception as e:
                        self._log(f"[PLC] 命令回调异常: {e}")

    def _close_sock(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        self._connected = False

    def _wait_stop(self, seconds: float):
        """等待停止事件，最多 seconds 秒。"""
        self._stop_event.wait(seconds)

    def stop(self):
        """停止线程（关闭 socket 促使其退出）。"""
        self._stop_event.set()
        with self._send_lock:
            if self._sock is not None:
                try:
                    self._sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        self._close_sock()
