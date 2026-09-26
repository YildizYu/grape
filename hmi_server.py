"""上位机 (HMI) ↔ ZED Box 55 AA TCP 服务模块。

与 PLC 通道共用同一帧协议 (见 PLC_TCP_PROTOCOL.md)，区别只在角色:
- Box 作 TCP 服务器监听 (默认 0.0.0.0:5000)，上位机主动连接
- 帧格式/命令码/状态码与 PLC 通道完全一致，上位机可复用 PLC 侧解析代码
- Box 周期上报状态帧 0x81 (兼作心跳)，状态变化立即上报
- 上位机命令: 0x01 START 开始采摘 / 0x02 ABORT 中止
- 同一时刻服务一个客户端；客户端断开后回到 accept 继续等待

线程模型: HmiServer 为独立守护线程（与 PlcClient 的模式一致），收到
的命令通过 on_command 回调抛给主流程处理（回调应只做轻量操作，如置
标志位），主流程通过 send_status() 上报状态（线程安全）。
"""

import socket
import threading
import time
from typing import Callable, Optional

from grape_stem_3d import plc_comm as plc


class HmiServer(threading.Thread):
    """Box 侧上位机 TCP 服务器（守护线程）。

    - 启动后循环: 监听 → 接受连接 → 补发终态/上报当前状态 →
      收帧分发/周期心跳 → 客户端断开后回到 accept
    - send_status() / send_frame() 线程安全
    - 收到上位机命令通过 on_command(cmd, data) 回调（在 HmiServer 线程内
      调用，回调应只做轻量操作，如设置标志位）

    Args:
        host: 监听地址 (0.0.0.0 = 所有网卡)
        port: 监听端口 (默认 5000；0 = 随机端口, 测试用, 见 bound_port)
        on_command: 命令回调
        status_interval_s: 心跳/状态周期上报间隔（默认 1.0 秒）
        logger: 日志函数（默认 print）
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        on_command: Optional[Callable[[int, bytes], None]] = None,
        status_interval_s: float = 1.0,
        logger: Callable[[str], None] = print,
    ):
        super().__init__(daemon=True, name="HmiServer")
        self._host = host
        self._port = port
        self._on_command = on_command
        self._status_interval_s = status_interval_s
        self._log = logger

        self._stop_event = threading.Event()
        self._srv_sock: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._send_lock = threading.Lock()
        self._current_state = plc.STATE_IDLE
        self._last_terminal_event: Optional[int] = None
        self._connected = False
        self._bound_port: Optional[int] = None

    # ── 对外接口（线程安全） ───────────────────────
    def set_on_command(self, on_command: Optional[Callable[[int, bytes], None]]) -> None:
        """设置命令回调（可在 start() 之前调用）。"""
        self._on_command = on_command

    def send_status(self, state: int):
        """设置当前状态并立即上报（线程安全）。

        终态事件（DONE/FAIL_*/ABORTED）会被记录, 新上位机连入时先补发
        一次终态再发当前状态（与 PLC 通道的断线补发语义一致）。
        """
        self._current_state = state
        if plc.is_terminal_state(state):
            self._last_terminal_event = state
        self.send_frame(plc.CMD_BOX_STATUS, bytes([state]))

    def send_ack(self, cmd: int, result: int):
        """应答上位机命令（线程安全）。"""
        self.send_frame(plc.CMD_BOX_ACK, bytes([cmd, result]))

    def send_drive(self) -> bool:
        """发送 0x83 DRIVE 事件帧（与 PlcClient 接口一致, 供 DriveGate 调用）。

        HMI-only 部署 (plc.enabled=false, hmi.enabled=true) 时 Drive 事件
        也上报给上位机, 由上位机驱动底盘换位。
        """
        sent = self.connected
        self.send_frame(plc.CMD_BOX_DRIVE, b"\x00")
        return sent

    def send_frame(self, cmd: int, data: bytes = b""):
        """发送一帧（线程安全，无上位机连接时静默丢弃）。"""
        frame = plc.build_frame(cmd, data)
        with self._send_lock:
            conn = self._conn
        if conn is None:
            return
        try:
            with self._send_lock:
                conn.sendall(frame)
        except OSError:
            # 连接已断, _serve 会返回并清理
            pass

    @property
    def connected(self) -> bool:
        """是否有上位机连接。"""
        return self._connected

    @property
    def bound_port(self) -> Optional[int]:
        """实际监听端口（绑定后可用；未启动/绑定失败为 None）。"""
        return self._bound_port

    def wait_ready(self, timeout_s: float = 5.0) -> bool:
        """等待监听就绪（测试/启动确认用）。"""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._bound_port is not None:
                return True
            if not self.is_alive():
                return False  # 绑定失败已退出
            time.sleep(0.02)
        return self._bound_port is not None

    # ── 线程主体 ────────────────────────────────────
    def run(self):
        try:
            self._srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._srv_sock.bind((self._host, self._port))
            self._srv_sock.listen(1)
            self._bound_port = self._srv_sock.getsockname()[1]
            self._log(f"[HMI] 监听 {self._host}:{self._bound_port} (55 AA, 与 PLC 同协议)")
        except OSError as e:
            self._log(f"[HMI] 监听失败 ({self._host}:{self._port}): {e}")
            return

        while not self._stop_event.is_set():
            try:
                self._srv_sock.settimeout(0.5)  # 半秒轮询停止标志
                conn, addr = self._srv_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break  # stop() 关闭了监听 socket

            self._log(f"[HMI] 上位机已连接 {addr[0]}:{addr[1]}")
            with self._send_lock:
                self._conn = conn
            self._connected = True

            # 新连接先补发断线窗口内可能丢失的终态事件, 再上报当前状态
            # (send_frame 内部加锁, 此处不能持锁调用)
            terminal = self._last_terminal_event
            if terminal is not None:
                self._last_terminal_event = None
                self.send_frame(plc.CMD_BOX_STATUS, bytes([terminal]))
            self.send_frame(plc.CMD_BOX_STATUS, bytes([self._current_state]))

            self._serve(conn)

            self._connected = False
            with self._send_lock:
                self._conn = None
            try:
                conn.close()
            except OSError:
                pass
            self._log("[HMI] 上位机断开, 等待下一个连接")

    def _serve(self, conn: socket.socket):
        """收帧 + 周期心跳循环，直到客户端断开或停止。"""
        parser = plc.FrameParser()
        last_status_time = time.time()
        conn.settimeout(0.5)  # 半秒超时以便周期心跳/退出检查

        while not self._stop_event.is_set():
            # 周期心跳: 重发当前状态
            now = time.time()
            if now - last_status_time >= self._status_interval_s:
                self.send_frame(plc.CMD_BOX_STATUS, bytes([self._current_state]))
                last_status_time = now

            # 收数据
            try:
                chunk = conn.recv(4096)
            except socket.timeout:
                continue
            except OSError as e:
                self._log(f"[HMI] 接收错误: {e}")
                return

            if not chunk:
                return

            for cmd, data in parser.feed(chunk):
                # 注意: 命令码 0x01/0x02 与状态码 0x01/0x02 重号,
                # 收帧日志必须用命令码名称表 (不能用 STATE_NAMES)
                rx_name = {
                    plc.CMD_PLC_START: "START",
                    plc.CMD_PLC_ABORT: "ABORT",
                }.get(cmd, f"0x{cmd:02x}")
                self._log(
                    f"[HMI] ← {rx_name} cmd={cmd:#04x} data={data.hex() or '-'}"
                )
                if self._on_command is not None:
                    try:
                        self._on_command(cmd, data)
                    except Exception as e:
                        self._log(f"[HMI] 命令回调异常: {e}")

    def stop(self):
        """停止服务（关闭监听 socket 与当前连接促使其退出）。"""
        self._stop_event.set()
        with self._send_lock:
            conn = self._conn
            self._conn = None
            if conn is not None:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    conn.close()
                except OSError:
                    pass
            if self._srv_sock is not None:
                try:
                    self._srv_sock.close()
                except OSError:
                    pass
