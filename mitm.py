"""55 AA 链路 MITM 转发 — 插在真实 PLC 与真实 Box 之间, 双向转发并解码记录。

转发为字节级原样透传 (坏 CRC 帧/干扰字节/半包全部穿过, 这正是 MITM 的
意义); FrameParser 双向各一个实例仅用于日志解码, 解码失败绝不影响转发
(坏帧由 FrameParser 按协议 §6.1 规则丢弃重同步, MITM 不做干预)。

线程模型: accept 线程接受 Box 连接 → create_connection 连真实 PLC →
两个 pump 线程 (daemon) 双向拷贝; 任一 pump 结束 (EOF/错误/停止) →
关闭两侧 socket 解除另一 pump 阻塞 → 回 accept 等待 Box 重连
(Box 的 PlcClient 断线后 3s 自动重连)。
"""

import socket
import threading
import time
from typing import Callable, Dict, Optional, Tuple

from grape_stem_3d import frame_desc
from grape_stem_3d import plc_comm as plc

# ── 方向常量 ────────────────────────────────────────────
DIR_BOX2PLC = "Box→PLC"
DIR_PLC2BOX = "PLC→Box"


def _timestamp() -> str:
    """毫秒级时间戳, 如 12:03:44.123。"""
    t = time.time()
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"


class MitmForwarder:
    """MITM 双向转发核心。

    Args:
        listen_host / listen_port: Box 改连的监听地址与端口 (listen_port=0
            时随机分配, 经 listen_port_actual 查询)
        target_host / target_port: 真实 PLC 的地址与端口
        logger: 日志函数 (默认 print)
    """

    def __init__(
        self,
        listen_host: str,
        listen_port: int,
        target_host: str,
        target_port: int,
        logger: Callable[[str], None] = print,
    ):
        self._listen_host = listen_host
        self._listen_port = listen_port
        self._target_host = target_host
        self._target_port = target_port
        self._log = logger

        self._stop = threading.Event()
        self._srv: Optional[socket.socket] = None
        self._socks_lock = threading.Lock()
        self._active_socks: set = set()  # 当前会话的 socket, stop() 时全部关闭

        self._stats_lock = threading.Lock()
        self._stats: Dict[str, int] = {
            "sessions": 0,
            "bytes_box2plc": 0,
            "bytes_plc2box": 0,
            "frames_box2plc": 0,
            "frames_plc2box": 0,
        }
        self._accept_thread: Optional[threading.Thread] = None

    # ── 对外接口 ────────────────────────────────────────
    def start(self) -> None:
        """绑定监听端口并启动 accept 线程 (daemon)。"""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self._listen_host, self._listen_port))
        srv.listen(5)
        self._srv = srv
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        self._log(f"[MITM] 监听 {self.listen_addr}, 转发到 {self._target_host}:{self._target_port}")

    @property
    def listen_addr(self) -> Tuple[str, int]:
        """实际监听地址 (listen_port=0 时为分配到的随机端口)。"""
        if self._srv is None:
            return (self._listen_host, self._listen_port)
        return self._srv.getsockname()

    @property
    def stats(self) -> Dict[str, int]:
        """统计快照: sessions/双向 bytes/双向 frames。"""
        with self._stats_lock:
            return dict(self._stats)

    def stop(self) -> None:
        """停止: 置停止事件并关闭监听与活动 socket 解除阻塞。"""
        self._stop.set()
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass
        with self._socks_lock:
            socks = list(self._active_socks)
        for s in socks:
            try:
                s.close()
            except OSError:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)

    # ── accept 与会话 ───────────────────────────────────
    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._srv.settimeout(0.5)
                box_conn, box_addr = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self._handle_session(box_conn, box_addr)

    def _handle_session(self, box_conn: socket.socket, box_addr) -> None:
        """一个 Box 连接的一次会话: 连真实 PLC → 双向 pump → 关闭。"""
        try:
            plc_conn = socket.create_connection(
                (self._target_host, self._target_port), timeout=3.0
            )
        except OSError as e:
            self._log(f"[MITM] 连接真实 PLC 失败: {e} (关闭 Box 连接, 等其重连)")
            self._close_sock(box_conn)
            return
        with self._socks_lock:
            self._active_socks.add(box_conn)
            self._active_socks.add(plc_conn)
        with self._stats_lock:
            self._stats["sessions"] += 1
            n = self._stats["sessions"]
        self._log(f"[MITM] 会话 #{n}: Box {box_addr} ↔ PLC {self._target_host}:{self._target_port}")

        done = threading.Event()

        def wrapper(src, dst, direction):
            try:
                self._pump(src, dst, direction)
            finally:
                done.set()

        t1 = threading.Thread(
            target=wrapper, args=(box_conn, plc_conn, DIR_BOX2PLC), daemon=True
        )
        t2 = threading.Thread(
            target=wrapper, args=(plc_conn, box_conn, DIR_PLC2BOX), daemon=True
        )
        t1.start()
        t2.start()
        done.wait()  # 任一 pump 结束即收尾

        self._close_sock(box_conn)
        self._close_sock(plc_conn)
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)
        self._log(f"[MITM] 会话 #{n} 结束")

    # ── 双向 pump ───────────────────────────────────────
    def _pump(self, src: socket.socket, dst: socket.socket, direction: str) -> None:
        """单向转发: src→dst 原样透传, 顺带解码记录。

        每个方向独立 FrameParser 实例 (会话重建时随 pump 新建, 绝不可
        共享 — 方向混流会把对端字节喂进错解析器)。
        """
        parser = plc.FrameParser()
        bytes_key = "bytes_box2plc" if direction == DIR_BOX2PLC else "bytes_plc2box"
        frames_key = "frames_box2plc" if direction == DIR_BOX2PLC else "frames_plc2box"
        try:
            src.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    chunk = src.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                with self._stats_lock:
                    self._stats[bytes_key] += len(chunk)
                # 解码仅用于日志; 坏帧由 FrameParser 丢弃, 不影响转发
                for cmd, data in parser.feed(chunk):
                    with self._stats_lock:
                        self._stats[frames_key] += 1
                    # 合法帧的接收字节与 build_frame 输出一致 (CRC 已验)
                    raw_hex = plc.build_frame(cmd, data).hex()
                    self._log(
                        frame_desc.format_log_line(
                            _timestamp(), direction, raw_hex, cmd, data
                        )
                    )
                dst.sendall(chunk)  # 原样转发 (含坏帧/干扰字节/半包)
        except OSError:
            pass

    def _close_sock(self, sock: socket.socket) -> None:
        with self._socks_lock:
            self._active_socks.discard(sock)
        try:
            sock.close()
        except OSError:
            pass
