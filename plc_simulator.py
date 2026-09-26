#!/usr/bin/env python3
"""PLC 模拟器 — 模拟 PLC 侧 TCP 服务器, 用于联调 ZED Box 的 55 AA 通信。

协议见 PLC_TCP_PROTOCOL.md。模拟 PLC 行为:
- 监听 TCP 端口, 接受 Box 客户端连接
- 解码并打印收到的每一帧 (状态/ACK/心跳, 带收帧计数 [#N])
- 键盘: 1=发START  2=发ABORT  a=自动模式  x=发原始字节  y=发坏CRC帧
        z=发干扰字节  c=收帧计数  h=帮助  q=退出

自动模式 (对齐协议 §6, 见 src/grape_stem_3d/plc_sim_auto.py):
- 已连接且见 Box IDLE → 自动发 START; DONE 后隔 retry-gap 再发
- 非 FAIL_CUT 失败码自动重试; FAIL_CUT 停机等人工 (按键 1 恢复)
- 发 START 后 --timeout 秒无终态 → 自动发 ABORT
- 连续 --watchdog 秒无帧 → 看门狗告警 (来帧自动恢复)

运行方式:
  cd /home/user/Grape_zed/DOBOT_6Axis_ROS2_V4-main/grape_stem_3d_zed_deploy
  python scripts/plc_simulator.py [--host 0.0.0.0] [--port 20001] [--auto]
"""

import argparse
import queue
import socket
import sys
import threading
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from grape_stem_3d import frame_desc
from grape_stem_3d import plc_comm as plc
from grape_stem_3d import plc_sim_auto as plc_auto

HELP_TEXT = """\
── 模拟 PLC 按键 ───────────────────────────────────────────────
  1          发 START (开始采摘)
  2          发 ABORT (中止)
  a          自动模式 开/关: 见 IDLE 自动发 START → 等终态;
             DONE 后隔 retry-gap 再发; 非 FAIL_CUT 失败码自动重试;
             FAIL_CUT 停机等人工 (按 1 恢复); 60s 无终态自动 ABORT;
             3s 无帧看门狗告警
  x HEX      发送原始字节流 (原样, 如 x 55aa0000017e80)
  y CMD [HEX]    发坏 CRC 帧 (合法帧破坏 CRC 末字节, 测 Box 容错)
  z HEX      发送干扰字节 (裸字节, 如 z 00ff55aa)
  c          显示收帧计数 (总量 + 分类)
  h          打印本帮助
  q          退出
"""


def parse_hex(token: str) -> bytes:
    """解析 hex 字符串 (容忍空格/0x 前缀/大小写), 奇数长度报错。"""
    token = token.replace(" ", "").replace("0x", "").replace("0X", "")
    if len(token) % 2:
        raise ValueError(f"奇数长度 hex: {token!r}")
    return bytes.fromhex(token)


class PlcSimulator:
    """TCP 服务器 + 键盘控制 + 自动模式。"""

    def __init__(self, host: str, port: int, *, timeout_s: float = 60.0,
                 abort_watchdog_s: float = 2.0, watchdog_s: float = 3.0,
                 retry_gap_s: float = 1.0, retry_fail: bool = True,
                 max_cycles: int = 0):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self._conn = None
        self._conn_lock = threading.Lock()
        self._stop = threading.Event()

        # 收帧计数 (recv 线程加锁累加)
        self._rx_count = 0
        self._rx_by_cmd = {}
        self._rx_lock = threading.Lock()

        # 自动模式: recv/连接事件入队, 单线程分发 (见 plc_sim_auto.py)
        self._events = queue.Queue()
        self._auto = plc_auto.AutoPlc(
            send_fn=self.send,
            result_timeout_s=timeout_s,
            abort_watchdog_s=abort_watchdog_s,
            watchdog_s=watchdog_s,
            retry_gap_s=retry_gap_s,
            retry_fail=retry_fail,
            max_cycles=max_cycles,
        )

    def run(self):
        print(f"[模拟PLC] 监听 {self._srv.getsockname()}, 等待 Box 连接...")
        print("  键盘: 1=发START  2=发ABORT  a=自动模式  x/y/z=发帧  h=帮助  q=退出\n")

        threading.Thread(target=self._accept_loop, daemon=True).start()
        threading.Thread(target=self._auto_loop, daemon=True).start()

        try:
            while not self._stop.is_set():
                try:
                    raw = input().strip()
                except EOFError:
                    self._stop.wait(1)
                    continue
                self._handle_key(raw)
        finally:
            self.close()

    # ── 键盘处理 ────────────────────────────────────────
    def _handle_key(self, raw: str) -> None:
        parts = raw.lower().split()
        if not parts:
            return
        key = parts[0]
        if key == "q":
            self._stop.set()
        elif key == "1":
            if self._auto.state != plc_auto.S_DISABLED:
                self._auto.on_manual_start()  # 自动机接管本轮 (含 FAIL_CUT 恢复)
                print("→ START (开始采摘) 已发送 [自动机接管本轮]")
            else:
                self.send(plc.CMD_PLC_START)
                print(f"→ START (开始采摘) 已发送 {plc.build_frame(plc.CMD_PLC_START).hex()}")
        elif key == "2":
            self.send(plc.CMD_PLC_ABORT)
            print(f"→ ABORT (中止) 已发送 {plc.build_frame(plc.CMD_PLC_ABORT).hex()}")
        elif key == "a":
            self._auto.set_enabled(self._auto.state == plc_auto.S_DISABLED)
        elif key == "x" or key == "z":
            if len(parts) < 2:
                print(f"  用法: {key} HEX, 如 {key} 55aa0000017e80")
                return
            try:
                raw = parse_hex(parts[1])
            except ValueError as e:
                print(f"  参数错误: {e}")
                return
            self.send_raw(raw)
            print(f"→ 原始字节已发送 ({len(raw)} 字节)")
        elif key == "y":
            if len(parts) < 2:
                print("  用法: y CMD [HEX], 如 y 01 / y 81 0a")
                return
            try:
                cmd = int(parts[1], 16)
                data = parse_hex(parts[2]) if len(parts) > 2 else b""
                if not 0 <= cmd <= 0xFF:
                    raise ValueError(f"CMD 越界: {cmd:#x}")
                bad = self.build_bad_crc_frame(cmd, data)
                self.send_raw(bad)
                print(f"→ 坏 CRC 帧已发送 (CRC 末字节取反): {bad.hex()}")
            except ValueError as e:
                print(f"  参数错误: {e}")
        elif key == "c":
            with self._rx_lock:
                print(f"  收帧计数: 共 {self._rx_count} 帧")
                for cmd, n in sorted(self._rx_by_cmd.items()):
                    name = frame_desc.CMD_NAMES.get(cmd, f"0x{cmd:02x}")
                    print(f"    {name}: {n}")
        elif key == "h":
            print(HELP_TEXT)
        else:
            print(f"  未知按键: {raw!r} (1=START 2=ABORT a=自动 h=帮助 q=退出)")

    # ── 自动模式事件循环 (单线程分发) ────────────────────
    def _auto_loop(self):
        """消费连接/帧事件; 队列空闲时以 0.5s 周期驱动 tick。"""
        while not self._stop.is_set():
            try:
                ev = self._events.get(timeout=0.5)
            except queue.Empty:
                self._auto.tick()
                continue
            kind, arg = ev
            if kind == "connected":
                self._auto.on_connected()
            elif kind == "disconnected":
                self._auto.on_disconnected()
            elif kind == "frame":
                cmd, data = arg
                self._auto.on_frame(cmd, data)

    # ── 网络 ────────────────────────────────────────────
    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                self._srv.settimeout(0.5)
                conn, addr = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            print(f"[模拟PLC] Box 已连接: {addr}")
            with self._conn_lock:
                self._conn = conn
            self._events.put(("connected", None))
            self._recv_loop(conn)
            print("[模拟PLC] Box 连接断开 (等待重连...)")
            with self._conn_lock:
                self._conn = None
            self._events.put(("disconnected", None))

    def _recv_loop(self, conn: socket.socket):
        parser = plc.FrameParser()
        conn.settimeout(0.5)
        while not self._stop.is_set():
            try:
                chunk = conn.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            if not chunk:
                return
            for cmd, data in parser.feed(chunk):
                with self._rx_lock:
                    self._rx_count += 1
                    self._rx_by_cmd[cmd] = self._rx_by_cmd.get(cmd, 0) + 1
                    n = self._rx_count
                raw = plc.build_frame(cmd, data).hex()  # CRC 已通过, 重建即原始字节
                print(f" [#{n}] {raw} " + frame_desc.describe_frame(cmd, data))
                self._events.put(("frame", (cmd, data)))

    # ── 发送 ────────────────────────────────────────────
    def send(self, cmd: int, data: bytes = b""):
        frame = plc.build_frame(cmd, data)
        self.send_raw(frame)

    def send_raw(self, raw: bytes):
        with self._conn_lock:
            conn = self._conn
        if conn is None:
            print("  [模拟PLC] Box 未连接, 无法发送")
            return
        try:
            conn.sendall(raw)
        except OSError as e:
            print(f"  [模拟PLC] 发送失败: {e}")

    @staticmethod
    def build_bad_crc_frame(cmd: int, data: bytes = b"") -> bytes:
        """构造一帧合法帧后取反 CRC 末字节 (模拟传输错误)。"""
        frame = plc.build_frame(cmd, data)
        return frame[:-1] + bytes([frame[-1] ^ 0xFF])

    def close(self):
        self._stop.set()
        with self._conn_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except OSError:
                    pass
        try:
            self._srv.close()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="PLC 模拟器 (55 AA TCP 服务器)")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0)")
    parser.add_argument("--port", type=int, default=20001, help="监听端口 (默认 20001)")
    parser.add_argument("--auto", action="store_true", help="启动即开自动模式")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="START 后无终态自动 ABORT 的超时秒 (现场值 60)")
    parser.add_argument("--abort-watchdog", type=float, default=2.0,
                        help="ABORT 后等 ABORTED 的超时秒 (协议 §6.6)")
    parser.add_argument("--watchdog", type=float, default=3.0,
                        help="无帧看门狗阈值秒 (协议 §6.3)")
    parser.add_argument("--retry-gap", type=float, default=1.0,
                        help="终态后到重发 START 的间隔秒")
    parser.add_argument("--no-retry-fail", action="store_true",
                        help="失败码不自动重试 (停机等人工)")
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="自动模式最大采摘轮数 (0=不限)")
    args = parser.parse_args()

    try:
        sim = PlcSimulator(
            args.host,
            args.port,
            timeout_s=args.timeout,
            abort_watchdog_s=args.abort_watchdog,
            watchdog_s=args.watchdog,
            retry_gap_s=args.retry_gap,
            retry_fail=not args.no_retry_fail,
            max_cycles=args.max_cycles,
        )
    except OSError as e:
        print(f"[模拟PLC] 监听 {args.host}:{args.port} 失败: {e}")
        print("  提示: 端口可能被占用 (ss -tlnp | grep 端口), 换 --port 重试")
        sys.exit(1)

    if args.auto:
        sim._auto.set_enabled(True)
    sim.run()


if __name__ == "__main__":
    main()
