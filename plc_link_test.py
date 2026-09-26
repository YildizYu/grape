#!/usr/bin/env python3
"""PLC 链路测试 (Box 侧) — 原始字节 + 协议帧全量双向打印。

与旧版的区别: **每一个 recv 原始字节块都先打印 (hex + ascii)**,
再打印其中解析出的协议帧。PLC 下发的任何内容 — 乱码/半包/坏 CRC/
非协议数据/未知命令 — 都会显示, 不再只打印"协议内"的帧,
便于排查连不上、收不到、数据不对的问题。

自带 Box 侧演示应答 (与旧版一致):
- 连接即上报 IDLE, 周期心跳 0x81
- START → ACK(成功) → PICKING → DONE → IDLE
- ABORT → ACK(成功) → ABORTED → IDLE
- 每 drive_count 次 DONE 自动发一次 0x83 DRIVE
- 断线自动重连

运行方式 (现场, 连真实 PLC):
  python scripts/plc_link_test.py --host 192.168.0.11 --port 5001

  ⚠ 真实 PLC 通常只接一个 TCP 连接: 测试前先停 Box 侧程序
  (run_grasp_pipeline.py), 否则测试端连不上或会把 Box 挤下线。

  ⚠ Box 多网卡时默认路由可能走错网口 → 连接超时。现场网段
  (box=192.168.0.10) 可用 --bind 192.168.0.10 强制走现场网口。

本地联调 (终端1 先开模拟器):
  python scripts/plc_simulator.py --port 20001
  python scripts/plc_link_test.py --host 127.0.0.1 --port 20001

输出示例:
  [12:00:01] [PLC] 连接 192.168.0.11:5001 ...
  [12:00:01] [PLC] 已连接 192.168.0.11:5001, 本地 192.168.0.10:40012
  [12:00:01] [TX] 55 AA 00 01 81 00 E0 91
  [12:00:02] [RX RAW] len=7  55 AA 00 01 01 2E F5  |UU....|
  [12:00:02] [RX OK] START cmd=0x01 data=-          (PLC 发来 START)
"""

import argparse
import socket
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from grape_stem_3d import plc_comm as plc

# 收帧命令码名称 (0x01/0x02 与状态码重号, 必须用命令码名称表)
_RX_CMD_NAMES = {
    plc.CMD_PLC_START: "START",
    plc.CMD_PLC_ABORT: "ABORT",
    plc.CMD_BOX_STATUS: "STATUS",
    plc.CMD_BOX_ACK: "ACK",
    plc.CMD_BOX_DRIVE: "DRIVE",
}
_ACK_RESULTS = {
    plc.ACK_OK: "OK",
    plc.ACK_BUSY: "BUSY",
    plc.ACK_UNKNOWN_CMD: "UNKNOWN_CMD",
}


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _raw_line(chunk: bytes) -> str:
    """原始字节块: hex + 可打印 ascii (不可打印显示为 .)。"""
    hexs = chunk.hex(" ").upper()
    ascii_ = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
    return f"[{_now()}] [RX RAW] len={len(chunk)}  {hexs}  |{ascii_}|"


def _rx_frame_line(cmd: int, data: bytes) -> str:
    name = _RX_CMD_NAMES.get(cmd, f"0x{cmd:02x}")
    line = f"[{_now()}] [RX OK] {name} cmd={cmd:#04x} data={data.hex() or '-'}"
    if cmd == plc.CMD_BOX_STATUS and len(data) == 1:
        line += f" → 状态: {plc.state_name(data[0])}"
    elif cmd == plc.CMD_BOX_ACK and len(data) == 2:
        line += (
            f" → 应答 {_RX_CMD_NAMES.get(data[0], hex(data[0]))} "
            f"结果 {_ACK_RESULTS.get(data[1], hex(data[1]))}"
        )
    elif cmd == plc.CMD_BOX_DRIVE and len(data) == 1:
        line += f" → 换位: {data[0]}"
    return line


def _tx(sock: socket.socket, frame: bytes) -> None:
    sock.sendall(frame)
    print(f"[{_now()}] [TX] {frame.hex(' ').upper()}", flush=True)


def _handle_command(sock, cmd, data, args, state) -> None:
    """模拟 Box 侧命令处理: START → 演示一次采摘状态循环。"""
    if cmd == plc.CMD_PLC_START:
        _tx(sock, plc.build_frame(plc.CMD_BOX_ACK, bytes([cmd, plc.ACK_OK])))
        state["cur"] = plc.STATE_PICKING
        _tx(sock, plc.build_frame(plc.CMD_BOX_STATUS, bytes([plc.STATE_PICKING])))
        time.sleep(2.0)  # 演示: 假装执行采摘
        state["cur"] = plc.STATE_DONE
        _tx(sock, plc.build_frame(plc.CMD_BOX_STATUS, bytes([plc.STATE_DONE])))
        state["cur"] = plc.STATE_IDLE
        _tx(sock, plc.build_frame(plc.CMD_BOX_STATUS, bytes([plc.STATE_IDLE])))
        # 累计 N 次 DONE → 发 DRIVE (0x83) 验证换位信号链路
        if args.drive_count > 0:
            state["done_cnt"] += 1
            if state["done_cnt"] >= args.drive_count:
                state["done_cnt"] = 0
                print(f"[{_now()}] [Box] 已累计 {args.drive_count} 次 DONE → 发送 DRIVE (0x83)")
                _tx(sock, plc.build_frame(plc.CMD_BOX_DRIVE, bytes([0])))
    elif cmd == plc.CMD_PLC_ABORT:
        _tx(sock, plc.build_frame(plc.CMD_BOX_ACK, bytes([cmd, plc.ACK_OK])))
        state["cur"] = plc.STATE_ABORTED
        _tx(sock, plc.build_frame(plc.CMD_BOX_STATUS, bytes([plc.STATE_ABORTED])))
        state["cur"] = plc.STATE_IDLE
        _tx(sock, plc.build_frame(plc.CMD_BOX_STATUS, bytes([plc.STATE_IDLE])))
    else:
        print(f"[{_now()}] [Box] 未知命令 {cmd:#04x}, 应答 UNKNOWN_CMD")
        _tx(sock, plc.build_frame(plc.CMD_BOX_ACK, bytes([cmd, plc.ACK_UNKNOWN_CMD])))


def _recv_loop(sock, args, state) -> None:
    """收帧 + 周期心跳循环, 直到断线或停止。

    每个 recv 块先打印 RX RAW (hex+ascii), 再打印解析出的协议帧,
    保证 PLC 下发的任何字节都可见。
    """
    parser = plc.FrameParser()
    last_status = time.time()
    status_interval_s = 1.0

    while True:
        # 周期心跳: 重发当前状态
        if time.time() - last_status >= status_interval_s:
            _tx(sock, plc.build_frame(plc.CMD_BOX_STATUS, bytes([state["cur"]])))
            last_status = time.time()

        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            continue
        except OSError as e:
            print(f"[{_now()}] [PLC] 接收错误: {e!r}")
            return

        if not chunk:
            print(f"[{_now()}] [PLC] PLC 关闭了连接")
            return

        print(_raw_line(chunk), flush=True)
        for cmd, data in parser.feed(chunk):
            print(_rx_frame_line(cmd, data), flush=True)
            _handle_command(sock, cmd, data, args, state)


def run(args) -> None:
    deadline = time.time() + args.seconds if args.seconds > 0 else None
    state = {"done_cnt": 0, "cur": plc.STATE_IDLE}

    while deadline is None or time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if args.bind:
                sock.bind((args.bind, 0))
            print(f"[{_now()}] [PLC] 连接 {args.host}:{args.port} ...", flush=True)
            sock.settimeout(3.0)
            sock.connect((args.host, args.port))
            sock.settimeout(0.5)
            local = sock.getsockname()
            print(f"[{_now()}] [PLC] 已连接 {args.host}:{args.port}, "
                  f"本地 {local[0]}:{local[1]}")
            # 连接即上报当前状态 (与 PlcClient 一致)
            _tx(sock, plc.build_frame(plc.CMD_BOX_STATUS, bytes([state["cur"]])))
            _recv_loop(sock, args, state)
        except OSError as e:
            print(f"[{_now()}] [PLC] 连接失败: {e!r}, 3s 后重连")
        finally:
            sock.close()
        time.sleep(3.0)


def main():
    parser = argparse.ArgumentParser(
        description="PLC 链路测试 (Box 侧, 原始字节 + 协议帧全量打印)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="192.168.0.11", help="PLC IP (默认 192.168.0.11)")
    parser.add_argument("--port", type=int, default=5001, help="PLC 端口 (默认 5001)")
    parser.add_argument("--bind", default="", help="绑定本地源 IP (如 192.168.0.10, 多网卡指定现场网口)")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="运行时长秒 (默认 0 = 一直运行, Ctrl+C 退出)")
    parser.add_argument("--drive-count", type=int, default=5,
                        help="每 N 次 DONE 发一次 0x83 DRIVE (默认 5, 0 = 不发)")
    args = parser.parse_args()

    print("=" * 60)
    print("PLC 链路测试 (Box 侧) 已启动 — 原始字节全量打印")
    print(f"  目标: {args.host}:{args.port}" + (f" (源地址绑定 {args.bind})" if args.bind else ""))
    print(f"  每 {args.drive_count} 次 DONE 发一次 DRIVE (0x83)" if args.drive_count > 0
          else "  不发 DRIVE")
    print("  请让 PLC 侧下发 0x01 START / 0x02 ABORT 观察双向收发")
    print("  Ctrl+C 退出")
    print("=" * 60)

    try:
        run(args)
    except KeyboardInterrupt:
        pass
    finally:
        print("[Box] 已退出")


if __name__ == "__main__":
    main()
