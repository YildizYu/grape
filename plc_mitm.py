#!/usr/bin/env python3
"""PLC ↔ Box 55 AA 链路 MITM 抓包 — 双向转发并解码记录。

插在真实 PLC 与真实 Box 之间: Box 的 TCP 客户端改连本工具监听端口,
本工具转发到真实 PLC。双向字节级原样透传 (坏帧/干扰字节也穿过),
用 FrameParser 双向解码记录 (时间戳 + 方向 + hex + 描述)。

现场部署 (跑在 Box 机 192.168.0.10 上):
  python scripts/plc_mitm.py --listen 192.168.0.10 --listen-port 20001 \\
                             --target 192.168.0.11 --target-port 20001
再把 Box 的 configs/fusion_pipeline.yaml 里 plc.host 改成 127.0.0.1
(用完务必改回)。

本地联调 (两个模拟器之间抓包):
  终端1: python scripts/plc_simulator.py --port 20001
  终端2: python scripts/plc_mitm.py --listen-port 20002 --target 127.0.0.1 --target-port 20001
  终端3: python scripts/box_simulator.py --port 20002
"""

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from grape_stem_3d import mitm as mitm_mod

HELP_TEXT = """\
── MITM 抓包按键 ──────────────────────────────────────────────
  c          显示统计 (会话/字节/帧)
  h          打印本帮助
  q          退出
"""


def main():
    parser = argparse.ArgumentParser(description="PLC ↔ Box 55 AA 链路 MITM 抓包")
    parser.add_argument("--listen", default="0.0.0.0", help="Box 改连的监听地址 (默认 0.0.0.0)")
    parser.add_argument("--listen-port", type=int, default=20002, help="监听端口 (默认 20002)")
    parser.add_argument("--target", default="192.168.0.11", help="真实 PLC IP")
    parser.add_argument("--target-port", type=int, default=20001, help="真实 PLC 端口")
    parser.add_argument("--log-file", default=None, help="追加写日志文件 (与 stdout 同格式)")
    args = parser.parse_args()

    log_file = open(args.log_file, "a", encoding="utf-8") if args.log_file else None

    def logger(msg: str) -> None:
        print(msg, flush=True)
        if log_file is not None:
            log_file.write(msg + "\n")
            log_file.flush()

    fwd = mitm_mod.MitmForwarder(
        args.listen, args.listen_port, args.target, args.target_port, logger=logger
    )
    try:
        fwd.start()
    except OSError as e:
        print(f"[MITM] 监听 {args.listen}:{args.listen_port} 失败: {e}")
        sys.exit(1)
    print(HELP_TEXT)

    try:
        while True:
            try:
                raw = input().strip()
            except EOFError:
                time.sleep(1)
                continue
            parts = raw.lower().split()
            if not parts:
                continue
            key = parts[0]
            if key == "q":
                break
            elif key == "c":
                s = fwd.stats
                print(f"  会话: {s['sessions']}")
                print(f"  字节: Box→PLC {s['bytes_box2plc']} / PLC→Box {s['bytes_plc2box']}")
                print(f"  帧:   Box→PLC {s['frames_box2plc']} / PLC→Box {s['frames_plc2box']}")
            elif key == "h":
                print(HELP_TEXT)
            else:
                print(f"  未知按键: {raw!r} (c=统计 h=帮助 q=退出)")
    finally:
        fwd.stop()
        if log_file is not None:
            log_file.close()
        print("[MITM] 已退出")


if __name__ == "__main__":
    main()
