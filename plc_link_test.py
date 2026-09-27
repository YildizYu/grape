#!/usr/bin/env python3
"""PLC 链路测试 (Box 侧) — 连接真实 PLC 或模拟器, 验证 55 AA 链路并双向打印。

不依赖相机/机械臂/ROS2, 验证:
- TCP 连接 + 心跳 (0x81 周期状态, 兼作 PLC 看门狗信号)
- 收到 START → ACK(成功) → PICKING → DONE → IDLE 演示循环
- 收到 ABORT → ACK(成功) → ABORTED → IDLE
- 每 drive_count 次 DONE 自动发一次 0x83 DRIVE (验证换位信号链路)
- 断线自动重连
- **双向打印所有收发帧** (方向 + 命令名 + 状态解码, 收帧含原始 hex)

运行方式 (现场, 连真实 PLC):
  python scripts/plc_link_test.py --host 192.168.0.11 --port 5000

  ⚠ 真实 PLC 通常只接一个 TCP 连接: 测试前先停 Box 侧程序
  (run_grasp_pipeline.py), 否则测试端连不上或会把 Box 挤下线。

本地联调 (终端1 先开模拟器):
  python scripts/plc_simulator.py --port 20001
  python scripts/plc_link_test.py --host 127.0.0.1 --port 20001

输出示例:
  [PLC] 连接 127.0.0.1:20001 ...
  [PLC] 已连接 127.0.0.1:20001
  [PLC] → STATUS cmd=0x81 data=00 → 状态: IDLE      (连接即上报)
  [PLC] ← START cmd=0x01 data=-                     (PLC 发来 START)
  [PLC] → ACK cmd=0x82 data=0100                   (应答 START 成功)
  [PLC] → STATUS cmd=0x81 data=01 → 状态: PICKING
  [PLC] → STATUS cmd=0x81 data=02 → 状态: DONE
  [PLC] → STATUS cmd=0x81 data=00 → 状态: IDLE
  ... (第 5 次 DONE 后)
  [PLC] → DRIVE cmd=0x83 data=-                     (0x83 换位信号)
"""

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from grape_stem_3d import plc_comm as plc


def main():
    parser = argparse.ArgumentParser(description="PLC 链路测试 (Box 侧, 双向打印)")
    parser.add_argument("--host", default="192.168.0.11", help="PLC IP (默认 192.168.0.11)")
    parser.add_argument("--port", type=int, default=5000, help="PLC 端口 (默认 5000)")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="运行时长秒 (默认 0 = 一直运行, Ctrl+C 退出)")
    parser.add_argument("--drive-count", type=int, default=5,
                        help="每 N 次 DONE 发一次 0x83 DRIVE (默认 5, 0 = 不发)")
    args = parser.parse_args()

    state = {"done_cnt": 0}

    client = plc.PlcClient(
        host=args.host,
        port=args.port,
        on_command=None,  # 回调在下方设置
        reconnect_interval_s=3.0,
        status_interval_s=1.0,
        log_sent=True,  # 打印发出的每一帧
    )

    def on_command(cmd, data):
        """模拟 Box 侧命令处理: START → 演示一次采摘状态循环。"""
        if cmd == plc.CMD_PLC_START:
            client.send_ack(cmd, plc.ACK_OK)
            client.send_status(plc.STATE_PICKING)
            time.sleep(2.0)  # 演示: 假装执行采摘
            client.send_status(plc.STATE_DONE)
            client.send_status(plc.STATE_IDLE)
            # 累计 N 次 DONE → 发 DRIVE (0x83) 验证换位信号链路
            if args.drive_count > 0:
                state["done_cnt"] += 1
                if state["done_cnt"] >= args.drive_count:
                    state["done_cnt"] = 0
                    print(f"[Box] 已累计 {args.drive_count} 次 DONE → 发送 DRIVE (0x83)")
                    client.send_drive()
        elif cmd == plc.CMD_PLC_ABORT:
            client.send_ack(cmd, plc.ACK_OK)
            client.send_status(plc.STATE_ABORTED)
            client.send_status(plc.STATE_IDLE)
        else:
            client.send_ack(cmd, plc.ACK_UNKNOWN_CMD)

    client.set_on_command(on_command)
    client.start()

    print("=" * 60)
    print(f"PLC 链路测试 (Box 侧) 已启动")
    print(f"  目标: {args.host}:{args.port}")
    print(f"  每 {args.drive_count} 次 DONE 发一次 DRIVE (0x83)" if args.drive_count > 0
          else "  不发 DRIVE")
    print("  请让 PLC 侧下发 0x01 START / 0x02 ABORT 观察双向收发")
    print("  Ctrl+C 退出")
    print("=" * 60)

    try:
        if args.seconds > 0:
            time.sleep(args.seconds)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        client.join(timeout=3.0)
        print("[Box] 已退出")


if __name__ == "__main__":
    main()
