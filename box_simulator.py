#!/usr/bin/env python3
"""ZED Box 模拟器 — 模拟 Box 侧 TCP 客户端 + 采摘流程状态机。

配合 scripts/plc_simulator.py (或真实 PLC) 联调 55 AA 协议。
协议见 PLC_TCP_PROTOCOL.md。行为:
- 以真实 PlcClient 身份连接 PLC (心跳/重连/终态补发与现场 Box 一致)
- 收到 START → ACK → PICKING → 六阶段推进 → DONE/FAIL → IDLE
- ABORT 任意时刻生效 (≤0.5s 上报 ABORTED)
- 键盘: 1=注入START  a=注入ABORT  f=预设故障  r=随机故障  t/x=手动发帧
- 帧收发全量日志 (← 收到 / → 发出)

运行方式:
  cd /home/user/Grape_zed/DOBOT_6Axis_ROS2_V4-main/grape_stem_3d_zed_deploy
  python scripts/box_simulator.py [--host 127.0.0.1] [--port 20001]
"""

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from grape_stem_3d import box_sim
from grape_stem_3d import plc_comm as plc

HELP_TEXT = """\
── ZED Box 模拟器按键 ──────────────────────────────────────────
  1          注入 START, 执行一轮自动采摘流程 (走 on_command, 与真实收到一致)
  a          注入 ABORT (任意时刻生效: 立即 ACK → ABORTED → IDLE)
  f STAGE:CODE  预设下轮故障, 如: f detect:03 (无目标) / f cut:0a (剪枝失败)
  r          随机故障注入 开/关 (随机选阶段与阶段内 FAIL 码)
  t HEX      手动上报任意状态码 (绕过状态机), 如: t 0a / t ff / t 00
  x CMD [HEX]   手动发任意帧 (自动 CRC), 如: x 82 0100 / x 81 09
  s          显示当前状态/阶段/故障配置
  h          打印本帮助
  q          退出
"""


def parse_hex(token: str) -> bytes:
    """解析 hex 字符串 (容忍空格/0x 前缀/大小写), 奇数长度报错。"""
    token = token.replace(" ", "").replace("0x", "").replace("0X", "")
    if len(token) % 2:
        raise ValueError(f"奇数长度 hex: {token!r}")
    return bytes.fromhex(token)


def parse_stage_durations(csv: str) -> dict:
    """解析 --stage-durations "2,4,4,3,4,2" 为阶段时长字典。"""
    parts = csv.split(",")
    if len(parts) != len(box_sim.STAGE_NAMES):
        raise argparse.ArgumentTypeError(
            f"需要 {len(box_sim.STAGE_NAMES)} 个数 (顺序 {','.join(box_sim.STAGE_NAMES)})"
        )
    return dict(zip(box_sim.STAGE_NAMES, map(float, parts)))


def show_state(sim: box_sim.BoxSim) -> None:
    """s 键: 显示当前状态与注入配置。"""
    busy = "忙" if sim.is_busy else "空闲"
    pending = sim.pending_fault
    if pending is not None:
        pending = f"{pending[0]} → {plc.state_name(pending[1])}"
    print(f"  状态: {busy}  当前阶段: {sim.current_stage or '-'}")
    print(f"  阶段时长: {sim.stage_durations}")
    print(
        f"  预设故障: {pending or '无'}  "
        f"随机注入: {'开' if sim.random_fault_enabled else '关'}"
    )


def main():
    parser = argparse.ArgumentParser(description="ZED Box 模拟器 (55 AA TCP 客户端)")
    parser.add_argument("--host", default="127.0.0.1", help="PLC/模拟器 IP (默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=20001, help="PLC 端口 (默认 20001)")
    parser.add_argument("--reconnect-interval", type=float, default=3.0, help="断线重连间隔秒")
    parser.add_argument("--status-interval", type=float, default=1.0, help="心跳间隔秒")
    parser.add_argument("--fault", metavar="STAGE:CODE", default=None,
                        help="启动即预设一次故障, 如 detect:03 / cut:0a")
    parser.add_argument("--fault-mode", choices=["off", "random"], default="off",
                        help="随机故障注入 (默认 off)")
    parser.add_argument("--fault-prob", type=float, default=0.2, help="随机注入概率")
    parser.add_argument("--stage-durations", type=parse_stage_durations, default=None,
                        metavar="CSV", help="阶段时长秒, 如 \"2,4,4,3,4,2\"")
    parser.add_argument("--idle-gap", type=float, default=0.2, help="终态→IDLE 间隔秒")
    args = parser.parse_args()

    durations = args.stage_durations or box_sim.DEFAULT_STAGE_DURATIONS
    total = sum(durations.values())
    mark = "✓ 在协议 10~30s 内" if 10.0 <= total <= 30.0 else "⚠ 超出协议 10~30s"
    print(f"[Box模拟] 阶段时长: {durations} (合计 {total:.1f}s, {mark})")

    client = plc.PlcClient(
        host=args.host,
        port=args.port,
        reconnect_interval_s=args.reconnect_interval,
        status_interval_s=args.status_interval,
    )
    sim = box_sim.BoxSim(
        client, stage_durations=durations, terminal_idle_gap_s=args.idle_gap
    )
    client.set_on_command(sim.on_command)

    if args.fault:
        try:
            stage, code_s = args.fault.split(":", 1)
            code = int(code_s, 16)
            sim.set_fault(stage, code)
            print(f"[Box模拟] 预设故障: 阶段 {stage} → {plc.state_name(code)}")
        except ValueError as e:
            parser.error(f"--fault 参数错误: {e}")
    if args.fault_mode == "random":
        sim.set_random_fault(True, args.fault_prob)
        print(f"[Box模拟] 随机故障注入: 开 (概率 {args.fault_prob})")

    client.start()
    print(HELP_TEXT)
    print(f"[Box模拟] 正在连接 PLC {args.host}:{args.port} ...")

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
            elif key == "1":
                print("→ 注入 START (开始采摘)...")
                sim.on_command(plc.CMD_PLC_START, b"")
            elif key == "a":
                print("→ 注入 ABORT (中止)...")
                sim.on_command(plc.CMD_PLC_ABORT, b"")
            elif key == "f":
                spec = parts[1] if len(parts) > 1 else None
                if not spec or ":" not in spec:
                    print(f"  用法: f 阶段:状态码  阶段: {','.join(box_sim.STAGE_NAMES)}")
                    print("  常用: detect:03 无目标 / detect:04 深度无效 / "
                          "detect:06 相机 / cut:0a 剪枝失败 / approach:08 运动异常")
                    continue
                try:
                    stage, code_s = spec.split(":", 1)
                    code = int(code_s, 16)
                    sim.set_fault(stage, code)
                    print(f"  已预设: 阶段 {stage} → {plc.state_name(code)}")
                except ValueError as e:
                    print(f"  参数错误: {e}")
            elif key == "r":
                enabled = not sim.random_fault_enabled
                sim.set_random_fault(enabled, args.fault_prob)
                print(f"  随机故障注入: {'开' if enabled else '关'} (概率 {args.fault_prob})")
            elif key == "t":
                if len(parts) < 2:
                    print("  用法: t HEX, 如 t 0a (FAIL_CUT) / t 00 (IDLE)")
                    continue
                try:
                    data = parse_hex(parts[1])
                    if len(data) != 1:
                        raise ValueError("状态码必须 1 字节")
                    sim.manual_status(data[0])
                except ValueError as e:
                    print(f"  参数错误: {e}")
            elif key == "x":
                if len(parts) < 2:
                    print("  用法: x CMD [HEX], 如 x 82 0100 / x 81 09")
                    continue
                try:
                    cmd = int(parts[1], 16)
                    data = parse_hex(parts[2]) if len(parts) > 2 else b""
                    if not 0 <= cmd <= 0xFF:
                        raise ValueError(f"CMD 越界: {cmd:#x}")
                    sim.manual_frame(cmd, data)
                except ValueError as e:
                    print(f"  参数错误: {e}")
            elif key == "s":
                show_state(sim)
            elif key == "h":
                print(HELP_TEXT)
            else:
                print(f"  未知按键: {raw!r} (h=帮助)")
    finally:
        sim.stop()
        client.stop()
        print("[Box模拟] 已退出")


if __name__ == "__main__":
    main()
