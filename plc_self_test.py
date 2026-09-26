#!/usr/bin/env python3
"""PLC↔Box 55 AA 协议全链路自测 — 无机械臂 / 无 PLC 硬件 / 无相机 / 无 ROS2。

在 127.0.0.1 回环上用「内置假 PLC + 真实 PlcClient + DriveGateStatusSink」
(接线方式与 run_grasp_pipeline.py 完全一致) 跑完整协议交互, 双向打印
所有收发帧, 覆盖:

  1. TCP 连接 + 首帧 IDLE + 周期心跳
  2. START → ACK → PICKING → DONE → IDLE 采摘循环
  3. 累计 5 次 DONE → 发 0x83 DRIVE (换位信号)
  4. 发出 DRIVE 后未收到 START → 每 2 秒补发一次 DRIVE
  5. START 解除 gate 后连续 10 次 FAIL_NO_TARGET → 再发 0x83 DRIVE
  6. ABORT → ACK → ABORTED → IDLE
  7. 忙时 START → ACK(忙)

最后自动判定 通过/失败 (退出码 0/1)。仅依赖标准库 + 项目 src/。

运行:
  python3 scripts/plc_self_test.py
"""

import socket
import sys
import threading
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from grape_stem_3d import frame_desc
from grape_stem_3d import plc_comm as plc
from grape_stem_3d.drive_gate import DriveGateStatusSink


# ── 内置假 PLC (TCP 服务器) ─────────────────────────────
class FakePlc:
    """收帧解码打印 + 记录, 可发 START/ABORT。"""

    def __init__(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))  # 随机可用端口
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self._conn = None
        self._parser = plc.FrameParser()
        self.received = []  # [(cmd, data), ...] 按到达顺序
        self._lock = threading.Lock()
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)

    def start(self):
        self._recv_thread.start()

    def _recv_loop(self):
        try:
            self._conn, addr = self._srv.accept()
            print(f"[假PLC] Box 已连接: {addr}")
            self._conn.settimeout(0.5)
            while True:
                try:
                    chunk = self._conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                with self._lock:
                    for cmd, data in self._parser.feed(chunk):
                        self.received.append((cmd, data))
                        print(f"[假PLC] 收到 {frame_desc.describe_frame(cmd, data)}")
        except OSError:
            pass

    def send_frame(self, cmd, data=b""):
        name = {plc.CMD_PLC_START: "START", plc.CMD_PLC_ABORT: "ABORT"}.get(cmd, f"0x{cmd:02x}")
        self._conn.sendall(plc.build_frame(cmd, data))
        print(f"[假PLC] 发出 {name}")

    def frames(self):
        with self._lock:
            return list(self.received)

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
        try:
            self._srv.close()
        except OSError:
            pass


# ── 自测主流程 ──────────────────────────────────────────
CHECKS = []  # [(描述, 通过?, 失败详情)]


def check(desc, ok, detail=""):
    CHECKS.append((desc, bool(ok), detail))
    mark = "✔" if ok else "✘"
    extra = f" — {detail}" if (detail and not ok) else ""
    print(f"  {mark} {desc}{extra}")


def main():
    print("=" * 60)
    print("PLC↔Box 55 AA 协议全链路自测 (离线, 无机械臂/PLC/相机)")
    print("=" * 60)

    fake = FakePlc()
    fake.start()

    # 真 PlcClient + DriveGate 计数门 (与主程序相同接线)
    client = plc.PlcClient(
        host="127.0.0.1",
        port=fake.port,
        reconnect_interval_s=0.5,
        status_interval_s=0.5,
        log_sent=True,  # 打印发出的每一帧
    )
    drive_gate = DriveGateStatusSink(
        client, client,
        {"enabled": True, "done_threshold": 5, "no_target_threshold": 10},
    )

    flags = {"start": False, "abort": False}
    busy = [False]
    pick_mode = {"fail": False}  # True = 本轮采摘报 FAIL_NO_TARGET
    pick_done = threading.Event()

    def on_command(cmd, data):
        """与 run_grasp_pipeline.py 的 _make_command_handler 一致。"""
        if cmd == plc.CMD_PLC_START:
            if busy[0] or flags["start"]:
                client.send_ack(cmd, plc.ACK_BUSY)
            else:
                client.send_ack(cmd, plc.ACK_OK)
                drive_gate.on_plc_start()  # START 被接受 → 解除 Drive gate
                flags["start"] = True
        elif cmd == plc.CMD_PLC_ABORT:
            client.send_ack(cmd, plc.ACK_OK)
            flags["abort"] = True
        else:
            client.send_ack(cmd, plc.ACK_UNKNOWN_CMD)

    client.set_on_command(on_command)
    client.start()

    def simulate_pick():
        """模拟一次采摘流程状态上报 (经 drive_gate, 与真实 pick_flow 一致)。"""
        busy[0] = True
        drive_gate.send_status(plc.STATE_PICKING)
        time.sleep(0.3)
        if pick_mode["fail"]:
            drive_gate.send_status(plc.STATE_FAIL_NO_TARGET)
        else:
            drive_gate.send_status(plc.STATE_DONE)
        drive_gate.send_status(plc.STATE_IDLE)
        busy[0] = False
        pick_done.set()

    def consume_flags(timeout_s=5.0):
        """主循环: 消费 START/ABORT 标志 (与 run_grasp_pipeline.py 主循环一致)。"""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if flags["start"]:
                flags["start"] = False
                pick_done.clear()
                simulate_pick()
            if flags["abort"]:
                flags["abort"] = False
                drive_gate.send_status(plc.STATE_ABORTED)
                drive_gate.send_status(plc.STATE_IDLE)
                pick_done.set()
            if pick_done.is_set():
                return True
            time.sleep(0.02)
        return False

    def start_and_wait():
        pick_done.clear()
        fake.send_frame(plc.CMD_PLC_START)
        return consume_flags()

    try:
        # ── 场景 1: 连接 + 首帧 IDLE ──
        deadline = time.time() + 5.0
        while time.time() < deadline and not fake.frames():
            time.sleep(0.05)
        first = fake.frames()
        check(
            "TCP 连接 + 首帧 IDLE",
            bool(first) and first[0] == (plc.CMD_BOX_STATUS, b"\x00"),
            f"首帧={first[0] if first else '无'}",
        )

        # ── 场景 2: 5 次采摘 → 第 5 次 DONE 后发 DRIVE ──
        print("\n── 场景 2: 5 次 START 采摘循环 (预期第 5 次 DONE 后发 0x83 DRIVE)")
        ok = True
        for i in range(1, 6):
            if not start_and_wait():
                ok = False
                break
        check("5 次采摘流程全部完成", ok)

        # ── 场景 3: Drive 重发 (未收到 START 时每 2s 补发) ──
        print("\n── 场景 3: 未收到 START 时 Drive 每 2 秒补发")
        time.sleep(2.5)
        n_retry = sum(1 for f in fake.frames() if f == (plc.CMD_BOX_DRIVE, b"\x00"))
        check("Drive 自动补发 (≥2 帧)", n_retry >= 2, f"实际 {n_retry}")

        # ── 场景 4: 连续 10 次无果 → 第 10 次后发 DRIVE ──
        print("\n── 场景 4: 连续 10 次检测不到 (预期第 10 次 FAIL_NO_TARGET 后发 0x83 DRIVE)")
        pick_mode["fail"] = True
        ok = True
        for _ in range(10):
            if not start_and_wait():
                ok = False
                break
        pick_mode["fail"] = False
        check("10 次无果流程全部完成", ok)

        # ── 场景 5: ABORT ──
        print("\n── 场景 5: ABORT")
        pick_done.clear()
        fake.send_frame(plc.CMD_PLC_ABORT)
        check("ABORT 流程完成", consume_flags())

        # ── 场景 6: 忙时 START → ACK(忙) ──
        print("\n── 场景 6: 采摘进行中再发 START (预期 ACK 忙)")
        pick_done.clear()
        fake.send_frame(plc.CMD_PLC_START)  # 第 1 个 START: 被接受
        time.sleep(0.15)                    # 此时流程执行中/待消费
        fake.send_frame(plc.CMD_PLC_START)  # 第 2 个 START: 应回 ACK(忙)
        consume_flags()

        # 等心跳几帧后汇总判定
        time.sleep(1.2)

        frames = fake.frames()
        done_idx = [i for i, f in enumerate(frames) if f == (plc.CMD_BOX_STATUS, b"\x02")]
        notarget_idx = [i for i, f in enumerate(frames) if f == (plc.CMD_BOX_STATUS, b"\x03")]
        drive_idx = [i for i, f in enumerate(frames) if f == (plc.CMD_BOX_DRIVE, b"\x00")]
        idle_cnt = frames.count((plc.CMD_BOX_STATUS, b"\x00"))
        ack_ok = frames.count((plc.CMD_BOX_ACK, b"\x01\x00"))
        ack_busy = frames.count((plc.CMD_BOX_ACK, b"\x01\x01"))
        ack_abort = frames.count((plc.CMD_BOX_ACK, b"\x02\x00"))
        aborted_cnt = frames.count((plc.CMD_BOX_STATUS, b"\x09"))

        print("\n── 汇总判定 ──")
        check("DONE 状态帧数 ≥ 6 (场景2×5 + 场景5×1)", len(done_idx) >= 6,
              f"实际 {len(done_idx)}")
        check("FAIL_NO_TARGET 状态帧数 = 10", len(notarget_idx) == 10,
              f"实际 {len(notarget_idx)}")
        check("DRIVE (0x83) 至少 3 次 (触发 2 次 + 补发 ≥1 次)",
              len(drive_idx) >= 3, f"实际 {len(drive_idx)}")
        if len(done_idx) >= 5 and len(drive_idx) >= 1:
            check("第 1 次 DRIVE 在第 5 次 DONE 之后",
                  drive_idx[0] > done_idx[4])
        if len(notarget_idx) >= 10 and len(drive_idx) >= 1:
            check("最后一次 DRIVE 在第 10 次 FAIL_NO_TARGET 之后",
                  drive_idx[-1] > notarget_idx[9])
        check("START 应答 ACK(成功) = 16", ack_ok == 16, f"实际 {ack_ok}")
        check("忙时 START 应答 ACK(忙) ≥ 1", ack_busy >= 1, f"实际 {ack_busy}")
        check("ABORT 应答 ACK(成功) = 1", ack_abort == 1, f"实际 {ack_abort}")
        check("ABORTED 状态帧 = 1", aborted_cnt == 1, f"实际 {aborted_cnt}")
        check("IDLE 心跳帧 ≥ 3", idle_cnt >= 3, f"实际 {idle_cnt}")

        failed = [c for c in CHECKS if not c[1]]
        print("=" * 60)
        if failed:
            print(f"✘ 自测未通过: {len(failed)} 项失败")
            for desc, _, detail in failed:
                print(f"   - {desc} {detail}")
            return 1
        print("✔ 全部通过: 协议交互与 0x83 Drive 均正常")
        print("=" * 60)
        return 0
    finally:
        client.stop()
        client.join(timeout=3.0)
        fake.close()


if __name__ == "__main__":
    sys.exit(main())
