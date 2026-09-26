"""55 AA 协议帧的可读描述 — PLC 模拟器 / Box 模拟器 / MITM 抓包共用。

从 scripts/plc_simulator.py 迁出 CMD_NAMES / describe_frame, 输出格式与
旧版逐字一致 (迁移回归由 tests/test_frame_desc.py 保证)。

注意: 0x01/0x02 命令码与状态码重号 (START 与 PICKING 同码), 收帧方向
必须用命令码名称表描述, 与 plc_comm.PlcClient 的收帧日志保持一致。
"""

from grape_stem_3d import plc_comm as plc

# ── 命令码名称表 ────────────────────────────────────────
# 覆盖双向: PLC→Box (0x01/0x02) 与 Box→PLC (0x81/0x82)
CMD_NAMES = {
    plc.CMD_PLC_START: "START (开始采摘)",
    plc.CMD_PLC_ABORT: "ABORT (中止)",
    plc.CMD_BOX_STATUS: "STATUS (状态上报)",
    plc.CMD_BOX_ACK: "ACK (命令应答)",
    plc.CMD_BOX_DRIVE: "DRIVE (驱动底盘到下一位置)",
}

# ── ACK 结果码名称表 ────────────────────────────────────
_ACK_RESULT_NAMES = {
    plc.ACK_OK: "成功",
    plc.ACK_BUSY: "忙",
    plc.ACK_UNKNOWN_CMD: "未知命令",
}

# ── 帧描述 ──────────────────────────────────────────────
def describe_frame(cmd: int, data: bytes) -> str:
    """生成帧的人类可读描述 (含方向箭头, PLC 视角)。

    返回如 "← STATUS (状态上报)  data=00 → 状态: IDLE"。
    ← 表示 PLC 收到 (来自 Box), 与旧版 plc_simulator.py 输出一致。
    """
    name = CMD_NAMES.get(cmd, f"UNKNOWN({cmd:#04x})")
    if cmd == plc.CMD_BOX_STATUS and data:
        state = data[0]
        desc = f"  data={data.hex()} → 状态: {plc.state_name(state)}"
    elif cmd == plc.CMD_BOX_ACK and len(data) == 2:
        ack_cmd = CMD_NAMES.get(data[0], f"{data[0]:#04x}")
        desc = (
            f"  data={data.hex()} → 应答 {ack_cmd}: "
            f"{_ACK_RESULT_NAMES.get(data[1], hex(data[1]))}"
        )
    else:
        desc = f"  data={data.hex() or '-'}"
    return f"← {name}{desc}"


def format_log_line(
    ts: str, direction: str, raw_hex: str, cmd: int, data: bytes
) -> str:
    """组装一条带时间戳/方向/原始 hex 的日志行 (MITM / Box 模拟器用)。

    Args:
        ts: 时间戳字符串 (如 "12:03:44.123")
        direction: 传输方向, "Box→PLC" 或 "PLC→Box"
        raw_hex: 完整帧的原始 hex
        cmd, data: 解析出的命令码与数据 (仅用于追加解码描述)

    返回如 "[12:03:44.123] Box→PLC 55aa0000017e80 ← START (开始采摘)"
    """
    arrow = "←" if direction == "Box→PLC" else "→"
    body = describe_frame(cmd, data)[2:]  # 去掉 describe_frame 自带的 ←
    return f"[{ts}] {direction} {raw_hex} {arrow} {body}"
