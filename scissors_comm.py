"""剪刀手 RS485 直连控制模块（ZED Box USB转485 → 剪刀手控制器）。

协议见 剪刀手控制协议0820.doc，MODBUS-RTU 要点：
- 9600/N/8/1，默认设备地址 0x01
- 功能码仅 03（读保持寄存器）/ 06（写单保持寄存器），从站不支持 FC16
- 寄存器:
    0x01 电机运行  写 4 = 电机轴伸出合剪（剪断果梗）
    0x03 停止      写 1
    0x04 速度      写/读 1-800 圈/min（轴收缩速度 = 电机转速/12）
    0x05 脉冲数    写/读 0-65535，一圈 1600 脉冲，9000 = 25mm 全行程
    0x0A 回原点    写 1 = 电机轴缩回开剪（放果）
    0x02 运动状态  读: 1=运动中, 0=已停止（寄存器表未列出，见协议注 1）
    0x08 设备地址  读: 01-247，可用作通信 ping
- 已知怪癖:
    1. 上电时控制器会上发干扰帧 00 00 01 06 00 03 00 01 B8 0A（前两字节
       00 00 为干扰码），收发必须按 CRC 校验过滤，不能按字节序盲拼。
    2. 动作指令（合剪/开剪）后控制器回两帧: 第 1 帧协议响应帧（原样回显），
       第 2 帧电机停止帧 01 06 00 03 00 01 B8 0A。停止帧可能残留在总线上
       污染下一次读取，故读帧时丢弃不匹配的帧。
    3. 官方防呆建议: 发动作指令后先延时再轮询 0x02（1=运动, 0=停止），
       读失败重读几次。

用法:
    from grape_stem_3d.scissors_comm import Scissors485
    sc = Scissors485(port="/dev/ttyUSB0")   # 不传则自动探测
    sc.scissors_cut(timeout_s=15.0, abort_event=threading.Event())
    sc.scissors_open_wait(timeout_s=5.0, abort_event=threading.Event())
    sc.shutdown()

CLI 自测（接线后现场验证用）:
    python3 -m grape_stem_3d.scissors_comm --port /dev/ttyUSB0 ping
    python3 -m grape_stem_3d.scissors_comm --port /dev/ttyUSB0 close
    python3 -m grape_stem_3d.scissors_comm --port /dev/ttyUSB0 open
    python3 -m grape_stem_3d.scissors_comm --port /dev/ttyUSB0 stop
    python3 -m grape_stem_3d.scissors_comm --port /dev/ttyUSB0 status
"""

import glob
import os
import threading
import time
from typing import List, Optional, Tuple

import serial

from grape_stem_3d.plc_comm import crc16_modbus

# ── 协议常量 ────────────────────────────────────────────
FC_READ_HOLD = 0x03   # 读保持寄存器
FC_WRITE_SINGLE = 0x06  # 写单保持寄存器

REG_MOTOR_RUN = 0x01   # 电机运行: 写 4 = 伸出合剪
REG_MOTION = 0x02      # 运动状态: 读, 1=运动中 0=停止
REG_STOP = 0x03        # 停止: 写 1
REG_SPEED = 0x04       # 速度: 1-800 圈/min
REG_STROKE = 0x05      # 脉冲数: 9000 = 25mm 全行程
REG_HOME = 0x0A        # 回原点: 写 1 = 缩回开剪
REG_ADDR = 0x08        # 设备地址: 读, 通信自检用

VAL_CLOSE = 4          # 0x01 寄存器合剪动作值
VAL_STOP = 1           # 0x03 寄存器停止值
VAL_OPEN = 1           # 0x0A 寄存器回原点值

DEFAULT_BAUD = 9600
DEFAULT_SLAVE = 1
DEFAULT_TIMEOUT_S = 0.5
DEFAULT_RETRIES = 1    # 官方建议读 0x02 失败重读防呆

# 剪刀动作结果码（与 DobotClient 语义一致, 便于管线切换）
RESULT_DONE = "done"
RESULT_TIMEOUT = "timeout"
RESULT_ABORTED = "aborted"
RESULT_ERROR = "error"
RESULT_NO_MOTION = "no_motion"  # 0x02 从未观察到运动: 指令未生效/电机未动


# ── 帧编解码 ────────────────────────────────────────────
def build_rtu_request(slave_id: int, func: int, reg: int, value: int) -> bytes:
    """构造 RTU 请求帧: 地址 + 功能码 + 寄存器(2大端) + 值/数量(2大端) + CRC(2小端)。"""
    frame = bytes([slave_id, func]) + reg.to_bytes(2, "big") + value.to_bytes(2, "big")
    return frame + crc16_modbus(frame).to_bytes(2, "little")


def parse_rtu_frame(data: bytes) -> Optional[Tuple[int, int, bytes]]:
    """校验并解析一帧 RTU 数据，返回 (slave_id, func, payload)。

    payload 为功能码之后的原始字节（不含 CRC）。
    - CRC 错误 / 长度非法 → None（调用方丢弃重搜，天然过滤干扰码）
    - FC03: payload = 字节数(1) + 数据(N)
    - FC06: payload = 寄存器(2) + 值(2)，正常回显与请求完全一致
    """
    if len(data) < 4:
        return None
    slave_id, func = data[0], data[1]
    if crc16_modbus(data[:-2]) != int.from_bytes(data[-2:], "little"):
        return None
    if func == FC_READ_HOLD and len(data) >= 5:
        n = data[2]
        if len(data) == 5 + n:
            return slave_id, func, data[2 : 2 + 1 + n]
    elif func == FC_WRITE_SINGLE and len(data) == 8:
        return slave_id, func, data[2:6]
    return None


class Scissors485:
    """USB 转 RS485 直连剪刀手控制器的 Modbus-RTU 主站。

    - 线程安全: 同一时刻只允许一笔 Modbus 事务（内部锁）
    - 每次发送前清空接收缓冲（丢掉上一笔残留的"电机停止帧"等），
      接收时按 CRC 过滤干扰码与不匹配帧
    - 从站上电干扰帧 00 00 01 06 ... 会被 CRC 校验自动丢弃

    Args:
        port: 串口设备路径, None 时自动探测（/dev/ttyUSB* > ttyCH341USB* > ttyACM*）
        slave_id: 从站地址（默认 1）
        baud: 波特率（默认 9600）
        timeout_s: 单帧应答超时（秒）
        retries: 无应答时的重发次数（官方防呆建议重读）
        logger: 日志函数（默认 print）
    """

    def __init__(
        self,
        port: Optional[str] = None,
        slave_id: int = DEFAULT_SLAVE,
        baud: int = DEFAULT_BAUD,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        retries: int = DEFAULT_RETRIES,
        logger=print,
    ):
        self._port = port or self.detect_port()
        self._slave = slave_id
        self._baud = baud
        self._timeout_s = timeout_s
        self._retries = retries
        self._log = logger
        self._lock = threading.Lock()
        self._ser: Optional[serial.Serial] = None
        self._rx_buf = bytearray()  # 跨 read 调用的接收残留 (防同批多帧丢失)
        self._connected = False

    # ── 串口管理 ───────────────────────────────────────
    @staticmethod
    def detect_port() -> Optional[str]:
        """自动探测 USB 转 485 设备: /dev/ttyUSB* / ttyCH341USB* / ttyACM*。"""
        for pattern in ("/dev/ttyUSB*", "/dev/ttyCH341USB*", "/dev/ttyACM*"):
            hits = sorted(glob.glob(pattern))
            if hits:
                return hits[0]
        return None

    def connect(self) -> bool:
        """打开串口。"""
        if self._port is None:
            self._log("[SCISSORS] 未找到 USB 转 485 设备 (ttyUSB*/ttyCH341USB*/ttyACM*)")
            return False
        try:
            self._ser = serial.Serial(
                self._port,
                baudrate=self._baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.05,
                write_timeout=1.0,
            )
            self._connected = True
            self._log(f"[SCISSORS] 已打开 {self._port} @ {self._baud},N,8,1")
            return True
        except (OSError, serial.SerialException) as e:
            self._log(f"[SCISSORS] 打开 {self._port} 失败: {e}")
            return False

    def shutdown(self):
        """关闭串口（进程退出前调用）。"""
        self._connected = False
        if self._ser is not None:
            try:
                self._ser.close()
            except (OSError, serial.SerialException):
                pass
            self._ser = None
        self._log("[SCISSORS] 串口已关闭")

    @property
    def connected(self) -> bool:
        return self._connected

    # ── Modbus 事务 ────────────────────────────────────
    def _read_one_frame(self, deadline: float) -> Optional[bytes]:
        """读入字节流, 返回第一条 CRC 合法且长度合法的完整帧。

        干扰码（00 00 前缀、残余停止帧等）靠 CRC + 长度校验逐字节重搜过滤。
        一次 read 到多帧时, 未消费的字节保留在 self._rx_buf 供下次调用,
        避免"停止帧+应答帧同批到达"时丢失应答帧。
        """
        buf = self._rx_buf
        while time.time() < deadline and self._ser is not None:
            if len(buf) >= 4:
                # 先消费缓冲里的残留, 再读新数据
                for i in range(len(buf) - 3):
                    candidate = bytes(buf[i:])
                    frame = self._try_extract(candidate)
                    if frame is not None:
                        del buf[: i + len(frame)]
                        return frame
                if len(buf) > 256:  # 干扰字节泛滥时只保留尾部窗口
                    del buf[:-32]
            try:
                chunk = self._ser.read(64)
            except (OSError, serial.SerialException) as e:
                self._log(f"[SCISSORS] 串口读取错误: {e}")
                return None
            if chunk:
                buf.extend(chunk)
        return None

    @staticmethod
    def _try_extract(candidate: bytes) -> Optional[bytes]:
        """从字节流任意位置尝试提取一条合法帧（RTU 无帧头, 靠 CRC 判定）。"""
        if len(candidate) >= 8 and candidate[1] == FC_WRITE_SINGLE:
            if parse_rtu_frame(candidate[:8]) is not None:
                return candidate[:8]
        if len(candidate) >= 5 and candidate[1] == FC_READ_HOLD:
            n = candidate[2]
            total = 5 + n
            if len(candidate) >= total and parse_rtu_frame(candidate[:total]) is not None:
                return candidate[:total]
        return None

    def _transact(self, func: int, reg: int, value: int) -> Optional[bytes]:
        """发送一帧并等待从站应答帧（含重试）。"""
        request = build_rtu_request(self._slave, func, reg, value)
        with self._lock:
            if self._ser is None:
                return None
            for attempt in range(self._retries + 1):
                try:
                    self._ser.reset_input_buffer()  # 丢弃驱动层残留
                    self._rx_buf.clear()            # 丢弃应用层残留
                    self._ser.write(request)
                    self._ser.flush()
                except (OSError, serial.SerialException) as e:
                    self._log(f"[SCISSORS] 发送失败: {e}")
                    return None
                deadline = time.time() + self._timeout_s
                # 等待匹配的应答帧; FC06 要求逐字节回显, 残余停止帧
                # (01 06 00 03 00 01) 与请求不同, 会被继续丢弃
                while True:
                    frame = self._read_one_frame(deadline)
                    if frame is None:
                        break  # 超时, 重试或失败
                    sid, rx_func, _ = parse_rtu_frame(frame)
                    if sid != self._slave:
                        self._log(f"[SCISSORS] 忽略其他从站帧: {frame.hex()}")
                        continue
                    if rx_func != func:
                        self._log(f"[SCISSORS] 忽略残余帧: {frame.hex()}")
                        continue
                    if frame == request:  # FC06 回显; FC03 不会与此相同
                        return frame
                    if func == FC_READ_HOLD:
                        return frame  # FC03 应答与请求必然不同, 合法即用
                    self._log(f"[SCISSORS] 忽略非回显帧: {frame.hex()}")
                if attempt < self._retries:
                    self._log(f"[SCISSORS] {func:#04x}@{reg:#04x} 无应答, 重试 {attempt + 1}/{self._retries}")
            self._log(f"[SCISSORS] {func:#04x}@{reg:#04x} 无应答 (超时)")
            return None

    def read_reg(self, addr: int) -> Optional[int]:
        """读单保持寄存器（FC03）。"""
        frame = self._transact(FC_READ_HOLD, addr, 1)
        if frame is None:
            return None
        _, _, payload = parse_rtu_frame(frame)
        if payload is None or len(payload) < 3 or payload[0] < 2:
            return None
        return int.from_bytes(payload[1:3], "big")

    def write_reg(self, addr: int, value: int) -> bool:
        """写单保持寄存器（FC06），从站回显一致即成功。"""
        return self._transact(FC_WRITE_SINGLE, addr, value) is not None

    # ── 剪刀动作 ───────────────────────────────────────
    def ping(self) -> bool:
        """通信自检: 读 0x08 设备地址寄存器（只读, 安全）。"""
        addr = self.read_reg(REG_ADDR)
        self._log(f"[SCISSORS] ping 设备地址 = {addr}")
        return addr is not None

    def scissors_close(self) -> bool:
        """合剪: 0x01 写 4 = 电机轴伸出剪断果梗。"""
        ok = self.write_reg(REG_MOTOR_RUN, VAL_CLOSE)
        self._log(f"[SCISSORS] 合剪 0x01←4 {'OK' if ok else '失败'}")
        return ok

    def scissors_open(self) -> bool:
        """开剪: 0x0A 写 1 = 回原点, 电机轴缩回放果。"""
        ok = self.write_reg(REG_HOME, VAL_OPEN)
        self._log(f"[SCISSORS] 开剪 0x0A←1 {'OK' if ok else '失败'}")
        return ok

    def scissors_stop(self) -> bool:
        """停止: 0x03 写 1。"""
        ok = self.write_reg(REG_STOP, VAL_STOP)
        self._log(f"[SCISSORS] 停止 0x03←1 {'OK' if ok else '失败'}")
        return ok

    def scissors_busy(self) -> Optional[int]:
        """读 0x02 运动状态: 1=运动中, 0=已停止, None=读取失败。"""
        return self.read_reg(REG_MOTION)

    def set_speed(self, rpm: int) -> bool:
        """设置速度 0x04（1-800 圈/min）。"""
        if not 1 <= rpm <= 800:
            self._log(f"[SCISSORS] 速度 {rpm} 超出 1-800 范围")
            return False
        ok = self.write_reg(REG_SPEED, rpm)
        self._log(f"[SCISSORS] 速度 0x04←{rpm} {'OK' if ok else '失败'}")
        return ok

    def set_stroke(self, pulses: int) -> bool:
        """设置行程 0x05 脉冲数（0-65535, 9000=25mm 全行程）。"""
        if not 0 <= pulses <= 65535:
            self._log(f"[SCISSORS] 脉冲数 {pulses} 超出 0-65535 范围")
            return False
        ok = self.write_reg(REG_STROKE, pulses)
        self._log(f"[SCISSORS] 行程 0x05←{pulses} {'OK' if ok else '失败'}")
        return ok

    def _wait_idle(
        self,
        timeout_s: float,
        abort_event: threading.Event,
        interval_s: float,
        start_grace_s: float,
        require_active: bool = True,
    ) -> str:
        """轮询 0x02 直到剪刀停止（官方防呆: 发指令后延时再读）。

        剪刀手状态判断 (与机械臂 wait_motion_done 的 seen_active 判据同款):
        0x02 必须**观察到过运动(1)** 之后读到 0 才算动作有效完成 —
        一上来就读 0 说明电机从未起动 (指令未生效/无供电), 判 no_motion,
        避免"发出去就报成功"的假闭环。start_grace_s 内仍未观察到 1
        时提前判 no_motion, 不必等满整个超时窗口。

        require_active=False (开剪回原点用): 本控制器固件在回原点过程中
        不置 0x02 忙标志 (2026-09-27 实测 20ms 轮询全程为 0 而电机在动),
        此时以"写成功 + 0x02 稳定为 0 满 start_grace_s"判完成;
        读失败 (None) 不进入该分支, 仍不会误判成功。
        """
        deadline = time.time() + timeout_s
        start_deadline = time.time() + start_grace_s
        seen_active = False
        while time.time() < deadline:
            if abort_event.is_set():
                return RESULT_ABORTED
            busy = self.scissors_busy()
            if busy == 1:
                seen_active = True
            elif busy == 0 and seen_active:
                return RESULT_DONE
            elif (busy == 0 and not seen_active
                  and time.time() >= start_deadline):
                if require_active:
                    return RESULT_NO_MOTION
                self._log("[SCISSORS] 开剪: 未观察到 0x02 运动标志 "
                          "(固件回原点不报忙), 稳定 0 判定完成")
                return RESULT_DONE
            # busy=1 继续等; None 读失败重读（不计时重试）
            time.sleep(interval_s)
        return RESULT_TIMEOUT if seen_active else RESULT_NO_MOTION

    def scissors_cut(self, timeout_s: float, abort_event: threading.Event,
                     delay_s: float = 0.2, interval_s: float = 0.05,
                     start_grace_s: float = 1.0) -> str:
        """合剪并等待完成: 发 0x01←4 → 延时 → 轮询 0x02 直到 0。

        状态判断: 0x02 必须观察到运动(1) 后回到 0 才算 done,
        从未观察到运动 → no_motion (指令未生效)。

        Returns: "done" | "no_motion" | "timeout" | "aborted" | "error"
        """
        if not self.scissors_close():
            return RESULT_ERROR
        if abort_event.wait(delay_s):  # 控制器回两帧, 官方建议延时再读
            return RESULT_ABORTED
        return self._wait_idle(timeout_s, abort_event, interval_s, start_grace_s)

    def scissors_open_wait(self, timeout_s: float, abort_event: threading.Event,
                           delay_s: float = 0.2, interval_s: float = 0.05,
                           start_grace_s: float = 1.0) -> str:
        """开剪并等待完成（0x0A→1 后轮询 0x02, 判据同合剪）。"""
        if not self.scissors_open():
            return RESULT_ERROR
        if abort_event.wait(delay_s):
            return RESULT_ABORTED
        return self._wait_idle(timeout_s, abort_event, interval_s, start_grace_s)


# ── CLI 自测 ────────────────────────────────────────────
def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="剪刀手 RS485 直连自测")
    parser.add_argument("--port", default=None, help="串口设备 (默认自动探测)")
    parser.add_argument("--slave", type=lambda x: int(x, 0), default=DEFAULT_SLAVE,
                        help="从站地址 (默认 1)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ping", help="读 0x08 设备地址, 通信自检")
    sub.add_parser("status", help="读 0x02 运动状态")
    sub.add_parser("close", help="合剪 0x01←4")
    sub.add_parser("open", help="开剪 0x0A←1")
    sub.add_parser("stop", help="停止 0x03←1")
    spd = sub.add_parser("speed", help="设速度 0x04")
    spd.add_argument("rpm", type=int)
    stk = sub.add_parser("stroke", help="设行程 0x05")
    stk.add_argument("pulses", type=int)
    cut = sub.add_parser("cut", help="合剪并等待停止")
    cut.add_argument("--timeout", type=float, default=15.0)
    opn = sub.add_parser("open-wait", help="开剪并等待停止")
    opn.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    sc = Scissors485(port=args.port, slave_id=args.slave)
    if not sc.connect():
        return 1
    try:
        abort = threading.Event()
        if args.cmd == "ping":
            ok = sc.ping()
        elif args.cmd == "status":
            v = sc.scissors_busy()
            print(f"0x02 运动状态 = {v} ({'运动中' if v == 1 else '已停止' if v == 0 else '读取失败'})")
            ok = v is not None
        elif args.cmd == "close":
            ok = sc.scissors_close()
        elif args.cmd == "open":
            ok = sc.scissors_open()
        elif args.cmd == "stop":
            ok = sc.scissors_stop()
        elif args.cmd == "speed":
            ok = sc.set_speed(args.rpm)
        elif args.cmd == "stroke":
            ok = sc.set_stroke(args.pulses)
        elif args.cmd == "cut":
            print(sc.scissors_cut(args.timeout, abort))
            ok = True
        elif args.cmd == "open-wait":
            print(sc.scissors_open_wait(args.timeout, abort))
            ok = True
        else:
            ok = False
        return 0 if ok else 2
    finally:
        sc.shutdown()


if __name__ == "__main__":
    raise SystemExit(_main())
