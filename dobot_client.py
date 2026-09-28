"""DOBOT CR5 ROS2 服务客户端封装: 运动 + 剪刀手 Modbus 链 + 位姿订阅。

对接 dobot_bringup_v4 提供的服务（前缀 /dobot_bringup_ros2/srv/，见
DOBOT_6Axis_ROS2_V4-main/dobot_msgs_v4/srv/ 与 V4新增指令/README.md）。

关键事实（源码级核实）:
- MovL/MovJ 等队列指令**立即返回**，res=ErrorID(0 成功)，
  robot_return 形如 "{队列ID}"（带花括号，_parse_int_return 处理）
- 运动完成判据 = RobotMode() 返回 5（ENABLE 空闲）；7=RUNNING
- bringup 内部服务回调**无超时**（机器人不应答会永久阻塞），
  因此所有调用一律 call_async + 客户端超时兜底
- MovL 的 param_value 逐项原样拼接进 TCP 命令，必须传完整键值串
  （"user=0" 式）；speed 数值优先于 v；cp=0 为精确停点
- 剪刀手: 机械臂末端 485 透传 + Modbus 主站服务链
  （SetToolPower→SetToolMode→SetTool485→ModbusRTUCreate），
  寄存器 0x01 合剪(写 4) / 0x0A 开剪(写 1) / 0x02 运动状态(1=运动中, 0=停止)
  / 0x04 速度 / 0x05 行程(脉冲数, 9000=25mm)。
  从站只支持功能码 03/06: 写必须用 SetSingleHoldReg(FC06),
  SetHoldRegs(FC16 写多寄存器) 从站不应答, 返回 ErrorID=-1

线程模型: 由调用方传入 rclpy Node；订阅回调只做缓存+加锁；
所有服务调用可在任意线程发起（call_async + Event 等待响应）。
"""

import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np

from rclpy.node import Node

from dobot_msgs_v4.srv import (
    ClearError,
    GetHoldRegs,
    InverseKin,
    ModbusRTUCreate,
    MovL,
    RobotMode,
    SetSingleHoldReg,
    SetTool485,
    SetToolMode,
    SetToolPower,
    SpeedFactor,
    Stop,
)
from dobot_msgs_v4.msg import RobotStatus, ToolVectorActual

from grape_stem_3d.handeye_transform import tool_vector_to_T_base_tool


# ── RobotMode 模式码 ────────────────────────────────────
MODE_ENABLE = 5      # 空闲使能（运动完成判据）
MODE_RUNNING = 7     # 运行中（含脚本与 TCP 队列）
MODE_ERROR = 9       # 错误
MODE_PAUSE = 10      # 暂停
MODE_COLLISION = 11  # 碰撞

MODE_NAMES = {
    MODE_ENABLE: "ENABLE",
    MODE_RUNNING: "RUNNING",
    MODE_ERROR: "ERROR",
    MODE_PAUSE: "PAUSE",
    MODE_COLLISION: "COLLISION",
}

# wait_motion_done 的返回码
WAIT_DONE = "done"
WAIT_TIMEOUT = "timeout"
WAIT_ABORTED = "aborted"
WAIT_ROBOT_ERROR = "robot_error"
WAIT_SERVICE_LOST = "service_lost"


class ServiceTimeoutError(Exception):
    """服务调用客户端超时（bringup 内部无超时，Box 侧必须兜底）。"""


class ServiceUnavailableError(Exception):
    """服务在启动等待期内未出现。"""


class DobotClient:
    """DOBOT CR5 服务客户端封装。

    Args:
        node: rclpy Node（订阅与服务 client 挂在它上面）
        robot_cfg: configs/fusion_pipeline.yaml 的 robot: 段
        logger: 日志函数
    """

    def __init__(self, node: Node, robot_cfg: dict, logger: Callable[[str], None] = print):
        self._node = node
        self._cfg = robot_cfg
        self._log = logger
        self._prefix = robot_cfg.get("service_prefix", "/dobot_bringup_ros2/srv")
        self._service_timeout_s = float(robot_cfg.get("service_timeout_s", 5.0))

        # 剪刀手传输通道优先级:
        #   scissors.service_name 配置 → ROS 服务通道 (GHC scissor_control_node)
        #   scissors.port 配置 (如 /dev/ttyUSB0) → USB转485 直连
        #   均留空 → 机械臂末端 485 Modbus 链 (SetSingleHoldReg)
        self._scissors_service_name = (
            self._cfg.get("scissors", {}).get("service_name") or None)
        self._scissors_port = self._cfg.get("scissors", {}).get("port") or None

        # ── 服务 client ─────────────────────────────
        self._clients = {
            "movl": node.create_client(MovL, f"{self._prefix}/MovL"),
            "inverse_kin": node.create_client(InverseKin, f"{self._prefix}/InverseKin"),
            "stop": node.create_client(Stop, f"{self._prefix}/Stop"),
            "clear_error": node.create_client(ClearError, f"{self._prefix}/ClearError"),
            "robot_mode": node.create_client(RobotMode, f"{self._prefix}/RobotMode"),
            "speed_factor": node.create_client(SpeedFactor, f"{self._prefix}/SpeedFactor"),
            "modbus_create": node.create_client(ModbusRTUCreate, f"{self._prefix}/ModbusRTUCreate"),
            "get_hold_regs": node.create_client(GetHoldRegs, f"{self._prefix}/GetHoldRegs"),
            "set_tool_power": node.create_client(SetToolPower, f"{self._prefix}/SetToolPower"),
            "set_tool_mode": node.create_client(SetToolMode, f"{self._prefix}/SetToolMode"),
            "set_tool_485": node.create_client(SetTool485, f"{self._prefix}/SetTool485"),
        }
        # 末端 485 透传模式才需要 SetSingleHoldReg (FC06 写单寄存器)。
        # 直连串口模式不创建该客户端: 服务端 (dobot_bringup_v4) 尚未实现此服务,
        # 若创建, wait_for_services 会永远等不到它 → 误降级 detect-only。
        if self._scissors_port is None:
            self._clients["set_single_hold_reg"] = node.create_client(
                SetSingleHoldReg, f"{self._prefix}/SetSingleHoldReg")

        # ── 位姿/状态订阅（回调只缓存+加锁）──────────
        self._lock = threading.Lock()
        self._T_base_tool: Optional[np.ndarray] = None
        self._pose_time = 0.0
        self._is_enable = False
        self._is_connected = False

        pose_topic = robot_cfg.get("pose_topic", "/dobot_msgs_v4/msg/ToolVectorActual")
        status_topic = robot_cfg.get("status_topic", "/dobot_msgs_v4/msg/RobotStatus")
        self._pose_sub = node.create_subscription(
            ToolVectorActual, pose_topic, self._on_tool_vector, 10
        )
        self._status_sub = node.create_subscription(
            RobotStatus, status_topic, self._on_robot_status, 10
        )

        # ── 剪刀 Modbus 状态 ─────────────────────────
        self._modbus_index: Optional[int] = None  # ModbusRTUCreate 返回的主站 index
        self._scissors_ready = False
        self._scissors485 = None  # Scissors485 实例 (tty 模式)
        self._scissors_ros = None  # ScissorServiceClient 实例 (ROS 服务通道)

    # ── 订阅回调（executor 线程）──────────────────
    def _on_tool_vector(self, msg: ToolVectorActual):
        with self._lock:
            self._T_base_tool = tool_vector_to_T_base_tool(
                msg.x, msg.y, msg.z, msg.rx, msg.ry, msg.rz
            )
            self._pose_time = time.time()

    def _on_robot_status(self, msg: RobotStatus):
        with self._lock:
            self._is_enable = msg.is_enable
            self._is_connected = msg.is_connected

    # ── 通用服务调用 ───────────────────────────────
    def _call(self, client, request, timeout_s: Optional[float] = None):
        """call_async + 客户端超时兜底。

        Raises:
            ServiceTimeoutError: 超时或服务未就绪
        """
        if not client.service_is_ready():
            raise ServiceTimeoutError(
                f"服务未就绪: {client.srv_name}（bringup 未启动或机器人掉线）"
            )
        timeout = self._service_timeout_s if timeout_s is None else timeout_s
        done = threading.Event()
        box = {}

        def _cb(fut):
            box["future"] = fut
            done.set()

        future = client.call_async(request)
        future.add_done_callback(_cb)
        if not done.wait(timeout):
            raise ServiceTimeoutError(
                f"服务调用超时 {timeout:.1f}s: {client.srv_name}（机器人无响应）"
            )
        fut = box["future"]
        if fut.result() is None:
            raise ServiceTimeoutError(
                f"服务调用失败（无结果）: {client.srv_name}"
            )
        return fut.result()

    @staticmethod
    def _parse_int_return(robot_return) -> Optional[int]:
        """解析 robot_return 形如 "{13}" / "{2,3}"（取第一个值）。"""
        if robot_return is None:
            return None
        s = str(robot_return).strip().strip("{}").strip()
        if not s:
            return None
        first = s.split(",")[0].strip()
        try:
            return int(float(first))
        except ValueError:
            return None

    # ── 初始化 ─────────────────────────────────────
    def wait_for_services(self, timeout_s: Optional[float] = None) -> bool:
        """等待全部必需服务出现 (透传模式含 SetSingleHoldReg, 直连模式不含)。"""
        timeout = self._cfg.get("wait_service_timeout_s", 10.0) if timeout_s is None else timeout_s
        deadline = time.time() + timeout
        for name, client in self._clients.items():
            while time.time() < deadline:
                if client.service_is_ready():
                    break
                time.sleep(0.2)
            else:
                self._log(f"[DOBOT] 服务等待超时: {client.srv_name}")
                return False
        self._log(f"[DOBOT] 全部服务就绪 (prefix={self._prefix})")
        return True

    def initialize(self) -> bool:
        """启动初始化: 全局速度因子 + 剪刀 Modbus 链（失败只禁剪刀段）。"""
        ok = self.set_speed_factor(int(self._cfg.get("speed_factor", 50)))
        if not ok:
            self._log("[DOBOT] SpeedFactor 设置失败（继续，不致命）")
        if self._cfg.get("scissors", {}).get("enable", True):
            if self.initialize_scissors():
                self._log("[DOBOT] 剪刀 Modbus 链就绪")
            else:
                self._log("[DOBOT] 剪刀链初始化失败 → 剪枝段禁用（运动段仍可用）")
        return ok

    # ── 运动接口 ───────────────────────────────────
    def movl_pose(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        rx_deg: float,
        ry_deg: float,
        rz_deg: float,
        speed_mm_s: Optional[float] = None,
        user: Optional[int] = None,
        tool: Optional[int] = None,
        cp: Optional[int] = None,
    ) -> bool:
        """MovL 笛卡尔直线运动（队列指令，立即返回）。

        参数单位: 位置 mm，姿态 Rz·Ry·Rx 固定轴欧拉角（度）。
        """
        motion = self._cfg.get("motion", {})
        if user is None:
            user = int(motion.get("user_index", 0))
        if tool is None:
            tool = int(motion.get("tool_index", 0))
        if speed_mm_s is None:
            speed_mm_s = float(motion.get("speed_approach_mm_s", 200))
        if cp is None:
            cp = int(motion.get("cp", 0))
        # 数值字符串化: 整数值不带小数点 ("speed=200" 而非 "speed=200.0")
        speed_str = str(int(speed_mm_s)) if float(speed_mm_s).is_integer() else str(speed_mm_s)

        req = MovL.Request()
        req.mode = False  # 笛卡尔模式: a..f = x,y,z(mm) + rx,ry,rz(度)
        req.a = float(x_mm)
        req.b = float(y_mm)
        req.c = float(z_mm)
        req.d = float(rx_deg)
        req.e = float(ry_deg)
        req.f = float(rz_deg)
        req.param_value = [
            f"user={user}",
            f"tool={tool}",
            f"speed={speed_str}",
            f"cp={cp}",
        ]

        resp = self._call(self._clients["movl"], req)
        if resp.res != 0:
            self._log(f"[DOBOT] MovL 被拒: res={resp.res} ret={resp.robot_return}")
            return False
        queue_id = self._parse_int_return(resp.robot_return)
        self._log(f"[DOBOT] MovL 已入队 ({x_mm:.1f},{y_mm:.1f},{z_mm:.1f})mm "
                  f"speed={speed_mm_s} queue_id={queue_id}")
        return True

    def stop(self) -> bool:
        """立即停止运动中的动作并清空队列（全状态可用）。"""
        try:
            resp = self._call(self._clients["stop"], Stop.Request())
            return resp.res == 0
        except ServiceTimeoutError:
            return False

    def clear_error(self) -> bool:
        """清除机器人错误状态。"""
        try:
            resp = self._call(self._clients["clear_error"], ClearError.Request())
            return resp.res == 0
        except ServiceTimeoutError:
            return False

    def robot_mode(self) -> Optional[int]:
        """查询 RobotMode（5=ENABLE 空闲, 7=RUNNING, 9=ERROR, 10=PAUSE, 11=COLLISION）。"""
        resp = self._call(self._clients["robot_mode"], RobotMode.Request())
        return self._parse_int_return(resp.robot_return)

    def inverse_kin(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        rx_deg: float,
        ry_deg: float,
        rz_deg: float,
    ) -> Optional[bool]:
        """机械臂逆解预检: 目标位姿 (mm/度, base 系) 是否可解。

        Returns:
            True   — 有解 (robot_return 含关节角)
            False  — 无解 (res≠0)
            None   — 服务不可用/超时 (调用方按未知处理, 不阻塞流程)
        用途: 发 MovL 前预检, 避免不可达位姿让机械臂进 ERROR。
        """
        try:
            req = InverseKin.Request()
            req.x = float(x_mm)
            req.y = float(y_mm)
            req.z = float(z_mm)
            req.rx = float(rx_deg)
            req.ry = float(ry_deg)
            req.rz = float(rz_deg)
            req.use_joint_near = "0"
            req.joint_near = ""
            req.user = "0"
            req.tool = "0"
            resp = self._call(self._clients["inverse_kin"], req)
        except ServiceTimeoutError as e:
            self._log(f"[DOBOT] InverseKin 服务异常: {e}")
            return None
        if resp.res != 0:
            self._log(f"[DOBOT] InverseKin 无解: res={resp.res} "
                      f"pose=({x_mm:.1f},{y_mm:.1f},{z_mm:.1f})mm "
                      f"rpy=({rx_deg:.1f},{ry_deg:.1f},{rz_deg:.1f})")
            return False
        joints = (resp.robot_return or "").strip("{}")
        if not joints:
            return False
        return True

    def set_speed_factor(self, ratio: int) -> bool:
        """全局速率比 (1~100)%。"""
        req = SpeedFactor.Request()
        req.ratio = ratio
        resp = self._call(self._clients["speed_factor"], req)
        return resp.res == 0

    def emergency_brake(self) -> None:
        """Stop + ClearError 幂等组合（ABORT/失败路径专用）。"""
        for name, fn in (("Stop", self.stop), ("ClearError", self.clear_error)):
            try:
                fn()
            except Exception as e:
                self._log(f"[DOBOT] {name} 调用异常: {e}")

    # ── 运动完成等待 ───────────────────────────────
    def wait_motion_done(
        self,
        timeout_s: float,
        abort_event: threading.Event,
        poll_interval_s: Optional[float] = None,
    ) -> str:
        """轮询 RobotMode 直到运动完成。

        完成判据: 观察到 mode≠5（运动起动）之后回到 5, 或超过
        motion_start_grace_s（默认 0.3s, 覆盖"队列已入但控制器尚未切 7"
        的竞态窗口）仍为 5——两者满足其一才判 "done", 避免"尚未起动
        误判为已完成"。

        Returns:
            "done" — 运动完成 (见上)
            "timeout" — 超过 timeout_s
            "aborted" — abort_event 被置位
            "robot_error" — mode==9/10/11 (ERROR/PAUSE/COLLISION)
            "service_lost" — 连续 2 次服务调用失败
        """
        interval = poll_interval_s or float(
            self._cfg.get("motion", {}).get("poll_interval_s", 0.1)
        )
        grace = float(self._cfg.get("motion", {}).get("motion_start_grace_s", 0.3))
        fail_limit = int(self._cfg.get("service_fail_limit", 2))
        deadline = time.time() + timeout_s
        start = time.time()
        fail_cnt = 0
        seen_active = False  # 是否观察到过 mode≠5 (运动起动)

        while time.time() < deadline:
            if abort_event.is_set():
                return WAIT_ABORTED
            try:
                mode = self.robot_mode()
            except ServiceTimeoutError:
                fail_cnt += 1
                if fail_cnt >= fail_limit:
                    self._log("[DOBOT] RobotMode 连续调用失败 → service_lost")
                    return WAIT_SERVICE_LOST
                time.sleep(interval)
                continue
            if mode is None:
                fail_cnt += 1
                if fail_cnt >= fail_limit:
                    return WAIT_SERVICE_LOST
                time.sleep(interval)
                continue
            fail_cnt = 0
            if mode == MODE_ENABLE:
                if seen_active or (time.time() - start) >= grace:
                    return WAIT_DONE
                # 宽限期内且从未观察到起动: 可能是队列尚未切换, 继续观察
                time.sleep(interval)
                continue
            seen_active = True
            if mode in (MODE_ERROR, MODE_COLLISION, MODE_PAUSE):
                self._log(f"[DOBOT] RobotMode={MODE_NAMES.get(mode, mode)} → robot_error")
                return WAIT_ROBOT_ERROR
            time.sleep(interval)
        return WAIT_TIMEOUT

    # ── 位姿查询 ───────────────────────────────────
    def get_T_base_tool(self, max_age_s: Optional[float] = None) -> Optional[np.ndarray]:
        """返回 T_base_tool（4×4, 米）；超龄返回 None。"""
        with self._lock:
            if self._T_base_tool is None:
                return None
            if max_age_s is not None and time.time() - self._pose_time > max_age_s:
                return None
            return self._T_base_tool.copy()

    def pose_stale(self, timeout_s: Optional[float] = None) -> bool:
        """位姿话题是否停更（机器人掉线时 bringup 不再发布）。"""
        t = timeout_s or float(self._cfg.get("pose_timeout_s", 2.0))
        with self._lock:
            return self._pose_time == 0.0 or (time.time() - self._pose_time) > t

    def robot_ok(self) -> Tuple[bool, bool]:
        """返回 (is_enable, is_connected)。"""
        with self._lock:
            return self._is_enable, self._is_connected

    # ── 剪刀手 Modbus 链 ───────────────────────────
    def initialize_scissors(self) -> bool:
        """初始化剪刀手通道（一次，程序启动时调用）。

        通道优先级:
        1. 配置 scissors.service_name → ROS 服务通道 (/scissor/set_state,
           GHC scissor_control_node 独占剪刀 RS485, 本侧阻塞等待返回)
        2. 配置 scissors.port (如 /dev/ttyUSB0) → USB转485 直连 (Scissors485)
        3. 均未配置 → 机械臂末端 485 Modbus 链:
           SetToolPower(1) → SetToolMode(1,0) → SetTool485(9600)
           → ModbusRTUCreate(1,9600,'"N"',8,1) → 写速度(0x04)/行程(0x05)
        """
        if self._scissors_service_name:
            return self._init_scissors_ros()
        if self._scissors_port:
            return self._init_scissors_tty()
        sc = self._cfg.get("scissors", {})
        slave_id = int(sc.get("slave_id", 1))
        baud = int(sc.get("baud", 9600))

        try:
            if self._call(self._clients["set_tool_power"],
                          _req(SetToolPower, status=1)).res != 0:
                self._log("[DOBOT] SetToolPower(1) 失败")
                return False
            if self._call(self._clients["set_tool_mode"],
                          _req(SetToolMode, mode=1, type=0)).res != 0:
                self._log("[DOBOT] SetToolMode(1,0) 失败")
                return False
            if self._call(self._clients["set_tool_485"],
                          _req(SetTool485, baudrate=baud, parity="N",
                               stop=1, identify=1)).res != 0:
                self._log("[DOBOT] SetTool485 失败")
                return False

            # parity 带引号匹配官方 TCP 语法 ModbusRTUCreate(1,9600,"N",8,1)
            resp = self._call(
                self._clients["modbus_create"],
                _req(ModbusRTUCreate, slave_id=slave_id, baud=baud,
                     parity='"N"', data_bit=8, stop_bit=1),
            )
            if resp.res != 0:
                self._log(f"[DOBOT] ModbusRTUCreate 失败: res={resp.res}")
                return False
            index = self._parse_int_return(resp.robot_return)
            if index is None:
                # C++ 侧已修 (callRosService_f 回填 robot_return); 仍解析失败时
                # 首个主站默认 index=0, 回退并告警。
                self._log("[DOBOT] ModbusRTUCreate 未返回 index, 回退使用 index=0")
                index = 0
            self._modbus_index = index
            self._log(f"[DOBOT] Modbus 主站已创建 index={index}")

            # 可选: 速度 (0x04, 1-800 圈/min) 与行程 (0x05 脉冲数, 9000=25mm)
            speed = int(sc.get("close_speed_rpm", 300))
            stroke = int(sc.get("stroke_pulses", 9000))
            self._write_reg(int(sc.get("reg_close_speed", 4)), speed)
            self._write_reg(int(sc.get("reg_stroke", 5)), stroke)

            self._scissors_ready = True
            return True
        except ServiceTimeoutError as e:
            self._log(f"[DOBOT] 剪刀链初始化超时: {e}")
            return False

    @property
    def scissors_ready(self) -> bool:
        if self._scissors_ros is not None:
            # 锁存未知时不隐藏通道: 让动作路径显式失败并报 FAIL_CUT,
            # 避免静默跳过剪枝; 仅服务未上线时才报告不可用。
            return (self._scissors_ros.is_ready()
                    or self._scissors_ros.state_unknown)
        return self._scissors_ready

    # ── 剪刀手 USB转485 直连通道 ───────────────────────
    def _init_scissors_tty(self) -> bool:
        """Scissors485 直连初始化: 开串口 → ping → 写速度/行程。"""
        from grape_stem_3d.scissors_comm import Scissors485

        sc = self._cfg.get("scissors", {})
        try:
            self._scissors485 = Scissors485(
                port=self._scissors_port,
                slave_id=int(sc.get("slave_id", 1)),
                baud=int(sc.get("baud", 9600)),
                logger=self._log,
            )
            if not self._scissors485.connect():
                self._log(f"[SCISSORS] 串口打开失败: {self._scissors_port}")
                return False
            if not self._scissors485.ping():
                self._log("[SCISSORS] ping 无响应 (剪刀手未供电/接线?)")
                return False
            self._scissors485.set_speed(int(sc.get("close_speed_rpm", 300)))
            self._scissors485.set_stroke(int(sc.get("stroke_pulses", 9000)))
            self._scissors_ready = True
            self._log(f"[SCISSORS] USB485 直连就绪: {self._scissors_port} "
                      f"(slave={sc.get('slave_id', 1)}, baud={sc.get('baud', 9600)})")
            return True
        except Exception as e:
            self._log(f"[SCISSORS] 直连初始化异常: {e!r}")
            return False

    # ── 剪刀手 ROS 服务通道 (GHC scissor_control_node) ──
    def _init_scissors_ros(self) -> bool:
        """ROS 服务通道初始化: 创建 /scissor/set_state 客户端。

        剪刀 RS485 由 GHC 的 C++ 节点独占, 速度/行程寄存器配置与
        0x02 判停都在 C++ 端完成, 本侧 close()/open() 阻塞等待返回。
        """
        from grape_stem_3d.scissor_service_client import ScissorServiceClient

        sc = self._cfg.get("scissors", {})
        try:
            self._scissors_ros = ScissorServiceClient(
                self._node,
                timeout_sec=float(sc.get("service_timeout_s", 25.0)),
                service_name=self._scissors_service_name,
            )
        except Exception as e:
            self._log(f"[SCISSORS] ROS 服务客户端创建失败: {e!r}")
            return False
        self._scissors_ready = True
        if self._scissors_ros.wait_ready(timeout_sec=3.0):
            self._log(f"[SCISSORS] ROS 服务通道就绪: {self._scissors_service_name}")
        else:
            self._log(f"[SCISSORS] ROS 服务暂未上线 ({self._scissors_service_name}), "
                      f"上线后自动可用")
        return True

    def _ros_action(self, action: str, timeout_s: Optional[float],
                    abort_event: Optional[threading.Event],
                    force: bool = False) -> str:
        """ROS 通道动作适配: 阻塞等待 C++ 端完成, 映射为流程返回码。

        0x02 判停在 C++ 端完成, 本侧无需轮询; 结果映射:
        "done" | "no_motion" | "timeout" | "aborted" | "error"
        force=True (保安全开剪) 先清除未知锁存再执行 —— 合剪失败/ABORT
        时宁可开剪, 不夹持果梗滞留 (pick_flow 既定策略)。
        """
        from grape_stem_3d.scissor_service_client import (
            ScissorError,
            ScissorStateUnknown,
        )

        client = self._scissors_ros
        if force:
            client.reset()
        try:
            fn = client.close if action == "close" else client.open
            fn(abort_event)
            return "done"
        except ScissorStateUnknown as exc:
            msg = str(exc)
            if abort_event is not None and abort_event.is_set():
                return "aborted"
            # C++ 侧 message 为英文大写开头 ("No motion detected..." /
            # "Timeout: ..."), 小写化后再匹配, 避免超时类失败被误归为 "error"
            if "no motion" in msg.lower():
                return "no_motion"
            if "超时" in msg or "timeout" in msg.lower():
                return "timeout"
            return "error"
        except ScissorError as exc:
            msg = str(exc)
            if abort_event is not None and abort_event.is_set():
                return "aborted"
            self._log(f"[SCISSORS] ROS 通道动作失败: {msg}")
            return "error"
        except Exception as exc:
            self._log(f"[SCISSORS] ROS 通道动作异常: {exc!r}")
            return "error"

    def scissors_close(self) -> bool:
        """合剪（0x01 写 4）: 电机轴伸出剪断果梗。"""
        if self._scissors_ros is not None:
            return self._ros_action("close", None, None) == "done"
        if self._scissors485 is not None:
            return self._scissors485.scissors_close()
        sc = self._cfg.get("scissors", {})
        return self._write_reg(int(sc.get("reg_close", 1)),
                               int(sc.get("close_val", 4)))

    def scissors_open(self) -> bool:
        """开剪（0x0A 写 1）: 电机轴缩回放果。"""
        if self._scissors_ros is not None:
            return self._ros_action("open", None, None) == "done"
        if self._scissors485 is not None:
            return self._scissors485.scissors_open()
        return self._write_reg(int(self._cfg.get("scissors", {}).get("reg_open", 10)), 1)

    def scissors_busy(self) -> Optional[int]:
        """读 0x02 运动状态寄存器: 1=运动中, 0=已停止, None=读取失败。"""
        if self._scissors_ros is not None:
            # ROS 通道由 C++ 节点内部判停且服务调用串行化, 无独立状态
            # 可读, 视为空闲放行 (合剪前到位判定等流程仍照常执行)。
            return 0
        if self._scissors485 is not None:
            return self._scissors485.scissors_busy()
        return self._read_reg(int(self._cfg.get("scissors", {}).get("reg_motion", 2)))

    def scissors_cut(self, timeout_s: float, abort_event: threading.Event) -> str:
        """合剪并等待完成: 发 0x03 → 延时 cmd_delay_s → 轮询 0x02 直到 0。

        状态判断: 0x02 必须观察到运动(1) 后回到 0 才算 done,
        从未观察到运动 → "no_motion" (指令未生效)。

        Returns: "done" | "no_motion" | "timeout" | "aborted" | "error"
        """
        sc = self._cfg.get("scissors", {})
        start_grace = float(sc.get("start_grace_s", 1.0))
        if self._scissors_ros is not None:
            return self._ros_action("close", timeout_s, abort_event)
        if self._scissors485 is not None:
            return self._scissors485.scissors_cut(
                timeout_s, abort_event, start_grace_s=start_grace
            )
        delay = float(sc.get("cmd_delay_s", 0.2))
        interval = float(sc.get("poll_interval_s", 0.05))

        if not self.scissors_close():
            return "error"
        if abort_event.wait(delay):  # 官方建议: 发指令后延时（控制器回两帧）
            return "aborted"
        return self._wait_scissors_idle(
            timeout_s, abort_event, interval, start_grace
        )

    def scissors_open_wait(self, timeout_s: float, abort_event: threading.Event,
                           force: bool = False) -> str:
        """开剪并等待完成（0x0A → 轮询 0x02, 判据同合剪）。

        force=True: 仅 ROS 通道生效, 先清除未知锁存再开剪 —— 供保安全
        开剪使用 (合剪失败/ABORT 时宁可开剪, 不夹持果梗滞留)。
        """
        sc = self._cfg.get("scissors", {})
        start_grace = float(sc.get("start_grace_s", 1.0))
        if self._scissors_ros is not None:
            return self._ros_action("open", timeout_s, abort_event, force=force)
        if self._scissors485 is not None:
            return self._scissors485.scissors_open_wait(
                timeout_s, abort_event, start_grace_s=start_grace
            )
        delay = float(sc.get("cmd_delay_s", 0.2))
        interval = float(sc.get("poll_interval_s", 0.05))

        if not self.scissors_open():
            return "error"
        if abort_event.wait(delay):
            return "aborted"
        return self._wait_scissors_idle(
            timeout_s, abort_event, interval, start_grace
        )

    def _wait_scissors_idle(
        self,
        timeout_s: float,
        abort_event: threading.Event,
        interval: float,
        start_grace_s: float,
    ) -> str:
        """轮询 0x02 寄存器直到剪刀停止。

        剪刀手状态判断 (与 wait_motion_done 的 seen_active 判据同款):
        必须观察到过运动(1) 之后读到 0 才判 done; start_grace_s 内
        始终读 0 (电机未起动) → "no_motion", 避免假成功。
        """
        deadline = time.time() + timeout_s
        start_deadline = time.time() + start_grace_s
        seen_active = False
        while time.time() < deadline:
            if abort_event.is_set():
                return "aborted"
            busy = self.scissors_busy()
            if busy == 1:
                seen_active = True
            elif busy == 0 and seen_active:
                return "done"
            elif busy == 0 and not seen_active and time.time() >= start_deadline:
                return "no_motion"
            # busy=None 读失败不计时重试, 但累计到最后仍是 no_motion/timeout
            time.sleep(interval)
        return "timeout" if seen_active else "no_motion"

    def _write_reg(self, addr: int, value: int) -> bool:
        """SetSingleHoldReg 写单保持寄存器（FC06）。

        剪刀手从站只支持功能码 03(读)/06(写单), SetHoldRegs 走 FC16
        从站不应答, ErrorID=-1, 所以统一用本函数。
        """
        if self._modbus_index is None:
            self._log("[DOBOT] Modbus 主站未初始化, 无法写寄存器")
            return False
        client = self._clients.get("set_single_hold_reg")
        if client is None:
            self._log("[DOBOT] SetSingleHoldReg 客户端未创建 (直连串口模式), 无法写寄存器")
            return False
        req = SetSingleHoldReg.Request()
        req.index = self._modbus_index
        req.addr = addr
        req.val = value
        try:
            resp = self._call(client, req)
        except ServiceTimeoutError as e:
            self._log(f"[DOBOT] SetSingleHoldReg({addr:#04x}) 超时: {e}")
            return False
        if resp.res != 0:
            self._log(f"[DOBOT] SetSingleHoldReg({addr:#04x}) 失败: res={resp.res}")
        return resp.res == 0

    def _read_reg(self, addr: int, val_type: str = "U16") -> Optional[int]:
        """GetHoldRegs 读单寄存器, 返回解析后的值。"""
        if self._modbus_index is None:
            return None
        req = GetHoldRegs.Request()
        req.index = self._modbus_index
        req.addr = addr
        req.count = 1
        req.val_type = val_type
        try:
            resp = self._call(self._clients["get_hold_regs"], req)
        except ServiceTimeoutError as e:
            self._log(f"[DOBOT] GetHoldRegs({addr:#04x}) 超时: {e}")
            return None
        return self._parse_int_return(resp.robot_return)


def _req(srv_cls, **kwargs):
    """构造服务请求对象并赋值字段。"""
    req = srv_cls.Request()
    for k, v in kwargs.items():
        setattr(req, k, v)
    return req
