"""PLC 采摘流程控制器（纯逻辑，机器人/检测器/PLC 全部可注入）。

一次采摘流程（worker 线程内执行）:

    WAIT_FRAME ─► DETECT ─► MOVL_APPROACH ─► MOVL_GRASP ─► CUT(合剪)
    ─► MOVL_PLACE ─► OPEN(开剪放果) ─► MOVL_RETREAT ─► DONE ─► IDLE

motion.send_via_topic=true 时机械臂交由 MoveIt 侧执行, 运动段换为
(见 _topic_pick_sequence):
    发布 /target_pose ─► 延时 close_delay_s ─► CUT(合剪)
    ─► [place_position_mm 已示教] 发布放果点 ─► 延时 open_delay_s
    ─► OPEN(开剪放果) ─► 发布 home 回初始位 ─► DONE ─► IDLE

每个阻塞点三要素: 超时上限 + abort_event 检查 + 失败归类上报。
任何路径（含异常）finally 中复位 busy, 保证可再次 START。

安全隐患设计（对应审查结论）:
- ABORT: 主线程调用 abort() 时**立即** robot.emergency_brake()（Stop 清运动+队列），
  worker 在 ≤poll_interval 内发现 abort_event, 幂等重复刹车后上报
  ABORTED → IDLE。
- 相机故障: WAIT_FRAME 阶段要求帧龄 ≤ frame_wait_timeout_s（默认 0.5s），
  无新帧即上报 FAIL_CAMERA（主循环侧的连续读失败计数另行处理）。
- 运动超时/机器人 ERROR(9)/COLLISION(11): emergency_brake 后上报
  FAIL_ROBOT_MOTION；服务连续失败且位姿停更上报 FAIL_NO_ROBOT_POSE。
- 剪刀超时: 尽力开剪（保安全）后上报 FAIL_CUT，需人工确认再继续。
- 剪刀状态判断: 合剪前读 0x02 确认空闲（busy/无响应 → FAIL_CUT 不盲目合剪）;
  合剪/开剪后必须观察到运动(1)→停止(0) 才判成功, 从未运动判 no_motion
  → FAIL_CUT（防"指令未生效却报 DONE"的假闭环）。

线程模型: PLC 回调线程只调 is_busy（加锁布尔）+ 置标志；本控制器
内部有 _busy_lock / _frame_cond / _abort_event，均线程安全。
"""

import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np

from grape_stem_3d import plc_comm as plc
from grape_stem_3d.cut_pose import (
    calculate_target_pose,
    rotation_matrix_to_rpy_degrees,
    rpy_degrees_to_rotation_matrix,
)

from grape_stem_3d.calculate_cut_pose import calculate_cut_pose
from grape_stem_3d.handeye_transform import camera_to_base_link


# detect_fn 的返回状态
DETECT_OK = "ok"
DETECT_NO_TARGET = "no_target"
DETECT_INVALID_DEPTH = "invalid_depth"
DETECT_ERROR = "error"


class PickFlowController:
    """一次 START 触发一轮采摘流程；单飞 worker 线程执行。

    Args:
        plc_client: 有 send_status(state) 方法的对象（PlcClient 或 Fake）
        robot: DobotClient 实例；None 表示不驱动机械臂（detect-only）
        detect_fn: 检测函数, 签名 detect_fn(rgbd_frame)
                   → (status, camera_xyz[, cut_info]) 或 (status, None)
                   cut_info 为可选第三元素 (新链 keypoint):
                   {"p_up_camera": (x,y,z), "p_cut_camera": (x,y,z),
                    "p_down_camera": (x,y,z)} 相机系果梗三点;
                   缺失/None 时回退固定姿态
        T_tool_camera: 4×4 手眼矩阵（camera → tool TCP）
        cfg: configs/fusion_pipeline.yaml 完整配置
        inference_lock: 可选 threading.Lock, 检测时持有（与手动 d 键互斥）
        detect_only: True 时检测成功即报 DONE, 不调用任何机器人接口
        logger: 日志函数
    """

    def __init__(
        self,
        plc_client,
        robot,
        detect_fn: Callable[[object], Tuple[str, Optional[Tuple[float, float, float]], Optional[dict]]],
        T_tool_camera: np.ndarray,
        cfg: dict,
        inference_lock: Optional[threading.Lock] = None,
        detect_only: bool = False,
        logger: Callable[[str], None] = print,
        publish_target_pose: Optional[Callable] = None,
        publish_home_pose: Optional[Callable] = None,
    ):
        self._plc = plc_client
        self._robot = robot
        self._detect_fn = detect_fn
        self._T_tool_camera = T_tool_camera
        self._cfg = cfg
        self._motion_cfg = cfg.get("robot", {}).get("motion", {})
        self._scissors_cfg = cfg.get("robot", {}).get("scissors", {})
        self._inference_lock = inference_lock
        self._detect_only = detect_only or robot is None
        self._log = logger
        # 话题发送模式 (motion.send_via_topic): PICK 只发布 /target_pose
        # (MoveIt 侧 grape_arm_control 消费), 不直接调 DOBOT MovL
        self._send_via_topic = bool(
            self._motion_cfg.get("send_via_topic", False)
        )
        self._publish_target_pose = publish_target_pose
        self._publish_home_pose = publish_home_pose
        # 夹爪 TCP 相对 Link6 法兰的偏移 (Link6 系, 米) — 与发布端
        # publish_target_pose 的补偿同源同值
        self._gripper_offset_m = np.asarray(
            cfg.get("target_pose", {}).get("gripper_offset_m", [0.0, 0.0, 0.0]),
            dtype=float,
        )

        # 帧槽（主线程 submit_frame 写入, worker 读取）
        self._frame_cond = threading.Condition()
        self._frame_info: Optional[Tuple[object, float]] = None  # (frame, ts)

        # 流程状态
        self._busy = False
        self._busy_lock = threading.Lock()
        self._abort_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_target_base: Optional[Tuple[float, float, float]] = None
        self._scissors_closed = False  # 剪刀当前是否处于合剪(夹持)状态

        # d 键冻结的检测结果 (手动模式: PICK 直接使用, 不再重新取帧检测)
        self._det_lock = threading.Lock()
        self._last_detection: Optional[dict] = None

    @property
    def last_target_base(self) -> Optional[Tuple[float, float, float]]:
        """最近一次采摘的目标点（base 系, 米）; 仅 UI 显示用。"""
        return self._last_target_base

    # ── 主线程接口 ──────────────────────────────────
    @property
    def is_busy(self) -> bool:
        """PLC 回调线程用: START 时决定 ACK_OK / ACK_BUSY。"""
        with self._busy_lock:
            return self._busy

    def submit_frame(self, rgbd_frame) -> None:
        """主循环每帧调用: 存入帧槽并唤醒 worker。"""
        with self._frame_cond:
            self._frame_info = (rgbd_frame, time.time())
            self._frame_cond.notify_all()

    def set_last_detection(self, camera_xyz, cut_info, T_base_tool) -> None:
        """主循环 'd' 键检测成功后调用: 冻结本轮采摘使用的检测结果。

        PICK 启动时若存在冻结结果, 不再重新取帧/检测, 直接使用这里的
        camera_xyz 与 d 时刻的机械臂位姿 T_base_tool, 保证机械臂按
        d 键显示的目标坐标采摘; 重新按 d 即覆盖。
        """
        with self._det_lock:
            self._last_detection = {
                "camera_xyz": camera_xyz,
                "cut_info": cut_info,
                "T_base_tool": T_base_tool,
            }

    def start_pick(self) -> bool:
        """消费 START 后调用; 已在流程中返回 False。"""
        with self._busy_lock:
            if self._busy:
                return False
            self._busy = True
        self._abort_event.clear()
        self._worker = threading.Thread(
            target=self._run_pick, daemon=True, name="PickFlow"
        )
        self._worker.start()
        return True

    def abort(self) -> None:
        """消费 ABORT 后调用（主线程, 立即生效）:

        abort_event 置位 + **立即** Stop 急停+清队列, 不等待 worker。
        """
        self._log("[PICK] ABORT: 置中止标志 + 立即急停机械臂")
        self._abort_event.set()
        if self._robot is not None:
            self._robot.emergency_brake()

    def notify_camera_failed(self) -> None:
        """主循环检测到相机连续读失败时调用。

        PICKING 中的相机故障由 WAIT_FRAME 的帧龄检查兜底
        （新帧停更 > frame_wait_timeout_s 即报 FAIL_CAMERA）；
        此处记录日志便于现场定位。
        """
        self._log("[PICK] 相机读帧失败（主循环通知）")

    def report_camera_failed(self) -> None:
        """主循环相机连续读失败达阈值时调用（IDLE 态上报 FAIL_CAMERA）。"""
        self._log("[PICK] 相机连续读失败 → 上报 FAIL_CAMERA")
        self._plc.send_status(plc.STATE_FAIL_CAMERA)
        self._plc.send_status(plc.STATE_IDLE)

    def report_aborted(self) -> None:
        """空闲期收到 ABORT 时调用: 上报 ABORTED + IDLE, 避免 PLC 看门狗误报。"""
        self._log("[PICK] 空闲期 ABORT → 上报 ABORTED")
        # 【高-2】剪刀仍处于夹持状态时尽力开剪, 避免夹着果梗滞留
        # (话题模式下合剪后即报 DONE, 剪刀可能一直闭着直到下次 ABORT)
        if self._scissors_closed:
            self._safety_open()
        self._plc.send_status(plc.STATE_ABORTED)
        self._plc.send_status(plc.STATE_IDLE)

    def stop(self, join_timeout_s: float = 3.0) -> None:
        """退出清理: 置中止标志并等待 worker 结束。"""
        self._abort_event.set()
        w = self._worker
        if w is not None and w.is_alive():
            w.join(join_timeout_s)

    # ── worker: 状态机 ──────────────────────────────
    def _run_pick(self) -> None:
        try:
            self._log("[PICK] 采摘流程开始")
            self._plc.send_status(plc.STATE_PICKING)

            if self._aborted():
                self._finish_aborted()
                return

            # 优先使用 d 键冻结的检测结果 (手动模式): 不重新取帧/检测,
            # 机械臂按 d 时刻的坐标与位姿采摘; 无冻结结果 (PLC 自动模式)
            # 时才退回自动取帧检测路径。
            with self._det_lock:
                frozen = self._last_detection
            if frozen is not None:
                camera_xyz = frozen["camera_xyz"]
                cut_info = frozen["cut_info"]
                T_base_tool_frozen = frozen["T_base_tool"]
                self._log(
                    "[PICK] 使用 d 键冻结检测结果: camera=("
                    f"{camera_xyz[0]:.3f},{camera_xyz[1]:.3f},"
                    f"{camera_xyz[2]:.3f})m"
                )
            else:
                self._log("[PICK] 无 d 键检测结果, 退回自动取帧检测 (PLC 自动模式)")
                frame = self._wait_fresh_frame()
                if self._aborted():
                    self._finish_aborted()
                    return
                if frame is None:
                    self._fail(plc.STATE_FAIL_CAMERA, "无新帧（相机故障）")
                    return

                packed = self._detect(frame)
                status = packed[0]
                camera_xyz = packed[1] if len(packed) > 1 else None
                cut_info = packed[2] if len(packed) > 2 else None
                if self._aborted():
                    self._finish_aborted()
                    return
                if status != DETECT_OK:
                    fail = {
                        DETECT_NO_TARGET: plc.STATE_FAIL_NO_TARGET,
                        DETECT_INVALID_DEPTH: plc.STATE_FAIL_INVALID_DEPTH,
                        DETECT_ERROR: plc.STATE_FAIL_UNKNOWN,
                    }.get(status, plc.STATE_FAIL_UNKNOWN)
                    self._fail(fail, f"检测失败: {status}")
                    return
                T_base_tool_frozen = None  # 自动路径下用抓取时刻的实时位姿

            # ── 手眼变换: P_base = T_base_tool @ T_tool_camera @ P_camera
            target_base = None
            target_rpy = None  # 新链动态姿态; None 时回退固定姿态
            self._last_target_base = None
            if self._robot is not None:
                if T_base_tool_frozen is not None:
                    # d 键冻结路径: 用 d 时刻的位姿, 保证与 d 显示的坐标一致
                    T_base_tool = T_base_tool_frozen
                else:
                    pose_timeout = float(
                        self._cfg.get("robot", {}).get("pose_timeout_s", 2.0)
                    )
                    T_base_tool = self._robot.get_T_base_tool(max_age_s=pose_timeout)
                if T_base_tool is None:
                    self._fail(
                        plc.STATE_FAIL_NO_ROBOT_POSE,
                        "未收到机械臂 TCP 位姿 (d 键冻结时请重新按 d)",
                    )
                    return
                target_base = camera_to_base_link(
                    camera_xyz, self._T_tool_camera, T_base_tool
                )
                self._last_target_base = target_base
                self._log(
                    f"[PICK] 目标 base=({target_base[0]:.3f},{target_base[1]:.3f},"
                    f"{target_base[2]:.3f})m"
                )
                # 工作半径门禁已按现场要求移除: 不做距离检查, 直接下发

                # ── 六自由度剪切姿态 (新链 keypoint) ──
                # 按果梗三点方向动态计算工具姿态; 计算失败回退配置固定角。
                # use_dynamic_orientation: false 时始终用示教固定姿态
                if (cut_info is not None
                        and self._motion_cfg.get("use_dynamic_orientation", True)):
                    try:
                        p_up = camera_to_base_link(
                            cut_info["p_up_camera"], self._T_tool_camera, T_base_tool
                        )
                        p_cut = camera_to_base_link(
                            cut_info["p_cut_camera"], self._T_tool_camera, T_base_tool
                        )
                        p_down = camera_to_base_link(
                            cut_info["p_down_camera"], self._T_tool_camera, T_base_tool
                        )
                        # 当前工具姿态: T_base_tool 的旋转部分 = 工具轴在基系
                        # 下的表示, ZYX 约定与 DOBOT MovL 一致
                        current_rpy = rotation_matrix_to_rpy_degrees(T_base_tool[:3, :3])
                        self._log(
                            "[PICK] 果梗三点(base): "
                            f"up=({p_up[0]:.3f},{p_up[1]:.3f},{p_up[2]:.3f}) "
                            f"cut=({p_cut[0]:.3f},{p_cut[1]:.3f},{p_cut[2]:.3f}) "
                            f"down=({p_down[0]:.3f},{p_down[1]:.3f},{p_down[2]:.3f})"
                        )
                        self._log(
                            f"[PICK] d 时刻臂姿 rpy=({current_rpy[0]:.1f},"
                            f"{current_rpy[1]:.1f},{current_rpy[2]:.1f})"
                        )

                        p_arm = T_base_tool[:3, 3]          # 当前末端在基系中的位置


                        current_z = T_base_tool[:3, 2]
                        cut_pose = calculate_cut_pose(
                            p_up,      # 果梗上点 U
                            p_down,    # 果梗下点 D
                            p_cut,     # 剪切点 C
                            p_arm,     # 当前末端位置 E
                            current_z, # 当前 TCP Z 轴方向
                        )
                        pose = calculate_target_pose(p_up, p_cut, p_down, current_rpy)

                        self._log(f"[PICK] cut_pose raw = {cut_pose!r}")
                        self._log(f"[PICK] pose raw = {pose!r}")

                        rpy_camera = (float(pose[5]), float(pose[4]), float(pose[3]))
                        # rpy_camera = (-90.0 , 0.0, -180)

                        self._log(
                            f"[PICK] rpy_camera(相机系) = "
                            f"({rpy_camera[0]:.2f}, {rpy_camera[1]:.2f}, {rpy_camera[2]:.2f})"
                        )


                        # 目标姿态从相机系转到机械臂基系再下发:
                        # R_base = R_base_camera @ R_camera,
                        # R_base_camera = (T_base_tool @ T_tool_camera)[:3,:3]
                        R_base_camera = (T_base_tool @ self._T_tool_camera)[:3, :3]
                        target_rpy = tuple(
                            float(v) for v in rotation_matrix_to_rpy_degrees(
                                R_base_camera @ rpy_degrees_to_rotation_matrix(rpy_camera)
                            )
                        )

                        self._log(
                            f"[PICK] target_rpy(基系, 最终发送) = "
                            f"({target_rpy[0]:.2f}, {target_rpy[1]:.2f}, {target_rpy[2]:.2f})"
                        )

                        self._log(
                            f"[PICK] 动态剪切姿态(相机系) rx={rpy_camera[0]:.1f} "
                            f"ry={rpy_camera[1]:.1f} rz={rpy_camera[2]:.1f}"
                        )
                        self._log(
                            f"[PICK] 基系目标姿态 rx={target_rpy[0]:.1f} "
                            f"ry={target_rpy[1]:.1f} rz={target_rpy[2]:.1f}"
                        )
                    except (ValueError, KeyError, TypeError) as e:
                        self._log(f"[PICK] 动态姿态计算失败, 回退固定姿态: {e!r}")

            if self._detect_only:
                # detect-only: 检测成功即完成, 不动机器人
                self._log("[PICK] detect-only: 检测成功, 不驱动机械臂")
                self._plc.send_status(plc.STATE_DONE)
                self._plc.send_status(plc.STATE_IDLE)
                return

            # ── 运动前安全检查: 机器人必须已使能 ──
            # (C++ 侧无自动上电状态机, 未使能时下发队列指令会被拒或挂起;
            #  提前拦截并明确上报, 避免"队列挂起→超时"的模糊故障)
            try:
                is_enable, is_connected = self._robot.robot_ok()
            except Exception:
                is_enable, is_connected = True, True  # 注入式 Fake 无此接口时放行
            # 状态话题 is_enable 可能滞后/误报: 已连接但报未使能时,
            # 以 RobotMode 实测为准 (5=ENABLE 空闲, 与 wait_motion_done 判据一致)
            mode = None
            if is_connected and not is_enable:
                try:
                    mode = self._robot.robot_mode()
                except Exception:
                    mode = None
                if mode == 5:
                    is_enable = True
                    self._log("[PICK] 状态话题报未使能, RobotMode=5 实测已使能 → 放行")
            if not is_enable or not is_connected:
                self._fail(
                    plc.STATE_FAIL_ROBOT_MOTION,
                    f"机器人未就绪 (enable={is_enable}, connected={is_connected}, "
                    f"RobotMode={mode}), "
                    "请上电/使能/清错后重试",
                )
                return

            # ── 运动与剪枝 ──
            # 注: 不再做剪刀未就绪的 FAIL_CUT 门禁 — 剪刀不可用时
            # 各执行段自行跳过剪枝 (仅运动 + 放果), 不阻断流程。
            if target_rpy is not None:
                rx, ry, rz = target_rpy
            else:
                rx, ry, rz = self._end_orientation()
                self._log(
                    f"[PICK] 使用固定剪切姿态 rpy=({rx:.1f},{ry:.1f},{rz:.1f})"
                )

            # 话题发送模式: 只发布 /target_pose (MoveIt 侧执行), 不调 DOBOT MovL
            if self._send_via_topic:
                if self._publish_target_pose is None:
                    self._fail(
                        plc.STATE_FAIL_UNKNOWN,
                        "send_via_topic 开启但未注入 /target_pose 发布器",
                    )
                    return
                if self._topic_pick_sequence(target_base, (rx, ry, rz)):
                    self._plc.send_status(plc.STATE_DONE)
                    self._plc.send_status(plc.STATE_IDLE)
                return

            # 逆解预检已按现场要求移除: 不做 IK 检查, 直接下发 MovL
            # (不可达时机械臂会进 ERROR, 需 ClearError 恢复)
            self._log(
                f"[PICK] 跳过逆解预检, 直接发送目标 "
                f"base=({target_base[0]:.3f},{target_base[1]:.3f},"
                f"{target_base[2]:.3f})m rpy=({rx:.1f},{ry:.1f},{rz:.1f})"
            )

            approach = self._offset(target_base, self._motion_cfg.get("approach_offset_m", [0.0, 0.0, 0.0]))
            place, place_rx, place_ry, place_rz = self._place_pose(target_base, rx, ry, rz)

            phases = [
                ("接近点", approach, rx, ry, rz, "approach"),
                ("抓取点", target_base, rx, ry, rz, "grasp"),
            ]
            for label, xyz, prx, pry, prz, kind in phases:
                result = self._do_motion(label, xyz, prx, pry, prz, kind)
                if result == "aborted":
                    self._finish_aborted()
                    return
                if result != "done":
                    self._motion_fail(result)
                    return

            # ── 剪枝（合剪前状态判断 → 合剪 → 轮询 0x02）──
            cut_failed = False
            if self._scissors_usable():
                # 剪刀状态判断: 读 0x02 确认空闲再合剪 (busy/无响应 → FAIL_CUT)
                if not self._pre_cut_idle_check():
                    return
                result = self._robot.scissors_cut(
                    float(self._scissors_cfg.get("cut_timeout_s", 15.0)),
                    self._abort_event,
                )
                if result == "aborted":
                    # 【高-2】合剪途中 ABORT: 尽力开剪保安全后再上报
                    self._scissors_closed = True
                    self._safety_open()
                    self._finish_aborted()
                    return
                if result != "done":
                    if result == "no_motion":
                        self._log("[PICK] 剪刀无动作 (0x02 未观察到运动), "
                                  "合剪指令可能未生效")
                    else:
                        self._log(f"[PICK] 合剪失败: {result}, 尽力开剪保安全")
                    self._scissors_closed = True
                    self._safety_open()
                    cut_failed = True
                else:
                    self._scissors_closed = True
            else:
                self._log("[PICK] 剪刀不可用, 跳过剪枝段")

            # ── 放果点 ──
            result = self._do_motion("放果点", place, place_rx, place_ry, place_rz, "place")
            if result == "aborted":
                self._finish_aborted()
                return
            if result != "done":
                # 【中-3】剪刀状态未知时, 任何运动失败都按 FAIL_CUT 上报
                if cut_failed:
                    self._fail(plc.STATE_FAIL_CUT, f"放果运动失败 ({result}), 且剪刀状态未知")
                else:
                    self._motion_fail(result)
                return

            # ── 开剪放果 ──
            if self._scissors_usable():
                result = self._robot.scissors_open_wait(
                    float(self._scissors_cfg.get("open_timeout_s", 5.0)),
                    self._abort_event,
                )
                if result == "aborted":
                    self._finish_aborted()
                    return
                if result != "done":
                    if result == "no_motion":
                        self._log("[PICK] 开剪无动作 (0x02 未观察到运动), "
                                  "指令可能未生效")
                    else:
                        self._log(f"[PICK] 开剪失败: {result} — 继续退回, 最终报 FAIL_CUT")
                    cut_failed = True
                else:
                    self._scissors_closed = False

            # ── 退回接近点 ──
            result = self._do_motion("退回", approach, rx, ry, rz, "approach")
            if result == "aborted":
                self._finish_aborted()
                return
            if result != "done":
                if cut_failed:
                    self._fail(plc.STATE_FAIL_CUT, f"退回运动失败 ({result}), 且剪刀状态未知")
                else:
                    self._motion_fail(result)
                return

            if cut_failed:
                self._fail(plc.STATE_FAIL_CUT, "剪枝/开剪失败, 需人工确认剪刀与果藤状态")
                return

            self._log("[PICK] 采摘完成")
            self._plc.send_status(plc.STATE_DONE)
            self._plc.send_status(plc.STATE_IDLE)
        except Exception as e:
            self._log(f"[PICK] 流程异常: {e!r}")
            if self._robot is not None:
                self._robot.emergency_brake()
            self._plc.send_status(plc.STATE_FAIL_UNKNOWN)
            self._plc.send_status(plc.STATE_IDLE)
        finally:
            with self._busy_lock:
                self._busy = False

    # ── 内部工具 ────────────────────────────────────
    def _aborted(self) -> bool:
        return self._abort_event.is_set()

    def _finish_aborted(self) -> None:
        self._log("[PICK] ABORT: 上报 ABORTED → IDLE")
        if self._robot is not None:
            self._robot.emergency_brake()  # 幂等兜底
        # 【高-2】剪刀处于合剪(夹持)状态时, 尽力开剪保安全 (协议 §5.4)
        if self._scissors_closed:
            self._safety_open()
        self._plc.send_status(plc.STATE_ABORTED)
        self._plc.send_status(plc.STATE_IDLE)

    def _safety_open(self) -> None:
        """尽力开剪一次 (独立短超时, 不受 abort_event 影响)。

        用于: ABORT/合剪失败时保证剪刀不夹持果藤滞留。
        """
        try:
            result = self._robot.scissors_open_wait(
                float(self._scissors_cfg.get("open_timeout_s", 5.0)),
                threading.Event(),  # 保安全开剪不受 abort 影响
            )
            self._log(f"[PICK] 保安全开剪: {result}")
            if result == "done":
                self._scissors_closed = False
        except Exception as e:
            self._log(f"[PICK] 保安全开剪异常: {e!r}")

    def _fail(self, state: int, reason: str) -> None:
        self._log(f"[PICK] 失败 {plc.state_name(state)}: {reason}")
        self._plc.send_status(state)
        self._plc.send_status(plc.STATE_IDLE)

    def _wait_fresh_frame(self, timeout_s: Optional[float] = None) -> Optional[object]:
        """等待一帧足够新的图像（帧龄 ≤ timeout_s），超时返回 None。"""
        timeout = timeout_s or float(
            self._cfg.get("robot", {}).get("motion", {}).get("frame_wait_timeout_s", 0.5)
        )
        deadline = time.time() + timeout
        with self._frame_cond:
            while True:
                if self._abort_event.is_set():
                    return None  # ABORT: 立即退出等待, 由 _run_pick 上报 ABORTED
                if self._frame_info is not None:
                    frame, ts = self._frame_info
                    if time.time() - ts <= timeout:
                        return frame
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                # 小步长等待, 保证 ABORT 在 ≤0.1s 内被感知
                self._frame_cond.wait(min(remaining, 0.1))

    def _detect(self, frame) -> Tuple[
        str, Optional[Tuple[float, float, float]], Optional[dict]
    ]:
        if self._inference_lock is not None:
            with self._inference_lock:
                return self._detect_fn(frame)
        return self._detect_fn(frame)

    def _end_orientation(self) -> Tuple[float, float, float]:
        m = self._motion_cfg
        return (
            float(m.get("end_pose_rx_deg", 0.0)),
            float(m.get("end_pose_ry_deg", 0.0)),
            float(m.get("end_pose_rz_deg", 0.0)),
        )

    @staticmethod
    def _offset(xyz: Tuple[float, float, float], offset) -> Tuple[float, float, float]:
        off = offset or [0.0, 0.0, 0.0]
        return (xyz[0] + float(off[0]), xyz[1] + float(off[1]), xyz[2] + float(off[2]))

    def _place_pose(
        self, target_base, rx, ry, rz
    ) -> Tuple[Tuple[float, float, float], float, float, float]:
        """放果点: 固定点位优先, 否则目标点 + place_offset_m。"""
        fixed = self._motion_cfg.get("place_position_mm")
        if fixed:
            return (
                (float(fixed[0]) / 1000.0, float(fixed[1]) / 1000.0, float(fixed[2]) / 1000.0),
                float(fixed[3]), float(fixed[4]), float(fixed[5]),
            )
        place = self._offset(target_base, self._motion_cfg.get("place_offset_m", [0.0, 0.0, 0.08]))
        return place, rx, ry, rz

    def _return_home_and_wait(self) -> str:
        """话题模式流程收尾: 发布回初始位置并等待机械臂到位。

        DONE 只有在机械臂回到 home (距 home ≤ auto_pick.home_radius_m)
        之后才由调用方上报, 避免 PLC 在臂未复位时就发下一个 START。
        超时 (motion.home_wait_timeout_s) 不阻断上报, 仅记日志。

        Returns: "done" (已到位 / 无需等待 / 超时容忍) | "aborted"
        """
        home_xyz = self._motion_cfg.get("home_pose_xyz_m")
        home_quat = self._motion_cfg.get("home_quat_xyzw")
        if not (home_xyz and home_quat) or self._publish_home_pose is None:
            return "done"
        self._publish_home_pose(home_xyz, home_quat)
        self._log("[PICK] 已发布回初始位置 /target_pose (home)")
        if self._robot is None:
            return "done"

        timeout_s = float(self._motion_cfg.get("home_wait_timeout_s", 20.0))
        radius = float(
            self._cfg.get("auto_pick", {}).get("home_radius_m", 0.05)
        )
        home = np.asarray([float(v) for v in home_xyz])
        poll = float(self._motion_cfg.get("poll_interval_s", 0.1))
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._abort_event.wait(poll):
                return "aborted"
            T = self._robot.get_T_base_tool(max_age_s=2.0)
            if T is not None and float(np.linalg.norm(T[:3, 3] - home)) <= radius:
                self._log("[PICK] 机械臂已回到初始位置 → 上报 DONE")
                return "done"
        self._log(f"[PICK] 等待回初始位置超时 ({timeout_s:.0f}s), 照常上报")
        return "done"

    def _wait_arrival(self, target_base, rpy) -> str:
        """轮询机械臂位姿直到法兰到达目标 Link6 位姿（剪切点）。

        判据: ToolVectorActual 报的是 Link6 法兰位姿 (未含剪刀手工具长度),
        而 target_base 是夹爪 TCP 剪切点。因此与发布端 publish_target_pose
        同公式反推 Link6 目标位姿: P_link6 = P_target - R_target @ offset,
        再比较法兰实测位置距 P_link6 ≤ scissors.arrival_tolerance_m 视为
        到位。位姿停更 (超过 max_age 2s) 视为未到位, 防止陈旧位姿误判。

        Returns: "arrived" | "timeout" | "aborted"
        """
        if self._robot is None:
            return "arrived"  # 无位姿源 (Fake/测试) 时放行, 维持旧开环行为
        timeout_s = float(self._scissors_cfg.get("arrival_timeout_s", 10.0))
        tol_m = float(self._scissors_cfg.get("arrival_tolerance_m", 0.02))
        poll = float(self._motion_cfg.get("poll_interval_s", 0.1))
        target = np.asarray([float(v) for v in target_base])
        R_target = rpy_degrees_to_rotation_matrix(rpy)
        link6_target = target - R_target @ self._gripper_offset_m
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._abort_event.wait(poll):
                return "aborted"
            T = self._robot.get_T_base_tool(max_age_s=2.0)
            if T is not None and float(np.linalg.norm(T[:3, 3] - link6_target)) <= tol_m:
                return "arrived"
        return "timeout"

    # ── send_via_topic 模式时序: 发布位姿 → 到位判定合剪 → 固定点开剪 ──
    def _topic_pick_sequence(self, target_base, rpy) -> bool:
        """话题模式采摘时序 (机械臂由 MoveIt 侧 grape_arm_control 执行)。

           发布目标位姿 → 轮询 TCP 到位 (arrival_timeout_s) → 合剪
           → [place_position_mm 已示教] 发布放果点 → 延时 open_delay_s → 开剪

        合剪改为闭环到位判定: 位姿未到达剪切点 (IK 失败/执行失败/位姿停更)
        时跳过合剪并上报 FAIL_ROBOT_MOTION, 避免空中误剪。
        与 MovL 路径的差异: 本侧无后续运动可编排, 合剪失败即时报 FAIL_CUT
        (MovL 路径则继续走放果/退回, 末尾统一上报)。

        Returns:
            True  = 流程正常走完 (调用方上报 DONE)
            False = 已自行上报失败/中止, 调用方直接返回
        """
        approach_base = list(target_base)
        approach_base[1] += 0.05
        self._publish_target_pose(approach_base, rpy)
        approach_arrival = self._wait_arrival(approach_base, rpy)
        if approach_arrival == "aborted":
            self._finish_aborted()
            return False
        if approach_arrival != "arrived":
            self._log(f"[PICK] 机械臂未到达 approach 位置 ({approach_arrival}), 跳过合剪")
            self._fail(
                plc.STATE_FAIL_ROBOT_MOTION,
                "机械臂未到达 approach 位置, 已跳过合剪",
            )
            return False
        self._log("[PICK] 机械臂已到达 approach 位置")

    
        self._publish_target_pose(target_base, rpy)
        self._log("[PICK] 已发布 /target_pose, 机械臂由 MoveIt 侧执行")

        if not self._scissors_usable():
            self._log("[PICK] 剪刀不可用, 跳过剪枝段")
            if self._return_home_and_wait() == "aborted":
                self._finish_aborted()
                return False
            return True

        # ── 合剪前到位判定: 轮询法兰是否到达剪切点对应 Link6 位姿 ──
        # 替代原 close_delay_s 开环延时: IK 失败/执行失败时机械臂不会到位,
        # 盲等固定时长会在空中合剪。
        arrival = self._wait_arrival(target_base, rpy)
        if arrival == "aborted":
            self._finish_aborted()
            return False
        if arrival != "arrived":
            self._log(f"[PICK] 机械臂未到达剪切点 ({arrival}), 跳过合剪")
            self._fail(
                plc.STATE_FAIL_ROBOT_MOTION,
                "机械臂未到达剪切点, 已跳过合剪",
            )
            return False
        self._log("[PICK] 机械臂已到达剪切点, 开始合剪")

        # 剪刀状态判断: 读 0x02 确认空闲再合剪 (busy/无响应 → FAIL_CUT)
        if not self._pre_cut_idle_check():
            return False

        # 合剪指令已下发 → 无论成败都按"可能已夹持"处理, 便于保安全开剪
        self._scissors_closed = True
        result = self._robot.scissors_cut(
            float(self._scissors_cfg.get("cut_timeout_s", 15.0)),
            self._abort_event,
        )
        if result == "aborted":
            # 【高-2】合剪途中 ABORT: 尽力开剪保安全后再上报
            self._safety_open()
            self._finish_aborted()
            return False
        if result != "done":
            if result == "no_motion":
                self._log("[PICK] 剪刀无动作 (0x02 未观察到运动), "
                          "合剪指令可能未生效")
            else:
                self._log(f"[PICK] 合剪失败: {result}, 尽力开剪保安全")
            self._safety_open()
            self._fail(plc.STATE_FAIL_CUT, "合剪失败, 需人工确认剪刀与果藤状态")
            return False
        self._log("[PICK] 合剪完成 (夹持中)")

        # ── 固定放果点 → 开剪 ──
        # ⚠ 固定位置待现场示教: 在 configs/fusion_pipeline.yaml 填
        #   robot.motion.place_position_mm = [x, y, z, rx, ry, rz]
        #   (mm / 度, base 系, 夹爪工具中心 TCP 位姿, 与 target_base 同一约定)。
        #   填好后本段自动生效; 留空 (null) 则跳过, 剪刀保持合剪到下一轮或 ABORT。
        fixed = self._motion_cfg.get("place_position_mm")
        if not fixed:
            self._log("[PICK] 未配置固定放果点 place_position_mm, 跳过开剪 (剪刀保持合剪)")
            if self._return_home_and_wait() == "aborted":
                self._finish_aborted()
                return False
            return True

        fixed = [float(v) for v in fixed]
        place = (fixed[0] / 1000.0, fixed[1] / 1000.0, fixed[2] / 1000.0)
        self._publish_target_pose(place, (fixed[3], fixed[4], fixed[5]))
        self._log(
            f"[PICK] 已发布放果点 /target_pose: "
            f"({fixed[0]:.1f},{fixed[1]:.1f},{fixed[2]:.1f})mm "
            f"rpy=({fixed[3]:.1f},{fixed[4]:.1f},{fixed[5]:.1f})"
        )

        arrive = float(self._scissors_cfg.get("open_delay_s", 2.5))
        if arrive > 0.0 and self._abort_event.wait(arrive):
            self._finish_aborted()
            return False

        result = self._robot.scissors_open_wait(
            float(self._scissors_cfg.get("open_timeout_s", 5.0)),
            self._abort_event,
        )
        if result == "aborted":
            self._finish_aborted()
            return False
        if result != "done":
            if result == "no_motion":
                self._log("[PICK] 开剪无动作 (0x02 未观察到运动), 指令可能未生效")
            # 开剪失败不挡住回家: home 是安全位, 先回 home 再报 FAIL_CUT
            self._log(f"[PICK] 开剪失败: {result} — 先回 home 再报 FAIL_CUT")
            if self._return_home_and_wait() == "aborted":
                self._finish_aborted()
                return False
            self._fail(plc.STATE_FAIL_CUT, f"开剪失败: {result}, 剪刀状态未知")
            return False
        self._scissors_closed = False
        self._log("[PICK] 开剪完成 (放果)")
        if self._return_home_and_wait() == "aborted":
            self._finish_aborted()
            return False
        return True

    def _do_motion(self, label, xyz, rx, ry, rz, kind: str) -> str:
        """单段运动: MovL + 完成等待。

        Returns: "done" | "aborted" | "send_fail" | "motion_fail" | "service_lost"
        """
        speed = float(self._motion_cfg.get(
            "speed_grasp_mm_s" if kind == "grasp" else "speed_approach_mm_s", 200
        ))
        x_mm, y_mm, z_mm = xyz[0] * 1000.0, xyz[1] * 1000.0, xyz[2] * 1000.0
        try:
            ok = self._robot.movl_pose(x_mm, y_mm, z_mm, rx, ry, rz, speed_mm_s=speed)
        except Exception as e:  # ServiceTimeoutError 等: 机器人无响应
            self._log(f"[PICK] {label} MovL 调用异常: {e!r}")
            self._robot.emergency_brake()
            return "service_lost"
        if not ok:
            self._log(f"[PICK] {label} MovL 被机器人拒绝 (res!=0)")
            return "send_fail"

        result = self._robot.wait_motion_done(
            float(self._motion_cfg.get("timeout_s", 60.0)), self._abort_event
        )
        if result == "aborted":
            return "aborted"
        if result == "done":
            self._log(f"[PICK] {label} 完成")
            return "done"
        # timeout / robot_error / service_lost
        self._robot.emergency_brake()
        return result

    def _motion_fail(self, result: str) -> None:
        """运动失败归类上报。"""
        if result == "service_lost" and self._robot.pose_stale():
            self._fail(plc.STATE_FAIL_NO_ROBOT_POSE, "机器人位姿停更（掉线）")
        elif result == "send_fail":
            self._fail(plc.STATE_FAIL_GRASP_SEND, "抓取命令发送失败")
        else:
            self._fail(plc.STATE_FAIL_ROBOT_MOTION, f"机械臂运动异常 ({result})")

    def _scissors_usable(self) -> bool:
        return (
            self._scissors_cfg.get("enable", True)
            and self._robot is not None
            and getattr(self._robot, "scissors_ready", False)
        )

    def _pre_cut_idle_check(self) -> bool:
        """合剪前剪刀状态判断: 读 0x02 确认空闲再下发合剪。

        - 0x02 == 0 (停止): 允许合剪
        - 0x02 == 1 (运动中): 有界等待其停止 (scissors.pre_cut_idle_timeout_s,
          默认 2s), 超时 → FAIL_CUT
        - 0x02 == None (无响应): 状态未知 → FAIL_CUT, 不盲目合剪
        - ABORT: 立即中止
        - 机器人无状态查询接口 (注入式 Fake): 跳过判断, 放行

        Returns:
            True  = 剪刀空闲, 可以合剪
            False = 已自行上报失败/中止, 调用方直接 return
        """
        busy_fn = getattr(self._robot, "scissors_busy", None)
        if busy_fn is None:
            return True
        deadline = time.time() + float(
            self._scissors_cfg.get("pre_cut_idle_timeout_s", 2.0)
        )
        interval = float(self._scissors_cfg.get("poll_interval_s", 0.05))
        while True:
            if self._abort_event.wait(interval):
                self._finish_aborted()
                return False
            try:
                busy = busy_fn()
            except Exception as e:
                self._log(f"[PICK] 读剪刀状态异常: {e!r}")
                busy = None
            if busy == 0:
                return True
            if busy is None:
                self._log("[PICK] 剪刀状态读取失败 (0x02 无响应), 状态未知")
                self._fail(plc.STATE_FAIL_CUT, "剪刀状态读取失败 (0x02 无响应)")
                return False
            if time.time() >= deadline:
                self._log("[PICK] 剪刀持续运动中, 合剪前等待超时")
                self._fail(plc.STATE_FAIL_CUT, "剪刀持续运动中, 合剪前等待超时")
                return False
