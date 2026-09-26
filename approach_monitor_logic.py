"""接近目标监控与夹爪触发纯逻辑 (无 ROS 依赖, 便于单元测试)。

状态机:
    IDLE → (收到新 target) → TARGET_RECEIVED → PLANNING → MOVING
    MOVING: 实时距离 <= close_distance_threshold → APPROACHING
    APPROACHING: 连续 approach_confirm_count 次满足 → GRIPPING → close_gripper()
    → COMPLETED; 任何时刻 STOP/ESTOP/机器人 ERROR → ERROR

设计约束:
- 不阻塞回调: 本模块只做纯状态推进, 由外部 timer 每周期调用 tick()
- 一次性触发: gripper_closed_for_current_target, 同一 target 只关一次
- 目标切换: 新 target_id 重置 confirm 计数与已关标志, 防止串间状态污染
- tick() 只返回动作 ("close_gripper" 等), 不执行夹爪;
  调用方 (节点) 收到动作后在独立线程执行真实夹爪并回调 on_gripper_closed(),
  避免长时间 Modbus 等待阻塞 timer 回调
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

# ── 状态 ──────────────────────────────────────────
S_IDLE = "IDLE"
S_TARGET_RECEIVED = "TARGET_RECEIVED"
S_PLANNING = "PLANNING"
S_MOVING = "MOVING"
S_APPROACHING = "APPROACHING"
S_GRIPPING = "GRIPPING"
S_COMPLETED = "COMPLETED"
S_ERROR = "ERROR"

STATE_NAMES = {
    S_IDLE: "IDLE",
    S_TARGET_RECEIVED: "TARGET_RECEIVED",
    S_PLANNING: "PLANNING",
    S_MOVING: "MOVING",
    S_APPROACHING: "APPROACHING",
    S_GRIPPING: "GRIPPING",
    S_COMPLETED: "COMPLETED",
    S_ERROR: "ERROR",
}


@dataclass
class ApproachMonitorState:
    """纯状态容器, 供节点与测试共享。"""
    state: str = S_IDLE
    target_id: Optional[str] = None
    target_xyz: Optional[Tuple[float, float, float]] = None
    target_quat: Optional[Tuple[float, float, float, float]] = None
    current_xyz: Optional[Tuple[float, float, float]] = None
    last_distance_m: Optional[float] = None
    confirm_count: int = 0
    gripper_closed: bool = False
    last_reason: str = ""


@dataclass
class ApproachConfig:
    close_distance_threshold_m: float = 0.03   # 30 mm, 配置项, 现场标定
    approach_confirm_count: int = 3            # 连续 N 次满足才触发
    dry_run: bool = True                       # True 只打印不真关
    max_distance_m: Optional[float] = None     # 目标过远拒绝 (None 不检查)


def compute_distance(
    target_xyz: Tuple[float, float, float],
    current_xyz: Tuple[float, float, float],
) -> Optional[float]:
    """两点欧氏距离 (米); 输入含 NaN/Inf 返回 None。"""
    if target_xyz is None or current_xyz is None:
        return None
    t = np.asarray(target_xyz, dtype=np.float64)
    c = np.asarray(current_xyz, dtype=np.float64)
    if t.shape != (3,) or c.shape != (3,):
        return None
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(c)):
        return None
    return float(np.linalg.norm(t - c))


class ApproachMonitor:
    """接近监控状态机 (纯逻辑)。"""

    def __init__(
        self,
        config: ApproachConfig,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self._cfg = config
        self._log = logger or (lambda msg: None)
        self._st = ApproachMonitorState()

    @property
    def state(self) -> str:
        return self._st.state

    def new_target(
        self,
        target_id: str,
        target_xyz: Tuple[float, float, float],
        target_quat: Tuple[float, float, float, float],
    ) -> bool:
        """收到新目标: 重置串内状态, 进入 TARGET_RECEIVED。

        Returns:
            False 表示目标非法 (XYZ 非有限 / 四元数模长异常), 已忽略。
        """
        xyz = np.asarray(target_xyz, dtype=np.float64)
        quat = np.asarray(target_quat, dtype=np.float64)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            self._st.last_reason = "target XYZ 非法"
            return False
        if quat.shape != (4,) or not np.all(np.isfinite(quat)):
            self._st.last_reason = "target quaternion 非法"
            return False
        if abs(float(np.linalg.norm(quat)) - 1.0) > 1e-3:
            self._st.last_reason = f"quaternion 模长 {np.linalg.norm(quat):.4f} ≠ 1"
            return False
        if (self._cfg.max_distance_m is not None
                and float(np.linalg.norm(xyz)) > self._cfg.max_distance_m):
            self._st.last_reason = f"目标超出最大距离 {self._cfg.max_distance_m}m"
            return False

        # 目标切换 (含同一 id 重发): 重置一次性触发状态
        self._st.target_id = target_id
        self._st.target_xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        self._st.target_quat = (
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        self._st.confirm_count = 0
        self._st.gripper_closed = False
        self._st.state = S_TARGET_RECEIVED
        self._log("[APPROACH] 新目标已接收, 等待规划执行")
        return True

    def notify_planning(self):
        if self._st.state in (S_TARGET_RECEIVED, S_IDLE):
            self._st.state = S_PLANNING

    def notify_moving(self):
        if self._st.state in (S_PLANNING, S_TARGET_RECEIVED):
            self._st.state = S_MOVING

    def notify_robot_error(self, reason: str = "robot error"):
        self._st.state = S_ERROR
        self._st.last_reason = reason

    def update_current_xyz(self, current_xyz: Tuple[float, float, float]):
        self._st.current_xyz = current_xyz

    def tick(self) -> str:
        """每周期调用一次 (timer 驱动)。推进状态并返回 action 字符串。

        Returns:
            "close_gripper" — 本周期应执行关夹爪 (一次性)
            "target_reached" — 距离已进入阈值 (确认计数中)
            "moving" / "idle" / "error" / "completed" / "" — 常规推进
        """
        st = self._st
        if st.state in (S_IDLE, S_TARGET_RECEIVED, S_PLANNING):
            return ""

        if st.state in (S_ERROR, S_COMPLETED):
            return st.state.lower()

        # 计算距离 (米)
        dist = compute_distance(st.target_xyz, st.current_xyz)
        st.last_distance_m = dist
        if dist is None:
            # 位姿无效: 不推进不触发, 保持安全
            return ""

        if st.state == S_MOVING:
            if dist <= self._cfg.close_distance_threshold_m:
                st.state = S_APPROACHING
                st.confirm_count = 1
                return "target_reached"
            return "moving"

        if st.state == S_APPROACHING:
            if dist <= self._cfg.close_distance_threshold_m:
                st.confirm_count += 1
                if st.confirm_count >= self._cfg.approach_confirm_count \
                        and not st.gripper_closed:
                    st.gripper_closed = True
                    st.state = S_GRIPPING
                    return "close_gripper"
                return "target_reached"
            # 退出阈值区: 重新计数 (防运动过程偶然扫过阈值)
            st.confirm_count = 0
            st.state = S_MOVING
            return "moving"

        return st.state

    def on_gripper_closed(self, ok: bool):
        """关夹爪动作完成回调 (外部注入的 closer 返回后调用)。"""
        if ok:
            self._st.state = S_COMPLETED
        else:
            self._st.state = S_ERROR
            self._st.last_reason = "gripper close failed"

    def snapshot(self) -> ApproachMonitorState:
        return ApproachMonitorState(
            state=self._st.state,
            target_id=self._st.target_id,
            target_xyz=self._st.target_xyz,
            target_quat=self._st.target_quat,
            current_xyz=self._st.current_xyz,
            last_distance_m=self._st.last_distance_m,
            confirm_count=self._st.confirm_count,
            gripper_closed=self._st.gripper_closed,
            last_reason=self._st.last_reason,
        )
