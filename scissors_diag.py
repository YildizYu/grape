#!/usr/bin/env python3
"""剪刀手 Modbus 链诊断脚本 — 逐步执行初始化/写/读, 打印每一步 res。

用途: 定位 SetSingleHoldReg 失败 (res=-10000) 的原因。

运行前提: bringup 在跑、机械臂已使能且无报警 (ERROR 态会拒绝立即指令)。
    python3 scripts/scissors_diag.py --config configs/fusion_pipeline.yaml

注意: 本脚本会真实发合剪/开剪指令! 确保剪刀手远离人手/果藤。
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

try:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from dobot_msgs_v4.srv import (
        GetHoldRegs,
        ModbusRTUCreate,
        SetSingleHoldReg,
        SetTool485,
        SetToolMode,
        SetToolPower,
        RobotMode,
    )
except ImportError as e:
    sys.exit(f"缺少 ROS2 依赖: {e}\n请先: source install/setup.bash")


def main():
    parser = argparse.ArgumentParser(description="Scissors Modbus diagnostic")
    parser.add_argument("--config", type=str, default="configs/fusion_pipeline.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path
    import yaml
    config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    # 正式配置位于 robot.scissors；顶层键仅兼容旧现场配置。
    sc = config.get("robot", {}).get("scissors", config.get("scissors", {}))

    rclpy.init(args=None)
    node = Node("scissors_diag")

    def call(srv_type, srv_name, req, timeout_s=5.0):
        client = node.create_client(srv_type, f"/dobot_bringup_ros2/srv/{srv_name}")
        if not client.wait_for_service(timeout_sec=5.0):
            print(f"  ✗ {srv_name}: 服务不可用")
            return None
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout_s)
        if not fut.done():
            print(f"  ✗ {srv_name}: 调用超时")
            return None
        resp = fut.result()
        if resp is None:
            print(f"  ✗ {srv_name}: 无结果")
            return None
        print(f"  → {srv_name}: res={resp.res} return={resp.robot_return!r}")
        return resp

    def q(srv_type, srv_name, **fields):
        req = srv_type.Request()
        for k, v in fields.items():
            setattr(req, k, v)
        return call(srv_type, srv_name, req)

    print("== 0. 机器人状态 ==")
    q(RobotMode, "RobotMode")

    print("== 1. 末端供电 / 复用模式 / 485 格式 ==")
    q(SetToolPower, "SetToolPower", status=1)
    q(SetToolMode, "SetToolMode", mode=1, type=0)
    q(SetTool485, "SetTool485", baudrate=int(sc.get("baud", 9600)),
      parity="N", stop=1, identify=1)

    print("== 2. 创建 Modbus 主站 (slave_id, baud) ==")
    r = q(ModbusRTUCreate, "ModbusRTUCreate",
          slave_id=int(sc.get("slave_id", 1)), baud=int(sc.get("baud", 9600)),
          parity='"N"', data_bit=8, stop_bit=1)
    index = None
    if r is not None and r.res == 0:
        s = r.robot_return.strip().strip("{}")
        index = int(s) if s.isdigit() else 0
        print(f"  → 主站 index = {index}")
    else:
        print("  ✗ 主站创建失败, 终止")
        return 1

    print("== 3. 写速度 0x04 / 行程 0x05 ==")
    q(SetSingleHoldReg, "SetSingleHoldReg", index=index,
      addr=int(sc.get("reg_close_speed", 4)), val=int(sc.get("close_speed_rpm", 300)))
    q(SetSingleHoldReg, "SetSingleHoldReg", index=index,
      addr=int(sc.get("reg_stroke", 5)), val=int(sc.get("stroke_pulses", 9000)))

    print("== 4. 读运动状态 0x02 (FC03 读回验证) ==")
    q(GetHoldRegs, "GetHoldRegs", index=index,
      addr=int(sc.get("reg_motion", 2)), count=1)

    print("== 5. 合剪 0x01←4 (确认剪刀安全后再执行) ==")
    q(SetSingleHoldReg, "SetSingleHoldReg", index=index,
      addr=int(sc.get("reg_close", 1)), val=int(sc.get("close_val", 4)))

    print("== 6. 开剪 0x0A←1 ==")
    q(SetSingleHoldReg, "SetSingleHoldReg", index=index,
      addr=int(sc.get("reg_open", 10)), val=1)

    print("\n解读:")
    print("  - 第 1 步任意 res≠0 → 末端供电/485 链路问题 (航插/接线)")
    print("  - 第 2 步 res≠0 → Modbus 主站创建失败 (固件/端口占用)")
    print("  - 第 3~6 步 res=-10000 → 从站无响应: 查剪刀手供电/从站地址/波特率")
    print("  - 第 4 步能读到寄存器值 → 通讯正常, 问题只在写")
    return 0


if __name__ == "__main__":
    sys.exit(main())
