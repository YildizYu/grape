#!/usr/bin/env python3
"""ZED X Mini 相机快速检测脚本。

用法:
    python scripts/check_zed_camera.py [--open]

无参数: 枚举设备, 显示链路状态 (AVAILABLE / NOT AVAILABLE)。
--open: 进一步尝试打开相机 (HD1200@30fps, NEURAL 深度), 验证视频流。
"""

import argparse
import sys


def enumerate_devices():
    import pyzed.sl as sl

    devs = sl.Camera.get_device_list()
    if not devs:
        print("✗ 未枚举到任何 ZED 设备 — 检查 FAKRA 线缆与载板供电")
        return False

    ok = True
    for d in devs:
        print(f"  发现设备: {d}")
        if "NOT AVAILABLE" in str(d):
            print("  ✗ 状态: NOT AVAILABLE — GMSL 视频链路未锁定, 请重新插拔线缆并重启")
            ok = False
        elif "AVAILABLE" in str(d):
            print("  ✓ 状态: AVAILABLE — 链路正常")
    return ok


def try_open():
    import pyzed.sl as sl

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1200
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.NEURAL

    cam = sl.Camera()
    st = cam.open(init)
    if st != sl.ERROR_CODE.SUCCESS:
        print(f"✗ 打开失败: {st}")
        print("  提示: 重新插拔 FAKRA 线缆两端 → 断电重启 Jetson → 再试")
        return False

    print("✓ 相机打开成功, 视频流正常")
    info = cam.get_camera_information()
    cal = info.camera_configuration.calibration_parameters.left_cam
    print(f"  型号: {info.camera_model} | 序列号: {info.serial_number}")
    print(f"  分辨率: {info.camera_configuration.resolution} @ {info.camera_configuration.fps}fps")
    print(f"  内参: fx={cal.fx:.2f} fy={cal.fy:.2f} cx={cal.cx:.2f} cy={cal.cy:.2f}")
    cam.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="ZED X Mini 相机检测")
    parser.add_argument("--open", action="store_true", help="尝试打开相机验证视频流")
    args = parser.parse_args()

    print("== ZED 相机检测 ==")
    ok = enumerate_devices()
    if args.open and ok:
        ok = try_open()
    if not args.open:
        print("(可加 --open 进一步验证视频流)")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
