"""ZED X Mini 相机适配器（由 RealSense D435I 版迁移而来）。

封装 pyzed (ZED SDK 5.x)，提供与旧版完全一致的相机接口：
    camera = CameraAdapter(width=640, height=480, fps=30)
    intrinsics = camera.start()
    frame = camera.read()   # RGBDFrame (color_bgr, depth_data float32 米, ...)

设计要点:
- pyzed 在 start() 时懒导入，因此本模块可在无相机环境中正常 import。
- ZED 深度图原生与左目图像对齐，无需 rs.align 之类的对齐步骤。
- ZED retrieve_measure(MEASURE.DEPTH) 输出 float32 米，depth_scale 恒为 1.0。
- ZED 原生分辨率无 640×480：以 zed_resolution (默认 HD1200) 采集，
  中心裁剪为输出宽高比后缩放至 (width, height)，内参按同比例换算
  (见 _compute_crop_resize / _scale_intrinsics)。

硬件说明:
- ZED X Mini 通过 GMSL2 (FAKRA Z) 连接 ZED Box Mini 采集卡，
  系统需已安装 stereolabs-zedbox-mini DTB 包 (本机已装, 诊断通过)。
- 4mm 镜头最小深度 0.15m / 2.2mm 镜头 0.1m，最大深度 12m / 8m。
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import time

import numpy as np

from .types import CameraIntrinsics, RGBDFrame
from .coordinate_3d import intrinsics_from_zed
from .depth_sampler import sample_depth, DepthSample

# ZED X Mini 支持的原生分辨率（pyzed 5.2.3 实测）:
# 不支持 HD720 / VGA / SVGA（SDK 会返回 INVALID RESOLUTION）
_ZED_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "HD1200": (1920, 1200),
    "HD1080": (1920, 1080),
    "HD2K": (2208, 1242),
}

_DEFAULT_ZED_RESOLUTION = "HD1200"
_DEFAULT_ZED_DEPTH_MODE = "NEURAL"


def _compute_crop_resize(
    native_w: int,
    native_h: int,
    out_w: int,
    out_h: int,
) -> Tuple[Tuple[int, int, int, int], Tuple[float, float]]:
    """计算 中心裁剪(4:3) + 缩放 的几何参数。

    Args:
        native_w, native_h: ZED 原生采集分辨率
        out_w, out_h: 期望输出分辨率

    Returns:
        crop: (x0, y0, crop_w, crop_h) — 原生图上的裁剪窗口
        scale: (sx, sy) — crop → output 的缩放系数
    """
    target_aspect = out_w / out_h
    native_aspect = native_w / native_h

    if native_aspect >= target_aspect:
        # 原生更宽 → 裁左右
        crop_h = native_h
        crop_w = int(round(native_h * target_aspect))
        x0 = (native_w - crop_w) // 2
        y0 = 0
    else:
        # 原生更高 → 裁上下
        crop_w = native_w
        crop_h = int(round(native_w / target_aspect))
        x0 = 0
        y0 = (native_h - crop_h) // 2

    sx = out_w / crop_w
    sy = out_h / crop_h

    return (x0, y0, crop_w, crop_h), (sx, sy)


def _scale_intrinsics(
    k: CameraIntrinsics,
    crop: Tuple[int, int, int, int],
    scale: Tuple[float, float],
    out_w: int,
    out_h: int,
) -> CameraIntrinsics:
    """将原生内参换算为 裁剪+缩放 后的输出图像内参。"""
    x0, y0, _, _ = crop
    sx, sy = scale
    return CameraIntrinsics(
        fx=k.fx * sx,
        fy=k.fy * sy,
        cx=(k.cx - x0) * sx,
        cy=(k.cy - y0) * sy,
        width=out_w,
        height=out_h,
    )


def _as_numpy(pyzed_array: np.ndarray, dtype) -> np.ndarray:
    """将 pyzed get_data() 返回的数组重建为运行时 dtype 的干净 numpy 数组。

    pyzed 5.2 (aarch64) 按 numpy 1.20 ABI 编译, 其数组的 dtype 描述符
    与 numpy 1.26 不兼容 (view/打印会触发 TypeError / "not a numpy array")。
    用 frombuffer 基于原始内存重建即可彻底绕过, 且不产生拷贝。

    对普通 numpy 数组同样适用 (a.data 是其缓冲区 memoryview)。
    """
    a = pyzed_array
    itemsize = np.dtype(dtype).itemsize
    # frombuffer 的 count 是元素个数而非字节数
    return np.frombuffer(
        a.data, dtype=dtype, count=a.nbytes // itemsize
    ).reshape(a.shape)


def _crop_resize_color(bgra: np.ndarray, crop: Tuple[int, int, int, int],
                       out_w: int, out_h: int) -> np.ndarray:
    """ZED BGRA (H,W,4) → 裁剪 → 缩放 → BGR (H,W,3)。"""
    import cv2

    bgra = _as_numpy(bgra, np.uint8)
    x0, y0, cw, ch = crop
    bgr = bgra[y0:y0 + ch, x0:x0 + cw, :3]
    if (cw, ch) != (out_w, out_h):
        bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    else:
        bgr = bgr.copy()  # 与 pyzed 缓冲区脱钩, 防止下次 grab 覆盖
    return bgr


def _crop_resize_depth(depth: np.ndarray, crop: Tuple[int, int, int, int],
                       out_w: int, out_h: int) -> np.ndarray:
    """ZED float32 米深度图 → 裁剪 → 缩放 (最近邻, 深度值不变)。"""
    import cv2

    depth = _as_numpy(depth, np.float32)
    x0, y0, cw, ch = crop
    d = depth[y0:y0 + ch, x0:x0 + cw]
    if (cw, ch) != (out_w, out_h):
        d = cv2.resize(d, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    else:
        d = d.copy()  # 与 pyzed 缓冲区脱钩
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


class CameraAdapter:
    """ZED X Mini RGB-D 相机适配器（接口与 RealSense 版兼容）。

    使用示例:
        camera = CameraAdapter(width=640, height=480, fps=30)
        intrinsics = camera.start()
        try:
            while True:
                frame = camera.read()
                if frame is None:
                    continue
                # 处理 frame.color_bgr, frame.depth_data, ...
        finally:
            camera.stop()
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        align_to: str = "color",
        zed_resolution: str = _DEFAULT_ZED_RESOLUTION,
        zed_depth_mode: str = _DEFAULT_ZED_DEPTH_MODE,
        zed_depth_min_m: float = 0.15,
        zed_depth_max_m: float = 8.0,
    ):
        """
        Args:
            width: 输出彩色图宽度 (默认 640, 与 D435i 版一致)
            height: 输出彩色图高度 (默认 480)
            fps: 帧率
            align_to: 兼容参数 — ZED 深度原生与左图对齐, 此参数被忽略
            zed_resolution: ZED 原生采集分辨率 (HD1200/HD1080/HD2K)
            zed_depth_mode: ZED 深度模式 (NEURAL/NEURAL_PLUS/QUALITY/ULTRA/PERFORMANCE)
            zed_depth_min_m: ZED 最小深度 (4mm 镜头 0.15 / 2.2mm 镜头 0.1)
            zed_depth_max_m: ZED 最大深度 (4mm 镜头 12 / 2.2mm 镜头 8)
        """
        self._width = width
        self._height = height
        self._fps = fps
        self._align_to = align_to  # 保留以兼容旧接口, 实际不使用
        self._zed_resolution = zed_resolution
        self._zed_depth_mode = zed_depth_mode
        self._zed_depth_min_m = zed_depth_min_m
        self._zed_depth_max_m = zed_depth_max_m

        self._cam = None
        self._runtime = None
        self._depth_scale = 1.0  # ZED 深度为 float32 米, 无需缩放
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._crop = None
        self._scale = None
        self._frame_count = 0
        self._running = False

    @property
    def intrinsics(self) -> Optional[CameraIntrinsics]:
        """相机内参（start() 后可用，已换算到输出分辨率）。"""
        return self._intrinsics

    @property
    def depth_scale(self) -> float:
        """深度单位转换系数。ZED 输出 float32 米，恒为 1.0。"""
        return self._depth_scale

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> CameraIntrinsics:
        """启动 ZED X Mini 相机。

        懒导入 pyzed — 只在有相机的环境中才会触发。

        Returns:
            CameraIntrinsics: 相机内参（输出分辨率下）

        Raises:
            ImportError: pyzed 未安装
            RuntimeError: 相机连接失败
        """
        import pyzed.sl as sl

        if self._zed_resolution not in _ZED_RESOLUTIONS:
            raise ValueError(
                f"Unknown zed_resolution: {self._zed_resolution}. "
                f"Available: {list(_ZED_RESOLUTIONS)}"
            )
        native_w, native_h = _ZED_RESOLUTIONS[self._zed_resolution]

        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, self._zed_resolution)
        init.camera_fps = self._fps
        # pyzed 5.x: 设置 depth_mode 即启用深度 (无 enable_depth 属性)
        init.depth_mode = getattr(sl.DEPTH_MODE, self._zed_depth_mode)
        init.coordinate_units = sl.UNIT.METER
        init.depth_minimum_distance = self._zed_depth_min_m
        init.depth_maximum_distance = self._zed_depth_max_m

        cam = sl.Camera()
        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            cam.close()
            raise RuntimeError(
                f"ZED camera open failed: {status}. "
                f"请检查 ZED X Mini 与 ZED Box Mini (GMSL2) 连接, "
                f"或用 ZED_Diagnostic 工具诊断."
            )

        # 读取标定内参 (对应当前打开的分辨率)
        info = cam.get_camera_information()
        cal = info.camera_configuration.calibration_parameters.left_cam
        cal_intrinsics = intrinsics_from_zed(cal)

        # 防御: 若 SDK 返回的标定尺寸与打开分辨率不一致, 按比例换算
        cal_w, cal_h = cal_intrinsics.width, cal_intrinsics.height
        if (cal_w, cal_h) != (native_w, native_h):
            sx = native_w / cal_w
            sy = native_h / cal_h
            cal_intrinsics = CameraIntrinsics(
                fx=cal_intrinsics.fx * sx,
                fy=cal_intrinsics.fy * sy,
                cx=cal_intrinsics.cx * sx,
                cy=cal_intrinsics.cy * sy,
                width=native_w,
                height=native_h,
            )

        # 计算 裁剪 + 缩放 参数, 并换算输出内参
        self._crop, self._scale = _compute_crop_resize(
            native_w, native_h, self._width, self._height
        )
        self._intrinsics = _scale_intrinsics(
            cal_intrinsics, self._crop, self._scale, self._width, self._height
        )

        self._cam = cam
        self._runtime = sl.RuntimeParameters()
        self._running = True
        self._frame_count = 0

        print(f"Camera started: ZED X Mini @ {self._zed_resolution} "
              f"{native_w}x{native_h} {self._fps}fps "
              f"-> output {self._width}x{self._height}")
        print(f"Depth mode: {self._zed_depth_mode}, "
              f"range {self._zed_depth_min_m}-{self._zed_depth_max_m} m")
        print(f"Intrinsics: fx={self._intrinsics.fx:.2f}, fy={self._intrinsics.fy:.2f}, "
              f"cx={self._intrinsics.cx:.2f}, cy={self._intrinsics.cy:.2f} "
              f"(output {self._intrinsics.width}x{self._intrinsics.height})")
        print(f"Depth scale: {self._depth_scale} (float32 m, 无需缩放)")
        print("Alignment: ZED 深度原生与左图对齐 (无需 align)")

        return self._intrinsics

    def read(self) -> Optional[RGBDFrame]:
        """读取一帧同步的 RGB-D 数据。

        Returns:
            RGBDFrame: 同步帧（color_bgr + float32 米深度），失败返回 None
        """
        if not self._running:
            return None

        try:
            import pyzed.sl as sl

            if self._cam.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
                return None

            img = sl.Mat()
            measure = sl.Mat()
            self._cam.retrieve_image(img, sl.VIEW.LEFT)
            self._cam.retrieve_measure(measure, sl.MEASURE.DEPTH)

            bgra = img.get_data()       # (H, W, 4) uint8 BGRA
            depth_native = measure.get_data()  # (H, W) float32 米

            color_bgr = _crop_resize_color(
                bgra, self._crop, self._width, self._height
            )
            depth_data = _crop_resize_depth(
                depth_native, self._crop, self._width, self._height
            )

            self._frame_count += 1

            return RGBDFrame(
                color_bgr=color_bgr,
                depth_data=depth_data,
                depth_frame=None,   # ZED 版不使用 SDK 帧对象
                color_frame=None,
                intrinsics=self._intrinsics,
                depth_scale=self._depth_scale,
                timestamp=time.time(),
                frame_id=self._frame_count,
            )

        except Exception as e:
            print(f"Frame read error: {e}")
            return None

    def pixel_to_3d(
        self,
        u: int,
        v: int,
        depth_frame=None,
        depth_data: Optional[np.ndarray] = None,
    ) -> Optional[tuple]:
        """像素坐标 → 相机三维坐标。

        公式与旧 RealSense 版一致 (针孔模型反投影):
          X = (u - cx) * Z / fx
          Y = (v - cy) * Z / fy
          Z = depth (米)

        Args:
            u: 像素列坐标
            v: 像素行坐标
            depth_frame: 兼容参数（ZED 版忽略）
            depth_data: float32 米深度数组（与 read() 输出同尺寸）

        Returns:
            (X, Y, Z) 米，或 None
        """
        if depth_data is None or self._intrinsics is None:
            return None

        h, w = depth_data.shape[:2]
        if not (0 <= v < h and 0 <= u < w):
            return None

        depth_m = float(depth_data[v, u])
        if depth_m <= 0 or not np.isfinite(depth_m):
            return None

        X = (u - self._intrinsics.cx) * depth_m / self._intrinsics.fx
        Y = (v - self._intrinsics.cy) * depth_m / self._intrinsics.fy
        Z = depth_m

        return (X, Y, Z)

    def sample_depth_robust(
        self,
        depth_data: np.ndarray,
        u: float,
        v: float,
        mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> DepthSample:
        """鲁棒深度采样（使用 depth_sampler 模块）。

        ZED 深度为 float32 米, sample_depth() 对非 uint16 输入直接按米处理。
        """
        return sample_depth(
            depth_data=depth_data,
            u=u,
            v=v,
            depth_scale=self._depth_scale,
            mask=mask,
            **kwargs,
        )

    def stop(self) -> None:
        """关闭相机，释放资源。"""
        self._running = False
        if self._cam is not None:
            try:
                self._cam.close()
                print("Camera stopped.")
            except Exception as e:
                print(f"Camera stop error: {e}")
        self._cam = None
        self._runtime = None
