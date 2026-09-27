"""Orbbec Gemini 336 RGB-D camera adapter based on Orbbec SDK v2."""

from typing import Optional, Sequence
import time

import numpy as np

from .depth_sampler import DepthSample, sample_depth
from .types import CameraIntrinsics, RGBDFrame


class CameraAdapter:
    """Expose synchronized, depth-to-color aligned Orbbec frames to the pipeline."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        align_to: str = "color",
        depth_width: Optional[int] = None,
        depth_height: Optional[int] = None,
        depth_fps: Optional[int] = None,
        alignment_mode: str = "hardware",
        serial_number: Optional[str] = None,
    ):
        if align_to != "color":
            raise ValueError(
                "Gemini 336 adapter only supports align_to='color'; the detection "
                "pipeline requires depth pixels in the color image coordinate system"
            )
        if alignment_mode not in {"hardware", "software"}:
            raise ValueError("alignment_mode must be 'hardware' or 'software'")

        self._width = width
        self._height = height
        self._fps = fps
        self._depth_width = depth_width or width
        self._depth_height = depth_height or height
        self._depth_fps = depth_fps or fps
        self._align_to = align_to
        self._alignment_mode = alignment_mode
        self._serial_number = serial_number

        self._ob = None
        self._pipeline = None
        self._align_filter = None
        self._depth_scale = 0.001
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._frame_count = 0
        self._running = False
        self._device_model = "Orbbec Gemini 336"
        self._device_serial = None
        self._active_alignment_mode = alignment_mode

    @property
    def intrinsics(self) -> Optional[CameraIntrinsics]:
        return self._intrinsics

    @property
    def depth_scale(self) -> float:
        """Raw uint16 depth count to metres conversion factor."""
        return self._depth_scale

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def device_model(self) -> str:
        return self._device_model

    @property
    def device_serial(self) -> Optional[str]:
        return self._device_serial

    @property
    def active_alignment_mode(self) -> str:
        return self._active_alignment_mode

    def start(self) -> CameraIntrinsics:
        """Open Gemini 336 and start synchronized depth-to-color RGB-D streams."""
        try:
            import pyorbbecsdk as ob
        except ImportError as exc:
            raise ImportError(
                "Orbbec SDK v2 is not installed. Install requirements-thor.txt "
                "inside the Thor virtual environment."
            ) from exc

        self._ob = ob
        self._pipeline = self._create_pipeline(ob)
        config = ob.Config()

        color_profile = self._select_profile(
            self._pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR),
            self._width,
            self._height,
            self._fps,
            (ob.OBFormat.RGB, ob.OBFormat.BGR, ob.OBFormat.MJPG,
             ob.OBFormat.YUYV, ob.OBFormat.UYVY),
            "color",
        )

        if self._alignment_mode == "hardware":
            depth_profile = self._select_hardware_d2c_profile(color_profile)
            config.set_align_mode(ob.OBAlignMode.HW_MODE)
            self._active_alignment_mode = "hardware"
        else:
            depth_profile = self._select_profile(
                self._pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR),
                self._depth_width,
                self._depth_height,
                self._depth_fps,
                (ob.OBFormat.Y16, ob.OBFormat.Z16),
                "depth",
            )
            config.set_align_mode(ob.OBAlignMode.DISABLE)
            self._align_filter = ob.AlignFilter(
                align_to_stream=ob.OBStreamType.COLOR_STREAM
            )
            self._active_alignment_mode = "software"

        config.enable_stream(depth_profile)
        config.enable_stream(color_profile)
        if hasattr(config, "set_frame_aggregate_output_mode"):
            config.set_frame_aggregate_output_mode(
                ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
            )

        try:
            self._pipeline.enable_frame_sync()
        except Exception:
            # Some firmware synchronizes profiles without an explicit switch.
            pass

        try:
            self._pipeline.start(config)
            self._set_device_identity()
            self._intrinsics = self._intrinsics_from_profile(color_profile)
        except Exception:
            self.stop()
            raise

        self._running = True
        self._frame_count = 0
        print(
            f"Camera started: {self._device_model} "
            f"{self._intrinsics.width}x{self._intrinsics.height} @ {self._fps}fps"
        )
        print(
            f"Intrinsics: fx={self._intrinsics.fx:.2f}, "
            f"fy={self._intrinsics.fy:.2f}, cx={self._intrinsics.cx:.2f}, "
            f"cy={self._intrinsics.cy:.2f}"
        )
        print(f"Alignment: Depth-to-Color ({self._active_alignment_mode})")
        return self._intrinsics

    def _create_pipeline(self, ob):
        if not self._serial_number:
            return ob.Pipeline()

        context = ob.Context()
        devices = context.query_devices()
        for index in range(devices.get_count()):
            device = devices[index]
            info = device.get_device_info()
            if info.get_serial_number() == self._serial_number:
                return ob.Pipeline(device)
        raise RuntimeError(
            f"Orbbec camera serial {self._serial_number!r} was not found"
        )

    def _select_hardware_d2c_profile(self, color_profile):
        ob = self._ob
        profiles = self._pipeline.get_d2c_depth_profile_list(
            color_profile, ob.OBAlignMode.HW_MODE
        )
        if profiles is None or self._profile_count(profiles) == 0:
            raise RuntimeError(
                "Gemini 336 has no hardware D2C profile for the requested color "
                f"stream {self._width}x{self._height}@{self._fps}. Set camera."
                "alignment_mode to 'software' or choose a supported stream profile."
            )

        candidates = self._video_profiles(profiles)
        for profile in candidates:
            if (
                profile.get_width() == self._depth_width
                and profile.get_height() == self._depth_height
                and profile.get_fps() == self._depth_fps
            ):
                return profile
        # D2C defines the compatible depth modes for this color profile.
        return candidates[0]

    @classmethod
    def _select_profile(
        cls,
        profiles,
        width: int,
        height: int,
        fps: int,
        formats: Sequence,
        stream_name: str,
    ):
        available = cls._video_profiles(profiles)
        for requested_format in formats:
            for profile in available:
                if (
                    profile.get_width() == width
                    and profile.get_height() == height
                    and profile.get_fps() == fps
                    and profile.get_format() == requested_format
                ):
                    return profile

        summary = ", ".join(
            f"{p.get_width()}x{p.get_height()}@{p.get_fps()} {p.get_format()}"
            for p in available
        )
        raise RuntimeError(
            f"Requested Orbbec {stream_name} stream {width}x{height}@{fps} is "
            f"not supported. Available profiles: {summary}"
        )

    @staticmethod
    def _profile_count(profiles) -> int:
        if hasattr(profiles, "get_count"):
            return profiles.get_count()
        return len(profiles)

    @classmethod
    def _video_profiles(cls, profiles):
        result = []
        for index in range(cls._profile_count(profiles)):
            if hasattr(profiles, "get_stream_profile_by_index"):
                profile = profiles.get_stream_profile_by_index(index)
            else:
                profile = profiles[index]
            if hasattr(profile, "as_video_stream_profile"):
                profile = profile.as_video_stream_profile()
            result.append(profile)
        return result

    @staticmethod
    def _intrinsics_from_profile(profile) -> CameraIntrinsics:
        intrinsic = profile.get_intrinsic()
        return CameraIntrinsics(
            fx=float(intrinsic.fx),
            fy=float(intrinsic.fy),
            cx=float(intrinsic.cx),
            cy=float(intrinsic.cy),
            width=int(intrinsic.width),
            height=int(intrinsic.height),
        )

    def _set_device_identity(self) -> None:
        try:
            info = self._pipeline.get_device().get_device_info()
            self._device_model = info.get_name()
            self._device_serial = info.get_serial_number()
        except Exception:
            pass

    def read(self) -> Optional[RGBDFrame]:
        """Read one synchronized frame with depth pixels aligned to color."""
        if not self._running:
            return None

        try:
            frames = self._pipeline.wait_for_frames(5000)
            if not frames:
                return None
            if self._align_filter is not None:
                frames = self._align_filter.process(frames)
                if not frames:
                    return None
                frames = frames.as_frame_set()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return None

            color_bgr = self._color_to_bgr(color_frame)
            depth_data = np.frombuffer(
                depth_frame.get_data(), dtype=np.uint16
            ).reshape((depth_frame.get_height(), depth_frame.get_width())).copy()
            if color_bgr is None:
                return None
            if depth_data.shape != color_bgr.shape[:2]:
                raise RuntimeError(
                    "Orbbec D2C alignment returned mismatched RGB/depth sizes: "
                    f"RGB={color_bgr.shape[:2]}, depth={depth_data.shape}"
                )

            # Orbbec SDK reports depth scale in millimetres per raw count.
            self._depth_scale = float(depth_frame.get_depth_scale()) / 1000.0
            self._frame_count += 1
            return RGBDFrame(
                color_bgr=color_bgr,
                depth_data=depth_data,
                depth_frame=depth_frame,
                color_frame=color_frame,
                intrinsics=self._intrinsics,
                depth_scale=self._depth_scale,
                timestamp=time.time(),
                frame_id=self._frame_count,
                metadata={
                    "camera_model": self._device_model,
                    "camera_serial": self._device_serial,
                    "alignment": f"depth_to_color_{self._active_alignment_mode}",
                    "depth_scale_unit": "m/count",
                },
            )
        except Exception as exc:
            print(f"Frame read error: {exc}")
            return None

    def _color_to_bgr(self, frame) -> Optional[np.ndarray]:
        import cv2

        ob = self._ob
        width, height = frame.get_width(), frame.get_height()
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        color_format = frame.get_format()
        if color_format == ob.OBFormat.RGB:
            return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
        if color_format == ob.OBFormat.BGR:
            return data.reshape(height, width, 3).copy()
        if color_format == ob.OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if color_format in (ob.OBFormat.YUYV, ob.OBFormat.YUY2):
            return cv2.cvtColor(
                data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2
            )
        if color_format == ob.OBFormat.UYVY:
            return cv2.cvtColor(
                data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_UYVY
            )
        raise RuntimeError(f"Unsupported Orbbec color format: {color_format}")

    def pixel_to_3d(self, u: int, v: int, depth_frame=None) -> Optional[tuple]:
        if depth_frame is None or self._intrinsics is None:
            return None
        width, height = depth_frame.get_width(), depth_frame.get_height()
        if not (0 <= u < width and 0 <= v < height):
            return None
        data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(height, width)
        depth_m = float(data[v, u]) * float(depth_frame.get_depth_scale()) / 1000.0
        if depth_m <= 0 or not np.isfinite(depth_m):
            return None
        x = (u - self._intrinsics.cx) * depth_m / self._intrinsics.fx
        y = (v - self._intrinsics.cy) * depth_m / self._intrinsics.fy
        return (x, y, depth_m)

    def sample_depth_robust(
        self,
        depth_data: np.ndarray,
        u: float,
        v: float,
        mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> DepthSample:
        return sample_depth(
            depth_data=depth_data,
            u=u,
            v=v,
            depth_scale=self._depth_scale,
            mask=mask,
            **kwargs,
        )

    def stop(self) -> None:
        self._running = False
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
                print("Camera stopped.")
            except Exception as exc:
                print(f"Camera stop error: {exc}")
        self._pipeline = None
        self._align_filter = None
