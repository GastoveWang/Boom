"""Low-latency FLIR/Teledyne Spinnaker (SpinView) camera adapter.

PySpin is imported only when a camera is opened.  This keeps the rest of the
online package importable on development machines without the Spinnaker SDK.
"""

from __future__ import annotations

import importlib
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class SpinCameraConfig:
    serial: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    offset_x: Optional[int] = None
    offset_y: Optional[int] = None
    fps: Optional[float] = 30.0
    exposure_us: Optional[float] = None
    gain_db: Optional[float] = None
    timeout_ms: int = 1000


@dataclass(frozen=True)
class CapturedFrame:
    bgr: np.ndarray
    camera_frame_id: int
    camera_timestamp_ns: int
    captured_monotonic: float


def _load_pyspin() -> Any:
    try:
        return importlib.import_module("PySpin")
    except ImportError as exc:
        raise RuntimeError(
            "PySpin is not installed. Install the NVIDIA Jetson/aarch64 build "
            "of the FLIR/Teledyne Spinnaker SDK that matches the JetPack "
            "Python version; PySpin is supplied by that SDK, not by pip."
        ) from exc


class SpinCamera:
    """Context-managed Spinnaker camera returning independent BGR arrays."""

    def __init__(self, config: SpinCameraConfig) -> None:
        self.config = config
        self._pyspin: Any = None
        self._system: Any = None
        self._camera_list: Any = None
        self._camera: Any = None
        self._processor: Any = None
        self._acquiring = False
        self.serial = ""

    def __enter__(self) -> "SpinCamera":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def open(self) -> None:
        if self._camera is not None:
            return

        pyspin = _load_pyspin()
        system = pyspin.System.GetInstance()
        camera_list = system.GetCameras()
        try:
            self._pyspin = pyspin
            if camera_list.GetSize() == 0:
                raise RuntimeError(
                    "No Spinnaker camera found. Close SpinView, check USB/GigE "
                    "connection and permissions, then try again."
                )
            camera = self._select_camera(camera_list)
            camera.Init()
            self._system = system
            self._camera_list = camera_list
            self._camera = camera
            self.serial = self._read_string("DeviceSerialNumber") or "unknown"
            self._configure()
            processor = pyspin.ImageProcessor()
            processor.SetColorProcessing(
                pyspin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_NEAREST_NEIGHBOR
            )
            self._processor = processor
            camera.BeginAcquisition()
            self._acquiring = True
        except Exception:
            try:
                if "camera" in locals() and camera.IsInitialized():
                    camera.DeInit()
            finally:
                camera_list.Clear()
                system.ReleaseInstance()
            self._camera = None
            raise

    def close(self) -> None:
        grabber = getattr(self, '_active_grabber', None)
        if grabber is not None:
            grabber.stop()
            self._active_grabber = None
        camera = self._camera
        if camera is not None:
            try:
                if self._acquiring:
                    camera.EndAcquisition()
            finally:
                self._acquiring = False
                camera.DeInit()
        if self._camera_list is not None:
            self._camera_list.Clear()
        if self._system is not None:
            self._system.ReleaseInstance()
        self._processor = None
        self._camera = None
        self._camera_list = None
        self._system = None
        self._pyspin = None

    def read(self) -> Optional[CapturedFrame]:
        if not self._acquiring or self._camera is None:
            raise RuntimeError("Camera acquisition has not started.")

        try:
            image = self._camera.GetNextImage(max(int(self.config.timeout_ms), 1))
        except Exception as exc:
            error_code = getattr(exc, "errorcode", None)
            timeout_code = getattr(self._pyspin, "SPINNAKER_ERR_TIMEOUT", None)
            if error_code == timeout_code or "timeout" in str(exc).lower():
                return None
            raise

        try:
            if image.IsIncomplete():
                return None
            frame_id = int(image.GetFrameID())
            timestamp_ns = int(image.GetTimeStamp())
            converted = self._processor.Convert(
                image, self._pyspin.PixelFormat_BGR8
            )
            frame = np.asarray(converted.GetNDArray()).copy()
            return CapturedFrame(
                bgr=frame,
                camera_frame_id=frame_id,
                camera_timestamp_ns=timestamp_ns,
                captured_monotonic=time.monotonic(),
            )
        finally:
            image.Release()

    def _select_camera(self, camera_list: Any) -> Any:
        if not self.config.serial:
            return camera_list.GetByIndex(0)
        for index in range(camera_list.GetSize()):
            camera = camera_list.GetByIndex(index)
            node_map = camera.GetTLDeviceNodeMap()
            node = self._pyspin.CStringPtr(node_map.GetNode("DeviceSerialNumber"))
            if (
                self._pyspin.IsReadable(node)
                and node.GetValue() == self.config.serial
            ):
                return camera
        raise RuntimeError(
            f"Spinnaker camera serial {self.config.serial!r} was not found."
        )

    def _configure(self) -> None:
        # Keep only the newest frame so a slow detector does not build latency.
        self._set_enum("StreamBufferHandlingMode", "NewestOnly", stream=True)
        self._set_enum("AcquisitionMode", "Continuous")

        # Apply size before offset; offsets can become invalid after ROI changes.
        self._set_int("Width", self.config.width)
        self._set_int("Height", self.config.height)
        self._set_int("OffsetX", self.config.offset_x)
        self._set_int("OffsetY", self.config.offset_y)

        if self.config.fps is not None:
            self._set_bool("AcquisitionFrameRateEnable", True)
            self._set_float("AcquisitionFrameRate", self.config.fps)
        if self.config.exposure_us is not None:
            self._set_enum("ExposureAuto", "Off")
            self._set_float("ExposureTime", self.config.exposure_us)
        if self.config.gain_db is not None:
            self._set_enum("GainAuto", "Off")
            self._set_float("Gain", self.config.gain_db)

    def _node_map(self, stream: bool = False) -> Any:
        return (
            self._camera.GetTLStreamNodeMap()
            if stream
            else self._camera.GetNodeMap()
        )

    def _read_string(self, name: str) -> Optional[str]:
        node = self._pyspin.CStringPtr(self._camera.GetTLDeviceNodeMap().GetNode(name))
        return node.GetValue() if self._pyspin.IsReadable(node) else None

    def _set_enum(self, name: str, entry_name: str, stream: bool = False) -> None:
        node = self._pyspin.CEnumerationPtr(self._node_map(stream).GetNode(name))
        if not self._pyspin.IsWritable(node):
            return
        entry = node.GetEntryByName(entry_name)
        if self._pyspin.IsReadable(entry):
            node.SetIntValue(entry.GetValue())

    def _set_bool(self, name: str, value: bool) -> None:
        node = self._pyspin.CBooleanPtr(self._node_map().GetNode(name))
        if self._pyspin.IsWritable(node):
            node.SetValue(bool(value))

    def _set_int(self, name: str, value: Optional[int]) -> None:
        if value is None:
            return
        node = self._pyspin.CIntegerPtr(self._node_map().GetNode(name))
        if not self._pyspin.IsWritable(node):
            return
        bounded = min(max(int(value), node.GetMin()), node.GetMax())
        increment = max(int(node.GetInc()), 1)
        bounded = node.GetMin() + ((bounded - node.GetMin()) // increment) * increment
        node.SetValue(bounded)

    def _set_float(self, name: str, value: float) -> None:
        node = self._pyspin.CFloatPtr(self._node_map().GetNode(name))
        if self._pyspin.IsWritable(node):
            node.SetValue(min(max(float(value), node.GetMin()), node.GetMax()))


class LatestFrameGrabber:
    """Continuously acquire frames and expose only the newest unpublished one."""

    def __init__(self, camera: SpinCamera) -> None:
        self.camera = camera
        self.camera._active_grabber = self
        self._condition = threading.Condition()
        self._frame: Optional[CapturedFrame] = None
        self._sequence = 0
        self._published_sequence = 0
        self._stopping = False
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._capture_loop, name="spin-camera-capture", daemon=True
        )

    @property
    def overwritten_frames(self) -> int:
        with self._condition:
            return max(self._sequence - self._published_sequence - 1, 0)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if self._thread.is_alive():
            wait_seconds = max(self.camera.config.timeout_ms / 1000.0 + 1.0, 2.0)
            self._thread.join(timeout=wait_seconds)

    def get_latest(self, last_sequence: int, timeout: float = 2.0) -> tuple[int, CapturedFrame]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._sequence <= last_sequence and not self._stopping:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a Spinnaker frame.")
                self._condition.wait(remaining)
            if self._error is not None:
                raise RuntimeError("Spinnaker capture thread failed.") from self._error
            if self._frame is None or self._sequence <= last_sequence:
                raise RuntimeError("Camera capture stopped before a new frame arrived.")
            self._published_sequence = self._sequence
            return self._sequence, self._frame

    def _capture_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    if self._stopping:
                        return
                frame = self.camera.read()
                if frame is None:
                    continue
                with self._condition:
                    self._frame = frame
                    self._sequence += 1
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._stopping = True
                self._condition.notify_all()
