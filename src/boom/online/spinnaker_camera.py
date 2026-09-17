"""Public Spinnaker camera module used by the online runner."""

from __future__ import annotations

from typing import Any

if __package__:
    from .spin_camera import (
        CapturedFrame,
        LatestFrameGrabber,
        SpinCamera as _BaseSpinCamera,
        SpinCameraConfig,
        _load_pyspin,
    )
else:
    from spin_camera import (
        CapturedFrame,
        LatestFrameGrabber,
        SpinCamera as _BaseSpinCamera,
        SpinCameraConfig,
        _load_pyspin,
    )

__all__ = [
    "CapturedFrame",
    "LatestFrameGrabber",
    "SpinCamera",
    "SpinCameraConfig",
]


class SpinCamera(_BaseSpinCamera):
    """Open the SDK and select a camera before configuring acquisition."""

    def open(self) -> None:
        if self._camera is not None:
            return

        pyspin = _load_pyspin()
        system = pyspin.System.GetInstance()
        camera_list = system.GetCameras()
        camera: Any = None
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
                if camera is not None and camera.IsInitialized():
                    camera.DeInit()
            finally:
                camera_list.Clear()
                system.ReleaseInstance()
            self._camera = None
            self._camera_list = None
            self._system = None
            self._pyspin = None
            raise
