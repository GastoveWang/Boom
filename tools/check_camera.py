# 檔案功能：列出 PySpin SDK 與相機，可指定幀數驗證取像。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""Read-only SDK discovery; optionally acquire a bounded set of frames."""
from pathlib import Path
import argparse
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from boom.online.spin_camera import _load_pyspin
from boom.online.spinnaker_camera import SpinCamera, SpinCameraConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial")
    parser.add_argument("--frames", type=int, default=0,
                        help="0 lists cameras; positive number also tests acquisition.")
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    sdk = _load_pyspin()
    system = sdk.System.GetInstance()
    cameras = None
    try:
        version = system.GetLibraryVersion()
        print(f"SDK {version.major}.{version.minor}.{version.type}.{version.build}")
        cameras = system.GetCameras()
        count = cameras.GetSize()
        print(f"Cameras detected: {count}")
        for index in range(count):
            camera = cameras.GetByIndex(index)
            nodes = camera.GetTLDeviceNodeMap()
            fields = []
            for name in ("DeviceModelName", "DeviceSerialNumber"):
                node = sdk.CStringPtr(nodes.GetNode(name))
                fields.append(f"{name}={node.GetValue() if sdk.IsReadable(node) else 'unknown'}")
            print(f"  [{index}] " + " ".join(fields))
            del node, nodes, camera
    finally:
        if cameras is not None:
            cameras.Clear()
        system.ReleaseInstance()
    if count == 0:
        print("No camera detected. Connect USB3/GigE and close SpinView before acquisition.")
        return 2
    if args.frames:
        with SpinCamera(SpinCameraConfig(serial=args.serial)) as camera:
            started = time.monotonic()
            for index in range(args.frames):
                frame = camera.read()
                if frame is None:
                    raise RuntimeError("Incomplete frame or camera timeout")
                if index == 0:
                    print(f"First frame OK: {frame.bgr.shape}, serial={camera.serial}")
            elapsed = max(time.monotonic() - started, 1e-6)
            print(f"Acquired {args.frames} frames at {args.frames / elapsed:.2f} FPS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ImportError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
