# 檔案功能：主執行緒 HighGUI 與最新預覽畫面傳遞。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""Keep HighGUI on the main thread and processing on one owned worker.

The mailbox holds only the newest completed visualization. Dropping a preview
never drops offline inference, recorded frames, or event evidence.
"""
from __future__ import annotations

import queue
import signal
import threading
from typing import Callable

import cv2
import numpy as np

from .preview import preview_frame


class PreviewChannel:
    def __init__(self, enabled: bool, max_width: int = 1280):
        self.enabled = enabled
        self.max_width = max_width
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._scroll = queue.Queue(maxsize=32)

    def publish(self, frame):
        if not self.enabled or self.stop.is_set():
            return
        # Resize already owns new pixels; copy only when displaying at source size.
        snapshot = preview_frame(frame, self.max_width)
        if snapshot is frame:
            snapshot = frame.copy()
        with self._lock:
            self._latest = snapshot

    def take_latest(self):
        with self._lock:
            frame, self._latest = self._latest, None
            return frame

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            try:
                self._scroll.put_nowait(flags)
            except queue.Full:
                pass

    def take_scroll(self):
        while True:
            try:
                yield self._scroll.get_nowait()
            except queue.Empty:
                return


def run_with_preview(process: Callable, *, enabled: bool, max_width: int = 1280):
    """Run process(channel), propagate failures, and join before releasing UI.

    Headless processing stays synchronous. Signals are handled in the calling
    main thread and restored on exit; GPU/camera/storage remain worker-owned.
    """
    channel = PreviewChannel(enabled, max_width)
    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda *_: channel.stop.set())
    window = "Boom - Wilderness smoke detection"
    worker = None
    done = threading.Event()
    results = []
    errors = []
    window_open = False
    stopping_shown = False
    try:
        if not enabled:
            return process(channel)
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Display must be started from the main thread")

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        window_open = True
        splash = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(splash, "Starting camera / model...  Q: stop", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imshow(window, splash)
        cv2.setMouseCallback(window, channel.on_mouse)

        def work():
            try:
                results.append(process(channel))
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()

        worker = threading.Thread(target=work, name="boom-processing", daemon=False)
        worker.start()
        displayed = splash
        while True:
            if window_open and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                channel.stop.set()
                window_open = False
            frame = channel.take_latest()
            if frame is not None and window_open:
                cv2.imshow(window, frame)
                displayed = frame
            if channel.stop.is_set() and window_open and not stopping_shown:
                notice = displayed.copy()
                cv2.rectangle(notice, (0, 0), (notice.shape[1], 44), (20, 20, 20), -1)
                cv2.putText(notice, "Stopping: finishing active work...", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.imshow(window, notice)
                stopping_shown = True
            if done.is_set():
                # Give HighGUI one final pump to paint the final result.
                if window_open:
                    cv2.waitKey(1)
                break
            if window_open:
                key = cv2.waitKey(10) & 0xFF
                if key in (27, ord("q")):
                    channel.stop.set()
            else:
                done.wait(0.01)
        worker.join()
        if errors:
            raise errors[0]
        return results[0] if results else None
    finally:
        channel.stop.set()
        try:
            if worker is not None:
                # Never release a camera/model underneath active native work.
                worker.join()
            if window_open:
                try:
                    cv2.destroyWindow(window)
                except cv2.error:
                    pass  # The user or window manager may already have closed it.
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
