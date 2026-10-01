# 檔案功能：非同步影片輸出與佇列管理。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""Bounded, ordered background output. Errors always reach the caller."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque


class AsyncVideoOutput:
    """Overlap encoding with detection without dropping offline frames.

    A full queue applies backpressure. Frame copies isolate the encoder from
    subsequent drawing. Writes use one worker; close follows worker shutdown.
    """

    def __init__(self, output, queue_size=2):
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.output = output
        self.queue_size = queue_size
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-output")
        self._pending = deque()
        self._closed = False

    def write(self, frame):
        if self._closed:
            raise RuntimeError("Video output is closed")
        while self._pending and self._pending[0].done():
            self._pending.popleft().result()
        if len(self._pending) >= self.queue_size:
            self._pending.popleft().result()
        self._pending.append(self._executor.submit(self.output.write, frame.copy()))

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            # Drain every accepted frame even when a previous write failed.
            self._executor.shutdown(wait=True)
            for future in self._pending:
                future.result()
        finally:
            self.output.close()
            self._pending.clear()
