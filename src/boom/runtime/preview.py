# 檔案功能：預覽縮放，保持偵測座標不變。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""Reduce display transfer cost without changing detection coordinates."""
import cv2


def preview_frame(frame, max_width=1280):
    if max_width > 0 and frame.shape[1] > max_width:
        height = max(1, round(frame.shape[0] * max_width / frame.shape[1]))
        return cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_LINEAR)
    return frame
