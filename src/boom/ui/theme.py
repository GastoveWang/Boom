"""
==============================================================================
Boom 離線使用者介面 - 視覺色彩主題與圓角卡片工具 (theme.py)
==============================================================================

【檔案定位】
定義全系統視覺化監控介面的現代化暗色系調色盤（BGR 格式）與高品質圓角矩形卡片繪製工具。

【常數與工具】
- 色彩定義：`BACKGROUND`, `SURFACE`, `CARD`, `BORDER`, `ACCENT`, `TEXT`。
- `rounded_surface(canvas, rect, color, radius)`: 繪製平滑反鋸齒圓角矩形區塊。
==============================================================================
"""
import cv2

BACKGROUND = (36, 31, 26)
SURFACE = (48, 42, 35)
CARD = (57, 50, 42)
BORDER = (85, 74, 61)
ACCENT = (225, 172, 105)
TEXT = (245, 240, 232)


def rounded_surface(canvas, rect, color=SURFACE, radius=12):
    x, y, w, h = rect
    r = min(radius, w//2, h//2)
    cv2.rectangle(canvas, (x+r, y), (x+w-r-1, y+h-1), color, -1)
    cv2.rectangle(canvas, (x, y+r), (x+w-1, y+h-r-1), color, -1)
    for cx, cy in ((x+r, y+r), (x+w-r-1, y+r), (x+r, y+h-r-1), (x+w-r-1, y+h-r-1)):
        cv2.circle(canvas, (cx, cy), r, color, -1, cv2.LINE_AA)
