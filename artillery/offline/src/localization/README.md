# Localization

地圖、GPS 照片初始化與地面平面約束 VO 的共用功能。執行方式與定位限制見
[MAP_LOCALIZATION.md](../../MAP_LOCALIZATION.md)。

## 模組分工

| 檔案 | 負責功能 |
| --- | --- |
| `photo.py` | 照片讀取、GPS／XMP、相機內參與初始姿態 |
| `regions.py` | 地圖範圍索引、GPS／估計拍攝中心的候選區域選擇 |
| `geo_map.py` | GeoTIFF 載入、透明區域合成、快取與地圖座標轉換 |
| `geometry.py` | 地面投影、像素轉換與 PnP 姿態估計／驗證 |
| `registration.py` | 照片對地圖初始化、照片對影片銜接、地圖漂移校正 |
| `visual_odometry.py` | 逐格特徵追蹤、補點、重定位排程與姿態歷史 |
| `map_panel.py` | 地圖 UI、無人機圖示、軌跡與煙霧事件標點 |
| `map_widgets.py` | 台灣範圍小地圖、完整指北針與無人機方向扇形 |
| `localizer.py` | `MapLocalizer` 對外介面、session 狀態、事件投影與 JSONL 紀錄 |
| `__init__.py` | 公開匯入介面 |

Runner 透過 `MapLocalizer` 呼叫這些模組。匹配與 VO 函式共用 session 狀態，
幾何運算集中在 `geometry.py`，畫面繪製集中在 `map_panel.py`。
這次拆分保留原演算法、門檻、狀態與輸出行為。

台灣小地圖以離線 Natural Earth 輪廓呈現，框線依主視窗地理範圍更新；
小於可讀尺寸時放大定位框。主地圖固定北向上。目前假設雲台與機頭方向一致，
方向扇形與角度統一表示無人機朝向，不另外區分兩組方向。
扇形為方向示意，並非精確地面視野範圍；相機垂直朝下時不顯示水平朝向。
信心與定位狀態保留於資料紀錄，畫面只顯示事件編號、時間及座標。

台灣概覽與指北針整合為右上方同一張導航卡，採用與主畫面一致的深藍灰配色。
初始構圖會預留導航卡高度，再將地圖與無人機共同縮放到下方可視範圍，避免被卡片遮擋。
主地圖仍固定北向上；這些版面調整不修改地理座標或偵測／VO 演算法。

```python
from artillery.offline.src.localization import MapLocalizer
```

原本 `artillery.offline.src.pipelines.map_localization` 僅轉接公開符號，
方便既有程式沿用舊匯入路徑；新程式使用上述 package 路徑。

從專案根目錄執行測試：

```powershell
python -m unittest discover -s tests -p test_map_localization.py -v
```
