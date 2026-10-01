# 檔案功能：pipelines 套件初始化與公開匯出；不直接執行。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""
==============================================================================
Boom 離線執行管線進入點模組 (boom.pipelines)
==============================================================================

【模組定位】
本模組提供離線批次與驗證管線的參數解析與個別演算法進入點。

【包含進入點與工具】
1. `main.py`: 統一步驟調度器 (`main`)，由各演算法 entrypoint 統一調用。
2. `cli.py`: 命令列參數規範、型別轉換與合法性檢查 (`build_arg_parser`, `validate_args`)。
3. `optical_flow_pipeline.py`: 光流演算法專屬進入點。
4. `map_localization.py`: 提供向後相容的定位模組符號轉接。

【導出類別與函式】
- `build_arg_parser`, `validate_args`: CLI 參數解析與驗證
- `main`: 離線管線執行進入點
==============================================================================
"""

from .cli import build_arg_parser, map_matching_config, validate_args
from .main import main

__all__ = [
    "build_arg_parser",
    "validate_args",
    "map_matching_config",
    "main",
]
