# 檔案功能：config 套件初始化與公開匯出；不直接執行。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""
==============================================================================
Boom 設定模組 (config)
==============================================================================

提供系統設定讀取與管理介面。
==============================================================================
"""

from .loader import load_config, get_config, ConfigFacade
from .defaults import MapMatchingConfig

__all__ = ["load_config", "get_config", "ConfigFacade", "MapMatchingConfig"]
