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
