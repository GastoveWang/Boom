# 檔案功能：分類 YAML 讀取、深度合併與設定存取。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""
==============================================================================
Boom 設定載入器 (loader.py)
==============================================================================

【模組職責】
負責載入與解析階層式 YAML 設定檔，提供預設設定、覆寫合併與路徑自動解析。
支援自訂設定檔路徑與環境變數覆寫。
==============================================================================
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "default.yaml"
_CONFIG_PARTS = (
    "system.yaml",
    "ui.yaml",
    "pipeline.yaml",
    "detector.yaml",
    "detector-yolo.yaml",
    "detector-motion.yaml",
    "matching.yaml",
    "localization.yaml",
)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """遞迴合併兩個字典"""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    載入設定檔。若指定 config_path 則會與 default.yaml 深度合併。

    :param config_path: 自訂 YAML 設定檔路徑 (可選)
    :return: 包含完整參數的巢狀字典
    """
    cfg: Dict[str, Any] = {}
    for path in (*(_CONFIG_DIR / name for name in _CONFIG_PARTS), _DEFAULT_CONFIG_PATH):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    if config_path:
        custom_p = Path(config_path)
        if custom_p.exists():
            with open(custom_p, "r", encoding="utf-8") as f:
                custom_cfg = yaml.safe_load(f) or {}
                cfg = _deep_merge(cfg, custom_cfg)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    return cfg


class ConfigFacade:
    """提供屬性風格訪問的設定外觀物件"""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getitem__(self, item: str) -> Any:
        return self._data[item]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    @property
    def raw(self) -> Dict[str, Any]:
        return self._data


_GLOBAL_CONFIG: Optional[ConfigFacade] = None


def get_config(config_path: Optional[str | Path] = None, reload: bool = False) -> ConfigFacade:
    """取得全域單例設定物件"""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None or reload or config_path is not None:
        _GLOBAL_CONFIG = ConfigFacade(load_config(config_path))
    return _GLOBAL_CONFIG
