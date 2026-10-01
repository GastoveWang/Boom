# 檔案功能：離線流程選擇與啟動。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""
==============================================================================
Boom 離線管線 - 統一調度與啟動器 (main.py)
==============================================================================

【檔案定位】
本檔案為 YOLO 與光流管線的統一調度轉接核心。
將命令列解析、參數合法性驗證與執行期 `runner.run()` 串接為單一入口。

【核心功能】
- `main(default_detector, allow_detector_selection, argv)`:
  建立對應演算法的參數解析器，解析 `argv`、執行 `validate_args` 校驗，
  最後調用 `runtime.runner.run(args)` 啟動完整的離線處理迴圈。

【相依模組】
- 上游：被 `optical_flow_pipeline.py` 與 `run_offline.py` 調用。
- 下游：調用 `cli.build_arg_parser`, `cli.validate_args`, `runtime.runner.run`。
==============================================================================
"""
from boom.config.defaults import DETECTOR_BACKEND
from .cli import build_arg_parser, validate_args
from boom.runtime.runner import run


def main(default_detector=DETECTOR_BACKEND, *, allow_detector_selection=True, argv=None):
    args = build_arg_parser(default_detector, allow_detector_selection=allow_detector_selection).parse_args(argv)
    validate_args(args)
    return run(args)
