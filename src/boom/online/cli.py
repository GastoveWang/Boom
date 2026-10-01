# 檔案功能：即時應用程式啟動入口。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""Online application entrypoint."""
import sys
from .configuration import build_arg_parser
from .pipeline import run

def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        raise SystemExit(run(args))
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
