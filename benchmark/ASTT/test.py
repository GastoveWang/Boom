# 檔案功能：正式驗證／測試入口，必須提供本機 checkpoint。
# 執行設定：doc/RUN_SETTINGS.md；ASTT 正式參數見 runner.py parser()，歷史 config.py 不影響正式流程。
from runner import parser, infer

if __name__ == '__main__':
    infer(parser(training=False).parse_args())
