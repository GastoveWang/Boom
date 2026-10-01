# 檔案功能：正式訓練入口，設定由 runner.parser 定義。
# 執行設定：doc/RUN_SETTINGS.md；ASTT 正式參數見 runner.py parser()，歷史 config.py 不影響正式流程。
from runner import parser, train

if __name__ == '__main__':
    train(parser(training=True).parse_args())
