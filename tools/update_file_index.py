# 檔案功能：更新逐檔用途索引、Python 功能註記與 CLI 設定表。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""Regenerate Boom's file-purpose index, source header comments and CLI reference."""
import ast
import importlib.util
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PURPOSES = {
    'AGENTS.md': '專案目標、功能優先順序與修改／驗證原則。',
    'pyproject.toml': 'Boom 套件、Python 版本、基礎／ML／ONNX／地圖依賴與安裝設定。',
    'doc/PROJECT_STRUCTURE.md': '專案架構、偵測流程與離線／即時執行範例。',
    'doc/ONLINE.md': '即時 Spinnaker 相機、偵測、輸出與設備設定。',
    'doc/ACCELERATION.md': 'YOLO／Motion 執行效能、GPU 與顯示加速。',
    'doc/MAP_LOCALIZATION.md': 'GeoTIFF 地圖、特徵匹配、參考照片與地面定位。',
    'doc/ASSET_UI.md': '地圖與 UI 圖示資源的來源與用途。',
    'doc/TOOLS.md': '抽幀、YOLO 資料切分、訓練與 ONNX 匯出指令。',
    'doc/FILE_INDEX.md': '專案逐檔用途與資源目錄索引。',
    'doc/RUN_SETTINGS.md': '執行入口、目前 YAML 值與 CLI 完整參數表。',
    'doc/README.md': '文件入口與比較專案的 Git 追蹤規則。',
    'configs/README.md': '分類 YAML 用途、覆寫順序與地圖匹配設定。',
    'benchmark/ASTT/START_HERE.md': 'ASTT 正式／歷史檔案分组與訓練、測試快速操作。',
    'benchmark/ASTT/README_LOCAL.md': 'ASTT 本機適配、模型解讀、資料格式、設定與限制。',
    'benchmark/ASTT/README.md': 'ASTT 上游論文、作者與引用資訊。',
    'benchmark/ASTT/requirements.txt': '正式 ASTT 流程的 torch、NumPy、OpenCV 依賴。',
    'run_offline.py': '離線影片入口：選擇 motion 或 YOLO，執行偵測、顯示、保存與可選定位。',
    'run_realtime.py': 'Spinnaker 即時相機入口：取像、偵測、畫面顯示與事件保存。',
    'tools/check_camera.py': '列出 PySpin SDK 與相機，可指定幀數驗證取像。',
    'tools/video_to_frames.py': '將影片依時間間隔抽成圖片，保留原始解析度。',
    'tools/split_dataset.py': 'YOLO 偵測／分割資料版本管理、平衡切分與 GUI。',
    'tools/train_yolo.py': '互動式 YOLO 訓練、選擇本機模型與 ONNX 匯出。',
    'tools/benchmark_matchers.py': '同影像對比較 EDM 與 SuperPoint/LightGlue 的匹配表現。',
    'tools/update_file_index.py': '更新逐檔用途索引、Python 功能註記與 CLI 設定表。',
}
GROUPS = {
    'config': {'defaults':'共用預設常數與地圖匹配設定結構', 'loader':'分類 YAML 讀取、深度合併與設定存取'},
    'core': {'coordinates':'WGS84/TWD97 座標轉換與地面投影', 'events':'地理參考、確認事件與紀錄資料結構'},
    'detection': {'factory':'建立 motion/YOLO 偵測器及對應設定'},
    'optical_flow': {'detector':'影像變化、相機運動補償、光流與煙塵候選事件追蹤', 'runner':'motion 單影片／批次 runner 與獨立 CLI', 'types':'motion 設定、影格、候選區域、追蹤與輸出資料結構'},
    'yolo': {'detector':'YOLO 推論、目標類別篩選與事件追蹤'},
    'interfaces': {'detector':'偵測器共用抽象介面','localizer':'定位器共用抽象介面','matcher':'特徵匹配器共用抽象介面','sensor':'影格來源共用抽象介面'},
    'localization': {'edm_matching':'固定尺寸 EDM ONNX 影像對匹配後端','feature_matching':'SIFT、SuperPoint/LightGlue 後端與建立工廠','geometry':'相機姿態、平面 homography 與姿態估計','geo_map':'GeoTIFF 讀取、ROI 與地圖座標轉換','localizer':'地圖定位器外觀與 session 管理','map_panel':'地圖畫面與無人機圖示合成','map_widgets':'方位、羅盤、台灣概覽與導航卡','nearby':'鄰近地圖 ROI 特徵匹配與定位結果','photo':'DJI 照片 Exif/XMP 與相機姿態讀取','regions':'GeoTIFF 索引、邊界與鄰近圖幅選擇','registration':'參考照片／地圖配準與矩陣校正','visual_odometry':'影片視覺里程與相機姿態追蹤'},
    'online': {'cli':'即時應用程式啟動入口','configuration':'即時 CLI、相機及偵測設定校驗','display':'即時事件疊圖','logging_setup':'主控台與 session 檔案 logging','pipeline':'取像、即時偵測與證據輸出協調','positioning':'可選固定姿態的地面位置估計','runner':'既有即時入口的相容匯出','spinnaker_camera':'公開 Spinnaker 相機相容模組','spin_camera':'PySpin 相機設定、取像與最新影格執行緒','storage':'事件 JSONL/CSV、座標與截圖保存'},
    'pipelines': {'cli':'離線 CLI 定義、參數校驗與地圖匹配設定','main':'離線流程選擇與啟動','map_localization':'地圖定位舊匯入路徑相容層','optical_flow_pipeline':'motion 離線流程相容入口'},
    'runtime': {'async_output':'非同步影片輸出與佇列管理','events':'事件生命週期、地理定位與歷史','inputs':'影片及參考照片路徑解析與影片來源','live_preview':'主執行緒 HighGUI 與最新預覽畫面傳遞','output':'輸出路徑、座標日誌與影片寫入','preview':'預覽縮放，保持偵測座標不變','progress':'終端進度、耗時與結果統計','reference':'參考照片 GPS/XMP 地理資訊解析','runner':'離線影格處理、偵測、定位與輸出主迴圈'},
    'ui': {'composition':'影片、側欄與地圖多面板合成','drawing':'文字與確認事件標記繪製','event_style':'事件標籤與顏色','panels':'事件卡、debug/client 側欄與地圖事件面板','theme':'配色與圓角卡片繪圖'},
    'ASTT': {'train':'正式訓練入口，設定由 runner.parser 定義','test':'正式驗證／測試入口，必須提供本機 checkpoint','runner':'訓練、驗證、推論、CLI、checkpoint 與輸出核心','dataset':'每影片連續影格窗口、缺幀／標籤校驗與可選 ECC','model':'本機 STE/TTE/CSA 模型與下一幀預測 decoder','localization':'誤差候選 bbox 與事件去抖合併，非 GPS 定位','smoke_test':'暫存資料上的模型、資料、事件与指標自我檢查','model_original':'歷史原始模型，研究參考','model_baseline_ca':'歷史 baseline + cross attention 模型，研究參考','model_ste_tte':'歷史 STE/TTE 模型，研究參考','config':'歷史 ViT argparse 設定；正式入口不使用','data_utils':'歷史影格 loader；正式入口改用 dataset.py','data_loaders':'歷史 CIFAR/ImageNet loader；正式入口不使用','utils':'歷史 metrics、logging 與設定工具','checkpoint':'歷史 JAX/ViT 權重載入與轉換','check_jax':'歷史 JAX 權重檢查入口','eval':'歷史影像分類評估入口，非本機異常測試','Train_and_eval':'歷史資料巡覽與訓練評估腳本，含上游路徑'},
}


def purpose(path):
    rel = path.relative_to(ROOT).as_posix()
    if rel in PURPOSES:
        return PURPOSES[rel]
    if path.name == '__init__.py':
        return f'{path.parent.name} 套件初始化與公開匯出；不直接執行。'
    if rel == 'benchmark/ASTT/scripts/extract_frames.py':
        return '正式 MP4 抽幀入口，保存影格與來源幀號／時間 manifest。'
    if path.stem.startswith('train_val_') and 'benchmark/ASTT/' in rel:
        return '歷史 ' + path.stem.removeprefix('train_val_') + ' 資料集／模型實驗；含上游路徑，非正式入口。'
    group = path.parent.name
    if group in GROUPS and path.stem in GROUPS[group]:
        return GROUPS[group][path.stem] + '。'
    if path.suffix == '.py':
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        description = ast.get_docstring(tree)
        return (description.splitlines()[0] if description else '驗證／輔助模組；詳見檔案內函式與類別。').replace('|', '/')
    return {
        '.yaml': '執行設定；用途與合併順序見 configs/README.md。',
        '.md': '說明文件：' + path.stem + '。',
        '.toml': 'Python 套件、依賴與建置設定。',
        '.txt': '環境依賴清單。',
    }.get(path.suffix, '版本控制／目錄保留設定。')


def annotate(path, description):
    raw = path.read_bytes()
    bom = raw.startswith(b'\xef\xbb\xbf')
    text = raw.decode('utf-8-sig')
    newline = '\r\n' if '\r\n' in text else '\n'
    lines = text.splitlines(keepends=True)
    lines = [line for line in lines if not line.startswith('# 檔案功能：') and not line.startswith('# 執行設定：')]
    position = 1 if lines and lines[0].startswith('#!') else 0
    if len(lines) > position and re.search(r'coding[:=]', lines[position]):
        position += 1
    rel = path.relative_to(ROOT).as_posix()
    reference = 'doc/RUN_SETTINGS.md；ASTT 正式參數見 runner.py parser()，歷史 config.py 不影響正式流程。' if rel.startswith('benchmark/ASTT/') else 'doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。'
    lines[position:position] = ['# 檔案功能：' + description + newline, '# 執行設定：' + reference + newline]
    result = ''.join(lines).encode('utf-8')
    if bom:
        result = b'\xef\xbb\xbf' + result
    path.write_bytes(result)


def cli_table(parser):
    lines = ['| 參數 | 預設值 | 用途／限制 |', '|---|---|---|']
    for action in parser._actions:
        if action.dest == 'help':
            continue
        name = ', '.join(action.option_strings) or action.dest
        default = str(action.default).replace(str(ROOT), '<Boom>').replace('|', '/')
        help_text = (action.help or action.dest.replace('_', ' ')).replace('|', '/')
        if action.choices:
            help_text += '；可選 ' + ', '.join(map(str, action.choices))
        lines.append(f'| `{name}` | `{default}` | {help_text} |')
    return '\n'.join(lines)


def main():
    files = list(ROOT.glob('*.py'))
    for folder in ('src', 'tools', 'benchmark/ASTT'):
        files.extend((ROOT/folder).rglob('*.py'))
    files = sorted(files)
    for path in files:
        annotate(path, purpose(path))
    other = [ROOT/'pyproject.toml', ROOT/'.gitignore', ROOT/'AGENTS.md']
    for folder in ('configs', 'doc'):
        other.extend(p for p in (ROOT/folder).iterdir() if p.is_file())
    other.extend(p for p in (ROOT/'benchmark/ASTT').rglob('*') if p.is_file()
                 and (p.suffix in ('.md', '.txt') or p.name == '.gitignore'))
    rows = ['# Boom 逐檔功能索引', '', 'Boom 是一套用於飛行中無人機的荒野煙霧偵測與定位系統，目標是在影像中即時標示煙霧區域，並估算疑似煙霧源的地面座標。', '',
        '此索引涵蓋專案原始碼、工具、設定與文件；Python 檔案頂端也有功能註記。資料、權重、輸出、第三方標註軟體與 Git/快取不逐檔註記，以免修改使用者資料。', '',
        '直接執行的入口和參數見 [RUN_SETTINGS.md](RUN_SETTINGS.md)，ASTT 分組見 [START_HERE.md](../benchmark/ASTT/START_HERE.md)。', '', '| 檔案 | 功能 |', '|---|---|']
    for path in sorted(set(files + other)):
        if not path.exists():
            continue
        rel = path.relative_to(ROOT).as_posix()
        rows.append(f'| [{rel}](../{rel}) | {purpose(path)} |')
    rows.extend(['', '## 資源與未納入 Git 的目錄', '', '| 目錄 | 功能 |', '|---|---|',
        '| data/picture | 使用者影格與標註；不自動改寫或視為正常資料。 |',
        '| data/astt | ASTT 的 train、val、test 影格與逐幀標籤。 |',
        '| data/yolo_datasets | YOLO 原始／版本化訓練資料。 |',
        '| data/video、data/raw_video | 影片輸入。 |',
        '| models | 預訓練權重、checkpoint 與 ONNX，不提交大型模型。 |',
        '| asset | UI、字型、圖示與地圖資源；大型地圖不提交。 |',
        '| output | 離線、即時、訓練與比較結果，不提交。 |',
        '| tests | 本機驗證程式；目前根 .gitignore 排除，未更改此規則。 |',
        '| X-AnyLabeling、cvat | 外部標註軟體；不修改、不加入 Boom Git。 |',
        '', '更新索引：`python tools/update_file_index.py`（使用 boom 環境）。'])
    (ROOT/'doc/FILE_INDEX.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')
    sys.path[:0] = [str(ROOT/'src'), str(ROOT/'benchmark/ASTT')]
    from boom.pipelines.cli import build_arg_parser as offline_parser
    from boom.online.configuration import build_arg_parser as online_parser
    spec = importlib.util.spec_from_file_location('astt_runner_reference', ROOT/'benchmark/ASTT/runner.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sections = ['# 執行入口與完整設定', '',
        '從 Boom 根目錄執行。以下 CLI 表由目前 argparse 定義產生；絕對路徑以 `<Boom>` 表示專案根。ASTT 的資料／輸出根會再附加 astt/；修改 configs/*.yaml 不會改 ASTT 設定。', '',
        '## 常用入口', '', '```powershell', 'conda activate boom', 'python run_offline.py motion --video data/video/example.mp4 --no-map --display',
        'python run_offline.py yolo --video data/video/example.mp4 --no-map --display',
        'python run_realtime.py motion --display', 'python run_realtime.py yolo --display',
        'python tools/check_camera.py --frames 30', 'python tools/video_to_frames.py data/video/example.mp4 --interval 0.5',
        'python tools/split_dataset.py', 'python tools/train_yolo.py',
        'python benchmark/ASTT/train.py --experiment-name astt_first --epochs 30 --batch-size 1',
        'python benchmark/ASTT/test.py --checkpoint output/astt/astt_first/checkpoints/best.pth --experiment-name astt_test --batch-size 1', '```', '',
        '影片／模型路徑為示例，須替換成現有檔案。ASTT 訓練前準備正常 train 與獨立 val；測試需本機版權重。即時相機需原廠 PySpin SDK。YOLO 訓練的模型、資料集與超參數由互動提示選擇，細節見 [TOOLS.md](TOOLS.md)。', '',
        '## Boom YAML 設定', '', '讀取順序與作用見 [configs/README.md](../configs/README.md)。CLI 值可覆蓋對應設定；非所有 YAML 值都由每個 runner 使用，請以 CLI 和 loader 實作為準。保留目前使用者設定，這裡只列出內容。', '']
    for path in sorted((ROOT/'configs').glob('*.yaml')):
        sections.extend(['### '+path.name, '', '```yaml', path.read_text(encoding='utf-8-sig').rstrip(), '```', ''])
    for title, parser in [('Boom 離線 CLI', offline_parser()), ('Boom 即時 CLI', online_parser()),
                          ('ASTT 訓練 CLI', module.parser(True)), ('ASTT 測試 CLI', module.parser(False))]:
        sections.extend(['## '+title, '', cli_table(parser), ''])
    for filename in ('check_camera.py', 'video_to_frames.py', 'split_dataset.py', 'benchmark_matchers.py', 'train_yolo.py'):
        tree = ast.parse((ROOT/'tools'/filename).read_text(encoding='utf-8-sig'))
        sections.extend(['## tools/'+filename+' 設定', '', '| 參數 | 預設／形式 | 用途／限制 |', '|---|---|---|'])
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != 'add_argument':
                continue
            names = [ast.literal_eval(a) for a in node.args if isinstance(a, ast.Constant)]
            values = {kw.arg: kw.value for kw in node.keywords}
            default = ast.unparse(values['default']) if 'default' in values else '未指定'
            help_text = ast.literal_eval(values['help']) if 'help' in values and isinstance(values['help'], ast.Constant) else ', '.join(names)
            if 'action' in values:
                default += '; action=' + ast.unparse(values['action'])
            if 'nargs' in values:
                default += '; nargs=' + ast.unparse(values['nargs'])
            if 'choices' in values:
                help_text += '; choices=' + ast.unparse(values['choices'])
            sections.append(f'| `{", ".join(names)}` | `{default}` | {help_text.replace("|", "/")} |')
        if filename == 'train_yolo.py':
            sections.extend(['', 'YOLO 訓練不是以 argparse 設定全部超參數；模型與版本資料集由互動選單挑選，訓練參數定義在 main() 的 model.train() 呼叫：', '', '```python'])
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'train':
                    sections.append(ast.unparse(node))
            sections.extend(['```'])
        sections.append('')
    sections.extend(['## ASTT 抽幀設定', '', '| 參數 | 預設 | 用途 |', '|---|---|---|',
        '| video | 必填 | 來源影片路徑。 |', '| --fps | 10 | 抽幀 FPS，不得高於來源 FPS。 |',
        '| --name | 影片 stem | 片段資料夾名稱，已存在則拒絕覆寫。 |',
        '| --split | train | train、val 或 test。 |', '| --data-dir | <Boom>/data | 根目錄，會附加 astt/<split>。 |', '',
        '模型設定由 checkpoint 載入時，以權重內 model_config 為準；registration 必須與訓練一致。resume 要保持原總 epochs、每 epoch steps 和選模方式。threshold/min-area 等只影響候選區域，不是煙霧分類信心。更多格式和限制見 [ASTT 本機說明](../benchmark/ASTT/README_LOCAL.md)。'])
    (ROOT/'doc/RUN_SETTINGS.md').write_text('\n'.join(sections)+'\n', encoding='utf-8')
    print(f'Annotated {len(files)} Python files and regenerated reference documents.')


if __name__ == '__main__':
    main()
