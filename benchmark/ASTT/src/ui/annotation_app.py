# 檔案功能：Tkinter 影片播放、異常區段標記與匯出介面。
import base64
import copy
from datetime import datetime
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from ..annotations import (TARGET_FPS, ExportCancelled, Source, export_dataset, load_project,
                                   normalize_intervals, plan_split, sample_indices, save_project)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ASTT_ROOT = Path(__file__).resolve().parents[2]


class AnnotationApp:
    def __init__(self, root, project=None):
        self.root = root
        root.title('ASTT 影片標記與資料集準備')
        root.geometry('1180x900')
        root.minsize(980, 740)
        self.sources = []
        self.current = None
        self.frame_index = -1
        self.capture = None
        self.capture_next = -1
        self.pending = {}
        self.playing = False
        self.play_token = None
        self.busy = False
        self.closing = False
        self.cancel = threading.Event()
        self.messages = queue.Queue()
        self.project_path = Path(project).resolve() if project else (
            ASTT_ROOT / 'output' / 'annotations' / f'annotations_{datetime.now():%Y%m%d_%H%M%S_%f}.json')
        self.position = tk.IntVar(value=0)
        self.speed = tk.StringVar(value='1')
        self.origin = tk.StringVar()
        self.output_parent = tk.StringVar(value=str(ASTT_ROOT / 'data' / 'prepared'))
        self.dataset_name = tk.StringVar(value='artillery-shell')
        self.ratios = tk.StringVar(value='7:3')
        self.status = tk.StringVar(value='加入影片或影格資料夾後，標記你要偵測的變化區段。')
        self.info = tk.StringVar(value='未標記＝正常 0；異常區段＝1。每支看完按「完成此來源標記」。')
        self._build()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.bind('<space>', lambda e: self._key(e, self.toggle_play))
        root.bind('<Left>', lambda e: self._key(e, lambda: self.step(-1)))
        root.bind('<Right>', lambda e: self._key(e, lambda: self.step(1)))
        root.bind('<Shift-Left>', lambda e: self._key(e, lambda: self.jump(-1)))
        root.bind('<Shift-Right>', lambda e: self._key(e, lambda: self.jump(1)))
        root.bind('<Key-s>', lambda e: self._key(e, self.mark_start))
        root.bind('<Key-e>', lambda e: self._key(e, self.mark_end))
        self._pump()
        if project:
            self.sources = load_project(project)
            self.refresh_sources()
            self.status.set('已載入 30 FPS 標記時間軸；待標記來源請確認區段並按「完成此來源標記」。')

    def _build(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill='x')
        for text, command in [('加入影片', self.add_videos), ('加入影格資料夾', self.add_frames),
                              ('載入標記專案', self.open_project), ('另存標記專案', self.save_as)]:
            ttk.Button(top, text=text, command=command).pack(side='left', padx=3)
        ttk.Label(top, text='空白鍵播放／暫停　← →逐幀　S 起點／E 終點').pack(side='right')
        main = ttk.Panedwindow(self.root, orient='horizontal')
        main.pack(fill='both', expand=True, padx=8)
        left = ttk.Frame(main, padding=4)
        right = ttk.Frame(main, padding=4)
        main.add(left, weight=1)
        main.add(right, weight=4)
        ttk.Label(left, text='來源（整組分配到同一資料組）').pack(anchor='w')
        self.source_list = ttk.Treeview(left, columns=('state',), show='tree headings', selectmode='browse', height=14)
        self.source_list.heading('#0', text='影片／影格來源')
        self.source_list.column('#0', width=180)
        self.source_list.heading('state', text='狀態')
        self.source_list.column('state', width=60)
        self.source_list.pack(fill='both', expand=True)
        self.source_list.bind('<<TreeviewSelect>>', self.select_source)
        ttk.Button(left, text='移除選取來源', command=self.remove_source).pack(fill='x', pady=5)
        ttk.Label(left, text='原影片群組（同影片的片段用相同名稱）', wraplength=250).pack(anchor='w')
        self.origin_entry = ttk.Entry(left, textvariable=self.origin)
        self.origin_entry.pack(fill='x', pady=4)
        self.origin_entry.bind('<FocusOut>', lambda e: self.update_origin())
        self.origin_entry.bind('<Return>', lambda e: self.update_origin())
        ttk.Button(left, text='完成此來源標記', command=self.finish_source).pack(fill='x', pady=4)
        ttk.Label(left, text='沒有異常的來源也可直接按完成。\n同一影片的不同解析度／剪接片段請使用相同群組。', wraplength=260).pack(anchor='w', pady=5)
        self.canvas = tk.Canvas(right, bg='#171c23', highlightthickness=0, height=365)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda e: self.render_preview())
        self.timeline = tk.Canvas(right, height=26, bg='#e5e7eb', highlightthickness=0)
        self.timeline.pack(fill='x', pady=(5, 0))
        self.timeline.bind('<Configure>', lambda e: self.draw_timeline())
        self.timeline.bind('<Button-1>', self.timeline_seek)
        self.slider = tk.Scale(right, from_=0, to=1, orient='horizontal', showvalue=False,
                               variable=self.position, command=self.slider_seek)
        self.slider.pack(fill='x')
        playback = ttk.Frame(right)
        playback.pack(fill='x', pady=3)
        self.jump_buttons = []
        for text, command in [('−1 秒', lambda: self.jump(-1)), ('上一幀', lambda: self.step(-1)),
                              ('播放／暫停', self.toggle_play), ('下一幀', lambda: self.step(1)), ('＋1 秒', lambda: self.jump(1))]:
            button = ttk.Button(playback, text=text, command=command)
            button.pack(side='left', padx=2)
            if '秒' in text:
                self.jump_buttons.append(button)
        ttk.Label(playback, text='速度').pack(side='left', padx=(10, 3))
        ttk.Combobox(playback, textvariable=self.speed, values=('0.5', '1', '2', '4', '8'), width=4, state='readonly').pack(side='left')
        ttk.Label(right, textvariable=self.info).pack(anchor='w', pady=3)
        marks = ttk.Frame(right)
        marks.pack(fill='x')
        for text, command in [('異常開始（S）', self.mark_start), ('異常結束（E）', self.mark_end),
                              ('標記目前單幀', self.mark_single), ('取消待標起點', self.cancel_start), ('刪除選取區段', self.delete_interval)]:
            ttk.Button(marks, text=text, command=command).pack(side='left', padx=2)
        self.interval_list = ttk.Treeview(right, columns=('start', 'end'), show='headings', height=4, selectmode='browse')
        self.interval_list.heading('start', text='異常起點（包含此幀）')
        self.interval_list.heading('end', text='異常終點（包含此幀）')
        self.interval_list.pack(fill='x', pady=5)
        self.interval_list.bind('<Double-1>', self.go_interval)
        export = ttk.LabelFrame(self.root, text='自動分組與匯出', padding=8)
        export.pack(fill='x', padx=8, pady=6)
        ttk.Label(export, text='輸出位置').grid(row=0, column=0, sticky='w')
        ttk.Entry(export, textvariable=self.output_parent).grid(row=0, column=1, columnspan=5, sticky='ew', padx=5)
        ttk.Button(export, text='選擇', command=self.choose_output).grid(row=0, column=6)
        for column, text, variable, width in [(0, '資料集名稱', self.dataset_name, 14),
                                               (2, 'train,val 比例', self.ratios, 12)]:
            ttk.Label(export, text=text).grid(row=1, column=column, pady=6, sticky='w')
            ttk.Entry(export, textvariable=variable, width=width).grid(row=1, column=column + 1, padx=5, sticky='w')
        ttk.Label(export, text='標記與匯出固定 30 FPS').grid(row=1, column=4, columnspan=2, sticky='w', padx=5)
        export.columnconfigure(1, weight=1)
        controls = ttk.Frame(export)
        controls.grid(row=2, column=0, columnspan=7, sticky='ew')
        ttk.Button(controls, text='預覽分組', command=self.preview_split).pack(side='left', padx=3)
        ttk.Button(controls, text='產生資料集與 .npy', command=self.start_export).pack(side='left', padx=3)
        ttk.Button(controls, text='取消匯出', command=self.cancel.set).pack(side='left', padx=3)
        ttk.Label(controls, text='每次建立新版本；至少兩個獨立來源群組。').pack(side='right')
        ttk.Label(self.root, textvariable=self.status, wraplength=1120, padding=8).pack(fill='x')

    def _key(self, event, action):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox)):
            return
        action()
        return 'break'

    def error(self, error):
        messagebox.showerror('ASTT 標記工具', str(error), parent=self.root)

    def autosave(self):
        try:
            save_project(self.project_path, self.sources)
            self.status.set(f'標記已保存：{self.project_path}')
        except Exception as exc:
            self.error(exc)

    def add_paths(self, paths):
        if self.busy or not paths:
            return
        failures = []
        for path in paths:
            try:
                if any(Path(s.path).resolve() == Path(path).resolve() for s in self.sources):
                    continue
                source = Source.inspect(path)
                if any(s.name.casefold() == source.name.casefold() for s in self.sources):
                    source.name += '_' + source.id[:8]
                self.sources.append(source)
            except Exception as exc:
                failures.append(str(exc))
        self.refresh_sources()
        self.autosave()
        if failures:
            self.error('\n'.join(failures))

    def add_videos(self):
        self.pause()
        if not self.busy:
            self.add_paths(filedialog.askopenfilenames(parent=self.root, title='加入影片', initialdir=PROJECT_ROOT / 'data' / 'video',
                                                     filetypes=[('影片', '*.mp4 *.avi *.mov *.mkv *.m4v'), ('所有檔案', '*.*')]))

    def add_frames(self):
        self.pause()
        if self.busy:
            return
        path = filedialog.askdirectory(parent=self.root, title='選擇已是 30 FPS 的影格資料夾，或包含多支影片的父資料夾', initialdir=PROJECT_ROOT / 'data' / 'picture')
        if path:
            folder = Path(path)
            if any(p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'} for p in folder.iterdir()):
                self.add_paths([folder])
            else:
                self.add_paths(sorted(p for p in folder.iterdir() if p.is_dir() and any(f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'} for f in p.iterdir())))

    def refresh_sources(self):
        selected = self.current.id if self.current else None
        self.source_list.delete(*self.source_list.get_children())
        for source in self.sources:
            self.source_list.insert('', 'end', iid=source.id, text=source.name, values=('完成' if source.reviewed else '待標記',))
        if self.sources:
            selected = selected if any(s.id == selected for s in self.sources) else self.sources[0].id
            self.source_list.selection_set(selected)

    def select_source(self, event=None):
        selection = self.source_list.selection()
        if not selection:
            return
        source = next(s for s in self.sources if s.id == selection[0])
        if self.current is source:
            self.refresh_intervals()
            return
        self.pause()
        if self.capture:
            self.capture.release()
        self.current = source
        self.source_indices = sample_indices(source)
        self.origin.set(source.origin)
        self.capture = cv2.VideoCapture(source.path) if source.kind == 'video' else None
        self.capture_next = -1
        self.frame_index = -1
        self.slider.configure(to=source.frame_count - 1)
        self.refresh_intervals()
        self.seek(0)

    def remove_source(self):
        if self.busy or not self.current:
            return
        self.pause()
        self.sources.remove(self.current)
        self.pending.pop(self.current.id, None)
        if self.capture:
            self.capture.release()
        self.capture = None
        self.current = None
        self.frame_index = -1
        self.raw_preview = None
        self.canvas.delete('all')
        self.interval_list.delete(*self.interval_list.get_children())
        self.origin.set('')
        self.info.set('加入或選擇來源，再標記異常起訖。')
        self.refresh_sources()
        self.draw_timeline()
        self.autosave()

    def update_origin(self):
        if self.current and not self.busy:
            value = self.origin.get().strip() or self.current.name
            if self.current.origin != value:
                self.current.origin = value
                self.autosave()

    def seek(self, index):
        if not self.current:
            return
        index = max(0, min(self.current.frame_count - 1, index))
        if index == self.frame_index:
            return
        if self.capture is not None:
            source_index = int(self.source_indices[index])
            if self.frame_index >= 0 and source_index == int(self.source_indices[self.frame_index]):
                ok, image = True, self.raw_preview
            else:
                if source_index != self.capture_next:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
                ok, image = self.capture.read()
                self.capture_next = source_index + 1
            if not ok:
                self.pause()
                self.status.set(f'第 {index + 1} 幀無法讀取。')
                return
        else:
            image = cv2.imread(str(Path(self.current.path) / self.current.frame_files[index]))
            if image is None:
                self.pause()
                self.status.set('無法讀取目前影格。')
                return
        self.frame_index = index
        self.position.set(index)
        self.raw_preview = image
        self.render_preview()
        self.update_info()
        self.draw_timeline()

    def slider_seek(self, value):
        index = int(float(value))
        if index != self.frame_index:
            self.pause()
            self.seek(index)

    def render_preview(self):
        image = getattr(self, 'raw_preview', None)
        if image is None:
            return
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        h, w = image.shape[:2]
        factor = min(width / w, height / h)
        resized = cv2.resize(image, (max(1, round(w * factor)), max(1, round(h * factor))))
        ok, encoded = cv2.imencode('.png', resized)
        if ok:
            self.photo = tk.PhotoImage(data=base64.b64encode(encoded.tobytes()), master=self.root)
            self.canvas.delete('all')
            self.canvas.create_image(width // 2, height // 2, image=self.photo)

    def position_text(self, index):
        text = f'第 {index + 1} 幀'
        if self.current:
            seconds = index / TARGET_FPS
            text += f'（{int(seconds // 60):02d}:{seconds % 60:05.2f}）'
        return text

    def update_info(self):
        if self.current is None:
            return
        abnormal = any(a <= self.frame_index <= b for a, b in self.current.intervals)
        pending = self.pending.get(self.current.id)
        mark = f'　待標起點：{self.position_text(pending)}' if pending is not None else ''
        self.info.set(f'{self.position_text(self.frame_index)} / {self.current.frame_count}　'
                      f'{"異常 1" if abnormal else "正常 0"}　'
                      f'{"已完成" if self.current.reviewed else "待完成標記"}　30 FPS{mark}')

    def draw_timeline(self):
        self.timeline.delete('all')
        if not self.current:
            return
        width = max(1, self.timeline.winfo_width())
        count = self.current.frame_count
        for a, b in self.current.intervals:
            self.timeline.create_rectangle(a / count * width, 0, (b + 1) / count * width, 26, fill='#ef6b6b', outline='')
        pending = self.pending.get(self.current.id)
        if pending is not None:
            x = pending / count * width
            self.timeline.create_line(x, 0, x, 26, fill='#eab308', width=3)
        x = max(0, self.frame_index) / count * width
        self.timeline.create_line(x, 0, x, 26, fill='#2563eb', width=2)

    def timeline_seek(self, event):
        if self.current:
            self.pause()
            self.seek(int(event.x / max(1, self.timeline.winfo_width()) * self.current.frame_count))

    def step(self, count):
        self.pause()
        self.seek(self.frame_index + count)

    def jump(self, seconds):
        self.step(round(seconds * TARGET_FPS) if self.current else 0)

    def pause(self):
        self.playing = False
        if self.play_token is not None:
            self.root.after_cancel(self.play_token)
            self.play_token = None

    def toggle_play(self):
        if self.playing:
            self.pause()
        elif self.current:
            if self.frame_index == self.current.frame_count - 1:
                self.seek(0)
            self.playing = True
            self.play_token = self.root.after(self.play_delay(), self.tick)

    def play_delay(self, elapsed_ms=0.):
        # Display each 30 FPS timeline frame; slow decoding delays playback instead of skipping.
        return max(1, round(1000 / (TARGET_FPS * float(self.speed.get())) - elapsed_ms))

    def tick(self):
        self.play_token = None
        if self.playing and self.current:
            started = time.perf_counter()
            self.seek(self.frame_index + 1)
            if self.frame_index >= self.current.frame_count - 1:
                self.pause()
            elif self.playing:
                self.play_token = self.root.after(self.play_delay((time.perf_counter() - started) * 1000), self.tick)

    def mark_start(self):
        if self.current and not self.busy and self.frame_index >= 0:
            self.pending[self.current.id] = self.frame_index
            self.current.reviewed = False
            self.refresh_sources()
            self.autosave()
            self.update_info()
            self.draw_timeline()

    def mark_end(self):
        if self.current and not self.busy:
            start = self.pending.get(self.current.id)
            if start is None:
                self.status.set('請先按「異常開始」。')
            elif self.frame_index < start:
                self.status.set('終點不能早於起點；可先取消待標起點再重新標記。')
            else:
                self.current.intervals.append((start, self.frame_index))
                self.pending.pop(self.current.id)
                self.marks_changed()

    def mark_single(self):
        if self.current and not self.busy and self.frame_index >= 0:
            self.current.intervals.append((self.frame_index, self.frame_index))
            self.marks_changed()

    def cancel_start(self):
        if self.current and not self.busy:
            self.pending.pop(self.current.id, None)
            self.update_info()
            self.draw_timeline()

    def marks_changed(self):
        self.current.intervals = normalize_intervals(self.current.intervals, self.current.frame_count)
        self.current.reviewed = False
        self.refresh_sources()
        self.refresh_intervals()
        self.update_info()
        self.draw_timeline()
        self.autosave()

    def refresh_intervals(self):
        self.interval_list.delete(*self.interval_list.get_children())
        if self.current:
            for i, (a, b) in enumerate(self.current.intervals):
                self.interval_list.insert('', 'end', iid=str(i), values=(self.position_text(a), self.position_text(b)))

    def delete_interval(self):
        selection = self.interval_list.selection()
        if self.current and selection and not self.busy:
            self.current.intervals.pop(int(selection[0]))
            self.marks_changed()

    def go_interval(self, event=None):
        selection = self.interval_list.selection()
        if self.current and selection:
            self.pause()
            self.seek(self.current.intervals[int(selection[0])][0])

    def finish_source(self):
        if self.current and not self.busy:
            self.pause()
            if self.current.id in self.pending:
                self.status.set('請先標記異常終點或取消待標起點。')
                return
            self.update_origin()
            self.current.reviewed = True
            self.refresh_sources()
            self.update_info()
            self.autosave()

    def open_project(self):
        self.pause()
        if self.busy:
            return
        path = filedialog.askopenfilename(parent=self.root, title='載入本工具保存的標記 JSON', initialdir=self.project_path.parent, filetypes=[('標記專案', '*.json')])
        if path:
            try:
                sources = load_project(path)
                if self.capture:
                    self.capture.release()
                self.capture = None
                self.sources, self.current = sources, None
                self.raw_preview = None
                self.pending.clear()
                self.canvas.delete('all')
                self.interval_list.delete(*self.interval_list.get_children())
                self.timeline.delete('all')
                self.origin.set('')
                self.info.set('加入或選擇來源，再標記異常起訖。')
                self.project_path = Path(path).resolve()
                self.refresh_sources()
                self.status.set(f'已載入 30 FPS 時間軸：{self.project_path}；待標記來源請確認區段並按完成。')
            except Exception as exc:
                self.error(exc)

    def save_as(self):
        if self.busy:
            return
        path = filedialog.asksaveasfilename(parent=self.root, title='另存新的標記專案', initialdir=self.project_path.parent,
                                           defaultextension='.json', filetypes=[('標記專案', '*.json')], confirmoverwrite=False)
        if path:
            path = Path(path).resolve()
            if path.exists() and path != self.project_path:
                self.error('此檔案已存在，請使用新的名稱。')
                return
            self.project_path = path
            self.autosave()

    def choose_output(self):
        if not self.busy:
            path = filedialog.askdirectory(parent=self.root, title='資料集版本的存放位置', initialdir=ASTT_ROOT)
            if path:
                self.output_parent.set(path)

    def options(self):
        self.update_origin()
        if self.pending:
            raise ValueError('還有未完成的異常起點，請標記終點或取消。')
        ratio_text = self.ratios.get().replace('，', ',').replace('：', ',').replace(':', ',')
        ratios = tuple(float(v.strip()) for v in ratio_text.split(','))
        return dict(output_parent=self.output_parent.get(), name=self.dataset_name.get(), ratios=ratios)

    def report_text(self, report):
        lines = []
        for split in ('train', 'val'):
            row = report['summary'][split]
            lines.append(f'{split}：{report["group_counts"][split]} 群組、{row["frames"]} 幀、異常 {row["abnormal_frames"]} 幀、正常訓練窗口 {row["normal_windows"]}')
        lines += report['warnings']
        lines.append('每支來源完整保留在同一組；訓練只使用連續五幀皆正常的窗口。')
        return '\n'.join(lines)

    def preview_split(self):
        if self.busy:
            return
        try:
            options = self.options()
            report = plan_split(self.sources, ratios=options['ratios'])
            detail = '\n'.join(f'{s.name} → {report["assignments"][s.id]}' for s in self.sources)
            messagebox.showinfo('分組預覽', self.report_text(report) + '\n\n' + detail, parent=self.root)
        except Exception as exc:
            self.error(exc)

    def start_export(self):
        if self.busy:
            return
        self.pause()
        try:
            options = self.options()
            if not options['output_parent'].strip():
                raise ValueError('請指定輸出位置。')
            plan_split(self.sources, ratios=options['ratios'])
            save_project(self.project_path, self.sources)
        except Exception as exc:
            self.error(exc)
            return
        sources = copy.deepcopy(self.sources)
        self.busy = True
        self.cancel.clear()
        self.status.set('正在產生資料集；可繼續查看影片，或按取消匯出。')
        def work():
            try:
                report = export_dataset(sources, **options, cancel=self.cancel,
                                        progress=lambda text: self.messages.put(('progress', text)))
                self.messages.put(('done', report))
            except ExportCancelled as exc:
                self.messages.put(('cancel', str(exc)))
            except Exception as exc:
                self.messages.put(('error', str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def _pump(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == 'progress':
                    self.status.set(payload)
                else:
                    self.busy = False
                    if kind == 'done':
                        self.status.set(f'資料集已產生：{payload["output"]}')
                        if not self.closing:
                            messagebox.showinfo('匯出完成', self.report_text(payload) + f'\n\n資料集：{payload["output"]}\n操作指令已寫入此目錄 README.md。', parent=self.root)
                    elif kind == 'cancel':
                        self.status.set(payload)
                    else:
                        self.status.set(f'匯出失敗：{payload}')
                        if not self.closing:
                            self.error(payload)
        except queue.Empty:
            pass
        if self.closing and not self.busy:
            self.root.destroy()
        else:
            self.pump_token = self.root.after(100, self._pump)

    def close(self):
        self.pause()
        if self.capture:
            self.capture.release()
            self.capture = None
        if self.busy:
            self.closing = True
            self.cancel.set()
            self.root.withdraw()
        else:
            self.root.after_cancel(self.pump_token)
            self.root.destroy()


def launch(project=None):
    root = tk.Tk()
    AnnotationApp(root, project)
    root.mainloop()
