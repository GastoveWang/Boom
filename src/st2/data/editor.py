# 檔案功能：ST2 的影片區段與手動逐幀框選介面。
"""Tkinter interval and per-frame BBox editor; independent of original ASTT."""
import copy
from dataclasses import asdict
from datetime import datetime
import math
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from st2.data.annotations import (ANNOTATIONS_ROOT, DATASET_ROOT, PROJECT_ROOT,
    SmokeProject, SmokeSource, ExportCancelled, export_dataset, load_project,
    merge_intervals, plan_split, save_project)


class SmokeAnnotationApp:
    def __init__(self, root, project=None, output_parent=DATASET_ROOT, name='st2',
                 ratios=(7., 3.), history_frames=4, seed=2026):
        self.root = root
        root.title('ST2 逐幀區域標註')
        root.geometry(f'{min(1380, root.winfo_screenwidth()-80)}x{min(900, root.winfo_screenheight()-100)}')
        root.minsize(1020, 660)
        self.project = load_project(project) if project else SmokeProject()
        self.project_path = Path(project).resolve() if project else self._new_project_path()
        self.current = None
        self.frame_index = -1
        self.capture = None
        self.capture_next = 0
        self.pil_image = None
        self.photo = None
        self.selected_box = None
        self.hovered_box = None
        self.drag = None
        self.pan_start = None
        self.force_new = False
        self.zoom = 1.
        self.offset = (0., 0.)
        self.scale = 1.
        self.playing = False
        self.play_token = None
        self.busy = False
        self.closing = False
        self.save_failed = False
        self.saved_state = asdict(self.project)
        self.undo_stack = []
        self.cancel = threading.Event()
        self.messages = queue.Queue()
        self.position = tk.DoubleVar(value=0)
        self.start_frame = tk.IntVar(value=1)
        self.end_frame = tk.IntVar(value=1)
        self.origin = tk.StringVar()
        self.output_parent = tk.StringVar(value=str(output_parent))
        self.dataset_name = tk.StringVar(value=name)
        self.ratios = tk.StringVar(value=f'{ratios[0]:g}:{ratios[1]:g}')
        self.history_frames = tk.IntVar(value=history_frames)
        self.seed = seed
        self.info = tk.StringVar(value='加入影片或連續圖片，選擇時間區段後逐幀畫框。')
        self.status = tk.StringVar(value='拖曳空白處新增框、角落縮放、邊線平移；按儲存或 Ctrl＋S 保存修改。')
        self.path_info = tk.StringVar(value=str(self.project_path))
        self._build()
        self.origin.trace_add('write', lambda *args: self.update_save_state())
        self.start_frame.trace_add('write', lambda *args: self.draw_timeline())
        self.end_frame.trace_add('write', lambda *args: self.draw_timeline())
        self.refresh_sources()
        root.protocol('WM_DELETE_WINDOW', self.close)
        for key, action in [('<Left>', lambda: self.step(-1)), ('<Right>', lambda: self.step(1)),
                            ('<Shift-Left>', lambda: self.step(-round(self.current.fps) if self.current else -1)),
                            ('<Shift-Right>', lambda: self.step(round(self.current.fps) if self.current else 1)),
                            ('<space>', self.toggle_play), ('<Key-s>', self.mark_start), ('<Key-e>', self.mark_end),
                            ('<Key-n>', self.next_missing), ('<Delete>', self.delete_box)]:
            root.bind(key, lambda event, action=action: self.key(event, action))
        root.bind('<Control-s>', lambda e: self.save())
        root.bind('<Control-z>', lambda e: self.key(e, self.undo))
        self.pump_token = root.after(100, self.pump)
        if self.project.sources:
            self.source_list.selection_set(self.project.sources[0].id)
            self.select_source()
        self.update_save_state()

    def _new_project_path(self):
        return ANNOTATIONS_ROOT / f'st2_{datetime.now():%Y%m%d_%H%M%S_%f}.json'

    def _build(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill='x')
        for title, action in [('加入影片', self.add_videos), ('加入圖片資料夾', self.add_frames),
                              ('新專案', self.new_project), ('載入專案', self.open_project), ('儲存 Ctrl＋S', self.save), ('另存專案', self.save_as)]:
            ttk.Button(top, text=title, command=action).pack(side='left', padx=3)
        ttk.Label(top, text='同位套用 → 逐幀微調 · 不需物體類別').pack(side='right', padx=8)
        ttk.Label(self.root, textvariable=self.path_info, foreground='#64748b').pack(fill='x', padx=12)
        panes = ttk.Panedwindow(self.root, orient='horizontal')
        panes.pack(fill='both', expand=True, padx=8, pady=6)
        left = ttk.Frame(panes, width=245, padding=5)
        center = ttk.Frame(panes, padding=4)
        right = ttk.Frame(panes, width=190, padding=5)
        panes.add(left, weight=0)
        panes.add(center, weight=1)
        panes.add(right, weight=0)
        ttk.Label(left, text='來源與缺框數').pack(anchor='w')
        self.source_list = ttk.Treeview(left, columns=('state',), show='tree headings', height=6, selectmode='browse')
        self.source_list.heading('#0', text='來源')
        self.source_list.heading('state', text='框')
        self.source_list.column('#0', width=145)
        self.source_list.column('state', width=70, stretch=False)
        self.source_list.pack(fill='x', pady=4)
        self.source_list.bind('<<TreeviewSelect>>', lambda e: self.select_source())
        ttk.Button(left, text='移除來源', command=self.remove_source).pack(fill='x')
        ttk.Label(left, text='原始來源群組（同群組不跨 train / val）', wraplength=230).pack(anchor='w', pady=(10, 2))
        entry = ttk.Entry(left, textvariable=self.origin)
        entry.pack(fill='x')
        entry.bind('<Return>', lambda e: self.update_origin())
        entry.bind('<FocusOut>', lambda e: self.update_origin())
        ttk.Separator(left).pack(fill='x', pady=10)
        ttk.Label(left, text='時間區段（起訖都包含）').pack(anchor='w')
        self.interval_list = ttk.Treeview(left, columns=('span',), show='tree headings', height=7, selectmode='browse')
        self.interval_list.heading('#0', text='類型')
        self.interval_list.heading('span', text='標註時間（秒）')
        self.interval_list.column('#0', width=90, stretch=False)
        self.interval_list.column('span', width=130)
        self.interval_list.pack(fill='both', expand=True, pady=4)
        self.interval_list.bind('<Double-1>', lambda e: self.go_interval())
        ttk.Button(left, text='刪除選取區段', command=self.delete_interval).pack(fill='x')
        ttk.Label(left, text='區段內每幀須有框；不確定的影格可略過。區段外會匯出為無目標。', wraplength=230, foreground='#64748b').pack(fill='x', pady=12)
        ttk.Label(center, textvariable=self.info, wraplength=800).pack(fill='x', pady=(0, 5))
        self.canvas = tk.Canvas(center, background='#17212d', highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda e: self.render())
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.motion)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        self.canvas.bind('<Motion>', self.hover)
        self.canvas.bind('<Leave>', lambda e: self.canvas.configure(cursor='arrow') if not self.drag else None)
        self.canvas.bind('<MouseWheel>', self.wheel)
        self.canvas.bind('<ButtonPress-2>', self.pan_press)
        self.canvas.bind('<B2-Motion>', self.pan_motion)
        self.timeline = tk.Canvas(center, height=76, background='#f8fafc', highlightthickness=0, cursor='hand2')
        self.timeline.pack(fill='x', pady=(4, 0))
        self.timeline.bind('<Configure>', lambda e: self.draw_timeline())
        self.timeline.bind('<Button-1>', self.timeline_seek)
        self.timeline.bind('<B1-Motion>', self.timeline_seek)
        self.timeline.bind('<Motion>', self.timeline_hover)
        self.slider = ttk.Scale(center, from_=0, to=1, variable=self.position, command=self.slider_seek)
        self.slider.pack(fill='x', pady=5)
        navigation = ttk.Frame(center)
        navigation.pack(fill='x')
        for text, action in [('◀ 前一幀', lambda: self.step(-1)), ('播放／暫停', self.toggle_play),
                              ('後一幀 ▶', lambda: self.step(1)), ('下一缺框幀', self.next_missing), ('適合視窗', self.fit_view)]:
            ttk.Button(navigation, text=text, command=action).pack(side='left', padx=2)
        span = ttk.Frame(center)
        span.pack(fill='x', pady=(6, 0))
        ttk.Button(span, text='設起點 S', command=self.mark_start).pack(side='left')
        ttk.Entry(span, textvariable=self.start_frame, width=7).pack(side='left', padx=3)
        ttk.Button(span, text='設終點 E', command=self.mark_end).pack(side='left')
        ttk.Entry(span, textvariable=self.end_frame, width=7).pack(side='left', padx=3)
        ttk.Button(span, text='加入有物體區段', command=self.add_interval).pack(side='left', padx=3)
        ttk.Button(span, text='略過此區段', command=self.ignore_interval).pack(side='left')
        ttk.Label(right, text='目前影格的 BBox').pack(anchor='w')
        self.box_list = tk.Listbox(right, height=8, exportselection=False)
        self.box_list.pack(fill='x', pady=4)
        self.box_list.bind('<<ListboxSelect>>', self.select_box)
        for title, action in [('畫新框', self.new_box), ('套用目前框到區段', self.seed_interval),
                              ('刪除選取框', self.delete_box), ('撤銷上次修改', self.undo), ('略過目前影格', self.ignore_frame)]:
            ttk.Button(right, text=title, command=action).pack(fill='x', pady=3)
        ttk.Label(right, text='空白處拖曳：新增框\n角落拖曳：調整形狀\n邊線拖曳：平移框\n框內也能畫新框\nCtrl＋拖曳：強制新增\n滾輪縮放 · 中鍵平移\n\n← →：逐幀\nShift＋← →：移動一秒\nN：下一缺框幀\nDelete：刪框\nCtrl＋Z：撤銷\nCtrl＋S：儲存', wraplength=175, foreground='#64748b').pack(fill='x', pady=12)
        export = ttk.LabelFrame(self.root, text='資料集版本 · 依來源群組分配', padding=7)
        export.pack(fill='x', padx=8, pady=(2, 6))
        export.columnconfigure(1, weight=1)
        ttk.Label(export, text='輸出位置').grid(row=0, column=0, sticky='w')
        ttk.Entry(export, textvariable=self.output_parent).grid(row=0, column=1, columnspan=5, sticky='ew', padx=5)
        ttk.Button(export, text='選擇', command=self.choose_output).grid(row=0, column=6)
        options = ttk.Frame(export)
        options.grid(row=1, column=0, columnspan=7, sticky='ew', pady=(6, 0))
        for label, variable, width in [('名稱', self.dataset_name, 16), ('train:val', self.ratios, 7), ('歷史幀數', self.history_frames, 5)]:
            ttk.Label(options, text=label).pack(side='left', padx=(6, 2))
            ttk.Entry(options, textvariable=variable, width=width).pack(side='left')
        ttk.Button(options, text='預覽分組', command=self.preview_split).pack(side='left', padx=8)
        ttk.Button(options, text='產生資料集', command=self.start_export).pack(side='left')
        ttk.Button(options, text='取消匯出', command=self.cancel.set).pack(side='left', padx=6)
        ttk.Label(self.root, textvariable=self.status, anchor='w', wraplength=1300).pack(fill='x', padx=10, pady=(0, 6))

    def key(self, event, action):
        if isinstance(event.widget, (ttk.Entry, tk.Entry, ttk.Combobox, tk.Text)) or self.busy:
            return
        action()
        return 'break'

    def error(self, error):
        self.status.set(str(error))
        messagebox.showerror('Smoke Heatmap 標註', str(error), parent=self.root)

    def project_state(self):
        state = asdict(self.project)
        if self.current:
            for row in state['sources']:
                if row['id'] == self.current.id:
                    row['origin'] = self.origin.get().strip()
        return state

    def update_save_state(self):
        self.dirty = self.project_state() != self.saved_state
        state = '未儲存修改' if self.dirty else '已儲存' if self.project_path.exists() else '尚無修改'
        self.path_info.set(f'{state} · {self.project_path}')
        self.root.title(f'{"* " if self.dirty else ""}ST2 逐幀區域標註')

    def ensure_saved(self, action):
        if self.drag:
            self.status.set('請先放開滑鼠完成目前框的調整，再儲存或關閉。')
            return False
        self.update_save_state()
        if not self.dirty:
            return True
        if self.busy:
            self.status.set('仍有未儲存修改，請等待匯出結束後儲存。')
            return False
        if not messagebox.askyesno('尚有未儲存修改', f'必須先儲存才能{action}。\n按「是」儲存並繼續，按「否」留在目前專案。', parent=self.root):
            return False
        return bool(self.save())

    def save(self):
        if self.busy:
            return False
        if self.drag:
            self.status.set('請先放開滑鼠完成目前框的調整，再儲存。')
            return False
        try:
            if self.current:
                if not self.origin.get().strip():
                    raise ValueError('原始來源群組不可空白；填寫後才能儲存。')
                self.current.origin = self.origin.get().strip()
            save_project(self.project_path, self.project)
            self.save_failed = False
            self.saved_state = asdict(self.project)
            self.update_save_state()
            self.status.set('標註專案已儲存。')
            return True
        except Exception as exc:
            self.save_failed = True
            self.update_save_state()
            self.error(exc)
            return False

    def add_videos(self):
        if not self.busy:
            self.add_paths(filedialog.askopenfilenames(parent=self.root, title='加入影片', initialdir=PROJECT_ROOT / 'data/video',
                filetypes=[('影片', '*.mp4 *.MP4 *.mov *.MOV *.avi *.mkv'), ('所有檔案', '*.*')]))

    def add_frames(self):
        if self.busy:
            return
        path = filedialog.askdirectory(parent=self.root, title='選擇直接包含連續圖片的資料夾', initialdir=PROJECT_ROOT / 'data')
        if path:
            fps = simpledialog.askfloat('圖片序列 FPS', '這個圖片序列的實際幀率：', initialvalue=30., minvalue=.001, parent=self.root)
            if fps is not None:
                self.add_paths([path], fps)

    def add_paths(self, paths, images_fps=30.):
        if self.busy:
            return
        added = None
        for path in paths:
            try:
                canonical = str(Path(path).resolve()).casefold()
                if any(str(Path(s.path).resolve()).casefold() == canonical for s in self.project.sources):
                    raise ValueError('這個來源已經加入。')
                added = SmokeSource.inspect(path, images_fps)
                self.project.sources.append(added)
            except Exception as exc:
                self.error(exc)
        self.refresh_sources()
        if added:
            self.update_save_state()
            self.source_list.selection_set(added.id)
            self.select_source()

    def refresh_sources(self):
        selection = self.current.id if self.current else None
        self.source_list.delete(*self.source_list.get_children())
        for source in self.project.sources:
            missing = len(source.missing_boxes())
            self.source_list.insert('', 'end', iid=source.id, text=source.name,
                values=(f'缺 {missing} 幀' if missing else '有框' if any(source.boxes.values()) else '無框',))
        if selection and self.source_list.exists(selection):
            self.source_list.selection_set(selection)

    def select_source(self):
        if self.busy or not self.source_list.selection():
            return
        source = next(s for s in self.project.sources if s.id == self.source_list.selection()[0])
        if self.current is source:
            return
        self.pause()
        if not self.update_origin():
            self.source_list.selection_set(self.current.id)
            return
        self.clear_current()
        self.current = source
        self.origin.set(source.origin)
        if source.kind == 'video':
            self.capture = cv2.VideoCapture(source.path)
        self.slider.configure(to=max(1, source.frame_count - 1))
        self.start_frame.set(1)
        self.end_frame.set(1)
        self.seek(0)
        self.refresh_intervals()

    def clear_current(self):
        if self.capture:
            self.capture.release()
        self.capture = None
        self.capture_next = 0
        self.current = None
        self.frame_index = -1
        self.pil_image = None
        self.photo = None
        self.drag = None
        self.selected_box = None
        self.hovered_box = None
        self.undo_stack = []
        self.zoom = 1.
        self.origin.set('')
        self.canvas.delete('all')
        self.draw_timeline()
        self.box_list.delete(0, 'end')
        self.interval_list.delete(*self.interval_list.get_children())
        self.info.set('加入或選擇來源，再選時間區段並逐幀畫框。')

    def remove_source(self):
        if self.busy or not self.current:
            return
        if not messagebox.askyesno('移除來源', '從本專案移除此來源及標註？原始檔案會保留。', parent=self.root):
            return
        self.pause()
        self.project.sources.remove(self.current)
        self.clear_current()
        self.refresh_sources()
        self.update_save_state()

    def update_origin(self):
        if self.current and not self.busy:
            value = self.origin.get().strip()
            if not value:
                self.status.set('原始來源群組不可空白，請填寫後再切換來源或匯出。')
                return False
            if value != self.current.origin:
                self.current.origin = value
                self.update_save_state()
        return True

    def seek(self, index):
        if not self.current or self.busy:
            return
        index = max(0, min(self.current.frame_count - 1, int(index)))
        if index == self.frame_index:
            return
        self.drag = None
        if self.capture is not None:
            if index != self.capture_next:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = self.capture.read()
            self.capture_next = index + 1
        else:
            image = cv2.imread(str(Path(self.current.path) / self.current.frame_files[index]))
            ok = image is not None
        if not ok:
            self.pause()
            self.error(f'第 {index + 1} 幀無法讀取。')
            return
        self.frame_index = index
        self.position.set(index)
        self.selected_box = None
        self.hovered_box = None
        self.canvas.configure(cursor='crosshair')
        previous_size = self.pil_image.size if self.pil_image else None
        self.pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if previous_size != self.pil_image.size:
            self.zoom = 1.
        self.render()
        self.update_boxes()

    def slider_seek(self, value):
        index = int(float(value))
        if index != self.frame_index:
            self.pause()
            self.seek(index)

    def step(self, count):
        self.pause()
        self.seek(self.frame_index + count)

    def pause(self):
        self.playing = False
        if self.play_token:
            self.root.after_cancel(self.play_token)
            self.play_token = None

    def toggle_play(self):
        if self.busy or not self.current:
            return
        if self.playing:
            self.pause()
            return
        if self.frame_index == self.current.frame_count - 1:
            self.seek(0)
        self.playing = True
        self.tick()

    def tick(self):
        self.play_token = None
        if not self.playing or self.busy or not self.current:
            return
        started = time.perf_counter()
        self.seek(self.frame_index + 1)
        if self.frame_index == self.current.frame_count - 1:
            self.pause()
        elif self.playing:
            delay = max(1, round(1000 / self.current.fps - (time.perf_counter()-started)*1000))
            self.play_token = self.root.after(delay, self.tick)

    def render(self):
        self.canvas.delete('all')
        if self.pil_image is None:
            self.canvas.create_text(max(1, self.canvas.winfo_width()) / 2, max(1, self.canvas.winfo_height()) / 2,
                text='加入來源後，在這裡逐幀畫 BBox', fill='#a7b9c9', font=('Microsoft JhengHei', 14))
            return
        width, height = self.pil_image.size
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.scale = min(cw / width, ch / height) * self.zoom
        if self.zoom == 1:
            self.offset = ((cw - width * self.scale) / 2, (ch - height * self.scale) / 2)
        ox, oy = self.offset
        x1, y1 = max(0, math.floor(-ox / self.scale)), max(0, math.floor(-oy / self.scale))
        x2, y2 = min(width, math.ceil((cw - ox) / self.scale)), min(height, math.ceil((ch - oy) / self.scale))
        if x2 > x1 and y2 > y1:
            # Resize only the visible crop; zooming a 4K source does not create a giant bitmap.
            crop = self.pil_image.crop((x1, y1, x2, y2))
            size = (max(1, round((x2-x1)*self.scale)), max(1, round((y2-y1)*self.scale)))
            self.photo = ImageTk.PhotoImage(crop.resize(size, Image.Resampling.BILINEAR), master=self.root)
            self.canvas.create_image(ox+x1*self.scale, oy+y1*self.scale, image=self.photo, anchor='nw')
        boxes = self.drag['preview'] if self.drag else (
            self.current.boxes.get(str(self.frame_index), []) if self.current.labels()[self.frame_index] != -1 else [])
        for index, box in enumerate(boxes):
            points = self.box_canvas(box)
            color = '#fbbf24' if index in (self.selected_box, self.hovered_box) else '#4ade80'
            self.canvas.create_rectangle(*points, outline=color, width=2)
            self.canvas.create_text(points[0]+4, points[1]+4, text=f'#{index+1}', fill=color, anchor='nw')
            if index in (self.selected_box, self.hovered_box):
                for x, y in self.corners(points):
                    self.canvas.create_rectangle(x-4, y-4, x+4, y+4, fill=color, outline='#17212d')

    def box_canvas(self, box):
        width, height = self.pil_image.size
        ox, oy = self.offset
        return (ox + box[0]*width*self.scale, oy + box[1]*height*self.scale,
                ox + box[2]*width*self.scale, oy + box[3]*height*self.scale)

    @staticmethod
    def corners(box):
        return [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]

    def image_point(self, event, clamp=False):
        if self.pil_image is None:
            return None
        width, height = self.pil_image.size
        x = (event.x-self.offset[0]) / (self.scale*width)
        y = (event.y-self.offset[1]) / (self.scale*height)
        if not clamp and not (0 <= x <= 1 and 0 <= y <= 1):
            return None
        return max(0., min(1., x)), max(0., min(1., y))

    def hit_test(self, event):
        """Use screen pixels for consistent handles at every zoom level."""
        if not self.current or self.pil_image is None or self.current.labels()[self.frame_index] == -1:
            return 'draw', None, None
        if self.force_new or event.state & 4:
            return 'draw', None, None
        boxes = self.current.boxes.get(str(self.frame_index), [])
        order = list(reversed(range(len(boxes))))
        for index in order:
            corners = self.corners(self.box_canvas(boxes[index]))
            corner = min(range(4), key=lambda i: max(abs(event.x-corners[i][0]), abs(event.y-corners[i][1])))
            x, y = corners[corner]
            if abs(event.x-x) <= 7 and abs(event.y-y) <= 7:
                return 'resize', index, corner
        for index in order:
            x1, y1, x2, y2 = self.box_canvas(boxes[index])
            vertical = min(abs(event.x-x1), abs(event.x-x2)) <= 7 and y1 <= event.y <= y2
            horizontal = min(abs(event.y-y1), abs(event.y-y2)) <= 7 and x1 <= event.x <= x2
            if vertical or horizontal:
                return 'move', index, None
        return 'draw', None, None

    def hover(self, event):
        if self.drag:
            return
        mode, index, corner = self.hit_test(event)
        cursor = ('bottom_right_corner' if corner in (0, 2) else 'top_right_corner') if mode == 'resize' else 'fleur' if mode == 'move' else 'crosshair'
        if self.busy or self.pil_image is None or self.current.labels()[self.frame_index] == -1:
            cursor, index = 'arrow', None
        elif mode == 'draw' and self.image_point(event) is None:
            cursor = 'arrow'
        self.canvas.configure(cursor=cursor)
        if self.hovered_box != index:
            self.hovered_box = index
            self.render()

    def press(self, event):
        if self.busy or not self.current or self.pil_image is None:
            return
        self.pause()
        mode, selected, corner = self.hit_test(event)
        point = self.image_point(event, clamp=mode != 'draw')
        if point is None:
            return
        if self.current.labels()[self.frame_index] == -1:
            self.status.set('這一幀已略過；若要標註，請先刪除對應略過區段。')
            return
        boxes = copy.deepcopy(self.current.boxes.get(str(self.frame_index), []))
        self.force_new = False
        anchor = self.corners(boxes[selected])[(corner+2) % 4] if mode == 'resize' else point
        if mode == 'draw':
            selected = len(boxes)
            boxes.append([*point, *point])
        self.selected_box = selected
        self.drag = {'mode': mode, 'index': selected, 'point': point, 'anchor': anchor,
                     'original': copy.deepcopy(boxes), 'preview': boxes}
        self.render()
        self.canvas.configure(cursor='fleur' if mode == 'move' else ('bottom_right_corner' if corner in (0, 2) else 'top_right_corner') if mode == 'resize' else 'crosshair')

    def motion(self, event):
        if not self.drag or self.busy:
            return
        point = self.image_point(event, clamp=True)
        item = self.drag
        if item['mode'] == 'move':
            box = item['original'][item['index']]
            dx = max(-box[0], min(1-box[2], point[0]-item['point'][0]))
            dy = max(-box[1], min(1-box[3], point[1]-item['point'][1]))
            value = [box[0]+dx, box[1]+dy, box[2]+dx, box[3]+dy]
        else:
            ax, ay = item['anchor']
            value = [min(ax, point[0]), min(ay, point[1]), max(ax, point[0]), max(ay, point[1])]
        item['preview'][item['index']] = value
        self.render()

    def release(self, event):
        if not self.drag:
            return
        self.motion(event)
        boxes, index = self.drag['preview'], self.drag['index']
        box = boxes[index]
        width, height = self.pil_image.size
        self.drag = None
        if (box[2]-box[0])*width < 2 or (box[3]-box[1])*height < 2:
            self.selected_box = None
            self.render()
            return
        original = self.current.boxes.get(str(self.frame_index), [])
        if boxes != original:
            self.remember()
            try:
                self.current.set_boxes(self.frame_index, boxes)
                self.changed()
            except Exception as exc:
                self.undo_stack.pop()
                self.error(exc)
        self.render()
        self.update_boxes()
        self.hover(event)

    def wheel(self, event):
        if self.pil_image is None or self.drag or self.busy:
            return
        self.pause()
        new_zoom = max(1., min(20., self.zoom * (1.25 if event.delta > 0 else .8)))
        ratio = new_zoom / self.zoom
        self.offset = (event.x-(event.x-self.offset[0])*ratio, event.y-(event.y-self.offset[1])*ratio)
        self.zoom = new_zoom
        self.render()

    def pan_press(self, event):
        if not self.busy:
            self.pause()
            self.pan_start = (event.x, event.y, *self.offset)

    def pan_motion(self, event):
        if self.pan_start and self.zoom > 1 and not self.busy:
            x, y, ox, oy = self.pan_start
            self.offset = (ox+event.x-x, oy+event.y-y)
            self.render()

    def fit_view(self):
        self.zoom = 1.
        self.render()

    def new_box(self):
        if not self.busy:
            self.selected_box = None
            self.force_new = True
            self.pause()
            self.status.set('在目前影格拖曳畫新框。')
            self.render()

    def select_box(self, event=None):
        selection = self.box_list.curselection()
        if selection:
            self.selected_box = selection[0]
            self.render()

    def update_boxes(self):
        self.box_list.delete(0, 'end')
        if not self.current:
            return
        boxes = self.current.boxes.get(str(self.frame_index), [])
        if self.current.labels()[self.frame_index] == -1:
            boxes = []
        width, height = self.pil_image.size
        for index, (x1,y1,x2,y2) in enumerate(boxes):
            self.box_list.insert('end', f'#{index+1}  {round((x2-x1)*width)} × {round((y2-y1)*height)}')
        if self.selected_box is not None and self.selected_box < len(boxes):
            self.box_list.selection_set(self.selected_box)
        label = self.current.labels()[self.frame_index]
        state = '略過' if label == -1 else '有物體區段' if label == 1 else '區段外'
        missing = len(self.current.missing_boxes())
        self.info.set(f'第 {self.frame_index+1} / {self.current.frame_count} 幀 · {self.frame_index/self.current.fps:.3f} 秒 · '
                      f'{self.current.fps:.3f} FPS · {state} · {len(boxes)} 個框 · 缺框 {missing} 幀')
        self.draw_timeline()

    def remember(self):
        self.undo_stack.append((copy.deepcopy(self.current.intervals), copy.deepcopy(self.current.ignored),
                                copy.deepcopy(self.current.boxes)))
        self.undo_stack = self.undo_stack[-10:]

    def undo(self):
        if self.busy or not self.current or not self.undo_stack:
            return
        self.pause()
        self.drag = None
        (self.current.intervals, self.current.ignored, self.current.boxes) = self.undo_stack.pop()
        self.selected_box = None
        self.refresh_sources()
        self.refresh_intervals()
        self.update_save_state()
        self.update_boxes()
        self.render()

    def changed(self):
        self.refresh_sources()
        self.refresh_intervals()
        self.update_boxes()
        self.update_save_state()

    def delete_box(self):
        if self.busy or not self.current or self.selected_box is None:
            return
        if self.current.labels()[self.frame_index] == -1:
            return
        self.pause()
        boxes = copy.deepcopy(self.current.boxes.get(str(self.frame_index), []))
        if self.selected_box >= len(boxes):
            return
        self.remember()
        boxes.pop(self.selected_box)
        self.current.set_boxes(self.frame_index, boxes)
        self.selected_box = None
        self.changed()
        self.render()

    def mark_start(self):
        if self.current and not self.busy:
            self.pause()
            self.start_frame.set(self.frame_index+1)
            try:
                end = self.end_frame.get()
            except tk.TclError:
                end = -1
            if end < self.start_frame.get():
                self.end_frame.set(self.start_frame.get())
            self.draw_timeline()

    def mark_end(self):
        if self.current and not self.busy:
            self.pause()
            self.end_frame.set(self.frame_index+1)
            self.draw_timeline()

    def add_interval(self, ignored=False):
        if not self.current or self.busy:
            return
        self.pause()
        try:
            pair = [self.start_frame.get()-1, self.end_frame.get()-1]
            intervals = self.current.ignored if ignored else self.current.intervals
            merged = merge_intervals([*intervals, pair], self.current.frame_count)
            self.remember()
            if ignored:
                self.current.ignored = merged
            else:
                self.current.intervals = merged
            self.changed()
            self.status.set('區段已加入。可畫一個模板框，再按「套用目前框到區段」，或繼續逐幀框選。')
        except Exception as exc:
            self.error(exc)

    def ignore_interval(self):
        self.add_interval(ignored=True)

    def seed_interval(self):
        if not self.current or self.busy:
            return
        self.pause()
        try:
            self.remember()
            result = self.current.seed_boxes(self.frame_index, self.start_frame.get()-1, self.end_frame.get()-1)
        except Exception as exc:
            if self.undo_stack:
                self.undo_stack.pop()
            self.error(exc)
            return
        self.changed()
        self.render()
        self.status.set(f'同位置套用 {result["filled"]} 幀；保留已有框 {result["kept"]} 幀、略過 {result["ignored"]} 幀。'
                        '可直接逐幀微調，完成後按儲存。')

    def ignore_frame(self):
        if self.current and not self.busy:
            self.start_frame.set(self.frame_index+1)
            self.end_frame.set(self.frame_index+1)
            self.ignore_interval()

    def refresh_intervals(self):
        self.interval_list.delete(*self.interval_list.get_children())
        if self.current:
            for kind, intervals, title in [('roi', self.current.intervals, '有物體'), ('ignore', self.current.ignored, '略過')]:
                for index, (start, end) in enumerate(intervals):
                    self.interval_list.insert('', 'end', iid=f'{kind}_{index}', text=title,
                        values=(f'{start/self.current.fps:.3f} – {end/self.current.fps:.3f} s',))

    def go_interval(self):
        if self.current and self.interval_list.selection():
            kind, index = self.interval_list.selection()[0].split('_')
            pair = (self.current.intervals if kind == 'roi' else self.current.ignored)[int(index)]
            self.start_frame.set(pair[0]+1)
            self.end_frame.set(pair[1]+1)
            self.pause()
            self.seek(pair[0])
            self.draw_timeline()

    def delete_interval(self):
        if self.busy or not self.current or not self.interval_list.selection():
            return
        self.pause()
        self.remember()
        kind, index = self.interval_list.selection()[0].split('_')
        (self.current.intervals if kind == 'roi' else self.current.ignored).pop(int(index))
        labels = self.current.labels()
        self.current.boxes = {k: v for k, v in self.current.boxes.items() if labels[int(k)] != 0}
        self.changed()
        self.render()

    def next_missing(self):
        if not self.current or self.busy:
            return
        self.pause()
        missing = self.current.missing_boxes()
        if missing:
            self.seek(next((i for i in missing if i > self.frame_index), missing[0]))
        else:
            self.status.set('有物體區段已沒有缺框影格，可繼續逐幀微調或儲存。')

    @staticmethod
    def time_text(seconds):
        minutes, remainder = divmod(max(0., seconds), 60)
        hours, minutes = divmod(int(minutes), 60)
        return f'{hours:d}:{minutes:02d}:{remainder:04.1f}' if hours else f'{minutes:d}:{remainder:04.1f}'

    @staticmethod
    def consecutive_ranges(indices):
        ranges = []
        for index in sorted(indices):
            if ranges and index == ranges[-1][1]+1:
                ranges[-1][1] = index
            else:
                ranges.append([index, index])
        return ranges

    def timeline_geometry(self):
        return 48., max(49., self.timeline.winfo_width()-12.)

    def timeline_frame(self, x):
        left, right = self.timeline_geometry()
        fraction = max(0., min(1., (x-left)/(right-left)))
        return min(self.current.frame_count-1, int(fraction*self.current.frame_count))

    def timeline_seek(self, event):
        if self.current and not self.busy:
            self.pause()
            self.seek(self.timeline_frame(event.x))

    def timeline_hover(self, event):
        if not self.current or self.busy or event.state & 256:
            return
        index = self.timeline_frame(event.x)
        for intervals, title in [(self.current.ignored, '略過'), (self.current.intervals, '有物體')]:
            pair = next((pair for pair in intervals if pair[0] <= index <= pair[1]), None)
            if pair:
                self.status.set(f'{title}區段：{pair[0]/self.current.fps:.3f}–{pair[1]/self.current.fps:.3f} 秒 '
                                f'（第 {pair[0]+1}–{pair[1]+1} 幀）。點擊或拖曳時間軌可跳轉。')
                return
        self.status.set(f'時間軌：{index/self.current.fps:.3f} 秒（第 {index+1} 幀）。點擊或拖曳可跳轉。')

    def draw_timeline(self):
        self.timeline.delete('all')
        left, right = self.timeline_geometry()
        if right-left < 20:
            return
        self.timeline.create_text(4, 28, text='區段', anchor='w', fill='#475569')
        self.timeline.create_text(4, 48, text='框', anchor='w', fill='#475569')
        for y in (24, 44):
            self.timeline.create_rectangle(left, y, right, y+12, fill='#e2e8f0', outline='')
        if not self.current:
            return
        source = self.current
        count = source.frame_count
        span = right-left
        def x(index):
            return left + index/count*span
        def fill_ranges(ranges, y, color, tag):
            for start, end in ranges:
                self.timeline.create_rectangle(x(start), y, max(x(start)+1, x(end+1)), y+12,
                    fill=color, outline='', tags=(tag,))
        fill_ranges(source.intervals, 24, '#ef4444', 'roi')
        fill_ranges(source.ignored, 24, '#94a3b8', 'ignore')
        labels = source.labels()
        annotated = [int(key) for key, boxes in source.boxes.items() if boxes and labels[int(key)] == 1]
        fill_ranges(self.consecutive_ranges(annotated), 44, '#22c55e', 'annotated')
        fill_ranges(self.consecutive_ranges(source.missing_boxes()), 44, '#fecaca', 'missing')
        ticks = max(2, min(8, int(span/110)))
        duration = count/source.fps
        for tick in range(ticks+1):
            xx = left + span*tick/ticks
            self.timeline.create_line(xx, 16, xx, 20, fill='#94a3b8')
            anchor = 'w' if tick == 0 else 'e' if tick == ticks else 'center'
            self.timeline.create_text(xx, 8, text=self.time_text(duration*tick/ticks),
                anchor=anchor, fill='#475569', font=('TkDefaultFont', 9))
        try:
            for index, title in [(self.start_frame.get()-1, 'S'), (self.end_frame.get()-1, 'E')]:
                if 0 <= index < count:
                    xx = x(index)
                    self.timeline.create_line(xx, 22, xx, 38, fill='#7c3aed', width=2, dash=(2,2), tags=('selection',))
                    self.timeline.create_text(xx+3, 38, text=title, anchor='nw', fill='#7c3aed', font=('TkDefaultFont', 8))
        except tk.TclError:
            pass  # An unfinished edit in an entry does not break timeline rendering.
        if self.frame_index >= 0:
            xx = x(self.frame_index)
            self.timeline.create_line(xx, 19, xx, 57, fill='#2563eb', width=2, tags=('playhead',))
            self.timeline.create_polygon(xx-4, 17, xx+4, 17, xx, 22, fill='#2563eb', tags=('playhead',))
        self.timeline.create_text(left, 66, text='紅：有物體  灰：略過  綠：有框  淺紅：缺框',
            anchor='w', fill='#475569', font=('TkDefaultFont', 9))

    def new_project(self):
        if self.busy:
            return
        self.pause()
        if not self.ensure_saved('建立新專案'):
            return
        self.clear_current()
        self.project = SmokeProject()
        self.project_path = self._new_project_path()
        self.saved_state = asdict(self.project)
        self.update_save_state()
        self.refresh_sources()
        self.status.set('新專案已準備；加入來源後開始標註。')

    def open_project(self):
        if self.busy:
            return
        self.pause()
        path = filedialog.askopenfilename(parent=self.root, title='載入 Smoke Heatmap 標註專案',
            initialdir=self.project_path.parent, filetypes=[('標註專案', '*.json')])
        if path:
            try:
                project = load_project(path)
                if not self.ensure_saved('載入其他專案'):
                    return
                self.clear_current()
                self.project, self.project_path = project, Path(path).resolve()
                self.saved_state = asdict(self.project)
                self.update_save_state()
                self.refresh_sources()
                if project.sources:
                    self.source_list.selection_set(project.sources[0].id)
                    self.select_source()
            except Exception as exc:
                self.error(exc)

    def save_as(self):
        if self.busy:
            return
        self.pause()
        path = filedialog.asksaveasfilename(parent=self.root, title='另存新的標註專案',
            initialdir=self.project_path.parent, defaultextension='.json', filetypes=[('標註專案', '*.json')], confirmoverwrite=False)
        if path:
            path = Path(path).resolve()
            if path.exists() and path != self.project_path:
                self.error('檔案已存在，請使用新的名稱。')
                return
            previous = self.project_path
            self.project_path = path
            if not self.save():
                self.project_path = previous
                self.update_save_state()

    def choose_output(self):
        if not self.busy:
            path = filedialog.askdirectory(parent=self.root, title='資料集版本的保存位置', initialdir=self.output_parent.get())
            if path:
                self.output_parent.set(path)

    def options(self):
        if not self.update_origin():
            raise ValueError('原始來源群組不可空白。')
        if not self.output_parent.get().strip():
            raise ValueError('請指定輸出位置。')
        text = self.ratios.get().replace('：', ':').replace(',', ':').replace('，', ':')
        return {'output_parent': self.output_parent.get(), 'name': self.dataset_name.get(),
                'ratios': tuple(float(v) for v in text.split(':')), 'history_frames': self.history_frames.get(), 'seed': self.seed}

    @staticmethod
    def report_text(report):
        lines = []
        for split in ('train', 'val'):
            row = report['summary'][split]
            lines.append(f'{split}：{report["group_counts"][split]} 群組、{row["frames"]} 幀、'
                         f'{row["samples"]} 窗口、有框窗口 {row["positive_samples"]}、略過 {row["ignored_frames"]} 幀')
        lines.extend(report['warnings'])
        lines.append('每個群組完整保留在同一組；匯出所有歷史畫面，不自動產生 BBox。')
        return '\n'.join(lines)

    def preview_split(self):
        if self.busy:
            return
        self.pause()
        try:
            options = self.options()
            report = plan_split(self.project, options['ratios'], options['seed'], options['history_frames'])
            detail = '\n'.join(f'{s.name} → {report["assignments"][s.id]}' for s in self.project.sources)
            messagebox.showinfo('分組預覽', self.report_text(report)+'\n\n'+detail, parent=self.root)
        except Exception as exc:
            self.error(exc)

    def start_export(self):
        if self.busy:
            return
        self.pause()
        try:
            options = self.options()
            plan_split(self.project, options['ratios'], options['seed'], options['history_frames'])
            if not self.save():
                return
            snapshot = copy.deepcopy(self.project)
        except Exception as exc:
            self.error(exc)
            return
        self.cancel.clear()
        self.busy = True
        self.status.set('正在產生新版本，可按「取消匯出」；來源與既有版本會保留。')
        def work():
            try:
                result = export_dataset(snapshot, **options, progress=lambda text: self.messages.put(('progress', text)), cancel=self.cancel)
                self.messages.put(('done', result))
            except ExportCancelled as exc:
                self.messages.put(('cancelled', str(exc)))
            except Exception as exc:
                self.messages.put(('error', str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def pump(self):
        while not self.messages.empty():
            kind, value = self.messages.get_nowait()
            if kind == 'progress':
                self.status.set(value)
            else:
                self.busy = False
                if self.current:
                    self.position.set(self.frame_index)
                if kind == 'done':
                    self.status.set(f'已產生：{value["output"]}')
                    if not self.closing:
                        messagebox.showinfo('資料集完成', self.report_text(value)+'\n\n'+value['output'], parent=self.root)
                elif kind == 'cancelled':
                    self.status.set(value)
                elif not self.closing:
                    self.error(value)
        if self.closing and not self.busy:
            self.root.destroy()
        else:
            self.pump_token = self.root.after(100, self.pump)

    def close(self):
        self.pause()
        if not self.ensure_saved('關閉視窗'):
            return
        if self.busy:
            self.closing = True
            self.cancel.set()
            if self.capture:
                self.capture.release()
                self.capture = None
            self.root.withdraw()
            return
        if self.capture:
            self.capture.release()
        self.root.after_cancel(self.pump_token)
        self.root.destroy()


def launch(project=None, output_parent=DATASET_ROOT, name='st2', ratios=(7., 3.), history_frames=4, seed=2026):
    root = tk.Tk()
    SmokeAnnotationApp(root, project, output_parent, name, ratios, history_frames, seed)
    root.mainloop()
