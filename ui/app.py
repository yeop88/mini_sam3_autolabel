"""메인 창.

    ┌ 소스: [카메라 ▼] [해상도 ▼] [카메라 열기] [영상 파일...] [닫기]          소스 정보 · fps ┐
    │ 영상 화면 (VideoView)                              │ 탭: 1 라벨링 / 2 검토 / 3 학습 / 4 실시간 │
    └ 메시지 ─────────────────────────────────────────────────────────────────────────────┘

구조: 작업 스레드(카메라 읽기, SAM 3, 실시간 추론, 학습 출력 읽기)는 tkinter 를 직접 건드리지 않고
      App.post(함수) 로 큐에 넣는다. GUI 스레드의 tick 이 큐를 비우고, 보이는 탭의 화면만 다시 그린다.
"""
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

from core.camera import FrameSource, camera_choices, resolution_choices
from core.paths import SESSIONS_DIR
from ui.label_tab import LabelTab
from ui.live_tab import LiveTab
from ui.review_tab import ReviewTab
from ui.train_tab import TrainTab
from ui.video_view import VideoView

TICK_MS = 15
TEXT_INPUT_CLASSES = {"Entry", "TEntry", "Spinbox", "TSpinbox", "TCombobox", "Text"}


class App:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.q = queue.Queue()
        self.main_thread = threading.current_thread()
        self.source = FrameSource()
        self.opening = False
        self._info_t = 0.0

        root.title("SAM 3 Auto Label")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(1600, sw - 40)}x{min(950, sh - 90)}+10+5")      # 작업 표시줄에 가리지 않게
        root.minsize(1100, 700)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build()

        self.tabs = [LabelTab(self.nb, self), ReviewTab(self.nb, self), TrainTab(self.nb, self), LiveTab(self.nb, self)]
        for t in self.tabs:
            self.nb.add(t, text=t.TITLE)
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.video.click_handler = lambda x, y: self.current_tab().on_click(x, y)
        root.bind("<Key>", self._on_key)

        self._init_source()
        tab = {"label": 0, "review": 1, "train": 2, "live": 3}.get(args.tab, 0)
        self.nb.select(tab)
        root.after(TICK_MS, self._tick)
        # 라벨링 탭으로 시작하면 SAM 3 를 백그라운드로 미리 불러온다 (약 3초, 카메라 여는 동안). --no-sam 이면 안 함
        if tab == 0 and not getattr(args, "no_sam", False):
            root.after(300, self.tabs[0].auto_load_model)

    # ---------- 위젯 ----------
    def _build(self):
        root = self.root
        top = ttk.Frame(root, padding=(6, 6, 6, 4))
        top.pack(fill="x")
        ttk.Label(top, text="소스:").pack(side="left")
        self.cam_combo = ttk.Combobox(top, state="readonly", width=34)
        self.cam_combo.pack(side="left", padx=4)
        self.cam_combo.bind("<<ComboboxSelected>>", lambda e: self._fill_resolutions())
        self.res_combo = ttk.Combobox(top, state="readonly", width=36)
        self.res_combo.pack(side="left", padx=4)
        ttk.Button(top, text="카메라 열기", command=self.open_camera).pack(side="left", padx=2)
        ttk.Button(top, text="영상 파일...", command=self.choose_video).pack(side="left", padx=2)
        ttk.Button(top, text="닫기", command=self.close_source).pack(side="left", padx=2)
        self.source_label = ttk.Label(top, text="소스 없음", foreground="gray")
        self.source_label.pack(side="left", padx=10)

        bottom = ttk.LabelFrame(root, text="메시지", padding=4)
        bottom.pack(side="bottom", fill="x", padx=6, pady=(0, 6))
        self.log_box = tk.Text(bottom, height=7, state="disabled", wrap="word", font=("Malgun Gothic", 9))
        sc = ttk.Scrollbar(bottom, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sc.set)
        sc.pack(side="right", fill="y")
        self.log_box.pack(side="left", fill="x", expand=True)

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True, padx=6)
        self.nb = ttk.Notebook(body, width=440)
        self.nb.pack(side="right", fill="y")
        self.video = VideoView(body)
        self.video.pack(side="left", fill="both", expand=True, padx=(0, 6))

    # ---------- 스레드 -> GUI ----------
    def post(self, fn, *args):
        self.q.put((fn, args))

    def log(self, msg):
        if threading.current_thread() is self.main_thread:
            self._append_log(msg)
        else:
            self.post(self._append_log, msg)

    def _append_log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{datetime.now():%H:%M:%S}] {msg}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def run_bg(self, fn, done=None, fail=None):
        """fn 을 별도 스레드에서 실행하고, 끝나면 GUI 스레드에서 done(결과) / fail(예외)"""
        def work():
            try:
                r = fn()
            except Exception as e:
                self.post(fail or (lambda err: self.log(f"[오류] {err}")), e)
                return
            if done:
                self.post(done, r)
        threading.Thread(target=work, daemon=True).start()

    # ---------- 소스 ----------
    def _init_source(self):
        a = self.args
        cams, default = camera_choices()
        self.cams = cams
        self.cam_combo.configure(values=[f"[{i}] {n}" for i, n in cams])
        idx = a.camera if a.camera is not None else default
        pos = [i for i, _ in cams].index(idx) if idx in [i for i, _ in cams] else 0
        self.cam_combo.current(pos)
        self._fill_resolutions(a.size or a.mode)
        if a.video:
            self.open_video(a.video)
        elif not a.no_camera:
            self.open_camera()

    def _fill_resolutions(self, want=None):
        pos = self.cam_combo.current()
        if pos < 0:
            return
        opts, default = resolution_choices(self.cams[pos][1])
        keys = [k for k, _ in opts]
        if want and want not in keys:
            opts.append((want, want))
            keys.append(want)
        self.res_opts = opts
        self.res_combo.configure(values=[l for _, l in opts])
        self.res_combo.current(keys.index(want if want else default))

    def open_camera(self):
        if self.opening:
            return
        pos, rpos = self.cam_combo.current(), self.res_combo.current()
        if pos < 0 or rpos < 0:
            return
        idx, name = self.cams[pos]
        key = self.res_opts[rpos][0]
        self.opening = True
        self.source_label.configure(text=f"카메라 [{idx}] {name} 여는 중...", foreground="gray")
        self.log(f"카메라 [{idx}] {name} 여는 중 ({self.res_opts[rpos][1]})")

        def done(_):
            self.opening = False
            self.log(f"열림: {self.source.info}")

        def fail(e):
            self.opening = False
            self.log(f"[오류] {e}")
        self.run_bg(lambda: self.source.open_camera(idx, name, key), done, fail)

    def choose_video(self):
        path = filedialog.askopenfilename(title="영상 파일", initialdir=SESSIONS_DIR,
                                          filetypes=[("영상", "*.mp4 *.avi *.mkv *.mov"), ("모두", "*.*")])
        if path:
            self.open_video(path)

    def open_video(self, path):
        try:
            self.source.open_video(path)
            self.log(f"열림: {self.source.info}")
        except Exception as e:
            self.log(f"[오류] {e}")

    def close_source(self):
        for t in self.tabs:
            t.on_source_closing()
        self.source.close()
        self.log("소스 닫음")

    def _update_source_label(self):
        s = self.source
        if self.opening:
            return
        if s.kind is None:
            self.source_label.configure(text="소스 없음 (카메라 열기 또는 영상 파일)", foreground="gray")
            return
        text = f"{s.info}   {s.fps:.0f} fps"
        if s.recording():
            text += "   ● 원본 녹화 중"
        if s.ended:
            text += "   (영상 끝)" if s.kind == "video" else "   (카메라 끊김)"
        self.source_label.configure(text=text, foreground="black")

    # ---------- 탭 ----------
    def current_tab(self):
        return self.tabs[self.nb.index(self.nb.select())]

    def _on_tab_changed(self, _e=None):
        t = self.current_tab()
        t.on_show()
        self._force = True

    def _on_key(self, e):
        w = self.root.focus_get()
        if w is not None and w.winfo_class() in TEXT_INPUT_CLASSES:
            return None
        if self.current_tab().on_key(e.keysym):
            return "break"
        return None

    # ---------- 주기 ----------
    def _tick(self):
        try:
            for _ in range(200):
                fn, args = self.q.get_nowait()
                try:
                    fn(*args)
                except Exception as e:
                    self._append_log(f"[오류] {e}")
        except queue.Empty:
            pass
        now = time.time()
        if now - self._info_t > 0.5:
            self._info_t = now
            self._update_source_label()
        for t in self.tabs:
            t.poll()
        seq, frame = self.source.latest()
        out = self.current_tab().render(seq, frame, force=getattr(self, "_force", False))
        self._force = False
        if out is not None:
            if out.get("img") is None and not out.get("message"):
                out["message"] = "소스가 없습니다. 위에서 [카메라 열기] 또는 [영상 파일...]"
            self.video.show(**out)
        self.root.after(TICK_MS, self._tick)

    def close(self):
        busy = [t.busy_text() for t in self.tabs if t.busy_text()]
        if busy and not messagebox.askyesno("종료", "\n".join(busy) + "\n\n그래도 끝낼까요?"):
            return
        for t in self.tabs:
            try:
                t.shutdown()
            except Exception:
                pass
        self.source.close()
        self.root.destroy()
