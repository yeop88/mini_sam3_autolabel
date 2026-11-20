"""4. 실시간 탭 — 학습한 YOLO-seg 모델을 위에서 연 카메라·영상에 실시간으로 돌린다.

단축키: R 실행/멈춤, S 화면 저장 (captures\\)
"""
import os
import time
import tkinter as tk
from tkinter import filedialog, ttk

import cv2

from core import render
from core.paths import CAPTURE_DIR, ROOT, RUNS_DIR
from core.realtime import LiveDetector, list_trained_weights
from ui.common import BaseTab, get_num, open_folder, spin

GREEN, ORANGE = (80, 230, 80), (0, 170, 255)


class LiveTab(BaseTab):
    TITLE = " 4 실시간 "

    def __init__(self, master, app):
        super().__init__(master, app)
        self.det = None
        self.weights = []
        self.conf = tk.StringVar(value="0.4")
        self.imgsz = tk.StringVar(value="640")
        self.half = tk.BooleanVar(value=True)
        self.trt = tk.BooleanVar(value=True)
        self._stats_t = 0.0
        self._last_state = None
        self._build()
        self.refresh_weights()

    def _build(self):
        f = ttk.LabelFrame(self, text="모델 (runs\\*\\weights\\best.pt, 최근 것부터)", padding=6)
        f.pack(fill="x")
        self.combo = ttk.Combobox(f, state="readonly", width=52)
        self.combo.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Button(f, text="새로고침", command=self.refresh_weights).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(f, text="다른 파일...", command=self.browse).grid(row=1, column=1, sticky="w", pady=(4, 0))

        f = ttk.LabelFrame(self, text="설정", padding=6)
        f.pack(fill="x", pady=(6, 0))
        spin(f, 0, "점수 기준", self.conf, 0.05, 0.95, 0.05, help_text="실행 중에도 바뀜")
        ttk.Label(f, text="입력 크기").grid(row=1, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.imgsz, values=["480", "640", "800", "960"], width=7).grid(
            row=1, column=1, sticky="w", pady=1)
        ttk.Checkbutton(f, text="TensorRT FP16 (가장 빠름. 엔진이 없으면 처음 한 번 약 4분 변환)", variable=self.trt).grid(
            row=2, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(f, text="FP16 (TensorRT 를 끈 경우 PyTorch)", variable=self.half).grid(
            row=3, column=0, columnspan=3, sticky="w")

        f = ttk.LabelFrame(self, text="실행", padding=6)
        f.pack(fill="x", pady=(6, 0))
        b = ttk.Frame(f)
        b.grid(row=0, column=0, sticky="w")
        self.btn_run = ttk.Button(b, text="실행 (R)", command=self.toggle, width=12)
        self.btn_run.pack(side="left")
        ttk.Button(b, text="화면 저장 (S)", command=self.save).pack(side="left", padx=4)
        ttk.Button(b, text="저장 폴더", command=lambda: open_folder(CAPTURE_DIR if os.path.isdir(CAPTURE_DIR)
                                                                    else ROOT)).pack(side="left")
        self.stats = ttk.Label(f, text="멈춤", foreground="gray", wraplength=400)
        self.stats.grid(row=1, column=0, sticky="w", pady=(4, 0))

    def refresh_weights(self):
        self.weights = list_trained_weights()
        self.combo.configure(values=[os.path.relpath(p, ROOT) for p in self.weights])
        if self.weights:
            self.combo.current(0)
        else:
            self.combo.set("")
            self.stats.configure(text="학습한 모델이 없습니다. 3 학습 탭에서 먼저 학습하세요.")

    def browse(self):
        p = filedialog.askopenfilename(title="YOLO 가중치", initialdir=RUNS_DIR if os.path.isdir(RUNS_DIR) else ROOT,
                                       filetypes=[("PyTorch", "*.pt"), ("모두", "*.*")])
        if p:
            self.weights.insert(0, p)
            self.combo.configure(values=[os.path.relpath(x, ROOT) if x.startswith(ROOT) else x for x in self.weights])
            self.combo.current(0)

    def toggle(self):
        if self.det is not None:
            self.det.stop()
            self.det = None
            self.btn_run.configure(text="실행 (R)")
            self.stats.configure(text="멈춤")
            self.touch()
            return
        pos = self.combo.current()
        if pos < 0:
            self.app.log("모델을 고르세요.")
            return
        w = self.weights[pos]
        self.det = LiveDetector(w, self.app.source, conf=get_num(self.conf, 0.4, float, 0.01, 0.99),
                                imgsz=int(get_num(self.imgsz, 640, float, 32)), half=self.half.get(),
                                trt=self.trt.get())
        self._last_state = None
        self.btn_run.configure(text="멈춤 (R)")
        self.app.log(f"실시간 시작: {os.path.relpath(w, ROOT) if w.startswith(ROOT) else w}")

    def save(self):
        d = self.det
        r = d.result if d is not None else None
        if r is None:
            seq, frame = self.app.source.latest()
            if frame is None:
                return
            img = frame
        else:
            img = render.annotate_full(r["frame"], r["items"], d.names, [f"{len(r['items'])} obj  {r['ms']:.1f} ms"])
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        p = os.path.join(CAPTURE_DIR, time.strftime("%Y%m%d_%H%M%S") + "_live.jpg")
        cv2.imwrite(p, img)
        self.app.log(f"화면 저장: {p}")

    def on_key(self, k):
        if k in ("r", "R"):
            self.toggle()
        elif k in ("s", "S"):
            self.save()
        else:
            return False
        return True

    def poll(self):
        d = self.det
        if d is None:
            return
        if d.state != self._last_state:
            self._last_state = d.state
            if d.state == "running":
                self.app.log(f"실시간 모델 준비 ({d.backend}): 클래스 {d.names}")
            elif d.state == "exporting":
                self.app.log("TensorRT 엔진 만드는 중 (약 4분, 이 모델·입력 크기는 한 번만)...")
            elif d.state == "error":
                self.app.log(f"[오류] 실시간: {d.error}")
                self.det = None
                self.btn_run.configure(text="실행 (R)")
                self.stats.configure(text=f"오류: {d.error}")
                return
        d.set_conf(get_num(self.conf, 0.4, float, 0.01, 0.99))
        now = time.time()
        if now - self._stats_t > 0.3:
            self._stats_t = now
            if d.state == "loading":
                self.stats.configure(text="모델 불러오는 중...")
            elif d.state == "exporting":
                self.stats.configure(text="TensorRT 엔진 만드는 중 (약 4분, 한 번만)...")
            elif d.result is not None:
                self.stats.configure(text=f"{d.backend}  추론 {d.ms:.1f} ms ({1000 / max(d.ms, 1e-3):.0f} fps 가능), "
                                          f"객체 {len(d.result['items'])}개, 카메라 {self.app.source.fps:.0f} fps")

    def render_key(self, seq):
        d = self.det
        if d is not None and d.state == "running":
            return d.result_seq, self.version
        return seq, self.version, d is None

    def compose(self, frame):
        d = self.det
        if d is None:
            return dict(img=frame, lines=[("멈춤 — [실행 (R)] 을 누르세요", ORANGE)])
        if d.state != "running" or d.result is None:
            text = "TensorRT 엔진 만드는 중 (약 4분, 한 번만)..." if d.state == "exporting" else "모델 불러오는 중..."
            return dict(img=frame, lines=[(text, ORANGE)])
        r = d.result
        return dict(img=r["frame"], items=r["items"], names=d.names,
                    lines=[(f"{len(r['items'])} obj   {d.backend}  추론 {d.ms:.1f} ms", GREEN)])

    def shutdown(self):
        if self.det is not None:
            self.det.stop()
