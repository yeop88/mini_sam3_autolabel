"""3. 학습 탭 — 고른 세션들로 YOLO 학습. 별도 프로세스(python -m core.train_job)로 돌려 GUI 가 멈추지 않는다.

- 클래스: 세션마다 달라도 이름으로 합침 (모든 물체 세션 포함)
- 작업: 박스 세션이 하나라도 있으면 박스(detect) 모델, 모두 분할이면 분할(segment) 모델
결과: dataset\\<이름>\\, runs\\<이름>\\weights\\best.pt, runs\\<이름>\\train_summary.json
"""
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from core.dataset import default_model, is_seg_model, merge_classes, pretrained_choices
from core.paths import ROOT, RUNS_DIR
from core.session import list_sessions
from core.util import read_json, stamp, write_json
from ui.common import BaseTab, get_num, open_folder, spin

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PROGRESS = re.compile(r"(\d+/\d+.*(it/s|s/it))|(%\|)|(\d+% [━─╸]+)")
NOISE = re.compile(r"(ultralytics\.nn\.|torch\.nn\.|^\s*from\s+n\s+params)")     # 모델 구조 표
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def console_python():
    """pythonw 로 GUI 를 띄워도 학습은 python.exe 로 (출력을 읽기 위해)"""
    d = os.path.dirname(sys.executable)
    p = os.path.join(d, "python.exe")
    return p if os.path.isfile(p) else sys.executable


class TrainTab(BaseTab):
    TITLE = " 3 학습 "

    def __init__(self, master, app):
        super().__init__(master, app)
        self.sessions = []
        self.proc = None
        self.run_name = None
        self.last_run_dir = None
        self.model = tk.StringVar(value="yolo26s-seg.pt")
        self.epochs = tk.StringVar(value="80")
        self.imgsz = tk.StringVar(value="640")
        self.batch = tk.StringVar(value="16")
        self.patience = tk.StringVar(value="25")
        self.val_ratio = tk.StringVar(value="0.15")
        self.block = tk.StringVar(value="10")
        self.name = tk.StringVar(value="")
        self.trt = tk.BooleanVar(value=True)
        self.task = "segment"
        self.status = "대기"
        self._build()

    def _build(self):
        f = ttk.LabelFrame(self, text="학습할 세션 (Ctrl·Shift 로 여러 개, 클래스는 이름으로 합침)", padding=6)
        f.pack(fill="x")
        self.listbox = tk.Listbox(f, height=9, width=52, selectmode="extended", exportselection=False,
                                  font=("Malgun Gothic", 9))
        self.listbox.grid(row=0, column=0, columnspan=3, sticky="w")
        self.listbox.bind("<<ListboxSelect>>", lambda e: self._update_selection())
        ttk.Button(f, text="새로고침", command=self.refresh_sessions).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(f, text="모두 선택", command=lambda: (self.listbox.selection_set(0, "end"),
                                                      self._update_selection())).grid(row=1, column=1, sticky="w",
                                                                                      pady=(4, 0))
        self.sel_label = ttk.Label(f, text="", wraplength=410)
        self.sel_label.grid(row=2, column=0, columnspan=3, sticky="w")

        f = ttk.LabelFrame(self, text="설정", padding=6)
        f.pack(fill="x", pady=(6, 0))
        ttk.Label(f, text="시작 가중치").grid(row=0, column=0, sticky="w")
        self.model_combo = ttk.Combobox(f, textvariable=self.model, values=pretrained_choices("segment"), width=22)
        self.model_combo.grid(row=0, column=1, columnspan=2, sticky="w", pady=1)
        spin(f, 1, "epoch", self.epochs, 1, 1000, 10)
        ttk.Label(f, text="입력 크기").grid(row=2, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.imgsz, values=["480", "640", "800", "960"], width=7).grid(
            row=2, column=1, sticky="w", pady=1)
        spin(f, 3, "batch", self.batch, 1, 128, 1)
        spin(f, 4, "patience", self.patience, 0, 500, 5, help_text="이만큼 좋아지지 않으면 일찍 끝")
        spin(f, 5, "검증 비율", self.val_ratio, 0.05, 0.5, 0.05)
        spin(f, 6, "검증 묶음", self.block, 1, 100, 1, help_text="연속 N장 단위로 검증 분할")
        ttk.Label(f, text="이름 (비우면 시각)").grid(row=7, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.name, width=24).grid(row=7, column=1, columnspan=2, sticky="w", pady=1)
        ttk.Checkbutton(f, text="끝나면 TensorRT 엔진도 만들기 (약 4분 더, 실시간 2.6배 빠름)", variable=self.trt).grid(
            row=8, column=0, columnspan=3, sticky="w")

        f = ttk.LabelFrame(self, text="실행", padding=6)
        f.pack(fill="x", pady=(6, 0))
        b = ttk.Frame(f)
        b.grid(row=0, column=0, sticky="w")
        ttk.Button(b, text="학습 시작", command=self.start).pack(side="left")
        ttk.Button(b, text="2 epoch 시험", command=lambda: self.start(smoke=True)).pack(side="left", padx=4)
        ttk.Button(b, text="중지", command=self.stop).pack(side="left")
        ttk.Button(b, text="결과 폴더", command=lambda: open_folder(self.last_run_dir or RUNS_DIR)).pack(
            side="left", padx=4)
        self.status_label = ttk.Label(f, text="대기", wraplength=400)
        self.status_label.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.progress_label = ttk.Label(f, text="", foreground="gray", wraplength=400)
        self.progress_label.grid(row=2, column=0, sticky="w")
        self.result_label = ttk.Label(f, text="", wraplength=400)
        self.result_label.grid(row=3, column=0, sticky="w")

    def on_show(self):
        self.refresh_sessions()

    def refresh_sessions(self):
        keep = {self.sessions[i]["dir"] for i in self.listbox.curselection()} if self.sessions else set()
        self.sessions = list_sessions()
        self.listbox.delete(0, "end")
        for i, s in enumerate(self.sessions):
            kind = "박스" if s["format"] == "box" else "분할"
            mode = "모든 물체" if s["class_mode"] == "open" else "지정"
            self.listbox.insert("end", f"{s['name']}  ({s['images']}장, {kind}, {mode}, 클래스 {len(s['names'])}개)")
            if s["dir"] in keep:
                self.listbox.selection_set(i)
        self._update_selection()

    def _update_selection(self):
        """고른 세션 -> 합친 클래스·작업 표시, 작업에 맞는 시작 가중치 목록"""
        dirs = [self.sessions[i]["dir"] for i in self.listbox.curselection()]
        if not dirs:
            self.sel_label.configure(text="세션을 고르세요.")
            return
        names, task, infos = merge_classes(dirs)
        n_img = sum(s["images"] for s in infos)
        self.sel_label.configure(text=f"{len(dirs)}개 세션, {n_img}장 → 작업: {'박스(detect)' if task == 'detect' else '분할(segment)'}"
                                      f", 클래스 {len(names)}개: " + ", ".join(names[:20]) + (" ..." if len(names) > 20 else ""))
        if task != self.task:
            self.task = task
            self.model_combo.configure(values=pretrained_choices(task))
            if is_seg_model(self.model.get()) != (task == "segment"):
                self.model.set(default_model(task))

    def _set_status(self, text):
        self.status = text
        self.status_label.configure(text=text)
        self.touch()

    def start(self, smoke=False):
        if self.proc is not None:
            self.app.log("학습이 이미 진행 중입니다.")
            return
        dirs = [self.sessions[i]["dir"] for i in self.listbox.curselection()]
        if not dirs:
            messagebox.showinfo("학습", "학습할 세션을 고르세요.")
            return
        names, task, _ = merge_classes(dirs)
        model = self.model.get().strip() or default_model(task)
        if task == "detect" and is_seg_model(model):
            model = default_model("detect")
            self.model.set(model)
            self.app.log(f"박스 세션이 있어 박스 모델 {model} 로 학습합니다.")
        name = re.sub(r"[^\w\-]+", "_", self.name.get().strip()) or (stamp() + ("_smoke" if smoke else ""))
        cfg = dict(sessions=dirs, name=name, model=model,
                   epochs=2 if smoke else int(get_num(self.epochs, 80, float, 1)),
                   imgsz=int(get_num(self.imgsz, 640, float, 32)), batch=int(get_num(self.batch, 16, float, 1)),
                   patience=int(get_num(self.patience, 25, float, 0)),
                   val_ratio=get_num(self.val_ratio, 0.15, float, 0.01, 0.9), block=int(get_num(self.block, 10, float, 1)),
                   seed=0, device="0", workers=4, trt=bool(self.trt.get()) and not smoke)
        run_dir = os.path.join(RUNS_DIR, name)
        cfg_path = os.path.join(run_dir, "job_config.json")
        write_json(cfg_path, cfg)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen([console_python(), "-m", "core.train_job", cfg_path], cwd=ROOT, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                     errors="replace", bufsize=1, creationflags=NO_WINDOW)
        self.run_name, self.last_run_dir = name, run_dir
        self.result_label.configure(text="")
        self._set_status(f"학습 중: {name} ({cfg['model']}, {cfg['epochs']} epoch, 세션 {len(dirs)}개, "
                         f"{'박스' if task == 'detect' else '분할'}, 클래스 {len(names)}개)")
        self.app.log(f"학습 시작: {name}, 세션 {[os.path.basename(d) for d in dirs]}, {cfg['model']}, {cfg['epochs']} epoch")
        if self.app.tabs[0].labeler is not None:
            self.app.log("참고: SAM 3 가 GPU 메모리를 쓰고 있습니다. 메모리가 모자라면 1 라벨링 탭에서 [해제] 하세요.")
        threading.Thread(target=self._read_output, args=(self.proc,), daemon=True).start()

    def _read_output(self, proc):
        summary = None
        for line in proc.stdout:
            line = ANSI.sub("", line).strip()
            if not line:
                continue
            if line.startswith("@@SUMMARY "):
                summary = line[len("@@SUMMARY "):].strip()
            elif NOISE.search(line):
                continue
            elif PROGRESS.search(line):
                self.app.post(self.progress_label.configure, {"text": line[-160:]})
            else:
                self.app.log(f"[학습] {line}")
        rc = proc.wait()
        self.app.post(self._finished, rc, summary)

    def _finished(self, rc, summary):
        self.proc = None
        s = read_json(summary, None) if summary else None
        if s:
            m = s.get("metrics", {})
            mask = (f"mask mAP50 {m['mask_map50']:.3f}, mAP50-95 {m.get('mask_map50_95', 0):.3f} / "
                    if "mask_map50" in m else "")
            text = (f"끝 ({s.get('minutes')}분, {s.get('task', 'segment')}): {mask}box mAP50 {m.get('box_map50', 0):.3f}, "
                    f"mAP50-95 {m.get('box_map50_95', 0):.3f}\n가중치: {os.path.relpath(s['best'], ROOT)}"
                    + (f"\nTensorRT: {os.path.basename(s['engine'])}" if s.get("engine") else ""))
            self.result_label.configure(text=text)
            self._set_status(f"학습 끝: {self.run_name}")
            self.app.log(f"학습 끝: {self.run_name} {m}. 4 실시간 탭에서 바로 쓸 수 있습니다.")
            self.app.tabs[3].refresh_weights()
        else:
            self._set_status(f"학습 중지/실패 (종료 코드 {rc}) — 메시지 창의 [학습] 줄을 확인하세요")
            self.app.log(f"학습이 끝나지 못했습니다 (종료 코드 {rc}).")

    def stop(self):
        if self.proc is None:
            return
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True, creationflags=NO_WINDOW)
        self.app.log("학습 중지 요청")

    def busy_text(self):
        return "학습이 진행 중입니다 (끝내면 학습도 중지됩니다)." if self.proc is not None else ""

    def shutdown(self):
        self.stop()

    def compose(self, frame):
        return dict(img=frame, lines=[self.status])
