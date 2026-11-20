"""1. 라벨링 탭 — 클래스·프롬프트, SAM 3 모델, 촬영하면서 자동 라벨, 영상 파일 일괄 라벨.

단축키: A 자동 라벨 켜기/끄기, Space 지금 프레임 저장, U 마지막 저장 취소
세션: 첫 저장(또는 자동 라벨 시작) 때 sessions\\<시각>_<태그>\\ 를 만든다. 클래스를 바꾸면 다음 저장부터 새 세션.
"""
import os
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from core import classes_cfg
from core.autolabel import LabelWorker, label_video
from core.labeler import (FAKE, FINDER_CHOICES, ObjectFinder, OpenVocabLabeler, Sam3Labeler, list_sam3_weights,
                          make_labeler)
from core.paths import SESSIONS_DIR, WEIGHTS_DIR
from core.session import Session
from ui.common import BaseTab, get_num, make_scrollable, open_folder, spin

FORMAT_TEXT = {"segment": "분할(다각형)", "box": "박스"}
TARGET_TEXT = {"fixed": "지정 클래스", "open": "모든 물체"}

GREEN, ORANGE, YELLOW = (80, 230, 80), (0, 170, 255), (0, 230, 255)


class ClassDialog(tk.Toplevel):
    """클래스 하나 추가·수정: 이름 + 프롬프트(쉼표로 여러 개). result = (name, [prompts]) 또는 None"""

    def __init__(self, master, title, name="", prompts=()):
        super().__init__(master)
        self.title(title)
        self.transient(master.winfo_toplevel())
        self.resizable(False, False)
        self.result = None
        f = ttk.Frame(self, padding=10)
        f.pack(fill="both")
        ttk.Label(f, text="클래스 이름 (학습에 쓰는 이름, 영어 권장)").grid(row=0, column=0, sticky="w")
        self.name = ttk.Entry(f, width=40)
        self.name.insert(0, name)
        self.name.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(f, text="SAM 3 프롬프트 (영어 명사구, 쉼표로 여러 개. 비우면 이름을 씀)").grid(row=2, column=0, sticky="w")
        self.prompts = ttk.Entry(f, width=40)
        self.prompts.insert(0, ", ".join(prompts))
        self.prompts.grid(row=3, column=0, sticky="ew")
        ttk.Label(f, text="예: cup, mug   /   red apple   /   cell phone", foreground="gray").grid(row=4, column=0, sticky="w")
        b = ttk.Frame(f)
        b.grid(row=5, column=0, sticky="e", pady=(10, 0))
        ttk.Button(b, text="확인", command=self._ok).pack(side="left", padx=2)
        ttk.Button(b, text="취소", command=self.destroy).pack(side="left", padx=2)
        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self.destroy())
        self.name.focus_set()
        self.grab_set()
        self.wait_window()

    def _ok(self):
        name = self.name.get().strip()
        if not name:
            messagebox.showwarning("클래스", "이름을 입력하세요.", parent=self)
            return
        prompts = [p.strip() for p in self.prompts.get().split(",") if p.strip()] or [name]
        self.result = (name, prompts)
        self.destroy()


class LabelTab(BaseTab):
    TITLE = " 1 라벨링 "

    def __init__(self, master, app):
        super().__init__(master, app)
        self.classes = classes_cfg.load()
        self.labeler = None          # SAM 3 (또는 fake)
        self.loading = False
        self.finder = None           # 모든 물체: 이름 찾기 (YOLOE prompt-free)
        self.finder_loading = False
        self.worker = LabelWorker()
        self.session = None
        self.auto = False
        self.preview = False
        self.prompt_var = tk.StringVar()
        self.last_submit = 0.0
        self.last_submit_seq = -1
        self.batch_cancel = None
        self.batch_running = False
        self._stats_t = 0.0

        self.conf = tk.StringVar(value="0.4")
        self.imgsz = tk.StringVar(value="1008")
        self.interval = tk.StringVar(value="1.0")
        self.min_diff = tk.StringVar(value="6")
        self.empty_every = tk.StringVar(value="10")
        self.min_area = tk.StringVar(value="300")
        self.record = tk.BooleanVar(value=False)     # 원본 raw.mp4 녹화: 기본 꺼짐
        self.tag = tk.StringVar(value="live")
        self.every_sec = tk.StringVar(value="0.5")
        # 라벨 설정
        self.fmt = tk.StringVar(value="segment")         # segment / box
        self.target = tk.StringVar(value="fixed")        # fixed (classes.json) / open (모든 물체)
        self.finder_choice = tk.StringVar(value=FINDER_CHOICES[0])
        self.find_conf = tk.StringVar(value="0.25")
        self.max_names = tk.StringVar(value="6")
        self.verify = tk.BooleanVar(value=True)
        self.use_fixed = tk.BooleanVar(value=True)
        self.body = make_scrollable(self)
        self._build()
        self._show_classes()
        self.refresh_weights()

    # ---------- 위젯 ----------
    def _build(self):
        body = self.body
        f = ttk.LabelFrame(body, text="라벨 설정", padding=6)
        f.pack(fill="x")
        ttk.Label(f, text="라벨 형식").grid(row=0, column=0, sticky="w")
        for j, (v, t) in enumerate((("segment", "분할 (다각형)"), ("box", "박스"))):
            ttk.Radiobutton(f, text=t, value=v, variable=self.fmt, command=self._on_setting).grid(
                row=0, column=1 + j, sticky="w", padx=(4, 8))
        ttk.Label(f, text="라벨 대상").grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(f, text="지정 클래스", value="fixed", variable=self.target, command=self._on_setting).grid(
            row=1, column=1, sticky="w", padx=(4, 8))
        ttk.Radiobutton(f, text="모든 물체 (클래스 자동 생성)", value="open", variable=self.target,
                        command=self._on_setting).grid(row=1, column=2, columnspan=2, sticky="w", padx=(4, 0))
        o = ttk.Frame(f)
        o.grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Label(o, text="이름 찾기 모델").grid(row=0, column=0, sticky="w")
        cb = ttk.Combobox(o, textvariable=self.finder_choice, values=FINDER_CHOICES, state="readonly", width=22)
        cb.grid(row=0, column=1, columnspan=2, sticky="w", pady=1)
        cb.bind("<<ComboboxSelected>>", lambda e: self.load_finder() if self.target.get() == "open" else None)
        spin(o, 1, "찾기 점수 기준", self.find_conf, 0.05, 0.9, 0.05, help_text="낮추면 더 많은 이름 후보")
        spin(o, 2, "이름 수 (장당)", self.max_names, 1, 10, 1, help_text="SAM 3 확인은 이름이 많으면 느림")
        ttk.Checkbutton(o, text="SAM 3 로 확인 (틀린 이름 거르기·마스크 정확, 장당 약 0.5 s)",
                        variable=self.verify, command=self._on_setting).grid(row=3, column=0, columnspan=3, sticky="w")
        self.chk_use_fixed = ttk.Checkbutton(o, text="지정 클래스(아래 표)도 함께 찾기 (SAM 3 확인일 때)",
                                             variable=self.use_fixed)
        self.chk_use_fixed.grid(row=4, column=0, columnspan=3, sticky="w")
        self.open_widgets = o
        ttk.Label(f, text="모든 물체: YOLOE 가 물체와 이름(4,585종)을 찾고 SAM 3 가 그 이름으로 다시 찾아 확인합니다. "
                          "새 이름은 세션 클래스에 자동 추가. 박스 = 'cls cx cy w h' 라벨 (박스 모델로 학습)",
                  foreground="gray", wraplength=410).grid(row=3, column=0, columnspan=4, sticky="w", pady=(2, 0))

        f = ttk.LabelFrame(body, text="클래스 · SAM 3 프롬프트 (classes.json)", padding=6)
        f.pack(fill="x", pady=(6, 0))
        self.tree = ttk.Treeview(f, columns=("name", "prompts"), show="headings", height=4, selectmode="browse")
        self.tree.heading("name", text="클래스")
        self.tree.heading("prompts", text="프롬프트")
        self.tree.column("name", width=110)
        self.tree.column("prompts", width=280)
        self.tree.grid(row=0, column=0, columnspan=4, sticky="ew")
        self.tree.bind("<Double-1>", lambda e: self.edit_class())
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._fill_prompt_entry())
        ttk.Button(f, text="추가", command=self.add_class).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(f, text="수정", command=self.edit_class).grid(row=1, column=1, sticky="w", pady=(4, 0))
        ttk.Button(f, text="삭제", command=self.delete_class).grid(row=1, column=2, sticky="w", pady=(4, 0))
        p = ttk.Frame(f)
        p.grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self.prompt_label = ttk.Label(p, text="프롬프트:")
        self.prompt_label.pack(side="left")
        self.prompt_entry = ttk.Entry(p, textvariable=self.prompt_var, width=24, font=("Malgun Gothic", 10))
        self.prompt_entry.pack(side="left", padx=4)
        self.prompt_entry.bind("<Return>", lambda e: self.apply_prompt())
        self.prompt_entry.bind("<KP_Enter>", lambda e: self.apply_prompt())
        self.btn_preview = ttk.Button(p, text="미리보기 (P)", command=self.toggle_preview, width=16)
        self.btn_preview.pack(side="left")
        ttk.Label(f, text="선택한 클래스의 프롬프트를 바로 바꿈 (Enter, 쉼표로 여러 개, 모델 다시 불러오기 없음).\n"
                          "미리보기 = 저장 없이 계속 라벨해 화면에 마스크 표시. 클래스 이름을 바꾸면 새 세션",
                  foreground="gray", wraplength=410).grid(row=3, column=0, columnspan=4, sticky="w")

        f = ttk.LabelFrame(self.body, text="SAM 3 모델", padding=6)
        f.pack(fill="x", pady=(6, 0))
        ttk.Label(f, text="가중치").grid(row=0, column=0, sticky="w")
        self.weight_combo = ttk.Combobox(f, state="readonly", width=30)
        self.weight_combo.grid(row=0, column=1, columnspan=3, sticky="w", pady=1)
        spin(f, 1, "점수 기준", self.conf, 0.05, 0.95, 0.05, help_text="낮추면 더 많이 잡음 (오탐도 늘어남)")
        ttk.Label(f, text="입력 크기").grid(row=2, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.imgsz, values=["1008", "644"], width=7, state="readonly").grid(
            row=2, column=1, sticky="w", pady=1)
        ttk.Label(f, text="644 면 빠름 (작은 물체는 놓칠 수 있음)", foreground="gray").grid(row=2, column=2, sticky="w", padx=4)
        b = ttk.Frame(f)
        b.grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self.btn_load = ttk.Button(b, text="불러오기", command=self.load_model)
        self.btn_load.pack(side="left")
        ttk.Button(b, text="해제 (GPU 메모리 비우기)", command=self.unload_model).pack(side="left", padx=4)
        self.model_label = ttk.Label(f, text="모델 없음", foreground="gray", wraplength=400)
        self.model_label.grid(row=4, column=0, columnspan=4, sticky="w")

        f = ttk.LabelFrame(self.body, text="촬영하면서 자동 라벨 (위에서 연 카메라·영상)", padding=6)
        f.pack(fill="x", pady=(6, 0))
        b = ttk.Frame(f)
        b.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))
        self.btn_auto = ttk.Button(b, text="자동 라벨 시작 (A)", command=self.toggle_auto, width=18)
        self.btn_auto.pack(side="left")
        ttk.Button(b, text="지금 저장 (Space)", command=self.save_now).pack(side="left", padx=4)
        ttk.Button(b, text="취소 (U)", command=self.undo).pack(side="left")
        spin(f, 1, "라벨 간격(초)", self.interval, 0.2, 10, 0.1, help_text="자동일 때 이 간격으로 한 장")
        spin(f, 2, "장면 변화 기준", self.min_diff, 0, 50, 1, help_text="직전 저장과 이만큼 달라야 저장")
        spin(f, 3, "빈 장면 저장", self.empty_every, 0, 100, 1, help_text="N장마다 1장 (0 = 안 함)")
        spin(f, 4, "최소 크기(px)", self.min_area, 0, 100000, 50, help_text="이보다 작은 마스크는 버림")
        ttk.Checkbutton(f, text="원본 영상 raw.mp4 함께 저장 (카메라일 때)", variable=self.record).grid(
            row=5, column=0, columnspan=4, sticky="w")
        s = ttk.Frame(f)
        s.grid(row=6, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Label(s, text="세션 태그").pack(side="left")
        ttk.Entry(s, textvariable=self.tag, width=12).pack(side="left", padx=4)
        ttk.Button(s, text="새 세션", command=self.new_session).pack(side="left")
        ttk.Button(s, text="폴더 열기", command=lambda: open_folder(self.session.dir if self.session
                                                                    else SESSIONS_DIR)).pack(side="left", padx=4)
        self.session_label = ttk.Label(f, text="", wraplength=400)
        self.session_label.grid(row=7, column=0, columnspan=4, sticky="w")
        self.stats_label = ttk.Label(f, text="", foreground="gray", wraplength=400)
        self.stats_label.grid(row=8, column=0, columnspan=4, sticky="w")

        f = ttk.LabelFrame(self.body, text="영상 파일 일괄 라벨 (새 세션으로 저장)", padding=6)
        f.pack(fill="x", pady=(6, 0))
        spin(f, 0, "몇 초마다", self.every_sec, 0.05, 10, 0.05, help_text="장면 변화·빈 장면 기준은 위와 같음")
        b = ttk.Frame(f)
        b.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Button(b, text="영상 고르고 시작...", command=self.start_batch).pack(side="left")
        ttk.Button(b, text="중지", command=self.cancel_batch).pack(side="left", padx=4)
        self.batch_bar = ttk.Progressbar(f, length=380, maximum=100)
        self.batch_bar.grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self.batch_label = ttk.Label(f, text="", foreground="gray")
        self.batch_label.grid(row=3, column=0, columnspan=4, sticky="w")

    # ---------- 클래스 ----------
    def _show_classes(self, select=None):
        sel = self._selected_class() if select is None else select
        self.tree.delete(*self.tree.get_children())
        for i, c in enumerate(self.classes):
            self.tree.insert("", "end", iid=str(i), values=(c["name"], ", ".join(c["prompts"])))
        if self.classes:
            sel = 0 if sel is None or sel >= len(self.classes) else sel
            self.tree.selection_set(str(sel))
        self._fill_prompt_entry()

    def _selected_class(self):
        sel = self.tree.selection()
        return int(sel[0]) if sel else None

    def _fill_prompt_entry(self):
        i = self._selected_class()
        if i is None or i >= len(self.classes):
            self.prompt_label.configure(text="프롬프트 (새 클래스):")
            self.prompt_var.set("")
        else:
            self.prompt_label.configure(text=f"'{self.classes[i]['name']}' 프롬프트:")
            self.prompt_var.set(", ".join(self.classes[i]["prompts"]))

    def _set_classes(self, new, select=None):
        err = classes_cfg.check(new)
        if err:
            messagebox.showwarning("클래스", err)
            return False
        old_names = [c["name"] for c in self.classes]
        classes_cfg.save(new)
        self.classes = new
        self._show_classes(select)
        self.app.log("classes.json 저장: " + ", ".join(f"{c['name']}({'/'.join(c['prompts'])})" for c in new))
        if self.session:
            if [c["name"] for c in new] != old_names and self.session.class_mode == "fixed":
                self.close_session()
                self.app.log("클래스가 바뀌어 다음 저장부터 새 세션을 만듭니다.")
            else:          # 프롬프트만 바뀜: 같은 세션에서 계속 (session.json 에 기록)
                _, prompt_list, prompt_to_cls = self.flat()
                self.session.set_prompts(prompt_list, prompt_to_cls)
        self.touch()
        return True

    def apply_prompt(self):
        """프롬프트 입력 칸 Enter: 선택한 클래스의 프롬프트를 바로 바꿈 (클래스가 없으면 새로 만듦)"""
        prompts = [p.strip() for p in self.prompt_var.get().split(",") if p.strip()]
        if not prompts:
            return
        i = self._selected_class()
        if i is None or i >= len(self.classes):
            new, sel = self.classes + [dict(name=prompts[0], prompts=prompts)], len(self.classes)
        else:
            if prompts == self.classes[i]["prompts"]:
                return
            new, sel = [dict(x) for x in self.classes], i
            new[i] = dict(name=self.classes[i]["name"], prompts=prompts)
        if self._set_classes(new, select=sel):
            self.app.log(f"프롬프트 적용: {new[sel]['name']} <- {', '.join(prompts)}")
            if self.current_labeler() is not None and not self.auto and not self.preview:
                self.set_preview(True)

    def add_class(self):
        d = ClassDialog(self, "클래스 추가")
        if d.result:
            self._set_classes(self.classes + [dict(name=d.result[0], prompts=d.result[1])], select=len(self.classes))

    def edit_class(self):
        i = self._selected_class()
        if i is None:
            return
        c = self.classes[i]
        d = ClassDialog(self, "클래스 수정", c["name"], c["prompts"])
        if d.result:
            new = [dict(x) for x in self.classes]
            new[i] = dict(name=d.result[0], prompts=d.result[1])
            self._set_classes(new)

    def delete_class(self):
        i = self._selected_class()
        if i is None:
            return
        if messagebox.askyesno("클래스 삭제", f"'{self.classes[i]['name']}' 를 지울까요?\n(이미 저장한 라벨은 그대로입니다)"):
            self._set_classes([c for j, c in enumerate(self.classes) if j != i])

    def flat(self):
        return classes_cfg.flatten(self.classes)

    # ---------- 모델 ----------
    def refresh_weights(self):
        opts = list_sam3_weights() + [FAKE]
        self.weight_combo.configure(values=opts)
        self.weight_combo.current(0)
        if opts[0] == FAKE:
            self.model_label.configure(text=f"weights 폴더에 SAM 3 가중치가 없습니다 ({WEIGHTS_DIR})", foreground="red")

    def auto_load_model(self):
        """앱 시작 때: SAM 3 가중치가 있고 클래스가 올바르면 조용히 불러오기 (없으면 메시지만).
        카메라 열기(1080p 약 6 s)와 동시에 한다: 둘 다 약 6~7 s 에 준비 (카메라 다음에 하면 SAM 이 11 s)"""
        if self.weight_combo.get() == FAKE:
            self.app.log("SAM 3 자동 불러오기 건너뜀: weights 폴더에 가중치가 없습니다.")
            return
        err = classes_cfg.check(self.classes)
        if err:
            self.app.log(f"SAM 3 자동 불러오기 건너뜀: {err}")
            return
        self.app.log("SAM 3 자동 불러오기 (끄려면 run.bat --no-sam)")
        self.load_model()
        if self.target.get() == "open":
            self.load_finder()

    # ---------- 라벨 설정 ----------
    def _on_setting(self):
        fmt, target = self.fmt.get(), self.target.get()
        self.chk_use_fixed.state(["!disabled"] if self.verify.get() else ["disabled"])
        self.app.log(f"라벨 설정: 형식 {FORMAT_TEXT[fmt]}, 대상 {TARGET_TEXT[target]}"
                     + (f" ({self.finder_choice.get()}" + (" + SAM 3 확인)" if self.verify.get() else ", SAM 3 확인 없음)")
                        if target == "open" else ""))
        if target == "open" and self.finder is None:
            self.load_finder()
        if target == "open" and self.verify.get() and self.labeler is None and not self.loading:
            self.app.log("SAM 3 확인을 쓰려면 SAM 3 모델도 [불러오기] 하세요 (또는 'SAM 3 로 확인' 끄기).")
        s = self.session
        if s and (s.format != fmt or s.class_mode != target):
            self.close_session()
            self.app.log("라벨 설정이 바뀌어 다음 저장부터 새 세션을 만듭니다.")
        self.touch()

    def load_finder(self):
        """모든 물체 모드의 이름 찾기 모델(YOLOE prompt-free) 불러오기. weights\\ 에 없으면 내려받음"""
        if self.finder_loading:
            return
        choice = self.finder_choice.get()
        self.finder_loading = True
        self.finder = None
        self.app.log(f"이름 찾기 모델 불러오는 중: {choice}")

        def work():
            t0 = time.time()
            return ObjectFinder(choice), time.time() - t0

        def done(r):
            self.finder_loading = False
            self.finder = r[0]
            self.app.log(f"이름 찾기 모델 준비 ({r[1]:.1f}s): {self.finder.name}")
            self.touch()

        def fail(e):
            self.finder_loading = False
            self.app.log(f"[오류] 이름 찾기 모델 불러오기 실패: {e}")
        self.app.run_bg(work, done, fail)

    def current_labeler(self):
        """지금 설정으로 쓸 라벨러 (준비 안 됐으면 None)"""
        if self.target.get() == "fixed":
            return self.labeler
        if self.finder is None:
            return None
        sam = self.labeler if isinstance(self.labeler, Sam3Labeler) else None
        if self.verify.get() and sam is None:
            return None
        return OpenVocabLabeler(self.finder, sam if self.verify.get() else None,
                                find_conf=get_num(self.find_conf, 0.25, float, 0.01, 0.99),
                                max_names=int(get_num(self.max_names, 6, float, 1, 10)),
                                use_fixed=self.use_fixed.get())

    def _not_ready_reason(self):
        if self.target.get() == "open":
            if self.finder is None:
                return "이름 찾기 모델 " + ("불러오는 중..." if self.finder_loading else "이 없습니다 (라벨 대상을 다시 고르세요)")
            if self.verify.get() and not isinstance(self.labeler, Sam3Labeler):
                return "SAM 3 확인을 쓰려면 SAM 3 모델을 [불러오기] 하세요" + (" (불러오는 중)" if self.loading else "")
        return "먼저 SAM 3 모델을 [불러오기] 하세요." + (" (불러오는 중)" if self.loading else "")

    def load_model(self):
        if self.loading:
            return
        err = classes_cfg.check(self.classes)
        if err:
            messagebox.showwarning("클래스", err)
            return
        choice = self.weight_combo.get()
        conf = get_num(self.conf, 0.4, float, 0.01, 0.99)
        imgsz = int(get_num(self.imgsz, 1008, int))
        _, prompt_list, prompt_to_cls = self.flat()
        self.set_auto(False)
        self.labeler = None
        self.loading = True
        self.btn_load.state(["disabled"])
        self.model_label.configure(text=f"{choice} 불러오는 중... (처음엔 10~30초)", foreground="black")
        self.app.log(f"SAM 3 불러오는 중: {choice}, 점수 기준 {conf}, 입력 {imgsz}")

        def work():
            t0 = time.time()
            lab = make_labeler(choice, conf=conf, imgsz=imgsz)
            lab.detect(np.zeros((640, 640, 3), np.uint8), prompt_list, prompt_to_cls)     # 예열 (첫 호출이 느림)
            return lab, time.time() - t0

        def done(r):
            self.loading = False
            self.btn_load.state(["!disabled"])
            self.labeler = r[0]
            self.model_label.configure(text=f"준비: {self.labeler.name} ({r[1]:.1f}s)", foreground="green")
            mode = getattr(self.labeler, "load_mode", "")
            self.app.log(f"SAM 3 준비 ({r[1]:.1f}s{', ' + mode if mode else ''}): {self.labeler.name}")
            self.touch()

        def fail(e):
            self.loading = False
            self.btn_load.state(["!disabled"])
            self.model_label.configure(text=f"불러오기 실패: {e}", foreground="red")
            self.app.log(f"[오류] SAM 3 불러오기 실패: {e}")
        self.app.run_bg(work, done, fail)

    def unload_model(self):
        if self.labeler is None:
            return
        self.set_auto(False)
        self.labeler = None
        self.model_label.configure(text="모델 없음 (해제함)", foreground="gray")
        self.touch()

        def work():
            import gc

            import torch
            while self.worker.busy:
                time.sleep(0.05)
            gc.collect()
            torch.cuda.empty_cache()
        self.app.run_bg(work, lambda _: self.app.log("SAM 3 해제, GPU 메모리 비움"))

    # ---------- 세션 ----------
    def params(self):
        return dict(min_diff=get_num(self.min_diff, 6.0, float, 0),
                    min_area=int(get_num(self.min_area, 300, float, 0)),
                    empty_every=int(get_num(self.empty_every, 10, float, 0)),
                    format=self.fmt.get())

    def make_session(self, tag, labeler, source_text):
        names, prompt_list, prompt_to_cls = self.flat()
        target = self.target.get()
        return Session(names if target == "fixed" else [], tag=tag, fmt=self.fmt.get(), class_mode=target,
                       prompt_list=prompt_list, prompt_to_cls=prompt_to_cls,
                       meta=dict(source=source_text, backend=labeler.name, params=self.params()))

    def ensure_session(self, labeler):
        names, prompt_list, prompt_to_cls = self.flat()
        s, target = self.session, self.target.get()
        if s and s.format == self.fmt.get() and s.class_mode == target and (target == "open" or s.names == names):
            if s.prompt_list != prompt_list:
                s.set_prompts(prompt_list, prompt_to_cls)
            return s
        self.close_session()
        tag = re.sub(r"[^\w\-]+", "_", self.tag.get().strip()) or "live"
        src = self.app.source
        self.session = self.make_session(tag, labeler, src.info)
        self.worker.reset_similar()
        msg = (f"새 세션: {self.session.name} ({FORMAT_TEXT[self.session.format]}, "
               f"{TARGET_TEXT[self.session.class_mode]})")
        if self.record.get() and src.kind == "camera":
            src.start_recording(os.path.join(self.session.dir, "raw.mp4"))
            msg += " (원본 녹화 raw.mp4)"
        self.app.log(msg)
        return self.session

    def close_session(self):
        s = self.session
        if not s:
            return
        self.set_auto(False)
        raw = self.app.source.stop_recording()
        s.write_meta(stats=dict(self.worker.stats))
        self.app.log(f"세션 닫음 {s.name}: {s.count}장 (빈 {s.empty}), "
                     + ", ".join(f"{n} {c}" for n, c in zip(s.names, s.instances))
                     + (", raw.mp4 저장" if raw else ""))
        self.session = None
        self.touch()

    def new_session(self):
        self.close_session()
        self.app.log("다음 저장부터 새 세션을 만듭니다.")

    def on_source_closing(self):
        self.close_session()

    # ---------- 라벨 ----------
    def _ready(self):
        """-> (seq, frame, 라벨러) 또는 None (이유를 메시지로)"""
        lab = self.current_labeler()
        if lab is None:
            self.app.log(self._not_ready_reason())
            return None
        seq, frame = self.app.source.latest()
        if frame is None:
            self.app.log("영상이 없습니다. 위에서 카메라나 영상 파일을 여세요.")
            return None
        return seq, frame, lab

    def set_auto(self, on):
        if self.auto == on:
            return
        self.auto = on
        self.btn_auto.configure(text="자동 라벨 멈춤 (A)" if on else "자동 라벨 시작 (A)")
        self.app.log(f"자동 라벨 {'켜짐' if on else '꺼짐'}")
        self.touch()

    def set_preview(self, on):
        if self.preview == on:
            return
        self.preview = on
        self.btn_preview.configure(text="미리보기 끄기 (P)" if on else "미리보기 (P)")
        self.app.log("미리보기 켜짐 (저장 안 함, 프롬프트를 바꾸면 바로 반영)" if on else "미리보기 꺼짐")
        self.touch()

    def toggle_preview(self):
        if self.preview:
            self.set_preview(False)
        elif self._ready():
            self.set_preview(True)

    def toggle_auto(self):
        if self.auto:
            self.set_auto(False)
        else:
            r = self._ready()
            if r:
                self.ensure_session(r[2])
                self.last_submit = 0.0
                self.set_auto(True)

    def save_now(self):
        r = self._ready()
        if not r:
            return
        s = self.ensure_session(r[2])
        if not self.worker.submit(r[1], True, r[2], s, self.flat(), self.params()):
            self.app.log("SAM 처리 중 - 잠시 후 다시 누르세요.")

    def undo(self):
        if not self.session:
            self.app.log("취소할 저장이 없습니다.")
            return
        name = self.session.undo()
        self.worker.reset_similar()
        self.app.log(f"저장 취소: {name}" if name else "취소할 저장이 없습니다.")
        self.touch()

    def on_key(self, k):
        if k in ("a", "A"):
            self.toggle_auto()
        elif k == "space":
            self.save_now()
        elif k in ("u", "U"):
            self.undo()
        elif k in ("p", "P"):
            self.toggle_preview()
        else:
            return False
        return True

    def poll(self):
        while not self.worker.errors.empty():
            self.app.log(f"[라벨 오류] {self.worker.errors.get()}")
        now = time.time()
        busy = self.worker.busy
        if self.auto and self.session is not None and not busy:
            if now - self.last_submit >= get_num(self.interval, 1.0, float, 0.05):
                seq, frame = self.app.source.latest()
                lab = self.current_labeler() if frame is not None and seq != self.last_submit_seq else None
                if lab is not None:
                    if self.worker.submit(frame, False, lab, self.ensure_session(lab), self.flat(), self.params()):
                        self.last_submit, self.last_submit_seq = now, seq
        elif self.preview and not busy:      # 쉬지 않고 계속
            seq, frame = self.app.source.latest()
            lab = self.current_labeler() if frame is not None and seq != self.last_submit_seq else None
            if lab is not None:
                if self.worker.submit(frame, False, lab, None, self.flat(), self.params(), preview=True):
                    self.last_submit_seq = seq
        if now - self._stats_t > 0.3:
            self._stats_t = now
            s = self.session
            if s:
                self.session_label.configure(text=f"세션 {s.name}: {s.count}장 (빈 {s.empty})  "
                                             + "  ".join(f"{n} {c}" for n, c in zip(s.names, s.instances)))
            else:
                self.session_label.configure(text="세션 없음 (첫 저장 때 만듦)")
            st = self.worker.stats
            self.stats_label.configure(text=f"라벨 {st['labeled']}  저장 {st['saved']}  비슷해서 건너뜀 {st['skip_similar']}  "
                                            f"빈 장면 건너뜀 {st['skip_empty']}  마지막 {st['last_n']}개 {st['last_ms']:.0f} ms")

    # ---------- 화면 ----------
    def render_key(self, seq):
        return seq, self.version, self.worker.result_seq, self.worker.busy, self.auto

    def compose(self, frame):
        lines = []
        if self.current_labeler() is None:
            lines.append((self._not_ready_reason(), ORANGE))
        elif self.auto:
            lines.append(("자동 라벨 켜짐" + ("  [SAM 처리 중]" if self.worker.busy else ""), GREEN))
        elif self.preview:
            lines.append(("미리보기 (저장 안 함)  —  Space 지금 저장, A 자동 라벨", YELLOW))
        else:
            lines.append(("자동 라벨 꺼짐  —  P 미리보기, A 자동 라벨, Space 지금 저장", ORANGE))
        fixed_text = "   ".join(f"{c['name']} = {' / '.join(c['prompts'])}" for c in self.classes)
        if self.target.get() == "open":
            how = "SAM 3 확인" if self.verify.get() else "SAM 3 확인 없음"
            lines.append(f"[{FORMAT_TEXT[self.fmt.get()]}] 모든 물체 ({self.finder_choice.get()}, {how})"
                         + (f"  +  {fixed_text}" if self.use_fixed.get() and self.verify.get() else ""))
        else:
            lines.append(f"[{FORMAT_TEXT[self.fmt.get()]}] 프롬프트: {fixed_text}")
        s = self.session
        if s:
            inst = sorted(zip(s.names, s.instances), key=lambda x: -x[1])[:10]
            lines.append(f"{s.name}  저장 {s.count}장  클래스 {len(s.names)}개  "
                         + "  ".join(f"{n} {c}" for n, c in inst))
        r = self.worker.result
        if self.preview and not self.auto and r is not None and r.get("preview"):
            # 미리보기: 마지막 결과 마스크를 지금 화면 위에 (가만히 있는 물체 기준, 0.2초쯤 늦음)
            lines.append((f"{len(r['kept'])}개  {r['ms']:.0f} ms", GREEN))
            return dict(img=frame, items=r["kept"], names=r["names"], lines=lines)
        if self.worker.message:
            lines.append((self.worker.message, YELLOW))
        thumb = None
        if r is not None:
            thumb = dict(img=r["frame"], items=r["kept"], names=r["names"],
                         lines=[f"마지막 라벨 {len(r['kept'])}개 {r['ms']:.0f} ms" + ("" if r["saved"] else " (저장 안 함)")])
        return dict(img=frame, lines=lines, thumb=thumb)

    # ---------- 영상 파일 일괄 ----------
    def start_batch(self):
        if self.batch_running:
            self.app.log("영상 일괄 라벨이 이미 진행 중입니다.")
            return
        if self.current_labeler() is None:
            self.app.log(self._not_ready_reason())
            return
        path = filedialog.askopenfilename(title="라벨할 영상", initialdir=SESSIONS_DIR,
                                          filetypes=[("영상", "*.mp4 *.avi *.mkv *.mov"), ("모두", "*.*")])
        if not path:
            return
        self.run_batch(path)

    def run_batch(self, path):
        flat, labeler, params = self.flat(), self.current_labeler(), self.params()
        if labeler is None:
            self.app.log(self._not_ready_reason())
            return
        every = get_num(self.every_sec, 0.5, float, 0.01)
        stem = re.sub(r"[^\w\-]+", "_", os.path.splitext(os.path.basename(path))[0])[:24]
        session = self.make_session(f"video_{stem}", labeler, path)
        self.batch_cancel = threading.Event()
        self.batch_running = True
        self.batch_bar.configure(value=0)
        self.app.log(f"영상 일괄 라벨 시작: {os.path.basename(path)}, {every}초마다 -> {session.name}")

        def progress(i, total, saved, ms):
            self.app.post(self._batch_progress, i, total, saved, ms)

        def done(saved):
            self.batch_running = False
            self.batch_bar.configure(value=100)
            word = "중지" if self.batch_cancel.is_set() else "끝"
            self.batch_label.configure(text=f"{word}: {saved}장 저장 -> {session.name}")
            self.app.log(f"영상 일괄 라벨 {word}: {saved}장 (빈 {session.empty}), "
                         + ", ".join(f"{n} {c}" for n, c in zip(session.names, session.instances))
                         + f" -> {session.dir}")

        def fail(e):
            self.batch_running = False
            self.app.log(f"[오류] 영상 일괄 라벨: {e}")
        self.app.run_bg(lambda: label_video(path, labeler, session, flat, params, every, progress, self.batch_cancel),
                        done, fail)

    def _batch_progress(self, i, total, saved, ms):
        if total:
            self.batch_bar.configure(value=100 * i / total)
        self.batch_label.configure(text=f"{i}/{total} 프레임, 저장 {saved}장, 평균 {ms:.0f} ms/장")

    def cancel_batch(self):
        if self.batch_running and self.batch_cancel:
            self.batch_cancel.set()
            self.app.log("영상 일괄 라벨 중지 요청")

    def busy_text(self):
        return "영상 일괄 라벨이 진행 중입니다." if self.batch_running else ""

    def shutdown(self):
        self.cancel_batch()
        self.close_session()
        self.worker.stop()
