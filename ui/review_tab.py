"""2. 검토 탭 — 저장된 라벨(학습에 실제로 쓰일 다각형)을 보고 틀린 것을 고친다.

- 이 장 빼기: images, labels 를 세션의 rejected\\ 로 옮김 (학습에서 빠짐)
- 객체 지우기 / 클래스 바꾸기: 라벨 파일을 바로 고침 (화면에서 객체를 클릭하면 선택)
- 되돌리기: 빼기·지우기·바꾸기를 마지막 것부터 되돌림
단축키: ← / → (또는 A / D, Space) 이전·다음, X 빼기, Z 되돌리기, Delete 선택 객체 지우기
"""
import os
import shutil
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

from core.paths import SESSIONS_DIR
from core.session import label_path, list_sessions, read_labels, session_images, write_labels
from core.util import read_json
from ui.common import BaseTab, open_folder


def _read_text(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


class ReviewTab(BaseTab):
    TITLE = " 2 검토 "

    def __init__(self, master, app):
        super().__init__(master, app)
        self.sessions = []
        self.sdir = None
        self.names = []
        self.imgs = []
        self.idx = 0
        self.img = None
        self.objs = []
        self.fmt = "segment"
        self.selected = None
        self.undo_stack = []
        self._build()

    # ---------- 위젯 ----------
    def _build(self):
        f = ttk.LabelFrame(self, text="세션", padding=6)
        f.pack(fill="x")
        self.combo = ttk.Combobox(f, state="readonly", width=52)
        self.combo.grid(row=0, column=0, columnspan=3, sticky="w")
        self.combo.bind("<<ComboboxSelected>>", lambda e: self.load_session(self.sessions[self.combo.current()]["dir"]))
        ttk.Button(f, text="새로고침", command=self.refresh_sessions).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(f, text="폴더 열기", command=lambda: open_folder(self.sdir or SESSIONS_DIR)).grid(
            row=1, column=1, sticky="w", pady=(4, 0))
        self.info = ttk.Label(f, text="", wraplength=400)
        self.info.grid(row=2, column=0, columnspan=3, sticky="w")

        f = ttk.LabelFrame(self, text="이미지", padding=6)
        f.pack(fill="x", pady=(6, 0))
        ttk.Button(f, text="◀ 이전 (←)", command=lambda: self.move(-1)).grid(row=0, column=0, sticky="w")
        ttk.Button(f, text="다음 ▶ (→)", command=lambda: self.move(1)).grid(row=0, column=1, sticky="w", padx=4)
        self.counter = ttk.Label(f, text="")
        self.counter.grid(row=0, column=2, sticky="w")
        self.scale = ttk.Scale(f, from_=0, to=1, orient="horizontal", length=390, command=self._on_scale)
        self.scale.grid(row=1, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Button(f, text="이 장 빼기 (X)", command=self.reject).grid(row=2, column=0, sticky="w")
        ttk.Button(f, text="되돌리기 (Z)", command=self.undo).grid(row=2, column=1, sticky="w", padx=4)
        ttk.Label(f, text="뺀 장은 세션의 rejected 폴더로 옮겨져 학습에서 빠집니다.", foreground="gray").grid(
            row=3, column=0, columnspan=3, sticky="w")

        f = ttk.LabelFrame(self, text="이 장의 객체 (영상 화면을 클릭해도 선택)", padding=6)
        f.pack(fill="x", pady=(6, 0))
        self.listbox = tk.Listbox(f, height=8, width=52, exportselection=False, font=("Malgun Gothic", 9))
        self.listbox.grid(row=0, column=0, columnspan=3, sticky="w")
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)
        ttk.Button(f, text="선택 객체 지우기 (Del)", command=self.delete_obj).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.cls_combo = ttk.Combobox(f, state="readonly", width=14)
        self.cls_combo.grid(row=1, column=1, sticky="w", padx=4, pady=(4, 0))
        ttk.Button(f, text="클래스 바꾸기", command=self.change_class).grid(row=1, column=2, sticky="w", pady=(4, 0))

    # ---------- 세션 ----------
    def on_show(self):
        self.refresh_sessions()

    def refresh_sessions(self):
        self.sessions = list_sessions()
        self.combo.configure(values=[f"{s['name']}  ({s['images']}장, {'박스' if s['format'] == 'box' else '분할'}, "
                                     f"클래스 {len(s['names'])}개)" for s in self.sessions])
        dirs = [s["dir"] for s in self.sessions]
        if self.sdir in dirs:
            self.combo.current(dirs.index(self.sdir))
            self.load_session(self.sdir, keep_index=True)
        elif dirs:
            self.combo.current(len(dirs) - 1)
            self.load_session(dirs[-1])
        else:
            self.combo.set("")
            self.sdir, self.imgs, self.img = None, [], None
            self.info.configure(text="세션이 없습니다. 1 라벨링 탭에서 먼저 저장하세요.")
            self.touch()

    def load_session(self, sdir, keep_index=False):
        if sdir != self.sdir:
            self.undo_stack = []
            keep_index = False
        self.sdir = sdir
        meta = read_json(os.path.join(sdir, "session.json"), {}) or {}
        self.names = meta.get("names", [])
        self.fmt = meta.get("format", "segment")       # 고쳐 쓸 때도 이 형식으로
        self.cls_combo.configure(values=self.names)
        if self.names:
            self.cls_combo.current(0)
        self.imgs = session_images(sdir)
        if not keep_index:
            self.idx = 0
        self.load_image()
        self._update_info()

    def _update_info(self):
        if not self.sdir:
            return
        inst = [0] * len(self.names)
        empty = 0
        for ip in self.imgs:
            t = _read_text(label_path(ip)) or ""
            rows = [ln for ln in t.splitlines() if ln.strip()]
            empty += int(not rows)
            for ln in rows:
                c = int(ln.split()[0])
                if 0 <= c < len(inst):
                    inst[c] += 1
        n_rej = len([p for p in os.listdir(os.path.join(self.sdir, "rejected", "images"))]) \
            if os.path.isdir(os.path.join(self.sdir, "rejected", "images")) else 0
        top = sorted(zip(self.names, inst), key=lambda x: -x[1])
        self.info.configure(text=f"{len(self.imgs)}장 (빈 장면 {empty}), 뺀 장 {n_rej}, "
                                 f"{'박스' if self.fmt == 'box' else '분할(다각형)'}, 클래스 {len(self.names)}개\n객체: "
                            + ", ".join(f"{n} {c}" for n, c in top[:15]) + (" ..." if len(top) > 15 else ""))

    # ---------- 이미지 ----------
    def load_image(self):
        self.selected = None
        if not self.imgs:
            self.img, self.objs = None, []
        else:
            self.idx = min(max(0, self.idx), len(self.imgs) - 1)
            ip = self.imgs[self.idx]
            self.img = cv2.imread(ip)
            if self.img is None:
                self.objs = []
            else:
                h, w = self.img.shape[:2]
                self.objs = read_labels(label_path(ip), w, h)
        n = len(self.imgs)
        self.counter.configure(text=f"{self.idx + 1} / {n}" if n else "0 / 0")
        self.scale.configure(to=max(1, n - 1))
        self._scale_lock = True
        self.scale.set(self.idx)
        self._scale_lock = False
        self._fill_list()
        self.touch()

    def _fill_list(self):
        self.listbox.delete(0, "end")
        for i, (c, poly) in enumerate(self.objs):
            area = abs(cv2.contourArea(np.asarray(poly, np.float32)))
            name = self.names[c] if 0 <= c < len(self.names) else str(c)
            self.listbox.insert("end", f"{i + 1}. {name}   (넓이 {area:.0f} px, 점 {len(poly)}개)")
        if self.selected is not None:
            self.listbox.selection_set(self.selected)

    def _on_scale(self, v):
        if getattr(self, "_scale_lock", False) or not self.imgs:
            return
        i = int(round(float(v)))
        if i != self.idx:
            self.idx = i
            self.load_image()

    def move(self, d):
        if self.imgs:
            self.idx = min(max(0, self.idx + d), len(self.imgs) - 1)
            self.load_image()

    def _on_list_select(self, _e=None):
        sel = self.listbox.curselection()
        self.selected = sel[0] if sel else None
        if self.selected is not None and self.names:
            c = self.objs[self.selected][0]
            if 0 <= c < len(self.names):
                self.cls_combo.current(c)
        self.touch()

    def on_click(self, x, y):
        hits = []
        for i, (_c, poly) in enumerate(self.objs):
            p = np.asarray(poly, np.float32)
            if cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0:
                hits.append((abs(cv2.contourArea(p)), i))
        self.listbox.selection_clear(0, "end")
        if hits:
            self.listbox.selection_set(min(hits)[1])
        self._on_list_select()

    # ---------- 고치기 ----------
    def reject(self):
        if not self.imgs:
            return
        ip = self.imgs[self.idx]
        lp = label_path(ip)
        rej_img = os.path.join(self.sdir, "rejected", "images")
        rej_lbl = os.path.join(self.sdir, "rejected", "labels")
        os.makedirs(rej_img, exist_ok=True)
        os.makedirs(rej_lbl, exist_ok=True)
        shutil.move(ip, os.path.join(rej_img, os.path.basename(ip)))
        if os.path.isfile(lp):
            shutil.move(lp, os.path.join(rej_lbl, os.path.basename(lp)))
        self.undo_stack.append(("reject", ip, lp))
        self.imgs.pop(self.idx)
        self.app.log(f"뺌: {os.path.basename(ip)}")
        self.load_image()
        self._update_info()

    def _write_objs(self, old_text):
        ip = self.imgs[self.idx]
        h, w = self.img.shape[:2]
        write_labels(label_path(ip), self.objs, w, h, self.fmt)
        self.undo_stack.append(("label", ip, old_text))

    def delete_obj(self):
        if self.selected is None or self.img is None:
            return
        old = _read_text(label_path(self.imgs[self.idx]))
        c, _ = self.objs.pop(self.selected)
        self._write_objs(old)
        self.app.log(f"객체 지움: {self.names[c] if c < len(self.names) else c} ({os.path.basename(self.imgs[self.idx])})")
        self.selected = None
        self._fill_list()
        self._update_info()
        self.touch()

    def change_class(self):
        new = self.cls_combo.current()
        if self.selected is None or self.img is None or new < 0 or self.objs[self.selected][0] == new:
            return
        old = _read_text(label_path(self.imgs[self.idx]))
        _, poly = self.objs[self.selected]
        self.objs[self.selected] = (new, poly)
        self._write_objs(old)
        self.app.log(f"클래스 바꿈 -> {self.names[new]} ({os.path.basename(self.imgs[self.idx])})")
        self._fill_list()
        self._update_info()
        self.touch()

    def undo(self):
        if not self.undo_stack:
            self.app.log("되돌릴 것이 없습니다.")
            return
        kind, ip, extra = self.undo_stack.pop()
        if kind == "reject":
            lp = extra
            shutil.move(os.path.join(self.sdir, "rejected", "images", os.path.basename(ip)), ip)
            rl = os.path.join(self.sdir, "rejected", "labels", os.path.basename(lp))
            if os.path.isfile(rl):
                shutil.move(rl, lp)
            self.imgs = session_images(self.sdir)
            self.app.log(f"되돌림 (빼기): {os.path.basename(ip)}")
        else:
            lp = label_path(ip)
            if extra is None:
                if os.path.isfile(lp):
                    os.remove(lp)
            else:
                with open(lp, "w", encoding="utf-8") as f:
                    f.write(extra)
            self.app.log(f"되돌림 (라벨): {os.path.basename(ip)}")
        if ip in self.imgs:
            self.idx = self.imgs.index(ip)
        self.load_image()
        self._update_info()

    def on_key(self, k):
        if k in ("Left", "a", "A"):
            self.move(-1)
        elif k in ("Right", "d", "D", "space"):
            self.move(1)
        elif k in ("x", "X"):
            self.reject()
        elif k in ("z", "Z"):
            self.undo()
        elif k in ("Delete", "BackSpace"):
            self.delete_obj()
        else:
            return False
        return True

    # ---------- 화면 ----------
    def render_key(self, seq):
        return self.version          # 카메라 프레임과 상관없음

    def compose(self, frame):
        if self.img is None:
            return dict(img=None, message="검토할 이미지가 없습니다. (1 라벨링 탭에서 저장한 세션을 고르세요)")
        items = [dict(cls=c, poly=p) for c, p in self.objs]
        cnt = [sum(1 for c, _ in self.objs if c == k) for k in range(len(self.names))]
        lines = [f"[{self.idx + 1}/{len(self.imgs)}] {os.path.basename(self.imgs[self.idx])}",
                 "  ".join(f"{n} {c}" for n, c in zip(self.names, cnt) if c) + ("(빈 장면)" if not self.objs else "")]
        return dict(img=self.img, items=items, names=self.names, lines=lines, selected=self.selected)
