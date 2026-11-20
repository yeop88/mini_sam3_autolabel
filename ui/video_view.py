"""영상 화면: 창 크기에 맞춰 축소해 보여 주고, 다각형·글자를 그린 뒤, 클릭을 원본 좌표로 바꿔 알린다."""
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from core import render


class VideoView(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, bg="black", highlightthickness=0, **kw)
        self._item = self.create_image(0, 0, anchor="nw")
        self._photo = None
        self.map = None            # (scale, ox, oy, w, h)
        self.click_handler = None
        self.bind("<Button-1>", self._on_click)

    def _on_click(self, e):
        self.focus_set()          # 입력 칸에서 포커스를 빼서 단축키가 먹게
        if self.map and self.click_handler:
            s, ox, oy, w, h = self.map
            x, y = (e.x - ox) / s, (e.y - oy) / s
            if 0 <= x < w and 0 <= y < h:
                self.click_handler(x, y)

    def show(self, img=None, items=None, names=None, lines=None, thumb=None, selected=None, message=""):
        """img: 원본 BGR. items: dict(cls, poly(원본 좌표), score?). lines: 왼쪽 위 글자 [str 또는 (str, BGR)].
        thumb: dict(img, items, names, lines) — 오른쪽 위 작은 화면 (마지막 라벨 결과 등)"""
        W, H = max(self.winfo_width(), 50), max(self.winfo_height(), 50)
        texts = []
        if img is None:
            canvas = np.zeros((H, W, 3), np.uint8)
            if message:
                texts.append((20, H // 2 - 12, message, (200, 200, 200), 18))
            ox = oy = 0
            self.map = None
        else:
            h, w = img.shape[:2]
            s = min(W / w, H / h)
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
            canvas = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            if items:
                canvas = render.draw_items(canvas, items, names or [], s, texts, selected=selected)
            if thumb and thumb.get("img") is not None:
                canvas = canvas.copy()
                self._paste_thumb(canvas, thumb, texts)
            ox, oy = (W - nw) // 2, (H - nh) // 2
            self.map = (s, ox, oy, w, h)
        rows = [(line, (255, 255, 255)) if isinstance(line, str) else line for line in (lines or [])]
        canvas = render.draw_texts(canvas, texts, panel=(4, 4, rows, 16))
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
        self.coords(self._item, ox, oy)
        self.itemconfigure(self._item, image=self._photo)

    @staticmethod
    def _paste_thumb(canvas, thumb, texts):
        ch, cw = canvas.shape[:2]
        src = thumb["img"]
        tw = max(80, cw // 3)
        ts = tw / src.shape[1]
        th = min(ch, max(1, int(src.shape[0] * ts)))
        small = cv2.resize(src, (tw, th), interpolation=cv2.INTER_AREA)
        x0 = cw - tw
        small = render.draw_items(small, thumb.get("items"), thumb.get("names") or [], ts, texts,
                                  offset=(x0, 0), text_size=12)
        canvas[0:th, x0:cw] = small
        cv2.rectangle(canvas, (x0, 0), (cw - 1, th - 1), (255, 255, 255), 1)
        for i, line in enumerate(thumb.get("lines") or []):
            texts.append((x0 + 6, th - 22 - i * 18, line, (255, 255, 255), 13))
