"""탭 기본 클래스와 작은 위젯 도우미."""
import os
import tkinter as tk
from tkinter import ttk


class BaseTab(ttk.Frame):
    """단계 탭 하나. App 이 아래 함수를 부른다 (모두 GUI 스레드).

    render_key(seq) : 이 값이 바뀔 때만 compose() 를 다시 부른다 (화면 다시 그리기 줄이기)
    compose(frame)  : 영상 화면에 보낼 dict (VideoView.show 인자) 또는 None
    poll()          : 매 tick (작업 스레드 상태 반영)
    on_key(keysym)  : 단축키. 처리했으면 True
    on_click(x, y)  : 영상 화면 클릭 (원본 좌표)
    on_show()       : 탭이 보일 때
    shutdown()      : 끝낼 때
    """

    TITLE = ""

    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app
        self._render_key = None
        self.version = 0          # 화면을 다시 그려야 할 때 올림

    def touch(self):
        self.version += 1

    def render(self, seq, frame, force=False):
        key = self.render_key(seq)
        if key == self._render_key and not force:
            return None
        self._render_key = key
        return self.compose(frame)

    def render_key(self, seq):
        return seq, self.version

    def compose(self, frame):
        return dict(img=frame)

    def poll(self):
        pass

    def on_key(self, keysym):
        return False

    def on_click(self, x, y):
        pass

    def on_show(self):
        pass

    def on_source_closing(self):
        """소스를 닫기 직전 (녹화 정리 등)"""

    def busy_text(self):
        """끝낼 때 확인이 필요한 작업이 있으면 설명, 없으면 빈 문자열"""
        return ""

    def shutdown(self):
        pass


def make_scrollable(parent):
    """세로 스크롤되는 영역을 만들고 안쪽 Frame 을 돌려준다 (설정이 화면보다 길 때). 마우스 휠로 스크롤."""
    bg = ttk.Style().lookup("TFrame", "background") or None
    canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0, background=bg)
    sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)
    win = canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
    canvas.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    def wheel(e):
        if inner.winfo_height() > canvas.winfo_height():
            canvas.yview_scroll(int(-e.delta / 120), "units")
    inner.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", wheel))
    inner.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
    return inner


def spin(parent, row, text, var, frm, to, inc, width=7, col=0, help_text=""):
    ttk.Label(parent, text=text).grid(row=row, column=col, sticky="w", padx=(0, 4), pady=1)
    sb = ttk.Spinbox(parent, textvariable=var, from_=frm, to=to, increment=inc, width=width)
    sb.grid(row=row, column=col + 1, sticky="w", pady=1)
    if help_text:
        ttk.Label(parent, text=help_text, foreground="gray").grid(row=row, column=col + 2, sticky="w", padx=4)
    return sb


def get_num(var, default, cast=float, lo=None, hi=None):
    """Spinbox 값 읽기 (잘못 입력했으면 default)"""
    try:
        v = cast(var.get())
    except (tk.TclError, ValueError):
        return default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def open_folder(path):
    if path and os.path.exists(path):
        os.startfile(path)
