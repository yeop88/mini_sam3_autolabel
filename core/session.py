"""세션 폴더: sessions/<시각>_<tag>/ 에 images/, labels/, session.json, raw.mp4, rejected/

라벨 형식 (session.json 의 format)
  segment : YOLO-seg  'cls x1 y1 x2 y2 ...' (다각형, 0~1)
  box     : YOLO 박스 'cls cx cy w h' (0~1)
클래스 방식 (class_mode)
  fixed : classes.json 의 클래스 (names 고정)
  open  : 모든 물체 — 처음 보는 이름이 나오면 names 끝에 추가 (번호는 바뀌지 않음)
"""
import glob
import os
import threading
import time

import cv2
import numpy as np

from core.paths import SESSIONS_DIR
from core.util import read_json, stamp, write_json

FORMATS = ("segment", "box")


def poly_line(cid, poly, w, h):
    xy = np.clip(np.asarray(poly, np.float32) / np.array([w, h], np.float32), 0, 1)
    return f"{cid} " + " ".join(f"{v:.6f}" for v in xy.reshape(-1))


def box_line(cid, poly, w, h):
    p = np.asarray(poly, np.float32)
    x0, y0 = np.clip(p.min(axis=0), 0, [w, h])
    x1, y1 = np.clip(p.max(axis=0), 0, [w, h])
    return f"{cid} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"


class Session:
    def __init__(self, names, tag="live", meta=None, fmt="segment", class_mode="fixed",
                 prompt_list=None, prompt_to_cls=None):
        self.name = f"{stamp()}_{tag}" if tag else stamp()
        self.dir = os.path.join(SESSIONS_DIR, self.name)
        self.img_dir = os.path.join(self.dir, "images")
        self.lbl_dir = os.path.join(self.dir, "labels")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)
        self.names = list(names)
        self.format = fmt if fmt in FORMATS else "segment"
        self.class_mode = class_mode
        self.prompt_list, self.prompt_to_cls = list(prompt_list or []), list(prompt_to_cls or [])
        self.meta = dict(meta or {})
        self.count = 0
        self.empty = 0
        self.instances = [0] * len(self.names)
        self.saved = []          # 저장 순서 (되돌리기용)
        self.lock = threading.Lock()
        self.write_meta()

    def _class_id(self, name):
        if name not in self.names:      # 모든 물체 모드: 새 이름 -> 새 클래스
            self.names.append(name)
            self.instances.append(0)
        return self.names.index(name)

    def save(self, frame, kept):
        """kept: dict(name, poly) list. 세션 형식대로 라벨을 쓴다. -> 파일 이름(확장자 없음)"""
        h, w = frame.shape[:2]
        with self.lock:
            n_before = len(self.names)
            cids, lines = [], []
            for it in kept:
                cid = self._class_id(it["name"])
                cids.append(cid)
                lines.append((box_line if self.format == "box" else poly_line)(cid, it["poly"], w, h))
            stem = f"{self.name}_{self.count:06d}"
            img_path = os.path.join(self.img_dir, stem + ".jpg")
            lbl_path = os.path.join(self.lbl_dir, stem + ".txt")
            cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            self.count += 1
            self.empty += int(not lines)
            for c in cids:
                self.instances[c] += 1
            self.saved.append((img_path, lbl_path, cids, not lines))
            if self.count % 20 == 0 or len(self.names) != n_before:
                self._write_meta_locked()
        return stem

    def set_prompts(self, prompt_list, prompt_to_cls):
        """클래스는 그대로, 프롬프트만 바뀜 (실시간 입력). 바뀐 기록을 session.json 의 prompt_history 에 남긴다."""
        with self.lock:
            hist = self.meta.setdefault("prompt_history", [])
            hist.append(dict(from_image=self.count, prompts=list(prompt_list), prompt_to_cls=list(prompt_to_cls),
                             time=time.strftime("%H:%M:%S")))
            self.prompt_list, self.prompt_to_cls = list(prompt_list), list(prompt_to_cls)
            self._write_meta_locked()

    def undo(self):
        with self.lock:
            if not self.saved:
                return None
            img_path, lbl_path, clss, empty = self.saved.pop()
            for p in (img_path, lbl_path):
                if os.path.isfile(p):
                    os.remove(p)
            self.count -= 1
            self.empty -= int(empty)
            for c in clss:
                self.instances[c] -= 1
            return os.path.basename(img_path)

    def write_meta(self, **extra):
        with self.lock:
            self.meta.update(extra)
            self._write_meta_locked()

    def _write_meta_locked(self):
        write_json(os.path.join(self.dir, "session.json"), dict(
            name=self.name, format=self.format, class_mode=self.class_mode, names=self.names,
            prompts=self.prompt_list, prompt_to_cls=self.prompt_to_cls,
            images=self.count, empty_images=self.empty,
            instances={n: c for n, c in zip(self.names, self.instances)},
            updated=time.strftime("%Y-%m-%d %H:%M:%S"), **self.meta))


def session_info(sdir):
    meta = read_json(os.path.join(sdir, "session.json")) or {}
    return dict(dir=sdir, name=os.path.basename(sdir), names=meta.get("names", []),
                format=meta.get("format", "segment"), class_mode=meta.get("class_mode", "fixed"),
                images=len(glob.glob(os.path.join(sdir, "images", "*.jpg"))),
                rejected=len(glob.glob(os.path.join(sdir, "rejected", "images", "*.jpg"))))


def list_sessions():
    """-> [dict(dir, name, names, format, class_mode, images, rejected)]  오래된 것부터"""
    out = []
    for d in sorted(glob.glob(os.path.join(SESSIONS_DIR, "*"))):
        if os.path.isfile(os.path.join(d, "session.json")) and os.path.isdir(os.path.join(d, "images")):
            out.append(session_info(d))
    return out


def session_images(sdir):
    return sorted(glob.glob(os.path.join(sdir, "images", "*.jpg")))


def label_path(img_path):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(os.path.dirname(os.path.dirname(img_path)), "labels", stem + ".txt")


def read_labels(path, w, h):
    """YOLO 라벨 (다각형 또는 박스) -> [(cls, poly(N,2 픽셀))]. 박스는 네 꼭짓점 다각형으로."""
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            v = line.split()
            if len(v) == 5:
                cx, cy, bw, bh = (float(x) for x in v[1:])
                x0, y0, x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
                out.append((int(v[0]), np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)))
            elif len(v) >= 7:
                xy = np.array(v[1:], np.float32).reshape(-1, 2) * np.array([w, h], np.float32)
                out.append((int(v[0]), xy))
    return out


def write_labels(path, objs, w, h, fmt="segment"):
    make = box_line if fmt == "box" else poly_line
    lines = [make(c, poly, w, h) for c, poly in objs]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
