"""화면 그리기: 다각형(반투명 채우기 + 외곽선)과 글자(한글 가능, PIL + 맑은 고딕).

texts 항목: (x, y, 글자, BGR 색, 크기)
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
           (10, 249, 72), (23, 204, 146), (134, 219, 61), (211, 188, 0), (209, 99, 106),
           (203, 56, 255), (255, 149, 200)]
_FONT_FILES = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "malgun.ttf"),
               os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "malgunbd.ttf")]
_fonts = {}


def color(i):
    return PALETTE[i % len(PALETTE)]


def font(size):
    size = int(size)
    if size not in _fonts:
        f = None
        for p in _FONT_FILES:
            try:
                f = ImageFont.truetype(p, size)
                break
            except Exception:
                pass
        _fonts[size] = f or ImageFont.load_default()
    return _fonts[size]


def draw_items(img, items, names, scale, texts, offset=(0, 0), selected=None, alpha=0.4, text_size=15):
    """img(이미 scale 배로 줄인 그림) 위에 items 의 다각형을 그린다. 글자는 texts 에 추가만 (나중에 draw_texts)."""
    if not items:
        return img
    over = img.copy()
    pts_list = []
    for it in items:
        poly = it.get("poly")
        if poly is None or len(poly) < 3:
            pts_list.append(None)
            continue
        pts = np.round(np.asarray(poly, np.float32) * scale).astype(np.int32)
        pts_list.append(pts)
        cv2.fillPoly(over, [pts], color(it["cls"]))
    out = cv2.addWeighted(over, alpha, img, 1 - alpha, 0)
    for i, (it, pts) in enumerate(zip(items, pts_list)):
        if pts is None:
            continue
        c = color(it["cls"])
        if i == selected:
            cv2.polylines(out, [pts], True, (255, 255, 255), 5)
        cv2.polylines(out, [pts], True, c, 2)
        name = names[it["cls"]] if 0 <= it["cls"] < len(names) else str(it["cls"])
        txt = f"{name} {it['score']:.2f}" if it.get("score") is not None else name
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
        texts.append((x + offset[0], max(0, y - text_size - 6) + offset[1], txt, c, text_size))
    return out


def draw_texts(img, texts, panel=None):
    """panel = (x, y, 글자 list, 크기): 반투명 검은 배경 위에 여러 줄 (안내 글자가 마스크와 겹쳐도 읽히게)"""
    if not texts and not panel:
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if panel and panel[2]:
        px, py, rows, size = panel
        f = font(size)
        w = max(f.getlength(t) for t, _ in rows) + 12
        h = len(rows) * (size + 8) + 6
        over = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        ImageDraw.Draw(over).rectangle([px, py, px + w, py + h], fill=(0, 0, 0, 140))
        pil = Image.alpha_composite(pil.convert("RGBA"), over).convert("RGB")
        texts = list(texts) + [(px + 6, py + 4 + i * (size + 8), t, c, size) for i, (t, c) in enumerate(rows)]
    d = ImageDraw.Draw(pil)
    for x, y, t, c, size in texts:
        d.text((int(x), int(y)), t, font=font(size), fill=(c[2], c[1], c[0]),
               stroke_width=2, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def annotate_full(img, items, names, lines=()):
    """원본 해상도 그림에 그리기 (화면 저장용)"""
    k = max(1.0, img.shape[1] / 960)
    texts = []
    out = draw_items(img.copy(), items, names, 1.0, texts, text_size=int(15 * k))
    for i, line in enumerate(lines):
        texts.append((int(8 * k), int((6 + i * 24) * k), line, (255, 255, 255), int(17 * k)))
    return draw_texts(out, texts)
