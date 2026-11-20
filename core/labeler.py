"""라벨러.

- Sam3Labeler      : SAM 3 / 3.1 텍스트 프롬프트 -> 객체별 마스크 (지정 클래스)
- ObjectFinder     : YOLOE prompt-free — 프롬프트 없이 물체와 이름(4,585종 어휘)을 찾음
- OpenVocabLabeler : '모든 물체' — ObjectFinder 가 이름을 찾고, SAM 3 가 그 이름으로 다시 찾아 확인·마스크
                     (SAM 3 가 못 찾은 이름은 버려짐 -> 틀린 이름이 걸러짐). SAM 3 없이 YOLOE 결과만 쓸 수도 있음
- FakeLabeler      : 시험용

label_frame() -> (kept, names)
  kept 항목: dict(cls(names 의 번호), name, prompt, score, poly(N,2 픽셀))  — 화면 표시·저장 모두 이 다각형을 쓴다.
"""
import contextlib
import glob
import os
import threading

import cv2
import numpy as np
import torch

from core.paths import WEIGHTS_DIR

SAM3_ORDER = ["sam3.1_multiplex.pt", "sam3.pt"]     # 앞에 있는 것이 기본
FAKE = "fake (색 덩어리, 시험용)"
# 모든 물체 모드의 이름 찾기 모델 (weights\ 에 없으면 Ultralytics 가 내려받음). 앞에 있는 것이 기본
FINDER_CHOICES = ["yoloe-26l-seg-pf.pt", "yoloe-26s-seg-pf.pt", "yoloe-11l-seg-pf.pt", "yoloe-11s-seg-pf.pt"]


def list_sam3_weights():
    found = [os.path.basename(p) for p in glob.glob(os.path.join(WEIGHTS_DIR, "sam3*.pt"))]
    found.sort(key=lambda n: (SAM3_ORDER.index(n) if n in SAM3_ORDER else len(SAM3_ORDER), n))
    return found


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())


def merge_same_class(items, iou_thr=0.6):
    """같은 클래스로 묶인 프롬프트(예: cup, mug)가 같은 물체를 두 번 잡으면 점수 높은 것만 남긴다."""
    keep = []
    for it in sorted(items, key=lambda d: -d["score"]):
        if not any(k["cls"] == it["cls"] and mask_iou(k["mask"], it["mask"]) >= iou_thr for k in keep):
            keep.append(it)
    return keep


def mask_to_polygon(mask, min_area=300, eps_ratio=0.002):
    """가장 큰 외곽선을 다각형(N,2 픽셀)으로. 너무 작으면 None."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    eps = max(0.5, eps_ratio * cv2.arcLength(c, True))
    c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(c) < 3:
        return None
    return c.astype(np.float32)


def box_poly(poly):
    """다각형 -> 그것을 감싸는 박스의 네 꼭짓점 (박스 라벨 형식일 때 화면·저장용)"""
    p = np.asarray(poly, np.float32)
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


# ---------------------------------------------------------------- SAM 3
_INIT_MODULES = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Embedding,
                 torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.MultiheadAttention)


@contextlib.contextmanager
def skip_random_init():
    """모델을 만드는 동안 무작위 초기화를 건너뛴다 (곧 체크포인트 가중치로 덮어쓰므로 필요 없음).

    SAM 3.1(8.7억 파라미터) 불러오기: 7.9 s -> 2.5 s. 결과(객체 수·점수·마스크 넓이)는 같음 (2026-11-20 확인).
    torch.nn.init.*_ 와 흔한 층의 reset_parameters 를 잠시 아무것도 안 하게 바꿨다가 되돌린다.
    """
    inits = {n: getattr(torch.nn.init, n) for n in dir(torch.nn.init) if n.endswith("_") and not n.startswith("_")}
    resets = {c: c.reset_parameters for c in _INIT_MODULES if hasattr(c, "reset_parameters")}
    try:
        for n in inits:
            setattr(torch.nn.init, n, lambda t, *a, **k: t)
        for c in resets:
            c.reset_parameters = lambda self, *a, **k: None
        yield
    finally:
        for n, f in inits.items():
            setattr(torch.nn.init, n, f)
        for c, f in resets.items():
            c.reset_parameters = f


class Sam3Labeler:
    """텍스트 프롬프트 -> 객체별 마스크. 여러 스레드에서 불러도 한 번에 하나씩 돈다.
    주의: 프롬프트 수가 늘면 급격히 느려짐 (한 장: 1개 149 ms, 3개 191, 10개 461, 30개 16 s, 80개 65 s)"""

    def __init__(self, weights, conf=0.4, imgsz=1008, half=True, device="0", fast_load=True):
        from ultralytics.models.sam import SAM3SemanticPredictor

        path = weights if os.path.isfile(weights) else os.path.join(WEIGHTS_DIR, weights)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SAM 3 가중치가 없습니다: {path}")
        overrides = dict(conf=conf, task="segment", mode="predict", model=path, imgsz=imgsz,
                         quantize=16 if half else 32, save=False, verbose=False, device=device)
        self.pred = SAM3SemanticPredictor(overrides=overrides)
        self.load_mode = "보통"
        if fast_load:         # 모델을 지금 만든다 (안 하면 첫 호출 때 만듦)
            try:
                with skip_random_init():
                    self.pred.setup_model(verbose=False)
                self.load_mode = "빠른 불러오기"
            except Exception:  # 안 되면 보통 방법으로 (첫 호출 때 다시 만듦)
                self.pred = SAM3SemanticPredictor(overrides=overrides)
        self.lock = threading.Lock()
        self.name = f"SAM3 {os.path.basename(path)} (conf {conf}, {imgsz}px)"

    def detect(self, frame_bgr, prompt_list, prompt_to_cls):
        """-> [dict(cls, prompt, score, mask)]  cls = prompt_to_cls[프롬프트 번호]"""
        if not prompt_list:
            return []
        with self.lock:
            r = self.pred(source=frame_bgr, text=list(prompt_list))[0]
        if r.masks is None or r.boxes is None or len(r.boxes) == 0:
            return []
        masks = r.masks.data.cpu().numpy().astype(bool)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        h, w = frame_bgr.shape[:2]
        items = []
        for m, c, s in zip(masks, cls, conf):
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            if 0 <= c < len(prompt_to_cls):
                items.append(dict(cls=int(prompt_to_cls[c]), prompt=prompt_list[c], score=float(s), mask=m))
        return merge_same_class(items)

    def detect_named(self, frame_bgr, flat):
        names, prompt_list, prompt_to_cls = flat
        items = self.detect(frame_bgr, prompt_list, prompt_to_cls)
        for it in items:
            it["name"] = names[it["cls"]]
        return items, list(names)


class FakeLabeler:
    """SAM 3 없이 흐름만 시험: 채도 높은(색이 진한) 덩어리를 첫 번째 클래스로 잡는다."""

    name = FAKE

    def __init__(self):
        self.lock = threading.Lock()

    def detect(self, frame_bgr, prompt_list, prompt_to_cls):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        m = cv2.morphologyEx(cv2.inRange(hsv, (0, 90, 60), (180, 255, 255)), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        area_min = frame_bgr.shape[0] * frame_bgr.shape[1] * 0.002
        return [dict(cls=0, prompt=prompt_list[0] if prompt_list else "fake", score=0.99, mask=lab == i)
                for i in range(1, n) if stats[i][4] >= area_min]

    def detect_named(self, frame_bgr, flat):
        names = list(flat[0]) or ["object"]
        items = self.detect(frame_bgr, flat[1], flat[2])
        for it in items:
            it["name"] = names[0]
        return items, names


def make_labeler(choice, conf=0.4, imgsz=1008, half=True):
    """choice = 가중치 파일 이름 또는 FAKE"""
    if choice == FAKE:
        return FakeLabeler()
    return Sam3Labeler(choice, conf=conf, imgsz=imgsz, half=half)


# ---------------------------------------------------------------- 모든 물체 (클래스 자동 생성)
class ObjectFinder:
    """YOLOE prompt-free: 프롬프트 없이 물체·이름·마스크. 빠름(1080p 한 장 15~55 ms) 대신 이름이 자주 틀림
    (예: 체커보드 -> quilting / sheet music, 'desktop' 같은 장면 이름). 그래서 기본은 SAM 3 로 확인한다."""

    def __init__(self, weights, device="0"):
        from ultralytics import YOLOE

        path = weights if os.path.isfile(weights) else os.path.join(WEIGHTS_DIR, weights)
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        self.model = YOLOE(path)            # 없으면 Ultralytics 가 이 경로로 내려받음
        self.device = device
        self.lock = threading.Lock()
        self.name = os.path.basename(path)
        self.model.predict(np.zeros((480, 640, 3), np.uint8), verbose=False, device=device)     # 예열

    def find(self, frame_bgr, conf=0.25):
        """-> [dict(name, score, poly(원본 좌표))]"""
        with self.lock:
            r = self.model.predict(frame_bgr, conf=conf, device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        polys = r.masks.xy if r.masks is not None else [box_poly(np.array([[b[0], b[1]], [b[2], b[3]]]))
                                                        for b in r.boxes.xyxy.cpu().numpy()]
        out = []
        for c, s, p in zip(r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), polys):
            if len(p) >= 3:
                out.append(dict(name=self.model.names[int(c)], score=float(s), poly=np.asarray(p, np.float32)))
        return out


class OpenVocabLabeler:
    """모든 물체: ObjectFinder 로 이름 찾기 -> (SAM 3 로 확인) -> 객체. 클래스 이름은 찾은 이름 그대로.

    max_names : 한 장에서 SAM 3 에 넘길 이름 수 상한 (SAM 3 는 프롬프트가 많으면 매우 느림, 지정 클래스 포함 최대 10)
    use_fixed : 지정 클래스(classes.json) 프롬프트도 함께 찾기 (YOLOE 가 놓치는 물체 보완, 예: 안경)
    """

    MAX_PROMPTS = 10

    def __init__(self, finder, sam=None, find_conf=0.25, max_names=6, use_fixed=True):
        self.finder, self.sam = finder, sam
        self.find_conf, self.max_names, self.use_fixed = find_conf, max_names, use_fixed
        self.lock = threading.Lock()
        self.name = f"모든 물체: {finder.name}" + (" + SAM 3 확인" if sam is not None else " (SAM 3 확인 없음)")

    def detect_named(self, frame_bgr, flat):
        found = sorted(self.finder.find(frame_bgr, self.find_conf), key=lambda d: -d["score"])
        names = []
        for f in found:
            if f["name"] not in names:
                names.append(f["name"])
        names = names[:self.max_names]
        if self.sam is None:          # YOLOE 결과 그대로
            items = [dict(cls=names.index(f["name"]), name=f["name"], prompt="YOLOE", score=f["score"], poly=f["poly"])
                     for f in found if f["name"] in names]
            return items, names
        all_names, prompts, p2c = list(names), list(names), list(range(len(names)))
        if self.use_fixed:
            fixed_names, fixed_prompts, fixed_p2c = flat
            for p, c in zip(fixed_prompts, fixed_p2c):
                n = fixed_names[c]
                if n not in all_names:
                    all_names.append(n)
                if p not in prompts:
                    prompts.append(p)
                    p2c.append(all_names.index(n))
        prompts, p2c = prompts[:self.MAX_PROMPTS], p2c[:self.MAX_PROMPTS]
        items = self.sam.detect(frame_bgr, prompts, p2c)
        # 다른 이름이 같은 물체를 잡으면 (예: smartphone / phone) 점수 높은 것만
        keep = []
        for it in sorted(items, key=lambda d: -d["score"]):
            if not any(mask_iou(k["mask"], it["mask"]) >= 0.8 for k in keep):
                keep.append(it)
        for it in keep:
            it["name"] = all_names[it["cls"]]
        return keep, all_names


def label_frame(labeler, frame, flat, min_area=300, fmt="segment"):
    """-> (kept, names). fmt='box' 면 다각형을 감싸는 박스(네 꼭짓점)로 바꿔 둔다 (화면에 보이는 그대로 저장)."""
    items, names = labeler.detect_named(frame, flat)
    kept = []
    for it in items:
        poly = it.get("poly")
        if poly is None:
            poly = mask_to_polygon(it["mask"], min_area=min_area)
        elif len(poly) < 3 or abs(cv2.contourArea(np.asarray(poly, np.float32))) < min_area:
            poly = None
        if poly is None:
            continue
        if fmt == "box":
            poly = box_poly(poly)
        kept.append(dict(cls=it["cls"], name=it["name"], prompt=it.get("prompt", ""), score=it["score"],
                         poly=np.asarray(poly, np.float32)))
    return kept, names
