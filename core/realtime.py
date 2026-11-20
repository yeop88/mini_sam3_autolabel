"""학습한 YOLO-seg 모델 실시간 추론 스레드.

FrameSource 의 가장 최근 프레임을 가져와 추론하고, 결과(dict(frame, items, ms))를 result / result_seq 로 내놓는다.
items: dict(cls, score, poly(N,2 픽셀))  — 마스크가 없는 모델이면 박스를 다각형으로.
"""
import collections
import glob
import os
import threading
import time

import numpy as np

from core.paths import RUNS_DIR


def list_trained_weights():
    """runs\\*\\weights\\best.pt, 최근 것부터"""
    return sorted(glob.glob(os.path.join(RUNS_DIR, "*", "weights", "best.pt")), key=os.path.getmtime, reverse=True)


def engine_path(weights, imgsz):
    """TensorRT 엔진 경로: best.pt -> best_640.engine (입력 크기별로 따로)"""
    return f"{os.path.splitext(weights)[0]}_{imgsz}.engine"


def export_engine(weights, imgsz, device=0):
    """YOLO .pt -> TensorRT FP16 엔진 (RTX 4080 SUPER 에서 yolo26s-seg 약 4분, 모델·입력 크기마다 한 번).
    측정 (1080p 프레임, 640): PyTorch FP16 8.7 ms -> TensorRT FP16 3.4 ms (추론만 6.2 -> 1.5 ms)"""
    from ultralytics import YOLO

    out = engine_path(weights, imgsz)
    made = YOLO(weights).export(format="engine", imgsz=imgsz, quantize=16, device=device, verbose=False)
    os.replace(made, out)
    onnx = os.path.splitext(weights)[0] + ".onnx"      # 중간 파일
    if os.path.isfile(onnx):
        os.remove(onnx)
    return out


class LiveDetector:
    def __init__(self, weights, source, conf=0.4, imgsz=640, half=True, device="0", trt=False):
        self.weights, self.source, self.trt = weights, source, trt
        self.imgsz = imgsz
        self.kw = dict(imgsz=imgsz, conf=conf, device=device, verbose=False)
        if not (trt or weights.endswith(".engine")):
            self.kw["quantize"] = 16 if half else 32
        self.backend = ""
        self.state = "loading"     # loading / exporting / running / error / stopped
        self.error = ""
        self.names = []
        self.result = None
        self.result_seq = 0
        self.ms = 0.0              # 추론 시간 (최근 30장 중앙값, ms)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def set_conf(self, conf):
        self.kw["conf"] = conf

    def _run(self):
        try:
            from ultralytics import YOLO

            path = self.weights
            task = None
            if not path.endswith(".engine"):
                task = YOLO(path).task           # segment / detect (박스 모델)
            if self.trt and not path.endswith(".engine"):
                eng = engine_path(path, self.imgsz)
                if not os.path.isfile(eng):
                    self.state = "exporting"
                    eng = export_engine(path, self.imgsz)
                    if self._stop.is_set():
                        self.state = "stopped"
                        return
                path = eng
            model = YOLO(path, task=task) if task else YOLO(path)
            self.backend = "TensorRT FP16" if path.endswith(".engine") else (
                "PyTorch FP16" if self.kw.get("quantize") == 16 else "PyTorch FP32")
            self.names = [model.names[i] for i in sorted(model.names)]
            model.predict(np.zeros((480, 640, 3), np.uint8), **self.kw)       # 예열
            self.state = "running"
            last, n_done, recent = -1, 0, collections.deque(maxlen=30)
            while not self._stop.is_set():
                seq, frame = self.source.latest()
                if frame is None or seq == last:
                    time.sleep(0.003)
                    continue
                last = seq
                t0 = time.perf_counter()
                r = model.predict(frame, **self.kw)[0]
                ms = (time.perf_counter() - t0) * 1000
                n_done += 1
                if n_done > 3:        # 처음 몇 장은 느림 (cuDNN 준비) -> 빼고 최근 30장 중앙값
                    recent.append(ms)
                    self.ms = float(np.median(recent))
                self.result = dict(frame=frame, items=self._items(r), ms=ms)
                self.result_seq += 1
            self.state = "stopped"
        except Exception as e:
            self.error = str(e)
            self.state = "error"

    @staticmethod
    def _items(r):
        if r.boxes is None or len(r.boxes) == 0:
            return []
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        if r.masks is not None:
            polys = r.masks.xy
        else:
            polys = [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
                     for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy()]
        return [dict(cls=int(c), score=float(s), poly=p) for c, s, p in zip(cls, conf, polys) if len(p) >= 3]
