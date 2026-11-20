"""촬영 중 자동 라벨 스레드 + 영상 파일 일괄 라벨.

비슷한 장면 건너뛰기: 마지막 저장 장면과 64x36 흑백 평균 차가 min_diff 보다 작으면 저장하지 않는다.
물체 없는 프레임: empty_every 장마다 1장만 저장 (배경 학습용, 0 = 저장 안 함). 수동 저장은 항상 저장.
"""
import os
import queue
import threading
import time

import cv2
import numpy as np

from core.labeler import label_frame


def frame_signature(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def signature_diff(a, b):
    if a is None or b is None:
        return 1e9
    return float(np.abs(a - b).mean())


class LabelWorker:
    """요청을 하나씩 받아 SAM 3 로 라벨 -> 저장 여부 판단 -> 저장. 결과는 result / result_seq 로 읽는다."""

    def __init__(self):
        self.q = queue.Queue(maxsize=1)
        self.busy = False
        self.last_sig = None
        self.empty_seen = 0
        self.result = None       # dict(frame, kept, names, ms, saved)
        self.result_seq = 0
        self.message = ""
        self.errors = queue.Queue()
        self.stats = dict(labeled=0, saved=0, skip_similar=0, skip_empty=0, last_ms=0.0, last_n=0)
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, frame, manual, labeler, session, flat, params, preview=False):
        """params: dict(min_diff, min_area, empty_every). preview=True 면 라벨만 하고 저장 안 함. 바쁘면 False"""
        if self.busy:
            return False
        self.busy = True
        self.q.put(dict(frame=frame.copy(), manual=manual, labeler=labeler, session=session, flat=flat,
                        params=dict(params), preview=preview))
        return True

    def reset_similar(self):
        self.last_sig = None

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                job = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._process(**job)
            except Exception as e:      # 한 장 실패로 멈추지 않게
                self.message = f"라벨 오류: {e}"
                self.errors.put(str(e))
            finally:
                self.busy = False

    def _process(self, frame, manual, labeler, session, flat, params, preview=False):
        fmt = params.get("format", "segment")
        if preview:      # 미리보기: 라벨만 (프롬프트 맞춰 보기용)
            t0 = time.perf_counter()
            kept, names = label_frame(labeler, frame, flat, params["min_area"], fmt)
            ms = (time.perf_counter() - t0) * 1000
            self.result = dict(frame=frame, kept=kept, names=names, ms=ms, saved=None, preview=True,
                               prompts=list(flat[1]))
            self.result_seq += 1
            return
        sig = frame_signature(frame)
        if not manual and signature_diff(sig, self.last_sig) < params["min_diff"]:
            self.stats["skip_similar"] += 1
            self.message = "비슷한 장면 - 건너뜀"
            return
        t0 = time.perf_counter()
        kept, names = label_frame(labeler, frame, flat, params["min_area"], fmt)
        ms = (time.perf_counter() - t0) * 1000
        st = self.stats
        st["labeled"] += 1
        st["last_ms"], st["last_n"] = ms, len(kept)
        saved = None
        keep = bool(kept)
        if not kept:
            self.empty_seen += 1
            ee = params["empty_every"]
            keep = manual or (ee > 0 and (self.empty_seen - 1) % ee == 0)
        if keep:
            saved = session.save(frame, kept)
            self.last_sig = sig
            st["saved"] += 1
            self.message = f"저장 {saved} ({len(kept)}개)"
        else:
            st["skip_empty"] += 1
            self.message = "물체 없음 - 건너뜀"
        self.result = dict(frame=frame, kept=kept, names=names, ms=ms, saved=saved, preview=False,
                           prompts=list(flat[1]))
        self.result_seq += 1


def label_video(path, labeler, session, flat, params, every_sec, progress=None, cancel=None):
    """영상 파일을 every_sec 초마다 한 장씩 라벨해 session 에 저장. progress(i, total, saved, ms_avg). -> 저장 장수"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열지 못했습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(every_sec * fps)))
    last_sig, saved, empty_seen, idx, ms_sum, n_lab = None, 0, 0, -1, 0.0, 0
    ee = params["empty_every"]
    try:
        while cap.grab():
            if cancel is not None and cancel.is_set():
                break
            idx += 1
            if idx % step:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            sig = frame_signature(frame)
            if signature_diff(sig, last_sig) < params["min_diff"]:
                continue
            t0 = time.perf_counter()
            kept, _names = label_frame(labeler, frame, flat, params["min_area"], params.get("format", "segment"))
            ms_sum += (time.perf_counter() - t0) * 1000
            n_lab += 1
            if not kept:
                empty_seen += 1
                if ee <= 0 or (empty_seen - 1) % ee:
                    continue
            session.save(frame, kept)
            last_sig = sig
            saved += 1
            if progress:
                progress(idx, total, saved, ms_sum / max(1, n_lab))
    finally:
        cap.release()
    session.write_meta(source=os.path.abspath(path), every_sec=every_sec)
    return saved
