"""카메라 기본값·열기 + 프레임 공급 스레드.

카메라 기본값은 chapter05\\01_camera_recognized 를 따른다 (그 앱의 camera_profiles.json 을 읽기만 함)
  - 카메라는 이름으로 고른다 (번호는 연결 순서에 따라 바뀜). 기본 = 그 앱에서 마지막으로 쓴 카메라
  - 해상도 기본 = 그 카메라의 지난번 해상도, 형식은 그 해상도에서 측정된 가장 빠른 형식
  - 여는 순서: 해상도 -> 형식. FPS 는 지정하지 않음 (지정하면 ELP 5MP 가 YUY2 로 바뀜), 버퍼 1
  - DSHOW 로 안 열리면 MSMF 로 한 번 더

FrameSource: 카메라 또는 영상 파일을 별도 스레드에서 읽어 '가장 최근 프레임'만 들고 있다.
             원본 녹화(raw.mp4)도 이 스레드에서 한다 (GUI 가 느려도 프레임이 빠지지 않게).
"""
import os
import sys
import threading
import time

import cv2

from core.paths import CAMERA_PROFILE_PATH
from core.util import read_json

REQUEST_SIZE = (1920, 1080)          # 01_camera_recognized 의 기본 요청 해상도
PRESETS = ("request", "max_fps", "max_quality")

try:  # 없는 카메라 번호를 열 때 나오는 OpenCV 경고 숨기기
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


def parse_size_key(key):
    try:
        w, h = (int(v) for v in str(key).lower().split("x"))
        return w, h
    except ValueError:
        return None


def load_profile_file():
    return read_json(CAMERA_PROFILE_PATH, {}) or {}


def get_camera_names():
    """DirectShow 장치 이름 (순서 = OpenCV CAP_DSHOW 번호). pygrabber 가 없으면 빈 목록"""
    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except Exception:
        return []


def camera_choices():
    """-> ([(번호, 이름)], 기본 번호). 기본 = 01_camera_recognized 에서 마지막으로 쓴 카메라"""
    names = get_camera_names()
    last = load_profile_file().get("last_camera")
    if names:
        choices = list(enumerate(names))
        default = names.index(last) if last in names else 0
    else:   # 이름을 모르면 0~3 번을 보여 주고, 0번을 마지막 카메라로 본다
        choices = [(i, (last if (i == 0 and last) else f"Camera {i}")) for i in range(4)]
        default = 0
    return choices, default


def resolution_choices(name):
    """-> ([(키, 표시 글자)], 기본 키). 키 = request / max_fps / max_quality / 'WxH'"""
    prof = load_profile_file().get("cameras", {}).get(name) or {}
    best = prof.get("best") or {}
    out = [("request", f"기본 {REQUEST_SIZE[0]}x{REQUEST_SIZE[1]}")]
    for key, label in (("max_fps", "최고 프레임"), ("max_quality", "최고 화질")):
        m = best.get(key)
        if m:
            out.append((key, f"{label} {m['width']}x{m['height']} {m['fourcc']} {m['fps']:.0f}fps"))
    sizes = {}
    for m in prof.get("modes", []):
        if m.get("ok") and m.get("fps", 0) > 0:
            k = f"{m['width']}x{m['height']}"
            if k not in sizes or m["fps"] > sizes[k]["fps"]:
                sizes[k] = m
    for k, m in sorted(sizes.items(), key=lambda kv: -kv[1]["width"] * kv[1]["height"]):
        out.append((k, f"{k} {m['request_fourcc']} {m['fps']:.0f}fps"))
    if not sizes:
        out += [("1280x720", "1280x720"), ("640x480", "640x480")]
    keys = [k for k, _ in out]
    last = prof.get("last_resolution")
    if last and last not in keys and (last in PRESETS or parse_size_key(last)):
        out.append((last, last))
        keys.append(last)
    return out, (last if last in keys else "request")


def choose_mode(profile, key):
    """해상도 키 -> {width, height, fourcc, fps(측정값 또는 None)} (01_camera_recognized 와 같은 규칙)"""
    best = (profile or {}).get("best") or {}
    if key in ("max_fps", "max_quality") and best.get(key):
        m = best[key]
        return dict(width=m["width"], height=m["height"], fourcc=m["request_fourcc"], fps=m["fps"])
    w, h = parse_size_key(key) or REQUEST_SIZE
    ok = [m for m in (profile or {}).get("modes", [])
          if m.get("ok") and m.get("fps", 0) > 0 and (m["width"], m["height"]) == (w, h)]
    if ok:
        m = max(ok, key=lambda m: m["fps"])
        return dict(width=w, height=h, fourcc=m["request_fourcc"], fps=m["fps"])
    adv = [x for x in (profile or {}).get("advertised", []) if (x["width"], x["height"]) == (w, h)]
    fourcc = "MJPG" if not adv or any(x["fourcc"] == "MJPG" for x in adv) else adv[0]["fourcc"]
    return dict(width=w, height=h, fourcc=fourcc, fps=None)


def open_capture(index, width, height, fourcc="MJPG", backend=None):
    if backend is None:
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def fourcc_text(value):
    v = int(value)
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)) if v > 0 else "?"


def open_camera(index, name, key):
    """카메라 열기 -> (cap, 설명 문자열). 실패하면 RuntimeError"""
    prof = load_profile_file().get("cameras", {}).get(name)
    spec = choose_mode(prof, key)
    w, h, fc = spec["width"], spec["height"], spec["fourcc"]
    cap = open_capture(index, w, h, fc)
    ok = False
    if cap is not None:
        for _ in range(5):        # 첫 몇 장은 버림 (노출 안정)
            ok, _f = cap.read()
    if (cap is None or not ok) and sys.platform == "win32":
        if cap is not None:
            cap.release()
        cap = open_capture(index, w, h, fc, cv2.CAP_MSMF)
        ok = cap is not None and cap.read()[0]
    if cap is None or not ok:
        if cap is not None:
            cap.release()
        raise RuntimeError(f"카메라 [{index}] {name} 를 열지 못했습니다 (다른 프로그램이 쓰고 있거나 연결이 빠졌는지 확인)")
    ok, frame = cap.read()
    fh, fw = frame.shape[:2]
    measured = f", 측정 {spec['fps']:.0f}fps" if spec.get("fps") else ""
    info = f"카메라 [{index}] {name} {fw}x{fh} {fourcc_text(cap.get(cv2.CAP_PROP_FOURCC))} (요청 {fc} {w}x{h}{measured})"
    return cap, info


class FrameSource:
    """카메라/영상 파일을 별도 스레드에서 읽고, 가장 최근 프레임만 보관한다.

    latest() -> (seq, frame)   seq 는 새 프레임마다 1 씩 늘어남. frame 은 공유되므로 고치지 말 것 (필요하면 copy)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self.seq = 0
        self.fps = 0.0
        self.kind = None          # "camera" / "video" / None
        self.info = ""
        self.path = None          # 영상 파일 경로
        self.ended = False        # 영상 끝 / 카메라 끊김
        self._thread = None
        self._stop = threading.Event()
        self._rec_lock = threading.Lock()
        self._rec = None

    # ---------- 열기 / 닫기 ----------
    def is_open(self):
        return self._thread is not None and self._thread.is_alive()

    def open_camera(self, index, name, key):
        self.close()
        cap, info = open_camera(index, name, key)
        self._start(cap, "camera", info, 0.0)

    def open_video(self, path):
        self.close()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열지 못했습니다: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.path = path
        self._start(cap, "video", f"영상 {os.path.basename(path)} {w}x{h} {fps:.1f}fps {n}프레임", 1.0 / fps)

    def _start(self, cap, kind, info, pace):
        self.kind, self.info, self.ended, self.fps = kind, info, False, 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(cap, pace), daemon=True)
        self._thread.start()

    def close(self):
        self.stop_recording()
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=3)
        self._thread = None
        with self._lock:
            self._frame = None
        self.kind, self.info, self.path, self.fps = None, "", None, 0.0

    def latest(self):
        with self._lock:
            return self.seq, self._frame

    # ---------- 읽기 스레드 ----------
    def _run(self, cap, pace):
        t_next = time.perf_counter()
        n, t0, fails = 0, time.time(), 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.kind == "video":
                        self.ended = True
                        break
                    fails += 1
                    if fails > 100:
                        self.ended = True
                        break
                    time.sleep(0.02)
                    continue
                fails = 0
                with self._lock:
                    self._frame = frame
                    self.seq += 1
                self._record(frame)
                n += 1
                now = time.time()
                if now - t0 >= 1.0:
                    self.fps, n, t0 = n / (now - t0), 0, now
                if pace:        # 영상 파일은 원래 속도로
                    t_next += pace
                    d = t_next - time.perf_counter()
                    if d > 0:
                        time.sleep(d)
                    elif d < -0.5:
                        t_next = time.perf_counter()
        finally:
            cap.release()

    # ---------- 원본 녹화 ----------
    # 처음 약 1초 동안 실제 프레임 속도를 재서 그 fps 로 저장 (재생 속도가 실제와 같게)
    def start_recording(self, path):
        with self._rec_lock:
            self._rec = dict(path=path, writer=None, buf=[], t0=None)

    def stop_recording(self):
        with self._rec_lock:
            r, self._rec = self._rec, None
        if not r:
            return None
        if r["writer"] is None and len(r["buf"]) >= 2:
            self._open_writer(r, max(1.0, (len(r["buf"]) - 1) / max(1e-3, time.time() - r["t0"])))
        if r["writer"] is not None:
            r["writer"].release()
            return r["path"]
        return None

    def recording(self):
        return self._rec is not None

    def _record(self, frame):
        with self._rec_lock:
            r = self._rec
            if r is None:
                return
            if r["writer"] is not None:
                r["writer"].write(frame)
                return
            now = time.time()
            r["t0"] = r["t0"] or now
            r["buf"].append(frame)
            if len(r["buf"]) >= 30 and now - r["t0"] >= 1.0:
                self._open_writer(r, min(120.0, max(1.0, (len(r["buf"]) - 1) / (now - r["t0"]))))

    @staticmethod
    def _open_writer(r, fps):
        h, w = r["buf"][0].shape[:2]
        os.makedirs(os.path.dirname(r["path"]), exist_ok=True)
        r["writer"] = cv2.VideoWriter(r["path"], cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in r["buf"]:
            r["writer"].write(f)
        r["buf"] = []
        r["fps"] = fps
