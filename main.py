"""SAM 3 자동 라벨 GUI — 실행 진입점.

    카메라 -> 1 라벨링 (SAM 3 텍스트 프롬프트) -> 2 검토 -> 3 학습 (YOLO-seg) -> 4 실시간

실행 (콘솔 창 없이):
    run.bat                                  # 01_camera_recognized 에서 마지막으로 쓴 카메라·해상도로 시작
    pythonw main.py --camera 1 --size 1280x720
    pythonw main.py --mode max_fps           # 최고 프레임 (max_quality = 최고 화질, request = 1920x1080)
    pythonw main.py --video sessions\\...\\raw.mp4   # 카메라 대신 영상
    pythonw main.py --no-camera --tab review # 카메라 없이 검토 탭으로 (SAM 3 자동 불러오기도 안 함)
    pythonw main.py --no-sam                 # SAM 3 자동 불러오기 끄기 (GPU 메모리 약 3.4 GB 아낌)
pythonw 로 실행하면 표준 출력은 logs\\gui_<날짜>.log 에 남는다.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

if sys.stdout is None or sys.stderr is None:      # pythonw: 출력이 없으면 tqdm 등이 실패하므로 파일로
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    _log = open(os.path.join(ROOT, "logs", time.strftime("gui_%Y%m%d.log")), "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log
else:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="SAM 3 자동 라벨 -> 검토 -> YOLO-seg 학습 -> 실시간 GUI")
    ap.add_argument("--camera", type=int, default=None, help="카메라 번호 (기본: 01_camera_recognized 에서 마지막으로 쓴 카메라)")
    ap.add_argument("--size", default=None, help="해상도 (예: 1280x720). 기본: 그 카메라의 지난번 해상도")
    ap.add_argument("--mode", default=None, choices=["request", "max_fps", "max_quality"])
    ap.add_argument("--video", default=None, help="카메라 대신 이 영상으로 시작")
    ap.add_argument("--no-camera", action="store_true", help="시작할 때 카메라를 열지 않음")
    ap.add_argument("--tab", default="label", choices=["label", "review", "train", "live"])
    ap.add_argument("--no-sam", action="store_true",
                    help="시작할 때 SAM 3 를 자동으로 불러오지 않음 (기본: 라벨링 탭으로 시작하면 백그라운드로 불러옴)")
    args = ap.parse_args()

    try:      # 고해상도 화면에서 글자가 흐려지지 않게
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    import tkinter as tk

    from ui.app import App

    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
