"""폴더 경로 (모두 이 프로젝트 폴더 기준)."""
import os

# Ultralytics 가 필요한 패키지를 몰래 설치하지 않게 (TensorRT 변환 때 onnxruntime-gpu 등).
# 이 모듈은 ultralytics 보다 먼저 불러온다 (core 의 모든 모듈이 paths 를 먼저 import).
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES_JSON = os.path.join(ROOT, "classes.json")      # 클래스·프롬프트
WEIGHTS_DIR = os.path.join(ROOT, "weights")             # SAM 3 가중치, (선택) YOLO 사전학습 가중치
SESSIONS_DIR = os.path.join(ROOT, "sessions")           # 촬영·자동 라벨 결과 (세션 폴더별)
DATASET_DIR = os.path.join(ROOT, "dataset")             # 학습용 묶음
RUNS_DIR = os.path.join(ROOT, "runs")                   # YOLO 학습 결과
CAPTURE_DIR = os.path.join(ROOT, "captures")            # 화면 저장
LOG_DIR = os.path.join(ROOT, "logs")                    # pythonw 실행 시 표준 출력

# 사전학습 YOLO 가중치를 더 찾아보는 곳 (읽기만)
END2_WEIGHTS = os.path.normpath(os.path.join(ROOT, "..", "chapter04", "lecture_v8s_convnext03_end_2", "weights"))
# 카메라 기본값: chapter05\01_camera_recognized 의 카메라 스펙 파일 (읽기만)
CAMERA_PROFILE_PATH = os.path.normpath(os.path.join(ROOT, "..", "chapter05", "01_camera_recognized",
                                                    "camera_profiles.json"))
