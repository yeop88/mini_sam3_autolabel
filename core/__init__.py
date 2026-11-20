"""SAM 3 자동 라벨 GUI 의 핵심 로직 (tkinter 를 모름).

paths       폴더 경로
util        json 읽기·쓰기
classes_cfg classes.json (클래스·프롬프트)
camera      카메라 기본값(01_camera_recognized 프로필), 프레임 공급 스레드(카메라/영상), 원본 녹화
labeler     SAM 3 / 3.1 라벨러, 마스크 -> YOLO-seg 다각형
autolabel   촬영 중 라벨 스레드, 영상 파일 일괄 라벨
session     세션 폴더(images, labels, raw.mp4) 저장·읽기
dataset     세션 -> 학습 데이터셋 (연속 묶음 단위 검증 분할)
train_job   학습 프로세스 진입점 (python -m core.train_job 설정.json)
realtime    학습한 YOLO-seg 실시간 추론 스레드
render      화면 그리기 (다각형, 한글 글자)
"""
