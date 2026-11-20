# mini_sam3_autolabel — SAM 3 자동 라벨 → 검토 → YOLO-seg 학습 → 실시간 (GUI)

카메라로 물체를 찍는 동안 **SAM 3.1 이 텍스트 프롬프트("cup", "brown object")로 마스크를 자동으로 만들어** YOLO 분할 라벨로 저장합니다.
그 라벨을 검토·수정하고, 작은 YOLO-seg 모델을 학습해 카메라에서 실시간으로 돌립니다. 모든 단계를 창 하나에서 합니다.

```
run.bat
 ┌ 소스: [카메라 ▼] [해상도 ▼] [카메라 열기] [영상 파일...] [닫기]               소스 정보 · fps ┐
 │                                                  │ 1 라벨링  2 검토  3 학습  4 실시간       │
 │   영상 화면 (탭에 따라 카메라 / 라벨 / 추론 결과)    │  (단계별 설정·버튼)                       │
 └ 메시지 ──────────────────────────────────────────────────────────────────────────────────┘
```

예전 명령줄 스크립트(00~05)는 `backup\` 에 있습니다 (아래 "예전 스크립트").

## 1. 실행

```
run.bat                                   # 01_camera_recognized 에서 마지막으로 쓴 카메라·해상도로 시작
run.bat --camera 1 --size 1280x720
run.bat --mode max_fps                    # 최고 프레임 (max_quality = 최고 화질, request = 1920x1080)
run.bat --video D:\videos\desk.mp4        # 카메라 대신 영상
run.bat --no-camera --tab review          # 카메라 없이 검토 탭으로 (SAM 3 자동 불러오기도 안 함)
run.bat --no-sam                          # SAM 3 자동 불러오기 끄기 (GPU 메모리 약 3.4 GB 아낌)
```
- **SAM 3 자동 불러오기**: 라벨링 탭으로 시작하면 카메라를 여는 동안 백그라운드로 SAM 3.1 을 불러옵니다
  (가중치 콤보 첫 항목, 점수 기준 0.4, 입력 1008, `classes.json` 프롬프트로 예열).
  실측: SAM 준비 4.8 s, 카메라(1080p) 열림 6.4 s → 영상이 뜰 때 이미 준비됨. 다른 탭으로 시작하거나 `--no-sam` 이면 안 함.
- `run.bat` = `pythonw main.py` (콘솔 창 없음). 표준 출력은 `logs\gui_<날짜>.log`.
- 준비물(이미 됨, 2026-11-20): `weights\sam3.1_multiplex.pt`, CLIP(`clip` 1.0). 다시 설치할 때는 `backup\README.md` 의 "1. 준비".

## 2. 탭별 사용법

### 위 줄 — 소스
- 카메라 목록·해상도는 `chapter05\01_camera_recognized\camera_profiles.json` 을 **읽기만** 해서 채웁니다 (마지막 카메라, 그 카메라의 지난번 해상도, 측정된 가장 빠른 형식). 여는 방법도 그 앱과 같습니다 (DSHOW, 해상도 → 형식, FPS 지정 안 함, 안 되면 MSMF).
- [영상 파일...] 로 mp4 를 열면 원래 속도로 재생되고, 라벨·실시간 탭이 카메라처럼 씁니다.

### 1 라벨링
| 칸 | 내용 |
| --- | --- |
| 라벨 설정 · 형식 | **분할 (다각형)** = YOLO-seg `cls x1 y1 ...` / **박스** = YOLO `cls cx cy w h`. 화면에도 저장될 모양 그대로 보임. 세션마다 `session.json` 의 `format` |
| 라벨 설정 · 대상 | **지정 클래스** = classes.json 의 클래스 / **모든 물체 (클래스 자동 생성)** = 아래 참고. 형식·대상을 바꾸면 다음 저장부터 새 세션 |
| 클래스 · 프롬프트 | `classes.json` 편집 (추가·수정·삭제하면 바로 저장). 프롬프트는 **영어 명사구**, 쉼표로 여러 개 |
| 프롬프트 바로 입력 | 표에서 클래스를 고르고 아래 칸에 입력 → **Enter** 면 바로 적용 (모델 다시 불러오기 없음, 0.15 s). 클래스가 없으면 새로 만듦. 클래스 이름이 같으면 세션을 이어 쓰고 `session.json` 의 `prompt_history` 에 기록 |
| 미리보기 (**P**) | 저장 없이 SAM 3 를 계속 돌려 지금 화면 위에 마스크 표시 (장당 약 0.2 s). 프롬프트를 바꾸며 맞춰 볼 때. 프롬프트를 적용하면 자동으로 켜짐 |
| SAM 3 모델 | 가중치(`weights\sam3*.pt`, 시험용 fake), 점수 기준(0.4), 입력 크기(1008 / 644 빠름) → [불러오기] (약 10초). [해제] 로 GPU 메모리 비우기 |
| 촬영하면서 자동 라벨 | **A** 자동 켜기/끄기, **Space** 지금 프레임 저장, **U** 마지막 저장 취소. 라벨 간격, 장면 변화 기준(직전 저장과 비슷하면 건너뜀), 빈 장면 N장마다 1장, 최소 마스크 크기 |
| 세션 | 첫 저장 때 `sessions\<시각>_<태그>\` 생성. 원본 `raw.mp4` 녹화는 **기본 꺼짐** (체크하면 카메라일 때 저장). 클래스 이름을 바꾸면 다음 저장부터 새 세션 |
| 영상 파일 일괄 라벨 | 영상(예: 세션의 raw.mp4)을 N초마다 라벨해 **새 세션**으로 저장. 프롬프트를 바꿔 다시 라벨할 때 |

영상 화면: 큰 화면 = 실시간 소스, 오른쪽 위 작은 화면 = 마지막으로 라벨한 프레임과 마스크.
(A "자동 라벨" = 간격마다 찍어 저장하는 촬영 방식, "모든 물체" = 무엇을 라벨할지 정하는 대상 — 서로 다른 설정입니다.)

#### 모든 물체 (클래스 자동 생성)
SAM 3 는 텍스트 프롬프트가 있어야 찾으므로, 이름 후보를 먼저 찾습니다.
1. **이름 찾기**: YOLOE prompt-free (`weights\yoloe-26l-seg-pf.pt` 기본, 4,585종 어휘, 장당 15~55 ms) 가 물체와 이름을 찾음.
2. **SAM 3 로 확인** (기본 켜짐): 찾은 이름(장당 최대 6개) + 지정 클래스 프롬프트로 SAM 3 가 다시 찾음. SAM 3 가 못 찾은 이름은 버려지고 마스크가 정확해짐. 장당 약 0.5 s.
3. 처음 보는 이름은 세션 클래스 끝에 자동 추가 (`session.json` 의 `class_mode: open`, `names`).

실측 (2026-11-20, 실제 카메라):
| 방식 | 결과 |
| --- | --- |
| 책상 프레임: YOLOE 만 | desktop, smartphone, keyboard, **sheet music**, bag, mouse, **chemical compound**, pen (틀린 이름 섞임, 안경 놓침) |
| 같은 프레임: YOLOE + SAM 3 확인 | smartphone 0.89, keyboard 0.75, pen 0.68 (틀린 이름 걸러짐), 0.5 s |
| 자동 라벨 14장: + SAM 3 확인 | 클래스 6개 (phone, person, monitor, calendar, sheet music, movie poster) |
| 자동 라벨 12장: SAM 3 확인 없음 | 클래스 18개 (hacker, server room, collaboration, cold, bank vault ... 엉뚱한 이름 많음) |

→ **SAM 3 확인을 켜 두세요.** YOLOE 가 놓치는 물체(예: 안경)는 "지정 클래스도 함께 찾기"로 보완합니다.
SAM 3 에 단어 목록을 통째로 주는 방법은 프롬프트 수에 따라 급격히 느려져 쓰지 않습니다 (장당 1개 149 ms, 10개 461 ms, 30개 16 s, 80개 65 s).
이름이 틀린 객체는 2 검토 탭에서 지우거나 클래스를 바꾸세요.

### 2 검토
- 세션을 고르면 **학습에 실제로 쓰일 다각형**을 그려 보여 줍니다.
- **← / →** (A / D, Space) 이전·다음, 슬라이더로 이동.
- **X** 이 장 빼기 → 세션의 `rejected\` 로 옮김 (학습에서 빠짐).
- 영상 화면에서 객체를 **클릭**하면 선택 → **Delete** 로 지우기, 또는 [클래스 바꾸기].
- **Z** 되돌리기 (빼기·지우기·바꾸기를 마지막 것부터).

### 3 학습
- 세션을 여러 개 고름 → [학습 시작] 또는 [2 epoch 시험]. 고르면 합친 클래스와 작업이 표시됨.
- **클래스는 이름으로 합침** (세션마다 클래스가 달라도 됨, 모든 물체 세션 포함). 라벨 번호를 합친 목록 번호로 바꿔 데이터셋을 만듦.
- **작업**: 박스 세션이 하나라도 있으면 박스(detect, `yolo26s.pt`), 모두 분할이면 분할(segment, `yolo26s-seg.pt`). 시작 가중치 목록도 작업에 맞게 바뀜.
- 시작 가중치: `weights\`, `chapter04\lecture_v8s_convnext03_end_2\weights\` 에 있는 `*-seg.pt` (yolo26s-seg, yolo11s-seg) 또는 이름(Ultralytics 가 내려받음).
- 연속 프레임은 서로 비슷하므로 검증은 **연속 10장 묶음 단위**로 떼어 냅니다 (장 단위로 섞으면 검증 점수가 부풀려짐).
- 학습은 **별도 프로세스**(`python -m core.train_job`)로 돌아서 GUI 가 멈추지 않고, [중지] 하면 프로세스째 끝냅니다. 진행은 메시지 창 `[학습]` 줄.
- 결과: `dataset\<이름>\`, `runs\<이름>\weights\best.pt`, `runs\<이름>\train_summary.json` (mAP).

- [끝나면 TensorRT 엔진도 만들기] (기본 켜짐, 2 epoch 시험은 제외): 학습 프로세스 안에서 `best_<입력크기>.engine` 을 만들어 둠 (약 4분).

### 4 실시간
- `runs\*\weights\best.pt` (최근 것부터) 또는 [다른 파일...] → **R** 실행/멈춤, **S** 화면 저장(`captures\`).
- **TensorRT FP16 기본**: `best_<입력크기>.engine` 이 있으면 쓰고, 없으면 처음 한 번 변환(약 4분, 모델·입력 크기별). 끄면 PyTorch FP16.
- 점수 기준은 실행 중에도 바뀝니다. 표시 속도는 최근 30장 중앙값.

### TensorRT 측정 (2026-11-20, yolo26s-seg, 1080p 프레임)
| 조건 | PyTorch FP16 | TensorRT FP16 |
| --- | --- | --- |
| 연속 실행 (입력 640, predict 전체 / 추론만) | 8.7 / 6.2 ms | **3.4 / 1.5 ms** |
| 앱 안 실측 (영상 30 fps, 입력 480, 1초마다 10번) | 8~13 ms | 2~11 ms |

앱 안에서는 프레임 사이에 GPU 가 쉬어 클럭이 내려가 차이가 줄어듭니다. 둘 다 60 fps(16.7 ms) 안.
**SAM 3 는 TensorRT 불가**: Ultralytics 내보내기가 SAM 을 지원하지 않습니다.
Ultralytics 의 패키지 자동 설치는 꺼 둡니다 (`YOLO_AUTOINSTALL=False`, `core\paths.py`) — 변환 때 onnxruntime-gpu 를 설치하려 하지만 없어도 변환됨.

단축키는 입력 칸에 커서가 있으면 동작하지 않습니다 → 영상 화면을 한 번 클릭하세요.

## 3. 잘 되게 하는 요령
- **프롬프트 표현에 따라 결과가 크게 다릅니다.** 합성 컨베이어 영상의 수육 한 프레임(점수 기준 0.4):
  `meat` 0개, `sliced meat` 0개, `piece of meat` 7개(점수 ~0.45), `food` 7개(~0.8), `brown object` 7개(~0.96).
  안 잡히면 프롬프트를 바꿔 보고(색·모양 묘사가 잘 먹힘), 점수 기준을 0.2~0.3 으로 낮춰 보세요.
- 클래스당 100~300장, 각도·거리·조명·배경을 바꿔 가며. 컨베이어처럼 장면 변화가 작으면 "장면 변화 기준"을 2 정도로.
- 학습한 모델이 SAM 3 보다 나을 수는 없습니다(라벨 품질이 상한). 대신 수십 배 빠릅니다 (SAM 3.1 약 170~240 ms/장 vs YOLO-seg 수 ms).

## 4. 구조

GUI 패턴: **핵심 로직(core) + 단계별 탭(ui) + 작업 스레드와 큐**.
작업 스레드(카메라 읽기, SAM 3, 실시간 추론, 학습 출력 읽기)는 tkinter 를 직접 건드리지 않고 `App.post()` 큐로 넘기며,
GUI 스레드의 tick(15 ms)이 큐를 비우고 보이는 탭의 화면만 다시 그립니다. 학습만 별도 프로세스.

| 경로 | 내용 |
| --- | --- |
| `main.py`, `run.bat` | 진입점 |
| `core\paths.py` | 폴더 경로 |
| `core\classes_cfg.py` | classes.json 읽기·저장·검사 |
| `core\camera.py` | 카메라 기본값(01_camera_recognized 프로필), 카메라 열기, `FrameSource`(읽기 스레드 + raw.mp4 녹화) |
| `core\labeler.py` | SAM 3/3.1 라벨러(+fake), 같은 클래스 중복 합치기, 마스크 → 다각형 |
| `core\autolabel.py` | `LabelWorker`(촬영 중 라벨 스레드), `label_video`(영상 일괄) |
| `core\session.py` | 세션 저장·되돌리기, 라벨 읽기·쓰기 |
| `core\dataset.py`, `core\train_job.py` | 데이터셋 만들기, 학습 프로세스 |
| `core\realtime.py` | `LiveDetector`(실시간 추론 스레드) |
| `core\render.py` | 다각형·한글 글자 그리기 |
| `ui\app.py` | 메인 창, 소스 줄, 큐, tick |
| `ui\video_view.py` | 영상 화면 (창 크기 맞춤, 클릭 → 원본 좌표) |
| `ui\label_tab.py`, `review_tab.py`, `train_tab.py`, `live_tab.py` | 탭 1~4 |
| `classes.json` | 클래스·프롬프트 |
| `constraints.txt` | `physical_ai` 패키지 버전 고정 (`pip install ... -c constraints.txt`) |
| `weights\` | `sam3.1_multiplex.pt`, `yoloe-26l/26s/11l/11s-seg-pf.pt`(모든 물체 이름 찾기), `yolo26n.pt`(Ultralytics AMP 점검용) |
| `sessions\`, `dataset\`, `runs\`, `captures\`, `logs\` | 결과 (실행하면 생김) |
| `backup\` | 예전 명령줄 스크립트 00~05 + `sal_common.py` + 예전 README (로컬에만 보관) |

## 5. 예전 스크립트 (`backup\`, 로컬에만 보관 — 저장소에는 없음)
명령줄판 00~05 를 그대로 보관합니다. 경로를 자기 폴더 기준으로 잡으므로 `backup\` 에서 실행하면
`backup\weights\` 를 찾습니다 → `--weights ..\weights\sam3.1_multiplex.pt` 처럼 가중치를 지정해서 쓰세요.
결과도 `backup\sessions\` 등에 생겨 GUI 와 섞이지 않습니다.

## 6. 라이선스
- 이 저장소의 소스 코드: [MIT](LICENSE)
- 외부 라이브러리·모델 가중치는 각자의 라이선스를 따릅니다 → [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
  - Ultralytics(YOLO, SAM 3 실행, YOLOE)·YOLOE·Ultralytics CLIP: **AGPL-3.0**
  - SAM 3 / 3.1 가중치: Meta **SAM License** (Hugging Face 승인제, 배포 시 라이선스 사본 제공, 군사 등 금지 용도)
  - TensorRT: NVIDIA 독점 라이선스 (선택)
- 가중치·데이터는 저장소에 포함하지 않습니다.

## 7. 상태
- 2026-11-20 GUI 작성. 카메라 대신 `chapter05\02_synthetic_conveyor_video` 영상으로 전체 흐름 자동 시험 통과:
  SAM 3.1 불러오기(약 10초) → 자동 라벨(장당 160~240 ms, 20장 객체 218) → 영상 일괄 라벨(+중지) → 검토(클릭 선택·지우기·빼기·되돌리기)
  → 2 epoch 학습(별도 프로세스, yolo26s-seg, 0.7분) → 실시간 추론 → 화면 저장. 시험 결과물은 지움.
- 같은 날 실제 카메라 시험 통과: 카메라 1 Global Shutter 1920x1080 MJPG 60 fps, SAM 3.1 준비 10~11 s, 실시간 프롬프트 6개 차례로 바꿔 미리보기 (장당 195~254 ms, 예: `object` 로 키보드·원통·가방 4개), 자동 라벨 저장. 시험 세션은 지움.
- SAM 3 불러오기 시간 분석: 약 10 s 중 파일 읽기 1 s, **모델을 만들며 무작위 초기화하는 데 약 6 s** (곧 가중치로 덮어씀).
  → 같은 날 **빠른 불러오기 적용** (`core\labeler.py` `skip_random_init`: 모델을 만드는 동안만 초기화를 끔, 실패하면 보통 방법).
  [불러오기] 준비 시간(예열 포함) 7.1 s → 2.7 s (디스크 캐시 전 첫 실행 4.4 s), 카메라 프레임 결과(점수·마스크 넓이) 동일.
