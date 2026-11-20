# 외부 라이선스와 출처 (Third-Party Notices)

이 저장소의 소스 코드는 [MIT 라이선스](LICENSE)입니다.
그러나 실행에 필요한 아래 외부 라이브러리와 모델 가중치는 **각자의 라이선스**를 따르며, MIT 라이선스가 적용되지 않습니다.
이 저장소에는 외부 코드나 가중치 파일을 넣지 않았습니다 (설치·내려받기는 사용자가 직접 하며, 그때 각 라이선스에 동의하게 됩니다).

확인 기준: 2026-11-20, 아래 버전 (설치 패키지 메타데이터, 각 저장소의 LICENSE).
라이선스 조건은 바뀔 수 있으니 사용·배포 전에 원문을 확인하세요. 이 문서는 법률 자문이 아닙니다.

## 꼭 확인할 것

### 1. Ultralytics — AGPL-3.0
- 이 프로그램은 YOLO 학습·추론, SAM 3 실행(`SAM3SemanticPredictor`), YOLOE 실행에 Ultralytics 패키지를 불러옵니다.
- AGPL-3.0 은 이 패키지를 포함·결합해 **배포하거나 네트워크 서비스로 제공**할 때 해당 프로그램 전체의 소스 공개 등 조건을 요구할 수 있습니다.
  이 저장소의 코드 자체는 MIT 이지만, Ultralytics 와 함께 배포·서비스할 때는 AGPL-3.0 조건을 따라야 할 수 있습니다.
  상업적으로 소스 공개 없이 쓰려면 Ultralytics Enterprise License 를 확인하세요.
- Ultralytics 가 배포하는 사전학습 가중치(YOLO26, YOLO11, YOLOE)도 AGPL-3.0 으로 배포됩니다.
- 출처: https://github.com/ultralytics/ultralytics (LICENSE: https://github.com/ultralytics/ultralytics/blob/main/LICENSE), https://www.ultralytics.com/license

### 2. SAM 3 / SAM 3.1 모델 — Meta "SAM License" (2025-11-19)
- 사용 가중치: `sam3.1_multiplex.pt` (https://huggingface.co/facebook/sam3.1), 대안 `sam3.pt` (https://huggingface.co/facebook/sam3)
- Hugging Face 에서 **접근 승인(gated)** 을 받은 뒤 내려받습니다. 이 저장소에는 포함하지 않습니다.
- 라이선스 원문: https://github.com/facebookresearch/sam3/blob/main/LICENSE (각 Hugging Face 저장소의 `LICENSE` 도 같음)
- 주요 조건 (원문 요약, 전체는 원문 참조):
  - 사용·복제·수정·배포를 허용하는 비독점·무상 라이선스.
  - SAM Materials(가중치 등)나 그 파생물을 제3자에게 배포하면 **이 라이선스 사본을 함께 제공**해야 하고, 이 라이선스 조건으로만 배포할 수 있음.
  - 연구 결과를 발표하면 SAM Materials 사용을 **명시**해야 함.
  - 수출 통제 법규 준수, 군사·전쟁, 핵, 첩보, 무기 개발 등 금지된 용도로 사용 금지.
  - 보증 없음("AS IS").
- 이 프로그램으로 만든 라벨(SAM 3 출력)과 그 라벨로 학습한 모델을 배포할 때도 위 조건을 확인하세요.
- 논문: SAM 3: Segment Anything with Concepts (Meta, 2025). 코드: https://github.com/facebookresearch/sam3

### 3. YOLOE (모든 물체 모드의 이름 찾기) — AGPL-3.0
- 사용 가중치: `yoloe-26l-seg-pf.pt` (기본), `yoloe-26s-seg-pf.pt`, `yoloe-11l-seg-pf.pt`, `yoloe-11s-seg-pf.pt`
  (Ultralytics assets 에서 내려받음: https://github.com/ultralytics/assets/releases)
- 원 연구·코드: THU-MIG, "YOLOE: Real-Time Seeing Anything", https://github.com/THU-MIG/yoloe (AGPL-3.0)

### 4. NVIDIA TensorRT — NVIDIA 독점 라이선스 (선택)
- 실시간 탭의 TensorRT 엔진 변환·실행에 사용 (`tensorrt_cu12`). 쓰지 않아도 PyTorch 로 동작합니다.
- 라이선스: NVIDIA Software License Agreement — https://developer.nvidia.com/tensorrt , https://github.com/NVIDIA/TensorRT
- 만들어진 `.engine` 파일은 이 저장소에 포함하지 않습니다.

## 파이썬 패키지

| 패키지 | 버전 | 라이선스 (설치 메타데이터) | 쓰는 곳 | 출처 |
| --- | --- | --- | --- | --- |
| ultralytics | 8.4.193 | AGPL-3.0 | YOLO 학습·추론, SAM 3, YOLOE | https://github.com/ultralytics/ultralytics |
| clip (Ultralytics 판) | 1.0 | AGPL-3.0 (저장소 LICENSE) | SAM 3 텍스트 인코더 | https://github.com/ultralytics/CLIP (원본 OpenAI CLIP, MIT: https://github.com/openai/CLIP) |
| torch | 2.14.1 | BSD-3-Clause 계열 (번들 구성 요소 포함 복합: Apache-2.0, BSD-2-Clause 등) | 딥러닝 실행 | https://github.com/pytorch/pytorch |
| torchvision | 0.29.1 | BSD-3-Clause | Ultralytics 의존 | https://github.com/pytorch/vision |
| tensorrt_cu12 | 10.16.1.11 | NVIDIA 독점 (Proprietary) | TensorRT (선택) | https://github.com/nvidia/tensorrt |
| onnx | 1.23.2 | Apache-2.0 | TensorRT 변환 중간 단계 | https://github.com/onnx/onnx |
| onnxslim | 0.1.98 | MIT | TensorRT 변환 중간 단계 | https://github.com/inisis/OnnxSlim |
| opencv-python | 5.0.0.93 | Apache-2.0 | 카메라·영상·그리기 | https://github.com/opencv/opencv-python |
| numpy | 2.4.6 | BSD-3-Clause (번들: 0BSD, MIT, Zlib, CC0-1.0) | 배열 계산 | https://github.com/numpy/numpy |
| pillow | 12.3.0 | MIT-CMU | 화면 표시, 한글 글자 | https://github.com/python-pillow/Pillow |
| pygrabber | 0.2 | MIT | 카메라 이름 읽기 (DirectShow) | https://github.com/andreaschiavinato/python_grabber |
| huggingface_hub | 2.4.0 | Apache-2.0 | SAM 3 가중치 내려받기 | https://github.com/huggingface/huggingface_hub |
| ftfy | 6.3.1 | Apache-2.0 | CLIP 의존 | https://github.com/rspeer/python-ftfy |
| wcwidth | 0.9.4 | MIT | ftfy 의존 | https://github.com/jquast/wcwidth |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python | CLIP 의존 | https://github.com/mrabarnett/mrab-regex |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | Ultralytics 의존 | https://github.com/tqdm/tqdm |
| tkinter (Python 표준 라이브러리) | Python 3.11 | PSF License | GUI | https://www.python.org |

`constraints.txt` 는 개발 환경(`physical_ai`)의 전체 패키지 버전 목록이며, 위 표에 없는 패키지는 이 프로그램이 직접 쓰지 않습니다.

## 그 밖의 자원
- **글꼴**: 화면의 한글 글자는 Windows 시스템 글꼴 "맑은 고딕(Malgun Gothic, `C:\Windows\Fonts\malgun.ttf`)"을 불러 씁니다. 글꼴 파일은 포함하지 않습니다 (Microsoft 소유).
- **카메라 스펙 파일**: 카메라 기본값은 같은 작업 폴더의 다른 프로젝트(`chapter05\01_camera_recognized\camera_profiles.json`)를 읽기만 합니다. 이 저장소에는 포함하지 않으며, 없으면 기본값(1920x1080, MJPG)으로 동작합니다.
- **학습 데이터**: 저장소에 데이터셋·촬영 사진·학습 결과를 포함하지 않습니다. 사용자가 촬영해 만든 데이터의 권리와 개인정보(사람 얼굴 등)는 사용자 책임입니다.
