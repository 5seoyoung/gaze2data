# Gaze2Data: ROI-based Gaze Tracking System

ROI(Region of Interest) 기반 시선 추적 시스템입니다. YOLOv8, Gaze360, 6DRepNet을 활용하여 비디오에서 사람의 시선을 추적하고 ROI 매핑을 수행합니다.

## 프로젝트 구조

```
Code_verson2/
├── config/                          # ROI 설정 파일
│   ├── A2roi_default.json          # A2 그룹 ROI 설정
│   ├── C3roi_default.json          # C3 그룹 ROI 설정
│   ├── D5roi_default.json          # D5 그룹 ROI 설정
│   └── roi_default.json            # 기본 ROI 설정
│
├── scripts/                         # 실행 스크립트
│   └── batch_inference.sh          # 배치 인퍼런스 스크립트
│
├── roi_gaze_tracking_gaze360_real.py  # 메인 시선 추적 스크립트
├── setup_roi_only.py               # ROI 설정 도구
│
├── 6DRepNet/                        # 6DRepNet 모델 (헤드 포즈 추정)
├── gaze360/                         # Gaze360 모델 (시선 추정)
└── YOLOv8-DeepSORT-Object-Tracking/ # YOLOv8 및 DeepSORT (객체 탐지 및 추적)
```

## 주요 기능

- **다중 인물 시선 추적**: YOLOv8을 사용한 사람 탐지 및 추적
- **ROI 기반 시선 매핑**: 사용자 정의 ROI 영역에 대한 시선 분석
- **Gaze360 모델**: 정확한 시선 방향 추정
- **6DRepNet 모델**: 헤드 포즈 추정 (fallback)
- **배치 처리**: 여러 비디오 파일 일괄 처리

## 사용 방법

### 1. ROI 설정

```bash
python3 setup_roi_only.py --source Input/video.mov --output config/roi_config.json
```

### 2. 단일 비디오 인퍼런스

```bash
python3 roi_gaze_tracking_gaze360_real.py \
    --source Input/video.mov \
    --output Results/output_infer.mp4 \
    --roi_config config/D5roi_default.json \
    --gpu -1 \
    --min_frames 5
```

### 3. 배치 인퍼런스

```bash
bash scripts/batch_inference.sh
```

## 출력 파일

- `{video_name}_infer.mp4`: 시선 추적 결과 비디오
- `{video_name}_infer_gaze_journey.xlsx`: ROI 방문 기록 엑셀 파일

## 요구사항

- Python 3.8+
- PyTorch
- OpenCV
- pandas
- ultralytics (YOLOv8)
- 기타 의존성 (requirements.txt 참조)

## 모델 가중치

모델 가중치 파일은 `weights/` 폴더에 저장되어야 합니다:
- `6DRepNet_300W_LP_AFLW2000.pth`
- `gaze360_model.pth.tar`
- `yolov8n.pt`
- `yolov8n-face.pt`

**참고**: 가중치 파일은 크기 제한으로 인해 Git 저장소에 포함되지 않습니다.

## 라이선스

각 서브모듈의 라이선스를 확인하세요:
- 6DRepNet
- Gaze360
- YOLOv8-DeepSORT-Object-Tracking
