# From Gaze to Data: Creating New Marketing Insights

**Research Period:** September 2024 - December 2024  
**Authors:** Hyeonsik Jo*, Seoyoung Oh*, Junhyeok Lee  
*Equal contribution  
**Affiliation:** Kookmin University

## Abstract

This study proposes a gaze-journey analysis framework that addresses the inherent limitations of purchase-log–based approaches, which capture only outcomes but not the underlying consumer decision processes. The proposed model performs ROI-level gaze estimation and enhances robustness by selectively incorporating 6DRepNet head-pose cues only when the primary model, Gaze360, exhibits high predictive uncertainty. Frame-level gaze predictions are further integrated with multi-object tracking to reconstruct each shopper's continuous gaze flow as a coherent temporal sequence. By quantitatively mapping gaze directions to shelf ROIs and restoring customer-specific gaze transitions over time, the framework generates gaze-based shopping-journey data that cannot be obtained from purchase logs alone.

This design enables stable gaze estimation even under the challenging conditions of top-mounted CCTV—low facial resolution, occlusion, and frequent side/rear views. Moreover, the proposed framework provides practical utility for applications such as product placement optimization, attention analysis, and in-store path design.

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

### 1. Visibility-Aware Gaze Estimation
- **Binocular (양안)**: Gaze360을 사용한 고정밀 시선 추정
- **Monocular (단안)**: 6DRepNet 기반 헤드 포즈 추정으로 fallback
- **Non-visible (비가시)**: 신뢰도가 낮은 프레임 제외

### 2. ROI-based Gaze Mapping
- 점 기반 다각형 ROI 정의 (사다리꼴, 기울어진 형태 지원)
- 방향-거리 기반 ROI 매핑 (깊이 정보 불필요)
- 최소 거리 원칙을 통한 안정적인 ROI 할당

### 3. Multi-Person Tracking
- YOLOv8 기반 사람 탐지
- IoU 기반 경량 추적 알고리즘
- 일관된 ID 유지로 시간적 연속성 보장

### 4. Gaze Journey Reconstruction
- 프레임 레벨 시선 예측을 시간적 시퀀스로 통합
- 고객별 시선 전환 패턴 복원
- ROI 방문 기록을 엑셀 파일로 저장

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
  - ROI 영역 시각화
  - 사람별 ID 및 시선 벡터 표시
  - ROI 매핑 결과 오버레이

- `{video_name}_infer_gaze_journey.xlsx`: ROI 방문 기록 엑셀 파일
  - Person ID별 ROI 방문 기록
  - 시간 순서대로 방문한 ROI 시퀀스

## 실험 결과

### Set-Matching Accuracy

| RoI Transitions | Samples | Set-Matching Accuracy |
|----------------|---------|----------------------|
| 0 transitions  | 6       | 0.83                 |
| 1 transition    | 6       | 0.83                 |
| 2 transitions   | 6       | 0.83                 |
| 3 transitions   | 7       | 0.86                 |
| **Overall**     | **25**  | **0.84**             |

### Participant-wise Performance

- Hyunsik: 0.88
- Junhyeok: 0.80
- Seoyoung: 0.84

## 요구사항

- Python 3.8+
- PyTorch
- OpenCV
- pandas
- ultralytics (YOLOv8)
- 기타 의존성 (requirements.txt 참조)

## 모델 가중치

모델 가중치 파일은 `weights/` 폴더에 저장되어야 합니다:
- `6DRepNet_300W_LP_AFLW2000.pth` (6DRepNet 헤드 포즈 추정)
- `gaze360_model.pth.tar` (Gaze360 시선 추정)
- `yolov8n.pt` (YOLOv8 객체 탐지)
- `yolov8n-face.pt` (YOLOv8 얼굴 탐지)

**참고**: 가중치 파일은 크기 제한으로 인해 Git 저장소에 포함되지 않습니다.

## 데이터셋

- **환경**: 실제 편의점 환경 (서울 소재 대학교 내)
- **카메라**: 천장 고정 CCTV (높이 2.3-2.6m)
- **데이터**: 약 40개 비디오 클립 (각 10-20초)
- **시나리오**: 
  - 단일 인물 시나리오 (~25개)
  - 다중 인물 시나리오 (~15개)
- **ROI 구조**: 6개 Zone (A-F), 각 Zone 내 세부 ROI

## 주요 기여

1. **Overhead CCTV 환경 대응**: 낮은 얼굴 해상도, 비정면 시야, 부분 가림 등 실제 CCTV 환경의 제약 조건 처리
2. **Selective Correction 메커니즘**: Gaze360의 불확실성이 높을 때만 6DRepNet을 선택적으로 활용
3. **ROI 레벨 시선 분석**: 픽셀 단위가 아닌 의미론적 ROI(제품 카테고리, 선반 영역) 기반 분석
4. **Gaze Journey 데이터 생성**: 구매 로그만으로는 얻을 수 없는 고객 시선 여정 데이터 생성

## 라이선스

각 서브모듈의 라이선스를 확인하세요:
- 6DRepNet
- Gaze360
- YOLOv8-DeepSORT-Object-Tracking

## 인용

```bibtex
@article{jo2024gaze2data,
  title={From Gaze to Data: Creating New Marketing Insights},
  author={Jo, Hyeonsik and Oh, Seoyoung and Lee, Junhyeok},
  journal={Kookmin University},
  year={2024}
}
```

## 참고문헌

- Gaze360: Physically Unconstrained Gaze Estimation in the Wild (ICCV 2019)
- 6DRepNet: 6D Rotation Representation for Unconstrained Head Pose Estimation (ICIP 2022)
- YOLOv8: You Only Look Once (2023)
- RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild (CVPR 2020)
