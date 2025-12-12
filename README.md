# From Gaze to Data: Creating New Marketing Insights

**Research Period:** 2025.09.01-2025.12.12  
**Authors:** Hyeonsik Jo*, Seoyoung Oh*, Junhyeok Lee  
*Equal contribution  
**Affiliation:** Kookmin University

## Abstract

This study proposes a gaze-journey analysis framework that addresses the limitations of purchase-log–based approaches by capturing consumer decision processes through ROI-level gaze estimation. The framework integrates Gaze360 with 6DRepNet head-pose cues for robust estimation under challenging CCTV conditions, and employs multi-object tracking to reconstruct continuous gaze flows. By mapping gaze directions to shelf ROIs, the system generates shopping-journey data that cannot be obtained from purchase logs alone.

![Figure 1: Overview of the proposed visibility-aware gaze estimation framework.](https://github.com/user-attachments/assets/b5163cb5-69c8-49d2-9004-b652c68cebb2)

## Project Structure

```
Code_verson2/
├── config/                          # ROI configuration files
│   ├── A2roi_default.json          # A2 group ROI configuration
│   ├── C3roi_default.json          # C3 group ROI configuration
│   ├── D5roi_default.json          # D5 group ROI configuration
│   └── roi_default.json            # Default ROI configuration
│
├── scripts/                         # Execution scripts
│   └── batch_inference.sh          # Batch inference script
│
├── roi_gaze_tracking_gaze360_real.py  # Main gaze tracking script
├── setup_roi_only.py               # ROI setup tool
│
├── 6DRepNet/                        # 6DRepNet model (head pose estimation)
├── gaze360/                         # Gaze360 model (gaze estimation)
└── YOLOv8-DeepSORT-Object-Tracking/ # YOLOv8 and DeepSORT (object detection and tracking)
```

## Key Features

### 1. Visibility-Aware Gaze Estimation
- **Binocular**: High-precision gaze estimation using Gaze360
- **Monocular**: Fallback to 6DRepNet-based head pose estimation
- **Non-visible**: Exclude frames with low reliability

### 2. ROI-based Gaze Mapping
- Point-based polygon ROI definition (supports trapezoids, skewed shapes)
- Direction-distance based ROI mapping (no depth information required)
- Stable ROI assignment using minimum distance principle

![Figure 2: Visualization of three RoI examples defined through our point-based polygon annotation system.](https://github.com/user-attachments/assets/67298dde-cdd5-423f-ae38-0505c39adc32)

### 3. Multi-Person Tracking
- YOLOv8-based person detection
- Lightweight IoU-based tracking algorithm
- Consistent ID maintenance for temporal continuity

### 4. Gaze Journey Reconstruction
- Integration of frame-level gaze predictions into temporal sequences
- Restoration of customer-specific gaze transition patterns
- Export ROI visit records to Excel files

![Figure 3: Examples of RoI mapping results produced by our point-based polygon annotation system, illustrating stable detection and assignment in both single- and multi-person scenarios.](https://github.com/user-attachments/assets/52be8b78-0105-4637-b1be-ce529c7da1a0)

## Usage

### 1. ROI Configuration

```bash
python3 setup_roi_only.py --source Input/video.mov --output config/roi_config.json
```

### 2. Single Video Inference

```bash
python3 roi_gaze_tracking_gaze360_real.py \
    --source Input/video.mov \
    --output Results/output_infer.mp4 \
    --roi_config config/D5roi_default.json \
    --gpu -1 \
    --min_frames 5
```

### 3. Batch Inference

```bash
bash scripts/batch_inference.sh
```

## Output Files

- `{video_name}_infer.mp4`: Gaze tracking result video
  - ROI region visualization
  - Person ID and gaze vector display
  - ROI mapping result overlay

- `{video_name}_infer_gaze_journey.xlsx`: ROI visit record Excel file
  - ROI visit records per Person ID
  - ROI sequence visited in chronological order

## Experimental Results

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

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- pandas
- ultralytics (YOLOv8)
- Other dependencies (see requirements.txt)

## Model Weights

Model weight files should be stored in the `weights/` folder:
- `6DRepNet_300W_LP_AFLW2000.pth` (6DRepNet head pose estimation)
- `gaze360_model.pth.tar` (Gaze360 gaze estimation)
- `yolov8n.pt` (YOLOv8 object detection)
- `yolov8n-face.pt` (YOLOv8 face detection)

**Note**: Weight files are not included in the Git repository due to size limitations.

## Dataset

- **Environment**: Real convenience store environment (located in a university in Seoul)
- **Camera**: Ceiling-mounted fixed CCTV (height 2.3-2.6m)
- **Data**: Approximately 40 video clips (10-20 seconds each)
- **Scenarios**: 
  - Single-person scenarios (~25 clips)
  - Multi-person scenarios (~15 clips)
- **ROI Structure**: 6 Zones (A-F), detailed ROIs within each Zone

## Key Contributions

1. **Overhead CCTV Environment Adaptation**: Handles constraints of real CCTV environments including low facial resolution, non-frontal views, and partial occlusions
2. **Selective Correction Mechanism**: Selectively incorporates 6DRepNet only when Gaze360 exhibits high uncertainty
3. **ROI-level Gaze Analysis**: Semantic ROI-based analysis (product categories, shelf regions) rather than pixel-level
4. **Gaze Journey Data Generation**: Generates customer gaze journey data that cannot be obtained from purchase logs alone

## License

Please check the license of each submodule:
- 6DRepNet
- Gaze360
- YOLOv8-DeepSORT-Object-Tracking

## References

- Gaze360: Physically Unconstrained Gaze Estimation in the Wild (ICCV 2019)
- 6DRepNet: 6D Rotation Representation for Unconstrained Head Pose Estimation (ICIP 2022)
- YOLOv8: You Only Look Once (2023)
- RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild (CVPR 2020)
