#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoI 기반 시선 추적 시스템 (Gaze360 버전)
YOLOv8 + IoU Tracking + Gaze360을 결합한 시스템 + RoI 응시 분석
"""

import os
import sys
import argparse
import math
import re
from collections import deque, defaultdict, OrderedDict

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn  # noqa: F401
from torchvision import transforms
from typing import Literal, Optional

try:
    import mediapipe as mp

    MP_AVAILABLE = True
except Exception as e:
    print(f"Warning: Mediapipe을 사용할 수 없습니다. 일부 기능이 제한됩니다. ({e})")
    MP_AVAILABLE = False

# =========================
# Gaze360 관련 import
# =========================
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "gaze360/code")
)
from model import GazeLSTM  # type: ignore[import]
from resnet import resnet18  # type: ignore[import]

# =========================
# 6DRepNet 관련 import (head pose fallback용)
# =========================
sixdrepnet_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "6DRepNet"
)
if os.path.exists(sixdrepnet_path):
    sys.path.insert(0, sixdrepnet_path)

SIXDREPNET_AVAILABLE = False
try:
    from sixdrepnet.model import SixDRepNet  # type: ignore[import]
    from sixdrepnet import utils as repnet_utils  # type: ignore[import]

    SIXDREPNET_AVAILABLE = True
except Exception as e:
    print(
        f"Warning: 6DRepNet을 import할 수 없습니다. "
        f"Head pose fallback이 비활성화됩니다. ({e})"
    )
    SIXDREPNET_AVAILABLE = False

# =========================
# YOLOv8 관련 import
# =========================
from ultralytics import YOLO

# =========================
# 얼굴 탐지 (RetinaFace) 관련
# =========================
FACE_DETECTOR_AVAILABLE = False
RetinaFace = None  # type: ignore[assignment]

try:
    # 일반 패키지 형태
    from face_detection import RetinaFace as _RetinaFace  # type: ignore[import]

    RetinaFace = _RetinaFace
    FACE_DETECTOR_AVAILABLE = True
except Exception:
    try:
        # sixdrepnet 경로 등에 함께 있을 수 있는 경우
        import face_detection  # type: ignore[import]

        from face_detection import RetinaFace as _RetinaFace  # type: ignore[import]

        RetinaFace = _RetinaFace
        FACE_DETECTOR_AVAILABLE = True
    except Exception as e2:
        print(
            f"Warning: RetinaFace를 import할 수 없습니다. "
            f"얼굴 탐지가 비활성화됩니다. ({e2})"
        )
        FACE_DETECTOR_AVAILABLE = False

# =========================
# JSON config 지원
# =========================
import json

# =========================
# MiDaS Depth 추정 관련
# =========================
DEPTH_AVAILABLE = False
try:
    import torch.hub
    DEPTH_AVAILABLE = True
except Exception:
    pass

# =========================
# 전역 상태 (간단 IoU tracker & ROI 드로잉용)
# =========================
data_deque: dict[int, deque] = {}
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
next_id = 1
tracked_objects: dict[int, dict] = {}
LEFT_EYE_IDX = [33, 133, 160, 158, 157, 173, 246, 161, 163]
RIGHT_EYE_IDX = [362, 263, 387, 385, 384, 398, 466, 373, 380]
RETINAFACE_EYE_ORDER = {
    "right": 0,
    "left": 1,
    "nose": 2,
}

# RoI 인터랙티브 설정용 전역 (마우스 콜백)
rois: list = []
current_polygon_points: list[tuple[int, int]] = []
drawing = False
current_roi = None
polygon_mode = False
roi_colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
]


def mouse_callback(event, x, y, flags, param):
    """다각형 RoI 설정을 위한 마우스 콜백 함수 (항상 polygon 전용)"""
    global rois, current_polygon_points

    if event == cv2.EVENT_LBUTTONDOWN:
        # 4개 점만 받도록 제한
        if len(current_polygon_points) < 4:
            current_polygon_points.append((x, y))
            print(f"폴리곤 점 추가: ({x}, {y}) ({len(current_polygon_points)}/4)")
            if len(current_polygon_points) == 4:
                print("4개 점 모두 추가됨. 'p' 키를 눌러 완료하세요.")
        else:
            print("이미 4개의 점이 추가되었습니다. 'p' 키를 눌러 완료하거나 'u' 키로 점을 제거하세요.")
    elif event == cv2.EVENT_RBUTTONDOWN and current_polygon_points:
        removed = current_polygon_points.pop()
        print(f"폴리곤 점 제거: {removed} (남은 점: {len(current_polygon_points)}/4)")


class RoIGazeTrackerGaze360:
    def __init__(
        self,
        yolov8_model_path: str = "weights/yolov8n.pt",
        gaze360_model_path: str = "weights/gaze360_model.pth.tar",
        sixdrepnet_model_path: str = "weights/6DRepNet_300W_LP_AFLW2000.pth",
        gpu_id: int = 0,
        min_frames_threshold: int = 5,
        gaze_distance_k: int = 300,
        sixd_case: int = 1,
        vector_weight: float = 0.35,
        eye_mode: str = "auto",
        yolo_face_model_path: Optional[str] = None,
        yaw_threshold_both: float = 40.0,
        yaw_threshold_single: float = 90.0,
        pitch_threshold_both: float = 30.0,
        pitch_threshold_single: float = 60.0,
        prediction_interval: int = 5,
        smoothing_window: int = 3,
        show_gaze_ray: bool = True,
        use_depth: bool = False,
        depth_model_type: str = "DPT_Large",
        ray_angle_thresh: float = 0.2,
        ray_shrink_px: int = 6,
        ray_max_len: float = 0.0,
        interactive_k: bool = False,
        interactive_face_center: bool = False,
    ):
        """
        RoI 기반 시선 추적 시스템 초기화 (Gaze360 버전)

        Args:
            yolov8_model_path: YOLOv8 모델 경로
            gaze360_model_path: Gaze360 모델 경로
            sixdrepnet_model_path: 6DRepNet 모델 경로 (head pose fallback용)
            gpu_id: GPU ID (CPU 사용시 -1)
            min_frames_threshold: RoI 응시 판단을 위한 최소 프레임 수
            gaze_distance_k: 시선 끝점 계산을 위한 거리 상수 k (픽셀, endpoint 모드용)
            sixd_case: 단안 모드에서 사용할 6D 보정 케이스 (1~4)
            vector_weight: 벡터 보정 가중치
            eye_mode: 눈 판별 모드 (auto/mediapipe/retinaface/yoloface/yaw_based)
            yolo_face_model_path: YOLO face 모델 경로 (landmark 포함 가중치)
            yaw_threshold_both: 양안 확보를 위한 Yaw 임계값 (기본값: 40도)
            yaw_threshold_single: 단안을 위한 Yaw 임계값 (기본값: 90도)
            pitch_threshold_both: 양안 확보를 위한 Pitch 임계값 (기본값: 30도)
            pitch_threshold_single: 단안을 위한 Pitch 임계값 (기본값: 60도)
            prediction_interval: 시선 추정을 수행할 프레임 간격 (기본값: 5, 매 프레임마다는 1)
            smoothing_window: 시선 벡터 스무딩을 위한 히스토리 윈도우 크기 (기본값: 3, 0이면 스무딩 비활성화)
        """
        self.gpu_id = gpu_id
        self.min_frames_threshold = min_frames_threshold
        self.gaze_distance_k = gaze_distance_k
        self.sixd_case = sixd_case
        self.vector_weight = vector_weight
        self.vector_depth = 1000.0
        self.eye_mode = eye_mode
        self.show_gaze_ray = show_gaze_ray
        self.yolo_face_model_path = yolo_face_model_path
        self.simple_eye_threshold = 0.35
        self.yaw_threshold_both = yaw_threshold_both
        self.yaw_threshold_single = yaw_threshold_single
        self.pitch_threshold_both = pitch_threshold_both
        self.pitch_threshold_single = pitch_threshold_single
        self.prediction_interval = max(1, prediction_interval)  # 최소 1
        self.smoothing_window = max(0, smoothing_window)  # 0이면 스무딩 비활성화
        # ROI 평면 (3D 교차용): origin/normal이 모두 설정될 때만 사용
        self.roi_plane_origin: Optional[np.ndarray] = None
        self.roi_plane_normal: Optional[np.ndarray] = None
        # Depth 추정 관련
        self.use_depth = use_depth
        self.depth_model_type = depth_model_type
        self.depth_model = None
        self.depth_transform = None
        self.last_depth_update_frame = -1
        self.ray_angle_thresh = ray_angle_thresh
        self.ray_shrink_px = max(0, int(ray_shrink_px))
        self.ray_max_len = float(ray_max_len)
        self.interactive_k = interactive_k
        self.interactive_face_center = interactive_face_center
        # k점 앵커 (interactive_k로 설정한 기준점)
        self.k_anchor_point: Optional[tuple[float, float]] = None
        # 수동 얼굴 중심 (옵션)
        self.manual_face_center: Optional[tuple[float, float]] = None
        # ROI 선택 각도 임계값 (도 단위, 기본값: 30도 - 더 엄격하게)
        self.roi_angle_threshold_deg = 30.0
        # 디버깅 모드: 계산값 출력 및 시각화
        self.debug_mode = False
        self.debug_roi_info = {}  # {roi_id: {"angle": deg, "distance": px, "score": val}}

        # 디바이스 설정
        if gpu_id >= 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"GPU {gpu_id} 사용")
        else:
            self.device = torch.device("cpu")
            print("CPU 사용")

        # Gaze360 모델
        self.init_gaze_model(gaze360_model_path)

        # 6DRepNet (fallback head pose)
        if SIXDREPNET_AVAILABLE:
            self.init_pose_model(sixdrepnet_model_path)
        else:
            self.pose_model = None

        # YOLOv8 사람 검출
        self.init_detection_model(yolov8_model_path)

        # 간단 IoU tracker 초기화
        self.init_tracker()

        # 전처리 변환들
        self.image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.transformations = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.image_normalize,
            ]
        )

        self.pose_transformations = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.body_transformations = transforms.Compose(
            [
                transforms.Resize((256, 192)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Mediapipe 초기화
        self.face_mesh = None
        self.pose_estimator = None
        if MP_AVAILABLE:
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.pose_estimator = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                print("Mediapipe FaceMesh/Pose 초기화 완료")
            except Exception as e:
                print(f"Warning: Mediapipe 초기화 실패 ({e})")
                self.face_mesh = None
                self.pose_estimator = None

        # YOLO face 모델 (landmark 지원)
        self.yolo_face_model = None
        if self.eye_mode in ("yoloface", "auto"):
            if yolo_face_model_path and os.path.exists(yolo_face_model_path):
                try:
                    self.yolo_face_model = YOLO(yolo_face_model_path)
                    print("YOLO face 모델 로딩 완료")
                except Exception as e:
                    print(f"Warning: YOLO face 모델 로딩 실패 ({e})")
                    self.yolo_face_model = None
            else:
                if self.eye_mode == "yoloface":
                    print(
                        "Warning: YOLO face 모델 경로가 유효하지 않습니다. "
                        "eye_mode를 mediapipe로 대체합니다."
                    )
                    self.eye_mode = "mediapipe"

        # 얼굴 탐지기
        if FACE_DETECTOR_AVAILABLE and RetinaFace is not None:
            try:
                face_gpu_id = -1 if self.device.type == "cpu" else gpu_id
                self.face_detector = RetinaFace(gpu_id=face_gpu_id)
            except Exception as e:
                print(f"Warning: RetinaFace 초기화 실패 ({e})")
                self.face_detector = None
        else:
            self.face_detector = None

        # Depth 모델 초기화 (선택적)
        if self.use_depth:
            self.init_depth_model()

        # RoI 리스트 (rect / poly 모두 dict 형태로 저장)
        self.rois: list[dict] = []

        # 사람별 시선 journey 저장
        self.person_gaze_data: dict[int, dict] = defaultdict(
            lambda: {
                "current_roi": None,
                "frame_count": 0,
                "weighted_count": 0.0,
                "gaze_journey": [],
                "temp_roi": None,
                "temp_frame_count": 0,
                "temp_weighted_count": 0.0,
                "last_gaze": None,
            }
        )

    # -------------------------
    # 모델 초기화
    # -------------------------
    def init_gaze_model(self, model_path: str):
        """Gaze360 모델 초기화"""
        print("Gaze360 모델을 로딩하는 중...")
        self.gaze_model = GazeLSTM()
        checkpoint = torch.load(model_path, map_location="cpu")

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            self.gaze_model.load_state_dict(new_state_dict)
        else:
            self.gaze_model.load_state_dict(checkpoint)

        self.gaze_model.to(self.device)
        self.gaze_model.eval()
        print("Gaze360 모델 로딩 완료")

    def init_pose_model(self, model_path: str):
        """6DRepNet 모델 초기화 (head pose fallback용)"""
        if not SIXDREPNET_AVAILABLE:
            self.pose_model = None
            return

        print("6DRepNet 모델을 로딩하는 중...")
        self.pose_model = SixDRepNet(
            backbone_name="RepVGG-B1g2",
            backbone_file="",
            deploy=True,
            pretrained=False,
        )

        saved_state_dict = torch.load(model_path, map_location="cpu")
        if "model_state_dict" in saved_state_dict:
            self.pose_model.load_state_dict(saved_state_dict["model_state_dict"])
        else:
            self.pose_model.load_state_dict(saved_state_dict)

        self.pose_model.to(self.device)
        self.pose_model.eval()
        print("6DRepNet 모델 로딩 완료")

    def init_depth_model(self):
        """MiDaS depth 모델 초기화"""
        if not DEPTH_AVAILABLE:
            print("Warning: torch.hub를 사용할 수 없습니다. Depth 추정이 비활성화됩니다.")
            self.use_depth = False
            return

        try:
            print(f"MiDaS depth 모델을 로딩하는 중... (모델: {self.depth_model_type})")
            # MiDaS 모델 로드 (torch.hub 사용)
            self.depth_model = torch.hub.load("intel-isl/MiDaS", self.depth_model_type)
            self.depth_model.to(self.device)
            self.depth_model.eval()

            # MiDaS 전처리 변환
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if self.depth_model_type == "DPT_Large" or self.depth_model_type == "DPT_Hybrid":
                self.depth_transform = midas_transforms.dpt_transform
            else:
                self.depth_transform = midas_transforms.small_transform

            print("MiDaS depth 모델 로딩 완료")
        except Exception as e:
            print(f"Warning: MiDaS depth 모델 로딩 실패 ({e}). Depth 추정이 비활성화됩니다.")
            self.use_depth = False
            self.depth_model = None
            self.depth_transform = None

    def init_detection_model(self, model_path: str):
        """YOLOv8 모델 초기화"""
        print("YOLOv8 모델을 로딩하는 중...")
        self.detection_model = YOLO(model_path)
        print("YOLOv8 모델 로딩 완료")

    def init_tracker(self):
        """간단 IoU 기반 추적기 초기화"""
        global next_id, tracked_objects, data_deque
        print("IoU 기반 추적기를 초기화하는 중...")
        next_id = 1
        tracked_objects = {}
        data_deque = {}
        print("추적기 초기화 완료")

    # -------------------------
    # IoU tracking
    # -------------------------
    @staticmethod
    def compute_iou(box1, box2) -> float:
        """두 바운딩 박스 간의 IoU 계산"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def simple_track(self, detections, frame_count: int):
        """간단한 IoU 기반 추적"""
        global next_id, tracked_objects, data_deque

        # 오래 안 보인 트랙 제거
        to_remove = [
            obj_id
            for obj_id, obj_data in tracked_objects.items()
            if frame_count - obj_data["last_seen"] > 5
        ]
        for obj_id in to_remove:
            tracked_objects.pop(obj_id, None)
            data_deque.pop(obj_id, None)

        matched_detections = set()
        matched_tracks = set()

        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            best_iou = 0.3
            best_match = None

            for obj_id, obj_data in tracked_objects.items():
                if obj_id in matched_tracks:
                    continue
                iou = self.compute_iou(
                    [x1, y1, x2, y2],
                    obj_data["bbox"],
                )
                if iou > best_iou:
                    best_iou = iou
                    best_match = obj_id

            if best_match is not None:
                tracked_objects[best_match]["bbox"] = [x1, y1, x2, y2]
                tracked_objects[best_match]["last_seen"] = frame_count
                matched_detections.add(i)
                matched_tracks.add(best_match)
            else:
                tracked_objects[next_id] = {
                    "bbox": [x1, y1, x2, y2],
                    "last_seen": frame_count,
                    "face_frames": deque(maxlen=7),
                    "last_prediction_frame": -1,
                    "cached_gaze_result": None,
                    "gaze_history": deque(maxlen=max(1, self.smoothing_window)) if self.smoothing_window > 0 else None,  # (yaw, pitch, gaze_vector) 튜플 저장
                }
                next_id += 1
                matched_detections.add(i)

        tracks = []
        for obj_id, obj_data in tracked_objects.items():
            if obj_data["last_seen"] == frame_count:
                x1, y1, x2, y2 = obj_data["bbox"]
                tracks.append([x1, y1, x2, y2, obj_id, 0])

        return tracks

    # -------------------------
    # Drawing helpers
    # -------------------------
    @staticmethod
    def compute_color_for_labels(label: int):
        """클래스에 따른 고정 색상 계산"""
        if label == 0:
            color = (85, 45, 255)
        else:
            color = [
                int((p * (label ** 2 - label + 1)) % 255) for p in palette
            ]
        return tuple(color)

    @staticmethod
    def draw_border(img, pt1, pt2, color, thickness, r, d):
        """둥근 모서리 박스 그리기"""
        x1, y1 = pt1
        x2, y2 = pt2

        cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
        cv2.ellipse(
            img,
            (x1 + r, y1 + r),
            (r, r),
            180,
            0,
            90,
            color,
            thickness,
        )

        cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(
            img,
            (x2 - r, y1 + r),
            (r, r),
            270,
            0,
            90,
            color,
            thickness,
        )

        cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(
            img,
            (x1 + r, y2 - r),
            (r, r),
            90,
            0,
            90,
            color,
            thickness,
        )

        cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(
            img,
            (x2 - r, y2 - r),
            (r, r),
            0,
            0,
            90,
            color,
            thickness,
        )

        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(
            img,
            (x1, y1 + r),
            (x2, y2 - r - d),
            color,
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
        cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
        cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
        cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)

        return img

    def UI_box(self, x, img, color=None, label=None, line_thickness=None):
        """이미지에 바운딩 박스 그리기"""
        tl = (
            line_thickness
            or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        )
        color = color or [
            np.random.randint(0, 255) for _ in range(3)
        ]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf
            )[0]
            top_left = (c1[0], c1[1] - t_size[1] - 4)
            bottom_right = (c1[0] + t_size[0] + 4, c1[1] + 4)

            self.draw_border(
                img,
                top_left,
                bottom_right,
                color,
                1,
                8,
                2,
            )

            cv2.putText(
                img,
                label,
                (c1[0] + 2, c1[1] - 2),
                0,
                tl / 3,
                (225, 255, 255),
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

    # -------------------------
    # 시선 추정 / 품질 / 융합
    # -------------------------
    def estimate_gaze(self, face_frames):
        """
        얼굴 프레임들에서 시선 추정 (Gaze360)
        Returns: (yaw_deg, pitch_deg, gaze_vector, mode, weight)
        """
        try:
            if len(face_frames) < 7:
                return None, None, None, "invalid", 0.0

            frames_tensor = torch.stack(face_frames).unsqueeze(0).to(
                self.device
            )

            with torch.no_grad():
                angular_output, var = self.gaze_model(frames_tensor)

            yaw = angular_output[0, 0].cpu().numpy()
            pitch = angular_output[0, 1].cpu().numpy()

            yaw_deg = float(yaw * 180.0 / np.pi)
            pitch_deg = float(pitch * 180.0 / np.pi)

            gaze_vector = np.array(
                [
                    np.cos(pitch) * np.sin(yaw),
                    np.sin(pitch),
                    -np.cos(pitch) * np.cos(yaw),
                ]
            )

            return yaw_deg, pitch_deg, gaze_vector, "gaze360", 1.0
        except Exception as e:
            print(f"시선 추정 오류: {e}")
            return None, None, None, "invalid", 0.0

    def estimate_head_pose(self, face_roi):
        """
        얼굴 영역에서 head pose 추정 (6DRepNet fallback)
        """
        if self.pose_model is None:
            return None, None, None, "invalid", 0.0

        try:
            face_img = Image.fromarray(face_roi)
            face_img = face_img.convert("RGB")
            face_img = self.pose_transformations(face_img)
            face_img = torch.Tensor(face_img[None, :]).to(self.device)

            with torch.no_grad():
                R_pred = self.pose_model(face_img)
                euler = (
                    repnet_utils.compute_euler_angles_from_rotation_matrices(
                        R_pred
                    )
                    * 180
                    / np.pi
                )
                pitch_deg = float(euler[:, 0].cpu().numpy()[0])
                yaw_deg = float(euler[:, 1].cpu().numpy()[0])

            yaw_rad = yaw_deg * np.pi / 180.0
            pitch_rad = pitch_deg * np.pi / 180.0

            gaze_vector = np.array(
                [
                    np.cos(pitch_rad) * np.sin(yaw_rad),
                    np.sin(pitch_rad),
                    -np.cos(pitch_rad) * np.cos(yaw_rad),
                ]
            )

            return yaw_deg, pitch_deg, gaze_vector, "headpose", 0.7
        except Exception as e:
            print(f"Head pose 추정 오류: {e}")
            return None, None, None, "invalid", 0.0

    @staticmethod
    def yaw_pitch_to_vector(yaw_deg: float, pitch_deg: float) -> np.ndarray:
        yaw_rad = np.radians(yaw_deg)
        pitch_rad = np.radians(pitch_deg)
        return np.array(
            [
                np.cos(pitch_rad) * np.sin(yaw_rad),
                np.sin(pitch_rad),
                -np.cos(pitch_rad) * np.cos(yaw_rad),
            ]
        )

    def vector_to_angles(self, dx: float, dy: float) -> tuple[float, float]:
        dz = self.vector_depth
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0, 0.0
        yaw_rad = math.atan2(dx, dz)
        hyp = math.sqrt(dx * dx + dz * dz)
        pitch_rad = math.atan2(dy, hyp)
        return float(np.degrees(yaw_rad)), float(np.degrees(pitch_rad))

    def smooth_gaze_result(
        self, gaze_history: deque, current_result: tuple
    ) -> tuple:
        """
        시선 추정 결과를 스무딩 (가중 평균)
        
        Args:
            gaze_history: 이전 예측 결과들의 히스토리 [(yaw, pitch, gaze_vector), ...]
            current_result: 현재 예측 결과 (yaw_deg, pitch_deg, gaze_vector, mode, weight)
            
        Returns:
            스무딩된 (yaw_deg, pitch_deg, gaze_vector, mode, weight)
        """
        if self.smoothing_window <= 0 or gaze_history is None:
            return current_result
        
        yaw_deg, pitch_deg, gaze_vector, mode, weight = current_result
        
        # 히스토리가 없거나 1개면 현재 결과 반환
        if len(gaze_history) == 0:
            gaze_history.append((yaw_deg, pitch_deg, gaze_vector.copy()))
            return current_result
        
        # 현재 결과를 히스토리에 추가
        gaze_history.append((yaw_deg, pitch_deg, gaze_vector.copy()))
        
        # 가중 평균 계산 (최근 것에 높은 가중치)
        history_list = list(gaze_history)
        n = len(history_list)
        
        # 선형 가중치: 최근 것일수록 높은 가중치
        weights = np.linspace(0.5, 1.0, n)  # 가장 오래된 것 0.5, 최신 것 1.0
        weights = weights / np.sum(weights)  # 정규화
        
        # 가중 평균 계산
        smoothed_yaw = sum(w * h[0] for w, h in zip(weights, history_list))
        smoothed_pitch = sum(w * h[1] for w, h in zip(weights, history_list))
        
        # 벡터도 가중 평균 (정규화 필요)
        smoothed_vector = np.zeros(3, dtype=float)
        for w, h in zip(weights, history_list):
            smoothed_vector += w * h[2]
        
        # 벡터 정규화
        vector_norm = np.linalg.norm(smoothed_vector)
        if vector_norm > 1e-6:
            smoothed_vector = smoothed_vector / vector_norm
        else:
            smoothed_vector = gaze_vector
        
        return (
            float(smoothed_yaw),
            float(smoothed_pitch),
            smoothed_vector,
            mode,
            weight,
        )

    def apply_vector_correction(
        self,
        yaw_deg: float,
        pitch_deg: float,
        base_point: Optional[tuple[float, float]],
        target_point: Optional[tuple[float, float]],
    ) -> tuple[float, float]:
        if (
            base_point is None
            or target_point is None
            or self.vector_weight <= 0.0
        ):
            return yaw_deg, pitch_deg

        dx = target_point[0] - base_point[0]
        dy = base_point[1] - target_point[1]
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return yaw_deg, pitch_deg

        corr_yaw, corr_pitch = self.vector_to_angles(dx, dy)
        w = np.clip(self.vector_weight, 0.0, 1.0)
        new_yaw = (1.0 - w) * yaw_deg + w * corr_yaw
        new_pitch = (1.0 - w) * pitch_deg + w * corr_pitch
        return new_yaw, new_pitch

    def apply_sixd_correction(
        self,
        yaw_deg: float,
        pitch_deg: float,
        geom_info: dict,
    ) -> tuple[float, float]:
        case = int(self.sixd_case)
        case = min(max(case, 1), 4)

        nose_point = geom_info.get("nose_point")
        neck_point = geom_info.get("neck_point")
        head_center = geom_info.get("head_center")

        if case == 1:
            return yaw_deg, pitch_deg

        if case == 2:
            return self.apply_vector_correction(
                yaw_deg,
                pitch_deg,
                neck_point,
                nose_point,
            )

        if case == 3:
            return self.apply_vector_correction(
                yaw_deg,
                pitch_deg,
                head_center,
                nose_point,
            )

        # case 4: neck->nose, then headcenter->nose
        yaw_deg, pitch_deg = self.apply_vector_correction(
            yaw_deg,
            pitch_deg,
            neck_point,
            nose_point,
        )
        yaw_deg, pitch_deg = self.apply_vector_correction(
            yaw_deg,
            pitch_deg,
            head_center,
            nose_point,
        )
        return yaw_deg, pitch_deg

    @staticmethod
    def _eye_patch_signal(
        face_crop,
        face_x: int,
        face_y: int,
        point_abs: tuple[int, int],
    ) -> float:
        if face_crop is None or face_crop.size == 0:
            return 0.0
        h, w = face_crop.shape[:2]
        if h == 0 or w == 0:
            return 0.0
        rel_x = int(point_abs[0] - face_x)
        rel_y = int(point_abs[1] - face_y)
        window = max(int(min(h, w) * 0.07), 6)
        x1 = max(rel_x - window, 0)
        y1 = max(rel_y - window, 0)
        x2 = min(rel_x + window, w)
        y2 = min(rel_y + window, h)
        patch = face_crop[y1:y2, x1:x2]
        if patch.size < 20:
            return 0.0
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        contrast = float(np.std(gray))
        edges = cv2.Canny(gray, 30, 90)
        edge_density = float(np.mean(edges > 0))
        score = 0.6 * (contrast / 30.0) + 0.4 * (edge_density * 5.0)
        return float(np.clip(score, 0.0, 1.5))

    def _state_from_landmark_points(
        self,
        face_crop,
        face_x: int,
        face_y: int,
        points: list[tuple[int, int]],
        nose_point: Optional[tuple[int, int]] = None,
    ) -> tuple[Literal["both", "single", "none"], dict]:
        info: dict = {}
        if face_crop is None or face_crop.size == 0:
            return "none", info

        if nose_point is not None:
            info["nose_point"] = nose_point

        left_score = 0.0
        right_score = 0.0
        if len(points) > 0:
            right_score = self._eye_patch_signal(face_crop, face_x, face_y, points[0])
        if len(points) > 1:
            left_score = self._eye_patch_signal(face_crop, face_x, face_y, points[1])

        threshold = 0.15
        visible_left = left_score > threshold
        visible_right = right_score > threshold

        if visible_left and visible_right:
            state = "both"
        elif visible_left or visible_right:
            state = "single"
        else:
            state = "none"

        return state, info

    @staticmethod
    def _region_visibility_score(region) -> float:
        if region is None or region.size == 0:
            return 0.0
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges > 0))
        brightness_variance = np.var(gray)
        contrast_score = min(contrast / 30.0, 1.0)
        edge_score = min(edge_density * 10.0, 1.0)
        variance_score = min(brightness_variance / 1000.0, 1.0)
        score = (
            0.4 * contrast_score
            + 0.4 * edge_score
            + 0.2 * variance_score
        )
        return float(np.clip(score, 0.0, 1.2))

    def eye_state_from_simple(self, face_crop, face_x: int, face_y: int):
        if face_crop is None or face_crop.size == 0:
            return "none", {}
        h, w = face_crop.shape[:2]
        if h == 0 or w == 0:
            return "none", {}
        top = face_crop[0 : int(h * 0.4), :]
        if top.size == 0:
            return "none", {}
        mid = w // 2
        left_region = top[:, :mid]
        right_region = top[:, mid:]
        left_score = self._region_visibility_score(left_region)
        right_score = self._region_visibility_score(right_region)
        threshold = self.simple_eye_threshold
        left_visible = left_score >= threshold
        right_visible = right_score >= threshold
        diff = abs(left_score - right_score)
        diff_threshold = 0.15

        if left_visible and right_visible:
            state = "both"
        elif diff >= diff_threshold:
            state = "single"
        else:
            state = "none"

        info = {
            "head_center": (
                face_x + w / 2.0,
                face_y + h / 2.0,
            ),
            "left_score": float(left_score),
            "right_score": float(right_score),
        }
        return state, info

    def eye_state_from_retinaface(
        self,
        face_crop,
        face_x: int,
        face_y: int,
        retinaface_info: Optional[dict] = None,
    ) -> tuple[Literal["both", "single", "none"], dict]:
        retinaface_landmarks = None
        retinaface_scores = None
        if retinaface_info is not None:
            retinaface_landmarks = retinaface_info.get("landmarks")
            retinaface_scores = retinaface_info.get("scores")

        if retinaface_landmarks is None or len(retinaface_landmarks) < 2:
            return "none", {}

        if retinaface_scores is None or len(retinaface_scores) < 2:
            retinaface_scores = []
            for idx in range(2):
                pt = retinaface_landmarks[idx]
                retinaface_scores.append(
                    self._eye_patch_signal(
                        face_crop,
                        face_x,
                        face_y,
                        pt,
                    )
                )

        eye_states = []
        confidence_threshold = 0.3
        for idx in range(2):
            if retinaface_scores is not None and len(retinaface_scores) > idx:
                if retinaface_scores[idx] < confidence_threshold:
                    eye_states.append(False)
                    continue
            eye_states.append(True)

        nose_point = None
        if len(retinaface_landmarks) > 2:
            nose_point = retinaface_landmarks[2]

        state, info = self._state_from_landmark_points(
            face_crop,
            face_x,
            face_y,
            retinaface_landmarks[:2],
            nose_point,
        )

        if eye_states:
            left_visible = eye_states[1] if len(eye_states) > 1 else False
            right_visible = eye_states[0]

            if state == "both":
                if not left_visible and not right_visible:
                    return "none", info
                if not left_visible or not right_visible:
                    return "single", info
            elif state == "single":
                combined = (left_visible or right_visible) and (
                    left_visible != right_visible
                )
                if left_visible and right_visible:
                    return "both", info
                if not combined:
                    return "none", info
            else:
                if left_visible and right_visible:
                    return "both", info
                if left_visible or right_visible:
                    return "single", info

        return state, info

    def eye_state_from_mediapipe(
        self,
        face_crop,
        face_x: int,
        face_y: int,
    ) -> tuple[Literal["both", "single", "none"], dict]:
        info = {}
        if (
            self.face_mesh is None
            or face_crop is None
            or face_crop.size == 0
        ):
            return "none", info

        h, w = face_crop.shape[:2]
        if h == 0 or w == 0:
            return "none", info

        try:
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return "none", info

            landmarks = results.multi_face_landmarks[0].landmark

            def eye_visible(idx_list):
                xs = [landmarks[i].x * w for i in idx_list]
                ys = [landmarks[i].y * h for i in idx_list]
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                area = width * height
                area_ratio = area / max(w * h, 1)
                return width > 4 and height > 2 and area_ratio > 0.0003

            left_visible = eye_visible(LEFT_EYE_IDX)
            right_visible = eye_visible(RIGHT_EYE_IDX)

            if left_visible and right_visible:
                state = "both"
            elif left_visible or right_visible:
                state = "single"
            else:
                state = "none"

            nose_idx = 1
            if 0 <= nose_idx < len(landmarks):
                nose = landmarks[nose_idx]
                nose_point = (
                    face_x + int(nose.x * w),
                    face_y + int(nose.y * h),
                )
                info["nose_point"] = nose_point

            info["head_center"] = (
                face_x + w / 2.0,
                face_y + h / 2.0,
            )
            return state, info
        except Exception:
            return "none", info

    def eye_state_from_yoloface(
        self,
        face_crop,
        face_x: int,
        face_y: int,
    ) -> tuple[Literal["both", "single", "none"], dict]:
        if self.yolo_face_model is None:
            return "none", {}
        if face_crop is None or face_crop.size == 0:
            return "none", {}
        try:
            results = self.yolo_face_model(face_crop, verbose=False)
            best_landmarks = None
            best_conf = 0.0
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    conf = float(box.conf[0].cpu().numpy())
                    if conf < 0.3 or conf < best_conf:
                        continue
                    landmarks = getattr(box, "landmarks", None)
                    if landmarks is None and hasattr(r, "keypoints"):
                        kp = r.keypoints
                        if kp is not None and len(kp.data) > 0:
                            landmarks = kp.data[0]
                    if landmarks is None:
                        continue
                    lm = landmarks.cpu().numpy()
                    best_landmarks = lm
                    best_conf = conf
            if best_landmarks is None:
                return "none", {}
            points = []
            for idx in range(min(2, best_landmarks.shape[0])):
                px = face_x + int(best_landmarks[idx][0])
                py = face_y + int(best_landmarks[idx][1])
                points.append((px, py))
            nose_point = None
            if best_landmarks.shape[0] >= 3:
                nose_point = (
                    face_x + int(best_landmarks[2][0]),
                    face_y + int(best_landmarks[2][1]),
                )
            return self._state_from_landmark_points(
                face_crop,
                face_x,
                face_y,
                points,
                nose_point,
            )
        except Exception:
            return "none", {}

    def determine_eye_state(
        self,
        face_crop,
        face_x: int,
        face_y: int,
        retinaface_data: Optional[dict] = None,
    ) -> tuple[Literal["both", "single", "none"], dict]:
        mode_order: list[str]
        if self.eye_mode == "auto":
            mode_order = []
            if retinaface_data is not None and retinaface_data.get("landmarks"):
                mode_order.append("retinaface")
            if self.yolo_face_model is not None:
                mode_order.append("yoloface")
            if self.face_mesh is not None:
                mode_order.append("mediapipe")
            mode_order.append("quality")
        else:
            mode_order = [self.eye_mode]

        for mode in mode_order:
            if mode == "retinaface":
                state, info = self.eye_state_from_retinaface(
                    face_crop,
                    face_x,
                    face_y,
                    retinaface_data,
                )
            elif mode == "yoloface":
                state, info = self.eye_state_from_yoloface(
                    face_crop,
                    face_x,
                    face_y,
                )
            elif mode == "mediapipe":
                state, info = self.eye_state_from_mediapipe(
                    face_crop,
                    face_x,
                    face_y,
                )
            elif mode == "quality":
                state, info = self.eye_state_from_simple(
                    face_crop,
                    face_x,
                    face_y,
                )
            else:
                continue

            if state != "none":
                return state, info

        return "none", {}

    def eye_state_from_yaw(
        self,
        yaw_deg: float,
        pitch_deg: float,
    ) -> tuple[Literal["both", "single", "none"], dict]:
        """
        6DRepNet의 Yaw/Pitch 각도로 양안/단안 판별
        
        판별 로직:
        - 양안 확보: |Yaw| < yaw_threshold_both AND |Pitch| < pitch_threshold_both
        - 단안/측면: |Yaw| < yaw_threshold_single AND |Pitch| < pitch_threshold_single 
                     AND (|Yaw| >= yaw_threshold_both OR |Pitch| >= pitch_threshold_both)
        - Skip: 그 외 (뒤통수 등)
        """
        abs_yaw = abs(yaw_deg)
        abs_pitch = abs(pitch_deg)
        
        # 양안 확보 구간
        if abs_yaw < self.yaw_threshold_both and abs_pitch < self.pitch_threshold_both:
            return "both", {"yaw": yaw_deg, "pitch": pitch_deg}
        
        # 단안/측면 구간
        if (abs_yaw < self.yaw_threshold_single 
            and abs_pitch < self.pitch_threshold_single
            and (abs_yaw >= self.yaw_threshold_both or abs_pitch >= self.pitch_threshold_both)):
            return "single", {"yaw": yaw_deg, "pitch": pitch_deg}
        
        # 그 외 (뒤통수 등) - Skip
        return "none", {"yaw": yaw_deg, "pitch": pitch_deg}

    def extract_pose_keypoints(
        self,
        person_roi,
        offset_x: int,
        offset_y: int,
    ) -> dict:
        if (
            self.pose_estimator is None
            or person_roi is None
            or person_roi.size == 0
        ):
            return {}

        h, w = person_roi.shape[:2]
        if h == 0 or w == 0:
            return {}

        try:
            rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            results = self.pose_estimator.process(rgb)
            if not results.pose_landmarks:
                return {}

            lm = results.pose_landmarks.landmark

            def to_point(landmark):
                return (
                    offset_x + int(landmark.x * w),
                    offset_y + int(landmark.y * h),
                )

            info = {}
            nose = lm[mp.solutions.pose.PoseLandmark.NOSE]
            if nose.visibility > 0.3:
                info["pose_nose"] = to_point(nose)

            left_shoulder = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = lm[
                mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
            ]
            if (
                left_shoulder.visibility > 0.5
                and right_shoulder.visibility > 0.5
            ):
                neck_point = (
                    offset_x
                    + int((left_shoulder.x + right_shoulder.x) * 0.5 * w),
                    offset_y
                    + int((left_shoulder.y + right_shoulder.y) * 0.5 * h),
                )
            elif left_shoulder.visibility > 0.5:
                neck_point = to_point(left_shoulder)
            elif right_shoulder.visibility > 0.5:
                neck_point = to_point(right_shoulder)
            else:
                neck_point = None

            info["neck_point"] = neck_point
            return info
        except Exception:
            return {}
    def compute_eye_visibility_quality(self, face_image) -> float:
        """
        좌/우 눈 패치를 분리해 최소 가시성 점수 반환 (quality_score: 0~1)
        """
        try:
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image

            h, w = gray.shape
            if h == 0 or w == 0:
                return 0.0

            eye_region = gray[0 : int(h * 0.4), :]
            if eye_region.size == 0:
                return 0.0

            mid = max(1, eye_region.shape[1] // 2)
            left_region = eye_region[:, :mid]
            right_region = eye_region[:, mid:]

            left_score = self._region_visibility_score(left_region)
            right_score = self._region_visibility_score(right_region)

            quality_score = min(left_score, right_score)
            return float(np.clip(quality_score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def adaptive_fusion(gaze360_result, headbody_result, quality_score):
        """
        적응적 융합: 눈 가시성에 따라 두 방법의 가중치 조정
        """
        g360_valid = (
            gaze360_result[0] is not None and gaze360_result[2] is not None
        )
        hb_valid = (
            headbody_result[0] is not None and headbody_result[2] is not None
        )

        if not g360_valid and not hb_valid:
            return None, None, None, "invalid", 0.0

        if not g360_valid:
            return headbody_result

        if not hb_valid:
            return gaze360_result

        if quality_score > 0.5:
            gaze360_weight = 0.8
            headbody_weight = 0.2
        else:
            gaze360_weight = 0.2
            headbody_weight = 0.8

        yaw_deg = (
            gaze360_weight * gaze360_result[0]
            + headbody_weight * headbody_result[0]
        )
        pitch_deg = (
            gaze360_weight * gaze360_result[1]
            + headbody_weight * headbody_result[1]
        )

        gaze_vector = (
            gaze360_weight * gaze360_result[2]
            + headbody_weight * headbody_result[2]
        )
        gaze_norm = np.linalg.norm(gaze_vector)
        if gaze_norm > 1e-6:
            gaze_vector = gaze_vector / gaze_norm
        else:
            gaze_vector = gaze360_result[2]

        final_weight = (
            quality_score * gaze360_result[4]
            + (1.0 - quality_score) * headbody_result[4]
        )
        final_mode = "fused"

        return float(yaw_deg), float(pitch_deg), gaze_vector, final_mode, float(
            final_weight
        )

    def compute_gaze_endpoint(
        self,
        face_center_x: int,
        face_center_y: int,
        gaze_x: float,
        gaze_y: float,
    ):
        """
        Gaze360 gaze_vector 기준 시선 끝점 계산
        """
        dx = -self.gaze_distance_k * gaze_x
        dy = -self.gaze_distance_k * gaze_y

        x_end = face_center_x + dx
        y_end = face_center_y + dy

        return int(x_end), int(y_end)

    # -------------------------
    # ROI 관련 로직
    # -------------------------
    def check_point_in_roi(self, x: int, y: int):
        """점이 어떤 RoI 안에 있는지 확인 (rect / polygon 모두 지원)"""
        for roi in self.rois:
            roi_type = roi.get("type", "rect")
            roi_id = roi.get("roi_id", 0)
            data = roi.get("data")

            if roi_type == "rect":
                x1, y1, x2, y2 = data
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return roi_id
            elif roi_type == "poly":
                if self.point_in_polygon(x, y, data):
                    return roi_id
        return None

    @staticmethod
    def point_in_polygon(x, y, polygon):
        """
        Ray casting 알고리즘으로 점이 다각형 안에 있는지 확인
        polygon: [(x1, y1), (x2, y2), ...]
        """
        n = len(polygon)
        if n < 3:
            return False

        inside = False
        p1x, p1y = polygon[0]

        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / float(
                                (p2y - p1y)
                            ) + p1x
                        else:
                            xinters = p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    # -------------------------
    # Ray-ROI intersection helpers
    # -------------------------
    @staticmethod
    def _segment_intersect(p1, p2, p3, p4):
        """선분 p1-p2와 p3-p4의 교차 여부 및 교차점 반환"""
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        a, b, c, d = p1, p2, p3, p4
        intersect = (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d))
        if not intersect:
            return False, None

        xdiff = (a[0] - b[0], c[0] - d[0])
        ydiff = (a[1] - b[1], c[1] - d[1])

        def det(p, q):
            return p[0] * q[1] - p[1] * q[0]

        div = det(xdiff, ydiff)
        if abs(div) < 1e-9:
            return False, None
        d0 = (det(a, b), det(c, d))
        x = det(d0, xdiff) / div
        y = det(d0, ydiff) / div
        return True, (x, y)

    def _shrink_roi(self, roi):
        """ROI를 안쪽으로 ray_shrink_px만큼 축소한 새 ROI 반환"""
        shrink = self.ray_shrink_px
        if shrink <= 0:
            return roi
        roi_type = roi.get("type", "rect")
        data = roi.get("data")
        if roi_type == "rect":
            x1, y1, x2, y2 = data
            return {
                "type": "rect",
                "roi_id": roi.get("roi_id", 0),
                "data": [x1 + shrink, y1 + shrink, x2 - shrink, y2 - shrink],
            }
        pts = data
        # 폴리곤은 단순히 centroid를 향해 축소
        cx = np.mean([p[0] for p in pts])
        cy = np.mean([p[1] for p in pts])
        shrunk = []
        for x, y in pts:
            vx, vy = x - cx, y - cy
            norm = math.hypot(vx, vy)
            if norm < 1e-6:
                shrunk.append((x, y))
            else:
                scale = max(0.0, (norm - shrink) / norm)
                shrunk.append((cx + vx * scale, cy + vy * scale))
        return {
            "type": "poly",
            "roi_id": roi.get("roi_id", 0),
            "data": shrunk,
        }

    def _ray_hit_roi(self, face_center: np.ndarray, dir_norm: np.ndarray, roi, max_len: float):
        """레이와 ROI 교차 여부 및 레이 상 거리 t 반환 (없으면 None)"""
        p0 = face_center
        p1 = face_center + dir_norm * max_len

        roi = self._shrink_roi(roi)
        roi_type = roi.get("type", "rect")
        data = roi.get("data")

        if roi_type == "rect":
            x1, y1, x2, y2 = data
            rect_pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            edges = list(zip(rect_pts, rect_pts[1:] + rect_pts[:1]))
        else:
            pts = data
            edges = list(zip(pts, pts[1:] + pts[:1]))

        candidates = []
        for e0, e1 in edges:
            hit, pt = self._segment_intersect((p0[0], p0[1]), (p1[0], p1[1]), e0, e1)
            if hit and pt is not None:
                v = np.array(pt) - p0
                t = np.dot(v, dir_norm)
                if t >= 0:
                    candidates.append(t)

        if len(candidates) == 0:
            return None
        return min(candidates)

    def interactive_set_k(self, video_path: str):
        """첫 프레임에서 클릭한 점까지의 거리를 k로 설정 (화면 중심 기준)"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"비디오를 열 수 없습니다: {video_path}")
            return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("첫 프레임을 읽을 수 없습니다.")
            return

        clicked = {"pt": None}
        window = "Set gaze_distance_k - 클릭 후 Enter, ESC로 취소"

        def cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked["pt"] = (x, y)

        cv2.namedWindow(window)
        cv2.setMouseCallback(window, cb)
        h, w = frame.shape[:2]
        center = (w * 0.5, h * 0.5)

        while True:
            disp = frame.copy()
            cv2.circle(disp, (int(center[0]), int(center[1])), 5, (0, 255, 255), -1)
            if clicked["pt"] is not None:
                cv2.circle(disp, clicked["pt"], 6, (0, 0, 255), -1)
                cv2.line(
                    disp,
                    (int(center[0]), int(center[1])),
                    (int(clicked["pt"][0]), int(clicked["pt"][1])),
                    (0, 255, 0),
                    2,
                )
            cv2.putText(
                disp,
                "Click target depth point, Enter=apply, ESC=skip",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10):  # Enter
                break
            if key == 27:  # ESC
                clicked["pt"] = None
                break
        cv2.destroyWindow(window)

        if clicked["pt"] is None:
            print("k 설정을 건너뜁니다.")
            return

        # 클릭한 점을 k점 앵커로 저장
        self.k_anchor_point = (float(clicked["pt"][0]), float(clicked["pt"][1]))
        
        dx = clicked["pt"][0] - center[0]
        dy = clicked["pt"][1] - center[1]
        k = float(math.hypot(dx, dy))
        # 너무 짧게 잡히면 화면 크기 기반으로 보정
        h, w = frame.shape[:2]
        min_len = max(h, w) * 0.4
        if k < 20.0:
            print(f"선택한 길이 {k:.1f}px 이 너무 짧습니다. {min_len:.1f}px로 보정합니다.")
            k = min_len

        self.gaze_distance_k = int(k)
        if self.ray_max_len <= 0 or self.ray_max_len < k:
            self.ray_max_len = k
        print(f"k점 앵커: ({self.k_anchor_point[0]:.1f}, {self.k_anchor_point[1]:.1f})")
        print(f"gaze_distance_k가 {k:.1f} (px)로 설정되었습니다. ray_max_len={self.ray_max_len:.1f}")

    def interactive_set_face_center(self, video_path: str):
        """첫 프레임에서 얼굴 중심을 수동으로 지정"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"비디오를 열 수 없습니다: {video_path}")
            return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("첫 프레임을 읽을 수 없습니다.")
            return

        clicked = {"pt": None}
        window = "Set face center - 클릭 후 Enter, ESC로 취소"

        def cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked["pt"] = (x, y)

        cv2.namedWindow(window)
        cv2.setMouseCallback(window, cb)

        while True:
            disp = frame.copy()
            if clicked["pt"] is not None:
                cv2.circle(disp, clicked["pt"], 6, (0, 255, 255), -1)
            cv2.putText(
                disp,
                "Click face center, Enter=apply, ESC=skip",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10):  # Enter
                break
            if key == 27:  # ESC
                clicked["pt"] = None
                break
        cv2.destroyWindow(window)

        if clicked["pt"] is None:
            print("수동 얼굴 중심 설정을 건너뜁니다.")
            return

        self.manual_face_center = (float(clicked["pt"][0]), float(clicked["pt"][1]))
        print(f"수동 얼굴 중심이 설정되었습니다: ({self.manual_face_center[0]:.1f}, {self.manual_face_center[1]:.1f})")

    @staticmethod
    def get_roi_center(roi):
        """ROI의 중심점 계산"""
        roi_type = roi.get("type", "rect")
        data = roi.get("data")

        if roi_type == "rect":
            x1, y1, x2, y2 = data
            return (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        elif roi_type == "poly":
            xs = [p[0] for p in data]
            ys = [p[1] for p in data]
            return (float(np.mean(xs)), float(np.mean(ys)))
        return None

    @staticmethod
    def is_point_in_roi(x: float, y: float, roi) -> bool:
        """점(x, y)이 ROI 내부인지 여부"""
        roi_type = roi.get("type", "rect")
        data = roi.get("data")

        if roi_type == "rect":
            x1, y1, x2, y2 = data
            return x1 <= x <= x2 and y1 <= y <= y2
        elif roi_type == "poly":
            return RoIGazeTrackerGaze360.point_in_polygon(x, y, data)
        return False

    @staticmethod
    def parse_vec3(text: Optional[str]) -> Optional[np.ndarray]:
        """콤마/공백 구분 3개 값 문자열을 np.ndarray(3,)로 변환"""
        if text is None:
            return None
        parts = [p for p in re.split(r"[,\s]+", text.strip()) if p]
        if len(parts) != 3:
            return None
        try:
            return np.array([float(p) for p in parts], dtype=float)
        except Exception:
            return None

    def intersect_gaze_with_plane(
        self,
        face_center_x: float,
        face_center_y: float,
        gaze_vector,
    ) -> Optional[np.ndarray]:
        """
        시선 레이(face_center, gaze_vector)와 ROI 평면(origin/normal)의 교차점 계산.
        평면 정보가 없거나 교차하지 않으면 None.
        """
        if self.roi_plane_origin is None or self.roi_plane_normal is None:
            return None

        o = np.array([float(face_center_x), float(face_center_y), 0.0], dtype=float)
        d = np.array(gaze_vector, dtype=float)
        n = self.roi_plane_normal

        # 평면 노멀을 시선과 반대 방향이면 뒤집어 정방향 교차가 되도록 함
        denom = float(np.dot(d, n))
        if denom > 0:
            n = -n
            denom = float(np.dot(d, n))

        if abs(denom) < 1e-6:
            return None  # 레이와 평면이 평행

        t = float(np.dot(self.roi_plane_origin - o, n) / denom)
        if t <= 0:
            return None  # 평면이 시선 반대 방향(또는 교차점이 뒤)

        return o + t * d

    def estimate_roi_plane_depth(self, frame: np.ndarray) -> Optional[float]:
        """
        MiDaS를 사용하여 ROI 영역의 평균 depth를 추정하고 roi_plane_z를 업데이트.
        
        Returns:
            추정된 depth 값 (픽셀 단위, None이면 실패)
        """
        if not self.use_depth or self.depth_model is None or self.depth_transform is None:
            return None

        if len(self.rois) == 0:
            return None

        try:
            # 프레임을 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self.depth_transform(frame_rgb).to(self.device)

            with torch.no_grad():
                prediction = self.depth_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()

            # 모든 ROI 영역의 depth 값 수집
            depth_values = []
            h, w = depth_map.shape

            for roi in self.rois:
                roi_type = roi.get("type", "rect")
                data = roi.get("data")

                if roi_type == "rect":
                    x1, y1, x2, y2 = data
                    x1, x2 = max(0, int(x1)), min(w, int(x2))
                    y1, y2 = max(0, int(y1)), min(h, int(y2))
                    if x2 > x1 and y2 > y1:
                        roi_depth = depth_map[y1:y2, x1:x2]
                        depth_values.extend(roi_depth.flatten().tolist())
                elif roi_type == "poly":
                    # 폴리곤 마스크 생성
                    mask = np.zeros((h, w), dtype=np.uint8)
                    pts = np.array(data, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 255)
                    roi_depth = depth_map[mask > 0]
                    depth_values.extend(roi_depth.flatten().tolist())

            if len(depth_values) == 0:
                return None

            # 중앙값 사용 (이상치에 강건함)
            median_depth = float(np.median(depth_values))
            
            # roi_plane_z 업데이트 (Gaze360 좌표계: -z가 전방)
            # MiDaS는 일반적으로 양수 depth를 반환하므로, -z로 변환
            estimated_z = -abs(median_depth)
            
            # 평면 정보 업데이트
            self.roi_plane_origin = np.array([0.0, 0.0, estimated_z], dtype=float)
            self.roi_plane_normal = np.array([0.0, 0.0, -1.0], dtype=float)
            
            return abs(median_depth)
        except Exception as e:
            print(f"Warning: Depth 추정 오류 ({e})")
            return None

    def select_roi_by_min_distance(
        self, face_center_x: int, face_center_y: int, gaze_vector
    ):
        """
        시선 반직선에서 각 ROI 중심점까지의 최단 거리를 계산하여
        가장 가까운 ROI로 매핑
        
        Args:
            face_center_x: 얼굴 중심 x 좌표
            face_center_y: 얼굴 중심 y 좌표
            gaze_vector: 시선 방향 벡터 [x, y, z]
            
        Returns:
            best_roi_id: 가장 가까운 ROI ID (없으면 None)
            min_distance: 최단 거리
        """
        if len(self.rois) == 0:
            return None, float('inf')
        # 1) ROI 평면이 정의된 경우: 레이-평면 교차점 기반 매핑
        intersection_point = self.intersect_gaze_with_plane(
            face_center_x, face_center_y, gaze_vector
        )
        if intersection_point is not None:
            ix, iy, _ = intersection_point

            # 1-1) 교차점이 ROI 내부에 있으면 바로 반환
            for roi in self.rois:
                if self.is_point_in_roi(ix, iy, roi):
                    return roi.get("roi_id", 0), 0.0

            # 1-2) 내부가 아니면 교차점-ROI 중심 거리 최소
            best_roi_id = None
            min_distance = float("inf")
            for roi in self.rois:
                roi_id = roi.get("roi_id", 0)
                roi_center = self.get_roi_center(roi)
                if roi_center is None:
                    continue
                roi_center_point = np.array(
                    [float(roi_center[0]), float(roi_center[1])]
                )
                distance = np.linalg.norm(roi_center_point - np.array([ix, iy]))
                if distance < min_distance:
                    min_distance = distance
                    best_roi_id = roi_id
            return best_roi_id, min_distance

        # 2) 평면이 없으면 기존 2D 시선 반직선 기반 매핑
        gaze_2d = np.array([gaze_vector[0], gaze_vector[1]], dtype=float)
        gaze_2d_norm = np.linalg.norm(gaze_2d)

        if gaze_2d_norm < 1e-6:
            return None, float('inf')

        gaze_2d_normalized = gaze_2d / gaze_2d_norm

        # 얼굴 중심점
        face_center = np.array([float(face_center_x), float(face_center_y)])

        best_roi_id = None
        min_distance = float('inf')

        for roi in self.rois:
            roi_id = roi.get("roi_id", 0)
            roi_center = self.get_roi_center(roi)

            if roi_center is None:
                continue

            # ROI 중심점
            roi_center_point = np.array([float(roi_center[0]), float(roi_center[1])])

            # 얼굴 중심에서 ROI 중심으로의 벡터
            v_to_roi = roi_center_point - face_center

            # 시선 방향으로의 투영 길이 계산
            # t = dot(v_to_roi, gaze_2d_normalized)
            t = np.dot(v_to_roi, gaze_2d_normalized)

            # t < 0이면 시선 반대 방향에 있는 ROI (시선 반직선 상에 없음)
            # 이 경우 유클리드 거리를 사용하거나 큰 거리로 처리
            if t < 0:
                # 시선 반대 방향이면 유클리드 거리 사용 (또는 큰 값)
                distance = np.linalg.norm(v_to_roi)
            else:
                # 시선 반직선 위의 가장 가까운 점 계산
                closest_point_on_ray = face_center + t * gaze_2d_normalized

                # ROI 중심점에서 시선 반직선까지의 최단 거리
                distance = np.linalg.norm(roi_center_point - closest_point_on_ray)

            if distance < min_distance:
                min_distance = distance
                best_roi_id = roi_id

        # 최단 거리가 너무 크면 매핑하지 않음 (선택적)
        # 화면 대각선의 약 2배를 임계값으로 설정 (1920x1080 기준 약 4400픽셀)
        max_distance_threshold = 5000.0  # 픽셀 단위
        if min_distance > max_distance_threshold:
            return None, min_distance

        # best_roi_id가 None이면 가장 가까운 ROI를 선택
        if best_roi_id is None:
            # 모든 ROI를 다시 순회하며 최소 거리 찾기
            for roi in self.rois:
                roi_id = roi.get("roi_id", 0)
                roi_center = self.get_roi_center(roi)
                if roi_center is None:
                    continue
                roi_center_point = np.array([float(roi_center[0]), float(roi_center[1])])
                v_to_roi = roi_center_point - face_center
                distance = np.linalg.norm(v_to_roi)
                if distance < min_distance:
                    min_distance = distance
                    best_roi_id = roi_id

        return best_roi_id, min_distance

    def select_roi_with_ray(
        self, face_center_x: int, face_center_y: int, gaze_vector
    ):
        """
        방향 기반 각도 매칭 (k점 기준 회전):
        - k점(앵커)을 중심으로 시선 방향(yaw/pitch)으로 회전하는 화살표가 ROI 선택
        - k점에서 각 ROI 중심까지의 벡터와 시선 방향 벡터의 각도 차이 계산
        - 각도 차이가 가장 작은 ROI 선택 (거리도 함께 고려)
        - k점이 필수: k점이 없으면 거리 기반 fallback
        """
        if len(self.rois) == 0:
            return None, float("inf")

        # k점이 필수: 없으면 거리 기반 fallback
        if self.k_anchor_point is None:
            return self.select_roi_by_min_distance(face_center_x, face_center_y, gaze_vector)

        gaze_2d = np.array([gaze_vector[0], gaze_vector[1]], dtype=float)
        norm = np.linalg.norm(gaze_2d)
        if norm < 1e-6:
            return None, float("inf")
        gaze_2d_normalized = gaze_2d / norm

        # k점을 기준점으로 사용 (필수)
        anchor_point = np.array([float(self.k_anchor_point[0]), float(self.k_anchor_point[1])])

        best_roi_id = None
        best_score = float("inf")  # 각도 차이 + 거리 가중치 (작을수록 좋음)
        
        # 디버깅: 모든 ROI의 각도와 거리 정보 저장
        roi_candidates = []  # (roi_id, angle_diff_deg, distance, score)
        self.debug_roi_info = {}  # 디버깅 정보 초기화

        for roi in self.rois:
            roi_id = roi.get("roi_id", 0)
            roi_center = self.get_roi_center(roi)
            if roi_center is None:
                continue

            roi_center_point = np.array([float(roi_center[0]), float(roi_center[1])])
            v_to_roi = roi_center_point - anchor_point
            v_norm = np.linalg.norm(v_to_roi)
            if v_norm < 1e-6:
                continue

            v_to_roi_normalized = v_to_roi / v_norm

            # 각도 차이 계산 (cosine similarity 사용)
            cos_angle = float(np.dot(gaze_2d_normalized, v_to_roi_normalized))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_diff = math.acos(cos_angle)  # 라디안 (0 ~ π)
            angle_diff_deg = math.degrees(angle_diff)  # 도 단위

            # 각도 차이 임계값: 설정된 임계값 이내만 고려
            if angle_diff_deg > self.roi_angle_threshold_deg:
                continue

            # 스코어: 각도 차이 우선, 거리는 보조 (각도가 훨씬 중요)
            # 각도 차이를 주로 고려하고, 거리는 동일 각도일 때만 고려
            angle_weight = 1.0  # 각도 가중치 (주요 기준)
            dist_weight = 0.05  # 거리 가중치 (보조, 매우 작게)
            
            # 거리 정규화: ROI들 간의 최대 거리를 기준으로 정규화
            # 모든 ROI 중심점 간의 최대 거리를 계산
            max_roi_dist = 0.0
            for other_roi in self.rois:
                other_center = self.get_roi_center(other_roi)
                if other_center is not None:
                    other_point = np.array([float(other_center[0]), float(other_center[1])])
                    dist_to_other = np.linalg.norm(roi_center_point - other_point)
                    max_roi_dist = max(max_roi_dist, dist_to_other)
            
            # 최대 거리가 0이면 기본값 사용
            if max_roi_dist < 1e-6:
                max_roi_dist = 2000.0  # 기본값 (픽셀)
            
            # 정규화된 거리 (0~1 범위)
            normalized_dist = min(v_norm / max_roi_dist, 1.0)
            
            # 스코어 계산: 각도 차이(도)를 주로, 거리는 보조로만
            # 각도 차이가 1도 차이나면 거리 20배 차이와 동일하게 취급
            score = angle_weight * angle_diff_deg + dist_weight * (normalized_dist * 20.0)
            
            # 디버깅: 후보 저장 및 정보 저장
            roi_candidates.append((roi_id, angle_diff_deg, v_norm, score))
            self.debug_roi_info[roi_id] = {
                "angle_diff_deg": angle_diff_deg,
                "distance": v_norm,
                "score": score,
                "center": (int(roi_center_point[0]), int(roi_center_point[1]))
            }

            if score < best_score:
                best_score = score
                best_roi_id = roi_id

        # 디버깅: 상위 3개 후보 출력
        if self.debug_mode and len(roi_candidates) > 0:
            roi_candidates.sort(key=lambda x: x[3])  # score로 정렬
            top_candidates = roi_candidates[:min(3, len(roi_candidates))]
            print(f"[DEBUG] Top ROI candidates:")
            for r in top_candidates:
                print(f"  ROI {r[0]}: angle={r[1]:.1f}°, dist={r[2]:.1f}px, score={r[3]:.2f}")
            print(f"[DEBUG] Selected ROI: {best_roi_id} (score: {best_score:.2f})")

        if best_roi_id is None:
            # 각도 기반으로 선택할 수 없으면 거리 기반 fallback
            return self.select_roi_by_min_distance(face_center_x, face_center_y, gaze_vector)

        return best_roi_id, best_score

    def update_gaze_journey(
        self,
        person_id: int,
        current_roi,
        weight: float = 1.0,
    ):
        """
        사람의 gaze_journey 업데이트 (가중치 기반)
        """
        person_data = self.person_gaze_data[person_id]

        if current_roi == person_data["temp_roi"]:
            person_data["temp_frame_count"] += 1
            person_data["temp_weighted_count"] += weight

            if (
                person_data["temp_weighted_count"]
                >= self.min_frames_threshold
            ):
                if current_roi != person_data["current_roi"]:
                    if (
                        person_data["current_roi"] is not None
                        and person_data["weighted_count"] > 0
                    ):
                        person_data["gaze_journey"].append(
                            (
                                person_data["current_roi"],
                                person_data["weighted_count"],
                            )
                        )
                    person_data["current_roi"] = current_roi
                    person_data["frame_count"] = person_data[
                        "temp_frame_count"
                    ]
                    person_data["weighted_count"] = person_data[
                        "temp_weighted_count"
                    ]
                else:
                    person_data["frame_count"] += 1
                    person_data["weighted_count"] += weight
        else:
            if (
                person_data["temp_weighted_count"]
                >= self.min_frames_threshold
            ):
                if (
                    person_data["current_roi"] is not None
                    and person_data["weighted_count"] > 0
                ):
                    person_data["gaze_journey"].append(
                        (
                            person_data["current_roi"],
                            person_data["weighted_count"],
                        )
                    )
                person_data["current_roi"] = None
                person_data["frame_count"] = 0
                person_data["weighted_count"] = 0.0

            person_data["temp_roi"] = current_roi
            person_data["temp_frame_count"] = 1
            person_data["temp_weighted_count"] = weight
            person_data["last_gaze"] = None

    def finalize_gaze_journey(self):
        """모든 사람의 gaze_journey 마무리"""
        for _, person_data in self.person_gaze_data.items():
            if (
                person_data["current_roi"] is not None
                and person_data["weighted_count"] > 0
            ):
                person_data["gaze_journey"].append(
                    (
                        person_data["current_roi"],
                        person_data["weighted_count"],
                    )
                )

    # -------------------------
    # 메인 드로잉 루프
    # -------------------------
    def draw_boxes_with_gaze(
        self,
        img,
        bbox,
        names,
        object_id,
        identities=None,
        offset=(0, 0),
        frame_count: int = 0,
    ):
        """바운딩 박스와 시선 정보를 그리기"""
        global data_deque, tracked_objects

        height, width, _ = img.shape
        active_rois = set()
        person_roi_mapping = {}  # {roi_id: [person_id1, person_id2, ...], ...}

        # 사라진 트랙 제거
        if identities is not None:
            for key in list(data_deque.keys()):
                if key not in identities:
                    data_deque.pop(key, None)

        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            center = (int((x2 + x1) / 2), int((y2 + y1) / 2))
            track_id = int(identities[i]) if identities is not None else 0

            if track_id not in data_deque:
                data_deque[track_id] = deque(maxlen=64)

            color = self.compute_color_for_labels(object_id[i])
            obj_name = names[object_id[i]]

            label = f"ID:{track_id} {obj_name}"

            if object_id[i] == 0:  # person
                person_roi = img[y1:y2, x1:x2]
                if person_roi.size > 0:
                    face_x, face_y = x1, y1
                    face_w, face_h = x2 - x1, y2 - y1
                    face_detected = False
                    retinaface_landmarks_abs: Optional[list[tuple[int, int]]] = None
                    retinaface_scores: Optional[list[float]] = None

                    # RetinaFace로 얼굴 검출
                    if self.face_detector is not None:
                        try:
                            faces = self.face_detector(person_roi)
                            if faces is not None and len(faces) > 0:
                                best_face = max(
                                    faces, key=lambda x: x[2]
                                )
                                score = best_face[2]
                                if score > 0.5:
                                    box_face, landmarks, scores = best_face
                                    fx_min, fy_min, fx_max, fy_max = [
                                        int(coord) for coord in box_face
                                    ]
                                    face_x = x1 + fx_min
                                    face_y = y1 + fy_min
                                    face_w = max(fx_max - fx_min, 1)
                                    face_h = max(fy_max - fy_min, 1)
                                    face_detected = True
                                    try:
                                        retinaface_landmarks_abs = []
                                        retinaface_scores = []
                                        for pt in landmarks:
                                            retinaface_landmarks_abs.append(
                                                (
                                                    x1 + int(pt[0]),
                                                    y1 + int(pt[1]),
                                                )
                                            )
                                        if scores is not None:
                                            for s in scores:
                                                retinaface_scores.append(float(s))
                                        else:
                                            retinaface_scores = None
                                    except Exception:
                                        retinaface_landmarks_abs = None
                                        retinaface_scores = None
                        except Exception:
                            pass

                    # 얼굴 탐지 실패 시 상단 40%를 얼굴로 가정
                    if not face_detected:
                        est_h = int((y2 - y1) * 0.4)
                        face_x = x1
                        face_y = y1
                        face_w = max(x2 - x1, 1)
                        face_h = max(est_h, 50)

                    face_crop = img[face_y : face_y + face_h, face_x : face_x + face_w]

                    if (
                        face_crop.size > 0
                        and face_crop.shape[0] > 0
                        and face_crop.shape[1] > 0
                        and track_id in tracked_objects
                    ):
                        face_center_x = face_x + face_w // 2
                        face_center_y = face_y + face_h // 2
                        # 수동 얼굴 중심이 설정된 경우 덮어씀
                        if self.manual_face_center is not None:
                            face_center_x = int(self.manual_face_center[0])
                            face_center_y = int(self.manual_face_center[1])

                        # 눈 상태 판별 (yaw_based 모드일 경우 6DRepNet 먼저 실행)
                        geom_info = {
                            "head_center": (
                                float(face_center_x),
                                float(face_center_y),
                            )
                        }
                        
                        # yaw_based 모드에서 사용할 6DRepNet 결과 저장
                        yaw_based_sixd_result = None
                        
                        if self.eye_mode == "yaw_based":
                            # yaw_based 모드: 6DRepNet으로 먼저 yaw/pitch 얻고 판별
                            if self.pose_model is None:
                                eye_state = "none"
                                mesh_info = {}
                            else:
                                sixd_yaw, sixd_pitch, sixd_gaze_vector, sixd_mode, sixd_weight = (
                                    self.estimate_head_pose(face_crop)
                                )
                                if sixd_yaw is not None and sixd_pitch is not None:
                                    eye_state, mesh_info = self.eye_state_from_yaw(
                                        sixd_yaw, sixd_pitch
                                    )
                                    # 나중에 재사용하기 위해 저장
                                    yaw_based_sixd_result = (
                                        sixd_yaw, sixd_pitch, sixd_gaze_vector, sixd_mode, sixd_weight
                                    )
                                else:
                                    eye_state = "none"
                                    mesh_info = {}
                        else:
                            # 기존 모드들: Mediapipe/RetinaFace/YOLO face/Quality 기반
                            eye_state, mesh_info = self.determine_eye_state(
                                face_crop,
                                face_x,
                                face_y,
                                {
                                    "landmarks": retinaface_landmarks_abs,
                                    "scores": retinaface_scores,
                                }
                            )
                        geom_info.update(mesh_info)

                        pose_info = self.extract_pose_keypoints(
                            person_roi, x1, y1
                        )
                        if pose_info.get("neck_point") is not None:
                            geom_info["neck_point"] = pose_info["neck_point"]
                        if (
                            geom_info.get("nose_point") is None
                            and pose_info.get("pose_nose") is not None
                        ):
                            geom_info["nose_point"] = pose_info["pose_nose"]

                        eye_label_map = {
                            "both": "E2",
                            "single": "E1",
                            "none": "E0",
                        }
                        eye_label = eye_label_map.get(eye_state, "E?")

                        final_result = None
                        waiting_label = ""
                        
                        # 예측 주기 체크: 마지막 예측 프레임으로부터 interval 이상 지났을 때만 예측
                        should_predict = False
                        if track_id in tracked_objects:
                            last_pred_frame = tracked_objects[track_id].get("last_prediction_frame", -1)
                            should_predict = (frame_count - last_pred_frame) >= self.prediction_interval
                        else:
                            should_predict = True  # 첫 프레임이면 예측

                        if eye_state == "both":
                            try:
                                face_tensor = self.transformations(
                                    Image.fromarray(
                                        cv2.cvtColor(
                                            face_crop,
                                            cv2.COLOR_BGR2RGB,
                                        )
                                    )
                                )
                                tracked_objects[track_id][
                                    "face_frames"
                                ].append(face_tensor)
                            except Exception:
                                pass

                            frames_ready = len(
                                tracked_objects[track_id]["face_frames"]
                            )
                            if frames_ready >= 7:
                                # 예측 주기 체크: 예측해야 할 때만 새로운 예측 수행
                                if should_predict:
                                    gaze360_result = self.estimate_gaze(
                                        list(
                                            tracked_objects[track_id][
                                                "face_frames"
                                            ]
                                        )
                                    )
                                    if gaze360_result[0] is not None:
                                        raw_result = (
                                            float(gaze360_result[0]),
                                            float(gaze360_result[1]),
                                            gaze360_result[2],
                                            "gaze360",
                                            1.0,
                                        )
                                        # 스무딩 적용
                                        gaze_history = tracked_objects[track_id].get("gaze_history")
                                        final_result = self.smooth_gaze_result(
                                            gaze_history, raw_result
                                        )
                                        # 예측 결과 캐시 및 프레임 번호 저장
                                        tracked_objects[track_id]["cached_gaze_result"] = final_result
                                        tracked_objects[track_id]["last_prediction_frame"] = frame_count
                                else:
                                    # 예측 주기 사이에는 이전 예측 결과 재사용
                                    cached_result = tracked_objects[track_id].get("cached_gaze_result")
                                    if cached_result is not None:
                                        final_result = cached_result
                            else:
                                waiting_label = f"[G360 {frames_ready}/7]"
                        else:
                            tracked_objects[track_id]["face_frames"].clear()

                        if eye_state == "single":
                            if self.pose_model is None:
                                waiting_label = "[6D DISABLED]"
                            else:
                                # 예측 주기 체크
                                if should_predict:
                                    # yaw_based 모드일 경우 이미 얻은 결과 재사용
                                    if self.eye_mode == "yaw_based" and yaw_based_sixd_result is not None:
                                        sixd_yaw, sixd_pitch, _, _, weight = yaw_based_sixd_result
                                    else:
                                        sixd_yaw, sixd_pitch, _, mode, weight = (
                                            self.estimate_head_pose(face_crop)
                                        )
                                    
                                    if sixd_yaw is not None and sixd_pitch is not None:
                                        sixd_yaw, sixd_pitch = self.apply_sixd_correction(
                                            sixd_yaw,
                                            sixd_pitch,
                                            geom_info,
                                        )
                                        gaze_vector = self.yaw_pitch_to_vector(
                                            sixd_yaw, sixd_pitch
                                        )
                                        raw_result = (
                                            sixd_yaw,
                                            sixd_pitch,
                                            gaze_vector,
                                            f"6D-CASE{self.sixd_case}",
                                            weight,
                                        )
                                        # 스무딩 적용
                                        gaze_history = tracked_objects[track_id].get("gaze_history")
                                        final_result = self.smooth_gaze_result(
                                            gaze_history, raw_result
                                        )
                                        # 예측 결과 캐시 및 프레임 번호 저장
                                        tracked_objects[track_id]["cached_gaze_result"] = final_result
                                        tracked_objects[track_id]["last_prediction_frame"] = frame_count
                                    else:
                                        waiting_label = "[6D FAIL]"
                                else:
                                    # 예측 주기 사이에는 이전 예측 결과 재사용
                                    cached_result = tracked_objects[track_id].get("cached_gaze_result")
                                    if cached_result is not None:
                                        final_result = cached_result

                        if eye_state == "none":
                            waiting_label = "[NO EYES]"

                        if final_result is not None:
                            yaw_deg, pitch_deg, gaze_vector, mode, weight = final_result
                            person_data = self.person_gaze_data[track_id]
                            last_gaze = person_data.get("last_gaze")
                            if (
                                last_gaze is not None
                                and mode == "gaze360"
                                and self.pose_model is not None
                            ):
                                yaw_delta = abs(last_gaze[0] - yaw_deg)
                                pitch_delta = abs(last_gaze[1] - pitch_deg)
                                if yaw_delta > 35.0 or pitch_delta > 25.0:
                                    sixd_yaw, sixd_pitch, _, mode, weight = (
                                        self.estimate_head_pose(face_crop)
                                    )
                                    if sixd_yaw is not None and sixd_pitch is not None:
                                        sixd_yaw, sixd_pitch = self.apply_sixd_correction(
                                            sixd_yaw,
                                            sixd_pitch,
                                            geom_info,
                                        )
                                        gaze_vector = self.yaw_pitch_to_vector(
                                            sixd_yaw,
                                            sixd_pitch,
                                        )
                                        yaw_deg = sixd_yaw
                                        pitch_deg = sixd_pitch
                                        mode = "headpose"
                                        weight = 0.7

                            self.person_gaze_data[track_id]["last_gaze"] = (
                                float(yaw_deg),
                                float(pitch_deg),
                            )

                            # 시선 방향 벡터로 ROI 매핑 (최단거리 방식 고정)
                            # Gaze360 좌표계 부호(-)를 매핑에도 동일하게 적용
                            mapped_gaze_vector = np.array(
                                [-gaze_vector[0], -gaze_vector[1], gaze_vector[2]],
                                dtype=float,
                            )
                            current_roi, _ = self.select_roi_with_ray(
                                face_center_x,
                                face_center_y,
                                mapped_gaze_vector,
                            )

                            self.update_gaze_journey(
                                track_id,
                                current_roi,
                                weight,
                            )

                            # person_roi_mapping에는 실제로 임계값을 넘어서 current_roi로 설정된 경우만 추가
                            # (temp_roi가 아닌 current_roi만 사용)
                            person_data = self.person_gaze_data[track_id]
                            confirmed_roi = person_data.get("current_roi")  # 임계값을 넘어서 확정된 ROI만
                            
                            if confirmed_roi is not None:
                                active_rois.add(confirmed_roi)
                                # ROI별로 보고 있는 사람 ID와 색상 기록
                                if confirmed_roi not in person_roi_mapping:
                                    person_roi_mapping[confirmed_roi] = []
                                # 사람 ID와 bounding box 색상 함께 저장
                                person_roi_mapping[confirmed_roi].append((track_id, color))

                            # 시선 화살표 시각화: k점에서 시선 방향으로 회전하는 화살표 (ROI 매핑용)
                            if self.show_gaze_ray:
                                # k점이 있으면 k점에서 시선 방향으로, 없으면 얼굴 중심에서 시선 방향으로
                                if self.k_anchor_point is not None:
                                    # k점에서 시선 방향 벡터로 화살표 그리기
                                    k_point = (int(self.k_anchor_point[0]), int(self.k_anchor_point[1]))
                                    arrow_start = k_point
                                    
                                    # 시선 방향 벡터 (Gaze360 좌표계 부호 적용)
                                    gaze_dir = np.array([-gaze_vector[0], -gaze_vector[1]], dtype=float)
                                    dir_norm = np.linalg.norm(gaze_dir)
                                    if dir_norm > 1e-6:
                                        gaze_dir /= dir_norm
                                        
                                        # 화살표 길이: gaze_distance_k 또는 화면 크기 기반 (더 짧게 조정)
                                        h, w = img.shape[:2]
                                        if self.gaze_distance_k > 0:
                                            # k점에서 시선 방향으로 gaze_distance_k만큼 (최대 화면 크기의 35%로 제한)
                                            max_len = max(h, w) * 0.35
                                            ray_len = float(min(self.gaze_distance_k, max_len))
                                        else:
                                            # 기본값: 화면 크기의 20%
                                            ray_len = float(max(h, w) * 0.2)
                                        
                                        arrow_end = (
                                            int(arrow_start[0] + gaze_dir[0] * ray_len),
                                            int(arrow_start[1] + gaze_dir[1] * ray_len),
                                        )
                                        
                                        # 화살표 두께: 선택된 ROI가 있으면 더 두껍게
                                        arrow_thickness = 5 if confirmed_roi is not None else 3
                                        # 화살표 색상: 선택된 ROI가 있으면 녹색, 없으면 노란색
                                        arrow_color = (0, 255, 0) if confirmed_roi is not None else (0, 255, 255)
                                        
                                        # k점 표시 (시작점)
                                        cv2.circle(
                                            img,
                                            k_point,
                                            10,
                                            (255, 0, 255),  # 마젠타색
                                            -1,
                                        )
                                        cv2.circle(
                                            img,
                                            k_point,
                                            10,
                                            (255, 255, 255),  # 흰색 테두리
                                            2,
                                        )
                                        # k점 라벨
                                        cv2.putText(
                                            img,
                                            "k",
                                            (k_point[0] - 8, k_point[1] - 12),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6,
                                            (255, 255, 255),
                                            2,
                                        )
                                        
                                        # 얼굴 중심점 표시 (참고용)
                                        cv2.circle(
                                            img,
                                            (face_center_x, face_center_y),
                                            6,
                                            (0, 255, 0),  # 녹색
                                            -1,
                                        )
                                        cv2.circle(
                                            img,
                                            (face_center_x, face_center_y),
                                            6,
                                            (255, 255, 255),  # 흰색 테두리
                                            2,
                                        )
                                        # 얼굴 중심 라벨
                                        cv2.putText(
                                            img,
                                            "face_center_x,y",
                                            (face_center_x + 10, face_center_y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            (0, 0, 255),  # 빨간색
                                            2,
                                        )
                                        
                                        # 화살표 그리기 (k점에서 시선 방향으로)
                                        cv2.arrowedLine(
                                            img,
                                            arrow_start,
                                            arrow_end,
                                            arrow_color,
                                            arrow_thickness,
                                            tipLength=0.15,
                                        )
                                        
                                        # yaw/pitch 각도 텍스트 표시 (k점 근처)
                                        angle_text = f"Y:{yaw_deg:.1f}° P:{pitch_deg:.1f}°"
                                        text_x = k_point[0] + 15
                                        text_y = k_point[1] - 15
                                    
                                    # 텍스트 배경
                                    (tw, th), _ = cv2.getTextSize(
                                        angle_text,
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        2,
                                    )
                                    cv2.rectangle(
                                        img,
                                        (text_x - 5, text_y - th - 5),
                                        (text_x + tw + 5, text_y + 5),
                                        (0, 0, 0),
                                        -1,
                                    )
                                    cv2.putText(
                                        img,
                                        angle_text,
                                        (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        arrow_color,
                                        2,
                                    )
                                else:
                                    # k점이 없으면 기존 방식: 얼굴 중심에서 시선 방향으로
                                    gaze_dir = np.array([-gaze_vector[0], -gaze_vector[1]], dtype=float)
                                    dir_norm = np.linalg.norm(gaze_dir)
                                    if dir_norm > 1e-6:
                                        gaze_dir /= dir_norm
                                        
                                        arrow_start = (face_center_x, face_center_y)
                                        h, w = img.shape[:2]
                                        ray_len = float(max(h, w) * 0.3)
                                        
                                        ray_end = (
                                            int(arrow_start[0] + gaze_dir[0] * ray_len),
                                            int(arrow_start[1] + gaze_dir[1] * ray_len),
                                        )
                                        
                                        arrow_thickness = 5 if confirmed_roi is not None else 3
                                        arrow_color = (0, 255, 0) if confirmed_roi is not None else (0, 255, 255)
                                        
                                        cv2.circle(
                                            img,
                                            arrow_start,
                                            6,
                                            (0, 255, 0),
                                            -1,
                                        )
                                        cv2.circle(
                                            img,
                                            arrow_start,
                                            6,
                                            (255, 255, 255),
                                            2,
                                        )
                                        
                                        cv2.arrowedLine(
                                            img,
                                            arrow_start,
                                            ray_end,
                                            arrow_color,
                                            arrow_thickness,
                                            tipLength=0.15,
                                        )
                                        
                                        angle_text = f"Y:{yaw_deg:.1f}° P:{pitch_deg:.1f}°"
                                        text_x = arrow_start[0] + 15
                                        text_y = arrow_start[1] - 15
                                        
                                        # 텍스트 배경
                                        (tw, th), _ = cv2.getTextSize(
                                            angle_text,
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6,
                                            2,
                                        )
                                        cv2.rectangle(
                                            img,
                                            (text_x - 5, text_y - th - 5),
                                            (text_x + tw + 5, text_y + 5),
                                            (0, 0, 0),
                                            -1,
                                        )
                                        cv2.putText(
                                            img,
                                            angle_text,
                                            (text_x, text_y),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6,
                                            arrow_color,
                                            2,
                                        )

                            # 라벨 표시는 임계값을 넘어서 확정된 ROI만 사용
                            roi_label = (
                                f"RoI:{confirmed_roi}"
                                if confirmed_roi is not None
                                else "No RoI"
                            )
                            mode_label = f"[{mode.upper()}]"
                            weight_label = f"W:{weight:.2f}" if weight > 0 else ""

                            label = (
                                f"ID:{track_id} {obj_name} {eye_label} "
                                f"Y:{yaw_deg:.1f}° P:{pitch_deg:.1f}° "
                                f"{mode_label} {weight_label} {roi_label}"
                            )
                        else:
                            if waiting_label:
                                label = (
                                    f"ID:{track_id} {obj_name} {eye_label} "
                                    f"{waiting_label}"
                                )
                            else:
                                label = (
                                    f"ID:{track_id} {obj_name} {eye_label} "
                                    "[NO INFERENCE]"
                                )

            data_deque[track_id].appendleft(center)
            self.UI_box(
                box,
                img,
                label=label,
                color=color,
                line_thickness=2,
            )

            # 트랙 궤적
            for j in range(1, len(data_deque[track_id])):
                if (
                    data_deque[track_id][j - 1] is None
                    or data_deque[track_id][j] is None
                ):
                    continue
                thickness = int(
                    math.sqrt(64.0 / float(j + j)) * 1.5
                )
                cv2.line(
                    img,
                    data_deque[track_id][j - 1],
                    data_deque[track_id][j],
                    color,
                    thickness,
                )

        return img, active_rois, person_roi_mapping

    def draw_rois(self, img, active_rois=None, person_roi_mapping=None):
        """RoI들을 이미지에 그리기 (rect / polygon 지원)
        
        Args:
            img: 입력 이미지
            active_rois: 활성 ROI ID 집합
            person_roi_mapping: {roi_id: [person_id1, person_id2, ...]} - 각 ROI를 보고 있는 사람들의 ID 리스트
        """
        global drawing, current_roi, data_deque

        if active_rois is None:
            active_rois = set()
        if person_roi_mapping is None:
            person_roi_mapping = {}

        for roi in self.rois:
            roi_type = roi.get("type", "rect")
            roi_id = roi.get("roi_id", 0)
            data = roi.get("data")
            
            # ROI를 보고 있는 사람들이 있으면 그들의 색상 사용, 없으면 기본 ROI 색상
            if roi_id in person_roi_mapping and len(person_roi_mapping[roi_id]) > 0:
                person_data_list = person_roi_mapping[roi_id]  # [(track_id, color), ...]
                
                if len(person_data_list) == 1:
                    # 한 명만 보고 있으면 그 사람의 색상 사용
                    _, color = person_data_list[0]
                else:
                    # 여러 사람이 보고 있으면 색상 혼합
                    colors = [person_color for _, person_color in person_data_list]
                    # RGB 각 채널의 평균 계산
                    mixed_color = (
                        int(np.mean([c[0] for c in colors])),
                        int(np.mean([c[1] for c in colors])),
                        int(np.mean([c[2] for c in colors])),
                    )
                    color = mixed_color
            else:
                color = roi_colors[(roi_id - 1) % len(roi_colors)]

            if roi_type == "rect":
                x1, y1, x2, y2 = data
                if roi_id in active_rois:
                    overlay = img.copy()
                    cv2.rectangle(
                        overlay,
                        (x1, y1),
                        (x2, y2),
                        color,
                        -1,
                    )
                    cv2.addWeighted(
                        overlay,
                        0.3,  # 더 진하게
                        img,
                        0.7,
                        0,
                        img,
                    )
                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        color,
                        4,
                    )

                    # 몇 명이 보고 있는지 표시
                    person_data_list = person_roi_mapping.get(roi_id, [])
                    person_count = len(person_data_list)
                    if person_count > 0:
                        person_ids = [str(pid) for pid, _ in person_data_list]
                        label = f"RoI {roi_id} [ID:{','.join(person_ids)}]"
                    else:
                        label = f"RoI {roi_id} [ACTIVE]"
                    (tw, th), _ = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        2,
                    )
                    cv2.rectangle(
                        img,
                        (x1, y1 - th - 15),
                        (x1 + tw + 10, y1 - 5),
                        color,
                        -1,
                    )
                    cv2.putText(
                        img,
                        label,
                        (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    
                    # 디버깅: 각도 차이 표시
                    if roi_id in self.debug_roi_info:
                        debug_info = self.debug_roi_info[roi_id]
                        angle_text = f"∠{debug_info['angle_diff_deg']:.1f}°"
                        dist_text = f"d:{debug_info['distance']:.0f}px"
                        score_text = f"s:{debug_info['score']:.1f}"
                        debug_label = f"{angle_text} {dist_text} {score_text}"
                        (dtw, dth), _ = cv2.getTextSize(
                            debug_label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            1,
                        )
                        cv2.rectangle(
                            img,
                            (x1, y2 + 5),
                            (x1 + dtw + 10, y2 + dth + 15),
                            (0, 0, 0),
                            -1,
                        )
                        cv2.putText(
                            img,
                            debug_label,
                            (x1 + 5, y2 + dth + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),  # 노란색
                            1,
                        )
                        
                        # k점에서 ROI 중심까지의 선 그리기 (시선 벡터 검증용)
                        if self.k_anchor_point is not None:
                            k_point = (int(self.k_anchor_point[0]), int(self.k_anchor_point[1]))
                            roi_center = debug_info['center']
                            # 각도에 따라 색상 변경 (작을수록 녹색, 클수록 빨간색)
                            if debug_info['angle_diff_deg'] < 15:
                                line_color = (0, 255, 0)  # 녹색
                            elif debug_info['angle_diff_deg'] < 30:
                                line_color = (0, 255, 255)  # 노란색
                            else:
                                line_color = (0, 0, 255)  # 빨간색
                            cv2.line(
                                img,
                                k_point,
                                roi_center,
                                line_color,
                                1,
                            )
                else:
                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )
                    label = f"RoI {roi_id}"
                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            elif roi_type == "poly":
                pts = np.array(data, np.int32).reshape((-1, 1, 2))
                if roi_id in active_rois:
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.addWeighted(
                        overlay,
                        0.3,  # 더 진하게
                        img,
                        0.7,
                        0,
                        img,
                    )
                    cv2.polylines(
                        img,
                        [pts],
                        True,
                        color,
                        4,
                    )
                    label_x, label_y = data[0]
                    # 몇 명이 보고 있는지 표시
                    person_data_list = person_roi_mapping.get(roi_id, [])
                    person_count = len(person_data_list)
                    if person_count > 0:
                        person_ids = [str(pid) for pid, _ in person_data_list]
                        label = f"RoI {roi_id} [ID:{','.join(person_ids)}]"
                    else:
                        label = f"RoI {roi_id} [ACTIVE]"
                    (tw, th), _ = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        2,
                    )
                    cv2.rectangle(
                        img,
                        (label_x, label_y - th - 15),
                        (label_x + tw + 10, label_y - 5),
                        color,
                        -1,
                    )
                    cv2.putText(
                        img,
                        label,
                        (label_x + 5, label_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    
                    # 디버깅: 각도 차이 표시 (polygon)
                    if roi_id in self.debug_roi_info:
                        debug_info = self.debug_roi_info[roi_id]
                        angle_text = f"∠{debug_info['angle_diff_deg']:.1f}°"
                        dist_text = f"d:{debug_info['distance']:.0f}px"
                        score_text = f"s:{debug_info['score']:.1f}"
                        debug_label = f"{angle_text} {dist_text} {score_text}"
                        (dtw, dth), _ = cv2.getTextSize(
                            debug_label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            1,
                        )
                        # polygon의 하단 중앙에 표시
                        poly_bottom = max([pt[1] for pt in data])
                        poly_left = min([pt[0] for pt in data])
                        cv2.rectangle(
                            img,
                            (poly_left, poly_bottom + 5),
                            (poly_left + dtw + 10, poly_bottom + dth + 15),
                            (0, 0, 0),
                            -1,
                        )
                        cv2.putText(
                            img,
                            debug_label,
                            (poly_left + 5, poly_bottom + dth + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),  # 노란색
                            1,
                        )
                        
                        # k점에서 ROI 중심까지의 선 그리기 (시선 벡터 검증용)
                        if self.k_anchor_point is not None:
                            k_point = (int(self.k_anchor_point[0]), int(self.k_anchor_point[1]))
                            roi_center = debug_info['center']
                            # 각도에 따라 색상 변경 (작을수록 녹색, 클수록 빨간색)
                            if debug_info['angle_diff_deg'] < 15:
                                line_color = (0, 255, 0)  # 녹색
                            elif debug_info['angle_diff_deg'] < 30:
                                line_color = (0, 255, 255)  # 노란색
                            else:
                                line_color = (0, 0, 255)  # 빨간색
                            cv2.line(
                                img,
                                k_point,
                                roi_center,
                                line_color,
                                1,
                            )
                else:
                    cv2.polylines(
                        img,
                        [pts],
                        True,
                        color,
                        2,
                    )
                    label_x, label_y = data[0]
                    label = f"RoI {roi_id}"
                    cv2.putText(
                        img,
                        label,
                        (label_x, label_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

        # 인터랙티브 드로잉 중인 직사각형
        if drawing and current_roi is not None:
            x1, y1, x2, y2 = current_roi
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

        return img

    @staticmethod
    def draw_info_panel(img, active_rois, num_people: int):
        """화면 상단에 정보 패널 그리기"""
        height, width = img.shape[:2]

        panel_height = 120
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (width, panel_height),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        cv2.putText(
            img,
            "RoI Gaze Tracking System (Gaze360)",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            img,
            f"Tracking: {num_people} people",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            img,
            f"Active RoIs: {len(active_rois)}",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        legend_x = width - 350
        cv2.putText(
            img,
            "Legend:",
            (legend_x, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.circle(
            img,
            (legend_x + 10, 45),
            6,
            (255, 0, 0),
            -1,
        )
        cv2.circle(
            img,
            (legend_x + 10, 45),
            8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            "Face Center",
            (legend_x + 25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.circle(
            img,
            (legend_x + 10, 70),
            10,
            (0, 0, 255),
            -1,
        )
        cv2.circle(
            img,
            (legend_x + 10, 70),
            12,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            img,
            "Gaze in RoI",
            (legend_x + 25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.circle(
            img,
            (legend_x + 10, 95),
            10,
            (0, 255, 0),
            -1,
        )
        cv2.circle(
            img,
            (legend_x + 10, 95),
            12,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            "Gaze outside",
            (legend_x + 25, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return img

    # -------------------------
    # 프레임 처리
    # -------------------------
    def process_frame(self, frame, frame_count: int = 0):
        """단일 프레임 처리"""
        # Depth 추정 (prediction_interval과 동일한 간격으로 업데이트)
        if self.use_depth and (frame_count % self.prediction_interval == 0 or self.last_depth_update_frame < 0):
            estimated_depth = self.estimate_roi_plane_depth(frame)
            if estimated_depth is not None:
                self.last_depth_update_frame = frame_count
        
        results = self.detection_model(frame, verbose=False)
        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                if int(box.cls) == 0:  # person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    detections.append(
                        [float(x1), float(y1), float(x2), float(y2), conf, cls]
                    )

        active_rois = set()
        num_people = len(detections)

        if len(detections) > 0:
            tracks = self.simple_track(detections, frame_count)
            if len(tracks) > 0:
                bbox_xyxy = []
                identities = []
                object_id = []
                for tr in tracks:
                    x1, y1, x2, y2, track_id, cls = tr
                    bbox_xyxy.append([x1, y1, x2, y2])
                    identities.append(track_id)
                    object_id.append(cls)

                bbox_xyxy = np.array(bbox_xyxy)
                identities = np.array(identities)
                object_id = np.array(object_id)

                frame, active_rois, person_roi_mapping = self.draw_boxes_with_gaze(
                    frame,
                    bbox_xyxy,
                    self.detection_model.names,
                    object_id,
                    identities,
                    offset=(0, 0),
                    frame_count=frame_count,
                )
            else:
                person_roi_mapping = {}
        else:
            person_roi_mapping = {}

        frame = self.draw_rois(frame, active_rois, person_roi_mapping)
        frame = self.draw_info_panel(frame, active_rois, num_people)

        return frame

    # -------------------------
    # ROI config / 설정
    # -------------------------
    def load_rois_from_config(self, config_path: str) -> bool:
        """JSON config 파일에서 ROI 로드"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            rois_list = []
            for roi_config in config.get("rois", []):
                roi_type = roi_config.get("type", "rect")
                roi_id = roi_config.get("roi_id", len(rois_list) + 1)
                data = roi_config.get("data")
                if roi_type not in ("rect", "poly"):
                    continue
                if data is None:
                    continue

                rois_list.append(
                    {
                        "type": roi_type,
                        "data": data,
                        "roi_id": roi_id,
                    }
                )

            self.rois = rois_list

            print(f"\nConfig 파일에서 {len(self.rois)}개의 ROI를 로드했습니다:")
            for roi in self.rois:
                roi_type = roi.get("type", "rect")
                roi_id = roi.get("roi_id", 0)
                if roi_type == "rect":
                    x1, y1, x2, y2 = roi["data"]
                    print(
                        f"  RoI {roi_id} (rect): "
                        f"({x1}, {y1}) - ({x2}, {y2})"
                    )
                else:
                    print(
                        f"  RoI {roi_id} (poly): "
                        f"{len(roi['data'])} points"
                    )

            # 선택적: ROI 평면 정보 로드 (3D 레이-평면 교차용)
            plane_origin = config.get("roi_plane_origin")
            plane_normal = config.get("roi_plane_normal")
            if (
                isinstance(plane_origin, (list, tuple))
                and len(plane_origin) == 3
                and isinstance(plane_normal, (list, tuple))
                and len(plane_normal) == 3
            ):
                try:
                    parsed_origin = np.array(plane_origin, dtype=float)
                    parsed_normal = np.array(plane_normal, dtype=float)
                    norm = float(np.linalg.norm(parsed_normal))
                    if norm > 1e-6:
                        parsed_normal = parsed_normal / norm
                        self.roi_plane_origin = parsed_origin
                        self.roi_plane_normal = parsed_normal
                        print(
                            "ROI 평면 정보 로드됨: "
                            f"origin={self.roi_plane_origin.tolist()}, "
                            f"normal={self.roi_plane_normal.tolist()}"
                        )
                except Exception:
                    # 평면 정보 파싱 실패 시 무시하고 기존 상태 유지
                    pass
            return True
        except Exception as e:
            print(f"Config 파일 로드 오류: {e}")
            return False

    def save_rois_to_config(self, save_path: str):
        """현재 self.rois를 JSON 파일로 저장 (rect/poly 및 평면 정보)"""
        try:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            data = {"rois": self.rois}
            if (
                getattr(self, "roi_plane_origin", None) is not None
                and getattr(self, "roi_plane_normal", None) is not None
            ):
                data["roi_plane_origin"] = self.roi_plane_origin.tolist()
                data["roi_plane_normal"] = self.roi_plane_normal.tolist()
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"ROI 설정을 저장했습니다: {save_path}")
        except Exception as e:
            print(f"ROI 저장 실패: {e}")

    def setup_rois(self, video_path: str, config_path: Optional[str] = None):
        """
        비디오의 첫 프레임에서 RoI 설정
        """
        global rois, current_roi, drawing, polygon_mode, current_polygon_points

        # 1) Config 우선
        if config_path and os.path.exists(config_path):
            if self.load_rois_from_config(config_path):
                return

        # 2) 인터랙티브 모드
        rois = []
        current_roi = None
        drawing = False
        polygon_mode = False
        current_polygon_points = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"비디오 파일을 열 수 없습니다: {video_path}")
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise IOError("비디오의 첫 프레임을 읽을 수 없습니다")

        window_name = "RoI 설정 - 드래그하여 RoI 생성, 'c' 완료, 'r' 초기화"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        print("\n=== RoI 설정 모드 ===")
        print("여러 개의 ROI를 설정할 수 있습니다!")
        print()
        print("Rect 모드(기본): 드래그하여 직사각형 RoI 생성")
        print("  - 드래그하여 RoI 생성 후, 다시 드래그하면 추가 RoI 생성 가능")
        print()
        print("Poly 모드: 'm' 키로 전환 후 왼쪽 클릭으로 꼭짓점 4개 추가")
        print("  - 정확히 4개의 점을 찍어 사각형(또는 사다리꼴) ROI를 생성")
        print("  - 'p' 키로 현재 RoI 완료 후, 다시 4개 점을 찍어 다음 RoI 추가 가능")
        print("  - 'u' 키로 마지막 점 삭제")
        print()
        print("키보드 단축키:")
        print("  'c' 키: 모든 RoI 설정 완료 (최소 1개 필요)")
        print("  'r' 키: 모든 RoI 초기화")
        print("  'm' 키: Rect/Poly 모드 전환")
        print("====================\n")

        while True:
            display_frame = frame.copy()

            for roi_entry in rois:
                if isinstance(roi_entry, dict):
                    roi_type = roi_entry.get("type", "rect")
                    roi_id = roi_entry.get("roi_id", len(rois))
                    color = roi_colors[(roi_id - 1) % len(roi_colors)]
                    if roi_type == "rect":
                        x1, y1, x2, y2 = roi_entry["data"]
                        cv2.rectangle(
                            display_frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2,
                        )
                        cv2.putText(
                            display_frame,
                            f"RoI {roi_id}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )
                    elif roi_type == "poly":
                        pts = np.array(roi_entry["data"], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(
                            display_frame,
                            [pts],
                            True,
                            color,
                            2,
                        )
                        label_x, label_y = roi_entry["data"][0]
                        cv2.putText(
                            display_frame,
                            f"RoI {roi_id}",
                            (label_x, label_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )
                else:
                    # 레거시 튜플 처리
                    if (
                        isinstance(roi_entry, tuple)
                        and len(roi_entry) == 5
                    ):
                        x1, y1, x2, y2, roi_id = roi_entry
                        color = roi_colors[(roi_id - 1) % len(roi_colors)]
                        cv2.rectangle(
                            display_frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2,
                        )
                        cv2.putText(
                            display_frame,
                            f"RoI {roi_id}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

            if drawing and current_roi is not None:
                x1, y1, x2, y2 = current_roi
                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2,
                )

            if polygon_mode and len(current_polygon_points) > 0:
                pts = np.array(current_polygon_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    display_frame,
                    [pts],
                    False,
                    (0, 255, 255),
                    2,
                )
                # 각 점에 번호 표시
                for idx, pt in enumerate(current_polygon_points):
                    cv2.circle(display_frame, pt, 6, (0, 255, 255), -1)
                    cv2.circle(display_frame, pt, 8, (255, 255, 255), 2)
                    # 점 번호 표시 (1, 2, 3, 4)
                    cv2.putText(
                        display_frame,
                        str(idx + 1),
                        (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                # 4개 점이 모두 있으면 닫힌 다각형 표시
                if len(current_polygon_points) == 4:
                    cv2.polylines(
                        display_frame,
                        [pts],
                        True,  # 닫힌 다각형
                        (0, 255, 0),
                        2,
                    )

            # 상단 안내 메시지
            if polygon_mode:
                guide_text = "Poly Mode: Click 4 points, then press 'p' to add RoI | 'c': Finish | 'r': Reset All"
            else:
                guide_text = "Rect Mode: Drag to create RoI | 'm': Switch mode | 'c': Finish | 'r': Reset All"
            
            cv2.putText(
                display_frame,
                guide_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            
            # 현재 설정된 ROI 개수 (더 눈에 띄게)
            roi_count_text = f"Total RoIs: {len(rois)}"
            if polygon_mode and len(current_polygon_points) > 0:
                roi_count_text += f" | Current: {len(current_polygon_points)}/4 points"
            
            cv2.putText(
                display_frame,
                roi_count_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            
            # 여러 ROI 추가 안내
            if len(rois) > 0:
                cv2.putText(
                    display_frame,
                    f"RoI {len(rois)} completed. You can add more RoIs!",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
            # 모드 표시 (하단으로 이동)
            mode_text = "POLY" if polygon_mode else "RECT"
            height, width = display_frame.shape[:2]
            if polygon_mode:
                mode_display = f"Mode: {mode_text} ({len(current_polygon_points)}/4 points) | Press 'm' to toggle"
            else:
                mode_display = f"Mode: {mode_text} | Press 'm' to toggle"
            
            cv2.putText(
                display_frame,
                mode_display,
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                if len(rois) == 0:
                    print("최소 1개 이상의 RoI를 설정해주세요.")
                else:
                    break
            elif key == ord("r"):
                rois = []
                current_polygon_points = []
                print("모든 RoI가 초기화되었습니다.")
            elif key == ord("m"):
                polygon_mode = not polygon_mode
                current_polygon_points = []
                mode_label = "POLY" if polygon_mode else "RECT"
                print(f"RoI 모드 전환: {mode_label}")
            elif key == ord("p") and polygon_mode:
                if len(current_polygon_points) != 4:
                    print(f"폴리곤 RoI는 정확히 4개의 점이 필요합니다. (현재: {len(current_polygon_points)}/4)")
                else:
                    roi_id = len(rois) + 1
                    rois.append(
                        {
                            "type": "poly",
                            "data": current_polygon_points.copy(),
                            "roi_id": roi_id,
                        }
                    )
                    print(
                        f"✓ RoI {roi_id} (poly, 4점) 추가됨: "
                        f"{current_polygon_points}"
                    )
                    print(f"  현재 총 {len(rois)}개의 RoI가 설정되었습니다. 더 추가하려면 다시 4개 점을 찍으세요.")
                    current_polygon_points = []
            elif key == ord("u") and polygon_mode:
                if current_polygon_points:
                    removed = current_polygon_points.pop()
                    print(f"폴리곤 점 제거: {removed}")

        cv2.destroyWindow(window_name)

        # 전역 튜플 RoI -> self.rois(dict) 변환
        self.rois = []
        for roi_entry in rois:
            if isinstance(roi_entry, dict):
                self.rois.append(roi_entry.copy())
            elif (
                isinstance(roi_entry, tuple)
                and len(roi_entry) == 5
            ):
                x1, y1, x2, y2, roi_id = roi_entry
                self.rois.append(
                    {
                        "type": "rect",
                        "data": [x1, y1, x2, y2],
                        "roi_id": roi_id,
                    }
                )

        print(f"\n총 {len(self.rois)}개의 RoI가 설정되었습니다:")
        for roi in self.rois:
            roi_id = roi.get("roi_id", 0)
            if roi.get("type", "rect") == "rect":
                x1, y1, x2, y2 = roi["data"]
                print(f"  RoI {roi_id} (rect): ({x1}, {y1}) - ({x2}, {y2})")
            else:
                print(
                    f"  RoI {roi_id} (poly): {len(roi['data'])} points"
                )

    # -------------------------
    # Gaze journey 저장
    # -------------------------
    def save_gaze_journey_to_excel(self, output_path: str):
        """gaze_journey 데이터를 엑셀 파일로 저장"""
        if not self.person_gaze_data:
            print("저장할 gaze_journey 데이터가 없습니다.")
            return

        self.finalize_gaze_journey()
        excel_path = output_path.replace(".mp4", "_gaze_journey.xlsx")

        try:
            data = []
            # 모든 사람의 ROI 번호만 수집
            all_roi_numbers = set()
            
            for person_id, person_data in self.person_gaze_data.items():
                if len(person_data["gaze_journey"]) > 0:
                    # ROI 번호만 추출 (순서 유지, 중복 제거)
                    roi_numbers = [roi_id for roi_id, _ in person_data["gaze_journey"]]
                    all_roi_numbers.update(roi_numbers)
                    
                    # 순서 유지하면서 중복 제거 (첫 번째 발생만 유지)
                    seen = set()
                    unique_ordered = []
                    for roi_id in roi_numbers:
                        if roi_id not in seen:
                            seen.add(roi_id)
                            unique_ordered.append(roi_id)
                    
                    # 결과 출력 형식: ROI 번호만 순서대로 (예: 1,2,3,4)
                    roi_str = ",".join(map(str, unique_ordered))
                    data.append(
                        {
                            "person_id": person_id,
                            "roi_numbers": roi_str,
                        }
                    )

            if len(data) == 0:
                print("저장할 유효한 gaze_journey가 없습니다.")
                return

            df = pd.DataFrame(data)
            df.to_excel(excel_path, index=False)

            print(f"\ngaze_journey 데이터가 저장되었습니다: {excel_path}")
            print(f"총 {len(data)}명의 데이터가 저장되었습니다.")

            print("\n=== ROI 매핑 결과 (방문 순서) ===")
            for person_id, person_data in sorted(
                self.person_gaze_data.items()
            ):
                if len(person_data["gaze_journey"]) > 0:
                    # ROI 번호만 추출 (순서 유지, 중복 제거)
                    roi_numbers = [roi_id for roi_id, _ in person_data["gaze_journey"]]
                    # 순서 유지하면서 중복 제거 (첫 번째 발생만 유지)
                    seen = set()
                    unique_ordered = []
                    for roi_id in roi_numbers:
                        if roi_id not in seen:
                            seen.add(roi_id)
                            unique_ordered.append(roi_id)
                    roi_str = ",".join(map(str, unique_ordered))
                    print(f"Person ID {person_id}: {roi_str}")
            
            # 전체 ROI 번호 출력
            if all_roi_numbers:
                all_roi_str = ",".join(map(str, sorted(all_roi_numbers)))
                print(f"\n전체 ROI: {all_roi_str}")
            print("========================\n")

        except Exception as e:
            print(f"엑셀 저장 오류: {e}")

    # -------------------------
    # 비디오 실행
    # -------------------------
    def run_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        roi_config_path: Optional[str] = None,
        save_roi_config: Optional[str] = None,
        interactive_k: bool = False,
        interactive_face_center: bool = False,
    ):
        """비디오 파일 처리"""
        self.setup_rois(video_path, roi_config_path)

        if interactive_k:
            try:
                self.interactive_set_k(video_path)
            except Exception as e:
                print(f"interactive k 설정 중 오류: {e}")

        if interactive_face_center:
            try:
                self.interactive_set_face_center(video_path)
            except Exception as e:
                print(f"interactive face center 설정 중 오류: {e}")

        # gaze 데이터 초기화
        self.person_gaze_data = defaultdict(
            lambda: {
                "current_roi": None,
                "frame_count": 0,
                "weighted_count": 0.0,
                "gaze_journey": [],
                "temp_roi": None,
                "temp_frame_count": 0,
                "temp_weighted_count": 0.0,
            }
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"비디오 파일을 열 수 없습니다: {video_path}")

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25.0
            fps = float(fps)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height),
            )
        else:
            out = None

        print(f"\n비디오 처리를 시작합니다: {video_path}")
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                print(
                    f"프레임 처리 중: {frame_count}/{total_frames}",
                    end="\r",
                )

                processed_frame = self.process_frame(
                    frame,
                    frame_count,
                )
                frame_count += 1

                if out is not None:
                    out.write(processed_frame)
                else:
                    cv2.imshow(
                        "RoI 기반 시선 추적 (Gaze360)",
                        processed_frame,
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

        print(
            f"\n비디오 처리 완료: "
            f"{output_path if output_path else '화면 표시'}"
        )

        if output_path:
            self.save_gaze_journey_to_excel(output_path)

        if save_roi_config:
            self.save_rois_to_config(save_roi_config)


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="RoI 기반 시선 추적 시스템 (Gaze360)"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="입력 비디오 파일 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 비디오 파일 경로 (선택사항)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU ID (CPU 사용시 -1, 기본값: -1)",
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="weights/yolov8n.pt",
        help="YOLOv8 모델 경로",
    )
    parser.add_argument(
        "--gaze_model",
        type=str,
        default="weights/gaze360_model.pth.tar",
        help="Gaze360 모델 경로",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=5,
        help="RoI 응시 판단을 위한 최소 프레임 수 (기본값: 5)",
    )
    parser.add_argument(
        "--gaze_distance",
        type=int,
        default=300,
        help="시선 끝점 계산을 위한 거리 상수 k (기본값: 300)",
    )
    parser.add_argument(
        "--gaze_distance_k",
        type=int,
        default=None,
        help="시선 끝점 거리 k 별도 지정 (지정 시 gaze_distance를 덮어씀)",
    )
    parser.add_argument(
        "--show_ray",
        dest="show_ray",
        action="store_true",
        default=True,
        help="시선 ray를 화면에 표시 (기본: 표시)",
    )
    parser.add_argument(
        "--no_show_ray",
        dest="show_ray",
        action="store_false",
        help="시선 ray를 표시하지 않음",
    )
    # ROI 매핑은 최단거리 방식으로 고정하므로 추가 옵션이 없습니다.
    parser.add_argument(
        "--sixdrepnet_model",
        type=str,
        default="weights/6DRepNet_300W_LP_AFLW2000.pth",
        help="6DRepNet 모델 경로 (head pose fallback용)",
    )
    parser.add_argument(
        "--roi_config",
        type=str,
        default=None,
        help="ROI config JSON 파일 경로 (선택사항)",
    )
    parser.add_argument(
        "--save_roi_config",
        type=str,
        default=None,
        help="인터랙티브로 설정한 ROI를 저장할 JSON 경로 (선택사항)",
    )
    parser.add_argument(
        "--roi_plane_origin",
        type=str,
        default=None,
        help="ROI 평면의 한 점 (x,y,z). 예: \"0,0,500\"",
    )
    parser.add_argument(
        "--roi_plane_normal",
        type=str,
        default=None,
        help="ROI 평면 법선 (nx,ny,nz). 예: \"0,0,1\"",
    )
    parser.add_argument(
        "--roi_plane_z",
        type=float,
        default=None,
        help="카메라 전방 +z 거리(픽셀 가정). origin=(0,0,-z), normal=(0,0,-1)로 설정",
    )
    parser.add_argument(
        "--sixd_case",
        type=int,
        default=1,
        help="6DRepNet 보정 케이스 (1:원본, 2:목-코, 3:머리-코, 4:혼합)",
    )
    parser.add_argument(
        "--vector_weight",
        type=float,
        default=0.35,
        help="벡터 보정 가중치 (0~1, 기본 0.35)",
    )
    parser.add_argument(
        "--eye_mode",
        type=str,
        default="auto",
        choices=["auto", "mediapipe", "retinaface", "yoloface", "quality", "yaw_based"],
        help="눈 가시성 판별 모드 (auto/mediapipe/retinaface/yoloface/quality/yaw_based)",
    )
    parser.add_argument(
        "--yolo_face_model",
        type=str,
        default=None,
        help="YOLO face 랜드마크 모델 경로 (yoloface 모드용)",
    )
    parser.add_argument(
        "--yaw_threshold_both",
        type=float,
        default=40.0,
        help="양안 확보를 위한 Yaw 임계값 (yaw_based 모드용, 기본값: 40도)",
    )
    parser.add_argument(
        "--yaw_threshold_single",
        type=float,
        default=90.0,
        help="단안을 위한 Yaw 임계값 (yaw_based 모드용, 기본값: 90도)",
    )
    parser.add_argument(
        "--pitch_threshold_both",
        type=float,
        default=30.0,
        help="양안 확보를 위한 Pitch 임계값 (yaw_based 모드용, 기본값: 30도)",
    )
    parser.add_argument(
        "--pitch_threshold_single",
        type=float,
        default=60.0,
        help="단안을 위한 Pitch 임계값 (yaw_based 모드용, 기본값: 60도)",
    )
    parser.add_argument(
        "--prediction_interval",
        type=int,
        default=5,
        help="시선 추정을 수행할 프레임 간격 (기본값: 5, 매 프레임마다는 1)",
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=3,
        help="시선 벡터 스무딩을 위한 히스토리 윈도우 크기 (기본값: 3, 0이면 스무딩 비활성화)",
    )
    parser.add_argument(
        "--use_depth",
        action="store_true",
        default=False,
        help="MiDaS depth 추정을 사용하여 ROI 평면 깊이를 동적으로 업데이트 (기본값: False)",
    )
    parser.add_argument(
        "--depth_model_type",
        type=str,
        default="DPT_Large",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        help="MiDaS depth 모델 타입 (기본값: DPT_Large)",
    )
    parser.add_argument(
        "--interactive_k",
        action="store_true",
        default=False,
        help="첫 프레임에서 클릭해 gaze_distance_k를 설정 (화면 중심 기준 거리)",
    )
    parser.add_argument(
        "--ray_max_len",
        type=float,
        default=0.0,
        help="레이 최대 길이 클립 (0이면 화면크기 기반 자동)",
    )
    parser.add_argument(
        "--ray_shrink_px",
        type=int,
        default=6,
        help="레이 교차 판정 시 ROI를 안쪽으로 축소할 픽셀(경계 접촉 감소)",
    )
    parser.add_argument(
        "--ray_angle_thresh",
        type=float,
        default=0.2,
        help="레이-ROI 중심 cos 임계값 (여러 교차 시 방향 필터)",
    )
    parser.add_argument(
        "--interactive_face_center",
        action="store_true",
        default=False,
        help="첫 프레임에서 클릭으로 얼굴 중심을 수동 지정",
    )
    parser.add_argument(
        "--roi_angle_threshold",
        type=float,
        default=30.0,
        help="ROI 선택 각도 임계값 (도 단위, 기본값: 30도)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="디버깅 모드 활성화: 계산값 출력 및 시각화 개선",
    )
    # ROI 매핑은 최단거리 방식으로 고정 (추가 설정 없음)

    args = parser.parse_args()

    if args.output is None:
        input_filename = os.path.basename(args.source)
        input_name, _ = os.path.splitext(input_filename)
        os.makedirs("Results", exist_ok=True)
        args.output = os.path.join(
            "Results",
            f"roi_gaze360_{input_name}.mp4",
        )

    tracker = RoIGazeTrackerGaze360(
        yolov8_model_path=args.yolo_model,
        gaze360_model_path=args.gaze_model,
        sixdrepnet_model_path=args.sixdrepnet_model,
        gpu_id=args.gpu,
        min_frames_threshold=args.min_frames,
        gaze_distance_k=(args.gaze_distance_k if args.gaze_distance_k is not None else args.gaze_distance),
        sixd_case=args.sixd_case,
        vector_weight=args.vector_weight,
        eye_mode=args.eye_mode,
        yolo_face_model_path=args.yolo_face_model,
        yaw_threshold_both=args.yaw_threshold_both,
        yaw_threshold_single=args.yaw_threshold_single,
        pitch_threshold_both=args.pitch_threshold_both,
        pitch_threshold_single=args.pitch_threshold_single,
        prediction_interval=args.prediction_interval,
        smoothing_window=args.smoothing_window,
        show_gaze_ray=args.show_ray,
        use_depth=args.use_depth,
        depth_model_type=args.depth_model_type,
        ray_max_len=args.ray_max_len,
        ray_shrink_px=args.ray_shrink_px,
        ray_angle_thresh=args.ray_angle_thresh,
        interactive_face_center=args.interactive_face_center,
    )

    # ROI 평면 설정 (CLI가 설정되면 config보다 우선 적용)
    plane_origin = RoIGazeTrackerGaze360.parse_vec3(args.roi_plane_origin)
    plane_normal = RoIGazeTrackerGaze360.parse_vec3(args.roi_plane_normal)
    if args.roi_plane_z is not None:
        # Gaze360에서 전방은 -Z 이므로, 사용자가 입력한 +z 거리를 음수로 배치
        plane_origin = np.array([0.0, 0.0, -abs(float(args.roi_plane_z))], dtype=float)
        plane_normal = np.array([0.0, 0.0, -1.0], dtype=float)

    if plane_normal is not None:
        norm = float(np.linalg.norm(plane_normal))
        if norm > 1e-6:
            plane_normal = plane_normal / norm
        else:
            plane_normal = None

    if plane_origin is not None and plane_normal is not None:
        tracker.roi_plane_origin = plane_origin
        tracker.roi_plane_normal = plane_normal
    
    # ROI 각도 임계값 설정
    tracker.roi_angle_threshold_deg = args.roi_angle_threshold
    # 디버깅 모드 설정
    tracker.debug_mode = args.debug

    tracker.run_video(
        args.source,
        args.output,
        args.roi_config,
        args.save_roi_config,
        args.interactive_k,
        args.interactive_face_center,
    )


if __name__ == "__main__":
    main()
