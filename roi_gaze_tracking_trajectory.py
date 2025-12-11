#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoI 기반 시선 추적 시스템 (시선 궤적 통과 기반)
Gaze360 1차 추정 + 정상 프레임 판단 + 시선 궤적 통과 기반 ROI 선택
"""

import os
import sys
import argparse
import math
import json
from collections import deque, defaultdict, OrderedDict
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

# 기존 모듈 import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gaze360/code"))
from model import GazeLSTM
from resnet import resnet18

# 6DRepNet import
sixdrepnet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "6DRepNet")
if os.path.exists(sixdrepnet_path):
    sys.path.insert(0, sixdrepnet_path)

try:
    from sixdrepnet.model import SixDRepNet
    from sixdrepnet import utils as repnet_utils
    SIXDREPNET_AVAILABLE = True
except Exception:
    SIXDREPNET_AVAILABLE = False

from ultralytics import YOLO

# 전역 변수
data_deque = {}
tracked_objects = {}
roi_colors = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
]


class RoIGazeTrackerTrajectory:
    """시선 궤적 통과 기반 ROI 선택 시스템"""
    
    def __init__(
        self,
        yolov8_model_path: str = "weights/yolov8n.pt",
        gaze360_model_path: str = "weights/gaze360_model.pth.tar",
        sixdrepnet_model_path: str = "weights/6DRepNet_300W_LP_AFLW2000.pth",
        gpu_id: int = 0,
        min_frames_threshold: int = 5,
        gaze_distance_k: int = 300,
        prediction_interval: int = 5,
        smoothing_window: int = 3,
        show_gaze_ray: bool = True,
        trajectory_samples: int = 100,
        pass_score_threshold: float = 0.1,
        angle_weight: float = 0.2,
        pass_score_weight: float = 0.7,
        distance_weight: float = 0.1,
        debug_mode: bool = False,
    ):
        """
        Args:
            trajectory_samples: 시선 궤적 샘플링 점 개수
            pass_score_threshold: ROI 통과 점수 최소 임계값
            angle_weight: 각도 가중치
            pass_score_weight: 통과 점수 가중치
            distance_weight: 거리 가중치
        """
        self.gpu_id = gpu_id
        self.min_frames_threshold = min_frames_threshold
        self.gaze_distance_k = gaze_distance_k
        self.prediction_interval = prediction_interval
        self.smoothing_window = smoothing_window
        self.show_gaze_ray = show_gaze_ray
        self.trajectory_samples = trajectory_samples
        self.pass_score_threshold = pass_score_threshold
        self.angle_weight = angle_weight
        self.pass_score_weight = pass_score_weight
        self.distance_weight = distance_weight
        self.debug_mode = debug_mode
        
        # ROI 및 설정
        self.rois = []
        self.k_anchor_point: Optional[Tuple[float, float]] = None
        self.manual_face_center: Optional[Tuple[float, float]] = None
        
        # 디버깅 정보
        self.debug_roi_info = {}
        
        # 디바이스 설정
        if gpu_id >= 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        
        # 모델 초기화
        self.init_gaze_model(gaze360_model_path)
        self.init_pose_model(sixdrepnet_model_path)
        self.init_yolo_model(yolov8_model_path)
        
        # Gaze 데이터 저장
        self.person_gaze_data = defaultdict(lambda: {
            "current_roi": None,
            "temp_roi": None,
            "temp_frame_count": 0,
            "temp_weighted_count": 0.0,
            "last_gaze": None,
            "gaze_history": deque(maxlen=smoothing_window),
            "roi_journey": [],  # ROI 방문 순서: [(roi_id, frame_count, pass_score), ...]
            "active_rois": set(),  # 현재 프레임에서 통과 점수가 임계값 이상인 ROI들
            "last_active_rois": set(),  # 이전 프레임의 활성 ROI들
        })
    
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
        self.transformations = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Gaze360 모델 로딩 완료")
    
    def init_pose_model(self, model_path: str):
        """6DRepNet 모델 초기화"""
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
        
        # 6DRepNet용 transform 정의
        self.pose_transformations = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        print("6DRepNet 모델 로딩 완료")
    
    def init_yolo_model(self, model_path: str):
        """YOLOv8 모델 초기화"""
        print("YOLOv8 모델을 로딩하는 중...")
        self.yolo_model = YOLO(model_path)
        print("YOLOv8 모델 로딩 완료")
    
    def estimate_gaze(self, face_frames):
        """Gaze360으로 시선 추정"""
        if len(face_frames) < 7:
            return None, None, None
        
        try:
            # batch dimension 추가: (7, 3, 224, 224) -> (1, 7, 3, 224, 224)
            face_tensors = torch.stack(face_frames[-7:]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                angular_output, var = self.gaze_model(face_tensors)
            
            # angular_output shape: (1, 2) -> [yaw, pitch] in radians
            yaw_rad = angular_output[0, 0].cpu().numpy()
            pitch_rad = angular_output[0, 1].cpu().numpy()
            
            yaw = float(yaw_rad * 180.0 / np.pi)
            pitch = float(pitch_rad * 180.0 / np.pi)
            
            # Gaze360 좌표계: x=cos(pitch)*sin(yaw), y=sin(pitch), z=-cos(pitch)*cos(yaw)
            gaze_vector = np.array([
                np.cos(pitch_rad) * np.sin(yaw_rad),
                np.sin(pitch_rad),
                -np.cos(pitch_rad) * np.cos(yaw_rad)
            ])
            
            return yaw, pitch, gaze_vector
        except Exception as e:
            if self.debug_mode:
                print(f"Gaze360 추정 에러: {e}")
            return None, None, None
    
    def estimate_head_pose(self, face_crop):
        """6DRepNet으로 머리 자세 추정"""
        if self.pose_model is None:
            return None, None, None, None, None
        
        try:
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_tensor = self.pose_transformations(face_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                R_pred = self.pose_model(face_tensor)
                euler = repnet_utils.compute_euler_angles_from_rotation_matrices(R_pred)
                pitch, yaw, roll = euler[0].cpu().numpy()
            
            # 6DRepNet 좌표계 변환
            yaw_deg = float(np.degrees(yaw))
            pitch_deg = float(np.degrees(pitch))
            
            # 벡터 변환
            gaze_vector = np.array([
                -np.sin(np.radians(yaw_deg)),
                -np.sin(np.radians(pitch_deg)),
                np.cos(np.radians(pitch_deg)) * np.cos(np.radians(yaw_deg))
            ])
            
            return yaw_deg, pitch_deg, gaze_vector, "6DRepNet", 0.7
        except Exception:
            return None, None, None, None, None
    
    def is_normal_frame(self, current_gaze, last_gaze, mode):
        """
        정상 프레임 판단
        조건: 급격한 변화, 몸 방향과 모순, 불확실성, 눈 가시성
        """
        if last_gaze is None:
            return True
        
        yaw_curr, pitch_curr = current_gaze[0], current_gaze[1]
        yaw_last, pitch_last = last_gaze[0], last_gaze[1]
        
        # 급격한 변화 체크
        yaw_delta = abs(yaw_curr - yaw_last)
        pitch_delta = abs(pitch_curr - pitch_last)
        
        if yaw_delta > 35.0 or pitch_delta > 25.0:
            if self.debug_mode:
                print(f"[DEBUG] 급격한 변화 감지: yaw_delta={yaw_delta:.1f}°, pitch_delta={pitch_delta:.1f}°")
            return False
        
        # Gaze360 모드면 정상으로 판단
        if mode == "gaze360":
            return True
        
        # 6DRepNet 모드면 추가 검증 필요
        return True
    
    def point_in_polygon(self, x: float, y: float, polygon: list) -> bool:
        """점이 폴리곤 내부에 있는지 확인 (ray casting 알고리즘)"""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def point_in_rect(self, x: float, y: float, rect: list) -> bool:
        """점이 직사각형 내부에 있는지 확인"""
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def is_point_in_roi(self, x: float, y: float, roi) -> bool:
        """점이 ROI 내부에 있는지 확인"""
        roi_type = roi.get("type", "rect")
        data = roi.get("data")
        
        if roi_type == "rect":
            return self.point_in_rect(x, y, data)
        elif roi_type == "poly":
            polygon = [(float(p[0]), float(p[1])) for p in data]
            return self.point_in_polygon(x, y, polygon)
        return False
    
    def get_roi_center(self, roi):
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
    
    def select_roi_by_min_distance(
        self, face_center_x: int, face_center_y: int, gaze_vector
    ):
        """
        얼굴 중심점에서 시선 방향으로 레이저를 쏴서 ROI에 걸치는지 확인
        1. 시선 끝점이 ROI 내부에 있으면 즉시 선택
        2. 그렇지 않으면 시선 반직선에서 각 ROI 중심점까지의 최단 거리 계산
        
        원본 roi_gaze_tracking_gaze360.py의 select_roi 로직 사용
        """
        if len(self.rois) == 0:
            return None, float('inf')
        
        # 시선 방향 벡터를 2D로 변환 (x, y만 사용)
        gaze_2d = np.array([gaze_vector[0], gaze_vector[1]], dtype=float)
        gaze_2d_norm = np.linalg.norm(gaze_2d)
        
        if gaze_2d_norm < 1e-6:
            return None, float('inf')
        
        gaze_2d_normalized = gaze_2d / gaze_2d_norm
        
        # 얼굴 중심점
        face_center = np.array([float(face_center_x), float(face_center_y)])
        
        # 1단계: 시선 끝점이 ROI 내부에 있는지 먼저 확인 (우선순위 높음)
        ray_length = self.gaze_distance_k if self.gaze_distance_k > 0 else 1000.0
        endpoint = face_center + gaze_2d_normalized * float(ray_length)
        for roi in self.rois:
            if self.is_point_in_roi(endpoint[0], endpoint[1], roi):
                if self.debug_mode:
                    print(f"[DEBUG] 시선 끝점이 ROI {roi.get('roi_id', 0)} 내부에 있음")
                return roi.get("roi_id", 0), 0.0
        
        # 2단계: 시선 반직선에서 각 ROI 중심점까지의 최단 거리 계산
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
            t = np.dot(v_to_roi, gaze_2d_normalized)
            
            # t < 0이면 시선 반대 방향에 있는 ROI (시선 반직선 상에 없음)
            if t < 0:
                # 시선 반대 방향이면 유클리드 거리 사용
                distance = np.linalg.norm(v_to_roi)
            else:
                # 시선 반직선 위의 가장 가까운 점 계산
                closest_point_on_ray = face_center + t * gaze_2d_normalized
                
                # ROI 중심점에서 시선 반직선까지의 최단 거리
                distance = np.linalg.norm(roi_center_point - closest_point_on_ray)
            
            if distance < min_distance:
                min_distance = distance
                best_roi_id = roi_id
        
        # 최단 거리가 너무 크면 매핑하지 않음
        max_distance_threshold = 5000.0  # 픽셀 단위
        if min_distance > max_distance_threshold:
            return None, min_distance
        
        # best_roi_id가 None이면 가장 가까운 ROI를 선택
        if best_roi_id is None:
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
        
        if self.debug_mode:
            print(f"[DEBUG] Selected ROI: {best_roi_id} (distance: {min_distance:.1f}px)")
        
        return best_roi_id, min_distance
    
    
    def smooth_gaze_result(self, gaze_history, new_result):
        """시선 결과 스무딩"""
        if self.smoothing_window == 0 or len(gaze_history) == 0:
            return new_result
        
        gaze_history.append(new_result)
        if len(gaze_history) < 2:
            return new_result
        
        # 가중 평균
        weights = np.linspace(0.5, 1.0, len(gaze_history))
        weights = weights / weights.sum()
        
        yaw_avg = sum(r[0] * w for r, w in zip(gaze_history, weights))
        pitch_avg = sum(r[1] * w for r, w in zip(gaze_history, weights))
        
        # 벡터 재계산
        gaze_vector = np.array([
            -np.sin(np.radians(yaw_avg)),
            -np.sin(np.radians(pitch_avg)),
            np.cos(np.radians(pitch_avg)) * np.cos(np.radians(yaw_avg))
        ])
        
        return (yaw_avg, pitch_avg, gaze_vector, new_result[3], new_result[4])
    
    def update_gaze_journey(self, person_id: int, current_roi, weight: float = 1.0, 
                            active_rois: list = None, frame_count: int = 0):
        """Gaze journey 업데이트 - ROI 전환 추적"""
        person_data = self.person_gaze_data[person_id]
        
        # 기존 로직: 현재 ROI 확정
        if current_roi == person_data["temp_roi"]:
            person_data["temp_frame_count"] += 1
            person_data["temp_weighted_count"] += weight
            
            if person_data["temp_weighted_count"] >= self.min_frames_threshold:
                # ROI가 확정되면 journey에 추가 (중복 방지)
                confirmed_roi = current_roi
                if confirmed_roi is not None:
                    # 마지막 ROI와 다르면 journey에 추가
                    if (not person_data["roi_journey"] or 
                        person_data["roi_journey"][-1][0] != confirmed_roi):
                        person_data["roi_journey"].append((
                            confirmed_roi,
                            frame_count,
                            0.0,  # pass_score (사용 안 함)
                            0.0,  # angle_diff (사용 안 함)
                            0.0   # distance (사용 안 함)
                        ))
                        if self.debug_mode:
                            print(f"[DEBUG] ROI {confirmed_roi} 확정 및 journey 추가 (frame={frame_count})")
                
                person_data["current_roi"] = confirmed_roi
        else:
            person_data["temp_roi"] = current_roi
            person_data["temp_frame_count"] = 1
            person_data["temp_weighted_count"] = weight
    
    def load_rois_from_config(self, config_path: str) -> bool:
        """ROI 설정 파일 로드"""
        if not os.path.exists(config_path):
            return False
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if "rois" in config:
                self.rois = config["rois"]
                print(f"Config 파일에서 {len(self.rois)}개의 ROI를 로드했습니다:")
                for roi in self.rois:
                    roi_type = roi.get("type", "rect")
                    roi_id = roi.get("roi_id", 0)
                    if roi_type == "poly":
                        points = len(roi.get("data", []))
                        print(f"  RoI {roi_id} ({roi_type}): {points} points")
                    else:
                        print(f"  RoI {roi_id} ({roi_type})")
                return True
        except Exception as e:
            print(f"ROI 설정 로드 실패: {e}")
        
        return False
    
    def draw_trajectory_heatmap(self, img, trajectory_points, roi_id, pass_score):
        """시선 궤적 히트맵 그리기"""
        if pass_score <= 0:
            return
        
        # ROI 영역에 히트맵 오버레이
        roi = next((r for r in self.rois if r.get("roi_id") == roi_id), None)
        if roi is None:
            return
        
        roi_type = roi.get("type", "rect")
        roi_data = roi.get("data")
        
        # 히트맵 색상 (통과 점수에 따라)
        intensity = int(pass_score * 255)
        heat_color = (0, intensity, 255 - intensity)  # 파란색 -> 빨간색
        
        overlay = img.copy()
        if roi_type == "rect":
            x1, y1, x2, y2 = roi_data
            cv2.rectangle(overlay, (x1, y1), (x2, y2), heat_color, -1)
        elif roi_type == "poly":
            pts = np.array(roi_data, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], heat_color)
        
        # 알파 블렌딩
        alpha = pass_score * 0.3  # 최대 30% 투명도
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    def draw_trajectory_line(self, img, trajectory_points):
        """시선 궤적 점 그리기 (선 연결 없이 점만)"""
        if len(trajectory_points) == 0:
            return
        
        points = np.array(trajectory_points, np.int32)
        
        # 모든 궤적 점을 작은 원으로 표시
        for i, pt in enumerate(points):
            # 샘플링 간격에 따라 점 크기 조정 (중간 점은 작게, 시작/끝은 크게)
            if i == 0:
                # 시작점 (k점) - 마젠타, 이미 별도로 그려짐
                continue
            elif i == len(points) - 1:
                # 끝점 - 녹색, 크게
                cv2.circle(img, tuple(pt), 6, (0, 255, 0), -1)
                cv2.circle(img, tuple(pt), 6, (255, 255, 255), 1)
            else:
                # 중간 점들 - 흰색, 작게
                if i % 10 == 0:  # 10개마다만 그려서 너무 많지 않게
                    cv2.circle(img, tuple(pt), 2, (255, 255, 255), -1)
    
    def draw_rois_with_heatmap(self, img, active_rois=None, person_roi_mapping=None, trajectory_points=None):
        """ROI 그리기 (간단한 버전)"""
        if active_rois is None:
            active_rois = set()
        if person_roi_mapping is None:
            person_roi_mapping = {}
        
        # ROI 그리기
        for roi in self.rois:
            roi_type = roi.get("type", "rect")
            roi_id = roi.get("roi_id", 0)
            data = roi.get("data")
            
            # 활성 ROI 색상
            if roi_id in person_roi_mapping and len(person_roi_mapping[roi_id]) > 0:
                _, color = person_roi_mapping[roi_id][0]
            else:
                color = roi_colors[(roi_id - 1) % len(roi_colors)]
            
            # 선택된 ROI 강조 표시
            is_active = roi_id in active_rois
            thickness = 6 if is_active else 2
            
            if roi_type == "rect":
                x1, y1, x2, y2 = data
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                
                # 라벨
                if is_active:
                    label = f"RoI {roi_id} [SELECTED]"
                    if roi_id in self.debug_roi_info:
                        info = self.debug_roi_info[roi_id]
                        label += f" pass:{info['pass_score']:.2f}"
                else:
                    label = f"RoI {roi_id}"
                
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            elif roi_type == "poly":
                pts = np.array(data, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, color, thickness)
                
                label_x, label_y = data[0]
                if is_active:
                    label = f"RoI {roi_id} [SELECTED]"
                    if roi_id in self.debug_roi_info:
                        info = self.debug_roi_info[roi_id]
                        label += f" pass:{info['pass_score']:.2f}"
                else:
                    label = f"RoI {roi_id}"
                
                cv2.putText(img, label, (label_x, label_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 디버깅 정보 표시
            if roi_id in self.debug_roi_info and is_active:
                info = self.debug_roi_info[roi_id]
                debug_text = f"pass:{info['pass_score']:.2f} ∠{info['angle_diff_deg']:.1f}° d:{info['distance']:.0f}"
                if roi_type == "rect":
                    cv2.putText(img, debug_text, (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    poly_bottom = max([pt[1] for pt in data])
                    poly_left = min([pt[0] for pt in data])
                    cv2.putText(img, debug_text, (poly_left, poly_bottom + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="시선 궤적 통과 기반 ROI 추적")
    parser.add_argument("--source", type=str, required=True, help="입력 비디오 경로")
    parser.add_argument("--output", type=str, default=None, help="출력 비디오 경로")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU ID (-1: CPU)")
    parser.add_argument("--roi_config", type=str, default=None, help="ROI 설정 파일")
    parser.add_argument("--min_frames", type=int, default=20, help="최소 프레임 수")
    parser.add_argument("--trajectory_samples", type=int, default=100, help="궤적 샘플링 점 개수")
    parser.add_argument("--pass_score_threshold", type=float, default=0.1, help="통과 점수 임계값")
    parser.add_argument("--gaze_distance_k", type=int, default=1200, help="시선 거리 k")
    parser.add_argument("--interactive_k", action="store_true", help="k점 수동 설정")
    parser.add_argument("--interactive_face_center", action="store_true", help="얼굴 중심 수동 설정")
    parser.add_argument("--debug", action="store_true", help="디버깅 모드")
    
    args = parser.parse_args()
    
    # 출력 경로 설정
    if args.output is None:
        input_name = os.path.splitext(os.path.basename(args.source))[0]
        os.makedirs("Results", exist_ok=True)
        args.output = f"Results/{input_name}_trajectory.mp4"
    
    # Tracker 초기화
    tracker = RoIGazeTrackerTrajectory(
        gpu_id=args.gpu,
        min_frames_threshold=args.min_frames,
        gaze_distance_k=args.gaze_distance_k,
        trajectory_samples=args.trajectory_samples,
        pass_score_threshold=args.pass_score_threshold,
        debug_mode=args.debug,
    )
    
    # ROI 로드
    if args.roi_config:
        tracker.load_rois_from_config(args.roi_config)
    
    # k점 설정
    if args.interactive_k:
        cap_temp = cv2.VideoCapture(args.source)
        ret, frame_temp = cap_temp.read()
        cap_temp.release()
        if ret:
            clicked = {"pt": None}
            def cb(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    clicked["pt"] = (x, y)
            cv2.namedWindow("Set k point")
            cv2.setMouseCallback("Set k point", cb)
            h, w = frame_temp.shape[:2]
            center = (w * 0.5, h * 0.5)
            while True:
                disp = frame_temp.copy()
                cv2.circle(disp, (int(center[0]), int(center[1])), 5, (0, 255, 255), -1)
                if clicked["pt"]:
                    cv2.circle(disp, clicked["pt"], 6, (0, 0, 255), -1)
                    cv2.line(disp, (int(center[0]), int(center[1])), clicked["pt"], (0, 255, 0), 2)
                cv2.putText(disp, "Click k point, Enter=apply, ESC=skip", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Set k point", disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (13, 10):
                    break
                if key == 27:
                    clicked["pt"] = None
                    break
            cv2.destroyAllWindows()
            if clicked["pt"]:
                tracker.k_anchor_point = (float(clicked["pt"][0]), float(clicked["pt"][1]))
                dx = clicked["pt"][0] - center[0]
                dy = clicked["pt"][1] - center[1]
                k = float(math.hypot(dx, dy))
                if k < 50:
                    k = max(h, w) * 0.4
                tracker.gaze_distance_k = int(k)
                print(f"k점 앵커: {tracker.k_anchor_point}, gaze_distance_k={tracker.gaze_distance_k}")
    
    # 얼굴 중심 설정
    if args.interactive_face_center:
        cap_temp = cv2.VideoCapture(args.source)
        ret, frame_temp = cap_temp.read()
        cap_temp.release()
        if ret:
            clicked = {"pt": None}
            def cb(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    clicked["pt"] = (x, y)
            cv2.namedWindow("Set face center")
            cv2.setMouseCallback("Set face center", cb)
            while True:
                disp = frame_temp.copy()
                if clicked["pt"]:
                    cv2.circle(disp, clicked["pt"], 8, (0, 255, 0), -1)
                cv2.putText(disp, "Click face center, Enter=apply, ESC=skip", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Set face center", disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (13, 10):
                    break
                if key == 27:
                    clicked["pt"] = None
                    break
            cv2.destroyAllWindows()
            if clicked["pt"]:
                tracker.manual_face_center = (float(clicked["pt"][0]), float(clicked["pt"][1]))
                print(f"수동 얼굴 중심: {tracker.manual_face_center}")
    
    # 비디오 처리
    print(f"비디오 처리를 시작합니다: {args.source}")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {args.source}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_count = 0
    active_rois = set()
    person_roi_mapping = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # YOLO로 사람 검출
        results = tracker.yolo_model.track(frame, persist=True, classes=[0])
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else None
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                track_id = int(track_ids[i]) if track_ids is not None else 0
                
                # 얼굴 영역 추출
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size == 0:
                    continue
                
                # 얼굴 중심 계산
                if tracker.manual_face_center:
                    face_center_x, face_center_y = int(tracker.manual_face_center[0]), int(tracker.manual_face_center[1])
                else:
                    face_center_x = x1 + (x2 - x1) // 2
                    face_center_y = y1 + int((y2 - y1) * 0.3)  # 상단 30%
                
                # 얼굴 크롭 (Gaze360용)
                face_crop = person_roi[:int((y2-y1)*0.4), :]
                if face_crop.size == 0:
                    continue
                
                # Gaze360 1차 추정
                try:
                    face_tensor = tracker.transformations(Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)))
                    if track_id not in tracked_objects:
                        tracked_objects[track_id] = {"face_frames": deque(maxlen=7)}
                    tracked_objects[track_id]["face_frames"].append(face_tensor)
                    
                    if len(tracked_objects[track_id]["face_frames"]) >= 7:
                        # Gaze360 추정
                        yaw, pitch, gaze_vector = tracker.estimate_gaze(list(tracked_objects[track_id]["face_frames"]))
                        
                        if yaw is not None:
                            # 정상 프레임 판단
                            person_data = tracker.person_gaze_data[track_id]
                            last_gaze = person_data.get("last_gaze")
                            
                            is_normal = tracker.is_normal_frame((yaw, pitch), last_gaze, "gaze360")
                            
                            if not is_normal and tracker.pose_model:
                                # 이상 프레임 → 6DRepNet 보정
                                sixd_yaw, sixd_pitch, sixd_gaze_vector, _, _ = tracker.estimate_head_pose(face_crop)
                                if sixd_yaw is not None:
                                    yaw, pitch, gaze_vector = sixd_yaw, sixd_pitch, sixd_gaze_vector
                                    mode = "6DRepNet"
                                else:
                                    mode = "gaze360"
                            else:
                                mode = "gaze360"
                            
                            # 스무딩
                            raw_result = (yaw, pitch, gaze_vector, mode, 1.0)
                            final_result = tracker.smooth_gaze_result(
                                person_data["gaze_history"], raw_result
                            )
                            yaw, pitch, gaze_vector, mode, weight = final_result
                            
                            # ROI 선택 (얼굴 중심에서 레이저 기반)
                            mapped_gaze_vector = np.array([-gaze_vector[0], -gaze_vector[1], gaze_vector[2]])
                            current_roi, _ = tracker.select_roi_by_min_distance(
                                face_center_x, face_center_y, mapped_gaze_vector
                            )
                            
                            # Gaze journey 업데이트 (ROI 전환 추적)
                            tracker.update_gaze_journey(track_id, current_roi, weight, 
                                                       None, frame_count)
                            
                            # 활성 ROI 업데이트
                            confirmed_roi = tracker.person_gaze_data[track_id].get("current_roi")
                            if confirmed_roi is not None:
                                active_rois.add(confirmed_roi)
                                if confirmed_roi not in person_roi_mapping:
                                    person_roi_mapping[confirmed_roi] = []
                                person_roi_mapping[confirmed_roi].append((track_id, (255, 0, 0)))
                            
                            # k점 설정 (없으면 face_center 사용)
                            if tracker.k_anchor_point:
                                k_point = np.array([float(tracker.k_anchor_point[0]), float(tracker.k_anchor_point[1])])
                            else:
                                # k점이 없으면 face_center를 k점으로 사용
                                k_point = np.array([float(face_center_x), float(face_center_y)])
                                if tracker.gaze_distance_k <= 0:
                                    h, w = frame.shape[:2]
                                    tracker.gaze_distance_k = int(max(h, w) * 0.35)
                            
                            # 얼굴 중심 표시
                            face_center_display = (face_center_x, face_center_y)
                            cv2.circle(frame, face_center_display, 8, (0, 255, 0), -1)
                            cv2.circle(frame, face_center_display, 8, (255, 255, 255), 2)
                            
                            # 시선 레이저 시각화 (얼굴 중심에서)
                            if tracker.show_gaze_ray:
                                gaze_2d = np.array([-gaze_vector[0], -gaze_vector[1]])
                                norm = np.linalg.norm(gaze_2d)
                                if norm > 1e-6:
                                    gaze_2d_normalized = gaze_2d / norm
                                    ray_length = tracker.gaze_distance_k if tracker.gaze_distance_k > 0 else 1000.0
                                    
                                    # 레이저 끝점
                                    ray_end = (face_center_display[0] + int(gaze_2d_normalized[0] * ray_length), 
                                              face_center_display[1] + int(gaze_2d_normalized[1] * ray_length))
                                    
                                    # 레이저 선 그리기
                                    arrow_color = (0, 255, 0) if confirmed_roi else (0, 255, 255)
                                    arrow_thickness = 3 if confirmed_roi else 2
                                    cv2.arrowedLine(frame, face_center_display, ray_end, 
                                                   arrow_color, arrow_thickness, tipLength=0.15)
                                    
                                    # yaw/pitch 표시
                                    yaw_pitch_text = f"Y:{yaw:.1f}° P:{pitch:.1f}°"
                                    cv2.putText(frame, yaw_pitch_text, 
                                               (face_center_display[0] + 15, face_center_display[1] - 10),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            # ROI 그리기
                            tracker.draw_rois_with_heatmap(frame, active_rois, person_roi_mapping, [])
                            
                            # 라벨 표시
                            roi_label = f"RoI:{confirmed_roi}" if confirmed_roi else "No RoI"
                            label = f"ID:{track_id} {mode} {roi_label}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                            
                            # ROI Journey 표시 (최근 5개)
                            journey = person_data.get("roi_journey", [])
                            if journey:
                                journey_text = "Journey: " + " → ".join([f"R{roi_id}" for roi_id, _, _, _, _ in journey[-5:]])
                                cv2.putText(frame, journey_text, (x1, y2 + 20),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            
                            person_data["last_gaze"] = (float(yaw), float(pitch))
                except Exception as e:
                    if tracker.debug_mode:
                        print(f"Error processing person {track_id}: {e}")
        
        out.write(frame)
        if frame_count % 50 == 0:
            print(f"프레임 처리 중: {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    print(f"비디오 처리 완료: {args.output}")
    
    # ROI Journey 출력
    print("\n=== ROI 방문 순서 (Journey) ===")
    for person_id, person_data in tracker.person_gaze_data.items():
        journey = person_data.get("roi_journey", [])
        if journey:
            print(f"\nPerson {person_id}의 ROI 방문 순서:")
            for i, (roi_id, frame_num, _, _, _) in enumerate(journey, 1):
                print(f"  {i}. ROI {roi_id} (프레임 {frame_num})")
        else:
            print(f"\nPerson {person_id}: ROI 방문 기록 없음")


if __name__ == "__main__":
    main()

