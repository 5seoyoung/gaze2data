#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoI 기반 시선 추적 시스템
6DRepNet + YOLOv8 추적을 이용한 간접 시선 추정 및 RoI 응시 분석
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.backends import cudnn
import pandas as pd
from collections import deque, defaultdict
from torchvision import transforms
from PIL import Image

# 6DRepNet 관련 import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '6DRepNet'))
from sixdrepnet.model import SixDRepNet
from sixdrepnet import utils
from face_detection import RetinaFace

# YOLOv8 관련 import
from ultralytics import YOLO

# 전역 변수
data_deque = {}
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
next_id = 1
tracked_objects = {}

# RoI 설정 전역 변수
rois = []  # [(x1, y1, x2, y2, roi_id), ...]
current_roi = None
drawing = False
roi_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)
]


def mouse_callback(event, x, y, flags, param):
    """RoI 설정을 위한 마우스 콜백 함수"""
    global rois, current_roi, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_roi = [x, y, x, y]
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_roi[2] = x
            current_roi[3] = y
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_roi[2] = x
        current_roi[3] = y
        
        # RoI 추가 (최소 크기 체크)
        x1, y1, x2, y2 = current_roi
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            roi_id = len(rois) + 1
            rois.append((x1, y1, x2, y2, roi_id))
            print(f"RoI {roi_id} 추가됨: ({x1}, {y1}) - ({x2}, {y2})")
        
        current_roi = None


class RoIGazeTracker:
    def __init__(self, 
                 yolov8_model_path="weights/yolov8n.pt",
                 sixdrepnet_model_path="weights/6DRepNet_300W_LP_AFLW2000.pth",
                 gpu_id=0,
                 min_frames_threshold=5,
                 gaze_distance_k=300):
        """
        RoI 기반 시선 추적 시스템 초기화
        
        Args:
            yolov8_model_path: YOLOv8 모델 경로
            sixdrepnet_model_path: 6DRepNet 모델 경로
            gpu_id: GPU ID (CPU 사용시 -1)
            min_frames_threshold: RoI 응시 판단을 위한 최소 프레임 수
            gaze_distance_k: 시선 끝점 계산을 위한 거리 상수 (픽셀)
        """
        self.gpu_id = gpu_id
        self.min_frames_threshold = min_frames_threshold
        self.gaze_distance_k = gaze_distance_k
        
        # CUDA 사용 가능 여부 확인
        if gpu_id >= 0 and torch.cuda.is_available():
            self.device = torch.device('cuda:%d' % gpu_id)
            print(f"GPU {gpu_id} 사용")
        else:
            self.device = torch.device('cpu')
            print("CPU 사용")
        
        # 6DRepNet 모델 초기화
        self.init_pose_model(sixdrepnet_model_path)
        
        # YOLOv8 모델 초기화
        self.init_detection_model(yolov8_model_path)
        
        # 추적 시스템 초기화
        self.init_tracker()
        
        # 이미지 전처리 변환
        self.transformations = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 얼굴 탐지기 초기화
        face_gpu_id = -1 if self.device.type == 'cpu' else gpu_id
        self.face_detector = RetinaFace(gpu_id=face_gpu_id)
        
        # RoI 설정
        self.rois = []
        
        # 각 사람별 시선 데이터 저장
        # {person_id: {'current_roi': roi_id or None, 'frame_count': int, 'gaze_journey': [(roi_id, frame_count), ...]}}
        self.person_gaze_data = defaultdict(lambda: {
            'current_roi': None, 
            'frame_count': 0, 
            'gaze_journey': [],
            'temp_roi': None,  # 임시로 보고 있는 RoI
            'temp_frame_count': 0  # 임시 RoI를 본 프레임 수
        })
        
    def init_pose_model(self, model_path):
        """6DRepNet 모델 초기화"""
        print("6DRepNet 모델을 로딩하는 중...")
        self.pose_model = SixDRepNet(
            backbone_name='RepVGG-B1g2',
            backbone_file='',
            deploy=True,
            pretrained=False
        )
        
        # 가중치 로드
        saved_state_dict = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in saved_state_dict:
            self.pose_model.load_state_dict(saved_state_dict['model_state_dict'])
        else:
            self.pose_model.load_state_dict(saved_state_dict)
        
        self.pose_model.to(self.device)
        self.pose_model.eval()
        print("6DRepNet 모델 로딩 완료")
        
    def init_detection_model(self, model_path):
        """YOLOv8 모델 초기화"""
        print("YOLOv8 모델을 로딩하는 중...")
        self.detection_model = YOLO(model_path)
        print("YOLOv8 모델 로딩 완료")
        
    def init_tracker(self):
        """간단한 IoU 기반 추적기 초기화"""
        global next_id, tracked_objects
        print("IoU 기반 추적기를 초기화하는 중...")
        next_id = 1
        tracked_objects = {}
        print("추적기 초기화 완료")
        
    def compute_iou(self, box1, box2):
        """두 바운딩 박스 간의 IoU 계산"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # 교집합 영역 계산
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # 각 박스의 면적 계산
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        # 합집합 면적 계산
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0.0
            
        return intersection_area / union_area
        
    def simple_track(self, detections, frame_count):
        """간단한 IoU 기반 추적"""
        global next_id, tracked_objects
        
        # 기존 추적 객체 업데이트 (5프레임 이상 보이지 않으면 제거)
        to_remove = []
        for obj_id, obj_data in tracked_objects.items():
            if frame_count - obj_data['last_seen'] > 5:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del tracked_objects[obj_id]
            if obj_id in data_deque:
                del data_deque[obj_id]
        
        # 새로운 탐지에 대해 기존 추적 객체와 매칭
        matched_detections = set()
        matched_tracks = set()
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            best_iou = 0.3  # IoU 임계값
            best_match = None
            
            for obj_id, obj_data in tracked_objects.items():
                if obj_id in matched_tracks:
                    continue
                    
                iou = self.compute_iou([x1, y1, x2, y2], obj_data['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = obj_id
            
            if best_match is not None:
                # 기존 추적 객체 업데이트
                tracked_objects[best_match]['bbox'] = [x1, y1, x2, y2]
                tracked_objects[best_match]['last_seen'] = frame_count
                matched_detections.add(i)
                matched_tracks.add(best_match)
            else:
                # 새로운 추적 객체 생성
                tracked_objects[next_id] = {
                    'bbox': [x1, y1, x2, y2],
                    'last_seen': frame_count
                }
                next_id += 1
                matched_detections.add(i)
        
        # 추적 결과 반환
        tracks = []
        for obj_id, obj_data in tracked_objects.items():
            if obj_data['last_seen'] == frame_count:  # 현재 프레임에서 업데이트된 객체만
                x1, y1, x2, y2 = obj_data['bbox']
                tracks.append([x1, y1, x2, y2, obj_id, 0])  # [x1, y1, x2, y2, track_id, class]
        
        return tracks
        
    def compute_color_for_labels(self, label):
        """클래스에 따른 고정 색상 계산"""
        if label == 0:  # person
            color = (85, 45, 255)
        else:
            color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)
        
    def draw_border(self, img, pt1, pt2, color, thickness, r, d):
        """둥근 모서리 박스 그리기"""
        x1, y1 = pt1
        x2, y2 = pt2
        # Top left
        cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
        
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
        
        cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
        cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
        cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
        cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)
        
        return img
        
    def UI_box(self, x, img, color=None, label=None, line_thickness=None):
        """이미지에 바운딩 박스 그리기"""
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        color = color or [np.random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            
            img = self.draw_border(img, (c1[0], c1[1] - t_size[1] - 3), 
                                 (c1[0] + t_size[0], c1[1] + 3), color, 1, 8, 2)
            
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, 
                       [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                       
    def estimate_pose(self, face_roi):
        """얼굴 영역에서 포즈 추정"""
        try:
            # 이미지 전처리
            face_img = Image.fromarray(face_roi)
            face_img = face_img.convert('RGB')
            face_img = self.transformations(face_img)
            face_img = torch.Tensor(face_img[None, :]).to(self.device)
            
            # 포즈 추정
            with torch.no_grad():
                R_pred = self.pose_model(face_img)
                euler = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
                pitch = float(euler[:, 0].cpu().numpy()[0])
                yaw = float(euler[:, 1].cpu().numpy()[0])
                roll = float(euler[:, 2].cpu().numpy()[0])
                
            return pitch, yaw, roll
        except Exception as e:
            print(f"포즈 추정 오류: {e}")
            return None, None, None
    
    def compute_gaze_endpoint(self, face_center_x, face_center_y, yaw, pitch):
        """
        머리 포즈로부터 시선 끝점 계산
        
        Gaze360과 동일한 좌표계 사용 (일관성):
        - integrated_gaze_tracking.py의 draw_gaze_arrow 방식과 통일
        
        수식:
        gaze_x = cos(pitch) * sin(yaw)
        gaze_y = sin(pitch)
        dx = -k * gaze_x
        dy = -k * gaze_y
        """
        # 각도를 라디안으로 변환
        yaw_rad = yaw * np.pi / 180
        pitch_rad = pitch * np.pi / 180
        
        # 시선 방향 벡터 계산 (Gaze360과 동일한 방식)
        gaze_x = np.cos(pitch_rad) * np.sin(yaw_rad)
        gaze_y = np.sin(pitch_rad)
        gaze_z = -np.cos(pitch_rad) * np.cos(yaw_rad)
        
        # ✅ Gaze360과 동일하게 음수 부호 적용
        dx = -self.gaze_distance_k * gaze_x
        dy = -self.gaze_distance_k * gaze_y
        
        x_end = face_center_x + dx
        y_end = face_center_y + dy
        
        return int(x_end), int(y_end), (gaze_x, gaze_y, gaze_z)
    
    def check_point_in_roi(self, x, y):
        """점이 어떤 RoI 안에 있는지 확인"""
        for roi in self.rois:
            x1, y1, x2, y2, roi_id = roi
            if x1 <= x <= x2 and y1 <= y <= y2:
                return roi_id
        return None
    
    def update_gaze_journey(self, person_id, current_roi):
        """
        사람의 gaze_journey 업데이트
        N프레임 이상 지속되어야 기록됨
        """
        person_data = self.person_gaze_data[person_id]
        
        # 현재 RoI가 임시 RoI와 같으면 카운트 증가
        if current_roi == person_data['temp_roi']:
            person_data['temp_frame_count'] += 1
            
            # 임계값 이상이면 정식으로 기록
            if person_data['temp_frame_count'] >= self.min_frames_threshold:
                # 이전 RoI와 다르면 새로운 journey 추가
                if current_roi != person_data['current_roi']:
                    # 이전 RoI 마무리
                    if person_data['current_roi'] is not None and person_data['frame_count'] > 0:
                        person_data['gaze_journey'].append(
                            (person_data['current_roi'], person_data['frame_count'])
                        )
                    
                    # 새로운 RoI 시작
                    person_data['current_roi'] = current_roi
                    person_data['frame_count'] = person_data['temp_frame_count']
                else:
                    # 같은 RoI 지속
                    person_data['frame_count'] += 1
        else:
            # 다른 RoI로 변경됨
            # 이전 임시 RoI가 임계값을 넘었다면 기록
            if person_data['temp_frame_count'] >= self.min_frames_threshold:
                if person_data['current_roi'] is not None and person_data['frame_count'] > 0:
                    person_data['gaze_journey'].append(
                        (person_data['current_roi'], person_data['frame_count'])
                    )
                person_data['current_roi'] = None
                person_data['frame_count'] = 0
            
            # 새로운 임시 RoI 설정
            person_data['temp_roi'] = current_roi
            person_data['temp_frame_count'] = 1
    
    def finalize_gaze_journey(self):
        """모든 사람의 gaze_journey 마무리"""
        for person_id, person_data in self.person_gaze_data.items():
            if person_data['current_roi'] is not None and person_data['frame_count'] > 0:
                person_data['gaze_journey'].append(
                    (person_data['current_roi'], person_data['frame_count'])
                )
                
    def draw_boxes_with_gaze(self, img, bbox, names, object_id, identities=None, offset=(0, 0)):
        """바운딩 박스와 시선 정보를 그리기"""
        global data_deque
        
        height, width, _ = img.shape
        active_rois = set()  # 현재 프레임에서 누군가가 보고 있는 RoI들
        
        # 추적되지 않는 객체의 버퍼 제거
        for key in list(data_deque):
            if key not in identities:
                data_deque.pop(key)
                
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            
            # 중심점 찾기
            center = (int((x2 + x1) / 2), int((y2 + y1) / 2))
            
            # 객체 ID 가져오기
            id = int(identities[i]) if identities is not None else 0
            
            # 새 객체에 대한 새 버퍼 생성
            if id not in data_deque:
                data_deque[id] = deque(maxlen=64)
                
            color = self.compute_color_for_labels(object_id[i])
            obj_name = names[object_id[i]]
            
            # 사람인 경우에만 포즈 및 시선 추정
            if object_id[i] == 0:  # person class
                # 얼굴 영역 추출 (바운딩 박스 내에서)
                face_roi = img[y1:y2, x1:x2]
                if face_roi.size > 0:
                    # 얼굴 탐지
                    faces = self.face_detector(face_roi)
                    if len(faces) > 0:
                        # 가장 높은 점수의 얼굴 선택
                        best_face = max(faces, key=lambda x: x[2])
                        if best_face[2] > 0.5:  # 신뢰도 임계값
                            box_face, landmarks, score = best_face
                            fx_min, fy_min, fx_max, fy_max = [int(coord) for coord in box_face]
                            
                            # 얼굴 영역을 원본 이미지 좌표로 변환
                            fx_min += x1
                            fy_min += y1
                            fx_max += x1
                            fy_max += y1
                            
                            # 얼굴 영역 추출
                            face_crop = img[fy_min:fy_max, fx_min:fx_max]
                            if face_crop.size > 0:
                                # 포즈 추정
                                pitch, yaw, roll = self.estimate_pose(face_crop)
                                
                                if pitch is not None and yaw is not None:
                                    # 얼굴 중심 계산
                                    face_center_x = fx_min + (fx_max - fx_min) // 2
                                    face_center_y = fy_min + (fy_max - fy_min) // 2
                                    
                                    # 시선 끝점 계산
                                    gaze_end_x, gaze_end_y, gaze_vector = self.compute_gaze_endpoint(
                                        face_center_x, face_center_y, yaw, pitch
                                    )
                                    
                                    # 시선 끝점이 어느 RoI에 있는지 확인
                                    current_roi = self.check_point_in_roi(gaze_end_x, gaze_end_y)
                                    
                                    # gaze_journey 업데이트
                                    self.update_gaze_journey(id, current_roi)
                                    
                                    # 현재 보고 있는 RoI 추가
                                    if current_roi is not None:
                                        active_rois.add(current_roi)
                                    
                                    # 얼굴 중심점 표시 (파란색)
                                    cv2.circle(img, (face_center_x, face_center_y), 6, (255, 0, 0), -1)
                                    cv2.circle(img, (face_center_x, face_center_y), 8, (255, 255, 255), 2)
                                    
                                    # RoI 안에 있는지에 따라 색상 변경
                                    if current_roi is not None:
                                        # RoI 안에 있으면 빨간색
                                        arrow_color = (0, 0, 255)
                                        endpoint_color = (0, 0, 255)
                                        endpoint_outer_color = (255, 255, 0)
                                    else:
                                        # RoI 밖에 있으면 초록색
                                        arrow_color = (0, 255, 0)
                                        endpoint_color = (0, 255, 0)
                                        endpoint_outer_color = (255, 255, 255)
                                    
                                    # 시선 화살표 그리기 (더 두껍게)
                                    cv2.arrowedLine(img, (face_center_x, face_center_y), 
                                                  (gaze_end_x, gaze_end_y),
                                                  arrow_color, 3, tipLength=0.2)
                                    
                                    # 시선 끝점 그리기 (더 크게, 이중 원)
                                    cv2.circle(img, (gaze_end_x, gaze_end_y), 10, endpoint_color, -1)
                                    cv2.circle(img, (gaze_end_x, gaze_end_y), 12, endpoint_outer_color, 2)
                                    
                                    # 라벨 생성
                                    roi_label = f"RoI:{current_roi}" if current_roi is not None else "No RoI"
                                    label = f'ID:{id} {obj_name} Y:{yaw:.1f}° P:{pitch:.1f}° {roi_label}'
                                else:
                                    label = f'ID:{id} {obj_name}'
                            else:
                                label = f'ID:{id} {obj_name}'
                        else:
                            label = f'ID:{id} {obj_name}'
                    else:
                        label = f'ID:{id} {obj_name}'
                else:
                    label = f'ID:{id} {obj_name}'
            else:
                label = f'ID:{id} {obj_name}'
                
            # 중심점을 버퍼에 추가
            data_deque[id].appendleft(center)
            self.UI_box(box, img, label=label, color=color, line_thickness=2)
            
            # 추적 경로 그리기
            for j in range(1, len(data_deque[id])):
                if data_deque[id][j - 1] is None or data_deque[id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + j)) * 1.5)
                cv2.line(img, data_deque[id][j - 1], data_deque[id][j], color, thickness)
                
        return img, active_rois
    
    def draw_rois(self, img, active_rois=None):
        """
        RoI들을 이미지에 그리기
        active_rois: 현재 누군가가 보고 있는 RoI ID들의 set
        """
        if active_rois is None:
            active_rois = set()
        
        for roi in self.rois:
            x1, y1, x2, y2, roi_id = roi
            color = roi_colors[(roi_id - 1) % len(roi_colors)]
            
            # 누군가가 보고 있는 RoI는 강조 (더 두껍게, 반투명 배경)
            if roi_id in active_rois:
                # 반투명 배경 추가
                overlay = img.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
                
                # 두꺼운 테두리
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
                
                # RoI ID 표시 (강조)
                label = f"RoI {roi_id} [ACTIVE]"
                # 배경 박스
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(img, (x1, y1 - text_height - 15), 
                            (x1 + text_width + 10, y1 - 5), color, -1)
                cv2.putText(img, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
            else:
                # 일반 RoI 박스 그리기
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # RoI ID 표시
                label = f"RoI {roi_id}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, 2)
        
        # 현재 그리는 중인 RoI 표시
        if drawing and current_roi is not None:
            x1, y1, x2, y2 = current_roi
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                
        return img
        
    def process_frame(self, frame, frame_count=0):
        """단일 프레임 처리"""
        # YOLOv8로 객체 탐지
        results = self.detection_model(frame, verbose=False)
        
        # 탐지 결과 처리
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # 사람 클래스만 필터링 (class 0)
                    if int(box.cls) == 0:  # person
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        detections.append([x1, y1, x2, y2, conf, cls])
        
        active_rois = set()  # 현재 보고 있는 RoI들
        
        # 간단한 추적 수행
        if len(detections) > 0:
            tracks = self.simple_track(detections, frame_count)
            
            if len(tracks) > 0:
                bbox_xyxy = []
                identities = []
                object_id = []
                
                for track in tracks:
                    x1, y1, x2, y2, track_id, cls = track
                    bbox_xyxy.append([x1, y1, x2, y2])
                    identities.append(track_id)
                    object_id.append(cls)
                
                bbox_xyxy = np.array(bbox_xyxy)
                identities = np.array(identities)
                object_id = np.array(object_id)
                
                # 바운딩 박스와 시선 정보 그리기
                frame, active_rois = self.draw_boxes_with_gaze(frame, bbox_xyxy, 
                                                self.detection_model.names, 
                                                object_id, identities)
        
        # RoI 그리기 (active_rois 정보 포함)
        frame = self.draw_rois(frame, active_rois)
        
        # 화면 상단에 정보 패널 그리기
        frame = self.draw_info_panel(frame, active_rois, len(detections) if len(detections) > 0 else 0)
                                            
        return frame
    
    def draw_info_panel(self, img, active_rois, num_people):
        """화면 상단에 정보 패널 그리기"""
        height, width = img.shape[:2]
        
        # 반투명 배경 패널
        panel_height = 120
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # 제목
        cv2.putText(img, "RoI Gaze Tracking System", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 현재 추적 중인 사람 수
        cv2.putText(img, f"Tracking: {num_people} people", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 현재 활성화된 RoI 수
        cv2.putText(img, f"Active RoIs: {len(active_rois)}/{len(self.rois)}", (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 범례 (오른쪽)
        legend_x = width - 350
        cv2.putText(img, "Legend:", (legend_x, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 얼굴 중심
        cv2.circle(img, (legend_x + 10, 45), 6, (255, 0, 0), -1)
        cv2.circle(img, (legend_x + 10, 45), 8, (255, 255, 255), 2)
        cv2.putText(img, "Face Center", (legend_x + 25, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 시선 끝점 (RoI 안)
        cv2.circle(img, (legend_x + 10, 70), 10, (0, 0, 255), -1)
        cv2.circle(img, (legend_x + 10, 70), 12, (255, 255, 0), 2)
        cv2.putText(img, "Gaze in RoI", (legend_x + 25, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 시선 끝점 (RoI 밖)
        cv2.circle(img, (legend_x + 10, 95), 10, (0, 255, 0), -1)
        cv2.circle(img, (legend_x + 10, 95), 12, (255, 255, 255), 2)
        cv2.putText(img, "Gaze outside", (legend_x + 25, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def setup_rois(self, video_path):
        """비디오의 첫 프레임에서 RoI 설정"""
        global rois, current_roi, drawing
        
        rois = []
        current_roi = None
        drawing = False
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise IOError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        # 첫 프레임 읽기
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise IOError("비디오의 첫 프레임을 읽을 수 없습니다")
        
        # RoI 설정 윈도우
        window_name = "RoI 설정 - 드래그하여 RoI 생성, 'c' 완료, 'r' 초기화"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        print("\n=== RoI 설정 모드 ===")
        print("마우스로 드래그하여 RoI를 설정하세요.")
        print("'c' 키: 설정 완료")
        print("'r' 키: 모든 RoI 초기화")
        print("====================\n")
        
        while True:
            display_frame = frame.copy()
            
            # 기존 RoI 그리기
            for roi in rois:
                x1, y1, x2, y2, roi_id = roi
                color = roi_colors[(roi_id - 1) % len(roi_colors)]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"RoI {roi_id}", (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 현재 그리는 중인 RoI 그리기
            if drawing and current_roi is not None:
                x1, y1, x2, y2 = current_roi
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            # 안내 메시지
            cv2.putText(display_frame, "Drag to create RoI | 'c': Complete | 'r': Reset",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"RoIs: {len(rois)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # 완료
                if len(rois) == 0:
                    print("최소 1개 이상의 RoI를 설정해주세요.")
                else:
                    break
            elif key == ord('r'):  # 초기화
                rois = []
                print("모든 RoI가 초기화되었습니다.")
        
        cv2.destroyWindow(window_name)
        
        self.rois = rois
        print(f"\n총 {len(self.rois)}개의 RoI가 설정되었습니다:")
        for roi in self.rois:
            x1, y1, x2, y2, roi_id = roi
            print(f"  RoI {roi_id}: ({x1}, {y1}) - ({x2}, {y2})")
    
    def save_gaze_journey_to_excel(self, output_path):
        """gaze_journey 데이터를 엑셀 파일로 저장"""
        if not self.person_gaze_data:
            print("저장할 gaze_journey 데이터가 없습니다.")
            return
        
        # 마무리 처리
        self.finalize_gaze_journey()
        
        excel_path = output_path.replace('.mp4', '_gaze_journey.xlsx')
        
        try:
            # 데이터 준비
            data = []
            for person_id, person_data in self.person_gaze_data.items():
                if len(person_data['gaze_journey']) > 0:
                    # gaze_journey를 문자열로 변환
                    journey_str = ', '.join([f"({roi_id},{frame_count})" 
                                            for roi_id, frame_count in person_data['gaze_journey']])
                    data.append({
                        'person_id': person_id,
                        'gaze_journey': journey_str
                    })
            
            # DataFrame 생성
            df = pd.DataFrame(data)
            
            # 엑셀로 저장
            df.to_excel(excel_path, index=False)
            
            print(f"\ngaze_journey 데이터가 저장되었습니다: {excel_path}")
            print(f"총 {len(data)}명의 데이터가 저장되었습니다.")
            
            # 각 사람의 journey 출력
            print("\n=== Gaze Journey 요약 ===")
            for person_id, person_data in sorted(self.person_gaze_data.items()):
                if len(person_data['gaze_journey']) > 0:
                    journey_str = ', '.join([f"({roi_id},{frame_count})" 
                                            for roi_id, frame_count in person_data['gaze_journey']])
                    print(f"Person ID {person_id}: {journey_str}")
            print("========================\n")
            
        except Exception as e:
            print(f"엑셀 저장 오류: {e}")
        
    def run_video(self, video_path, output_path=None):
        """비디오 파일 처리"""
        # RoI 설정
        self.setup_rois(video_path)
        
        # person_gaze_data 초기화
        self.person_gaze_data = defaultdict(lambda: {
            'current_roi': None, 
            'frame_count': 0, 
            'gaze_journey': [],
            'temp_roi': None,
            'temp_frame_count': 0
        })
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise IOError(f"비디오 파일을 열 수 없습니다: {video_path}")
            
        # 출력 비디오 설정
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
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
                    
                print(f"프레임 처리 중: {frame_count}/{total_frames}", end='\r')
                
                # 프레임 처리
                processed_frame = self.process_frame(frame, frame_count)
                frame_count += 1
                
                # 결과 저장
                if out:
                    out.write(processed_frame)
                else:
                    cv2.imshow("RoI 기반 시선 추적", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"\n비디오 처리 완료: {output_path if output_path else '화면 표시'}")
        
        # gaze_journey 데이터 엑셀 저장
        if output_path:
            self.save_gaze_journey_to_excel(output_path)


def main():
    parser = argparse.ArgumentParser(description='RoI 기반 시선 추적 시스템')
    parser.add_argument('--source', type=str, required=True,
                       help='입력 비디오 파일 경로')
    parser.add_argument('--output', type=str, default=None,
                       help='출력 비디오 파일 경로 (선택사항)')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU ID (CPU 사용시 -1, 기본값: -1)')
    parser.add_argument('--yolo_model', type=str, default='weights/yolov8n.pt',
                       help='YOLOv8 모델 경로')
    parser.add_argument('--pose_model', type=str, default='weights/6DRepNet_300W_LP_AFLW2000.pth',
                       help='6DRepNet 모델 경로')
    parser.add_argument('--min_frames', type=int, default=5,
                       help='RoI 응시 판단을 위한 최소 프레임 수 (기본값: 5)')
    parser.add_argument('--gaze_distance', type=int, default=300,
                       help='시선 끝점 계산을 위한 거리 상수 (기본값: 300)')
    
    args = parser.parse_args()
    
    # 출력 경로 설정
    if args.output is None:
        # 기본 출력 경로 생성
        input_filename = os.path.basename(args.source)
        input_name = os.path.splitext(input_filename)[0]
        args.output = f"Results/roi_gaze_{input_name}.mp4"
        
        # Results 디렉토리 생성
        os.makedirs("Results", exist_ok=True)
    
    # RoI 기반 시선 추적 시스템 초기화
    tracker = RoIGazeTracker(
        yolov8_model_path=args.yolo_model,
        sixdrepnet_model_path=args.pose_model,
        gpu_id=args.gpu,
        min_frames_threshold=args.min_frames,
        gaze_distance_k=args.gaze_distance
    )
    
    # 비디오 파일 처리
    tracker.run_video(args.source, args.output)


if __name__ == "__main__":
    main()

