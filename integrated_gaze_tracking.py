#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합된 시선 추정 및 객체 추적 시스템
YOLOv8 + IoU Tracking + Gaze360을 결합한 시스템
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
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import csv
from collections import defaultdict

# Gaze360 관련 import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gaze360/code'))
from model import GazeLSTM
from resnet import resnet18

# YOLOv8 관련 import
from ultralytics import YOLO
from collections import deque

# 얼굴 탐지를 위한 import
try:
    from face_detection import RetinaFace
    FACE_DETECTOR_AVAILABLE = True
except:
    FACE_DETECTOR_AVAILABLE = False
    print("Warning: RetinaFace를 import할 수 없습니다. 얼굴 탐지를 위해 전체 바운딩 박스를 사용합니다.")

# 전역 변수
data_deque = {}
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
next_id = 1
tracked_objects = {}  # {id: {'bbox': [x1,y1,x2,y2], 'last_seen': frame_count, 'face_frames': deque}}


class IntegratedGazeTracker:
    def __init__(self, 
                 yolov8_model_path="weights/yolov8n.pt",
                 gaze360_model_path="weights/gaze360_model.pth.tar",
                 gpu_id=0):
        """
        통합된 시선 추정 및 추적 시스템 초기화
        
        Args:
            yolov8_model_path: YOLOv8 모델 경로
            gaze360_model_path: Gaze360 모델 경로
            gpu_id: GPU ID (CPU 사용시 -1)
        """
        self.gpu_id = gpu_id
        # CUDA 사용 가능 여부 확인
        if gpu_id >= 0 and torch.cuda.is_available():
            self.device = torch.device('cuda:%d' % gpu_id)
            print(f"GPU {gpu_id} 사용")
        else:
            self.device = torch.device('cpu')
            print("CPU 사용")
        
        # Gaze360 모델 초기화
        self.init_gaze_model(gaze360_model_path)
        
        # YOLOv8 모델 초기화
        self.init_detection_model(yolov8_model_path)
        
        # 추적 시스템 초기화
        self.init_tracker()
        
        # 시선 데이터 저장용
        self.gaze_data = []  # [(frame_num, person_id, yaw, pitch, gaze_vector_x, y, z), ...]
        
        # 이미지 전처리 변환
        self.image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        self.transformations = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.image_normalize
        ])
        
        # 얼굴 탐지기 초기화
        if FACE_DETECTOR_AVAILABLE:
            try:
                # CPU 모드인 경우 -1로 설정
                face_gpu_id = -1 if self.device.type == 'cpu' else gpu_id
                self.face_detector = RetinaFace(gpu_id=face_gpu_id)
            except:
                self.face_detector = None
                print("Warning: RetinaFace 초기화 실패")
        else:
            self.face_detector = None
        
    def init_gaze_model(self, model_path):
        """Gaze360 모델 초기화"""
        print("Gaze360 모델을 로딩하는 중...")
        self.gaze_model = GazeLSTM()
        
        # 가중치 로드
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            # DataParallel로 저장된 경우
            state_dict = checkpoint['state_dict']
            # 'module.' 접두사 제거
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')  # remove 'module.' prefix
                new_state_dict[name] = v
            self.gaze_model.load_state_dict(new_state_dict)
        else:
            self.gaze_model.load_state_dict(checkpoint)
        
        self.gaze_model.to(self.device)
        self.gaze_model.eval()
        print("Gaze360 모델 로딩 완료")
        
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
                    'last_seen': frame_count,
                    'face_frames': deque(maxlen=7)  # 최근 7개 프레임 저장
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
        elif label == 2:  # Car
            color = (222, 82, 175)
        elif label == 3:  # Motobike
            color = (0, 204, 255)
        elif label == 5:  # Bus
            color = (0, 149, 255)
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
    
    def spherical2cartesian(self, x):
        """구면 좌표를 카르테시안 좌표로 변환"""
        output = np.zeros((x.shape[0], 3))
        output[:, 2] = -np.cos(x[:, 1]) * np.cos(x[:, 0])
        output[:, 0] = np.cos(x[:, 1]) * np.sin(x[:, 0])
        output[:, 1] = np.sin(x[:, 1])
        return output
    
    def estimate_gaze(self, face_frames):
        """얼굴 프레임들에서 시선 추정"""
        try:
            if len(face_frames) < 7:
                return None, None, None
            
            # 7개의 프레임을 텐서로 변환
            frames_tensor = torch.stack(face_frames).unsqueeze(0).to(self.device)
            
            # 시선 추정
            with torch.no_grad():
                angular_output, var = self.gaze_model(frames_tensor)
                
            # 각도를 numpy로 변환 (radians)
            yaw = angular_output[0, 0].cpu().numpy()
            pitch = angular_output[0, 1].cpu().numpy()
            
            # 라디안을 degree로 변환
            yaw_deg = float(yaw * 180 / np.pi)
            pitch_deg = float(pitch * 180 / np.pi)
            
            # 3D 시선 벡터 계산 (단위 벡터)
            # Gaze360의 좌표계: x는 좌우, y는 상하, z는 깊이
            gaze_vector = np.array([
                np.cos(pitch) * np.sin(yaw),      # x
                np.sin(pitch),                     # y
                -np.cos(pitch) * np.cos(yaw)      # z (카메라 방향이 -z)
            ])
            
            return yaw_deg, pitch_deg, gaze_vector
        except Exception as e:
            print(f"시선 추정 오류: {e}")
            return None, None, None
    
    def draw_gaze_arrow(self, img, x, y, yaw, pitch, length=100, color=(0, 255, 0), thickness=2):
        """시선 방향을 화살표로 그리기"""
        try:
            # 각도를 라디안으로 변환
            yaw_rad = yaw * np.pi / 180
            pitch_rad = pitch * np.pi / 180
            
            # 3D 벡터 계산
            dx = -length * np.cos(pitch_rad) * np.sin(yaw_rad)
            dy = -length * np.sin(pitch_rad)
            
            # 시작점과 끝점 계산
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + dx), int(y + dy)
            
            # 화살표 그리기
            cv2.arrowedLine(img, (x1, y1), (x2, y2), color, thickness, tipLength=0.3)
            
            return img
        except Exception as e:
            print(f"화살표 그리기 오류: {e}")
            return img
            
    def draw_boxes_with_gaze(self, img, bbox, names, object_id, identities=None, offset=(0, 0), frame_num=0):
        """바운딩 박스와 시선 정보를 그리기"""
        global data_deque, tracked_objects
        
        height, width, _ = img.shape
        
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
            
            # 사람인 경우에만 시선 추정
            if object_id[i] == 0:  # person class
                # 얼굴 영역 추출
                face_roi = img[y1:y2, x1:x2]
                if face_roi.size > 0:
                    face_x, face_y, face_w, face_h = x1, y1, x2 - x1, y2 - y1
                    
                    # 얼굴 탐지 시도
                    if self.face_detector is not None:
                        try:
                            faces = self.face_detector(face_roi)
                            if len(faces) > 0:
                                # 가장 높은 점수의 얼굴 선택
                                best_face = max(faces, key=lambda x: x[2])
                                if best_face[2] > 0.5:  # 신뢰도 임계값
                                    box_face, landmarks, score = best_face
                                    fx_min, fy_min, fx_max, fy_max = [int(coord) for coord in box_face]
                                    
                                    # 얼굴 영역을 원본 이미지 좌표로 변환
                                    face_x = x1 + fx_min
                                    face_y = y1 + fy_min
                                    face_w = fx_max - fx_min
                                    face_h = fy_max - fy_min
                        except Exception as e:
                            print(f"얼굴 탐지 오류: {e}")
                    
                    # 얼굴 영역 크롭 및 전처리
                    try:
                        face_crop = img[face_y:face_y+face_h, face_x:face_x+face_w]
                        if face_crop.size > 0 and face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                            face_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                            face_tensor = self.transformations(face_img)
                            
                            # face_frames deque에 추가
                            if id in tracked_objects:
                                tracked_objects[id]['face_frames'].append(face_tensor)
                                
                                # 7개 프레임이 모이면 시선 추정
                                if len(tracked_objects[id]['face_frames']) == 7:
                                    yaw_deg, pitch_deg, gaze_vector = self.estimate_gaze(
                                        list(tracked_objects[id]['face_frames'])
                                    )
                                    
                                    if yaw_deg is not None and pitch_deg is not None and gaze_vector is not None:
                                        # 시선 데이터 저장
                                        self.gaze_data.append({
                                            'frame': frame_num,
                                            'person_id': id,
                                            'yaw_deg': yaw_deg,
                                            'pitch_deg': pitch_deg,
                                            'gaze_x': float(gaze_vector[0]),
                                            'gaze_y': float(gaze_vector[1]),
                                            'gaze_z': float(gaze_vector[2]),
                                            'face_center_x': face_x + face_w // 2,
                                            'face_center_y': face_y + face_h // 2
                                        })
                                        
                                        # 시선 방향 화살표 그리기
                                        face_center_x = face_x + face_w // 2
                                        face_center_y = face_y + face_h // 2
                                        self.draw_gaze_arrow(img, face_center_x, face_center_y, 
                                                           yaw_deg, pitch_deg, 
                                                           length=100, color=(0, 255, 0), thickness=3)
                                        
                                        # 라벨에 시선 정보 추가
                                        label = f'ID:{id} {obj_name} Yaw:{yaw_deg:.1f}° Pitch:{pitch_deg:.1f}°'
                                    else:
                                        label = f'ID:{id} {obj_name}'
                                else:
                                    label = f'ID:{id} {obj_name} ({len(tracked_objects[id]["face_frames"])}/7)'
                            else:
                                label = f'ID:{id} {obj_name}'
                        else:
                            label = f'ID:{id} {obj_name}'
                    except Exception as e:
                        print(f"얼굴 전처리 오류: {e}")
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
        
        if len(detections) == 0:
            return frame
        
        # 간단한 추적 수행
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
            frame = self.draw_boxes_with_gaze(frame, bbox_xyxy, 
                                            self.detection_model.names, 
                                            object_id, identities, frame_num=frame_count)
                                            
        return frame
    
    def save_gaze_data_to_csv(self, output_path):
        """시선 데이터를 CSV 파일로 저장"""
        if not self.gaze_data:
            print("저장할 시선 데이터가 없습니다.")
            return
        
        csv_path = output_path.replace('.mp4', '_gaze_data.csv')
        
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['frame', 'person_id', 'yaw_deg', 'pitch_deg', 
                            'gaze_x', 'gaze_y', 'gaze_z', 'face_center_x', 'face_center_y']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for data in self.gaze_data:
                    writer.writerow(data)
            
            print(f"시선 데이터가 저장되었습니다: {csv_path}")
            print(f"총 {len(self.gaze_data)}개의 시선 데이터가 저장되었습니다.")
        except Exception as e:
            print(f"CSV 저장 오류: {e}")
        
    def run_camera(self, cam_id=0):
        """카메라에서 실시간 처리"""
        cap = cv2.VideoCapture(cam_id)
        
        if not cap.isOpened():
            raise IOError("카메라를 열 수 없습니다")
            
        print("실시간 처리를 시작합니다. 'q'를 눌러 종료하세요.")
        
        frame_count = 0
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 프레임 처리
                processed_frame = self.process_frame(frame, frame_count)
                frame_count += 1
                
                # 결과 표시
                cv2.imshow("통합 시선 추정 및 추적", processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        cap.release()
        cv2.destroyAllWindows()
        
    def run_video(self, video_path, output_path=None):
        """비디오 파일 처리"""
        # 시선 데이터 초기화
        self.gaze_data = []
        
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
            
        print(f"비디오 처리를 시작합니다: {video_path}")
        
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
                    cv2.imshow("통합 시선 추정 및 추적", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"\n비디오 처리 완료: {output_path if output_path else '화면 표시'}")
        
        # 시선 데이터 CSV 저장
        if output_path and len(self.gaze_data) > 0:
            self.save_gaze_data_to_csv(output_path)


def main():
    parser = argparse.ArgumentParser(description='통합된 시선 추정 및 객체 추적 시스템')
    parser.add_argument('--source', type=str, default='0', 
                       help='입력 소스 (카메라 ID 또는 비디오 파일 경로)')
    parser.add_argument('--output', type=str, default=None,
                       help='출력 비디오 파일 경로 (선택사항)')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU ID (CPU 사용시 -1, 기본값: -1)')
    parser.add_argument('--yolo_model', type=str, default='weights/yolov8n.pt',
                       help='YOLOv8 모델 경로')
    parser.add_argument('--gaze_model', type=str, default='weights/gaze360_model.pth.tar',
                       help='Gaze360 모델 경로')
    
    args = parser.parse_args()
    
    # 통합 시스템 초기화
    tracker = IntegratedGazeTracker(
        yolov8_model_path=args.yolo_model,
        gaze360_model_path=args.gaze_model,
        gpu_id=args.gpu
    )
    
    # 입력 소스에 따라 실행
    if args.source.isdigit():
        # 카메라
        tracker.run_camera(int(args.source))
    else:
        # 비디오 파일
        tracker.run_video(args.source, args.output)


if __name__ == "__main__":
    main()

