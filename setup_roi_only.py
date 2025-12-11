#!/usr/bin/env python3
"""
ROI만 설정하고 저장하는 간단한 스크립트
"""
import cv2
import json
import os
import sys
from typing import Optional

# roi_gaze_tracking_gaze360.py에서 필요한 전역 변수와 함수 가져오기
import numpy as np

# 전역 변수
rois = []
current_polygon_points = []

roi_colors = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
]

def mouse_callback(event, x, y, flags, param):
    """점 4개를 찍어서 ROI 생성 (항상 polygon 모드)"""
    global rois, current_polygon_points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 4개 점만 받도록 제한
        if len(current_polygon_points) < 4:
            current_polygon_points.append((x, y))
            print(f"점 추가: ({x}, {y}) ({len(current_polygon_points)}/4)")
            if len(current_polygon_points) == 4:
                print("4개 점 모두 추가됨. 자동으로 ROI가 생성됩니다.")
                # 자동으로 ROI 생성
                roi_id = len(rois) + 1
                rois.append({
                    "type": "poly",
                    "roi_id": roi_id,
                    "data": current_polygon_points.copy()
                })
                print(f"RoI {roi_id} 생성 완료! 다음 ROI를 그리려면 점을 다시 찍으세요.")
                current_polygon_points = []
        else:
            print("이미 4개의 점이 추가되었습니다. 'u' 키로 점을 제거하거나 'c' 키로 완료하세요.")
    elif event == cv2.EVENT_RBUTTONDOWN and current_polygon_points:
        removed = current_polygon_points.pop()
        print(f"점 제거: {removed} (남은 점: {len(current_polygon_points)}/4)")

def save_rois_to_config(config_path: str):
    """ROI를 JSON 파일로 저장"""
    config = {"rois": rois}
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nROI 설정을 저장했습니다: {config_path}")
    print(f"총 {len(rois)}개의 ROI가 저장되었습니다.")

def main():
    global rois, current_polygon_points
    
    if len(sys.argv) < 3:
        print("사용법: python3 setup_roi_only.py <비디오_경로> <저장할_JSON_경로>")
        print("예: python3 setup_roi_only.py Input/IMG_6553.mov roi_default.json")
        sys.exit(1)
    
    video_path = sys.argv[1]
    config_path = sys.argv[2]
    
    # 기존 설정 로드 (있는 경우)
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if "rois" in config and len(config["rois"]) > 0:
                    rois = config["rois"]
                    print(f"기존 ROI {len(rois)}개를 로드했습니다. 추가로 ROI를 그릴 수 있습니다.")
        except Exception as e:
            print(f"기존 설정 로드 실패: {e}")
    
    # 비디오 첫 프레임 읽기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        sys.exit(1)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("비디오의 첫 프레임을 읽을 수 없습니다.")
        sys.exit(1)
    
    window_name = "RoI 설정 - 드래그하여 RoI 생성, 'c' 완료, 'r' 초기화"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n=== RoI 설정 모드 (점 4개 방식) ===")
    print("여러 개의 ROI를 설정할 수 있습니다!")
    print()
    print("사용 방법:")
    print("  1. 왼쪽 클릭으로 점 4개를 순서대로 찍으세요")
    print("  2. 4개 점을 모두 찍으면 자동으로 ROI가 생성됩니다")
    print("  3. 다음 ROI를 그리려면 다시 점 4개를 찍으세요")
    print()
    print("키보드 단축키:")
    print("  'c' 키: 모든 RoI 설정 완료 및 저장")
    print("  'r' 키: 모든 RoI 초기화")
    print("  'u' 키: 현재 그리는 ROI의 마지막 점 제거")
    print("  'q' 또는 ESC: 종료 (저장 안 함)")
    print("====================\n")
    
    while True:
        display_frame = frame.copy()
        
        # 기존 ROI 그리기
        for roi_entry in rois:
            if isinstance(roi_entry, dict):
                roi_type = roi_entry.get("type", "rect")
                roi_id = roi_entry.get("roi_id", len(rois))
                color = roi_colors[(roi_id - 1) % len(roi_colors)]
                if roi_type == "rect":
                    x1, y1, x2, y2 = roi_entry["data"]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"RoI {roi_id}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                elif roi_type == "poly":
                    pts = np.array(roi_entry["data"], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [pts], True, color, 2)
                    label_x, label_y = roi_entry["data"][0]
                    cv2.putText(display_frame, f"RoI {roi_id}", (label_x, label_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 현재 그리는 점들 표시
        if len(current_polygon_points) > 0:
            pts = np.array(current_polygon_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], False, (0, 255, 255), 2)
            for idx, pt in enumerate(current_polygon_points):
                cv2.circle(display_frame, pt, 6, (0, 255, 255), -1)
                cv2.circle(display_frame, pt, 8, (255, 255, 255), 2)
                cv2.putText(display_frame, str(idx + 1), (pt[0] + 10, pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 상태 표시
        cv2.putText(display_frame, "점 4개 방식", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, f"ROI 개수: {len(rois)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if len(current_polygon_points) > 0:
            cv2.putText(display_frame, f"점: {len(current_polygon_points)}/4", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            if len(rois) > 0:
                save_rois_to_config(config_path)
                break
            else:
                print("최소 1개의 ROI가 필요합니다!")
        elif key == ord('r'):
            rois = []
            current_polygon_points = []
            print("모든 ROI가 초기화되었습니다.")
        elif key == ord('u'):
            if current_polygon_points:
                removed = current_polygon_points.pop()
                print(f"점 제거: {removed} (남은 점: {len(current_polygon_points)}/4)")
        elif key == 27 or key == ord('q'):  # ESC or 'q'
            print("종료합니다. (저장 안 함)")
            break
    
    cv2.destroyAllWindows()
    print("ROI 설정 완료!")

if __name__ == "__main__":
    main()

