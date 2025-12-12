#!/bin/bash

# 배치 인퍼런스 스크립트
# Results/A2_1people 폴더에 {영상이름}_infer.mp4 형식으로 저장

cd /Users/ohseoyoung/Downloads/capcap/Code_verson2
source ~/.pyenv/versions/cephalo-env/bin/activate

# 출력 디렉토리 생성
mkdir -p Results/A2_1people

# 영상 목록
videos=(
    "IMG_5268.MOV"
    "IMG_5269.MOV"
    "IMG_5270.MOV"
    "IMG_5271.MOV"
    "IMG_5272.MOV"
    "IMG_5273.MOV"
    "IMG_5274.MOV"
    "IMG_5275.MOV"
    "IMG_5276.MOV"
    "IMG_5277.MOV"
    "IMG_5278.MOV"
    "IMG_5279.MOV"
)

# 총 영상 개수
total=${#videos[@]}
current=0

echo "=========================================="
echo "배치 인퍼런스 시작"
echo "총 ${total}개 영상 처리 예정"
echo "=========================================="
echo ""

# 각 영상 처리
for video in "${videos[@]}"; do
    current=$((current + 1))
    
    # 영상 이름 추출 (확장자 제거)
    video_name="${video%.*}"
    
    # 입력/출력 경로
    input_path="Input/${video}"
    output_path="Results/A2_1people/${video_name}_infer.mp4"
    
    echo "[${current}/${total}] 처리 중: ${video}"
    echo "  입력: ${input_path}"
    echo "  출력: ${output_path}"
    echo ""
    
    # 인퍼런스 실행
    python3 roi_gaze_tracking_gaze360_real.py \
        --source "${input_path}" \
        --output "${output_path}" \
        --roi_config config/A2roi_default.json \
        --gpu -1 \
        --min_frames 5
    
    if [ $? -eq 0 ]; then
        echo "  ✓ 완료: ${video}"
    else
        echo "  ✗ 실패: ${video}"
    fi
    
    echo ""
    echo "----------------------------------------"
    echo ""
done

echo "=========================================="
echo "배치 인퍼런스 완료"
echo "결과 저장 위치: Results/A2_1people/"
echo "=========================================="

