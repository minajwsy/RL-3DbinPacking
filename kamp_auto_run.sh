#!/bin/bash
# KAMP 서버용 Maskable PPO 3D Bin Packing 자동 실행 스크립트
# 기존 auto_run.sh의 구조를 최대한 유지하면서 Maskable PPO 적용

# 오류 발생시엔 스크립트 중단  
set -e

# 브랜치 설정 (인자로 전달하거나 기본값 사용)
BRANCH=${1:-master}

# 환경 변수 설정 (기존 구조 유지)
WORK_DIR="${HOME}/RL-3DbinPacking"
REPO_URL="https://github.com/YOUR_USERNAME/RL-3DbinPacking.git"  # 실제 저장소로 변경 필요
LOG_DIR="${WORK_DIR}/logs"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# 로그 설정 (기존과 동일)
mkdir -p "${LOG_DIR}"
exec 1> >(tee "${LOG_DIR}/ppo_3dbp_maskable_${TIMESTAMP}.log")
exec 2>&1

echo "=== Maskable PPO 3D Bin Packing 실험 시작: ${TIMESTAMP} ==="
echo "브랜치: ${BRANCH}"

# Git 저장소 확인 (기존과 동일)
if [ ! -d "${WORK_DIR}/.git" ]; then
    echo "Git 저장소가 없습니다. 클론을 시작합니다."
    git clone "${REPO_URL}" "${WORK_DIR}"
fi

# 작업 디렉토리 설정
cd "${WORK_DIR}" || exit 1

# Git 상태 정리 및 동기화 (기존 auto_run.sh 로직 완전 유지)
echo "=== Git 저장소 동기화 중 ==="
# 원격(git서버)에서 삭제된 브랜치를 로컬(kamp서버)에서도 자동으로 제거 위해 --prune 옵션 추가
git fetch origin --prune 

# 원격(git서버)에서 삭제된 로컬(kamp서버) 브랜치 정리
echo "=== 삭제된 원격 브랜치 정리 중 ==="
for local_branch in $(git branch | grep -v "^*" | tr -d ' '); do
  if [ "${local_branch}" != "${BRANCH}" ] && ! git ls-remote --heads origin ${local_branch} | grep -q ${local_branch}; then
    echo "로컬 브랜치 '${local_branch}'가 원격에 없으므로 삭제합니다."
    git branch -D ${local_branch} || true
  fi
done

# 브랜치 체크아웃 또는 생성 (기존과 동일)
git checkout ${BRANCH} 2>/dev/null || git checkout -b ${BRANCH} origin/${BRANCH} 2>/dev/null || git checkout -b ${BRANCH}

# 원격 브랜치가 있으면 동기화
if git ls-remote --heads origin ${BRANCH} | grep -q ${BRANCH}; then
    git reset --hard "origin/${BRANCH}"
fi

git clean -fd || true

# Python 환경 설정 (원본 구조 기반으로 개선)
echo "=== Python 환경 설정 ==="
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# uv 패키지 관리자 사용
if command -v uv &> /dev/null; then
    echo "uv 패키지 관리자를 사용하여 의존성 설치"
    uv sync
else
    echo "uv를 찾을 수 없습니다. pip으로 설치 진행"
    pip install -r requirements.txt
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
fi

# GPU 상태 확인 (기존과 동일하지만 더 자세한 정보)
echo "=== 하드웨어 정보 확인 ==="
if nvidia-smi > /dev/null 2>&1; then
    echo "GPU 사용 가능:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    export CUDA_VISIBLE_DEVICES=0
    export FORCE_CPU=false
else
    echo "GPU를 찾을 수 없습니다. CPU 모드로 실행합니다."
    export FORCE_CPU=true
fi

# 디바이스 정보 출력 (신규 추가)
echo "=== 디바이스 정보 확인 ==="
python -c "
import sys
sys.path.insert(0, 'src')
try:
    from device_utils import print_device_info
    print_device_info()
except ImportError:
    print('디바이스 유틸리티를 찾을 수 없습니다.')
"

# 실험 환경 준비 (기존 구조 기반)
mkdir -p results models logs gifs

# Maskable PPO 실험 실행 (기존 패턴 유지하면서 새로운 훈련 스크립트 호출)
echo "=== Maskable PPO 3D Bin Packing 학습 시작: $(date '+%Y-%m-%d %H:%M:%S') ==="

# 환경 변수 설정 (실험 파라미터)
export TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-100000}
export CONTAINER_SIZE="10,10,10"
export NUM_BOXES=${NUM_BOXES:-64}

# 메인 훈련 실행 (기존 실행 패턴과 유사)
if python -m src.train 2>&1 | tee results/training_output_${TIMESTAMP}.txt; then
    echo "Maskable PPO 3D Bin Packing 학습이 성공적으로 완료되었습니다."
    TRAINING_SUCCESS=true
else
    echo "학습 실행 중 오류가 발생했습니다. 오류 로그를 확인하세요."
    TRAINING_SUCCESS=false
fi

# 학습 성공/실패와 관계없이 5초 대기 (기존과 동일)
echo "다음 단계로 진행하기 전에 5초 대기합니다..."
sleep 5

# 모델 평가 실행 (있다면)
if [ "${TRAINING_SUCCESS}" = true ] && [ -d "models" ] && [ "$(ls -A models/)" ]; then
    echo "=== 모델 평가 시작: $(date '+%Y-%m-%d %H:%M:%S') ==="
    
    # 가장 최근 모델 찾기
    LATEST_MODEL=$(find models -name "*.zip" -type f -exec ls -t {} + | head -n 1)
    
    if [ -n "${LATEST_MODEL}" ]; then
        echo "평가할 모델: ${LATEST_MODEL}"
        
        # 평가 스크립트가 있다면 실행
        if [ -f "src/eval.py" ]; then
            python -m src.eval --model_path="${LATEST_MODEL}" 2>&1 | tee results/evaluation_${TIMESTAMP}.txt
        else
            echo "평가 스크립트를 찾을 수 없습니다."
        fi
    else
        echo "평가할 모델을 찾을 수 없습니다."
    fi
fi

# 결과 처리 (기존 auto_run.sh 패턴 완전 유지)
echo "=== 변경사항 확인 중 ==="
git status

# 파일 존재 여부 확인 후 추가 (기존 패턴 기반으로 확장)
echo "=== 결과 파일 추가 중 ==="

# 훈련 결과 파일 추가
if [ -f "results/training_output_${TIMESTAMP}.txt" ]; then
    git add "results/training_output_${TIMESTAMP}.txt"
    echo "훈련 결과 파일이 추가되었습니다."
else
    echo "훈련 결과 파일이 존재하지 않습니다."
fi

# 평가 결과 파일 추가
if [ -f "results/evaluation_${TIMESTAMP}.txt" ]; then
    git add "results/evaluation_${TIMESTAMP}.txt"
    echo "평가 결과 파일이 추가되었습니다."
else
    echo "평가 결과 파일이 존재하지 않습니다."
fi

# 최신 결과 파일들 추가 (기존 패턴 유지)
RESULT_FILES=$(find "results" -name "*.txt" -newer "results/training_output_${TIMESTAMP}.txt" 2>/dev/null || echo "")
if [ -n "$RESULT_FILES" ]; then
    git add $RESULT_FILES
    echo "최신 결과 파일들이 추가되었습니다."
fi

# 모델 파일 추가 (신규)
MODEL_FILES=$(find "models" -name "*.zip" 2>/dev/null)
if [ -n "$MODEL_FILES" ]; then
    git add $MODEL_FILES
    echo "훈련된 모델 파일들이 추가되었습니다."
else
    echo "모델 파일이 존재하지 않습니다."
fi

# PNG 파일 추가 (기존과 동일)
PNG_FILES=$(find "results" -name "*.png" 2>/dev/null)
if [ -n "$PNG_FILES" ]; then
    git add $PNG_FILES
    echo "PNG 파일이 추가되었습니다."
else
    echo "PNG 파일이 존재하지 않습니다."
fi

# GIF 파일 추가 (신규)
GIF_FILES=$(find "gifs" -name "*.gif" 2>/dev/null)
if [ -n "$GIF_FILES" ]; then
    git add $GIF_FILES
    echo "GIF 파일이 추가되었습니다."
else
    echo "GIF 파일이 존재하지 않습니다."
fi

# 로그 파일 추가 (기존 패턴 확장)
LOG_FILES=$(find "logs" -name "*.log" -o -name "*.txt" 2>/dev/null)
if [ -n "$LOG_FILES" ]; then
    git add $LOG_FILES
    echo "로그 파일이 추가되었습니다."
else
    echo "로그 파일이 존재하지 않습니다."
fi

# 변경사항이 있는지 확인 (기존과 동일)
if git diff --cached --quiet; then
    echo "=== 변경사항 없음 ==="
else
    # 커밋 및 푸시 (기존 패턴 기반으로 메시지 개선)
    if [ "${TRAINING_SUCCESS}" = true ]; then
        COMMIT_MESSAGE="Maskable PPO 3D Bin Packing 실험 성공: ${TIMESTAMP}"
    else
        COMMIT_MESSAGE="Maskable PPO 3D Bin Packing 실험 (일부 오류): ${TIMESTAMP}"
    fi
    
    git commit -m "${COMMIT_MESSAGE}"
    git push origin "${BRANCH}"
    echo "=== 결과 업로드 완료 ==="
fi

echo "=== KAMP 서버 작업 완료: $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "실행 요약:"
echo "- 브랜치: ${BRANCH}"
echo "- 훈련 성공: ${TRAINING_SUCCESS}"
echo "- 로그 위치: ${LOG_DIR}/ppo_3dbp_maskable_${TIMESTAMP}.log"

# 완료 알림 (신규 추가)
if command -v notify-send &> /dev/null; then
    if [ "${TRAINING_SUCCESS}" = true ]; then
        notify-send "KAMP 실험 완료" "Maskable PPO 3D Bin Packing 실험이 성공적으로 완료되었습니다."
    else
        notify-send "KAMP 실험 완료" "Maskable PPO 3D Bin Packing 실험이 완료되었습니다 (일부 오류 발생)."
    fi
fi 