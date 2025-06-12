#!/bin/bash
# KAMP 서버용 Maskable PPO 3D Bin Packing 자동 실행 스크립트
# 기존 auto_run.sh 기반, uv를 활용한 의존성 관리와 GPU/CPU 자동 선택 기능 추가
# 사용법: ./kamp_auto_run.sh [브랜치명]   <- 기본값은 master

# 오류 발생시엔 스크립트 중단  
set -e

# 브랜치 설정 (인자로 전달하거나 기본값 사용)
BRANCH=${1:-master}

# 환경 변수 설정
WORK_DIR="${HOME}/RL-3DbinPacking"
REPO_URL="https://github.com/your-username/RL-3DbinPacking.git"  # 실제 저장소 URL로 변경 필요
LOG_DIR="${WORK_DIR}/logs"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# 로그 설정
mkdir -p "${LOG_DIR}"
exec 1> >(tee "${LOG_DIR}/kamp_auto_run_${TIMESTAMP}.log")
exec 2>&1

echo "=== KAMP 서버 Maskable PPO 3D-BinPacking 실행 시작: ${TIMESTAMP} ==="

# Git 저장소 확인 및 클론
if [ ! -d "${WORK_DIR}/.git" ]; then
    echo "Git 저장소가 없습니다. 클론을 시작합니다."
    git clone "${REPO_URL}" "${WORK_DIR}"
fi

# 작업 디렉토리 설정
cd "${WORK_DIR}" || exit 1

# Git 상태 정리 및 동기화 (기존 auto_run.sh 로직 유지)
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

# 브랜치 체크아웃 또는 생성 (중요: 이 부분 추가)
git checkout ${BRANCH} 2>/dev/null || git checkout -b ${BRANCH} origin/${BRANCH} 2>/dev/null || git checkout -b ${BRANCH}

# 원격 브랜치가 있으면 동기화
if git ls-remote --heads origin ${BRANCH} | grep -q ${BRANCH}; then
    git reset --hard "origin/${BRANCH}"
fi

git clean -fd || true

# uv 설치 확인 및 설치
echo "=== uv 패키지 매니저 확인 중 ==="
if ! command -v uv &> /dev/null; then
    echo "uv가 설치되어 있지 않습니다. 설치를 진행합니다."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

echo "uv 버전: $(uv --version)"

# Python 환경 설정
echo "=== Python 환경 설정 중 ==="
# Python 3.9 설치 및 가상환경 생성
uv python install 3.9
uv venv .venv --python 3.9
source .venv/bin/activate

# GPU 확인 및 적절한 의존성 설치
echo "=== GPU 상태 확인 및 의존성 설치 중 ==="
if nvidia-smi > /dev/null 2>&1; then
    echo "GPU가 감지되었습니다. GPU 버전 패키지를 설치합니다."
    nvidia-smi
    
    # GPU 버전 PyTorch 설치
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # 기본 의존성 설치
    uv sync
    echo "GPU 환경 설정 완료"
else
    echo "GPU를 찾을 수 없습니다. CPU 버전 패키지를 설치합니다."
    
    # CPU 버전 PyTorch 설치
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    # 기본 의존성 설치
    uv sync
    echo "CPU 환경 설정 완료"
fi

# Python 경로 설정
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# 디렉토리 생성
mkdir -p models results logs gifs

# Maskable PPO 실험 실행
echo "=== Maskable PPO 3D Bin Packing 학습 시작: $(date '+%Y-%m-%d %H:%M:%S') ==="

# 학습 실행 (기존 auto_run.sh의 실행 패턴 유지)
if python -m src.train_maskable_ppo --timesteps 100000 2>&1 | tee results/training_output_${TIMESTAMP}.txt; then
    echo "Maskable PPO 학습이 성공적으로 완료되었습니다."
else
    echo "학습 실행 중 오류가 발생했습니다. 오류 로그를 확인하세요."
    # 오류가 있더라도 계속 진행 (기존 로직 유지)
fi

# 실험이 성공/실패와 관계없이 5초 대기 (기존 로직 유지)
echo "다음 단계로 진행하기 전에 5초 대기합니다..."
sleep 5

# 모델 평가 실행
echo "=== 모델 평가 시작: $(date '+%Y-%m-%d %H:%M:%S') ==="
if [ -f "models/ppo_mask_*.zip" ]; then
    # 가장 최근 모델 찾기
    LATEST_MODEL=$(ls -t models/ppo_mask_*.zip | head -n1)
    echo "최신 모델 평가 중: ${LATEST_MODEL}"
    
    # 간단한 평가 스크립트 실행
    python -c "
import sys
sys.path.append('src')
from train_maskable_ppo import evaluate_model
try:
    evaluate_model('${LATEST_MODEL}', num_episodes=5)
except Exception as e:
    print(f'평가 중 오류: {e}')
" 2>&1 | tee results/evaluation_${TIMESTAMP}.txt
else
    echo "평가할 모델이 없습니다."
fi

# 결과 처리 (기존 auto_run.sh 로직 유지)
echo "=== 변경사항 확인 중 ==="
git status

# 파일 존재 여부 확인 후 추가 (기존 로직 확장)
echo "=== 결과 파일 추가 중 ==="

# 학습 결과 파일 추가
if [ -f "results/training_output_${TIMESTAMP}.txt" ]; then
    git add "results/training_output_${TIMESTAMP}.txt"
    echo "학습 결과 파일이 추가되었습니다."
fi

# 학습 결과 요약 파일 추가
RESULTS_FILES=$(find "results" -name "training_results_*.txt" 2>/dev/null)
if [ -n "$RESULTS_FILES" ]; then
    git add $RESULTS_FILES
    echo "학습 결과 요약 파일이 추가되었습니다."
fi

# 평가 결과 파일 추가
if [ -f "results/evaluation_${TIMESTAMP}.txt" ]; then
    git add "results/evaluation_${TIMESTAMP}.txt"
    echo "평가 결과 파일이 추가되었습니다."
fi

# 모델 파일 추가 (용량이 클 수 있으므로 선택적)
MODEL_FILES=$(find "models" -name "ppo_mask_*.zip" -newer "results/training_output_${TIMESTAMP}.txt" 2>/dev/null || true)
if [ -n "$MODEL_FILES" ]; then
    echo "새로운 모델 파일 발견, 추가 여부 확인 중..."
    # 파일 크기 확인 (100MB 이하만 추가)
    for model_file in $MODEL_FILES; do
        if [ -f "$model_file" ]; then
            file_size=$(stat -c%s "$model_file" 2>/dev/null || stat -f%z "$model_file" 2>/dev/null || echo 0)
            if [ "$file_size" -lt 104857600 ]; then  # 100MB
                git add "$model_file"
                echo "모델 파일 추가됨: $model_file (크기: $file_size bytes)"
            else
                echo "모델 파일이 너무 큽니다: $model_file (크기: $file_size bytes)"
            fi
        fi
    done
fi

# PNG/GIF 파일 추가 (기존 로직 유지)
PNG_FILES=$(find "results" -name "*.png" 2>/dev/null)
if [ -n "$PNG_FILES" ]; then
    git add $PNG_FILES
    echo "PNG 파일이 추가되었습니다."
fi

GIF_FILES=$(find "gifs" -name "*.gif" -newer "results/training_output_${TIMESTAMP}.txt" 2>/dev/null || true)
if [ -n "$GIF_FILES" ]; then
    git add $GIF_FILES
    echo "GIF 파일이 추가되었습니다."
fi

# 로그 파일 추가
LOG_FILES=$(find "logs" -name "*${TIMESTAMP}*" 2>/dev/null)
if [ -n "$LOG_FILES" ]; then
    git add $LOG_FILES
    echo "로그 파일이 추가되었습니다."
fi

# 변경사항이 있는지 확인 (기존 로직 유지)
if git diff --cached --quiet; then
    echo "=== 변경사항 없음 ==="
else
    # 커밋 및 푸시 (실험 모드에 따라 메시지 구분)
    git commit -m "Maskable PPO 3D-BinPacking 실험 및 평가 결과 업데이트: ${TIMESTAMP}"
    git push origin "${BRANCH}"
    echo "=== 결과 업로드 완료 ==="
fi

# 시스템 정보 로깅
echo "=== 시스템 정보 ==="
echo "Python 버전: $(python --version)"
echo "uv 버전: $(uv --version)"
echo "Git 브랜치: $(git branch --show-current)"
echo "작업 디렉토리: $(pwd)"
if nvidia-smi > /dev/null 2>&1; then
    echo "GPU 정보:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
fi

echo "=== KAMP 서버 작업 완료: $(date '+%Y-%m-%d %H:%M:%S') ===" 