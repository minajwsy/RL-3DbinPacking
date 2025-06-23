#!/bin/bash
# KAMP 서버용 Maskable PPO 3D Bin Packing 자동 실행 스크립트
# 기존 auto_run.sh 기반, uv를 활용한 의존성 관리와 GPU/CPU 자동선택 기능 및
# narwhals 에러 해결 스크립트 추가
# 사용법: ./kamp_auto_run.sh [브랜치명]   <- 기본값은 master

# 오류 발생시엔 스크립트 중단  
set -e

# 브랜치 설정 (인자로 전달하거나 기본값 사용)
BRANCH=${1:-main}

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
    # curl -LsSf https://astral.sh/uv/install.sh | sh
    # source $HOME/.cargo/env
    wget -qO- https://astral.sh/uv/install.sh | sh 
    ls -al ~/.local/bin/uv 
    export PATH="$HOME/.local/bin:$PATH"    
fi

echo "uv 버전: $(uv --version)"

# Python 환경설정 및 가상환경 생성/활성화 확인
echo "=== Python 환경설정 및 가상환경 생성 중 ==="
uv python install 3.11      
uv venv .venv --python 3.11
source .venv/bin/activate
echo "현재 Python 환경: $VIRTUAL_ENV"
echo "Python 버전: $(python --version)"

# GPU 확인 및 적절한 의존성 설치
echo "=== GPU 상태 확인 및 의존성 설치 중 ==="

# uv.lock 파일 삭제하여 새로 생성하도록 함
echo "=== 기존 lock 파일 정리 중 ==="
rm -f uv.lock

if nvidia-smi > /dev/null 2>&1; then
    echo "GPU가 감지되었습니다. GPU 버전 패키지를 설치합니다."
    nvidia-smi
    
    # GPU 버전 PyTorch 설치
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # 필수 패키지 설치 (narwhals 포함)
    echo "=== 필수 패키지 개별 설치 중 ==="
    # 핵심 패키지들을 개별적으로 설치
    uv pip install gymnasium numpy nptyping pillow pandas jupyter pytest
    uv pip install sb3-contrib stable-baselines3[extra] tensorboard matplotlib opencv-python tqdm rich
    
    # narwhals 및 관련 의존성 설치
    echo "=== 누락된 의존성 설치 중 ==="
    uv pip install narwhals
    uv pip install tenacity packaging
    uv pip install rich tqdm

    # plotly 재설치 (의존성 포함)
    echo "=== plotly 재설치 중 ==="
    uv pip uninstall plotly -- yes || true
    uv pip install plotly

    # plotly_gif 설치 (누락된 의존성)
    echo "=== plotly_gif 설치 중 ==="
    uv pip install plotly_gif
    
    # stable-baselines3[extra] 재설치 (rich, tqdm 포함)
    #echo "=== stable-baselines3[extra] 재설치 중 ==="
    #uv pip uninstall stable-baselines3 -- yes || true
    #uv pip install stable-baselines3[extra]

    echo "GPU 환경 설정 완료 (narwhals 및 plotly_gif 포함)"
else
    echo "GPU를 찾을 수 없습니다. CPU 버전 패키지를 설치합니다."
    
    # CPU 버전 PyTorch 설치
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # 필수 패키지 설치 (narwhals 포함)
    echo "=== 필수 패키지 개별 설치 중 ==="
    # 핵심 패키지들을 개별적으로 설치
    uv pip install gymnasium numpy nptyping pillow pandas jupyter pytest
    uv pip install sb3-contrib stable-baselines3[extra] tensorboard matplotlib opencv-python tqdm rich
    
    # narwhals 및 관련 의존성 설치
    echo "=== 누락된 의존성 설치 중 ==="
    uv pip install narwhals
    uv pip install tenacity packaging
    uv pip install rich tqdm

    # plotly 재설치 (의존성 포함)
    echo "=== plotly 재설치 중 ==="
    uv pip uninstall plotly -- yes || true
    uv pip install plotly

    # plotly_gif 설치 (누락된 의존성)
    echo "=== plotly_gif 설치 중 ==="
    uv pip install plotly_gif

    # stable-baselines3[extra] 재설치 (rich, tqdm 포함)
    #echo "=== stable-baselines3[extra] 재설치 중 ==="
    #uv pip uninstall stable-baselines3 -- yes || true
    #uv pip install stable-baselines3[extra]
    
    echo "CPU 환경 설정 완료 (narwhals 및 plotly_gif 포함)"
fi

# 의존성 설치완료 확인
echo "=== 의존성 설치완료 확인 중 ==="
python -c "
try:
    import narwhals
    print('✅ narwhals 설치 확인')
    print(f'   버전: {narwhals.__version__}')
except ImportError as e:
    print(f'❌ narwhals 설치 실패: {e}')

try:
    import plotly
    print('✅ plotly 설치 확인')
    print(f'   버전: {plotly.__version__}')
except ImportError as e:
    print(f'❌ plotly 설치 실패: {e}')

try:
    import plotly.express as px
    print('✅ plotly.express 임포트 성공')
except ImportError as e:
    print(f'❌ plotly.express 임포트 실패: {e}')

try:
    from _plotly_utils.basevalidators import ColorscaleValidator
    print('✅ _plotly_utils.basevalidators 임포트 성공')
except ImportError as e:
    print(f'❌ _plotly_utils.basevalidators 임포트 실패: {e}')

try:
    import plotly_gif
    print('✅ plotly_gif 설치 확인')
    print(f'   버전: {plotly_gif.__version__}')
except ImportError as e:
    print(f'❌ plotly_gif 설치 실패: {e}')
except AttributeError:
    print('✅ plotly_gif 설치 확인 (버전 정보 없음)')
"

# 학습 스크립트 테스트
echo "=== 학습 스크립트 임포트 테스트 ==="
python -c "
try:
    import sys
    sys.path.append('src')
    from packing_kernel import *
    print('✅ packing_kernel 임포트 성공')
except ImportError as e:
    print(f'❌ packing_kernel 임포트 실패: {e}')

try:
    from src.train_maskable_ppo import *
    print('✅ train_maskable_ppo 임포트 성공')
except ImportError as e:
    print(f'❌ train_maskable_ppo 임포트 실패: {e}')
"

echo "=== narwhals 에러 해결 완료 ==="
echo "이제 다음 명령어로 학습을 시작 :"
echo "python -m src.ultimate_train_fix --timesteps 20000 --eval-freq 1500 --container-size 10 10 10 --num-boxes 32 --curriculum-learning --improved-rewards"

# Python 경로 설정
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# 디렉토리 생성
mkdir -p models results logs gifs

# Maskable PPO 실험 실행
echo "=== 개선된 Maskable PPO 3D Bin Packing 학습 시작: $(date '+%Y-%m-%d %H:%M:%S') ==="

# 999 스텝 문제 완전 해결을 위한 Ultimate 학습 실행
# PYTHONPATH 설정으로 환경 등록 문제 해결
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "🚀 999 스텝 문제 완전 해결 학습 시작"
echo "📋 설정: 안전한 평가 주기 사용, GIF 및 성능 그래프 생성"

if python src.ultimate_train_fix.py \
    --timesteps 30000 \
    --eval-freq 2500 \
    --num-boxes 18 \
    2>&1 | tee results/ultimate_training_output_${TIMESTAMP}.txt; then
    echo "✅ Ultimate 학습이 성공적으로 완료되었습니다."
    
    # 학습 성과 요약 출력
    echo "=== Ultimate 학습 결과 요약 ==="
    if ls results/ultimate_results_*.txt >/dev/null 2>&1; then
        echo "📊 최신 학습 요약:"
        tail -20 $(ls -t results/ultimate_results_*.txt | head -1)
    fi
    
    # 성능 대시보드 확인
    if ls results/ultimate_dashboard_*.png >/dev/null 2>&1; then
        echo "📈 최종 성과 대시보드 생성됨:"
        ls -la results/ultimate_dashboard_*.png | tail -1
    fi
    
    # GIF 파일 확인
    if ls gifs/ultimate_demo_*.gif >/dev/null 2>&1; then
        echo "🎬 학습 데모 GIF 생성됨:"
        ls -la gifs/ultimate_demo_*.gif | tail -1
    fi
    
else
    echo "❌ Ultimate 학습 실행 중 오류가 발생했습니다."
    echo "🔄 대안으로 콜백 없는 학습을 시도합니다..."
    
    # 대안: 콜백 없는 순수 학습
    if python src.no_callback_train.py \
        --timesteps 8000 \
        --num-boxes 20 \
        2>&1 | tee results/fallback_training_output_${TIMESTAMP}.txt; then
        echo "✅ 대안 학습이 성공적으로 완료되었습니다."
    else
        echo "❌ 대안 학습도 실패했습니다. 로그를 확인하세요."
    fi
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
if [ -f "results/ultimate_training_output_${TIMESTAMP}.txt" ]; then
    git add "results/ultimate_training_output_${TIMESTAMP}.txt"
    echo "학습 결과 파일이 추가되었습니다."
fi

# 학습 결과 요약 파일 추가
RESULTS_FILES=$(find "results" -name "ultimate_results_*.txt" 2>/dev/null)
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
MODEL_FILES=$(find "models" -name "ppo_mask_*.zip" -newer "results/ultimate_training_output_${TIMESTAMP}.txt" 2>/dev/null || true)
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

GIF_FILES=$(find "gifs" -name "*.gif" -newer "results/ultimate_training_output_${TIMESTAMP}.txt" 2>/dev/null || true)
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