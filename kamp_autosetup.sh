#!/bin/bash
# KAMP 서버용 Maskable PPO 3D Bin Packing 실행을 위한 개발환경의 자동 구축 스크립트
# 기존 auto_run.sh 기반, uv를 활용한 의존성 관리와 GPU/CPU 자동선택 기능 및
# narwhals 에러 해결 스크립트 추가
# 사용법: ./kamp_autosetup.sh [브랜치명]   <- 기본값은 master

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