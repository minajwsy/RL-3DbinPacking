#!/bin/bash
# 로컬 환경용 3D Bin Packing PPO 개발 자동화 스크립트
# Windows/Mac 호환, conda 환경 사용

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 브랜치 설정
BRANCH=${1:-master}
CONDA_ENV_NAME="ppo-3dbp"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

echo -e "${BLUE}=== 로컬 PPO-3D Bin Packing 개발환경 시작: ${TIMESTAMP} ===${NC}"
echo -e "${YELLOW}브랜치: ${BRANCH}${NC}"

# 운영체제 감지
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    OS_TYPE="Windows"
    CONDA_CMD="conda.exe"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="Mac"
    CONDA_CMD="conda"
else
    OS_TYPE="Linux"
    CONDA_CMD="conda"
fi

echo -e "${BLUE}운영체제: ${OS_TYPE}${NC}"

# Conda 환경 확인 및 생성
check_conda_env() {
    echo -e "${BLUE}=== Conda 환경 확인 중 ===${NC}"
    
    if ! command -v ${CONDA_CMD} &> /dev/null; then
        echo -e "${RED}Conda가 설치되지 않았습니다. Conda를 먼저 설치하세요.${NC}"
        exit 1
    fi
    
    # 환경 존재 여부 확인
    if ${CONDA_CMD} env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo -e "${GREEN}Conda 환경 '${CONDA_ENV_NAME}' 이미 존재${NC}"
    else
        echo -e "${YELLOW}Conda 환경 '${CONDA_ENV_NAME}' 생성 중...${NC}"
        ${CONDA_CMD} create -n ${CONDA_ENV_NAME} python=3.8 -y
        echo -e "${GREEN}Conda 환경 생성 완료${NC}"
    fi
}

# uv 설치 및 의존성 관리
setup_uv_deps() {
    echo -e "${BLUE}=== uv 패키지 관리자 설정 ===${NC}"
    
    # Conda 환경 활성화
    source $(${CONDA_CMD} info --base)/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV_NAME}
    
    # uv 설치 확인
    if ! command -v uv &> /dev/null; then
        echo -e "${YELLOW}uv 설치 중...${NC}"
        pip install uv
    fi
    
    # 의존성 설치
    echo -e "${YELLOW}의존성 설치 중...${NC}"
    uv sync --dev
    
    # GPU 감지 및 PyTorch 설치
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}GPU 감지됨 - CUDA 버전 PyTorch 설치${NC}"
        uv add "torch>=1.11.0+cu118" "torchvision>=0.12.0+cu118" --index-url https://download.pytorch.org/whl/cu118
    else
        echo -e "${YELLOW}GPU 없음 - CPU 버전 PyTorch 사용${NC}"
        export FORCE_CPU=true
    fi
    
    echo -e "${GREEN}의존성 설치 완료${NC}"
}

# Git 동기화
sync_git() {
    echo -e "${BLUE}=== Git 저장소 동기화 ===${NC}"
    
    # Git 상태 확인
    if ! git rev-parse --is-inside-work-tree &> /dev/null; then
        echo -e "${RED}Git 저장소가 아닙니다.${NC}"
        exit 1
    fi
    
    # 원격 저장소 동기화
    git fetch origin --prune
    
    # 브랜치 체크아웃
    if git show-ref --verify --quiet refs/heads/${BRANCH}; then
        git checkout ${BRANCH}
        git pull origin ${BRANCH}
    else
        git checkout -b ${BRANCH}
    fi
    
    echo -e "${GREEN}Git 동기화 완료${NC}"
}

# 로컬 개발 환경 테스트
test_local_env() {
    echo -e "${BLUE}=== 로컬 환경 테스트 ===${NC}"
    
    # Conda 환경 활성화
    source $(${CONDA_CMD} info --base)/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV_NAME}
    
    # Python 경로 설정
    export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
    
    # 디바이스 정보 출력
    python -c "
import sys
sys.path.insert(0, 'src')
try:
    from device_utils import print_device_info
    print_device_info()
except ImportError as e:
    print(f'디바이스 유틸리티 로드 실패: {e}')
    print('기본 환경 정보:')
    print(f'Python 버전: {sys.version}')
"
    
    # 환경 검증
    python -c "
import sys
sys.path.insert(0, 'src')
try:
    import gym
    from src.packing_env import PackingEnv
    print('✅ 환경 로드 성공')
except ImportError as e:
    print(f'❌ 환경 로드 실패: {e}')
"
    
    echo -e "${GREEN}로컬 환경 테스트 완료${NC}"
}

# 코드 변경사항 커밋 및 푸시
commit_and_push() {
    echo -e "${BLUE}=== 변경사항 커밋 및 푸시 ===${NC}"
    
    # 변경사항 확인
    if git diff --quiet && git diff --staged --quiet; then
        echo -e "${YELLOW}변경사항 없음${NC}"
        return 0
    fi
    
    # 변경사항 추가
    git add .
    
    # 커밋 메시지 입력
    echo -e "${YELLOW}커밋 메시지를 입력하세요 (Enter 시 기본 메시지 사용):${NC}"
    read -r commit_message
    
    if [[ -z "$commit_message" ]]; then
        commit_message="로컬 개발 업데이트: ${TIMESTAMP}"
    fi
    
    git commit -m "$commit_message"
    git push origin ${BRANCH}
    
    echo -e "${GREEN}변경사항 푸시 완료${NC}"
}

# 클라우드 실행 대기
wait_for_cloud() {
    echo -e "${BLUE}=== 클라우드 실행 대기 ===${NC}"
    echo -e "${YELLOW}KAMP 서버에서 실행을 기다립니다...${NC}"
    echo -e "${YELLOW}실행이 완료되면 아무 키나 누르세요${NC}"
    read -n 1 -s -r -p ""
    echo
}

# 결과 가져오기
pull_results() {
    echo -e "${BLUE}=== 실행 결과 가져오기 ===${NC}"
    
    git pull origin ${BRANCH}
    
    # 결과 파일 확인
    echo -e "${BLUE}=== 생성된 결과 파일 ===${NC}"
    
    if [[ -d "results" ]]; then
        echo -e "${GREEN}Results 폴더:${NC}"
        ls -la results/ | head -10
    fi
    
    if [[ -d "models" ]]; then
        echo -e "${GREEN}Models 폴더:${NC}"
        ls -la models/ | head -5
    fi
    
    if [[ -d "gifs" ]]; then
        echo -e "${GREEN}GIFs 폴더:${NC}"
        ls -la gifs/*.gif 2>/dev/null | head -5 || echo "GIF 파일 없음"
    fi
    
    echo -e "${GREEN}결과 가져오기 완료${NC}"
}

# 메인 실행 함수
main() {
    echo -e "${GREEN}🚀 로컬 개발 환경 자동화 시작${NC}"
    
    check_conda_env
    setup_uv_deps
    sync_git
    test_local_env
    
    # 대화형 메뉴
    while true; do
        echo -e "\n${BLUE}=== 메뉴 선택 ===${NC}"
        echo "1) 변경사항 커밋 및 푸시"
        echo "2) 클라우드 실행 대기"
        echo "3) 결과 가져오기"
        echo "4) 전체 프로세스 (1→2→3)"
        echo "5) 종료"
        
        read -p "선택하세요 (1-5): " choice
        
        case $choice in
            1)
                commit_and_push
                ;;
            2)
                wait_for_cloud
                ;;
            3)
                pull_results
                ;;
            4)
                commit_and_push
                wait_for_cloud
                pull_results
                ;;
            5)
                echo -e "${GREEN}스크립트 종료${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}잘못된 선택입니다${NC}"
                ;;
        esac
    done
}

# 도움말
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "사용법: $0 [브랜치명]"
    echo ""
    echo "옵션:"
    echo "  브랜치명    사용할 Git 브랜치 (기본값: master)"
    echo "  -h, --help  이 도움말 표시"
    echo ""
    echo "예시:"
    echo "  $0            # master 브랜치 사용"
    echo "  $0 feature    # feature 브랜치 사용"
    exit 0
fi

# 스크립트 실행
main 