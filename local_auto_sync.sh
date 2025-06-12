#!/bin/bash
# 로컬 환경용 자동 동기화 스크립트 (Windows/Mac 호환)
# Maskable PPO 3D Bin Packing 개발환경 자동화
# 사용법: ./local_auto_sync.sh [브랜치명] [--setup]

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수 정의
print_color() {
    printf "${2}${1}${NC}\n"
}

print_header() {
    echo ""
    print_color "============================================" $BLUE
    print_color "$1" $BLUE
    print_color "============================================" $BLUE
}

print_success() {
    print_color "✓ $1" $GREEN
}

print_warning() {
    print_color "⚠ $1" $YELLOW
}

print_error() {
    print_color "✗ $1" $RED
}

# 운영체제 감지
detect_os() {
    case "$(uname -s)" in
        Darwin*) echo "mac" ;;
        Linux*)  echo "linux" ;;
        CYGWIN*|MINGW*|MSYS*) echo "windows" ;;
        *) echo "unknown" ;;
    esac
}

# 초기 설정
OS=$(detect_os)
BRANCH=${1:-master}
SETUP_MODE=false

# 파라미터 처리
for arg in "$@"; do
    case $arg in
        --setup)
            SETUP_MODE=true
            shift
            ;;
    esac
done

# 환경 변수 설정
WORK_DIR=$(pwd)
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

print_header "로컬 Maskable PPO 3D Bin Packing 개발환경 시작"
print_color "운영체제: $OS" $BLUE
print_color "브랜치: $BRANCH" $BLUE
print_color "작업 디렉토리: $WORK_DIR" $BLUE
print_color "타임스탬프: $TIMESTAMP" $BLUE

# uv 설치 확인
check_uv() {
    print_header "uv 패키지 매니저 확인"
    
    if command -v uv &> /dev/null; then
        UV_VERSION=$(uv --version)
        print_success "uv가 이미 설치되어 있습니다: $UV_VERSION"
        return 0
    else
        print_warning "uv가 설치되어 있지 않습니다."
        
        if [ "$SETUP_MODE" = true ]; then
            print_color "uv 설치를 시작합니다..." $YELLOW
            
            case $OS in
                "mac")
                    curl -LsSf https://astral.sh/uv/install.sh | sh
                    ;;
                "linux")
                    curl -LsSf https://astral.sh/uv/install.sh | sh
                    ;;
                "windows")
                    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
                    ;;
                *)
                    print_error "지원되지 않는 운영체제입니다: $OS"
                    return 1
                    ;;
            esac
            
            # PATH 설정
            export PATH="$HOME/.local/bin:$PATH"
            
            if command -v uv &> /dev/null; then
                print_success "uv 설치가 완료되었습니다."
                return 0
            else
                print_error "uv 설치에 실패했습니다."
                return 1
            fi
        else
            print_error "uv가 필요합니다. --setup 옵션으로 다시 실행하거나 수동으로 설치하세요."
            return 1
        fi
    fi
}

# Python 환경 설정
setup_python_env() {
    print_header "Python 환경 설정"
    
    # Python 3.9 설치 (필요시)
    if ! uv python list | grep -q "3.9"; then
        print_color "Python 3.9 설치 중..." $YELLOW
        uv python install 3.9
    fi
    
    # 가상환경 생성 또는 활성화
    if [ ! -d ".venv" ]; then
        print_color "가상환경 생성 중..." $YELLOW
        uv venv .venv --python 3.9
    fi
    
    # 가상환경 활성화 스크립트
    if [ "$OS" = "windows" ]; then
        VENV_ACTIVATE=".venv/Scripts/activate"
    else
        VENV_ACTIVATE=".venv/bin/activate"
    fi
    
    if [ -f "$VENV_ACTIVATE" ]; then
        source "$VENV_ACTIVATE"
        print_success "가상환경 활성화 완료"
    else
        print_error "가상환경 활성화 스크립트를 찾을 수 없습니다."
        return 1
    fi
    
    # 의존성 설치
    print_color "의존성 설치 중..." $YELLOW
    
    # CPU 버전 PyTorch 설치 (로컬은 보통 CPU)
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # 기본 의존성 설치
    if [ -f "pyproject.toml" ]; then
        uv sync
        print_success "의존성 설치 완료 (pyproject.toml 기반)"
    elif [ -f "requirements.txt" ]; then
        uv pip install -r requirements.txt
        print_success "의존성 설치 완료 (requirements.txt 기반)"
    else
        print_warning "의존성 파일을 찾을 수 없습니다. 수동으로 설치가 필요할 수 있습니다."
    fi
}

# Git 상태 확인 및 동기화
sync_git() {
    print_header "Git 저장소 동기화"
    
    # Git 저장소 확인
    if [ ! -d ".git" ]; then
        print_error "Git 저장소가 아닙니다. git init을 먼저 실행하세요."
        return 1
    fi
    
    # 현재 상태 확인
    print_color "현재 Git 상태 확인 중..." $YELLOW
    git status --porcelain > /tmp/git_status_$$
    
    if [ -s /tmp/git_status_$$ ]; then
        print_warning "변경사항이 있습니다:"
        cat /tmp/git_status_$$
        
        read -p "변경사항을 커밋하시겠습니까? (y/n): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git add .
            
            # 커밋 메시지 입력
            echo "커밋 메시지를 입력하세요 (기본값: 로컬 개발 업데이트 $TIMESTAMP):"
            read -r COMMIT_MSG
            
            if [ -z "$COMMIT_MSG" ]; then
                COMMIT_MSG="로컬 개발 업데이트: $TIMESTAMP"
            fi
            
            git commit -m "$COMMIT_MSG"
            print_success "변경사항이 커밋되었습니다."
        fi
    else
        print_success "변경사항이 없습니다."
    fi
    
    rm -f /tmp/git_status_$$
    
    # 원격 저장소 동기화
    print_color "원격 저장소와 동기화 중..." $YELLOW
    
    # 현재 브랜치 확인
    CURRENT_BRANCH=$(git branch --show-current)
    
    if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
        print_color "브랜치를 $BRANCH로 전환합니다..." $YELLOW
        git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH"
    fi
    
    # 원격에서 변경사항 가져오기
    if git fetch origin "$BRANCH" 2>/dev/null; then
        if git diff --quiet HEAD "origin/$BRANCH" 2>/dev/null; then
            print_success "원격 저장소와 동기화되어 있습니다."
        else
            print_warning "원격 저장소에 새로운 변경사항이 있습니다."
            
            read -p "원격 변경사항을 가져오시겠습니까? (y/n): " -n 1 -r
            echo
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                git pull origin "$BRANCH"
                print_success "원격 변경사항을 가져왔습니다."
            fi
        fi
    else
        print_warning "원격 저장소에 접근할 수 없거나 브랜치가 존재하지 않습니다."
    fi
}

# 로컬 개발 환경 테스트
test_local_env() {
    print_header "로컬 개발 환경 테스트"
    
    # Python 환경 테스트
    print_color "Python 환경 테스트 중..." $YELLOW
    
    python -c "
import sys
print(f'Python 버전: {sys.version}')

try:
    import torch
    print(f'PyTorch 버전: {torch.__version__}')
    print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
except ImportError:
    print('PyTorch가 설치되지 않았습니다.')

try:
    import gym
    print(f'Gym 설치됨')
except ImportError:
    print('Gym이 설치되지 않았습니다.')

try:
    from sb3_contrib import MaskablePPO
    print('Maskable PPO 사용 가능')
except ImportError:
    print('Maskable PPO가 설치되지 않았습니다.')

# 프로젝트 모듈 테스트
sys.path.append('src')
try:
    from src.device_utils import get_device, log_system_info
    print('디바이스 유틸리티 사용 가능')
    log_system_info()
except ImportError as e:
    print(f'프로젝트 모듈 로드 실패: {e}')
"
    
    if [ $? -eq 0 ]; then
        print_success "로컬 개발 환경 테스트 통과"
    else
        print_error "로컬 개발 환경 테스트 실패"
        return 1
    fi
}

# 간단한 학습 테스트 실행
run_quick_test() {
    print_header "간단한 학습 테스트"
    
    read -p "간단한 학습 테스트를 실행하시겠습니까? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_color "간단한 학습 테스트 실행 중..." $YELLOW
        
        if python -m src.train_maskable_ppo --timesteps 10 --force-cpu --no-gif; then
            print_success "간단한 학습 테스트 완료"
        else
            print_error "학습 테스트 실패"
            return 1
        fi
    fi
}

# KAMP 서버 실행 대기
wait_for_kamp() {
    print_header "KAMP 서버 실행 대기"
    
    print_color "로컬 작업이 완료되었습니다." $GREEN
    print_color "이제 KAMP 서버에서 다음 명령을 실행하세요:" $YELLOW
    print_color "" $YELLOW
    print_color "  chmod +x kamp_auto_run.sh" $YELLOW
    print_color "  ./kamp_auto_run.sh $BRANCH" $YELLOW
    print_color "" $YELLOW
    
    read -p "KAMP 서버 실행이 완료되면 Enter를 누르세요..."
    
    print_color "결과를 가져오는 중..." $YELLOW
    
    # 결과 가져오기
    if git pull origin "$BRANCH"; then
        print_success "KAMP 서버 실행 결과를 가져왔습니다."
        
        # 결과 파일 확인
        if [ -d "results" ] && [ "$(ls -A results)" ]; then
            print_success "실행 결과 파일들:"
            ls -la results/
        fi
        
        if [ -d "models" ] && [ "$(ls -A models)" ]; then
            print_success "모델 파일들:"
            ls -la models/
        fi
        
        if [ -d "gifs" ] && [ "$(ls -A gifs)" ]; then
            print_success "GIF 파일들:"
            ls -la gifs/
        fi
        
    else
        print_error "결과를 가져오는 데 실패했습니다."
        return 1
    fi
}

# 메인 실행 함수
main() {
    # uv 확인 및 설치
    if ! check_uv; then
        exit 1
    fi
    
    # 설정 모드일 때만 환경 설정
    if [ "$SETUP_MODE" = true ]; then
        setup_python_env
        if [ $? -ne 0 ]; then
            print_error "Python 환경 설정에 실패했습니다."
            exit 1
        fi
        
        test_local_env
        if [ $? -ne 0 ]; then
            print_error "환경 테스트에 실패했습니다."
            exit 1
        fi
    fi
    
    # Git 동기화
    sync_git
    if [ $? -ne 0 ]; then
        print_error "Git 동기화에 실패했습니다."
        exit 1
    fi
    
    # 설정 모드가 아닐 때는 간단 테스트와 KAMP 대기
    if [ "$SETUP_MODE" = false ]; then
        run_quick_test
        wait_for_kamp
    fi
    
    print_header "로컬 자동화 스크립트 완료"
    print_success "모든 작업이 완료되었습니다!"
}

# 도움말 표시
show_help() {
    echo "사용법: $0 [브랜치명] [옵션]"
    echo ""
    echo "옵션:"
    echo "  --setup    초기 환경 설정 (uv 설치, Python 환경 설정 등)"
    echo "  --help     이 도움말 표시"
    echo ""
    echo "예시:"
    echo "  $0                    # master 브랜치로 동기화"
    echo "  $0 develop            # develop 브랜치로 동기화"  
    echo "  $0 master --setup     # 초기 환경 설정 후 동기화"
    echo ""
    echo "워크플로우:"
    echo "  1. 로컬에서 코드 개발 및 수정"
    echo "  2. ./local_auto_sync.sh 실행"
    echo "  3. KAMP 서버에서 ./kamp_auto_run.sh 실행"
    echo "  4. 결과 확인 및 반복"
}

# 파라미터 확인
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# 메인 실행
main 