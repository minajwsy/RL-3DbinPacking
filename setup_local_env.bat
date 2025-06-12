@echo off
REM Windows용 PPO-3D Bin Packing 환경 설정 스크립트
echo === PPO-3D Bin Packing 로컬 환경 설정 시작 ===

REM 관리자 권한 확인
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo 관리자 권한으로 실행해주세요.
    pause
    exit /b 1
)

REM Conda 설치 확인
where conda >nul 2>&1
if %errorLevel% neq 0 (
    echo Conda가 설치되지 않았습니다.
    echo https://docs.conda.io/en/latest/miniconda.html 에서 Miniconda를 설치하세요.
    pause
    exit /b 1
)

echo Conda 설치 확인됨

REM Conda 환경 생성
echo === Conda 환경 생성 중 ===
call conda create -n ppo-3dbp python=3.8 -y

REM 환경 활성화
echo === 환경 활성화 ===
call conda activate ppo-3dbp

REM uv 설치
echo === uv 패키지 매니저 설치 ===
pip install uv

REM PyTorch 설치 (GPU 감지)
echo === PyTorch 설치 ===
nvidia-smi >nul 2>&1
if %errorLevel% equ 0 (
    echo GPU 감지됨 - CUDA 버전 설치
    call uv add torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    echo GPU 없음 - CPU 버전 설치
    call uv add torch torchvision
)

REM 의존성 설치
echo === 프로젝트 의존성 설치 ===
call uv sync

REM 환경 확인
echo === 환경 설정 확인 ===
python -c "import torch; print(f'PyTorch 버전: {torch.__version__}'); print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"

echo === 환경 설정 완료 ===
echo 사용법: conda activate ppo-3dbp
pause 