@echo off
setlocal enabledelayedexpansion

REM Windows용 PPO-3D Bin Packing 개발 자동화 스크립트
echo === 로컬 PPO-3D Bin Packing 개발환경 시작 ===

REM 브랜치 설정
set BRANCH=%1
if "%BRANCH%"=="" set BRANCH=master

REM 환경 변수 설정
set CONDA_ENV_NAME=ppo-3dbp
set TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

echo 브랜치: %BRANCH%
echo 타임스탬프: %TIMESTAMP%

REM Conda 환경 활성화 확인
call conda activate %CONDA_ENV_NAME%
if %errorLevel% neq 0 (
    echo Conda 환경 '%CONDA_ENV_NAME%'이 없습니다.
    echo setup_local_env.bat를 먼저 실행하세요.
    pause
    exit /b 1
)

echo Conda 환경 활성화 완료

REM Git 상태 확인
git status >nul 2>&1
if %errorLevel% neq 0 (
    echo Git 저장소가 아닙니다.
    pause
    exit /b 1
)

REM Git 동기화
echo === Git 저장소 동기화 ===
git fetch origin
git checkout %BRANCH%
git pull origin %BRANCH%

REM 의존성 업데이트
echo === 의존성 업데이트 확인 ===
call uv sync

REM 환경 테스트
echo === 로컬 환경 테스트 ===
set PYTHONPATH=%cd%\src;%PYTHONPATH%

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

REM 메뉴 루프
:menu
echo.
echo === 메뉴 선택 ===
echo 1) 변경사항 커밋 및 푸시
echo 2) 클라우드 실행 대기
echo 3) 결과 가져오기
echo 4) 전체 프로세스 (1→2→3)
echo 5) 로컬에서 직접 훈련
echo 6) 종료

set /p choice="선택하세요 (1-6): "

if "%choice%"=="1" goto commit_push
if "%choice%"=="2" goto wait_cloud
if "%choice%"=="3" goto pull_results
if "%choice%"=="4" goto full_process
if "%choice%"=="5" goto local_train
if "%choice%"=="6" goto end
echo 잘못된 선택입니다.
goto menu

:commit_push
echo === 변경사항 커밋 및 푸시 ===
git status
git add .

git diff --cached --quiet
if %errorLevel% equ 0 (
    echo 변경사항 없음
    goto menu
)

set /p commit_message="커밋 메시지를 입력하세요 (Enter 시 기본 메시지): "
if "%commit_message%"=="" set commit_message=로컬 개발 업데이트: %TIMESTAMP%

git commit -m "%commit_message%"
git push origin %BRANCH%
echo 변경사항 푸시 완료
goto menu

:wait_cloud
echo === 클라우드 실행 대기 ===
echo KAMP 서버에서 실행을 기다립니다...
echo 실행이 완료되면 아무 키나 누르세요
pause
goto menu

:pull_results
echo === 실행 결과 가져오기 ===
git pull origin %BRANCH%

echo === 생성된 결과 파일 ===
if exist "results" (
    echo Results 폴더:
    dir results /b | head -10
)

if exist "models" (
    echo Models 폴더:
    dir models /b | head -5
)

if exist "gifs" (
    echo GIFs 폴더:
    dir gifs\*.gif /b 2>nul | head -5
)

echo 결과 가져오기 완료
goto menu

:full_process
call :commit_push
call :wait_cloud
call :pull_results
goto menu

:local_train
echo === 로컬에서 직접 훈련 ===
set FORCE_CPU=true
set TOTAL_TIMESTEPS=10000
echo CPU 모드로 빠른 테스트 훈련 시작...
python -m src.train
echo 로컬 훈련 완료
goto menu

:end
echo 스크립트 종료
pause 