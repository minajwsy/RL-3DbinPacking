# 수동 실행 가이드

## 로컬에서 직접 훈련

### Windows 환경
```cmd
# Conda 환경 활성화
conda activate ppo-3dbp

# Python 경로 설정
set PYTHONPATH=%cd%\src;%PYTHONPATH%

# 디바이스 정보 확인
python -c "from src.device_utils import print_device_info; print_device_info()"

# 훈련 실행 (CPU 모드)
set FORCE_CPU=true
set TOTAL_TIMESTEPS=10000
python -m src.train

# 모델 평가
python -m src.eval --model_path models/ppo_3dbp_latest.zip
```

### Mac/Linux 환경
```bash
# Conda 환경 활성화
conda activate ppo-3dbp

# Python 경로 설정
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# 디바이스 정보 확인
python -c "from src.device_utils import print_device_info; print_device_info()"

# 훈련 실행 (GPU 자동 감지)
export TOTAL_TIMESTEPS=50000
python -m src.train

# 모델 평가
python -m src.eval --model_path models/ppo_3dbp_latest.zip
```

## KAMP 서버에서 수동 실행

```bash
# 서버 접속
ssh username@kamp-server

# 프로젝트 디렉토리 이동
cd ~/RL-3DbinPacking

# Git 업데이트
git pull origin master

# 의존성 설치
uv sync

# 환경 변수 설정
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export TOTAL_TIMESTEPS=100000
export CUDA_VISIBLE_DEVICES=0

# GPU 상태 확인
nvidia-smi

# 훈련 실행
python -m src.train

# 결과 업로드
git add results/ models/ gifs/
git commit -m "훈련 결과 업데이트: $(date)"
git push origin master
```

## 환경별 실행 옵션

### 빠른 테스트 (로컬)
```bash
# 환경 활성화
conda activate ppo-3dbp

# 빠른 테스트 설정
export FORCE_CPU=true
export TOTAL_TIMESTEPS=1000

# 실행
python -m src.train
```

### 전체 훈련 (KAMP 서버)
```bash
# 환경 설정
export TOTAL_TIMESTEPS=200000
export NUM_BOXES=64
export CONTAINER_SIZE="15,15,15"

# 실행
python -m src.train
```

### 모델 비교 평가
```bash
# 모든 모델 비교
python -m src.eval --compare

# 특정 모델 평가
python -m src.eval --model_path models/ppo_3dbp_20250101_120000.zip --num_episodes 50

# 렌더링과 함께 평가
python -m src.eval --model_path models/latest_model.zip --render
```

## 문제 해결

### 의존성 문제
```bash
# 환경 재생성
conda deactivate
conda remove -n ppo-3dbp --all -y
conda create -n ppo-3dbp python=3.8 -y
conda activate ppo-3dbp
uv sync
```

### GPU 문제
```bash
# GPU 상태 확인
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# CUDA 재설치 (필요시)
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Git 동기화 문제
```bash
# 강제 리셋
git fetch origin
git reset --hard origin/master
git clean -fd
```
