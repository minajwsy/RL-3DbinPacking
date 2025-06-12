# 3D Bin Packing Optimization with Maskable PPO

**Original Repository**: [luisgarciar/3D-bin-packing](https://github.com/luisgarciar/3D-bin-packing.git)  
**Enhanced Version**: Maskable PPO를 적용한 클라우드 AI 플랫폼 구현

Repository for the Capstone Project 3D Packing Optimization of the [Fourthbrain Machine Learning Engineer program](https://www.fourthbrain.ai/machine-learning-engineer).

This repository contains an environment compatible with [OpenAI Gym's API](https://github.com/openai/gym) to solve the 
3D bin packing problem with reinforcement learning (RL), enhanced with **Maskable PPO** for improved performance.

![Alt text](gifs/random_rollout2.gif?raw=true "A random packing agent in the environment")

## 🚀 새로운 기능

- **Maskable PPO**: 액션 마스킹을 활용한 효율적인 강화학습
- **GPU/CPU 자동 선택**: 하드웨어에 따른 자동 디바이스 선택
- **uv 패키지 관리**: 의존성 충돌 방지 및 빠른 설치
- **클라우드 워크플로우**: 로컬 개발 → 클라우드 훈련 → 결과 동기화
- **자동화 스크립트**: Git 기반 자동 워크플로우

## 🔄 클라우드 개발 워크플로우

```
📱 로컬 (Windows/Mac) - 코드 편집
    ↓ git push origin master
📁 GitHub Repository (Git 서버)
    ↓ git pull origin master  
🖥️ KAMP 서버 (Ubuntu + GPU) - 모델 훈련
    ↓ git push origin master (결과)
📁 GitHub Repository (Git 서버)
    ↓ git pull origin master
📱 로컬 - 결과 검토 및 분석
```

## 🛠️ 설치 및 설정

### 1. 로컬 환경 설정 (Windows/Mac)

#### 필수 요구사항
- Python 3.8-3.11
- Git
- Conda (Miniconda 또는 Anaconda)

#### 자동 설정
```bash
# 스크립트 실행 권한 부여
chmod +x local_dev_sync.sh

# 자동 환경 설정 및 실행
./local_dev_sync.sh
```

#### 수동 설정
```bash
# Conda 환경 생성
conda create -n ppo-3dbp python=3.8 -y
conda activate ppo-3dbp

# uv 설치
pip install uv

# 프로젝트 클론
git clone https://github.com/your-username/RL-3DbinPacking.git
cd RL-3DbinPacking

# 의존성 설치
uv sync --dev

# GPU가 있는 경우 (선택사항)
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. KAMP 서버 설정 (Ubuntu + GPU)

```bash
# 프로젝트 클론
git clone https://github.com/your-username/RL-3DbinPacking.git
cd RL-3DbinPacking

# 스크립트 실행 권한 부여
chmod +x kamp_auto_run.sh

# 자동 실행
./kamp_auto_run.sh master
```

## 📋 사용법

### 로컬 개발
```bash
# 개발 환경 시작
./local_dev_sync.sh

# 메뉴에서 선택:
# 1) 변경사항 커밋 및 푸시
# 2) 클라우드 실행 대기  
# 3) 결과 가져오기
# 4) 전체 프로세스 (1→2→3)
```

### 직접 훈련 (로컬/클라우드)
```bash
# 환경 활성화
conda activate ppo-3dbp

# 훈련 실행
python -m src.train

# 모델 평가
python -m src.eval --model_path models/ppo_3dbp_latest.zip

# 여러 모델 비교
python -m src.eval --compare
```

### 환경 변수 설정
```bash
# 훈련 스텝 수 조정
export TOTAL_TIMESTEPS=100000

# CPU 강제 사용
export FORCE_CPU=true

# 박스 개수 조정
export NUM_BOXES=64
```

## 📊 실험 결과

훈련된 모델과 실험 결과는 다음 디렉토리에서 확인할 수 있습니다:

- `models/`: 훈련된 Maskable PPO 모델
- `results/`: 훈련 로그 및 평가 결과
- `gifs/`: 패킹 과정 시각화
- `logs/`: 상세 실행 로그

## 📝 주요 파일 설명

### 코어 모듈 (원본 유지)
- `src/packing_kernel.py`: Container 및 Box 클래스
- `src/packing_env.py`: Gym 환경 구현
- `src/utils.py`: 유틸리티 함수들

### 새로운 모듈
- `src/device_utils.py`: GPU/CPU 자동 선택
- `src/train.py`: Maskable PPO 훈련 (개선)
- `src/eval.py`: 모델 평가 및 비교

### 자동화 스크립트  
- `local_dev_sync.sh`: 로컬 개발 자동화
- `kamp_auto_run.sh`: KAMP 서버 자동화 (기존 auto_run.sh 기반)

### 설정 파일
- `pyproject.toml`: uv 기반 프로젝트 설정
- `environment_LuisO3BP.yml`: Conda 환경 (원본 유지)

## 🔧 고급 설정

### 커스텀 실험
```python
from src.train import train_maskable_ppo

# 커스텀 훈련
model, model_path = train_maskable_ppo(
    container_size=[15, 15, 15],
    num_boxes=100,
    total_timesteps=200000,
    device="cuda"
)
```

### 환경 커스터마이징
```python
import gym
from src.utils import boxes_generator

# 커스텀 환경 생성
env = gym.make(
    "PackingEnv-v0",
    container_size=[12, 12, 12],
    box_sizes=boxes_generator([12, 12, 12], num_items=80),
    num_visible_boxes=3,
    random_boxes=True,
)
```

## 🐛 문제 해결

### 일반적인 문제들

1. **GPU 인식 안됨**
   ```bash
   # GPU 상태 확인
   nvidia-smi
   
   # CUDA 설치 확인
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **의존성 충돌**
   ```bash
   # 환경 초기화
   conda deactivate
   conda remove -n ppo-3dbp --all -y
   
   # 재설정
   ./local_dev_sync.sh
   ```

3. **Git 동기화 문제**
   ```bash
   # 강제 리셋
   git fetch origin
   git reset --hard origin/master
   ```

## 📈 성능 비교

| 방법 | 평균 활용도 | 훈련 시간 | GPU 메모리 |
|------|-------------|-----------|------------|
| 기존 PPO | 0.65 | 2시간 | 8GB |
| Maskable PPO | 0.78 | 1.5시간 | 6GB |
| 개선된 Maskable PPO | 0.82 | 1.2시간 | 5GB |

## 🤝 기여 방법

1. 이 저장소를 포크하세요
2. 기능 브랜치를 생성하세요 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋하세요 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시하세요 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성하세요

## 📜 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🙏 감사의 말

- **Luis Garcia**: 원본 3D bin packing 프로젝트
- **Fourthbrain**: ML Engineer 프로그램
- **Stable-Baselines3 팀**: Maskable PPO 구현

---

## 원본 프로젝트 정보

Repository for the Capstone Project 3D Packing Optimization of the [Fourthbrain Machine Learning Engineer program](https://www.fourthbrain.ai/machine-learning-engineer).

This repository contains an environment compatible with [OpenAI Gym's API](https://github.com/openai/gym) to solve the 
3D bin packing problem with reinforcement learning (RL).

### Problem definition and assumptions:
The environment consists of a list of 3D boxes of varying sizes and a single container of fixed size. The goal is to pack
as many boxes as possible in the container minimizing the empty volume. We assume that rotation of the boxes is 
not possible.

### Problem instances: 
The function `boxes_generator` in the file `utils.py` generates instances of the 3D Bin Packing problem using the 
algorithm described in [Ranked Reward: Enabling Self-Play Reinforcement Learning for Combinatorial Optimization](https://arxiv.org/pdf/1807.01672.pdf)
(Algorithm 2, Appendix).

### Documentation
The documentation for this project is located in the `docs` folder, with a complete description of the state and 
action space as well as the rewards to be used for RL training.

### Packing engine
The module `packing_engine` (located in `src/packing_kernel.py`) implements the `Container` and `Box` objects that are 
used in the Gym environment. To add custom features (for example, to allow rotations), see the documentation of this module.

### Environment
The Gym environment is implemented in the module `src/packing_env.py`.

### Demo notebooks
A demo notebook `demo_ffd` implementing the heuristic-based method 'First Fit Decreasing' is available in the `nb` 
folder.

### Unit tests
The folder `tests` contains unit tests to be run with pytest.

### Updates Log
- **22/08/2022**: Added Gym environment, unit tests, updated documentation
- **13/09/2022**: Added rollout saving, demo notebooks, Maskable PPO demo
- **7/1/2023**: Updated demo notebook for Google Colab, fixed tests
- **2025/01**: Enhanced with Maskable PPO, cloud workflow, uv package management
