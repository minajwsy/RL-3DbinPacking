# Maskable PPO 3D Bin Packing 개발환경 구축 가이드

이 가이드는 luisgarciar의 3D bin packing 오픈소스를 기반으로 Maskable PPO를 적용하고, 로컬 ↔ GitHub ↔ KAMP 서버 간의 자동화된 개발 워크플로우를 구축하는 방법을 설명합니다.

## 🏗️ 전체 아키텍처

```
📱 로컬 환경 (Windows/Mac + Cursor AI Editor)
    ↕️ Git Push/Pull
📁 GitHub Repository (중앙 Git 서버)
    ↕️ Git Push/Pull  
🖥️ KAMP 서버 (Ubuntu + GPU 클라우드 환경)
```

## 🚀 빠른 시작

### 1단계: 로컬 환경 초기 설정

```bash
# 프로젝트 클론
git clone <YOUR_REPO_URL>
cd RL-3DbinPacking

# 초기 환경 설정 (한 번만 실행)
./local_auto_sync.sh --setup
```

### 2단계: 개발 워크플로우

```bash
# 1. 로컬에서 코드 수정 후 동기화
./local_auto_sync.sh

# 2. KAMP 서버에서 학습 실행 (SSH로 접속 후)
./kamp_auto_run.sh

# 3. 결과 확인 (로컬에서)
git pull origin master
```

## 📋 상세 설치 가이드

### 로컬 환경 (Windows/Mac)

#### 필수 요구사항
- Git
- Bash 셸 (Windows: Git Bash 또는 WSL, Mac: 기본 터미널)
- Python 3.8-3.11

#### 설정 단계

1. **uv 패키지 매니저 설치**
   ```bash
   # Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Mac/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **프로젝트 환경 설정**
   ```bash
   # 초기 설정 (자동으로 uv 설치, Python 환경 구성, 의존성 설치)
   ./local_auto_sync.sh --setup
   ```

3. **환경 테스트**
   ```bash
   # 가상환경 활성화 (Windows)
   .venv\Scripts\activate
   
   # 가상환경 활성화 (Mac/Linux)
   source .venv/bin/activate
   
   # 환경 테스트
   python test_env.py
   ```

### KAMP 서버 환경 (Ubuntu + GPU)

#### 필수 요구사항
- Ubuntu 18.04+
- NVIDIA GPU (권장)
- CUDA Toolkit (GPU 사용시)
- Git

#### 설정 단계

1. **프로젝트 클론**
   ```bash
   git clone <YOUR_REPO_URL>
   cd RL-3DbinPacking
   ```

2. **스크립트 실행 권한 설정**
   ```bash
   chmod +x kamp_auto_run.sh
   ```

3. **자동 실행**
   ```bash
   # 자동으로 uv 설치, 환경 설정, GPU/CPU 감지, 학습 실행
   ./kamp_auto_run.sh
   ```

## 🛠️ 주요 컴포넌트

### 1. 디바이스 자동 선택 (`src/device_utils.py`)
- GPU/CPU 자동 감지
- 최적화된 학습 설정 제공
- 시스템 정보 로깅

### 2. Maskable PPO 학습 (`src/train_maskable_ppo.py`)
- 기존 골격 유지하면서 Maskable PPO 적용
- GPU/CPU 자동 선택
- 학습 결과 자동 저장

### 3. 자동화 스크립트
- `local_auto_sync.sh`: 로컬 환경용 자동 동기화
- `kamp_auto_run.sh`: KAMP 서버용 자동 실행

### 4. 환경 설정
- `pyproject.toml`: uv 기반 의존성 관리
- `requirements.txt`: 호환성 유지

## 📝 사용법

### 로컬 개발 워크플로우

1. **일반 동기화**
   ```bash
   ./local_auto_sync.sh
   ```

2. **다른 브랜치 사용**
   ```bash
   ./local_auto_sync.sh develop
   ```

3. **환경 재설정**
   ```bash
   ./local_auto_sync.sh --setup
   ```

### KAMP 서버 실행

1. **기본 실행**
   ```bash
   ./kamp_auto_run.sh
   ```

2. **특정 브랜치 실행**
   ```bash
   ./kamp_auto_run.sh develop
   ```

### 수동 학습 실행

```bash
# 간단한 테스트 (50 스텝)
python -m src.train_maskable_ppo --timesteps 50 --force-cpu --no-gif

# 전체 학습 (100,000 스텝)
python -m src.train_maskable_ppo --timesteps 100000

# GPU 강제 사용
python -m src.train_maskable_ppo --timesteps 100000

# CPU 강제 사용
python -m src.train_maskable_ppo --timesteps 100000 --force-cpu
```

## 🔧 설정 옵션

### 학습 파라미터

```bash
python -m src.train_maskable_ppo \
    --timesteps 100000 \
    --container-size 10 10 10 \
    --num-boxes 64 \
    --visible-boxes 3 \
    --seed 42 \
    --force-cpu \
    --no-gif \
    --eval-freq 10000
```

### 환경 변수

- `WORK_DIR`: 작업 디렉토리 (기본: `${HOME}/RL-3DbinPacking`)
- `REPO_URL`: Git 저장소 URL
- `BRANCH`: 사용할 브랜치 (기본: `master`)

## 📊 결과 확인

### 학습 결과 파일
- `results/training_results_*.txt`: 학습 요약
- `models/ppo_mask_*.zip`: 저장된 모델
- `logs/`: 상세 로그
- `gifs/`: 시각화 결과

### 모니터링
- TensorBoard: `logs/tensorboard/`
- 실시간 로그: `logs/kamp_auto_run_*.log`

## 🚨 문제 해결

### 일반적인 문제

1. **uv 설치 실패**
   ```bash
   # 수동 설치
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **GPU 인식 안됨**
   ```bash
   # CUDA 설치 확인
   nvidia-smi
   
   # PyTorch CUDA 버전 확인
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **환경 충돌**
   ```bash
   # 가상환경 삭제 후 재생성
   rm -rf .venv
   ./local_auto_sync.sh --setup
   ```

4. **Git 동기화 문제**
   ```bash
   # 강제 리셋
   git reset --hard origin/master
   git clean -fd
   ```

### 로그 확인

```bash
# KAMP 서버 로그
tail -f logs/kamp_auto_run_*.log

# 학습 로그
tail -f logs/training_monitor_*.csv

# TensorBoard 실행
tensorboard --logdir logs/tensorboard
```

## 🔄 개발 워크플로우 예시

### 일반적인 개발 사이클

1. **로컬에서 코드 수정**
   ```bash
   # Cursor AI Editor에서 코드 편집
   # 예: src/train_maskable_ppo.py 수정
   ```

2. **로컬 동기화**
   ```bash
   ./local_auto_sync.sh
   # → 변경사항 커밋, GitHub에 푸시
   ```

3. **KAMP 서버 실행**
   ```bash
   # SSH로 KAMP 서버 접속
   ssh user@kamp-server
   cd RL-3DbinPacking
   ./kamp_auto_run.sh
   # → 코드 가져오기, 학습 실행, 결과 푸시
   ```

4. **결과 확인**
   ```bash
   # 로컬에서 결과 가져오기
   git pull origin master
   
   # 결과 파일 확인
   ls -la results/ models/ gifs/
   ```

5. **반복**
   - 결과 분석 후 코드 수정
   - 다시 1단계부터 반복

## 📚 참고 자료

- [Original 3D Bin Packing Repository](https://github.com/luisgarciar/3D-bin-packing)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Maskable PPO Documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
- [uv Documentation](https://docs.astral.sh/uv/)

## 🤝 기여

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Test your changes (`python test_env.py`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

💡 **팁**: 문제가 발생하면 먼저 `python test_env.py`로 환경을 테스트하고, 로그 파일을 확인해보세요! 