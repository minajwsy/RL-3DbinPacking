# 🚀 KAMP 서버 실행 가이드 (개선된 버전)

## 📋 사전 준비사항

### 1. 환경 확인
```bash
# Python 환경 확인
python --version  # Python 3.9+ 권장

# GPU 확인 (선택사항)
nvidia-smi

# 작업 디렉토리 확인
pwd
ls -la
```

### 2. 의존성 설치 확인
```bash
# 필수 패키지 설치 확인
pip list | grep -E "(torch|stable-baselines3|sb3-contrib|gymnasium)"

# 누락된 패키지가 있다면 설치
pip install torch torchvision torchaudio
pip install stable-baselines3[extra]
pip install sb3-contrib
pip install gymnasium
```

## 🎯 실행 방법

### 1. 기본 실행 (개선된 설정)
```bash
# 기본 설정으로 실행 (커리큘럼 학습 + 개선된 보상 활성화)
python -m src.train_maskable_ppo

# 또는 더 짧은 학습으로 테스트
python -m src.train_maskable_ppo --timesteps 10000
```

### 2. 빠른 테스트 실행
```bash
# 매우 짧은 테스트 (100 스텝)
python -m src.train_maskable_ppo --timesteps 100 --no-gif

# 중간 테스트 (7000 스텝)
python -m src.train_maskable_ppo --timesteps 7000 --no-gif
```

### 3. 적극적인 학습 모드
```bash
# 장시간 고성능 학습
python -m src.train_maskable_ppo --aggressive-training

# 커스텀 설정
python -m src.train_maskable_ppo \
    --timesteps 300000 \
    --eval-freq 20000 \
    --num-boxes 48 \
    --curriculum-learning \
    --improved-rewards
```

### 4. 자동 실행 스크립트 사용
```bash
# KAMP 서버용 자동 스크립트 실행
./kamp_auto_run.sh

# 특정 브랜치로 실행
./kamp_auto_run.sh main
```

## 🔧 문제 해결

### 환경 생성 오류
```bash
# 환경 등록 확인
python -c "import gymnasium as gym; print(gym.envs.registry.keys())"

# 패키지 재설치
pip uninstall rl-3d-bin-packing
pip install -e .
```

### GPU 관련 오류
```bash
# CPU 강제 사용
python -m src.train_maskable_ppo --force-cpu --timesteps 5000

# CUDA 버전 확인
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### 메모리 부족 오류
```bash
# 더 작은 배치 크기로 실행
python -m src.train_maskable_ppo --timesteps 5000 --no-curriculum --no-improved-rewards
```

## 📊 실행 결과 확인

### 1. 실시간 모니터링
- 학습 중 콘솔에서 실시간 진행 상황 확인
- 매 10 에피소드마다 성능 지표 출력
- 활용률, 보상, 성공률 등 종합 정보 제공

### 2. 결과 파일 확인
```bash
# 학습 결과 확인
ls -la results/
cat results/improved_training_results_*.txt

# 학습 통계 확인
ls -la results/comprehensive_training_stats_*.npy

# 대시보드 이미지 확인
ls -la results/improved_final_dashboard_*.png
```

### 3. 모델 파일 확인
```bash
# 저장된 모델 확인
ls -la models/
ls -la models/improved_ppo_mask_*
```

## 🎯 성능 지표 해석

### 활용률 (Utilization Rate)
- **80% 이상**: 🥇 우수
- **60-80%**: 🥈 양호  
- **60% 미만**: 🥉 개선 필요

### 성공률 (Success Rate)
- **80% 이상**: 🥇 우수
- **50-80%**: 🥈 양호
- **50% 미만**: 🥉 개선 필요

### 학습 안정성 (Stability)
- **0.7 이상**: 🥇 안정적
- **0.5-0.7**: 🥈 보통
- **0.5 미만**: 🥉 불안정

## 🚨 주의사항

### 1. 학습 시간
- 기본 설정: 약 2-4시간 (200,000 스텝)
- 적극적 모드: 약 8-12시간 (500,000+ 스텝)
- 테스트 모드: 약 5-10분 (7,000 스텝)

### 2. 디스크 공간
- 모델 파일: 약 50-100MB
- 로그 파일: 약 10-50MB
- 결과 이미지: 약 5-20MB

### 3. 메모리 사용량
- GPU 모드: 약 2-4GB VRAM
- CPU 모드: 약 1-2GB RAM

## 📈 성능 개선 팁

### 1. 하이퍼파라미터 조정
```bash
# 더 적극적인 학습률
python -m src.train_maskable_ppo --timesteps 50000

# 커리큘럼 학습 비활성화 (빠른 테스트용)
python -m src.train_maskable_ppo --no-curriculum --timesteps 20000
```

### 2. 환경 설정 조정
```bash
# 더 많은 박스로 시작
python -m src.train_maskable_ppo --num-boxes 64 --timesteps 100000

# 더 작은 컨테이너
python -m src.train_maskable_ppo --container-size 8 8 8 --timesteps 50000
```

### 3. 평가 주기 조정
```bash
# 더 자주 평가 (더 세밀한 모니터링)
python -m src.train_maskable_ppo --eval-freq 5000 --timesteps 50000

# 덜 자주 평가 (더 빠른 학습)
python -m src.train_maskable_ppo --eval-freq 25000 --timesteps 100000
```

## 🔄 다음 단계

1. **초기 테스트**: `--timesteps 7000`으로 빠른 테스트
2. **중간 검증**: `--timesteps 50000`으로 성능 확인
3. **전체 학습**: 기본 설정 또는 적극적 모드로 완전 학습
4. **결과 분석**: 생성된 대시보드와 요약 보고서 검토
5. **모델 평가**: 최종 모델로 추가 평가 수행

## 🔧 해결된 문제들

### 1. ActionMaskingWrapper 임포트 오류
- **문제**: `ActionMaskingWrapper`를 찾을 수 없음
- **해결**: `ActionMasker`로 변경하고 `sb3_contrib.common.wrappers`에서 임포트

### 2. 환경 생성 실패
- **문제**: 여러 환경 ID에서 생성 실패
- **해결**: `PackingEnv-v0` 환경 직접 등록 및 `boxes_generator` 사용

### 3. train_and_evaluate 함수 누락
- **문제**: 함수가 정의되지 않음
- **해결**: 완전한 함수 구현 추가

### 4. 환경 등록 문제
- **문제**: `PackingEnv-v0` 환경이 등록되지 않음
- **해결**: 코드 시작 시 자동 환경 등록 및 PYTHONPATH 설정

### 5. boxes_generator 파라미터 오류
- **문제**: `num_boxes`, `container_size` 파라미터 불일치
- **해결**: `num_items`, `bin_size` 파라미터로 수정

---

이 가이드를 따라 실행하면 개선된 학습 성능을 확인할 수 있습니다! 