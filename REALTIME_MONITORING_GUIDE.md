# 실시간 모니터링 기능 사용 가이드

## 🚀 개요

Maskable PPO 3D Bin Packing 학습 과정에서 **실시간으로 성능 지표를 모니터링**할 수 있는 기능이 추가되었습니다. 이 기능을 통해 학습 진행 상황을 실시간으로 확인하고, 성능 개선 양상을 시각화할 수 있습니다.

## 📊 주요 기능

### 1. 실시간 성능 지표 추적
- **에피소드별 보상 추이** (원시 데이터 + 이동평균)
- **평가 성능 변화** (주기적 모델 평가)
- **성공률 모니터링** (빈 패킹 성공률)
- **에피소드 길이 분석** (학습 효율성 측정)

### 2. 실시간 시각화
- 4개 서브플롯으로 구성된 종합 대시보드
- 학습 진행에 따른 자동 플롯 업데이트
- 고해상도 차트 이미지 자동 저장

### 3. 자동 성과 분석
- 학습 안정성 분석 (보상 추세, 변동성)
- 성공률 평가 및 분류
- 개선 제안 자동 생성

## 🛠️ 사용 방법

### 기본 학습 (실시간 모니터링 포함)

```bash
# 기본 실행 (100,000 스텝)
python -m src.train_maskable_ppo --timesteps 100000

# 더 짧은 평가 주기로 더 자주 모니터링
python -m src.train_maskable_ppo --timesteps 50000 --eval-freq 5000

# CPU 강제 사용 + 실시간 모니터링
python -m src.train_maskable_ppo --timesteps 30000 --force-cpu
```

### 학습 통계 분석 (학습 완료 후)

```bash
# 특정 학습 결과 분석
python -m src.train_maskable_ppo --analyze-only results/training_stats_20241201_143022.npy

# 대시보드만 생성
python -m src.train_maskable_ppo --dashboard-only results/training_stats_20241201_143022.npy
```

### KAMP 서버에서 자동 실행

```bash
# 자동화 스크립트 실행 (실시간 모니터링 포함)
./kamp_auto_run.sh master
```

## 📈 모니터링 대시보드 설명

### 좌상단: 에피소드별 보상
- **파란색 선**: 원시 에피소드 보상
- **빨간색 선**: 이동평균 (50 에피소드)
- **해석**: 상승 추세면 학습이 잘 진행됨

### 우상단: 평가 성능
- **초록색 선**: 주기적 모델 평가 결과
- **검은색 점선**: 0점 기준선
- **해석**: 0 이상이면 양호, 지속적 상승이 이상적

### 좌하단: 성공률
- **주황색 선**: 빈 패킹 성공률 (%)
- **목표**: 80% 이상이 우수
- **해석**: 급격한 변동 없이 상승해야 함

### 우하단: 에피소드 길이
- **보라색 선**: 원시 에피소드 길이
- **빨간색 선**: 이동평균 (20 에피소드)
- **해석**: 너무 짧으면 조기 종료, 너무 길면 비효율

## 🔧 고급 설정

### 모니터링 주기 조정

```python
# train_maskable_ppo.py 수정
monitor_callback = RealTimeMonitorCallback(
    eval_env=eval_env,
    eval_freq=2000,  # 2000 스텝마다 평가 (기본값의 절반)
    n_eval_episodes=10,  # 평가 에피소드 수 증가
    verbose=1
)
```

### TensorBoard 동시 사용

```bash
# 별도 터미널에서 TensorBoard 실행
tensorboard --logdir=logs/tensorboard --port=6006

# 브라우저에서 http://localhost:6006 접속
```

## 🎯 성과 분석 지표

### 자동 생성되는 분석 항목

1. **기본 통계**
   - 총 에피소드 수
   - 총 학습 스텝 수
   - 최종/최고 평가 보상
   - 최종/최고 성공률

2. **학습 안정성**
   - 평가 보상 추세 (기울기)
   - 보상 변동성 (변동계수)

3. **성공률 등급**
   - ✅ 우수 (80% 이상)
   - ⚠️ 보통 (50-80%)
   - ❌ 개선 필요 (50% 미만)

4. **개선 제안**
   - 보상 함수 조정 필요성
   - 학습 파라미터 조정 제안
   - 과적합 가능성 경고

## 📁 출력 파일

### 자동 생성되는 파일들

```
results/
├── training_progress_YYYYMMDD_HHMMSS.png    # 중간 진행 상황
├── final_training_progress_YYYYMMDD_HHMMSS.png  # 최종 진행 상황
├── final_dashboard_YYYYMMDD_HHMMSS.png      # 최종 대시보드
├── training_stats_YYYYMMDD_HHMMSS.npy       # 학습 통계 (분석용)
└── training_results_YYYYMMDD_HHMMSS.txt     # 결과 요약
```

## 🚨 문제 해결

### GUI 환경이 없는 서버에서 오류 발생 시

```python
# 이미 설정되어 있지만, 추가 확인용
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경용 백엔드
```

### 메모리 부족 시

```bash
# 평가 주기를 늘려서 메모리 사용량 감소
python -m src.train_maskable_ppo --timesteps 50000 --eval-freq 20000
```

### 플롯 업데이트가 너무 느린 경우

```python
# RealTimeMonitorCallback에서 eval_freq 증가
eval_freq=10000  # 더 적은 빈도로 업데이트
```

## 📚 실제 사용 예시

### 1. 빠른 테스트 (5분 이내)

```bash
python -m src.train_maskable_ppo --timesteps 5000 --eval-freq 1000
```

### 2. 본격 학습 (1-2시간)

```bash
python -m src.train_maskable_ppo --timesteps 100000 --eval-freq 5000
```

### 3. 장기 학습 (하룻밤)

```bash
nohup python -m src.train_maskable_ppo --timesteps 500000 --eval-freq 10000 > training.log 2>&1 &
```

### 4. 기존 결과 분석

```bash
# 최근 결과 찾기
ls -t results/training_stats_*.npy | head -1

# 분석 실행
python -m src.train_maskable_ppo --analyze-only results/training_stats_20241201_143022.npy
```

## 💡 팁과 권장사항

1. **모니터링 주기**: 너무 짧으면 성능 저하, 너무 길면 트렌드 놓칠 수 있음
2. **성공률 목표**: 80% 이상을 목표로 설정
3. **조기 중단**: 성능이 계속 악화되면 하이퍼파라미터 조정 고려
4. **결과 보관**: `.npy` 파일을 보관하여 나중에 재분석 가능
5. **비교 분석**: 여러 실험 결과를 비교하여 최적 설정 찾기

---

📧 **문의 사항이나 개선 제안이 있으시면 이슈로 등록해 주세요!** 