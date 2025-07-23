# 하이퍼파라미터 최적화 가이드

## 🚀 개요

`ultimate_train_fix.py`에 Optuna와 W&B Sweep을 이용한 자동 하이퍼파라미터 최적화 기능이 추가되었습니다.

## 📦 설치 요구사항

```bash
# 필수 라이브러리 설치
pip install optuna wandb plotly scikit-learn

# 또는 전체 requirements 설치
pip install -r requirements.txt
```

## 🎯 최적화 목표

- **평균 에피소드 보상 (Mean Episode Reward)**: 70% 가중치
- **공간 활용률 (Utilization Rate)**: 30% 가중치
- **종합 점수**: `보상 × 0.7 + 활용률 × 100 × 0.3`

## 🔧 최적화 대상 하이퍼파라미터

| 파라미터 | 탐색 범위 | 분포 |
|----------|-----------|------|
| `learning_rate` | 1e-6 ~ 1e-3 | log-uniform |
| `n_steps` | [1024, 2048, 4096] | categorical |
| `batch_size` | [64, 128, 256] | categorical |
| `n_epochs` | 3 ~ 15 | uniform integer |
| `clip_range` | 0.1 ~ 0.4 | uniform |
| `ent_coef` | 1e-4 ~ 1e-1 | log-uniform |
| `vf_coef` | 0.1 ~ 1.0 | uniform |
| `gae_lambda` | 0.9 ~ 0.99 | uniform |

## 🚀 사용법

### 1. Optuna 최적화

```bash
# 기본 Optuna 최적화 (30회 시행)
python src/ultimate_train_fix.py --optimize --optimization-method optuna --n-trials 30

# W&B 로깅과 함께
python src/ultimate_train_fix.py --optimize --optimization-method optuna --n-trials 50 --use-wandb --wandb-project my-project
```

### 2. W&B Sweep 최적화

```bash
# W&B Sweep 최적화 (베이지안 최적화)
python src/ultimate_train_fix.py --optimize --optimization-method wandb --use-wandb --n-trials 30

# 커스텀 프로젝트명
python src/ultimate_train_fix.py --optimize --optimization-method wandb --use-wandb --wandb-project my-custom-project --n-trials 50
```

### 3. 두 방법 모두 사용

```bash
# Optuna + W&B Sweep 동시 실행
python src/ultimate_train_fix.py --optimize --optimization-method both --use-wandb --n-trials 30
```

### 4. 최적 하이퍼파라미터로 최종 학습

```bash
# Optuna 결과로 최종 학습
python src/ultimate_train_fix.py --train-with-best results/optuna_results_20240722_143052.json --timesteps 50000
```

## 📊 결과 분석

### 생성되는 파일들

- `results/optuna_results_[timestamp].json`: Optuna 최적화 결과
- `results/optuna_visualization_[timestamp].png`: 최적화 과정 시각화
- `results/hyperparameter_optimization_summary_[timestamp].json`: 통합 결과 요약

### Optuna 시각화 내용

1. **Optimization History**: 최적화 진행 과정
2. **Parameter Importances**: 파라미터 중요도 분석
3. **Parallel Coordinate Plot**: 최적 파라미터 조합 패턴
4. **Hyperparameter Slice Plot**: 개별 파라미터 영향도

## 🎛️ 고급 옵션

### Trial 설정 커스터마이징

```bash
# 더 긴 trial 학습 (정확도 향상)
python src/ultimate_train_fix.py --optimize --trial-timesteps 15000 --n-trials 20

# 더 많은 컨테이너/박스 설정
python src/ultimate_train_fix.py --optimize --container-size 12 12 12 --num-boxes 20 --n-trials 30
```

### W&B 설정

```bash
# W&B 로그인 (최초 1회)
wandb login

# 특정 프로젝트로 최적화
python src/ultimate_train_fix.py --optimize --optimization-method wandb --use-wandb --wandb-project "3d-binpacking-optimization"
```

## 🔍 성능 분석 예시

### 최적화 결과 해석

```json
{
  "best_value": 8.45,
  "best_params": {
    "learning_rate": 0.0002341,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 8,
    "clip_range": 0.234,
    "ent_coef": 0.0156,
    "vf_coef": 0.67,
    "gae_lambda": 0.95
  }
}
```

- **best_value (8.45)**: 종합 점수 (보상 × 0.7 + 활용률 × 100 × 0.3)
- 만약 보상=6.5, 활용률=63%라면: 6.5 × 0.7 + 63 × 0.3 = 4.55 + 18.9 = 23.45

## 📈 모범 사례

### 1. 단계적 접근

```bash
# 1단계: 빠른 탐색 (20회, 짧은 학습)
python src/ultimate_train_fix.py --optimize --n-trials 20 --trial-timesteps 5000

# 2단계: 정밀 탐색 (50회, 중간 학습)
python src/ultimate_train_fix.py --optimize --n-trials 50 --trial-timesteps 10000

# 3단계: 최종 학습 (최적 파라미터로 긴 학습)
python src/ultimate_train_fix.py --train-with-best results/optuna_results_xxx.json --timesteps 100000
```

### 2. 리소스 효율적 사용

```bash
# Pruning으로 효율적 최적화 (성능 낮은 trial 조기 중단)
python src/ultimate_train_fix.py --optimize --n-trials 100 --trial-timesteps 8000
```

### 3. 실험 관리

```bash
# 의미있는 W&B 프로젝트명 사용
python src/ultimate_train_fix.py --optimize --optimization-method both --use-wandb --wandb-project "3d-binpacking-16boxes-optimization"
```

## 🛠️ 문제 해결

### 일반적인 오류

1. **ImportError: No module named 'optuna'**
   ```bash
   pip install optuna
   ```

2. **W&B 로그인 필요**
   ```bash
   wandb login
   ```

3. **메모리 부족**
   - `--trial-timesteps` 값을 낮춤 (예: 5000)
   - `--batch-size` 범위 축소

### 성능 이슈

- **최적화가 너무 오래 걸림**: `--n-trials` 감소, `--trial-timesteps` 감소
- **결과가 좋지 않음**: `--trial-timesteps` 증가, 더 많은 trials 실행
- **불안정한 결과**: 시드 고정, 더 긴 평가 (n_episodes 증가)

## 📚 추가 자료

- [Optuna 공식 문서](https://optuna.readthedocs.io/)
- [W&B Sweeps 가이드](https://docs.wandb.ai/guides/sweeps)
- [PPO 하이퍼파라미터 가이드](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) 