# 3D Bin Packing Optimization

Repository for the Capstone Project 3D Packing Optimization of the [Fourthbrain Machine Learning Engineer program](https://www.fourthbrain.ai/machine-learning-engineer).

This repository contains an environment compatible with [OpenAI Gym's API](https://github.com/openai/gym) to solve the 
3D bin packing problem with reinforcement learning (RL).


![Alt text](gifs/random_rollout2.gif?raw=true "A random packing agent in the environment")

## Problem definition and assumptions:
The environment consists of a list of 3D boxes of varying sizes and a single container of fixed size. The goal is to pack
as many boxes as possible in the container minimizing the empty volume. We assume that rotation of the boxes is 
not possible.

##  Problem instances: 
The function `boxes_generator` in the file `utils.py` generates instances of the 3D Bin Packing problem using the 
algorithm described in [Ranked Reward: Enabling Self-Play Reinforcement Learning for Combinatorial Optimization](https://arxiv.org/pdf/1807.01672.pdf)
(Algorithm 2, Appendix).

## Documentation
The documentation for this project is located in the `doc` folder, with a complete description of the state and 
action space as well as the rewards to be used for RL training.

## Installation instructions
We recommend that you create a virtual environment with Python 3.8 (for example, using conda environments). 
In your terminal window, activate your environment and clone the repository:
``` 
git clone https://github.com/luisgarciar/3D-bin-packing.git
```

To run the code, you need to install a few dependencies. Go to the cloned directory and install the required packages:
```
cd 3D-bin-packing
pip install -r requirements.txt
```

## Packing engine
The module `packing_engine` (located in `src/packing_engine.py`) implements the `Container` and `Box` objects that are 
used in the Gym environment. To add custom features (for example, to allow rotations), see the documentation of this module.

## Environment
The Gym environment is implemented in the module `src/packing_env.py`.

## Demo notebooks
A demo notebook `demo_ffd` implementing the heuristic-based method 'First Fit Decreasing' is available in the `nb` 
folder.

## Unit tests
The folder `tests` contains unit tests to be run with pytest.

## Update: 22/08/2022
The following updates have been made to the repository:
- Added the `packing_env.py` file with the Gym environment.
- Added unit tests for the Gym environment.
- Updated the documentation with the full description of the state and action space.
- Updated the demo notebooks.

## Update: 13/09/2022
The following updates have been made to the repository:
- Added functionality for saving rollouts of a policy in a .gif file and
- Added a demo notebook for the random policy.
- Updated the requirements.txt file with the required packages.
- Added a demo script for training agents with Maskable PPO. 

## Update: 7/1/2023 
The following updates have been made to the repository:
- Updated the demo notebook for training agents with Maskable PPO in Google colab.
- Fixed issues with the tests.

## 🚀 Update: 종합 실시간 모니터링 기능 대폭 강화! (2024년 12월)
**활용률과 안정성 지표까지 포함한** 새로운 **종합 실시간 모니터링 및 성능 분석** 기능이 추가되었습니다!

### ✨ 주요 신기능

#### 🎯 **컨테이너 활용률 실시간 추적**
- **에피소드별 활용률**: 각 에피소드에서 달성한 컨테이너 공간 활용률 실시간 모니터링
- **평가 활용률**: 주기적 평가에서의 활용률 성능 추적
- **최대 활용률 기록**: 지금까지 달성한 최고 활용률 지속 추적
- **목표선 표시**: 80% 목표 활용률 기준선으로 성과 측정

#### ⚖️ **학습 안정성 지표 분석**
- **보상 안정성**: 최근 에피소드들의 보상 표준편차로 학습 안정성 측정
- **활용률 안정성**: 활용률의 변동성을 통한 성능 일관성 평가
- **학습 smoothness**: 학습 곡선의 부드러움 정도 (0~1 범위)

#### 📊 **6개 차트 종합 대시보드**
- **상단**: 에피소드별 보상 + 컨테이너 활용률 (이동평균 및 목표선 포함)
- **중단**: 평가 성능 (보상 & 활용률) + 성공률 (목표선 포함)
- **하단**: 학습 안정성 지표 + 에피소드 길이 & 최대 활용률

#### ⚡ **더 빈번한 실시간 업데이트**
- **빠른 업데이트**: 1000 스텝마다 주요 차트 업데이트
- **전체 업데이트**: 기존 주기로 모든 차트 종합 업데이트
- **실시간 콘솔 출력**: 활용률과 안정성 정보 포함

### 🛠️ 간단한 사용법
```bash
# 종합 실시간 모니터링과 함께 학습 시작
python -m src.train_maskable_ppo --timesteps 100000

# 더 자주 업데이트하는 실시간 모니터링
python -m src.train_maskable_ppo --timesteps 50000 --eval-freq 5000

# 종합 분석 (활용률 & 안정성 포함)
python -m src.train_maskable_ppo --analyze-only results/comprehensive_training_stats_YYYYMMDD_HHMMSS.npy

# KAMP 서버 자동 실행 (종합 모니터링 포함)
./kamp_auto_run.sh master
```

### 📈 실시간 출력 예시
```
스텝: 25,000 | 에피소드: 156 | 최근 10 에피소드 평균 보상: 0.642 | 평균 활용률: 64.2% | 최대 활용률: 78.5% | 경과 시간: 180.5초

평가 수행 중... (스텝: 25,000)
평가 완료: 평균 보상 0.685, 평균 활용률 68.5%, 성공률 80.0%
```

### 📁 자동 생성 파일
- `comprehensive_training_progress_*.png`: 6개 차트 종합 대시보드
- `comprehensive_training_stats_*.npy`: 활용률+안정성 포함 종합 통계
- `comprehensive_summary_*.txt`: 한글 성과 요약 보고서 (등급 평가 포함)

### 🏆 자동 등급 평가 시스템
- **활용률**: 🥇 우수(80%+) / 🥈 양호(60-80%) / 🥉 개선필요(60%-)
- **성공률**: 🥇 우수(80%+) / 🥈 양호(50-80%) / 🥉 개선필요(50%-)
- **학습 안정성**: 🥇 매우안정적(0.7+) / 🥈 안정적(0.5-0.7) / 🥉 불안정(0.5-)