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

## 🚀 Update: 실시간 모니터링 기능 추가 (2024년 12월)
새로운 **실시간 모니터링 및 성능 분석** 기능이 추가되었습니다!

### ✨ 주요 기능
- **실시간 학습 진행 상황 모니터링**: 에피소드별 보상, 성공률, 평가 성능을 실시간으로 추적
- **자동 시각화**: 4개 차트로 구성된 종합 대시보드 자동 생성 및 업데이트
- **지능형 성과 분석**: 학습 안정성, 성공률 등급, 개선 제안 자동 생성
- **서버 환경 최적화**: GUI 없는 KAMP 서버에서도 완벽 작동

### 🛠️ 간단한 사용법
```bash
# 실시간 모니터링과 함께 학습 시작
python -m src.train_maskable_ppo --timesteps 100000

# 기존 결과 분석
python -m src.train_maskable_ppo --analyze-only results/training_stats_YYYYMMDD_HHMMSS.npy

# KAMP 서버 자동 실행 (모니터링 포함)
./kamp_auto_run.sh master
```

📖 **자세한 사용법**: [`REALTIME_MONITORING_GUIDE.md`](REALTIME_MONITORING_GUIDE.md) 를 참조하세요.
