#!/usr/bin/env python3
"""
999 스텝 문제 해결을 위한 간단한 학습 스크립트
평가 콜백을 최소화하여 학습 중단 문제 방지
"""

import os
import sys
import time
import datetime
import warnings

# 경고 메시지 억제
warnings.filterwarnings("ignore")

# 경로 설정
sys.path.append('src')
os.environ['PYTHONPATH'] = os.getcwd() + ':' + os.getcwd() + '/src'

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# 로컬 모듈 import
from packing_kernel import *
from train_maskable_ppo import make_env, get_action_masks

def quick_train(timesteps=3000, eval_freq=1500):
    """999 스텝 문제 해결을 위한 빠른 학습"""
    
    print(f"=== 빠른 학습 시작 ===")
    print(f"목표 스텝: {timesteps:,}")
    print(f"평가 주기: {eval_freq:,}")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 환경 생성 (간단한 설정)
    env = make_env(
        container_size=[10, 10, 10],
        num_boxes=16,  # 적은 박스 수로 시작
        num_visible_boxes=3,
        seed=42,
        render_mode=None,
        random_boxes=False,
        only_terminal_reward=False,
        improved_reward_shaping=True,
    )()
    
    # 평가용 환경
    eval_env = make_env(
        container_size=[10, 10, 10],
        num_boxes=16,
        num_visible_boxes=3,
        seed=43,
        render_mode=None,
        random_boxes=False,
        only_terminal_reward=False,
        improved_reward_shaping=True,
    )()
    
    # 모니터링 설정
    env = Monitor(env, f"logs/quick_train_{timestamp}.csv")
    eval_env = Monitor(eval_env, f"logs/quick_eval_{timestamp}.csv")
    
    print("환경 설정 완료")
    print(f"액션 스페이스: {env.action_space}")
    print(f"관찰 스페이스: {env.observation_space}")
    
    # 간단한 콜백만 사용
    callbacks = []
    
    # 최소한의 평가 콜백만 사용
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best_quick",
        log_path="logs/quick_eval",
        eval_freq=eval_freq,
        n_eval_episodes=3,  # 적은 평가 에피소드
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # 체크포인트 콜백
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path="models/quick_checkpoints",
        name_prefix=f"quick_model_{timestamp}",
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # 간단한 하이퍼파라미터
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="logs/quick_tensorboard"
    )
    
    print("\n=== 학습 시작 ===")
    start_time = time.time()
    
    try:
        # 학습 실행
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name=f"quick_maskable_ppo_{timestamp}",
        )
        
        training_time = time.time() - start_time
        print(f"\n학습 완료! 소요 시간: {training_time:.2f}초")
        
        # 모델 저장
        model_path = f"models/quick_ppo_mask_{timestamp}"
        model.save(model_path)
        print(f"모델 저장 완료: {model_path}")
        
        # 간단한 평가
        print("\n=== 최종 평가 ===")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=5, deterministic=True
        )
        
        print(f"평가 완료:")
        print(f"  평균 보상: {mean_reward:.4f} ± {std_reward:.4f}")
        
        # 환경 정리
        env.close()
        eval_env.close()
        
        return model_path, mean_reward
        
    except KeyboardInterrupt:
        print("\n학습이 중단되었습니다.")
        model.save(f"models/interrupted_quick_{timestamp}")
        return None, None
    
    except Exception as e:
        print(f"\n학습 중 오류 발생: {e}")
        model.save(f"models/error_quick_{timestamp}")
        raise e

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="빠른 학습 스크립트")
    parser.add_argument("--timesteps", type=int, default=3000, help="총 학습 스텝")
    parser.add_argument("--eval-freq", type=int, default=1500, help="평가 주기")
    
    args = parser.parse_args()
    
    # 디렉토리 생성
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("🚀 빠른 학습 스크립트 시작")
    
    try:
        model_path, reward = quick_train(
            timesteps=args.timesteps,
            eval_freq=args.eval_freq
        )
        
        if model_path:
            print(f"\n🎉 성공적으로 완료!")
            print(f"모델: {model_path}")
            print(f"성능: {reward:.4f}")
        else:
            print("❌ 학습이 완료되지 않았습니다.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        sys.exit(1) 