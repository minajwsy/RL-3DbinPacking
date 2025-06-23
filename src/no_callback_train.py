#!/usr/bin/env python3
"""
999 스텝 문제 완전 해결: 콜백 없는 순수 학습 스크립트
평가 콜백을 완전히 제거하여 학습 중단 문제 방지
"""

import os
import sys
import time
import datetime
import warnings

# 경고 메시지 억제
warnings.filterwarnings("ignore")

# matplotlib 백엔드 설정 (서버 환경 대응)
import matplotlib
matplotlib.use('Agg')

# 경로 설정
sys.path.append('src')
os.environ['PYTHONPATH'] = os.getcwd() + ':' + os.getcwd() + '/src'

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# 로컬 모듈 import
from packing_kernel import *
from train_maskable_ppo import make_env, get_action_masks

class SimpleProgressCallback:
    """간단한 진행상황 출력 클래스 (콜백 없음)"""
    def __init__(self, total_timesteps, print_freq=1000):
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.start_time = time.time()
        
    def print_progress(self, current_step):
        if current_step % self.print_freq == 0 or current_step == self.total_timesteps:
            elapsed = time.time() - self.start_time
            progress = current_step / self.total_timesteps * 100
            eta = elapsed / current_step * (self.total_timesteps - current_step) if current_step > 0 else 0
            
            print(f"진행: {current_step:,}/{self.total_timesteps:,} ({progress:.1f}%) | "
                  f"경과: {elapsed:.1f}s | 예상 남은 시간: {eta:.1f}s")

def no_callback_train(timesteps=3000, container_size=[10, 10, 10], num_boxes=16):
    """콜백 없는 순수 학습 함수"""
    
    print(f"=== 콜백 없는 순수 학습 시작 ===")
    print(f"목표 스텝: {timesteps:,}")
    print(f"컨테이너 크기: {container_size}")
    print(f"박스 개수: {num_boxes}")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 환경 생성 (간단한 설정)
    env = make_env(
        container_size=container_size,
        num_boxes=num_boxes,
        num_visible_boxes=3,
        seed=42,
        render_mode=None,
        random_boxes=False,
        only_terminal_reward=False,
        improved_reward_shaping=True,
    )()
    
    # 모니터링 설정 (로그만 기록)
    env = Monitor(env, f"logs/no_callback_train_{timestamp}.csv")
    
    print("환경 설정 완료")
    print(f"액션 스페이스: {env.action_space}")
    print(f"관찰 스페이스: {env.observation_space}")
    
    # 간단한 하이퍼파라미터 (빠른 학습용)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,  # 더 작은 스텝으로 빠른 업데이트
        batch_size=32,  # 작은 배치 크기
        n_epochs=3,  # 적은 에포크
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,  # 최소 출력
        tensorboard_log=None  # 텐서보드 비활성화
    )
    
    print("\n=== 순수 학습 시작 (콜백 없음) ===")
    start_time = time.time()
    
    # 진행상황 추적용
    progress = SimpleProgressCallback(timesteps, print_freq=500)
    
    try:
        # 콜백 없이 순수 학습만 실행
        print("⚠️  콜백 없이 학습 시작 - 평가 없음, 중단 없음")
        
        model.learn(
            total_timesteps=timesteps,
            callback=None,  # 콜백 완전 제거
            progress_bar=False,  # 진행바 비활성화
            tb_log_name=None,  # 텐서보드 로그 비활성화
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ 순수 학습 완료! 소요 시간: {training_time:.2f}초")
        
        # 모델 저장
        model_path = f"models/no_callback_ppo_{timestamp}"
        model.save(model_path)
        print(f"모델 저장 완료: {model_path}")
        
        # 학습 후 간단한 테스트 (별도 환경 사용)
        print("\n=== 학습 후 간단한 테스트 ===")
        test_env = make_env(
            container_size=container_size,
            num_boxes=num_boxes,
            num_visible_boxes=3,
            seed=999,
            render_mode=None,
            random_boxes=False,
            only_terminal_reward=False,
            improved_reward_shaping=True,
        )()
        
        # 3번의 짧은 테스트
        total_reward = 0
        test_count = 3
        
        for test_idx in range(test_count):
            obs, _ = test_env.reset()
            episode_reward = 0
            step_count = 0
            max_test_steps = 30  # 매우 짧은 테스트
            
            while step_count < max_test_steps:
                try:
                    action_masks = get_action_masks(test_env)
                    action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    episode_reward += reward
                    step_count += 1
                    
                    if terminated or truncated:
                        break
                except Exception as e:
                    print(f"테스트 {test_idx} 스텝 {step_count} 오류: {e}")
                    break
            
            total_reward += episode_reward
            print(f"테스트 {test_idx + 1}: {step_count}스텝, 보상 = {episode_reward:.4f}")
        
        avg_reward = total_reward / test_count
        print(f"평균 테스트 보상: {avg_reward:.4f}")
        
        # 환경 정리
        env.close()
        test_env.close()
        
        return model_path, avg_reward
        
    except KeyboardInterrupt:
        print("\n학습이 중단되었습니다.")
        model.save(f"models/interrupted_no_callback_{timestamp}")
        return None, None
    
    except Exception as e:
        print(f"\n학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        model.save(f"models/error_no_callback_{timestamp}")
        return None, None

def progressive_train(max_timesteps=10000):
    """점진적 학습: 작은 단위로 나누어 학습"""
    
    print(f"=== 점진적 학습 시작 ===")
    print(f"최대 스텝: {max_timesteps:,}")
    
    # 단계별 학습
    stages = [
        (1000, [8, 8, 8], 8),    # 1단계: 작은 문제
        (2000, [10, 10, 10], 12), # 2단계: 중간 문제
        (max_timesteps, [10, 10, 10], 16), # 3단계: 목표 문제
    ]
    
    best_model_path = None
    best_reward = -float('inf')
    
    for stage_idx, (timesteps, container_size, num_boxes) in enumerate(stages):
        print(f"\n🎯 단계 {stage_idx + 1}: {timesteps} 스텝, 컨테이너 {container_size}, 박스 {num_boxes}개")
        
        try:
            model_path, reward = no_callback_train(
                timesteps=timesteps,
                container_size=container_size,
                num_boxes=num_boxes
            )
            
            if model_path and reward > best_reward:
                best_model_path = model_path
                best_reward = reward
                print(f"✅ 단계 {stage_idx + 1} 완료: 보상 {reward:.4f}")
            else:
                print(f"❌ 단계 {stage_idx + 1} 실패")
                
        except Exception as e:
            print(f"❌ 단계 {stage_idx + 1} 오류: {e}")
            continue
    
    return best_model_path, best_reward

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="콜백 없는 순수 학습 스크립트")
    parser.add_argument("--timesteps", type=int, default=3000, help="총 학습 스텝")
    parser.add_argument("--container-size", nargs=3, type=int, default=[10, 10, 10], help="컨테이너 크기")
    parser.add_argument("--num-boxes", type=int, default=16, help="박스 개수")
    parser.add_argument("--progressive", action="store_true", help="점진적 학습 모드")
    
    args = parser.parse_args()
    
    # 디렉토리 생성
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("🚀 콜백 없는 순수 학습 스크립트 시작")
    print("📝 특징: 평가 없음, 콜백 없음, 999 스텝 문제 없음")
    
    try:
        if args.progressive:
            print("📈 점진적 학습 모드")
            model_path, reward = progressive_train(args.timesteps)
        else:
            print("⚡ 단일 학습 모드")
            model_path, reward = no_callback_train(
                timesteps=args.timesteps,
                container_size=args.container_size,
                num_boxes=args.num_boxes
            )
        
        if model_path:
            print(f"\n🎉 성공적으로 완료!")
            print(f"모델: {model_path}")
            print(f"성능: {reward:.4f}")
            print(f"✅ 999 스텝 문제 없이 완료됨")
        else:
            print("❌ 학습이 완료되지 않았습니다.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        sys.exit(1) 