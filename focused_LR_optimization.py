#!/usr/bin/env python3
"""
🎯 집중 학습률 최적화 (동일 난이도)
이전 최고 성능 기반 정밀 튜닝
"""

import os
import sys
import warnings
import datetime
import time
import json
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings("ignore")
sys.path.append('src')

print("🎯 집중 학습률 최적화")

def create_environment(container_size, num_boxes, seed):
    """환경 생성 (기존과 동일)"""
    try:
        import gymnasium as gym
        from gymnasium.envs.registration import register
        from packing_env import PackingEnv
        from utils import boxes_generator
        
        if 'PackingEnv-v0' not in gym.envs.registry:
            register(id='PackingEnv-v0', entry_point='packing_env:PackingEnv')
        
        box_sizes = boxes_generator(container_size, num_boxes, seed)
        env = gym.make(
            "PackingEnv-v0",
            container_size=container_size,
            box_sizes=box_sizes,
            num_visible_boxes=min(3, num_boxes),
            render_mode=None,
            random_boxes=False,
            only_terminal_reward=False,
        )
        
        try:
            from train_maskable_ppo import ImprovedRewardWrapper
            env = ImprovedRewardWrapper(env)
        except:
            pass
        
        from sb3_contrib.common.wrappers import ActionMasker
        
        def get_masks(env):
            try:
                from train_maskable_ppo import get_action_masks
                return get_action_masks(env)
            except:
                import numpy as np
                return np.ones(env.action_space.n, dtype=bool)
        
        env = ActionMasker(env, get_masks)
        return env
        
    except Exception as e:
        print(f"❌ 환경 생성 오류: {e}")
        return None

def focused_objective(trial):
    """집중된 목적 함수 - 좁은 범위 탐색"""
    try:
        # === 🎯 집중 탐색 범위 ===
        params = {
            # 학습률: 2e-4 ~ 2.8e-4 정밀 탐색 (이전 최고 근처)
            'learning_rate': trial.suggest_float('learning_rate', 2.0e-4, 2.8e-4, log=True),
            
            # n_steps: 이전 최고 근처 집중
            'n_steps': trial.suggest_categorical('n_steps', [512, 768]),  # 384 제외
            
            # batch_size: 64 중심 탐색
            'batch_size': trial.suggest_categorical('batch_size', [64, 96]),  # 48 제외
            
            # n_epochs: 효율적 범위
            'n_epochs': trial.suggest_int('n_epochs', 3, 4),  # 5는 너무 많음
            
            # 🔧 이전 최적값 기반 미세 조정
            'clip_range': trial.suggest_float('clip_range', 0.17, 0.22),  # 이전 최고 근처
            'ent_coef': trial.suggest_float('ent_coef', 0.05, 0.08),      # 이전 최고 근처
            'vf_coef': trial.suggest_float('vf_coef', 0.40, 0.55),        # 이전 최고 근처
            'gae_lambda': trial.suggest_float('gae_lambda', 0.94, 0.96)   # 이전 최고 근처
        }
        
        # === 🏗️ 동일한 문제 설정 (성능 저하 방지) ===
        container_size = [8, 8, 8]   # 이전과 동일
        num_boxes = 6                # 이전과 동일
        train_steps = 10000          # 적당한 학습 시간
        
        print(f"\n🔍 Trial {trial.number}: LR={params['learning_rate']:.2e}, "
              f"Steps={params['n_steps']}, Batch={params['batch_size']}")
        
        # === 환경 및 모델 생성 ===
        env = create_environment(container_size, num_boxes, 42)
        if env is None:
            return 0.0
        
        import torch
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(2)
        
        from sb3_contrib import MaskablePPO
        import torch.nn as nn
        
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            gamma=0.99,
            gae_lambda=params['gae_lambda'],
            clip_range=params['clip_range'],
            ent_coef=params['ent_coef'],
            vf_coef=params['vf_coef'],
            max_grad_norm=0.5,
            verbose=0,
            device='cpu',
            policy_kwargs=dict(
                net_arch=[128, 128],
                activation_fn=nn.ReLU,
            )
        )
        
        # === 🚀 안정적 학습 (조기 종료 완화) ===
        intermediate_steps = [3000, 6000, 10000]
        
        for step in intermediate_steps:
            remaining = step - (0 if step == 3000 else intermediate_steps[intermediate_steps.index(step)-1])
            model.learn(total_timesteps=remaining, progress_bar=False)
            
            # 중간 평가 (관대한 기준)
            score = evaluate_model(model, container_size, num_boxes, n_episodes=8)
            trial.report(score, step)
            
            # 🔧 조기 종료 완화 (더 기다림)
            if step > 6000 and trial.should_prune():  # 6000 스텝 이후에만 prune
                env.close()
                del model
                raise optuna.TrialPruned()
            
            print(f"   Step {step}: {score:.2f}")
        
        # === 최종 평가 ===
        final_score = evaluate_model(model, container_size, num_boxes, n_episodes=12)
        
        env.close()
        del model
        
        print(f"   🏆 최종: {final_score:.2f}")
        return final_score
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"❌ Trial 실패: {e}")
        return 0.0

def evaluate_model(model, container_size, num_boxes, n_episodes=12):
    """모델 평가 (기존과 동일한 방식)"""
    try:
        import numpy as np
        all_rewards, all_utils, placements = [], [], []
        
        for ep in range(n_episodes):
            env = create_environment(container_size, num_boxes, 100 + ep * 13)
            if env is None: continue
            
            obs, _ = env.reset(seed=100 + ep * 13)
            reward = 0.0
            
            for _ in range(25):  # 이전과 동일한 스텝 수
                try:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, r, done, trunc, _ = env.step(action)
                    reward += r
                    if done or trunc: break
                except: break
            
            # 활용률 및 배치 개수 계산
            util, placed = 0.0, 0
            try:
                if hasattr(env.unwrapped, 'container'):
                    placed_vol = sum(box.volume for box in env.unwrapped.container.boxes 
                                   if box.position is not None)
                    total_vol = env.unwrapped.container.volume
                    util = placed_vol / total_vol if total_vol > 0 else 0.0
                    placed = sum(1 for box in env.unwrapped.container.boxes 
                               if box.position is not None)
            except: pass
            
            all_rewards.append(reward)
            all_utils.append(util)
            placements.append(placed)
            env.close()
        
        if not all_rewards: return 0.0
        
        # 🎯 이전과 동일한 점수 계산 방식
        mean_reward = np.mean(all_rewards)
        mean_util = np.mean(all_utils)
        success_rate = sum(1 for u in all_utils if u >= 0.3) / len(all_utils)
        
        # 종합 점수: 보상 30% + 활용률 70% (이전과 동일)
        combined_score = mean_reward * 0.3 + mean_util * 100 * 0.7
        
        return combined_score
        
    except: return 0.0

def run_focused_optimization(n_trials=20):
    """집중 최적화 실행"""
    try:
        print(f"🚀 집중 학습률 최적화 시작 ({n_trials} trials)")
        print(f"📊 목표: 18.57 점수 초과")
        
        # 🔧 조기 종료 완화된 설정
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42, n_startup_trials=5),
            pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=2, interval_steps=1)
        )
        
        study.optimize(focused_objective, n_trials=n_trials, timeout=3600)
        
        print(f"\n📈 === 집중 최적화 결과 ===")
        print(f"✅ 완료 trials: {len(study.trials)}")
        print(f"🏆 최고 점수: {study.best_value:.2f}")
        
        print(f"\n🎯 === 최적 파라미터 ===")
        for key, value in study.best_params.items():
            if key == 'learning_rate':
                print(f"   {key}: {value:.2e}")
            else:
                print(f"   {key}: {value}")
        
        # 성능 비교
        baseline = 18.57
        improvement = ((study.best_value - baseline) / baseline) * 100
        print(f"\n📊 === 성능 비교 ===")
        print(f"   이전 최고: {baseline:.2f}")
        print(f"   신규 최고: {study.best_value:.2f}")
        print(f"   개선율: {improvement:+.1f}%")
        
        # 결과 저장
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/focused_optimization_{timestamp}.json"
        
        os.makedirs('results', exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'optimization_type': 'focused_learning_rate',
                'problem_config': {
                    'container_size': [8, 8, 8],
                    'num_boxes': 6,
                    'train_steps': 10000
                },
                'best_params': study.best_params,
                'best_score': study.best_value,
                'baseline_score': baseline,
                'improvement_percent': improvement,
                'n_trials': len(study.trials)
            }, f, indent=2)
        
        print(f"💾 결과 저장: {results_file}")
        
        if study.best_value > baseline:
            print(f"🎉 성능 개선 성공!")
            print(f"💡 이 파라미터를 ultimate_train_fix.py에 적용하세요")
        else:
            print(f"📋 추가 최적화가 필요합니다")
        
        return study.best_params, study.best_value
        
    except Exception as e:
        print(f"❌ 집중 최적화 실패: {e}")
        return None, 0

if __name__ == "__main__":
    print("🎯 집중 학습률 최적화")
    print(f"🚀 Python: {sys.version}")
    print(f"📁 작업 디렉토리: {os.getcwd()}")
    
    best_params, best_score = run_focused_optimization(n_trials=20)
    
    if best_params and best_score > 18.57:
        print(f"\n🎉 집중 최적화 성공!")
        print(f"🏆 새로운 최적 파라미터 발견!")
    else:
        print(f"\n📊 추가 분석이 필요합니다")