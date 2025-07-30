#!/usr/bin/env python3
"""
🎯 정밀한 2차 하이퍼파라미터 최적화
강화 최적 파라미터 기반 세밀 조정
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

def create_environment(container_size, num_boxes, seed):
    """최적화된 환경 생성"""
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

def refined_objective(trial):
    """정밀한 목적 함수"""
    try:
        # === 정밀한 파라미터 범위 설정 ===
        params = {
            # 🎯 학습률: 최적 구간 세밀 탐색
            'learning_rate': trial.suggest_float('learning_rate', 1.5e-4, 3e-4, log=True),
            
            # 🎯 n_steps: 강화 최적 근처 탐색
            'n_steps': trial.suggest_categorical('n_steps', [384, 512, 768]),
            
            # 🎯 batch_size: 효과적인 범위 탐색
            'batch_size': trial.suggest_categorical('batch_size', [48, 64, 96]),
            
            # 🎯 n_epochs: 효율성 고려
            'n_epochs': trial.suggest_int('n_epochs', 3, 5),
            
            # 🎯 기존 최적값 근처 미세 조정
            'clip_range': trial.suggest_float('clip_range', 0.15, 0.25),
            'ent_coef': trial.suggest_float('ent_coef', 0.04, 0.10),
            'vf_coef': trial.suggest_float('vf_coef', 0.3, 0.7),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.92, 0.98)
        }
        
        # === 더 어려운 문제 설정 ===
        container_size = [10, 10, 10]  # 큰 컨테이너
        num_boxes = 8  # 더 많은 박스
        train_steps = 12000  # 긴 학습 시간
        
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
        
        # === 점진적 학습 및 조기 종료 ===
        for step in [4000, 8000, 12000]:
            remaining = step - (0 if step == 4000 else [4000, 8000][step//4000-2])
            model.learn(total_timesteps=remaining, progress_bar=False)
            
            # 중간 평가
            score = quick_evaluate(model, container_size, num_boxes)
            trial.report(score, step)
            
            if trial.should_prune():
                env.close()
                del model
                raise optuna.TrialPruned()
            
            print(f"   Step {step}: {score:.2f}")
        
        # === 최종 평가 ===
        final_score = comprehensive_evaluate(model, container_size, num_boxes)
        
        env.close()
        del model
        
        print(f"   🏆 최종: {final_score:.2f}")
        return final_score
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"❌ Trial 실패: {e}")
        return 0.0

def quick_evaluate(model, container_size, num_boxes, n_episodes=6):
    """빠른 중간 평가"""
    try:
        import numpy as np
        all_rewards, all_utils = [], []
        
        for ep in range(n_episodes):
            env = create_environment(container_size, num_boxes, 200 + ep * 7)
            if env is None: continue
            
            obs, _ = env.reset(seed=200 + ep * 7)
            reward = 0.0
            
            for _ in range(30):
                try:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, r, done, trunc, _ = env.step(action)
                    reward += r
                    if done or trunc: break
                except: break
            
            # 활용률 계산
            util = 0.0
            try:
                if hasattr(env.unwrapped, 'container'):
                    placed_vol = sum(box.volume for box in env.unwrapped.container.boxes 
                                   if box.position is not None)
                    total_vol = env.unwrapped.container.volume
                    util = placed_vol / total_vol if total_vol > 0 else 0.0
            except: pass
            
            all_rewards.append(reward)
            all_utils.append(util)
            env.close()
        
        if not all_rewards: return 0.0
        return np.mean(all_rewards) * 0.3 + np.mean(all_utils) * 100 * 0.7
        
    except: return 0.0

def comprehensive_evaluate(model, container_size, num_boxes, n_episodes=10):
    """종합 최종 평가"""
    try:
        import numpy as np
        all_rewards, all_utils, placements = [], [], []
        
        for ep in range(n_episodes):
            env = create_environment(container_size, num_boxes, 300 + ep * 11)
            if env is None: continue
            
            obs, _ = env.reset(seed=300 + ep * 11)
            reward = 0.0
            
            for _ in range(35):
                try:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, r, done, trunc, _ = env.step(action)
                    reward += r
                    if done or trunc: break
                except: break
            
            # 상세 메트릭 계산
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
        
        # 🎯 종합 점수: 보상 20% + 활용률 60% + 성공률 20%
        mean_reward = np.mean(all_rewards)
        mean_util = np.mean(all_utils)
        success_rate = sum(1 for u in all_utils if u >= 0.4) / len(all_utils)
        
        return (mean_reward * 0.2 + mean_util * 100 * 0.6 + success_rate * 20 * 0.2)
        
    except: return 0.0

def run_refined_optimization(n_trials=30):
    """정밀한 최적화 실행"""
    try:
        print(f"🚀 정밀한 2차 최적화 시작 ({n_trials} trials)")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42, n_startup_trials=8),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        )
        
        study.optimize(refined_objective, n_trials=n_trials, timeout=7200)
        
        print(f"\n📈 === 정밀 최적화 결과 ===")
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
        results_file = f"results/refined_optimization_{timestamp}.json"
        
        os.makedirs('results', exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'best_params': study.best_params,
                'best_score': study.best_value,
                'baseline_score': baseline,
                'improvement_percent': improvement,
                'n_trials': len(study.trials)
            }, f, indent=2)
        
        print(f"💾 결과 저장: {results_file}")
        
        if study.best_value > baseline:
            print(f"🎉 새로운 최적 파라미터 발견!")
        
        return study.best_params, study.best_value
        
    except Exception as e:
        print(f"❌ 최적화 실패: {e}")
        return None, 0

if __name__ == "__main__":
    print("🎯 정밀한 2차 하이퍼파라미터 최적화")
    print(f"🚀 Python: {sys.version}")
    
    best_params, best_score = run_refined_optimization(n_trials=30)
    
    if best_params:
        print(f"\n🎉 정밀 최적화 완료!")
        print(f"🏆 최종 최적 파라미터 획득!")
    else:
        print(f"\n❌ 최적화 실패")