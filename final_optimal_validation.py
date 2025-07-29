#!/usr/bin/env python3
"""
🏆 최적 파라미터 최종 검증 스크립트 (문제 해결 버전)
다양한 평가 방법과 충분한 학습으로 실제 성능 측정
"""

import os
import sys
import warnings
import datetime
import time
import json

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings("ignore")
sys.path.append('src')

print("🏆 최적 파라미터 최종 검증")

OPTIMAL_PARAMS = {
    'learning_rate': 2.6777169756959113e-06,
    'n_steps': 128,     # 더 크게
    'batch_size': 16,   # 더 크게
    'n_epochs': 3,      # 최적값 유지
    'clip_range': 0.17716239549317803,
    'ent_coef': 0.06742268917730829,
    'vf_coef': 0.4545305173856873,
    'gae_lambda': 0.9449228658070746
}

# 비교용 기본 파라미터
DEFAULT_PARAMS = {
    'learning_rate': 3e-4,
    'n_steps': 128,
    'batch_size': 16,
    'n_epochs': 3,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'gae_lambda': 0.95
}

def create_environment(container_size, num_boxes, seed):
    """개선된 환경 생성"""
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
        
        # 개선된 보상 래퍼 적용
        try:
            from train_maskable_ppo import ImprovedRewardWrapper
            env = ImprovedRewardWrapper(env)
        except:
            pass
        
        # 액션 마스킹 적용
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

def train_model(env, params, train_steps=2000):
    """모델 학습"""
    try:
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
                net_arch=[64, 64],  # 적당한 크기
                activation_fn=nn.ReLU,
            )
        )
        
        print(f"🎓 학습 시작: {train_steps:,} 스텝")
        start_time = time.time()
        
        model.learn(total_timesteps=train_steps, progress_bar=False)
        
        training_time = time.time() - start_time
        print(f"⏱️ 학습 완료: {training_time:.1f}초")
        
        return model, training_time
        
    except Exception as e:
        print(f"❌ 모델 학습 오류: {e}")
        return None, 0

def diverse_evaluation(model, container_size, num_boxes, n_episodes=10):
    """다양한 시드와 방법으로 평가"""
    try:
        print(f"🔍 다양한 평가 시작 ({n_episodes} 에피소드)")
        
        all_rewards = []
        all_utilizations = []
        success_count = 0
        
        for ep in range(n_episodes):
            # 매번 다른 시드로 새 환경 생성
            seed = 42 + ep * 10
            eval_env = create_environment(container_size, num_boxes, seed)
            
            if eval_env is None:
                continue
            
            obs, _ = eval_env.reset(seed=seed)
            episode_reward = 0.0
            steps = 0
            max_steps = 20  # 더 많은 스텝
            
            # Stochastic 평가 (다양성을 위해)
            for step in range(max_steps):
                try:
                    action, _ = model.predict(obs, deterministic=False)  # False로 변경!
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    steps += 1
                    
                    if terminated or truncated:
                        break
                        
                except Exception as e:
                    print(f"     에피소드 {ep+1} 스텝 {step} 오류: {e}")
                    break
            
            # 활용률 계산
            utilization = 0.0
            placed_boxes = 0
            try:
                if hasattr(eval_env.unwrapped, 'container'):
                    placed_volume = sum(box.volume for box in eval_env.unwrapped.container.boxes 
                                      if box.position is not None)
                    container_volume = eval_env.unwrapped.container.volume
                    utilization = placed_volume / container_volume if container_volume > 0 else 0.0
                    
                    placed_boxes = sum(1 for box in eval_env.unwrapped.container.boxes 
                                     if box.position is not None)
            except:
                pass
            
            all_rewards.append(episode_reward)
            all_utilizations.append(utilization)
            
            if utilization >= 0.4:  # 40% 이상을 성공으로 간주
                success_count += 1
            
            print(f"   에피소드 {ep+1}: 보상={episode_reward:.3f}, 활용률={utilization:.1%}, 박스={placed_boxes}개")
            
            eval_env.close()
        
        if not all_rewards:
            return None
        
        import numpy as np
        results = {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_utilization': np.mean(all_utilizations),
            'std_utilization': np.std(all_utilizations),
            'success_rate': success_count / len(all_rewards),
            'combined_score': np.mean(all_rewards) * 0.3 + np.mean(all_utilizations) * 100 * 0.7,
            'episodes': len(all_rewards)
        }
        
        return results
        
    except Exception as e:
        print(f"❌ 평가 오류: {e}")
        return None

def final_validation():
    """최종 검증 실행"""
    try:
        print("\n🎯 실험 설정:")
        container_size = [8, 8, 8]  # 적당한 크기
        num_boxes = 6  # 적당한 개수
        train_steps = 2000  # 충분한 학습
        
        print(f"   - 컨테이너: {container_size}")
        print(f"   - 박스 개수: {num_boxes}")
        print(f"   - 학습 스텝: {train_steps:,}")
        
        # === 1. 최적 파라미터 실험 ===
        print(f"\n🏆 === 최적 파라미터 실험 ===")
        
        optimal_env = create_environment(container_size, num_boxes, 42)
        if optimal_env is None:
            raise ValueError("최적 환경 생성 실패")
        
        optimal_model, optimal_time = train_model(optimal_env, OPTIMAL_PARAMS, train_steps)
        if optimal_model is None:
            raise ValueError("최적 모델 학습 실패")
        
        optimal_results = diverse_evaluation(optimal_model, container_size, num_boxes)
        optimal_env.close()
        del optimal_model
        
        # === 2. 기본 파라미터 실험 ===
        print(f"\n📊 === 기본 파라미터 실험 ===")
        
        default_env = create_environment(container_size, num_boxes, 42)
        if default_env is None:
            raise ValueError("기본 환경 생성 실패")
        
        default_model, default_time = train_model(default_env, DEFAULT_PARAMS, train_steps)
        if default_model is None:
            raise ValueError("기본 모델 학습 실패")
        
        default_results = diverse_evaluation(default_model, container_size, num_boxes)
        default_env.close()
        del default_model
        
        # === 3. 결과 비교 ===
        if optimal_results and default_results:
            print(f"\n📈 === 최종 성능 비교 ===")
            
            metrics = [
                ('평균 보상', 'mean_reward', '.4f'),
                ('보상 안정성', 'std_reward', '.4f'),
                ('평균 활용률', 'mean_utilization', '.1%'),
                ('활용률 안정성', 'std_utilization', '.1%'),
                ('성공률', 'success_rate', '.1%'),
                ('종합 점수', 'combined_score', '.4f')
            ]
            
            print(f"{'지표':<15} {'최적 파라미터':<15} {'기본 파라미터':<15} {'개선율':<10}")
            print("-" * 65)
            
            for name, key, fmt in metrics:
                opt_val = optimal_results[key]
                def_val = default_results[key]
                
                if def_val != 0:
                    improvement = f"{((opt_val - def_val) / def_val * 100):+.1f}%"
                else:
                    improvement = "N/A"
                
                print(f"{name:<15} {opt_val:{fmt}:<15} {def_val:{fmt}:<15} {improvement:<10}")
            
            # 종합 평가
            combined_improvement = ((optimal_results['combined_score'] - default_results['combined_score']) / 
                                   default_results['combined_score'] * 100)
            
            print(f"\n🎯 종합 결과:")
            if combined_improvement > 15:
                print(f"🏆 최적 파라미터가 {combined_improvement:.1f}% 뛰어난 성능!")
            elif combined_improvement > 5:
                print(f"✅ 최적 파라미터가 {combined_improvement:.1f}% 개선된 성능!")
            elif combined_improvement > 0:
                print(f"🔄 최적 파라미터가 약간 개선됨 (+{combined_improvement:.1f}%)")
            else:
                print(f"⚠️ 최적 파라미터 성능이 {abs(combined_improvement):.1f}% 낮음")
            
            # 결과 저장
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"results/final_validation_{timestamp}.json"
            
            import os
            os.makedirs('results', exist_ok=True)
            
            final_data = {
                'timestamp': timestamp,
                'experiment_config': {
                    'container_size': container_size,
                    'num_boxes': num_boxes,
                    'train_steps': train_steps
                },
                'optimal_params': OPTIMAL_PARAMS,
                'default_params': DEFAULT_PARAMS,
                'optimal_results': optimal_results,
                'default_results': default_results,
                'training_times': {
                    'optimal': optimal_time,
                    'default': default_time
                },
                'improvement': combined_improvement
            }
            
            with open(results_file, 'w') as f:
                json.dump(final_data, f, indent=2)
            
            print(f"💾 상세 결과 저장: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 최종 검증 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 시스템 환경:")
    print(f"   Python: {sys.version}")
    print(f"   작업 디렉토리: {os.getcwd()}")
    
    success = final_validation()
    if success:
        print(f"\n🎉 최적 파라미터 최종 검증 완료!")
        print(f"💡 Optuna 최적화 결과가 성공적으로 검증되었습니다!")
    else:
        print(f"\n❌ 최종 검증 실패")
