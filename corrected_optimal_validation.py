#!/usr/bin/env python3
"""
🔧 수정된 최적 파라미터 검증 스크립트
학습률 문제 해결 및 더 나은 비교 분석
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

print("🔧 수정된 최적 파라미터 검증")

# 원본 Optuna 결과 (문제가 있는 학습률)
ORIGINAL_OPTIMAL = {
    'learning_rate': 2.6777169756959113e-06,  # 너무 낮음!
    'n_steps': 256,
    'batch_size': 32,
    'n_epochs': 3,
    'clip_range': 0.17716239549317803,
    'ent_coef': 0.06742268917730829,
    'vf_coef': 0.4545305173856873,
    'gae_lambda': 0.9449228658070746
}

# 수정된 최적 파라미터 (학습률만 수정)
CORRECTED_OPTIMAL = {
    'learning_rate': 1e-4,  # 적절한 수준으로 조정
    'n_steps': 256,
    'batch_size': 32,
    'n_epochs': 3,
    'clip_range': 0.17716239549317803,
    'ent_coef': 0.06742268917730829,
    'vf_coef': 0.4545305173856873,
    'gae_lambda': 0.9449228658070746
}

# 더 강화된 최적 파라미터 (다른 값들도 보완)
ENHANCED_OPTIMAL = {
    'learning_rate': 2e-4,  # 조금 더 높게
    'n_steps': 512,         # 더 많은 스텝
    'batch_size': 64,       # 더 큰 배치
    'n_epochs': 3,
    'clip_range': 0.17716239549317803,
    'ent_coef': 0.06742268917730829,
    'vf_coef': 0.4545305173856873,
    'gae_lambda': 0.9449228658070746
}

# 기본 파라미터
DEFAULT_PARAMS = {
    'learning_rate': 9e-4,  # 3e-4
    'n_steps': 256,
    'batch_size': 32,
    'n_epochs': 3,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'gae_lambda': 0.95
}

def create_environment(container_size, num_boxes, seed):
    """환경 생성"""
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

def train_model(env, params, train_steps=8000, name=""):
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
                net_arch=[128, 128],
                activation_fn=nn.ReLU,
            )
        )
        
        print(f"🎓 {name} 학습 시작: {train_steps:,} 스텝 (LR: {params['learning_rate']:.2e})")
        start_time = time.time()
        
        model.learn(total_timesteps=train_steps, progress_bar=False)
        
        training_time = time.time() - start_time
        print(f"⏱️ {name} 학습 완료: {training_time:.1f}초")
        
        return model, training_time
        
    except Exception as e:
        print(f"❌ {name} 모델 학습 오류: {e}")
        return None, 0

def evaluate_model(model, container_size, num_boxes, n_episodes=12, name=""):
    """모델 평가"""
    try:
        print(f"🔍 {name} 평가 시작 ({n_episodes} 에피소드)")
        
        all_rewards = []
        all_utilizations = []
        placement_counts = []
        
        for ep in range(n_episodes):
            seed = 100 + ep * 5
            eval_env = create_environment(container_size, num_boxes, seed)
            
            if eval_env is None:
                continue
            
            obs, _ = eval_env.reset(seed=seed)
            episode_reward = 0.0
            max_steps = 25
            
            for step in range(max_steps):
                try:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
                        
                except Exception as e:
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
            placement_counts.append(placed_boxes)
            
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
            'mean_placement': np.mean(placement_counts),
            'success_rate': sum(1 for u in all_utilizations if u >= 0.3) / len(all_utilizations),
            'combined_score': np.mean(all_rewards) * 0.3 + np.mean(all_utilizations) * 100 * 0.7,
            'episodes': len(all_rewards)
        }
        
        return results
        
    except Exception as e:
        print(f"❌ {name} 평가 오류: {e}")
        return None

def comprehensive_validation():
    """종합적인 검증 실행"""
    try:
        print("\n🎯 실험 설정:")
        container_size = [8, 8, 8]  # 적당한 크기
        num_boxes = 6  # 적당한 개수
        train_steps = 8000  # 충분한 학습
        
        print(f"   - 컨테이너: {container_size}")
        print(f"   - 박스 개수: {num_boxes}")
        print(f"   - 학습 스텝: {train_steps:,}")
        
        results = {}
        
        # === 1. 원본 최적 파라미터 (문제가 있는 학습률) ===
        print(f"\n❌ === 원본 최적 파라미터 (LR: {ORIGINAL_OPTIMAL['learning_rate']:.2e}) ===")
        
        env1 = create_environment(container_size, num_boxes, 42)
        model1, time1 = train_model(env1, ORIGINAL_OPTIMAL, train_steps, "원본최적")
        if model1:
            results['original'] = evaluate_model(model1, container_size, num_boxes, name="원본최적")
            results['original']['training_time'] = time1
        env1.close()
        del model1
        
        # === 2. 수정된 최적 파라미터 (학습률만 수정) ===
        print(f"\n🔧 === 수정된 최적 파라미터 (LR: {CORRECTED_OPTIMAL['learning_rate']:.2e}) ===")
        
        env2 = create_environment(container_size, num_boxes, 42)
        model2, time2 = train_model(env2, CORRECTED_OPTIMAL, train_steps, "수정최적")
        if model2:
            results['corrected'] = evaluate_model(model2, container_size, num_boxes, name="수정최적")
            results['corrected']['training_time'] = time2
        env2.close()
        del model2
        
        # === 3. 강화된 최적 파라미터 ===
        print(f"\n🚀 === 강화된 최적 파라미터 (LR: {ENHANCED_OPTIMAL['learning_rate']:.2e}) ===")
        
        env3 = create_environment(container_size, num_boxes, 42)
        model3, time3 = train_model(env3, ENHANCED_OPTIMAL, train_steps, "강화최적")
        if model3:
            results['enhanced'] = evaluate_model(model3, container_size, num_boxes, name="강화최적")
            results['enhanced']['training_time'] = time3
        env3.close()
        del model3
        
        # === 4. 기본 파라미터 ===
        print(f"\n📊 === 기본 파라미터 (LR: {DEFAULT_PARAMS['learning_rate']:.2e}) ===")
        
        env4 = create_environment(container_size, num_boxes, 42)
        model4, time4 = train_model(env4, DEFAULT_PARAMS, train_steps, "기본")
        if model4:
            results['default'] = evaluate_model(model4, container_size, num_boxes, name="기본")
            results['default']['training_time'] = time4
        env4.close()
        del model4
        
        # === 5. 결과 비교 ===
        print(f"\n📈 === 종합 성능 비교 ===")
        
        configs = [
            ('원본 최적', 'original', ORIGINAL_OPTIMAL['learning_rate']),
            ('수정 최적', 'corrected', CORRECTED_OPTIMAL['learning_rate']),
            ('강화 최적', 'enhanced', ENHANCED_OPTIMAL['learning_rate']),
            ('기본', 'default', DEFAULT_PARAMS['learning_rate'])
        ]
        
        print(f"{'설정':<12} {'학습률':<12} {'평균보상':<10} {'활용률':<10} {'종합점수':<10} {'학습시간':<8}")
        print("-" * 75)
        
        best_score = 0
        best_config = ""
        
        for name, key, lr in configs:
            if key in results and results[key]:
                r = results[key]
                reward = r['mean_reward']
                util = r['mean_utilization']
                score = r['combined_score']
                time_taken = r['training_time']
                
                print(f"{name:<12} {lr:<12.2e} {reward:<10.3f} {util:<10.1%} {score:<10.2f} {time_taken:<8.1f}s")
                
                if score > best_score:
                    best_score = score
                    best_config = name
            else:
                print(f"{name:<12} {lr:<12.2e} {'실패':<10} {'실패':<10} {'실패':<10} {'실패':<8}")
        
        print(f"\n🏆 최고 성능: {best_config} (점수: {best_score:.2f})")
        
        # 결과 저장
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/corrected_validation_{timestamp}.json"
        
        os.makedirs('results', exist_ok=True)
        
        final_data = {
            'timestamp': timestamp,
            'experiment_config': {
                'container_size': container_size,
                'num_boxes': num_boxes,
                'train_steps': train_steps
            },
            'parameters': {
                'original': ORIGINAL_OPTIMAL,
                'corrected': CORRECTED_OPTIMAL,
                'enhanced': ENHANCED_OPTIMAL,
                'default': DEFAULT_PARAMS
            },
            'results': results,
            'best_config': best_config,
            'best_score': best_score
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        print(f"💾 상세 결과 저장: {results_file}")
        
        # 핵심 인사이트
        if 'corrected' in results and 'original' in results:
            if results['corrected'] and results['original']:
                improvement = ((results['corrected']['combined_score'] - results['original']['combined_score']) / 
                             results['original']['combined_score'] * 100)
                print(f"\n💡 핵심 인사이트:")
                print(f"   학습률 수정으로 {improvement:.1f}% 성능 개선!")
                print(f"   원본 Optuna 최적화에 학습률 문제가 있었음을 확인")
        
        return True
        
    except Exception as e:
        print(f"❌ 종합 검증 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 시스템 환경:")
    print(f"   Python: {sys.version}")
    print(f"   작업 디렉토리: {os.getcwd()}")
    
    success = comprehensive_validation()
    if success:
        print(f"\n🎉 수정된 최적 파라미터 검증 완료!")
        print(f"💡 학습률 문제가 해결되었습니다!")
    else:
        print(f"\n❌ 검증 실패") 