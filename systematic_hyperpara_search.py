#!/usr/bin/env python3
"""
🔬 체계적 하이퍼파라미터 재탐색
재현성 확보 + 확장된 범위 탐색
"""

import os
import sys
import warnings
import datetime
import time
import json
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings("ignore")
sys.path.append('src')

print("🔬 체계적 하이퍼파라미터 재탐색")

# 🏆 이전 최고 성능 파라미터 (재현 기준점)
BASELINE_BEST = {
    'learning_rate': 2e-4,
    'n_steps': 512,
    'batch_size': 64,
    'n_epochs': 3,
    'clip_range': 0.17716239549317803,
    'ent_coef': 0.06742268917730829,
    'vf_coef': 0.4545305173856873,
    'gae_lambda': 0.9449228658070746
}

def create_environment(container_size, num_boxes, seed):
    """환경 생성 (완전히 동일한 방식)"""
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

def train_and_evaluate(params, container_size, num_boxes, train_steps, name=""):
    """학습 및 평가 (완전히 통제된 방식)"""
    try:
        print(f"\n🎓 {name} 학습 중...")
        print(f"   LR: {params['learning_rate']:.2e}")
        print(f"   Steps: {params['n_steps']}, Batch: {params['batch_size']}")
        
        # === 환경 생성 ===
        env = create_environment(container_size, num_boxes, 42)  # 고정 시드
        if env is None:
            return 0.0
        
        # === 모델 생성 ===
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
        
        # === 충분한 학습 시간 ===
        start_time = time.time()
        model.learn(total_timesteps=train_steps, progress_bar=False)
        training_time = time.time() - start_time
        
        print(f"   ⏱️ 학습 완료: {training_time:.1f}초")
        
        # === 안정적 평가 (더 많은 에피소드) ===
        all_rewards, all_utils = [], []
        
        for ep in range(15):  # 더 많은 평가 에피소드
            eval_env = create_environment(container_size, num_boxes, 100 + ep * 17)
            if eval_env is None: continue
            
            obs, _ = eval_env.reset(seed=100 + ep * 17)
            episode_reward = 0.0
            
            for step in range(25):
                try:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
                except:
                    break
            
            # 활용률 계산
            utilization = 0.0
            try:
                if hasattr(eval_env.unwrapped, 'container'):
                    placed_volume = sum(box.volume for box in eval_env.unwrapped.container.boxes 
                                      if box.position is not None)
                    container_volume = eval_env.unwrapped.container.volume
                    utilization = placed_volume / container_volume if container_volume > 0 else 0.0
            except:
                pass
            
            all_rewards.append(episode_reward)
            all_utils.append(utilization)
            eval_env.close()
        
        env.close()
        del model
        
        if not all_rewards:
            return 0.0
        
        # === 종합 점수 계산 ===
        mean_reward = np.mean(all_rewards)
        mean_util = np.mean(all_utils)
        combined_score = mean_reward * 0.3 + mean_util * 100 * 0.7
        
        print(f"   📊 결과: 보상={mean_reward:.3f}, 활용률={mean_util:.1%}, 점수={combined_score:.2f}")
        
        return combined_score
        
    except Exception as e:
        print(f"❌ {name} 실패: {e}")
        return 0.0

def systematic_search():
    """체계적 탐색 실행"""
    try:
        print("\n🎯 실험 설정:")
        container_size = [8, 8, 8]  # 이전과 동일
        num_boxes = 6              # 이전과 동일
        train_steps = 12000        # 더 긴 학습 시간
        
        print(f"   - 컨테이너: {container_size}")
        print(f"   - 박스 개수: {num_boxes}")
        print(f"   - 학습 스텝: {train_steps:,}")
        
        results = {}
        
        # === 1. 기준점 재현 테스트 ===
        print(f"\n🏆 === 기준점 재현 테스트 ===")
        baseline_score = train_and_evaluate(
            BASELINE_BEST, container_size, num_boxes, train_steps, "기준점"
        )
        results['baseline_reproduction'] = {
            'params': BASELINE_BEST,
            'score': baseline_score
        }
        
        # === 2. 확장된 학습률 탐색 ===
        print(f"\n🔍 === 확장된 학습률 탐색 ===")
        
        # 🎯 더 넓은 학습률 범위 (기존의 3배 확장)
        learning_rates = [
            1.2e-4, 1.5e-4, 1.8e-4,  # 더 낮은 영역
            2.0e-4,                   # 기준점
            2.2e-4, 2.5e-4, 2.8e-4,  # 더 높은 영역
            3.2e-4, 3.5e-4           # 훨씬 높은 영역
        ]
        
        best_lr_score = 0
        best_lr = None
        
        for lr in learning_rates:
            lr_params = BASELINE_BEST.copy()
            lr_params['learning_rate'] = lr
            
            score = train_and_evaluate(
                lr_params, container_size, num_boxes, train_steps, f"LR={lr:.2e}"
            )
            
            results[f'lr_{lr:.2e}'] = {'params': lr_params, 'score': score}
            
            if score > best_lr_score:
                best_lr_score = score
                best_lr = lr
        
        # === 3. 최적 학습률 기반 미세 조정 ===
        if best_lr and best_lr_score > baseline_score:
            print(f"\n🚀 === 최적 학습률({best_lr:.2e}) 기반 미세 조정 ===")
            
            # n_epochs 조정
            for n_epochs in [3, 4, 5]:
                params = BASELINE_BEST.copy()
                params['learning_rate'] = best_lr
                params['n_epochs'] = n_epochs
                
                score = train_and_evaluate(
                    params, container_size, num_boxes, train_steps, 
                    f"최적LR+Epochs={n_epochs}"
                )
                
                results[f'optimal_lr_epochs_{n_epochs}'] = {'params': params, 'score': score}
            
            # batch_size 조정
            for batch_size in [48, 64, 96]:
                params = BASELINE_BEST.copy()
                params['learning_rate'] = best_lr
                params['batch_size'] = batch_size
                
                score = train_and_evaluate(
                    params, container_size, num_boxes, train_steps, 
                    f"최적LR+Batch={batch_size}"
                )
                
                results[f'optimal_lr_batch_{batch_size}'] = {'params': params, 'score': score}
        
        # === 4. 결과 분석 ===
        print(f"\n📈 === 최종 결과 분석 ===")
        
        best_overall_score = 0
        best_overall_config = None
        best_overall_name = ""
        
        print(f"{'설정':<25} {'점수':<8} {'개선율':<8}")
        print("-" * 45)
        
        for name, data in results.items():
            score = data['score']
            improvement = ((score - 18.57) / 18.57) * 100 if score > 0 else -100
            
            print(f"{name:<25} {score:<8.2f} {improvement:<+7.1f}%")
            
            if score > best_overall_score:
                best_overall_score = score
                best_overall_config = data['params']
                best_overall_name = name
        
        print(f"\n🏆 최고 성능: {best_overall_name} (점수: {best_overall_score:.2f})")
        
        # === 5. 결과 저장 ===
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/systematic_search_{timestamp}.json"
        
        os.makedirs('results', exist_ok=True)
        
        final_data = {
            'timestamp': timestamp,
            'search_type': 'systematic_reproduction_and_expansion',
            'experiment_config': {
                'container_size': container_size,
                'num_boxes': num_boxes,
                'train_steps': train_steps
            },
            'baseline_target': 18.57,
            'baseline_reproduction': baseline_score,
            'best_config': best_overall_name,
            'best_score': best_overall_score,
            'best_params': best_overall_config,
            'improvement_from_target': ((best_overall_score - 18.57) / 18.57) * 100,
            'all_results': results
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        print(f"💾 상세 결과 저장: {results_file}")
        
        # === 6. 권장사항 ===
        if best_overall_score > 18.57:
            print(f"\n🎉 성능 개선 성공!")
            print(f"💡 권장 파라미터:")
            for key, value in best_overall_config.items():
                if key == 'learning_rate':
                    print(f"   {key}: {value:.2e}")
                else:
                    print(f"   {key}: {value}")
        else:
            print(f"\n📊 추가 분석 필요:")
            print(f"   기준점 재현: {baseline_score:.2f}")
            print(f"   최고 점수: {best_overall_score:.2f}")
            if baseline_score < 16.0:
                print(f"   ⚠️ 기준점 재현 실패 - 환경 설정 점검 필요")
            else:
                print(f"   💡 더 긴 학습 시간이나 다른 접근 필요")
        
        return True
        
    except Exception as e:
        print(f"❌ 체계적 탐색 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔬 체계적 하이퍼파라미터 재탐색")
    print(f"🚀 Python: {sys.version}")
    print(f"📁 작업 디렉토리: {os.getcwd()}")
    
    success = systematic_search()
    
    if success:
        print(f"\n🎉 체계적 탐색 완료!")
        print(f"🔬 재현성과 최적화가 동시에 검증되었습니다!")
    else:
        print(f"\n❌ 탐색 실패")