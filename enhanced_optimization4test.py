#!/usr/bin/env python3
"""
🏆 프로덕션 최적 설정 최종 검증 스크립트
Phase 4에서 검증된 최고 성능 설정 (20.591점)으로 최종 테스트
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
import warnings

# 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings("ignore")
sys.path.append('src')

# 🏆 검증된 최고 성능 설정 (Phase 4 결과)
PRODUCTION_OPTIMAL = {
    "learning_rate": 0.00013,
    "n_steps": 768,
    "batch_size": 96,
    "n_epochs": 5,
    "clip_range": 0.18,
    "ent_coef": 0.008,
    "vf_coef": 0.5,
    "gae_lambda": 0.96,
    "net_arch": {"pi": [256, 128, 64], "vf": [256, 128, 64]}
}

def create_production_env(container_size=[10, 10, 10], num_boxes=12, seed=42):
    """프로덕션 환경 생성"""
    try:
        from train_maskable_ppo import make_env
        
        env = make_env(
            container_size=container_size,
            num_boxes=num_boxes,
            num_visible_boxes=3,
            seed=seed,
            render_mode=None,
            random_boxes=False,
            only_terminal_reward=False,
            improved_reward_shaping=True,
        )()
        
        print(f"✅ 프로덕션 환경 생성: 컨테이너{container_size}, 박스{num_boxes}개")
        return env
        
    except Exception as e:
        print(f"❌ 환경 생성 실패: {e}")
        return None

def train_production_model(env, timesteps=50000):
    """프로덕션 최적 설정으로 모델 학습"""
    try:
        import torch
        import torch.nn as nn
        from sb3_contrib import MaskablePPO
        
        print(f"🚀 프로덕션 학습 시작: {timesteps:,} 스텝")
        print(f"📊 최적 설정: LR={PRODUCTION_OPTIMAL['learning_rate']:.2e}, "
              f"Steps={PRODUCTION_OPTIMAL['n_steps']}, "
              f"Batch={PRODUCTION_OPTIMAL['batch_size']}")
        
        start_time = time.time()
        
        # 최적 설정으로 모델 생성
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=PRODUCTION_OPTIMAL['learning_rate'],
            n_steps=PRODUCTION_OPTIMAL['n_steps'],
            batch_size=PRODUCTION_OPTIMAL['batch_size'],
            n_epochs=PRODUCTION_OPTIMAL['n_epochs'],
            gamma=0.99,
            gae_lambda=PRODUCTION_OPTIMAL['gae_lambda'],
            clip_range=PRODUCTION_OPTIMAL['clip_range'],
            ent_coef=PRODUCTION_OPTIMAL['ent_coef'],
            vf_coef=PRODUCTION_OPTIMAL['vf_coef'],
            max_grad_norm=0.5,
            verbose=1,
            seed=42,
            policy_kwargs=dict(
                net_arch=PRODUCTION_OPTIMAL['net_arch'],
                activation_fn=nn.ReLU,
                share_features_extractor=True,
            )
        )
        
        # 학습 실행
        model.learn(total_timesteps=timesteps, progress_bar=True)
        
        duration = time.time() - start_time
        print(f"⏱️ 학습 완료: {duration/60:.1f}분")
        
        return model, duration
        
    except Exception as e:
        print(f"❌ 학습 실패: {e}")
        return None, 0

def evaluate_production_model(model, container_size=[10, 10, 10], num_boxes=12, n_episodes=50):
    """강화된 프로덕션 평가 (더 많은 에피소드)"""
    print(f"🔍 프로덕션 평가 시작: {n_episodes} 에피소드")
    
    all_rewards = []
    all_utilizations = []
    placement_counts = []
    success_count = 0
    
    for ep in range(n_episodes):
        seed = 200 + ep * 3  # 다양한 시드
        eval_env = create_production_env(container_size, num_boxes, seed)
        
        if eval_env is None:
            continue
        
        obs, _ = eval_env.reset(seed=seed)
        episode_reward = 0.0
        
        for step in range(50):  # 최대 50스텝
            try:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
                    
            except Exception as e:
                break
        
        # 성과 계산
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
        
        # 성공 기준: 활용률 25% 이상 또는 박스 50% 이상
        if utilization >= 0.25 or placed_boxes >= num_boxes * 0.5:
            success_count += 1
        
        all_rewards.append(episode_reward)
        all_utilizations.append(utilization)
        placement_counts.append(placed_boxes)
        
        if ep < 10 or ep % 10 == 0:
            print(f"   에피소드 {ep+1}: 보상={episode_reward:.3f}, "
                  f"활용률={utilization:.1%}, 박스={placed_boxes}개")
        
        eval_env.close()
    
    if not all_rewards:
        return None
    
    results = {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_utilization': np.mean(all_utilizations),
        'std_utilization': np.std(all_utilizations),
        'mean_placement': np.mean(placement_counts),
        'max_placement': max(placement_counts),
        'success_rate': success_count / len(all_rewards),
        'combined_score': np.mean(all_rewards) * 0.3 + np.mean(all_utilizations) * 100 * 0.7,
        'episodes': len(all_rewards),
        'all_rewards': all_rewards,
        'all_utilizations': all_utilizations
    }
    
    return results

def production_final_test(timesteps=50000, eval_episodes=50):
    """프로덕션 최종 테스트 실행"""
    print("🏆 프로덕션 최적 설정 최종 검증 시작")
    print(f"📊 목표: 20.591점 재현 및 안정성 검증")
    print("="*60)
    
    # 환경 생성
    container_size = [10, 10, 10]
    num_boxes = 12
    
    env = create_production_env(container_size, num_boxes, 42)
    if env is None:
        return False
    
    # 모델 학습
    print(f"\n🎓 1단계: 프로덕션 모델 학습 ({timesteps:,} 스텝)")
    model, train_time = train_production_model(env, timesteps)
    
    if model is None:
        env.close()
        return False
    
    # 모델 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/production_optimal_{timestamp}"
    os.makedirs('models', exist_ok=True)
    model.save(model_path)
    print(f"💾 모델 저장: {model_path}")
    
    # 강화된 평가
    print(f"\n📊 2단계: 강화된 평가 ({eval_episodes} 에피소드)")
    results = evaluate_production_model(model, container_size, num_boxes, eval_episodes)
    
    env.close()
    
    if results is None:
        return False
    
    # 결과 분석 및 출력
    print("\n" + "="*60)
    print("🏆 프로덕션 최종 테스트 결과")
    print("="*60)
    print(f"📊 종합 점수: {results['combined_score']:.3f}")
    print(f"🎯 목표 대비: {(results['combined_score']/20.591*100):.1f}% (목표: 20.591)")
    print(f"💰 평균 보상: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"📦 평균 활용률: {results['mean_utilization']:.1%} ± {results['std_utilization']:.1%}")
    print(f"🎲 평균 배치: {results['mean_placement']:.1f}개 (최대: {results['max_placement']}개)")
    print(f"✅ 성공률: {results['success_rate']:.1%}")
    print(f"⏱️ 학습 시간: {train_time/60:.1f}분")
    
    # 성능 판정
    if results['combined_score'] >= 20.0:
        print(f"🎉 우수! 목표 성능 달성 또는 근접")
    elif results['combined_score'] >= 18.57:
        print(f"✅ 성공! Phase 3 목표 달성")
    else:
        print(f"📈 개선 필요: 추가 튜닝 권장")
    
    # 상세 결과 저장
    final_results = {
        'timestamp': timestamp,
        'test_type': 'production_final',
        'params': PRODUCTION_OPTIMAL,
        'config': {
            'container_size': container_size,
            'num_boxes': num_boxes,
            'timesteps': timesteps,
            'eval_episodes': eval_episodes
        },
        'performance': results,
        'training_time_minutes': train_time/60,
        'model_path': model_path,
        'target_score': 20.591,
        'achievement_rate': results['combined_score']/20.591*100
    }
    
    results_file = f"results/production_final_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n💾 상세 결과 저장: {results_file}")
    
    return results['combined_score'] >= 18.57

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='프로덕션 최적 설정 최종 테스트')
    parser.add_argument('--timesteps', type=int, default=50000, help='학습 스텝 수')
    parser.add_argument('--episodes', type=int, default=50, help='평가 에피소드 수')
    parser.add_argument('--quick', action='store_true', help='빠른 테스트 (25000 스텝)')
    
    args = parser.parse_args()
    
    if args.quick:
        timesteps = 25000
        episodes = 30
        print("⚡ 빠른 테스트 모드")
    else:
        timesteps = args.timesteps
        episodes = args.episodes
        print("🏆 완전 테스트 모드")
    
    print(f"🚀 설정: {timesteps:,} 스텝, {episodes} 에피소드")
    
    start_time = time.time()
    success = production_final_test(timesteps, episodes)
    total_time = time.time() - start_time
    
    print(f"\n⏱️ 총 소요 시간: {total_time/60:.1f}분")
    
    if success:
        print("🎉 프로덕션 최종 테스트 성공!")
    else:
        print("📈 성능 개선이 필요합니다.")

if __name__ == "__main__":
    main() 