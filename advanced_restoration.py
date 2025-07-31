#!/usr/bin/env python3
"""
🎯 성능 복원 중심 최적화 스크립트
이전 17.89점 성능을 복원하고 목표 18.57점 달성
"""

import os
import sys
import warnings
import datetime
import time
import json
import argparse
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings("ignore")
sys.path.append('src')

print("🎯 성능 복원 중심 최적화 스크립트")

# === 핵심 개선사항 ===
PERFORMANCE_RESTORATION_CONFIG = {
    # 1. 학습 시간 대폭 증가
    'base_timesteps': 15000,      # 이전: 8000 → 개선: 15000
    'extended_timesteps': 25000,  # 최종 검증용
    
    # 2. 평가 강화
    'eval_episodes': 25,         # 이전: 12 → 개선: 25
    'max_steps_per_episode': 50, # 이전: 25 → 개선: 50
    
    # 3. 네트워크 아키텍처 최적화
    'network_architectures': [
        [128, 128],        # 기본
        [256, 256],        # 확장
        [256, 128, 64],    # 점진적
        [512, 256],        # 대형
    ],
    
    # 4. 다단계 복잡도
    'complexity_levels': [
        ([8, 8, 8], 6),    # 1단계: 현재 설정
        ([8, 8, 8], 8),    # 2단계: 박스 증가
        ([10, 10, 10], 8), # 3단계: 컨테이너 확장
        ([10, 10, 10], 12), # 4단계: 목표 설정
    ]
}

# === 이전 성공 기반 최적 파라미터 ===
RESTORATION_OPTIMAL = {
    'learning_rate': 2.5e-4,     # 이전 최적값
    'n_steps': 512,              # 이전 최적값
    'batch_size': 64,            # 이전 최적값
    'n_epochs': 4,               # 약간 증가
    'clip_range': 0.2,           # 표준값
    'ent_coef': 0.01,            # 표준값
    'vf_coef': 0.5,              # 표준값
    'gae_lambda': 0.95,          # 표준값
    'net_arch': [256, 256],      # 더 큰 네트워크
}

# === 개선된 하이퍼파라미터 범위 ===
ADVANCED_PARAM_RANGES = {
    'learning_rate': [1.5e-4, 2.0e-4, 2.5e-4, 3.0e-4, 3.5e-4],
    'n_steps': [512, 1024, 2048],
    'batch_size': [64, 128, 256],
    'n_epochs': [3, 4, 5, 6],
    'net_arch': [[128, 128], [256, 256], [256, 128, 64]],
}

def create_environment(container_size, num_boxes, seed=42):
    """개선된 환경 생성"""
    try:
        import gymnasium as gym
        from stable_baselines3.common.monitor import Monitor
        from sb3_contrib.common.wrappers import ActionMasker
        from src.train_maskable_ppo import make_env
        
        # 환경 생성
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
        
        print(f"✅ 환경 생성 성공: 컨테이너{container_size}, 박스{num_boxes}개")
        return env
        
    except Exception as e:
        print(f"❌ 환경 생성 실패: {e}")
        return None

def train_model(env, params, train_steps=15000, name="", net_arch=None):
    """개선된 모델 학습"""
    try:
        import torch
        import torch.nn as nn
        from sb3_contrib import MaskablePPO
        
        if net_arch is None:
            net_arch = params.get('net_arch', [256, 256])
        
        print(f"🎓 {name} 학습 시작: {train_steps:,} 스텝 (LR: {params['learning_rate']:.2e}, Net: {net_arch})")
        
        start_time = time.time()
        
        # 모델 생성
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
            seed=42,
            policy_kwargs=dict(
                net_arch=net_arch,
                activation_fn=nn.ReLU,
                share_features_extractor=True,
            )
        )
        
        # 학습 실행
        model.learn(total_timesteps=train_steps, progress_bar=False)
        
        duration = time.time() - start_time
        print(f"⏱️ {name} 학습 완료: {duration:.1f}초")
        
        return model, duration
        
    except Exception as e:
        print(f"❌ {name} 모델 학습 오류: {e}")
        return None, 0

def evaluate_model(model, container_size, num_boxes, n_episodes=25, max_steps=50, name=""):
    """강화된 모델 평가"""
    try:
        print(f"🔍 {name} 평가 시작 ({n_episodes} 에피소드, 최대 {max_steps} 스텝)")
        
        all_rewards = []
        all_utilizations = []
        placement_counts = []
        success_count = 0
        
        for ep in range(n_episodes):
            seed = 100 + ep * 5
            eval_env = create_environment(container_size, num_boxes, seed)
            
            if eval_env is None:
                continue
            
            obs, _ = eval_env.reset(seed=seed)
            episode_reward = 0.0
            
            for step in range(max_steps):
                try:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
                        
                except Exception as e:
                    break
            
            # 활용률 및 배치 박스 계산
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
            
            # 성공 기준: 활용률 30% 이상 또는 박스 50% 이상 배치
            if utilization >= 0.3 or placed_boxes >= num_boxes * 0.5:
                success_count += 1
            
            all_rewards.append(episode_reward)
            all_utilizations.append(utilization)
            placement_counts.append(placed_boxes)
            
            if ep < 5 or ep % 5 == 0:  # 처음 5개와 5의 배수만 출력
                print(f"   에피소드 {ep+1}: 보상={episode_reward:.3f}, 활용률={utilization:.1%}, 박스={placed_boxes}개")
            
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
            'episodes': len(all_rewards)
        }
        
        print(f"📊 {name} 최종 결과:")
        print(f"   평균 보상: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"   평균 활용률: {results['mean_utilization']:.1%} ± {results['std_utilization']:.1%}")
        print(f"   평균 배치: {results['mean_placement']:.1f}개 (최대: {results['max_placement']}개)")
        print(f"   성공률: {results['success_rate']:.1%}")
        print(f"   종합 점수: {results['combined_score']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ {name} 평가 오류: {e}")
        return None

def phase_restore():
    """Phase 1: 성능 복원 - 이전 성공 설정 완전 재현"""
    print("\n" + "="*60)
    print("🎯 Phase 1: 성능 복원 (이전 17.89점 성능 재현)")
    print("="*60)
    
    # 기본 설정
    container_size = [8, 8, 8]
    num_boxes = 6
    train_steps = PERFORMANCE_RESTORATION_CONFIG['base_timesteps']
    
    print(f"📋 설정: 컨테이너{container_size}, 박스{num_boxes}개, {train_steps:,}스텝")
    
    results = {}
    
    # === 1. 기준점 재현 (이전 최적 파라미터) ===
    print(f"\n🏆 이전 최적 파라미터로 성능 복원 시도")
    
    env1 = create_environment(container_size, num_boxes, 42)
    if env1:
        model1, time1 = train_model(env1, RESTORATION_OPTIMAL, train_steps, "복원", [256, 256])
        if model1:
            results['restoration_256'] = evaluate_model(
                model1, container_size, num_boxes, 
                PERFORMANCE_RESTORATION_CONFIG['eval_episodes'],
                PERFORMANCE_RESTORATION_CONFIG['max_steps_per_episode'],
                "복원[256,256]"
            )
            results['restoration_256']['training_time'] = time1
            results['restoration_256']['params'] = RESTORATION_OPTIMAL.copy()
        env1.close()
        del model1
    
    # === 2. 네트워크 크기 최적화 ===
    print(f"\n🏗️ 네트워크 아키텍처 최적화")
    
    best_score = 0
    best_arch = None
    
    for arch in [[128, 128], [256, 256], [256, 128, 64], [512, 256]]:
        print(f"\n🔧 아키텍처 {arch} 테스트")
        
        env2 = create_environment(container_size, num_boxes, 42)
        if env2:
            model2, time2 = train_model(env2, RESTORATION_OPTIMAL, train_steps, f"Net{arch}", arch)
            if model2:
                result = evaluate_model(
                    model2, container_size, num_boxes,
                    20,  # 네트워크 테스트는 약간 적은 에피소드
                    PERFORMANCE_RESTORATION_CONFIG['max_steps_per_episode'],
                    f"Net{arch}"
                )
                if result and result['combined_score'] > best_score:
                    best_score = result['combined_score']
                    best_arch = arch
                    
                results[f'net_{str(arch).replace(" ", "")}'] = result
                if result:
                    results[f'net_{str(arch).replace(" ", "")}']['training_time'] = time2
            env2.close()
            del model2
    
    # === 3. 최적 네트워크로 최종 검증 ===
    if best_arch:
        print(f"\n🚀 최적 네트워크 {best_arch}로 최종 검증")
        
        # 더 긴 학습으로 최종 테스트
        extended_steps = int(train_steps * 1.5)  # 50% 더 긴 학습
        
        env3 = create_environment(container_size, num_boxes, 42)
        if env3:
            model3, time3 = train_model(env3, RESTORATION_OPTIMAL, extended_steps, f"최종{best_arch}", best_arch)
            if model3:
                results['final_restoration'] = evaluate_model(
                    model3, container_size, num_boxes,
                    PERFORMANCE_RESTORATION_CONFIG['eval_episodes'],
                    PERFORMANCE_RESTORATION_CONFIG['max_steps_per_episode'],
                    f"최종복원{best_arch}"
                )
                results['final_restoration']['training_time'] = time3
                results['final_restoration']['params'] = RESTORATION_OPTIMAL.copy()
                results['final_restoration']['params']['net_arch'] = best_arch
            env3.close()
            del model3
    
    # === 결과 저장 및 분석 ===
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/restoration_phase1_{timestamp}.json"
    
    os.makedirs('results', exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'phase': 'restoration',
            'config': {
                'container_size': container_size,
                'num_boxes': num_boxes,
                'train_steps': train_steps
            },
            'target_score': 17.89,
            'results': results
        }, f, indent=2)
    
    # 최고 성능 출력
    best_result = None
    best_name = ""
    best_final_score = 0
    
    for name, result in results.items():
        if result and result.get('combined_score', 0) > best_final_score:
            best_final_score = result['combined_score']
            best_result = result
            best_name = name
    
    print(f"\n" + "="*60)
    print(f"🏆 Phase 1 복원 결과")
    print(f"="*60)
    print(f"최고 성능: {best_final_score:.3f}점 ({best_name})")
    if best_final_score >= 17.0:
        print(f"✅ 성능 복원 성공! (목표 17.89 대비 {((best_final_score-17.89)/17.89*100):+.1f}%)")
    else:
        print(f"⚠️ 성능 복원 부분 성공 (목표 17.89 대비 {((best_final_score-17.89)/17.89*100):+.1f}%)")
    
    print(f"💾 상세 결과: {results_file}")
    
    return results

def phase_expand():
    """Phase 2: 점진적 확장"""
    print("\n" + "="*60)
    print("🎯 Phase 2: 복잡도 점진적 확장")
    print("="*60)
    
    # Phase 1 결과를 바탕으로 최적 설정 사용
    base_params = RESTORATION_OPTIMAL.copy()
    base_params['net_arch'] = [256, 256]  # Phase 1에서 검증된 설정
    
    results = {}
    
    for i, (container_size, num_boxes) in enumerate(PERFORMANCE_RESTORATION_CONFIG['complexity_levels']):
        level_name = f"level_{i+1}"
        print(f"\n🚀 레벨 {i+1}: 컨테이너{container_size}, 박스{num_boxes}개")
        
        # 복잡도에 따라 학습 시간 조정
        train_steps = PERFORMANCE_RESTORATION_CONFIG['base_timesteps'] * (1 + i * 0.3)
        train_steps = int(train_steps)
        
        env = create_environment(container_size, num_boxes, 42)
        if env:
            model, train_time = train_model(env, base_params, train_steps, f"레벨{i+1}")
            if model:
                result = evaluate_model(
                    model, container_size, num_boxes,
                    PERFORMANCE_RESTORATION_CONFIG['eval_episodes'],
                    PERFORMANCE_RESTORATION_CONFIG['max_steps_per_episode'],
                    f"레벨{i+1}"
                )
                results[level_name] = result
                if result:
                    results[level_name]['training_time'] = train_time
                    results[level_name]['config'] = {
                        'container_size': container_size,
                        'num_boxes': num_boxes,
                        'train_steps': train_steps
                    }
            env.close()
            del model
    
    # 결과 저장
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/expansion_phase2_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'phase': 'expansion',
            'target_score': 18.57,
            'results': results
        }, f, indent=2)
    
    print(f"💾 Phase 2 결과: {results_file}")
    return results

def phase_optimize():
    """Phase 3: 목표 달성 최적화"""
    print("\n" + "="*60)
    print("🎯 Phase 3: 목표 18.57점 달성 최적화")
    print("="*60)
    
    # 최적화 대상 설정
    container_size = [10, 10, 10]
    num_boxes = 12
    train_steps = PERFORMANCE_RESTORATION_CONFIG['extended_timesteps']
    
    print(f"📋 최종 목표: 컨테이너{container_size}, 박스{num_boxes}개, {train_steps:,}스텝")
    
    results = {}
    best_score = 0
    best_params = None
    
    # 하이퍼파라미터 조합 최적화
    for lr in ADVANCED_PARAM_RANGES['learning_rate']:
        for batch_size in [64, 128]:
            for net_arch in [[256, 256], [256, 128, 64]]:
                
                params = RESTORATION_OPTIMAL.copy()
                params['learning_rate'] = lr
                params['batch_size'] = batch_size
                params['net_arch'] = net_arch
                
                config_name = f"lr{lr:.1e}_b{batch_size}_net{len(net_arch)}"
                print(f"\n🔧 {config_name} 최적화 중...")
                
                env = create_environment(container_size, num_boxes, 42)
                if env:
                    model, train_time = train_model(env, params, train_steps, config_name)
                    if model:
                        result = evaluate_model(
                            model, container_size, num_boxes,
                            PERFORMANCE_RESTORATION_CONFIG['eval_episodes'],
                            PERFORMANCE_RESTORATION_CONFIG['max_steps_per_episode'],
                            config_name
                        )
                        results[config_name] = result
                        if result:
                            results[config_name]['training_time'] = train_time
                            results[config_name]['params'] = params.copy()
                            
                            if result['combined_score'] > best_score:
                                best_score = result['combined_score']
                                best_params = params.copy()
                    env.close()
                    del model
    
    # 결과 저장 및 분석
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/optimization_phase3_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'phase': 'optimization',
            'target_score': 18.57,
            'best_score': best_score,
            'best_params': best_params,
            'results': results
        }, f, indent=2)
    
    print(f"\n" + "="*60)
    print(f"🏆 Phase 3 최적화 결과")
    print(f"="*60)
    print(f"최고 성능: {best_score:.3f}점")
    if best_score >= 18.57:
        print(f"🎉 목표 달성! ({best_score:.3f} >= 18.57)")
    else:
        print(f"📈 목표 근접 (목표 18.57 대비 {((best_score-18.57)/18.57*100):+.1f}%)")
    
    print(f"💾 Phase 3 결과: {results_file}")
    return results

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='고급 성능 복원 최적화')
    parser.add_argument('--phase', choices=['restore', 'expand', 'optimize'], 
                       default='restore', help='실행할 단계')
    parser.add_argument('--timesteps', type=int, help='학습 스텝 수 (선택사항)')
    parser.add_argument('--complexity', choices=['simple', 'progressive'], 
                       default='progressive', help='복잡도 설정')
    parser.add_argument('--target', type=float, default=18.57, help='목표 점수')
    
    args = parser.parse_args()
    
    print(f"🚀 시작: Phase {args.phase}")
    print(f"📊 Python: {sys.version}")
    print(f"📁 작업 디렉토리: {os.getcwd()}")
    
    start_time = time.time()
    
    try:
        if args.phase == 'restore':
            results = phase_restore()
        elif args.phase == 'expand':
            results = phase_expand()
        elif args.phase == 'optimize':
            results = phase_optimize()
        
        total_time = time.time() - start_time
        print(f"\n⏱️ 총 소요 시간: {total_time/60:.1f}분")
        print(f"🎉 Phase {args.phase} 완료!")
        
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()