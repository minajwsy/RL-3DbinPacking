#!/usr/bin/env python3
"""
��� Optuna 최적 파라미터 검증 및 성능 평가 스크립트 (수정 버전 v2)
클라우드 프로덕션 환경에서 도출된 최적 파라미터를 적용하여 실제 성능 측정
"""

import os
import sys
import warnings
import datetime
import time
import json

# === 환경 설정 및 등록 (가장 먼저 수행) ===
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings("ignore")
sys.path.append('src')

print("��� Optuna 최적 파라미터 검증 시작")

# === 환경 등록 ===
try:
    import gymnasium as gym
    from gymnasium.envs.registration import register
    from packing_env import PackingEnv
    
    if 'PackingEnv-v0' not in gym.envs.registry:
        register(id='PackingEnv-v0', entry_point='packing_env:PackingEnv')
        print("✅ PackingEnv-v0 환경 등록 완료")
    else:
        print("✅ PackingEnv-v0 환경 이미 등록됨")
except Exception as e:
    print(f"⚠️ 환경 등록 중 오류: {e}")

# === 전역 import ===
import numpy as np
import gc
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# === 최적 파라미터 (300 trials 결과) ===
OPTIMAL_PARAMS = {
    'learning_rate': 2.6777169756959113e-06,
    'n_steps': 512,
    'batch_size': 32,
    'n_epochs': 3,
    'clip_range': 0.17716239549317803,
    'ent_coef': 0.06742268917730829,
    'vf_coef': 0.4545305173856873,
    'gae_lambda': 0.9449228658070746
}

# === 비교용 기본 파라미터 ===
DEFAULT_PARAMS = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 256,
    'n_epochs': 10,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'gae_lambda': 0.95
}

def create_env(container_size, num_boxes, num_visible_boxes=3, seed=42):
    """환경 생성 함수"""
    try:
        from utils import boxes_generator
        
        # 박스 크기 생성
        box_sizes = boxes_generator(
            bin_size=container_size,
            num_items=num_boxes,
            seed=seed
        )
        
        print(f"생성된 박스 개수: {len(box_sizes)}")
        print(f"컨테이너 크기: {container_size}")
        
        # PackingEnv-v0 환경 생성
        env = gym.make(
            "PackingEnv-v0",
            container_size=container_size,
            box_sizes=box_sizes,
            num_visible_boxes=num_visible_boxes,
            render_mode=None,
            random_boxes=False,
            only_terminal_reward=False,
        )
        
        print(f"환경 생성 성공: PackingEnv-v0")
        
        # 개선된 보상 래퍼 적용
        try:
            from train_maskable_ppo import ImprovedRewardWrapper
            env = ImprovedRewardWrapper(env)
            print(f"개선된 보상 래퍼 적용됨")
        except:
            print(f"⚠️ ImprovedRewardWrapper 없음 - 기본 환경 사용")
        
        # 액션 마스킹 래퍼 적용
        from sb3_contrib.common.wrappers import ActionMasker
        
        def get_action_masks_local(env):
            """로컬 액션 마스크 함수"""
            try:
                from train_maskable_ppo import get_action_masks
                return get_action_masks(env)
            except:
                # 기본 액션 마스크 (모든 액션 허용)
                return np.ones(env.action_space.n, dtype=bool)
        
        env = ActionMasker(env, get_action_masks_local)
        print(f"액션 마스킹 래퍼 적용됨")
        
        # 시드 설정
        env.reset(seed=seed)
        print(f"시드 설정 완료: {seed}")
        
        return env
        
    except Exception as e:
        print(f"❌ 환경 생성 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_optimal_parameters():
    """최적 파라미터로 실제 PPO 학습 및 검증"""
    try:
        print("��� 모듈 로딩 중...")
        print("✅ 강화학습 모듈 로드")
        
        # 메모리 최적화
        gc.collect()
        
        # 실험 설정
        container_size = [10, 10, 10]
        num_boxes = 16
        train_timesteps = 10000
        
        print(f"��� 실험 설정:")
        print(f"   - 컨테이너 크기: {container_size}")
        print(f"   - 박스 개수: {num_boxes}")
        print(f"   - 학습 스텝: {train_timesteps:,}")
        
        # === 최적 파라미터로 학습 ===
        print("\n��� === 최적 파라미터 학습 시작 ===")
        optimal_results = train_and_evaluate(
            params=OPTIMAL_PARAMS,
            container_size=container_size,
            num_boxes=num_boxes,
            train_timesteps=train_timesteps,
            experiment_name="optimal"
        )
        
        # === 기본 파라미터로 학습 (비교용) ===
        print("\n��� === 기본 파라미터 학습 시작 (비교용) ===")
        default_results = train_and_evaluate(
            params=DEFAULT_PARAMS,
            container_size=container_size,
            num_boxes=num_boxes,
            train_timesteps=train_timesteps,
            experiment_name="default"
        )
        
        # === 결과 비교 및 분석 ===
        print("\n��� === 성능 비교 분석 ===")
        compare_results(optimal_results, default_results)
        
        # === 상세 결과 저장 ===
        save_validation_results(optimal_results, default_results)
        
        return True
        
    except Exception as e:
        print(f"❌ 검증 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_and_evaluate(params, container_size, num_boxes, train_timesteps, experiment_name):
    """파라미터를 사용한 학습 및 평가"""
    print(f"��� {experiment_name} 실험 시작")
    print(f"파라미터: {json.dumps(params, indent=2)}")
    
    # 환경 생성
    env = create_env(container_size, num_boxes, seed=42)
    eval_env = create_env(container_size, num_boxes, seed=43)
    
    if env is None or eval_env is None:
        raise ValueError("환경 생성 실패")
    
    # 모니터링 설정
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('logs', exist_ok=True)
    env = Monitor(env, f"logs/validate_{experiment_name}_train_{timestamp}.csv")
    eval_env = Monitor(eval_env, f"logs/validate_{experiment_name}_eval_{timestamp}.csv")
    
    # 진행상황 콜백
    class ProgressCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            
        def _on_step(self) -> bool:
            if self.num_timesteps % 1000 == 0:
                progress = (self.num_timesteps / train_timesteps) * 100
                print(f"   진행률: {progress:.1f}% ({self.num_timesteps:,}/{train_timesteps:,})")
            return True
    
    # PPO 모델 생성
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
            net_arch=[256, 256, 128],
            activation_fn="relu",
        )
    )
    
    # 학습 시작
    start_time = time.time()
    print(f"��� 학습 시작: {train_timesteps:,} 스텝")
    
    callback = ProgressCallback()
    model.learn(total_timesteps=train_timesteps, callback=callback, progress_bar=False)
    
    training_time = time.time() - start_time
    print(f"⏱️ 학습 완료: {training_time:.1f}초")
    
    # 모델 저장
    os.makedirs('models', exist_ok=True)
    model_path = f"models/validate_{experiment_name}_{timestamp}"
    model.save(model_path)
    print(f"��� 모델 저장: {model_path}")
    
    # === 상세 평가 수행 ===
    print("��� 상세 평가 시작...")
    evaluation_results = detailed_evaluation(model, eval_env, experiment_name)
    
    # 환경 정리
    env.close()
    eval_env.close()
    
    # 결과 정리
    results = {
        'experiment_name': experiment_name,
        'params': params,
        'training_time': training_time,
        'model_path': model_path,
        'timestamp': timestamp,
        **evaluation_results
    }
    
    # 메모리 정리
    del model
    gc.collect()
    
    return results

def detailed_evaluation(model, eval_env, experiment_name, n_episodes=10):
    """상세한 모델 평가"""
    print(f"��� {experiment_name} 상세 평가 ({n_episodes} 에피소드)")
    
    episode_rewards = []
    episode_lengths = []
    episode_utilizations = []
    episode_success_rates = []
    
    for ep in range(n_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done and episode_length < 100:
            # 액션 마스크 생성
            try:
                action_masks = np.ones(eval_env.action_space.n, dtype=bool)
            except:
                action_masks = None
                
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 활용률 계산
        try:
            if hasattr(eval_env.unwrapped, 'container'):
                placed_volume = sum(box.volume for box in eval_env.unwrapped.container.boxes 
                                  if box.position is not None)
                container_volume = eval_env.unwrapped.container.volume
                utilization = placed_volume / container_volume if container_volume > 0 else 0.0
                episode_utilizations.append(utilization)
                
                success = 1.0 if utilization >= 0.5 else 0.0
                episode_success_rates.append(success)
            else:
                episode_utilizations.append(0.0)
                episode_success_rates.append(0.0)
        except:
            episode_utilizations.append(0.0)
            episode_success_rates.append(0.0)
        
        if (ep + 1) % 5 == 0:
            print(f"   에피소드 {ep+1}/{n_episodes} 완료")
    
    # 통계 계산
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_utilization': np.mean(episode_utilizations),
        'std_utilization': np.std(episode_utilizations),
        'success_rate': np.mean(episode_success_rates),
        'combined_score': np.mean(episode_rewards) * 0.3 + np.mean(episode_utilizations) * 100 * 0.7,
        'episode_rewards': episode_rewards,
        'episode_utilizations': episode_utilizations
    }
    
    print(f"✅ {experiment_name} 평가 완료:")
    print(f"   평균 보상: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"   평균 활용률: {results['mean_utilization']:.1%} ± {results['std_utilization']:.1%}")
    print(f"   성공률: {results['success_rate']:.1%}")
    print(f"   종합 점수: {results['combined_score']:.4f}")
    
    return results

def compare_results(optimal_results, default_results):
    """최적 파라미터와 기본 파라미터 결과 비교"""
    print("��� === 상세 성능 비교 ===")
    
    metrics = [
        ('평균 보상', 'mean_reward', '.4f'),
        ('평균 활용률', 'mean_utilization', '.1%'),
        ('성공률', 'success_rate', '.1%'),
        ('종합 점수', 'combined_score', '.4f'),
        ('학습 시간', 'training_time', '.1f초')
    ]
    
    print(f"{'지표':<12} {'최적 파라미터':<15} {'기본 파라미터':<15} {'개선율':<10}")
    print("-" * 60)
    
    for metric_name, key, fmt in metrics:
        optimal_val = optimal_results[key]
        default_val = default_results[key]
        
        if key == 'training_time':
            improvement = f"{((default_val - optimal_val) / default_val * 100):+.1f}%"
        else:
            improvement = f"{((optimal_val - default_val) / default_val * 100):+.1f}%"
        
        print(f"{metric_name:<12} {optimal_val:{fmt}:<15} {default_val:{fmt}:<15} {improvement:<10}")
    
    # 종합 평가
    combined_improvement = ((optimal_results['combined_score'] - default_results['combined_score']) / 
                           default_results['combined_score'] * 100)
    
    print(f"\n��� 종합 평가:")
    if combined_improvement > 10:
        print(f"��� 최적 파라미터가 {combined_improvement:.1f}% 우수한 성능을 보입니다!")
    elif combined_improvement > 0:
        print(f"✅ 최적 파라미터가 {combined_improvement:.1f}% 개선된 성능을 보입니다.")
    else:
        print(f"⚠️ 최적 파라미터 성능이 {abs(combined_improvement):.1f}% 낮습니다.")

def save_validation_results(optimal_results, default_results):
    """검증 결과를 JSON 파일로 저장"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    results_file = f"results/validation_results_{timestamp}.json"
    
    validation_data = {
        'validation_timestamp': timestamp,
        'optimal_params': OPTIMAL_PARAMS,
        'default_params': DEFAULT_PARAMS,
        'optimal_results': optimal_results,
        'default_results': default_results,
        'improvement': {
            'combined_score': ((optimal_results['combined_score'] - default_results['combined_score']) / 
                              default_results['combined_score'] * 100),
            'mean_reward': ((optimal_results['mean_reward'] - default_results['mean_reward']) / 
                           default_results['mean_reward'] * 100),
            'mean_utilization': ((optimal_results['mean_utilization'] - default_results['mean_utilization']) / 
                                default_results['mean_utilization'] * 100)
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(validation_data, f, indent=2, default=str)
    
    print(f"��� 검증 결과 저장: {results_file}")
    
    # 간단한 요약 텍스트 파일도 생성
    summary_file = f"results/validation_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("=== Optuna 최적 파라미터 검증 결과 ===\n\n")
        f.write(f"검증 시간: {timestamp}\n")
        f.write(f"실험 설정: 컨테이너 [10,10,10], 박스 16개\n\n")
        
        f.write("최적 파라미터 성능:\n")
        f.write(f"  - 평균 보상: {optimal_results['mean_reward']:.4f}\n")
        f.write(f"  - 평균 활용률: {optimal_results['mean_utilization']:.1%}\n")
        f.write(f"  - 성공률: {optimal_results['success_rate']:.1%}\n")
        f.write(f"  - 종합 점수: {optimal_results['combined_score']:.4f}\n\n")
        
        f.write("기본 파라미터 성능:\n")
        f.write(f"  - 평균 보상: {default_results['mean_reward']:.4f}\n")
        f.write(f"  - 평균 활용률: {default_results['mean_utilization']:.1%}\n")  
        f.write(f"  - 성공률: {default_results['success_rate']:.1%}\n")
        f.write(f"  - 종합 점수: {default_results['combined_score']:.4f}\n\n")
        
        improvement = validation_data['improvement']['combined_score']
        f.write(f"종합 개선율: {improvement:.1f}%\n")
    
    print(f"��� 요약 저장: {summary_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Optuna 최적 파라미터 검증")
    parser.add_argument("--validate", action="store_true", help="최적 파라미터 검증 실행")
    parser.add_argument("--timesteps", type=int, default=10000, help="학습 스텝 수")
    
    try:
        args = parser.parse_args()
        
        if args.validate:
            print("��� Optuna 최적 파라미터 검증 모드")
            print(f"학습 스텝: {args.timesteps:,}")
            
            success = validate_optimal_parameters()
            if success:
                print("\n✅ 검증 완료! results/ 폴더에서 상세 결과를 확인하세요.")
            else:
                print("\n❌ 검증 실패!")
        else:
            print("��� 사용법:")
            print("  python validate_optimal_params_fixed.py --validate")
            print("  python validate_optimal_params_fixed.py --validate --timesteps 15000")
            print("\n��� 최적 파라미터 (300 trials):")
            for key, value in OPTIMAL_PARAMS.items():
                if 'learning_rate' in key:
                    print(f"  {key}: {value:.6e}")
                else:
                    print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ 메인 오류: {e}")

if __name__ == "__main__":
    main()
