#!/usr/bin/env python3
"""
🚀 클라우드 환경용 실제 PPO 학습 + Optuna 하이퍼파라미터 최적화
터미널 크래시 문제 해결 후 실제 학습 수행
"""

import os
import sys
import warnings
import datetime
import time

# 클라우드 환경 최적화 설정
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings("ignore")

# 메모리 최적화
import gc
gc.collect()

print("🚀 클라우드 프로덕션 3D Bin Packing 학습 시작")

def production_train_with_optuna():
    """실제 PPO 학습을 포함한 Optuna 최적화"""
    try:
        # 필수 모듈 순차 로딩
        print("📦 모듈 로딩 중...")
        
        import numpy as np
        import optuna
        import torch
        print("✅ 기본 모듈 로드")
        
        # 환경 경로 설정
        sys.path.append('src')
        
        # 로컬 모듈
        from train_maskable_ppo import make_env
        from packing_kernel import Container, Box
        print("✅ 로컬 모듈 로드")
        
        # 강화학습 모듈
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks
        from stable_baselines3.common.monitor import Monitor
        print("✅ 강화학습 모듈 로드")
        
        # 메모리 정리
        gc.collect()
        
        # Optuna 스터디 생성
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3,
                n_warmup_steps=500,
                interval_steps=200
            )
        )
        
        print("✅ Optuna 스터디 생성 완료")
        
        def objective(trial):
            """실제 PPO 학습을 포함한 목적 함수"""
            try:
                # 하이퍼파라미터 제안 (최적화된 범위)
                learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
                n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096])
                batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
                n_epochs = trial.suggest_int('n_epochs', 3, 15)
                clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
                ent_coef = trial.suggest_float('ent_coef', 1e-4, 1e-1, log=True)
                vf_coef = trial.suggest_float('vf_coef', 0.1, 1)
                gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
                
                print(f"🔬 Trial {trial.number}: lr={learning_rate:.6f}, steps={n_steps}, batch={batch_size}")
                
                # 환경 생성 (점진적으로 크기 증가)
                container_size = [10, 10, 10]  # 실제 크기
                num_boxes = 12  # 적당한 복잡성
                
                env = make_env(
                    container_size=container_size,
                    num_boxes=num_boxes,
                    num_visible_boxes=3,
                    seed=42 + trial.number,
                    render_mode=None,
                    random_boxes=False,
                    only_terminal_reward=False,
                    improved_reward_shaping=True,
                )()
                
                eval_env = make_env(
                    container_size=container_size,
                    num_boxes=num_boxes,
                    num_visible_boxes=3,
                    seed=43 + trial.number,
                    render_mode=None,
                    random_boxes=False,
                    only_terminal_reward=False,
                    improved_reward_shaping=True,
                )()
                
                # 모니터링
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                env = Monitor(env, f"logs/prod_train_trial_{trial.number}_{timestamp}.csv")
                eval_env = Monitor(eval_env, f"logs/prod_eval_trial_{trial.number}_{timestamp}.csv")
                
                # PPO 모델 생성 (수정됨)
                model = MaskablePPO(
                    "MultiInputPolicy",
                    env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    gae_lambda=gae_lambda,
                    gamma=0.99,
                    max_grad_norm=0.5,
                    verbose=0,
                    seed=42 + trial.number,
                    policy_kwargs=dict(
                        net_arch=[256, 256, 128],
                        activation_fn=torch.nn.ReLU,  # 수정: 문자열 대신 torch 객체
                        share_features_extractor=True,
                    )
                )
                
                # 실제 학습 수행 (짧은 시간)
                train_timesteps = 3000  # 클라우드 환경에 맞춘 적당한 길이
                start_time = time.time()
                
                print(f"   🎓 학습 시작: {train_timesteps} 스텝")
                model.learn(total_timesteps=train_timesteps, progress_bar=False)
                
                training_time = time.time() - start_time
                print(f"   ⏱️ 학습 완료: {training_time:.1f}초")
                
                # 모델 평가 (수정된 get_action_masks 사용법)
                total_rewards = []
                total_utilizations = []
                
                for ep in range(3):  # 3 에피소드 평가
                    obs, _ = eval_env.reset()
                    episode_reward = 0.0
                    done = False
                    step_count = 0
                    
                    while not done and step_count < 50:
                        # 수정: get_action_masks를 환경에서 직접 호출
                        if hasattr(eval_env, 'action_masks'):
                            action_masks = eval_env.action_masks()
                        else:
                            # 기본 마스크 (모든 액션 허용)
                            action_masks = np.ones(eval_env.action_space.n, dtype=bool)
                        
                        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                        obs, reward, terminated, truncated, info = eval_env.step(action)
                        episode_reward += reward
                        done = terminated or truncated
                        step_count += 1
                    
                    total_rewards.append(episode_reward)
                    
                    # 활용률 계산
                    if hasattr(eval_env.unwrapped, 'container'):
                        placed_volume = sum(box.volume for box in eval_env.unwrapped.container.boxes 
                                            if box.position is not None)
                        container_volume = eval_env.unwrapped.container.volume
                        utilization = placed_volume / container_volume if container_volume > 0 else 0.0
                        total_utilizations.append(utilization)
                    else:
                        total_utilizations.append(0.0)
                
                # 성능 지표 계산
                mean_reward = np.mean(total_rewards)
                mean_utilization = np.mean(total_utilizations)
                
                # 다중 목적 최적화: 공간 활용률 70% + 평균 보상 30%
                combined_score = mean_reward * 0.3 + mean_utilization * 100 * 0.7
                
                print(f"   📊 평균 보상: {mean_reward:.4f}")
                print(f"   📦 평균 활용률: {mean_utilization:.1%}")
                print(f"   🎯 종합 점수: {combined_score:.4f}")
                
                # 환경 정리
                env.close()
                eval_env.close()
                del model  # 메모리 해제
                gc.collect()
                
                return combined_score
                
            except Exception as e:
                print(f"❌ Trial {trial.number} 오류: {e}")
                import traceback
                traceback.print_exc()
                return 0.0
        
        # 최적화 실행
        n_trials = 10  # 실제 학습을 위한 적당한 trial 수
        print(f"🚀 프로덕션 최적화 시작 ({n_trials} trials)")
        
        study.optimize(objective, n_trials=n_trials)
        
        # 결과 출력
        print("\n🏆 프로덕션 최적화 완료!")
        print(f"최적 값: {study.best_value:.4f}")
        print(f"최적 파라미터:")
        for key, value in study.best_params.items():
            if 'learning_rate' in key:
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        # 결과 저장
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/production_optuna_results_{timestamp}.txt"
        os.makedirs('results', exist_ok=True)
        
        with open(results_file, 'w') as f:
            f.write("=== 클라우드 프로덕션 Optuna 최적화 결과 ===\n")
            f.write(f"실행 시간: {timestamp}\n")
            f.write(f"Trial 수: {n_trials}\n")
            f.write(f"최적 값: {study.best_value:.4f}\n")
            f.write(f"최적 파라미터:\n")
            for key, value in study.best_params.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\n모든 Trial 결과:\n")
            for trial in study.trials:
                f.write(f"Trial {trial.number}: {trial.value:.4f} - {trial.params}\n")
        
        print(f"📄 결과 저장: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 프로덕션 최적화 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="클라우드 프로덕션 3D Bin Packing")
    parser.add_argument("--optimize", action="store_true", help="프로덕션 Optuna 최적화")
    parser.add_argument("--trials", type=int, default=10, help="Trial 수")
    
    try:
        args = parser.parse_args()
        
        if args.optimize:
            print(f"🔬 프로덕션 Optuna 최적화 모드 ({args.trials} trials)")
            success = production_train_with_optuna()
            if success:
                print("✅ 프로덕션 최적화 성공!")
            else:
                print("❌ 프로덕션 최적화 실패!")
        else:
            print("💡 사용법:")
            print("  python production_train_cloud.py --optimize")
            print("  python production_train_cloud.py --optimize --trials 20")
            
    except Exception as e:
        print(f"❌ 메인 오류: {e}")

if __name__ == "__main__":
    main() 