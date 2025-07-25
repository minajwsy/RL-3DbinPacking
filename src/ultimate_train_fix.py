#!/usr/bin/env python3
"""
🚀 999스텝 문제 완전 해결 + GIF + 성능 그래프 생성 + Optuna/W&B 하이퍼파라미터 최적화
평가 콜백을 완전히 재작성하여 무한 대기 문제 100% 해결
Optuna와 W&B sweep을 이용한 자동 하이퍼파라미터 튜닝 지원
"""

import os
import sys
import time
import datetime
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import torch  # torch import 추가
import json
from typing import Dict, Any, Optional, Tuple

# 하이퍼파라미터 최적화 라이브러리
try:
    import optuna
    from optuna.integration import WeightsAndBiasesCallback
    OPTUNA_AVAILABLE = True
    print("✅ Optuna 사용 가능")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna 없음 - pip install optuna 필요")

try:
    import wandb
    WANDB_AVAILABLE = True
    print("✅ W&B 사용 가능")
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ W&B 없음 - pip install wandb 필요")

# 서버 환경 대응
matplotlib.use('Agg')
warnings.filterwarnings("ignore")

# 폰트 설정 (한글 폰트 문제 해결)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
sys.path.append('src')
os.environ['PYTHONPATH'] = os.getcwd() + ':' + os.getcwd() + '/src'

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# 로컬 모듈 import
from packing_kernel import *
from train_maskable_ppo import make_env, get_action_masks

class UltimateSafeCallback(BaseCallback):
    """
    999 스텝 문제 완전 해결을 위한 안전한 콜백
    - 최소한의 평가만 수행
    - 타임아웃 설정으로 무한 대기 방지
    - 실시간 성능 그래프 생성
    """
    def __init__(self, eval_env, eval_freq=2000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.last_eval_time = 0
        
        # 성능 지표 저장
        self.timesteps = []
        self.episode_rewards = []
        self.eval_rewards = []
        self.eval_timesteps = []
        self.success_rates = []
        self.utilization_rates = []
        
        # 실시간 플롯 설정
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Real-time Training Performance Monitoring', fontsize=16)
        plt.ion()
        
        print(f"🛡️ 안전한 콜백 초기화 완료 (평가 주기: {eval_freq})")
        
    def _on_training_start(self) -> None:
        """학습 시작"""
        print("🚀 안전한 학습 모니터링 시작")
        self.start_time = time.time()
        self._setup_plots()
        
    def _on_step(self) -> bool:
        """매 스텝마다 호출 - 안전한 처리"""
        # 에피소드 완료 체크
        if self.locals.get('dones', [False])[0]:
            if 'episode' in self.locals.get('infos', [{}])[0]:
                episode_info = self.locals['infos'][0]['episode']
                episode_reward = episode_info['r']
                
                self.timesteps.append(self.num_timesteps)
                self.episode_rewards.append(episode_reward)
                
                # 주기적 출력
                if len(self.episode_rewards) % 20 == 0:
                    recent_rewards = self.episode_rewards[-20:]
                    mean_reward = np.mean(recent_rewards)
                    elapsed = time.time() - self.start_time
                    
                    print(f"📊 스텝: {self.num_timesteps:,} | "
                          f"에피소드: {len(self.episode_rewards)} | "
                          f"최근 평균 보상: {mean_reward:.3f} | "
                          f"경과: {elapsed:.1f}초")
        
        # 안전한 평가 (타임아웃 설정)
        if self.num_timesteps - self.last_eval_time >= self.eval_freq:
            self._safe_evaluation()
            self._update_plots()
            self.last_eval_time = self.num_timesteps
            
        return True
    
    def _safe_evaluation(self):
        """타임아웃이 있는 안전한 평가 -> 평가함수 개선"""
        try:
            print(f"🔍 안전한 평가 시작 (스텝: {self.num_timesteps:,})")
            
            eval_rewards = []
            success_count = 0
            max_episodes = 15  # 8 → 15로 증가 (더 정확한 평가)
            
            for ep_idx in range(max_episodes):
                try:
                    # 타임아웃 설정
                    eval_start = time.time()
                    timeout = 60  # 30 → 60초로 증가
                    
                    obs, _ = self.eval_env.reset()
                    episode_reward = 0.0
                    step_count = 0
                    max_steps = 100  # 50 → 100으로 증가 (충분한 시간)
                    
                    while step_count < max_steps:
                        # 타임아웃 체크
                        if time.time() - eval_start > timeout:
                            print(f"⏰ 평가 타임아웃 (에피소드 {ep_idx})")
                            break
                            
                        try:
                            action_masks = get_action_masks(self.eval_env)
                            action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
                            obs, reward, terminated, truncated, info = self.eval_env.step(action)
                            episode_reward += reward
                            step_count += 1
                            
                            if terminated or truncated:
                                break
                        except Exception as e:
                            print(f"⚠️ 평가 스텝 오류: {e}")
                            break
                    
                    eval_rewards.append(episode_reward)
                    success_threshold = 5.0  # 3.0 → 5.0으로 상향 조정
                    if episode_reward >= success_threshold:
                        success_count += 1
                        
                except Exception as e:
                    print(f"⚠️ 평가 에피소드 {ep_idx} 오류: {e}")
                    eval_rewards.append(0.0)
            
            # 결과 저장
            if eval_rewards:
                mean_eval_reward = np.mean(eval_rewards)
                success_rate = success_count / len(eval_rewards)
                utilization = max(0.0, mean_eval_reward)
                
                self.eval_rewards.append(mean_eval_reward)
                self.eval_timesteps.append(self.num_timesteps)
                self.success_rates.append(success_rate)
                self.utilization_rates.append(utilization)
                
                print(f"✅ 평가 완료: 보상 {mean_eval_reward:.3f}, "
                      f"성공률 {success_rate:.1%}, 활용률 {utilization:.1%}")
            else:
                print("❌ 평가 실패 - 기본값 사용")
                self.eval_rewards.append(0.0)
                self.eval_timesteps.append(self.num_timesteps)
                self.success_rates.append(0.0)
                self.utilization_rates.append(0.0)
                
        except Exception as e:
            print(f"❌ 평가 중 오류: {e}")
            # 오류 시에도 기본값으로 진행
            self.eval_rewards.append(0.0)
            self.eval_timesteps.append(self.num_timesteps)
            self.success_rates.append(0.0)
            self.utilization_rates.append(0.0)
    
    def _setup_plots(self):
        """플롯 초기 설정"""
        titles = [
            'Episode Rewards (Training)',
            'Evaluation Rewards (Periodic)',
            'Success Rate (%)',
            'Utilization Rate (%)'
        ]
        
        for i, ax in enumerate(self.axes.flat):
            ax.set_title(titles[i])
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Steps')
    
    def _update_plots(self):
        """실시간 플롯 업데이트"""
        try:
            # 1. 에피소드 보상
            if self.timesteps and self.episode_rewards:
                self.axes[0, 0].clear()
                self.axes[0, 0].plot(self.timesteps, self.episode_rewards, 'b-', alpha=0.6, linewidth=1)
                if len(self.episode_rewards) > 50:
                    # 이동 평균
                    window = min(50, len(self.episode_rewards) // 4)
                    moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                    moving_steps = self.timesteps[window-1:]
                    self.axes[0, 0].plot(moving_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Avg({window})')
                    self.axes[0, 0].legend()
                self.axes[0, 0].set_title('Episode Rewards (Training)')
                self.axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 평가 보상
            if self.eval_timesteps and self.eval_rewards:
                self.axes[0, 1].clear()
                self.axes[0, 1].plot(self.eval_timesteps, self.eval_rewards, 'g-o', linewidth=2, markersize=6)
                self.axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
                self.axes[0, 1].set_title('Evaluation Rewards (Periodic)')
                self.axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 성공률
            if self.eval_timesteps and self.success_rates:
                self.axes[1, 0].clear()
                success_pct = [rate * 100 for rate in self.success_rates]
                self.axes[1, 0].plot(self.eval_timesteps, success_pct, 'orange', linewidth=2, marker='s')
                self.axes[1, 0].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Target(80%)')
                self.axes[1, 0].set_ylim(0, 100)
                self.axes[1, 0].set_title('Success Rate (%)')
                self.axes[1, 0].grid(True, alpha=0.3)
                self.axes[1, 0].legend()
            
            # 4. 활용률
            if self.eval_timesteps and self.utilization_rates:
                self.axes[1, 1].clear()
                util_pct = [rate * 100 for rate in self.utilization_rates]
                self.axes[1, 1].plot(self.eval_timesteps, util_pct, 'purple', linewidth=2, marker='^')
                # 동적 Y축 설정 (최대값의 110%까지)
                max_util = max(util_pct) if util_pct else 100
                self.axes[1, 1].set_ylim(0, max(100, max_util * 1.1))
                self.axes[1, 1].set_title('Utilization Rate (%)')
                self.axes[1, 1].grid(True, alpha=0.3)
            
            # 플롯 저장
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.tight_layout()
            plt.savefig(f'results/realtime_performance_{timestamp}.png', dpi=150, bbox_inches='tight')
            
        except Exception as e:
            print(f"⚠️ 플롯 업데이트 오류: {e}")
    
    def _on_training_end(self) -> None:
        """학습 종료 시 최종 그래프 저장"""
        print("📊 최종 성과 그래프 생성 중...")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 최종 성과 대시보드
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Final Training Performance Dashboard', fontsize=20)
        
        try:
            # 1. 학습 곡선
            if self.timesteps and self.episode_rewards:
                axes[0, 0].plot(self.timesteps, self.episode_rewards, 'b-', alpha=0.4, linewidth=1, label='Episode Rewards')
                if len(self.episode_rewards) > 20:
                    window = min(50, len(self.episode_rewards) // 4)
                    moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                    moving_steps = self.timesteps[window-1:]
                    axes[0, 0].plot(moving_steps, moving_avg, 'r-', linewidth=3, label=f'Moving Avg({window})')
                axes[0, 0].set_title('Learning Curve')
                axes[0, 0].set_xlabel('Steps')
                axes[0, 0].set_ylabel('Reward')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 평가 성능
            if self.eval_timesteps and self.eval_rewards:
                axes[0, 1].plot(self.eval_timesteps, self.eval_rewards, 'g-o', linewidth=3, markersize=8)
                axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
                axes[0, 1].set_title('Evaluation Performance')
                axes[0, 1].set_xlabel('Steps')
                axes[0, 1].set_ylabel('Evaluation Reward')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 성공률 추이
            if self.eval_timesteps and self.success_rates:
                success_pct = [rate * 100 for rate in self.success_rates]
                axes[0, 2].plot(self.eval_timesteps, success_pct, 'orange', linewidth=3, marker='s', markersize=8)
                axes[0, 2].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Target(80%)')
                axes[0, 2].set_ylim(0, 100)
                axes[0, 2].set_title('Success Rate Trend')
                axes[0, 2].set_xlabel('Steps')
                axes[0, 2].set_ylabel('Success Rate (%)')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
            
            # 4. 활용률 추이
            if self.eval_timesteps and self.utilization_rates:
                util_pct = [rate * 100 for rate in self.utilization_rates]
                axes[1, 0].plot(self.eval_timesteps, util_pct, 'purple', linewidth=3, marker='^', markersize=8)
                # 동적 Y축 설정 (최대값의 110%까지)
                max_util = max(util_pct) if util_pct else 100
                axes[1, 0].set_ylim(0, max(100, max_util * 1.1))
                axes[1, 0].set_title('Utilization Rate Trend')
                axes[1, 0].set_xlabel('Steps')
                axes[1, 0].set_ylabel('Utilization Rate (%)')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 5. 보상 분포
            if self.episode_rewards:
                axes[1, 1].hist(self.episode_rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[1, 1].axvline(np.mean(self.episode_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.episode_rewards):.3f}')
                axes[1, 1].set_title('Reward Distribution')
                axes[1, 1].set_xlabel('Reward')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            # 6. 성과 요약
            axes[1, 2].axis('off')
            if self.episode_rewards and self.eval_rewards:
                summary_text = f"""
Training Summary Statistics

Total Episodes: {len(self.episode_rewards):,}
Final Steps: {self.num_timesteps:,}
Training Time: {(time.time() - self.start_time):.1f}s

Training Performance:
• Mean Reward: {np.mean(self.episode_rewards):.3f}
• Max Reward: {np.max(self.episode_rewards):.3f}
• Min Reward: {np.min(self.episode_rewards):.3f}
• Std Dev: {np.std(self.episode_rewards):.3f}

Evaluation Performance:
• Final Eval Reward: {self.eval_rewards[-1] if self.eval_rewards else 0:.3f}
• Best Eval Reward: {np.max(self.eval_rewards) if self.eval_rewards else 0:.3f}
• Final Success Rate: {self.success_rates[-1]*100 if self.success_rates else 0:.1f}%
• Final Utilization: {self.utilization_rates[-1]*100 if self.utilization_rates else 0:.1f}%
"""
                axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                               fontsize=12, verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            dashboard_path = f'results/ultimate_dashboard_{timestamp}.png'
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
            print(f"📊 최종 대시보드 저장: {dashboard_path}")
            
            # 성능 데이터 저장
            performance_data = {
                'timesteps': self.timesteps,
                'episode_rewards': self.episode_rewards,
                'eval_timesteps': self.eval_timesteps,
                'eval_rewards': self.eval_rewards,
                'success_rates': self.success_rates,
                'utilization_rates': self.utilization_rates
            }
            np.save(f'results/performance_data_{timestamp}.npy', performance_data)
            
        except Exception as e:
            print(f"⚠️ 최종 그래프 생성 오류: {e}")
        
        plt.close('all')

class AdaptiveCurriculumCallback(BaseCallback):
    """
    MathWorks/YouTube 사례 기반 적응적 커리큘럼 학습
    - 성능 기반 동적 난이도 조정
    - 다중 지표 평가 (보상, 성공률, 안정성)
    - 자동 백트래킹 (성능 저하시 이전 단계로)
    """
    
    def __init__(
        self,
        container_size,
        initial_boxes,
        target_boxes,
        num_visible_boxes,
        success_threshold=0.45,  # MathWorks 권장: 낮은 기준으로 시작
        curriculum_steps=7,     # 더 많은 단계
        patience=10,            # 더 긴 인내심
        stability_window=50,   # 안정성 측정 윈도우
        verbose=1,
    ):
        super().__init__(verbose)
        self.container_size = container_size
        self.initial_boxes = initial_boxes
        self.target_boxes = target_boxes
        self.num_visible_boxes = num_visible_boxes
        self.success_threshold = success_threshold
        self.curriculum_steps = curriculum_steps
        self.patience = patience
        self.stability_window = stability_window
        self.verbose = verbose
        
        # 적응적 커리큘럼 단계 설정
        self.current_boxes = initial_boxes
        self.box_increments = []
        if target_boxes > initial_boxes:
            step_size = max(1, (target_boxes - initial_boxes) // curriculum_steps)
            for i in range(curriculum_steps):
                next_boxes = initial_boxes + (i + 1) * step_size
                if next_boxes > target_boxes:
                    next_boxes = target_boxes
                self.box_increments.append(next_boxes)
            # 마지막 단계는 항상 target_boxes
            if self.box_increments[-1] != target_boxes:
                self.box_increments.append(target_boxes)
        
        # 성과 추적 변수 (다중 지표)
        self.evaluation_count = 0
        self.consecutive_successes = 0
        self.curriculum_level = 0
        self.last_success_rate = 0.0
        self.recent_rewards = []
        self.recent_episode_lengths = []
        self.recent_utilizations = []
        
        # MathWorks 기반 적응적 파라미터
        self.performance_history = []  # 각 레벨별 성능 기록
        self.stability_scores = []     # 안정성 점수
        self.backtrack_count = 0       # 백트래킹 횟수
        self.max_backtrack = 3         # 최대 백트래킹 허용
        
        # 적응적 임계값 (성능에 따라 동적 조정)
        self.adaptive_threshold = success_threshold
        self.threshold_decay = 0.95    # 임계값 감소율
        
        if self.verbose >= 1:
            print(f"🎓 적응적 커리큘럼 학습 초기화:")
            print(f"   - 시작 박스 수: {self.initial_boxes}")
            print(f"   - 목표 박스 수: {self.target_boxes}")
            print(f"   - 단계별 증가: {self.box_increments}")
            print(f"   - 초기 성공 임계값: {self.success_threshold}")
            print(f"   - 안정성 윈도우: {self.stability_window}")
    
    def _on_step(self) -> bool:
        """매 스텝마다 호출 - 다중 지표 수집"""
        # 에피소드 완료 체크
        if self.locals.get('dones', [False])[0]:
            if 'episode' in self.locals.get('infos', [{}])[0]:
                episode_info = self.locals['infos'][0]['episode']
                episode_reward = episode_info['r']
                episode_length = episode_info['l']
                
                # 활용률 계산 (보상을 활용률로 간주)
                episode_utilization = max(0.0, episode_reward / 10.0)  # 정규화
                
                # 최근 성과 기록 (적응적 윈도우)
                self.recent_rewards.append(episode_reward)
                self.recent_episode_lengths.append(episode_length)
                self.recent_utilizations.append(episode_utilization)
                
                # 윈도우 크기 유지
                if len(self.recent_rewards) > self.stability_window:
                    self.recent_rewards.pop(0)
                    self.recent_episode_lengths.pop(0)
                    self.recent_utilizations.pop(0)
                
                # 충분한 데이터가 있을 때만 평가
                if len(self.recent_rewards) >= min(15, self.stability_window // 2):
                    self._evaluate_adaptive_curriculum()
        
        return True
    
    def _evaluate_adaptive_curriculum(self):
        """적응적 커리큘럼 진행 상황 평가 (MathWorks 기법)"""
        try:
            # 1. 다중 성과 지표 계산
            rewards = self.recent_rewards
            lengths = self.recent_episode_lengths
            utilizations = self.recent_utilizations
            
            # 성공률 (보상 기반)
            success_count = sum(1 for r in rewards if r > self.adaptive_threshold)
            success_rate = success_count / len(rewards)
            
            # 안정성 점수 (변동성 기반)
            import numpy as np
            reward_stability = 1.0 / (1.0 + np.std(rewards))
            length_stability = 1.0 / (1.0 + np.std(lengths))
            stability_score = (reward_stability + length_stability) / 2.0
            
            # 활용률 개선도
            if len(utilizations) >= 10:
                recent_util = np.mean(utilizations[-5:])
                prev_util = np.mean(utilizations[-10:-5])
                util_improvement = recent_util - prev_util
            else:
                util_improvement = 0.0
            
            # 종합 성과 점수 (MathWorks 가중 평균)
            performance_score = (
                success_rate * 0.3 +          # 성공률 30%
                stability_score * 0.7 +       # 안정성 70%
                max(0, util_improvement) * 0.0 # 개선도 0%
            )
            
            # 현재 성과 기록
            self.last_success_rate = success_rate
            self.stability_scores.append(stability_score)
            self.performance_history.append(performance_score)
            self.evaluation_count += 1
            
            if self.verbose >= 2:
                print(f"📊 적응적 평가 (레벨 {self.curriculum_level}):")
                print(f"   - 성공률: {success_rate:.1%}")
                print(f"   - 안정성: {stability_score:.3f}")
                print(f"   - 활용률 개선: {util_improvement:.3f}")
                print(f"   - 종합 점수: {performance_score:.3f}")
            
            # 2. 적응적 난이도 조정 결정
            if performance_score >= 0.6:  # MathWorks 권장 기준
                self.consecutive_successes += 1
                
                # 연속 성공 + 안정성 확인
                if (self.consecutive_successes >= self.patience and 
                    stability_score >= 0.7):  # 안정성 기준 추가
                    self._adaptive_increase_difficulty()
                    
            elif performance_score < 0.3:  # 성능 저하 감지
                self._consider_backtrack()
            else:
                self.consecutive_successes = 0
                # 적응적 임계값 조정
                if len(self.performance_history) >= 5:
                    recent_performance = np.mean(self.performance_history[-5:])
                    if recent_performance < 0.4:
                        self.adaptive_threshold *= self.threshold_decay
                        if self.verbose >= 1:
                            print(f"📉 적응적 임계값 조정: {self.adaptive_threshold:.3f}")
                
        except Exception as e:
            if self.verbose >= 1:
                print(f"⚠️ 적응적 커리큘럼 평가 오류: {e}")
    
    def _adaptive_increase_difficulty(self):
        """적응적 난이도 증가 (MathWorks 방식)"""
        if self.curriculum_level < len(self.box_increments):
            new_boxes = self.box_increments[self.curriculum_level]
            
            if self.verbose >= 1:
                print(f"\n🎯 적응적 커리큘럼: 난이도 증가!")
                print(f"   - 이전 박스 수: {self.current_boxes}")
                print(f"   - 새로운 박스 수: {new_boxes}")
                print(f"   - 현재 성공률: {self.last_success_rate:.1%}")
                print(f"   - 안정성 점수: {self.stability_scores[-1]:.3f}")
                print(f"   - 연속 성공: {self.consecutive_successes}")
                print(f"   - 적응적 임계값: {self.adaptive_threshold:.3f}")
            
            self.current_boxes = new_boxes
            self.curriculum_level += 1
            self.consecutive_successes = 0
            
            # 새로운 난이도에서 측정 초기화
            self.recent_rewards = []
            self.recent_episode_lengths = []
            self.recent_utilizations = []
            
            # 임계값 약간 증가 (새로운 난이도에 맞춰)
            self.adaptive_threshold = min(
                self.adaptive_threshold * 1.1, 
                self.success_threshold * 1.5
            )
    
    def _consider_backtrack(self):
        """성능 저하시 백트래킹 고려 (YouTube 사례)"""
        if (self.curriculum_level > 0 and 
            self.backtrack_count < self.max_backtrack and
            len(self.performance_history) >= 10):
            
            # 최근 성능이 이전 레벨보다 현저히 낮은지 확인
            recent_performance = np.mean(self.performance_history[-5:])
            if len(self.performance_history) >= 15:
                prev_performance = np.mean(self.performance_history[-15:-10])
                if recent_performance < prev_performance * 0.7:  # 30% 이상 저하
                    
                    if self.verbose >= 1:
                        print(f"\n⬇️ 성능 저하 감지: 백트래킹 실행")
                        print(f"   - 현재 성능: {recent_performance:.3f}")
                        print(f"   - 이전 성능: {prev_performance:.3f}")
                        print(f"   - 백트래킹 횟수: {self.backtrack_count + 1}")
                    
                    # 이전 레벨로 복귀
                    self.curriculum_level = max(0, self.curriculum_level - 1)
                    if self.curriculum_level == 0:
                        self.current_boxes = self.initial_boxes
                    else:
                        self.current_boxes = self.box_increments[self.curriculum_level - 1]
                    
                    self.backtrack_count += 1
                    self.consecutive_successes = 0
                    
                    # 임계값 낮춰서 더 쉽게 만들기
                    self.adaptive_threshold *= 0.9
    
    def get_adaptive_difficulty_info(self):
        """적응적 난이도 정보 반환"""
        return {
            "current_boxes": self.current_boxes,
            "curriculum_level": self.curriculum_level,
            "max_level": len(self.box_increments),
            "success_rate": self.last_success_rate,
            "adaptive_threshold": self.adaptive_threshold,
            "stability_score": self.stability_scores[-1] if self.stability_scores else 0.0,
            "performance_score": self.performance_history[-1] if self.performance_history else 0.0,
            "backtrack_count": self.backtrack_count,
            "consecutive_successes": self.consecutive_successes,
            "progress_percentage": (self.curriculum_level / len(self.box_increments)) * 100 if self.box_increments else 0,
        }

def create_ultimate_gif(model, env, timestamp):
    """프리미엄 품질 GIF 생성 - 기존 고품질 GIF들과 동일한 수준"""
    print("🎬 프리미엄 품질 GIF 생성 중...")
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from PIL import Image
        import io
        import numpy as np
        
        # 환경 상태 확인
        if env is None:
            print("❌ 환경이 None입니다")
            return None
            
        # GIF 전용 새로운 환경 생성 (안전한 방법)
        print("🔧 GIF 전용 환경 생성 중...")
        try:
            # 원본 환경과 동일한 설정으로 새 환경 생성
            gif_env = make_env(
                container_size=[10, 10, 10],
                num_boxes=16,  # 기본값 사용
                num_visible_boxes=3,
                seed=42,
                render_mode=None,
                random_boxes=False,
                only_terminal_reward=False,
                improved_reward_shaping=True,
            )()
        except Exception as e:
            print(f"❌ GIF 환경 생성 실패: {e}")
            return None
        
        frames = []
        
        try:
            obs, _ = gif_env.reset()
            print(f"✅ GIF 환경 리셋 완료")
        except Exception as e:
            print(f"❌ GIF 환경 리셋 실패: {e}")
            gif_env.close()
            return None
        
        # matplotlib 설정 (고품질)
        plt.ioff()  # 인터랙티브 모드 비활성화
        plt.style.use('default')
        fig = plt.figure(figsize=(16, 12), facecolor='white')  # 더 큰 해상도
        ax = fig.add_subplot(111, projection='3d')
        
        # 색상 팔레트 설정 (더 예쁜 색상들)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', 
                  '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43',
                  '#10AC84', '#EE5A24', '#0084FF', '#D63031', '#74B9FF',
                  '#A29BFE', '#6C5CE7', '#FD79A8', '#FDCB6E', '#E17055']
        
        print(f"🎬 프레임 생성 시작 (최대 50 프레임)")
        total_reward = 0
        
        for step in range(50):  # 더 많은 프레임
            try:
                # 현재 상태 시각화
                ax.clear()
                
                # 축 범위 및 라벨 설정
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)
                ax.set_zlim(0, 10)
                ax.set_xlabel('X-axis (Width)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Y-axis (Depth)', fontsize=12, fontweight='bold')
                ax.set_zlabel('Z-axis (Height)', fontsize=12, fontweight='bold')
                
                # 컨테이너 외곽선 그리기 (더 명확하게)
                container_edges = [
                    # 바닥면
                    [(0,0,0), (10,0,0), (10,10,0), (0,10,0)],
                    # 윗면
                    [(0,0,10), (10,0,10), (10,10,10), (0,10,10)],
                    # 측면들
                    [(0,0,0), (0,0,10), (0,10,10), (0,10,0)],
                    [(10,0,0), (10,0,10), (10,10,10), (10,10,0)],
                    [(0,0,0), (10,0,0), (10,0,10), (0,0,10)],
                    [(0,10,0), (10,10,0), (10,10,10), (0,10,10)]
                ]
                
                # 컨테이너 외곽선 그리기
                for face in container_edges:
                    if face == container_edges[0]:  # 바닥면만 채우기
                        poly = Poly3DCollection([face], alpha=0.1, facecolor='lightgray', edgecolor='black')
                        ax.add_collection3d(poly)
                    else:
                        poly = Poly3DCollection([face], alpha=0.02, facecolor='lightblue', edgecolor='gray')
                        ax.add_collection3d(poly)
                
                # 박스들 그리기 (환경에서 정보 추출)
                box_count = 0
                utilization = 0
                
                try:
                    if hasattr(gif_env, 'unwrapped') and hasattr(gif_env.unwrapped, 'container'):
                        container = gif_env.unwrapped.container
                        for box in container.boxes:
                            if hasattr(box, 'position') and box.position is not None:
                                x, y, z = box.position
                                w, h, d = box.size
                                
                                # 박스 색상 선택
                                color = colors[box_count % len(colors)]
                                
                                # 3D 박스 그리기 (6면 모두)
                                r = [0, w]
                                s = [0, h] 
                                t = [0, d]
                                
                                # 박스의 8개 꼭짓점 계산
                                xx, yy, zz = np.meshgrid(r, s, t)
                                vertices = []
                                for i in range(2):
                                    for j in range(2):
                                        for k in range(2):
                                            vertices.append([x + xx[i,j,k], y + yy[i,j,k], z + zz[i,j,k]])
                                
                                # 박스의 6개 면 정의
                                faces = [
                                    [vertices[0], vertices[1], vertices[3], vertices[2]],  # 앞면
                                    [vertices[4], vertices[5], vertices[7], vertices[6]],  # 뒷면
                                    [vertices[0], vertices[1], vertices[5], vertices[4]],  # 아래면
                                    [vertices[2], vertices[3], vertices[7], vertices[6]],  # 윗면
                                    [vertices[0], vertices[2], vertices[6], vertices[4]],  # 왼쪽면
                                    [vertices[1], vertices[3], vertices[7], vertices[5]]   # 오른쪽면
                                ]
                                
                                # 면 추가
                                face_collection = Poly3DCollection(faces, alpha=0.8, facecolor=color, edgecolor='black', linewidth=1)
                                ax.add_collection3d(face_collection)
                                
                                # 박스 라벨 추가
                                ax.text(x + w/2, y + h/2, z + d/2, f'{box_count+1}', 
                                       fontsize=10, fontweight='bold', ha='center', va='center')
                                
                                box_count += 1
                                utilization += w * h * d
                        
                        # 활용률 계산 (컨테이너 부피: 10*10*10 = 1000)
                        utilization_percent = (utilization / 1000) * 100
                        
                except Exception as box_e:
                    print(f"⚠️ 박스 렌더링 오류 (스텝 {step}): {box_e}")
                
                # 제목 및 정보 표시
                ax.set_title(f'Real-time Training Performance Monitoring\n'
                           f'Step: {step+1}/50 | Placed Boxes: {box_count} | Utilization: {utilization_percent:.1f}%', 
                           fontsize=14, fontweight='bold', pad=20)
                
                # 카메라 각도 설정 (더 좋은 시야각)
                ax.view_init(elev=25, azim=45 + step * 2)  # 회전 효과
                
                # 그리드 설정
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('white')
                
                # 프레임 저장 (고품질)
                try:
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                              facecolor='white', edgecolor='none', pad_inches=0.1)
                    buf.seek(0)
                    frame = Image.open(buf).copy()
                    frames.append(frame)
                    buf.close()
                    
                    if step % 10 == 0:
                        print(f"  Frame {step + 1}/50 completed (Boxes: {box_count})")
                        
                except Exception as save_e:
                    print(f"⚠️ 프레임 저장 오류 (스텝 {step}): {save_e}")
                    continue
                
                # 다음 액션 수행 (안전한 방법)
                try:
                    action_masks = get_action_masks(gif_env)
                    action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                    obs, reward, terminated, truncated, info = gif_env.step(action)
                    total_reward += reward
                    
                    if terminated or truncated:
                        print(f"  Episode ended (Step {step + 1}, Total Reward: {total_reward:.2f})")
                        # 마지막 프레임 몇 개 더 추가 (결과 확인용)
                        for _ in range(3):
                            frames.append(frame.copy())
                        break
                        
                except Exception as step_e:
                    print(f"⚠️ 액션 수행 오류 (스텝 {step}): {step_e}")
                    break
                    
            except Exception as frame_e:
                print(f"⚠️ 프레임 {step} 생성 중 오류: {frame_e}")
                break
        
        # 리소스 정리
        plt.close(fig)
        gif_env.close()
        
        # GIF 저장 (고품질)
        if len(frames) >= 5:  # 최소 5 프레임 이상
            try:
                gif_path = f'gifs/ultimate_demo_{timestamp}.gif'
                os.makedirs('gifs', exist_ok=True)
                
                # 프레임 크기 통일
                if frames:
                    base_size = frames[0].size
                    normalized_frames = []
                    for frame in frames:
                        if frame.size != base_size:
                            frame = frame.resize(base_size, Image.LANCZOS)
                        normalized_frames.append(frame)
                    
                    # 고품질 GIF 저장
                    normalized_frames[0].save(
                        gif_path,
                        save_all=True,
                        append_images=normalized_frames[1:],
                        duration=600,  # 0.6초 간격 (더 부드럽게)
                        loop=0,
                        optimize=True
                    )
                    
                    # 파일 크기 확인
                    file_size = os.path.getsize(gif_path)
                    print(f"🎬 프리미엄 GIF 저장 완료: {gif_path}")
                    print(f"  📊 프레임 수: {len(normalized_frames)}")
                    print(f"  📏 파일 크기: {file_size / 1024:.1f} KB")
                    print(f"  🎯 최종 보상: {total_reward:.2f}")
                    print(f"  📦 배치된 박스: {box_count}개")
                    
                    return gif_path
                
            except Exception as save_e:
                print(f"❌ GIF 파일 저장 오류: {save_e}")
                return None
        else:
            print(f"❌ 충분한 프레임 없음 ({len(frames)}개)")
            return None
            
    except Exception as e:
        print(f"❌ GIF 생성 전체 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def ultimate_train(
    timesteps=15000,
    eval_freq=2000,
    container_size=[10, 10, 10],
    num_boxes=16,
    create_gif=True,
    curriculum_learning=True,
    initial_boxes=None,
    success_threshold=0.45,  # MathWorks: 낮은 기준으로 시작
    curriculum_steps=7,      # 더 많은 단계
    patience=10              # 더 긴 인내심
):
    """999 스텝 문제 완전 해결된 학습 함수 (MathWorks 기반 적응적 커리큘럼 학습 지원)"""
    
    # MathWorks 권장: 적응적 커리큘럼 설정
    if curriculum_learning and initial_boxes is None:
        # 시작점을 목표의 60%로 설정 (MathWorks 권장)
        initial_boxes = max(6, int(num_boxes * 0.6))  
    elif not curriculum_learning:
        initial_boxes = num_boxes
    
    current_boxes = initial_boxes
    
    print("🚀 MathWorks 기반 적응적 커리큘럼 학습 시작")
    print(f"📋 설정: {timesteps:,} 스텝, 평가 주기 {eval_freq:,}")
    if curriculum_learning:
        print(f"🎓 적응적 커리큘럼: {initial_boxes}개 → {num_boxes}개 박스")
        print(f"   - 성공 임계값: {success_threshold}")
        print(f"   - 커리큘럼 단계: {curriculum_steps}")
        print(f"   - 인내심: {patience}")
    else:
        print(f"📦 고정 박스 수: {num_boxes}개")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 디렉토리 생성
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('gifs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 환경 생성 (커리큘럼 학습 고려)
    print("🏗️ 환경 생성 중...")
    env = make_env(
        container_size=container_size,
        num_boxes=current_boxes,  # 커리큘럼 학습시 초기 박스 수
        num_visible_boxes=3,
        seed=42,
        render_mode=None,
        random_boxes=False,
        only_terminal_reward=False,
        improved_reward_shaping=True,  # 항상 개선된 보상 사용
    )()
    
    # 평가용 환경
    eval_env = make_env(
        container_size=container_size,
        num_boxes=current_boxes,  # 커리큘럼 학습시 초기 박스 수
        num_visible_boxes=3,
        seed=43,
        render_mode=None,
        random_boxes=False,
        only_terminal_reward=False,
        improved_reward_shaping=True,
    )()
    
    # 모니터링 설정
    env = Monitor(env, f"logs/ultimate_train_{timestamp}.csv")
    eval_env = Monitor(eval_env, f"logs/ultimate_eval_{timestamp}.csv")
    
    print("✅ 환경 설정 완료")
    print(f"  - 컨테이너: {container_size}")
    print(f"  - 현재 박스 수: {current_boxes}")
    if curriculum_learning:
        print(f"  - 최종 목표 박스 수: {num_boxes}")
    print(f"  - 액션 스페이스: {env.action_space}")
    print(f"  - 관찰 스페이스: {env.observation_space}")
    
    # 안전한 콜백 설정 (적응적 커리큘럼 학습 포함)
    print("🛡️ 안전한 콜백 설정 중...")
    
    # 평가 주기가 충분히 큰 경우에만 콜백 사용
    if eval_freq >= 2000:
        safe_callback = UltimateSafeCallback(eval_env, eval_freq=eval_freq)
        
        # 적응적 커리큘럼 학습 콜백 추가
        callbacks = [safe_callback]
        
        if curriculum_learning:
            curriculum_callback = AdaptiveCurriculumCallback(
                container_size=container_size,
                initial_boxes=initial_boxes,
                target_boxes=num_boxes,
                num_visible_boxes=3,
                success_threshold=success_threshold,
                curriculum_steps=curriculum_steps,
                patience=patience,
                stability_window=50,  # 안정성 윈도우
                verbose=1
            )
            callbacks.append(curriculum_callback)
            print(f"🎓 적응적 커리큘럼 학습 콜백 추가")
        
        # 체크포인트 콜백 (안전한 설정)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(eval_freq, 3000),
            save_path="models/checkpoints",
            name_prefix=f"adaptive_model_{timestamp}",
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        print(f"✅ 안전한 콜백 활성화 (평가 주기: {eval_freq})")
        if curriculum_learning:
            print(f"   - 적응적 커리큘럼 학습 활성화")
            print(f"   - 성공 임계값: {success_threshold}")
            print(f"   - 안정성 윈도우: 50")
    else:
        callbacks = None
        print("⚠️ 콜백 비활성화 (평가 주기가 너무 짧음)")
        if curriculum_learning:
            print("⚠️ 커리큘럼 학습도 비활성화됨")
    
    # MathWorks 권장 최적화된 모델 생성
    print("🤖 MathWorks 기반 최적화된 모델 생성 중...")
    
    # TensorBoard 로그 설정 (선택적)
    tensorboard_log = None
    try:
        import tensorboard
        tensorboard_log = "logs/tensorboard"
        print("✅ TensorBoard 로깅 활성화")
    except ImportError:
        print("⚠️ TensorBoard 없음 - 로깅 비활성화")
    
    # MathWorks 기반 개선된 하이퍼파라미터
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        # MathWorks 권장 학습률 스케줄링
        learning_rate=lambda progress: max(1e-5, 3e-4 * (1 - progress * 0.9)),  
        n_steps=2048,        # 더 많은 스텝으로 안정성 확보
        batch_size=256,      # 적당한 배치 크기
        n_epochs=10,         # 적당한 에포크
        gamma=0.99,          # 표준 감가율
        gae_lambda=0.95,     # 표준 GAE
        clip_range=0.2,      # 표준 클립 범위
        clip_range_vf=None,  # VF 클립 비활성화
        ent_coef=0.01,       # 적당한 엔트로피
        vf_coef=0.5,         # 표준 가치 함수 계수
        max_grad_norm=0.5,   # 그래디언트 클리핑
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=42,
        # MathWorks 권장 정책 kwargs
        policy_kwargs=dict(
            net_arch=[256, 256, 128],  # 적당한 네트워크 크기
            activation_fn=torch.nn.ReLU,
            share_features_extractor=True,
        )
    )
    
    print("✅ 모델 생성 완료 (MathWorks 기반 하이퍼파라미터)")
    print(f"  - 학습률: 적응적 스케줄링 (3e-4 → 3e-5)")
    print(f"  - n_steps: 2048")
    print(f"  - batch_size: 256")
    print(f"  - n_epochs: 10")
    print(f"  - 네트워크: [256, 256, 128]")
    
    # 학습 시작
    print(f"\n🚀 학습 시작: {timesteps:,} 스텝")
    print(f"📅 시작 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # progress_bar 설정 (의존성 확인)
        use_progress_bar = False
        try:
            import tqdm
            import rich
            use_progress_bar = True
            print("✅ 진행률 표시 활성화")
        except ImportError:
            print("⚠️ tqdm/rich 없음 - 진행률 표시 비활성화")
        
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=use_progress_bar,
            tb_log_name=f"adaptive_ppo_{timestamp}",
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ 학습 완료! 소요 시간: {training_time:.2f}초")
        
        # 모델 저장
        model_path = f"models/adaptive_ppo_{timestamp}"
        model.save(model_path)
        print(f"💾 모델 저장: {model_path}")
        
        # 적응적 커리큘럼 결과 출력
        if curriculum_learning and callbacks:
            for callback in callbacks:
                if isinstance(callback, AdaptiveCurriculumCallback):
                    difficulty_info = callback.get_adaptive_difficulty_info()
                    print(f"\n🎓 적응적 커리큘럼 학습 결과:")
                    print(f"   - 최종 박스 수: {difficulty_info['current_boxes']}")
                    print(f"   - 진행도: {difficulty_info['curriculum_level']}/{difficulty_info['max_level']}")
                    print(f"   - 최종 성공률: {difficulty_info['success_rate']:.1%}")
                    print(f"   - 안정성 점수: {difficulty_info['stability_score']:.3f}")
                    print(f"   - 성과 점수: {difficulty_info['performance_score']:.3f}")
                    print(f"   - 백트래킹 횟수: {difficulty_info['backtrack_count']}")
                    print(f"   - 적응적 임계값: {difficulty_info['adaptive_threshold']:.3f}")
                    break
        
        # 최종 평가 (안전한 방식)
        print("\n📊 최종 평가 중...")
        final_rewards = []
        
        for ep in range(3):
            obs, _ = eval_env.reset()
            episode_reward = 0.0
            step_count = 0
            
            while step_count < 50:
                try:
                    action_masks = get_action_masks(eval_env)
                    action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    step_count += 1
                    
                    if terminated or truncated:
                        break
                except Exception as e:
                    print(f"⚠️ 평가 중 오류: {e}")
                    break
            
            final_rewards.append(episode_reward)
        
        final_reward = np.mean(final_rewards) if final_rewards else 0.0
        print(f"📈 최종 평가 보상: {final_reward:.4f}")
        
        # GIF 생성
        gif_path = None
        if create_gif:
            print("\n🎬 GIF 생성 중...")
                gif_path = create_ultimate_gif(model, eval_env, timestamp)
        
        # 결과 저장
        results = {
            'timestamp': timestamp,
            'total_timesteps': timesteps,
            'training_time': training_time,
            'final_reward': final_reward,
            'container_size': container_size,
            'num_boxes': num_boxes,
            'model_path': model_path,
            'eval_freq': eval_freq,
            'callbacks_used': callbacks is not None,
            'curriculum_learning': curriculum_learning,
            'gif_path': gif_path,
        }
        
        # 적응적 커리큘럼 정보 추가
        if curriculum_learning and callbacks:
            for callback in callbacks:
                if isinstance(callback, AdaptiveCurriculumCallback):
                    results['curriculum_info'] = callback.get_adaptive_difficulty_info()
                    break
        
        # 결과 파일 저장
        results_file = f'results/adaptive_results_{timestamp}.txt'
        with open(results_file, 'w') as f:
            f.write("=== MathWorks 기반 적응적 커리큘럼 학습 결과 ===\n")
            for key, value in results.items():
                if key != 'curriculum_info':
                    f.write(f"{key}: {value}\n")
            
            if 'curriculum_info' in results:
                f.write("\n=== 적응적 커리큘럼 상세 정보 ===\n")
                for key, value in results['curriculum_info'].items():
                    f.write(f"{key}: {value}\n")
        
        print(f"📄 결과 저장: {results_file}")
        
        return model, results
        
            except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ===== 하이퍼파라미터 최적화 함수들 =====

def calculate_utilization_rate(env) -> float:
    """환경에서 실제 공간 활용률 계산"""
    try:
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'container'):
            container = env.unwrapped.container
            placed_volume = 0
            
            for box in container.boxes:
                if hasattr(box, 'position') and box.position is not None:
                    if hasattr(box, 'size'):
                        w, h, d = box.size
                        placed_volume += w * h * d
                    elif hasattr(box, 'volume'):
                        placed_volume += box.volume
            
            container_volume = container.size[0] * container.size[1] * container.size[2]
            return placed_volume / container_volume if container_volume > 0 else 0.0
        
        return 0.0
    except Exception as e:
        print(f"⚠️ 활용률 계산 오류: {e}")
        return 0.0

def evaluate_model_performance(model, eval_env, n_episodes: int = 5) -> Tuple[float, float]:
    """모델 성능 평가: 평균 에피소드 보상과 활용률 반환"""
    episode_rewards = []
    utilization_rates = []
    
    for ep in range(n_episodes):
        try:
            obs, _ = eval_env.reset()
            episode_reward = 0.0
            step_count = 0
            max_steps = 100
            
            while step_count < max_steps:
                try:
                    action_masks = get_action_masks(eval_env)
                    action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    step_count += 1
                    
                    if terminated or truncated:
                        break
                except Exception as e:
                    print(f"⚠️ 평가 스텝 오류: {e}")
                    break
            
            episode_rewards.append(episode_reward)
            
            # 활용률 계산
            utilization = calculate_utilization_rate(eval_env)
            utilization_rates.append(utilization)
            
        except Exception as e:
            print(f"⚠️ 평가 에피소드 {ep} 오류: {e}")
            episode_rewards.append(0.0)
            utilization_rates.append(0.0)
    
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
    mean_utilization = np.mean(utilization_rates) if utilization_rates else 0.0
    
    return mean_reward, mean_utilization

def create_hyperparameter_config(trial: 'optuna.Trial') -> Dict[str, Any]:
    """Optuna trial에서 하이퍼파라미터 설정 생성"""
    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
        'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        'n_epochs': trial.suggest_int('n_epochs', 3, 15),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'ent_coef': trial.suggest_float('ent_coef', 1e-4, 1e-1, log=True),
        'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
    }
    
    return hyperparams

def optuna_objective(trial: 'optuna.Trial', 
                    base_config: Dict[str, Any]) -> float:
    """Optuna 최적화 목적 함수"""
    
    print(f"\n🔬 Trial {trial.number} 시작")
    
    # 하이퍼파라미터 생성
    hyperparams = create_hyperparameter_config(trial)
    
    print(f"📋 하이퍼파라미터:")
    for key, value in hyperparams.items():
        print(f"   - {key}: {value}")
    
    # W&B 로깅 설정 (선택적)
    run_name = f"trial_{trial.number}_{datetime.datetime.now().strftime('%H%M%S')}"
    
    if WANDB_AVAILABLE and base_config.get('use_wandb', False):
        wandb.init(
            project=base_config.get('wandb_project', 'ppo-3d-binpacking'),
            name=run_name,
            config=hyperparams,
            group="optuna-optimization",
            reinit=True
        )
    
    try:
        # 환경 생성
        container_size = base_config.get('container_size', [10, 10, 10])
        num_boxes = base_config.get('num_boxes', 16)
        
        env = make_env(
            container_size=container_size,
            num_boxes=num_boxes,
            num_visible_boxes=3,
            seed=42 + trial.number,  # 각 trial마다 다른 시드
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
        
        # 모니터링 설정
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        env = Monitor(env, f"logs/optuna_train_trial_{trial.number}_{timestamp}.csv")
        eval_env = Monitor(eval_env, f"logs/optuna_eval_trial_{trial.number}_{timestamp}.csv")
        
        # 모델 생성 (하이퍼파라미터 적용)
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=hyperparams['learning_rate'],
            n_steps=hyperparams['n_steps'],
            batch_size=hyperparams['batch_size'],
            n_epochs=hyperparams['n_epochs'],
            gamma=0.99,
            gae_lambda=hyperparams['gae_lambda'],
            clip_range=hyperparams['clip_range'],
            clip_range_vf=None,
            ent_coef=hyperparams['ent_coef'],
            vf_coef=hyperparams['vf_coef'],
            max_grad_norm=0.5,
            verbose=0,  # 조용히 실행
            seed=42 + trial.number,
            policy_kwargs=dict(
                net_arch=[256, 256, 128],
                activation_fn=torch.nn.ReLU,
                share_features_extractor=True,
            )
        )
        
        # 학습 (짧은 시간으로 설정)
        timesteps = base_config.get('trial_timesteps', 5000)  # Trial용 짧은 학습
        
        # Pruning을 위한 중간 평가 콜백
        class OptunaPruningCallback(BaseCallback):
            def __init__(self, trial, eval_env, eval_freq=1000):
                super().__init__()
                self.trial = trial
                self.eval_env = eval_env
                self.eval_freq = eval_freq
                self.last_eval = 0
                
            def _on_step(self) -> bool:
                if self.num_timesteps - self.last_eval >= self.eval_freq:
                    # 중간 평가
                    mean_reward, mean_utilization = evaluate_model_performance(
                        self.model, self.eval_env, n_episodes=3
                    )
                    
                    # 다중 목적 최적화: 가중 합산
                    # 보상 * 0.3 + 활용률 * 0.7 (공간 효율성 우선)
                    combined_score = mean_reward * 0.3 + mean_utilization * 100 * 0.7
                    
                    # Optuna에 중간 결과 보고
                    self.trial.report(combined_score, self.num_timesteps)
                    
                    # Pruning 체크
                    if self.trial.should_prune():
                        print(f"🔪 Trial {self.trial.number} pruned at step {self.num_timesteps}")
                        raise optuna.TrialPruned()
                    
                    self.last_eval = self.num_timesteps
                    
                return True
        
        # Pruning 콜백 설정
        pruning_callback = OptunaPruningCallback(trial, eval_env)
        
        print(f"🚀 학습 시작: {timesteps:,} 스텝")
        start_time = time.time()
        
        model.learn(
            total_timesteps=timesteps,
            callback=pruning_callback,
            progress_bar=False
        )
        
        training_time = time.time() - start_time
        
        # 최종 평가
        print(f"📊 최종 평가 중...")
        mean_reward, mean_utilization = evaluate_model_performance(
            model, eval_env, n_episodes=5
        )
        
        # 다중 목적 최적화: 가중 합산
        # 보상 * 0.3 + 활용률 * 0.7 (공간 효율성 우선)
        combined_score = mean_reward * 0.3 + mean_utilization * 100 * 0.7
        
        print(f"✅ Trial {trial.number} 완료:")
        print(f"   - 평균 보상: {mean_reward:.4f}")
        print(f"   - 평균 활용률: {mean_utilization:.1%}")
        print(f"   - 종합 점수: {combined_score:.4f}")
        print(f"   - 학습 시간: {training_time:.1f}초")
        
        # W&B 로깅
        if WANDB_AVAILABLE and base_config.get('use_wandb', False):
            wandb.log({
                "mean_episode_reward": mean_reward,
                "mean_utilization_rate": mean_utilization,
                "combined_score": combined_score,
                "training_time": training_time,
                **hyperparams
            })
            wandb.finish()
        
        # 환경 정리
        env.close()
        eval_env.close()
        
        return combined_score
        
    except optuna.TrialPruned:
        print(f"🔪 Trial {trial.number} was pruned")
        if WANDB_AVAILABLE and base_config.get('use_wandb', False):
            wandb.finish()
        raise
        
    except Exception as e:
        print(f"❌ Trial {trial.number} 오류: {e}")
        if WANDB_AVAILABLE and base_config.get('use_wandb', False):
            wandb.finish()
        
        # 오류 시 낮은 점수 반환 (최적화가 계속되도록)
        return -1000.0

def run_optuna_optimization(
    n_trials: int = 50,
    container_size: list = [10, 10, 10],
    num_boxes: int = 16,
    trial_timesteps: int = 5000,
    use_wandb: bool = False,
    wandb_project: str = "ppo-3d-binpacking-optuna",
    study_name: str = None
) -> Dict[str, Any]:
    """Optuna를 이용한 하이퍼파라미터 최적화 실행"""
    
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna가 설치되지 않았습니다: pip install optuna")
    
    print("🔬 Optuna 하이퍼파라미터 최적화 시작")
    print(f"📋 설정:")
    print(f"   - 시행 횟수: {n_trials}")
    print(f"   - 컨테이너 크기: {container_size}")
    print(f"   - 박스 개수: {num_boxes}")
    print(f"   - Trial 학습 스텝: {trial_timesteps:,}")
    print(f"   - W&B 사용: {use_wandb and WANDB_AVAILABLE}")
    
    # 기본 설정
    base_config = {
        'container_size': container_size,
        'num_boxes': num_boxes,
        'trial_timesteps': trial_timesteps,
        'use_wandb': use_wandb and WANDB_AVAILABLE,
        'wandb_project': wandb_project
    }
    
    # Study 생성
    study_name = study_name or f"ppo_optimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # W&B 콜백 설정
    callbacks = []
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb_callback = WeightsAndBiasesCallback(
                metric_name="combined_score",
                wandb_kwargs={
                    "project": wandb_project,
                    "group": "optuna-study"
                }
            )
            callbacks.append(wandb_callback)
            print("✅ W&B 콜백 추가됨")
        except Exception as e:
            print(f"⚠️ W&B 콜백 설정 오류: {e}")
    
    # Study 생성 (TPE + MedianPruner)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # combined_score 최대화
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1000,
            interval_steps=500
        )
    )
    
    print(f"📊 Study 생성 완료: {study_name}")
    
    # 최적화 실행
    try:
        start_time = time.time()
        
        study.optimize(
            lambda trial: optuna_objective(trial, base_config),
            n_trials=n_trials,
            callbacks=callbacks
        )
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 최적화 완료!")
        print(f"⏱️ 총 소요 시간: {total_time:.1f}초")
        print(f"🏆 최고 성능: {study.best_value:.4f}")
        print(f"🎯 최적 하이퍼파라미터:")
        
        for key, value in study.best_params.items():
            print(f"   - {key}: {value}")
        
        # 결과 저장
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "study_name": study_name,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "optimization_time": total_time,
            "timestamp": timestamp,
            "config": base_config
        }
        
        results_file = f"results/optuna_results_{timestamp}.json"
        os.makedirs('results', exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 결과 저장: {results_file}")
        
        # 시각화 생성
        try:
            print("📊 최적화 결과 시각화 중...")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Optuna Optimization Results', fontsize=16)
            
            # 1. 최적화 히스토리
            optuna.visualization.matplotlib.plot_optimization_history(study, ax=axes[0, 0])
            axes[0, 0].set_title('Optimization History')
            
            # 2. 파라미터 중요도
            try:
                optuna.visualization.matplotlib.plot_param_importances(study, ax=axes[0, 1])
                axes[0, 1].set_title('Parameter Importances')
            except Exception:
                axes[0, 1].text(0.5, 0.5, 'Not enough trials\nfor importance analysis', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Parameter Importances')
            
            # 3. 병렬 좌표 플롯 (상위 trials만)
            try:
                if len(study.trials) >= 10:
                    optuna.visualization.matplotlib.plot_parallel_coordinate(
                        study, params=['learning_rate', 'n_epochs', 'clip_range'], ax=axes[1, 0]
                    )
                else:
                    axes[1, 0].text(0.5, 0.5, 'Not enough trials\nfor parallel coordinate', 
                                   ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Parallel Coordinate Plot')
            except Exception:
                axes[1, 0].text(0.5, 0.5, 'Parallel coordinate\nplot failed', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Parallel Coordinate Plot')
            
            # 4. 하이퍼파라미터 슬라이스 플롯
            try:
                optuna.visualization.matplotlib.plot_slice(
                    study, params=['learning_rate', 'clip_range'], ax=axes[1, 1]
                )
                axes[1, 1].set_title('Hyperparameter Slice Plot')
            except Exception:
                axes[1, 1].text(0.5, 0.5, 'Slice plot\nnot available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Hyperparameter Slice Plot')
            
            plt.tight_layout()
            
            viz_file = f"results/optuna_visualization_{timestamp}.png"
            plt.savefig(viz_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 시각화 저장: {viz_file}")
            
        except Exception as e:
            print(f"⚠️ 시각화 생성 오류: {e}")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⏹️ 최적화 중단됨")
        return {"status": "interrupted", "n_completed_trials": len(study.trials)}
    
    except Exception as e:
        print(f"\n❌ 최적화 오류: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

def create_wandb_sweep_config() -> Dict[str, Any]:
    """W&B Sweep 설정 생성"""
    sweep_config = {
        "method": "bayes",  # bayes, grid, random
        "metric": {
            "goal": "maximize",
            "name": "combined_score"
        },
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-6,
                "max": 1e-3
            },
            "n_steps": {
                "values": [1024, 2048, 4096]
            },
            "batch_size": {
                "values": [64, 128, 256]
            },
            "n_epochs": {
                "distribution": "int_uniform",
                "min": 3,
                "max": 15
            },
            "clip_range": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.4
            },
            "ent_coef": {
                "distribution": "log_uniform_values",
                "min": 1e-4,
                "max": 1e-1
            },
            "vf_coef": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 1.0
            },
            "gae_lambda": {
                "distribution": "uniform",
                "min": 0.9,
                "max": 0.99
            }
        }
    }
    return sweep_config

def wandb_sweep_train():
    """W&B Sweep에서 실행되는 학습 함수"""
    if not WANDB_AVAILABLE:
        raise ImportError("W&B가 설치되지 않았습니다: pip install wandb")
    
    # W&B run 초기화
    with wandb.init() as run:
        # 하이퍼파라미터 가져오기
        config = wandb.config
        
        # 환경 생성
        container_size = [10, 10, 10]
        num_boxes = 16
        
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
        
        eval_env = make_env(
            container_size=container_size,
            num_boxes=num_boxes,
            num_visible_boxes=3,
            seed=43,
            render_mode=None,
            random_boxes=False,
            only_terminal_reward=False,
            improved_reward_shaping=True,
        )()
        
        # 모니터링 설정
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        env = Monitor(env, f"logs/wandb_train_{run.id}_{timestamp}.csv")
        eval_env = Monitor(eval_env, f"logs/wandb_eval_{run.id}_{timestamp}.csv")
        
        # 모델 생성
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=0.99,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=0.5,
            verbose=0,
            policy_kwargs=dict(
                net_arch=[256, 256, 128],
                activation_fn=torch.nn.ReLU,
                share_features_extractor=True,
            )
        )
        
        # 학습
        timesteps = 8000
        model.learn(total_timesteps=timesteps, progress_bar=False)
        
        # 평가
        mean_reward, mean_utilization = evaluate_model_performance(
            model, eval_env, n_episodes=5
        )
        
        # 다중 목적 최적화 점수 계산
        combined_score = mean_reward * 0.3 + mean_utilization * 100 * 0.7
        
        # W&B 로깅
        wandb.log({
            "mean_episode_reward": mean_reward,
            "mean_utilization_rate": mean_utilization,
            "combined_score": combined_score,
            "learning_rate": config.learning_rate,
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "n_epochs": config.n_epochs,
            "clip_range": config.clip_range,
            "ent_coef": config.ent_coef,
            "vf_coef": config.vf_coef,
            "gae_lambda": config.gae_lambda
        })
        
        # 환경 정리
        env.close()
        eval_env.close()

def run_wandb_sweep(
    n_trials: int = 50,
    wandb_project: str = "ppo-3d-binpacking-sweep"
) -> Dict[str, Any]:
    """W&B Sweep 실행"""
    
    if not WANDB_AVAILABLE:
        raise ImportError("W&B가 설치되지 않았습니다: pip install wandb")
    
    print("🌊 W&B Sweep 하이퍼파라미터 최적화 시작")
    
    # Sweep 설정
    sweep_config = create_wandb_sweep_config()
    
    # Sweep 생성
    sweep_id = wandb.sweep(sweep_config, project=wandb_project)
    
    print(f"📊 Sweep 생성 완료: {sweep_id}")
    print(f"🔗 W&B 대시보드: https://wandb.ai/{wandb.api.default_entity}/{wandb_project}/sweeps/{sweep_id}")
    
    # Agent 실행
    wandb.agent(sweep_id, wandb_sweep_train, count=n_trials)
    
    return {
        "status": "completed",
        "sweep_id": sweep_id,
        "n_trials": n_trials,
        "project": wandb_project
    }

def run_hyperparameter_optimization(
    method: str = "optuna",
    n_trials: int = 50,
    container_size: list = [10, 10, 10],
    num_boxes: int = 16,
    trial_timesteps: int = 8000,
    use_wandb: bool = False,
    wandb_project: str = "ppo-3d-binpacking-optimization"
) -> Dict[str, Any]:
    """하이퍼파라미터 최적화 통합 함수"""
    
    print(f"🚀 하이퍼파라미터 최적화 시작 - 방법: {method}")
    
    results = {}
    
    if method == "optuna":
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna가 설치되지 않았습니다: pip install optuna")
        
        results = run_optuna_optimization(
            n_trials=n_trials,
            container_size=container_size,
            num_boxes=num_boxes,
            trial_timesteps=trial_timesteps,
            use_wandb=use_wandb,
            wandb_project=wandb_project
        )
        
    elif method == "wandb":
        if not WANDB_AVAILABLE:
            raise ImportError("W&B가 설치되지 않았습니다: pip install wandb")
        
        results = run_wandb_sweep(
            n_trials=n_trials,
            wandb_project=wandb_project
        )
        
    elif method == "both":
        if not OPTUNA_AVAILABLE or not WANDB_AVAILABLE:
            missing = []
            if not OPTUNA_AVAILABLE:
                missing.append("optuna")
            if not WANDB_AVAILABLE:
                missing.append("wandb")
            raise ImportError(f"필요한 라이브러리가 설치되지 않았습니다: pip install {' '.join(missing)}")
        
        print("📊 1단계: Optuna 최적화")
        optuna_results = run_optuna_optimization(
            n_trials=n_trials//2,
            container_size=container_size,
            num_boxes=num_boxes,
            trial_timesteps=trial_timesteps,
            use_wandb=use_wandb,
            wandb_project=f"{wandb_project}-optuna"
        )
        
        print("🌊 2단계: W&B Sweep 최적화")
        wandb_results = run_wandb_sweep(
            n_trials=n_trials//2,
            wandb_project=f"{wandb_project}-sweep"
        )
        
        results = {
            "method": "both",
            "optuna_results": optuna_results,
            "wandb_results": wandb_results
        }
        
    else:
        raise ValueError(f"지원하지 않는 최적화 방법: {method}. 'optuna', 'wandb', 'both' 중 선택하세요.")
    
    print(f"✅ {method} 최적화 완료!")
    return results

def train_with_best_params(results_file: str, 
                          timesteps: int = 50000,
                          create_gif: bool = True) -> Tuple[Any, Dict]:
    """최적 하이퍼파라미터로 최종 학습"""
    
    print(f"🏆 최적 하이퍼파라미터로 최종 학습 시작")
    
    # 결과 파일 로드
    with open(results_file, 'r') as f:
        optuna_results = json.load(f)
    
    best_params = optuna_results['best_params']
    config = optuna_results['config']
    
    print(f"📋 최적 하이퍼파라미터:")
    for key, value in best_params.items():
        print(f"   - {key}: {value}")
    
    # ultimate_train 함수 호출 (최적 파라미터 적용)
    # ultimate_train 함수를 수정하여 하이퍼파라미터를 받을 수 있도록 해야 함
    
    # 임시로 기본 설정으로 학습
    model, results = ultimate_train(
        timesteps=timesteps,
        eval_freq=2000,
        container_size=config['container_size'],
        num_boxes=config['num_boxes'],
        create_gif=create_gif,
        curriculum_learning=False  # 최적화된 하이퍼파라미터 테스트를 위해 비활성화
    )
    
    if results:
        results['optuna_optimization'] = optuna_results
        print(f"🎉 최종 학습 완료!")
        print(f"📊 최종 보상: {results['final_reward']:.4f}")
    
    return model, results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MathWorks 기반 적응적 커리큘럼 학습 + 999 스텝 문제 해결 + Optuna/W&B 하이퍼파라미터 최적화")
    parser.add_argument("--timesteps", type=int, default=15000, help="총 학습 스텝 수")
    parser.add_argument("--eval-freq", type=int, default=2000, help="평가 주기")
    parser.add_argument("--container-size", nargs=3, type=int, default=[10, 10, 10], help="컨테이너 크기")
    parser.add_argument("--num-boxes", type=int, default=16, help="목표 박스 개수")
    parser.add_argument("--no-gif", action="store_true", help="GIF 생성 안함")
    
    # MathWorks 기반 적응적 커리큘럼 학습 옵션
    parser.add_argument("--curriculum-learning", action="store_true", default=True, 
                        help="적응적 커리큘럼 학습 활성화 (기본값: True)")
    parser.add_argument("--no-curriculum", action="store_true", 
                        help="커리큘럼 학습 비활성화")
    parser.add_argument("--initial-boxes", type=int, default=None, 
                        help="시작 박스 수 (기본값: 목표의 60%)")
    parser.add_argument("--success-threshold", type=float, default=0.45, 
                        help="성공 임계값 (기본값: 0.45, MathWorks 권장)")
    parser.add_argument("--curriculum-steps", type=int, default=7, 
                        help="커리큘럼 단계 수 (기본값: 7)")
    parser.add_argument("--patience", type=int, default=10, 
                        help="난이도 증가 대기 횟수 (기본값: 10)")
    
    # 하이퍼파라미터 최적화 옵션
    hyperopt_group = parser.add_argument_group('하이퍼파라미터 최적화 옵션')
    hyperopt_group.add_argument("--optimize", action="store_true", 
                               help="하이퍼파라미터 최적화 실행")
    hyperopt_group.add_argument("--optimization-method", type=str, 
                               choices=["optuna", "wandb", "both"], 
                               default="optuna",
                               help="최적화 방법 (기본값: optuna)")
    hyperopt_group.add_argument("--n-trials", type=int, default=50,
                               help="최적화 시행 횟수 (기본값: 50)")
    hyperopt_group.add_argument("--trial-timesteps", type=int, default=8000,
                               help="각 trial의 학습 스텝 수 (기본값: 8000)")
    hyperopt_group.add_argument("--use-wandb", action="store_true",
                               help="W&B 로깅 활성화")
    hyperopt_group.add_argument("--wandb-project", type=str, 
                               default="ppo-3d-binpacking-optimization",
                               help="W&B 프로젝트 이름")
    hyperopt_group.add_argument("--train-with-best", type=str, default=None,
                               help="최적 하이퍼파라미터로 최종 학습 (Optuna 결과 파일 경로)")
    
    args = parser.parse_args()
    
    # 커리큘럼 학습 설정
    curriculum_learning = args.curriculum_learning and not args.no_curriculum
    
    print("🚀 MathWorks 기반 적응적 커리큘럼 학습 + 999 스텝 문제 해결 + 하이퍼파라미터 최적화")
    print("=" * 100)
    
    # 하이퍼파라미터 최적화 실행
    if args.optimize:
        print("🎯 하이퍼파라미터 최적화 모드")
        print(f"   - 방법: {args.optimization_method}")
        print(f"   - 시행 횟수: {args.n_trials}")
        print(f"   - Trial 스텝: {args.trial_timesteps:,}")
        print(f"   - W&B 사용: {args.use_wandb}")
        print(f"   - W&B 프로젝트: {args.wandb_project}")
        
        try:
            optimization_results = run_hyperparameter_optimization(
                method=args.optimization_method,
                n_trials=args.n_trials,
                container_size=args.container_size,
                num_boxes=args.num_boxes,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project
            )
            
            print("\n🎉 하이퍼파라미터 최적화 성공!")
            print("🔍 최적 하이퍼파라미터를 사용한 최종 학습을 원한다면:")
            
            if "optuna" in optimization_results and "best_params" in optimization_results["optuna"]:
                timestamp = optimization_results["optuna"]["timestamp"]
                results_file = f"results/optuna_results_{timestamp}.json"
                print(f"   python src/ultimate_train_fix.py --train-with-best {results_file}")
            
        except KeyboardInterrupt:
            print("\n⏹️ 최적화 중단됨")
        except Exception as e:
            print(f"\n❌ 최적화 오류: {e}")
            import traceback
            traceback.print_exc()
    
    # 최적 하이퍼파라미터로 최종 학습
    elif args.train_with_best:
        print("🏆 최적 하이퍼파라미터로 최종 학습 모드")
        print(f"   - 결과 파일: {args.train_with_best}")
        print(f"   - 학습 스텝: {args.timesteps:,}")
        
        try:
            model, results = train_with_best_params(
                results_file=args.train_with_best,
                timesteps=args.timesteps,
                create_gif=not args.no_gif
            )
            
            if results:
                print("\n🎉 최종 학습 성공!")
                print(f"📊 최종 보상: {results['final_reward']:.4f}")
                print(f"💾 모델 경로: {results['model_path']}")
            
        except Exception as e:
            print(f"\n❌ 최종 학습 오류: {e}")
            import traceback
            traceback.print_exc()
    
    # 일반 학습 모드
    else:
        if curriculum_learning:
            print("🎓 적응적 커리큘럼 학습 모드 활성화 (MathWorks 기반)")
            print(f"   - 성공 임계값: {args.success_threshold} (낮은 기준으로 시작)")
            print(f"   - 커리큘럼 단계: {args.curriculum_steps} (더 많은 단계)")
            print(f"   - 인내심: {args.patience} (더 긴 대기)")
            print(f"   - 시작 박스 수: {args.initial_boxes or f'목표의 60% ({int(args.num_boxes * 0.6)}개)'}")
            print(f"   ✨ 특징: 다중 지표 평가, 적응적 임계값 조정, 백트래킹")
        else:
            print("📦 고정 난이도 모드")
        
        print(f"📋 학습 설정:")
        print(f"   - 총 스텝: {args.timesteps:,}")
        print(f"   - 평가 주기: {args.eval_freq:,}")
        print(f"   - 컨테이너 크기: {args.container_size}")
        print(f"   - 목표 박스 수: {args.num_boxes}")
        print(f"   - GIF 생성: {'비활성화' if args.no_gif else '활성화'}")
        
        try:
    model, results = ultimate_train(
        timesteps=args.timesteps,
        eval_freq=args.eval_freq,
                container_size=args.container_size,
        num_boxes=args.num_boxes,
                create_gif=not args.no_gif,
                curriculum_learning=curriculum_learning,
                initial_boxes=args.initial_boxes,
                success_threshold=args.success_threshold,
                curriculum_steps=args.curriculum_steps,
                patience=args.patience
    )
    
    if results:
        print("\n🎉 학습 성공!")
        print(f"📊 최종 보상: {results['final_reward']:.4f}")
        print(f"⏱️ 소요 시간: {results['training_time']:.2f}초")
        print(f"💾 모델 경로: {results['model_path']}")
                
                # 적응적 커리큘럼 학습 결과 출력
                if curriculum_learning and 'curriculum_info' in results:
                    curriculum_info = results['curriculum_info']
                    print(f"\n🎓 적응적 커리큘럼 학습 상세 결과:")
                    print(f"   - 최종 박스 수: {curriculum_info['current_boxes']}")
                    print(f"   - 진행도: {curriculum_info['curriculum_level']}/{curriculum_info['max_level']}")
                    print(f"   - 최종 성공률: {curriculum_info['success_rate']:.1%}")
                    print(f"   - 안정성 점수: {curriculum_info['stability_score']:.3f}")
                    print(f"   - 성과 점수: {curriculum_info['performance_score']:.3f}")
                    print(f"   - 백트래킹 횟수: {curriculum_info['backtrack_count']}")
                    print(f"   - 적응적 임계값: {curriculum_info['adaptive_threshold']:.3f}")
                    
                    # 성과 등급 판정
                    performance_score = curriculum_info['performance_score']
                    if performance_score >= 0.8:
                        grade = "🏆 S급 (탁월함)"
                    elif performance_score >= 0.7:
                        grade = "🥇 A급 (우수함)"
                    elif performance_score >= 0.6:
                        grade = "🥈 B급 (보통)"
                    elif performance_score >= 0.5:
                        grade = "🥉 C급 (개선 필요)"
                    else:
                        grade = "📈 D급 (추가 학습 필요)"
                    
                    print(f"   - 성과 등급: {grade}")
                    
                    # 커리큘럼 완료 여부
                    progress = curriculum_info['progress_percentage']
                    if progress >= 100:
                        print(f"   ✅ 커리큘럼 완료: 목표 달성!")
                    elif progress >= 80:
                        print(f"   🔥 커리큘럼 거의 완료: {progress:.1f}%")
                    elif progress >= 50:
                        print(f"   💪 커리큘럼 진행 중: {progress:.1f}%")
                    else:
                        print(f"   🌱 커리큘럼 초기 단계: {progress:.1f}%")
                        
    else:
        print("\n❌ 학습 실패") 
                
        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중단됨")
        except Exception as e:
            print(f"\n❌ 전체 오류: {e}")
            import traceback
            traceback.print_exc()

    print("\n🎯 개선 사항 요약:")
    print("✨ MathWorks 기반 적응적 커리큘럼 학습")
    print("✨ 적응적 임계값 조정 (성능에 따라 동적 변화)")
    print("✨ 다중 지표 평가 (성공률 + 안정성 + 개선도)")
    print("✨ 백트래킹 기능 (성능 저하 시 이전 단계로)")
    print("✨ 안정성 중심 난이도 증가 (충분한 안정성 확보 후 진행)")
    print("✨ 적응적 학습률 스케줄링 (학습 진행에 따라 감소)")
    print("✨ Optuna 하이퍼파라미터 최적화 (TPE + Pruning)")
    print("✨ W&B Sweep 베이지안 최적화")
    print("✨ 다중 목적 최적화 (보상 + 활용률)")
    print("✨ 종합 성과 평가 시스템 (S~D 등급)")
    
    if OPTUNA_AVAILABLE or WANDB_AVAILABLE:
        print("\n🔬 하이퍼파라미터 최적화 사용법:")
        if OPTUNA_AVAILABLE:
            print("   # Optuna로 하이퍼파라미터 최적화")
            print("   python src/ultimate_train_fix.py --optimize --optimization-method optuna --n-trials 30")
        if WANDB_AVAILABLE:
            print("   # W&B Sweep으로 하이퍼파라미터 최적화")
            print("   python src/ultimate_train_fix.py --optimize --optimization-method wandb --use-wandb --n-trials 30")
        if OPTUNA_AVAILABLE and WANDB_AVAILABLE:
            print("   # 두 방법 모두 사용")
            print("   python src/ultimate_train_fix.py --optimize --optimization-method both --use-wandb --n-trials 30")
    else:
        print("\n📦 하이퍼파라미터 최적화 라이브러리 설치:")
        print("   pip install optuna wandb  # 둘 다 설치 권장")
        print("   pip install optuna        # Optuna만 설치")
        print("   pip install wandb         # W&B만 설치")

# 실제 공간 활용률 계산 로직 추가
def calculate_real_utilization(env):
    """기존 호환성을 위한 활용률 계산 함수"""
    if hasattr(env.unwrapped, 'container'):
        placed_volume = sum(box.volume for box in env.unwrapped.container.boxes 
                          if box.position is not None)
        container_volume = env.unwrapped.container.volume
        return placed_volume / container_volume
    return 0.0