"""
Maskable PPO를 사용한 3D bin packing 학습 스크립트
GPU/CPU 자동 선택 기능 포함
실시간 모니터링 및 성능 지표 추적 기능 추가
"""
import os
import sys
import time
import warnings
import datetime
from typing import Optional, Dict, List
import threading
import queue

import gymnasium as gym
import numpy as np

# matplotlib 백엔드 설정 (GUI 환경이 없는 서버에서도 작동하도록)
import matplotlib
matplotlib.use('Agg')  # GUI가 없는 환경에서도 작동하는 백엔드
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.results_plotter import load_results, ts2xy

from src.utils import boxes_generator
from src.device_utils import setup_training_device, log_system_info, get_device

from plotly_gif import GIF
import io
from PIL import Image

# 경고 메시지 억제
warnings.filterwarnings("ignore")


class RealTimeMonitorCallback(BaseCallback):
    """
    실시간 모니터링을 위한 커스텀 콜백 클래스
    학습 진행 상황을 실시간으로 추적하고 시각화합니다.
    활용률, 안정성 지표 포함한 종합적인 성능 모니터링 제공
    """
    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=5, verbose=1, update_freq=1000):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.update_freq = update_freq  # 그래프 업데이트 주기 (더 빈번한 업데이트용)
        
        # 성능 지표 저장용 리스트
        self.timesteps = []
        self.episode_rewards = []
        self.mean_rewards = []
        self.eval_rewards = []
        self.eval_timesteps = []
        self.success_rates = []
        self.episode_lengths = []
        
        # 새로 추가: 활용률 및 안정성 지표
        self.utilization_rates = []  # 활용률 (컨테이너 공간 활용도)
        self.eval_utilization_rates = []  # 평가 시 활용률
        self.reward_stability = []  # 보상 안정성 (표준편차)
        self.utilization_stability = []  # 활용률 안정성
        self.learning_smoothness = []  # 학습 곡선 smoothness
        self.max_utilization_rates = []  # 최대 활용률 기록
        
        # 실시간 플롯 설정 (3x2 그리드로 확장)
        self.fig, self.axes = plt.subplots(3, 2, figsize=(16, 12))
        self.fig.suptitle('실시간 학습 진행 상황 - 성능 지표 종합 모니터링', fontsize=16)
        plt.ion()  # 인터랙티브 모드 활성화
        
        # 마지막 평가 및 업데이트 시점
        self.last_eval_time = 0
        self.last_update_time = 0
        
        # 에피소드별 통계
        self.current_episode_rewards = []
        self.current_episode_lengths = []
        self.current_episode_utilizations = []
        
        # 안정성 계산을 위한 윈도우 크기
        self.stability_window = 50
        
    def _on_training_start(self) -> None:
        """학습 시작 시 호출"""
        print("\n=== 실시간 모니터링 시작 ===")
        self.start_time = time.time()
        
        # 초기 플롯 설정
        self._setup_plots()
        
    def _on_step(self) -> bool:
        """매 스텝마다 호출"""
        # 에피소드 완료 체크
        if self.locals.get('dones', [False])[0]:
            # 에피소드 보상 및 활용률 기록
            if 'episode' in self.locals.get('infos', [{}])[0]:
                episode_info = self.locals['infos'][0]['episode']
                episode_reward = episode_info['r']
                episode_length = episode_info['l']
                
                # 활용률 계산 (보상이 곧 활용률이므로 동일)
                episode_utilization = max(0.0, episode_reward)  # 음수 보상 처리
                
                self.current_episode_rewards.append(episode_reward)
                self.current_episode_lengths.append(episode_length)
                self.current_episode_utilizations.append(episode_utilization)
                
                self.timesteps.append(self.num_timesteps)
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.utilization_rates.append(episode_utilization)
                
                # 실시간 출력 (더 자세한 정보 포함)
                if len(self.current_episode_rewards) % 10 == 0:
                    recent_rewards = self.current_episode_rewards[-10:]
                    recent_utilizations = self.current_episode_utilizations[-10:]
                    
                    mean_reward = np.mean(recent_rewards)
                    mean_utilization = np.mean(recent_utilizations)
                    max_utilization = np.max(recent_utilizations) if recent_utilizations else 0
                    elapsed_time = time.time() - self.start_time
                    
                    print(f"스텝: {self.num_timesteps:,} | "
                          f"에피소드: {len(self.episode_rewards)} | "
                          f"최근 10 에피소드 평균 보상: {mean_reward:.3f} | "
                          f"평균 활용률: {mean_utilization:.1%} | "
                          f"최대 활용률: {max_utilization:.1%} | "
                          f"경과 시간: {elapsed_time:.1f}초")
        
        # 빈번한 그래프 업데이트
        if self.num_timesteps - self.last_update_time >= self.update_freq:
            self._update_stability_metrics()
            self._quick_update_plots()
            self.last_update_time = self.num_timesteps
        
        # 주기적 평가 및 전체 플롯 업데이트
        if self.num_timesteps - self.last_eval_time >= self.eval_freq:
            self._perform_evaluation()
            self._update_plots()
            self.last_eval_time = self.num_timesteps
            
        return True
    
    def _update_stability_metrics(self):
        """안정성 지표 업데이트"""
        try:
            # 충분한 데이터가 있을 때만 계산
            if len(self.episode_rewards) >= self.stability_window:
                # 최근 윈도우의 데이터
                recent_rewards = self.episode_rewards[-self.stability_window:]
                recent_utilizations = self.utilization_rates[-self.stability_window:]
                
                # 보상 안정성 (표준편차)
                reward_std = np.std(recent_rewards)
                self.reward_stability.append(reward_std)
                
                # 활용률 안정성
                utilization_std = np.std(recent_utilizations)
                self.utilization_stability.append(utilization_std)
                
                # 학습 곡선 smoothness (연속된 값들의 차이의 평균)
                if len(recent_rewards) > 1:
                    reward_diffs = np.diff(recent_rewards)
                    smoothness = 1.0 / (1.0 + np.mean(np.abs(reward_diffs)))  # 0~1 범위
                    self.learning_smoothness.append(smoothness)
                
                # 최대 활용률 업데이트
                current_max_util = np.max(self.utilization_rates)
                self.max_utilization_rates.append(current_max_util)
                
        except Exception as e:
            if self.verbose > 0:
                print(f"안정성 지표 계산 중 오류: {e}")
    
    def _quick_update_plots(self):
        """빠른 플롯 업데이트 (일부 차트만)"""
        try:
            # 메인 성능 지표만 빠르게 업데이트
            if len(self.episode_rewards) > 10:
                # 보상 차트 업데이트
                self.axes[0, 0].clear()
                episodes = list(range(1, len(self.episode_rewards) + 1))
                self.axes[0, 0].plot(episodes, self.episode_rewards, 'b-', alpha=0.3, linewidth=0.8)
                
                # 이동평균
                if len(self.episode_rewards) >= 20:
                    window = min(50, len(self.episode_rewards) // 4)
                    moving_avg = []
                    for i in range(window-1, len(self.episode_rewards)):
                        avg = np.mean(self.episode_rewards[i-window+1:i+1])
                        moving_avg.append(avg)
                    self.axes[0, 0].plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'이동평균({window})')
                    self.axes[0, 0].legend()
                
                self.axes[0, 0].set_title('에피소드별 보상 (실시간)')
                self.axes[0, 0].set_xlabel('에피소드')
                self.axes[0, 0].set_ylabel('보상')
                self.axes[0, 0].grid(True, alpha=0.3)
                
                # 활용률 차트 업데이트
                self.axes[0, 1].clear()
                utilization_pct = [u * 100 for u in self.utilization_rates]
                self.axes[0, 1].plot(episodes, utilization_pct, 'g-', alpha=0.4, linewidth=0.8)
                
                # 활용률 이동평균
                if len(utilization_pct) >= 20:
                    window = min(30, len(utilization_pct) // 4)
                    moving_avg_util = []
                    for i in range(window-1, len(utilization_pct)):
                        avg = np.mean(utilization_pct[i-window+1:i+1])
                        moving_avg_util.append(avg)
                    self.axes[0, 1].plot(episodes[window-1:], moving_avg_util, 'darkgreen', linewidth=2, label=f'이동평균({window})')
                    self.axes[0, 1].legend()
                
                self.axes[0, 1].set_title('컨테이너 활용률 (실시간)')
                self.axes[0, 1].set_xlabel('에피소드')
                self.axes[0, 1].set_ylabel('활용률 (%)')
                self.axes[0, 1].set_ylim(0, 100)
                self.axes[0, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.draw()
                plt.pause(0.001)  # 매우 짧은 pause
                
        except Exception as e:
            if self.verbose > 0:
                print(f"빠른 플롯 업데이트 중 오류: {e}")
    
    def _perform_evaluation(self):
        """모델 평가 수행 (활용률 포함)"""
        try:
            print(f"\n평가 수행 중... (스텝: {self.num_timesteps:,})")
            
            # 평가 실행
            eval_rewards = []
            eval_utilizations = []
            success_count = 0
            
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                episode_reward = 0
                done = False
                truncated = False
                step_count = 0
                max_steps = 200
                
                while not (done or truncated) and step_count < max_steps:
                    action_masks = get_action_masks(self.eval_env)
                    action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    step_count += 1
                
                eval_rewards.append(episode_reward)
                
                # 활용률은 보상과 동일 (환경에서 활용률이 보상으로 사용됨)
                episode_utilization = max(0.0, episode_reward)
                eval_utilizations.append(episode_utilization)
                
                # 성공률 계산 (보상이 양수이면 성공으로 간주)
                if episode_reward > 0:
                    success_count += 1
            
            mean_eval_reward = np.mean(eval_rewards)
            mean_eval_utilization = np.mean(eval_utilizations)
            success_rate = success_count / self.n_eval_episodes
            
            # 결과 저장
            self.eval_rewards.append(mean_eval_reward)
            self.eval_utilization_rates.append(mean_eval_utilization)
            self.eval_timesteps.append(self.num_timesteps)
            self.success_rates.append(success_rate)
            
            print(f"평가 완료: 평균 보상 {mean_eval_reward:.3f}, "
                  f"평균 활용률 {mean_eval_utilization:.1%}, "
                  f"성공률 {success_rate:.1%}")
            
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            if self.verbose > 1:
                import traceback
                traceback.print_exc()
    
    def _setup_plots(self):
        """플롯 초기 설정 (3x2 그리드)"""
        # 상단 왼쪽: 에피소드별 보상
        self.axes[0, 0].set_title('에피소드별 보상')
        self.axes[0, 0].set_xlabel('에피소드')
        self.axes[0, 0].set_ylabel('보상')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # 상단 오른쪽: 컨테이너 활용률
        self.axes[0, 1].set_title('컨테이너 활용률')
        self.axes[0, 1].set_xlabel('에피소드')
        self.axes[0, 1].set_ylabel('활용률 (%)')
        self.axes[0, 1].set_ylim(0, 100)
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # 중단 왼쪽: 평가 성능 (보상 & 활용률)
        self.axes[1, 0].set_title('평가 성능')
        self.axes[1, 0].set_xlabel('학습 스텝')
        self.axes[1, 0].set_ylabel('평균 보상/활용률')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # 중단 오른쪽: 성공률
        self.axes[1, 1].set_title('성공률')
        self.axes[1, 1].set_xlabel('학습 스텝')
        self.axes[1, 1].set_ylabel('성공률 (%)')
        self.axes[1, 1].set_ylim(0, 100)
        self.axes[1, 1].grid(True, alpha=0.3)
        
        # 하단 왼쪽: 학습 안정성 지표
        self.axes[2, 0].set_title('학습 안정성')
        self.axes[2, 0].set_xlabel('학습 진행도')
        self.axes[2, 0].set_ylabel('안정성 지표')
        self.axes[2, 0].grid(True, alpha=0.3)
        
        # 하단 오른쪽: 에피소드 길이 & 최대 활용률
        self.axes[2, 1].set_title('에피소드 길이 & 최대 활용률')
        self.axes[2, 1].set_xlabel('에피소드')
        self.axes[2, 1].set_ylabel('길이 / 최대 활용률')
        self.axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)  # 서브플롯 간격 조정
    
    def _update_plots(self):
        """플롯 업데이트 (전체 6개 차트)"""
        try:
            # 모든 서브플롯 클리어
            for ax in self.axes.flat:
                ax.clear()
            
            # 플롯 재설정
            self._setup_plots()
            
            # 1. 에피소드별 보상 (상단 왼쪽)
            if self.episode_rewards:
                episodes = list(range(1, len(self.episode_rewards) + 1))
                self.axes[0, 0].plot(episodes, self.episode_rewards, 'b-', alpha=0.3, linewidth=0.5)
                
                # 이동 평균 (50 에피소드)
                if len(self.episode_rewards) >= 50:
                    moving_avg = []
                    for i in range(49, len(self.episode_rewards)):
                        avg = np.mean(self.episode_rewards[i-49:i+1])
                        moving_avg.append(avg)
                    self.axes[0, 0].plot(episodes[49:], moving_avg, 'r-', linewidth=2, label='이동평균(50)')
                    self.axes[0, 0].legend()
            
            # 2. 컨테이너 활용률 (상단 오른쪽)
            if self.utilization_rates:
                episodes = list(range(1, len(self.utilization_rates) + 1))
                utilization_pct = [u * 100 for u in self.utilization_rates]
                self.axes[0, 1].plot(episodes, utilization_pct, 'g-', alpha=0.4, linewidth=0.6)
                
                # 활용률 이동평균
                if len(utilization_pct) >= 30:
                    window = min(30, len(utilization_pct) // 4)
                    moving_avg_util = []
                    for i in range(window-1, len(utilization_pct)):
                        avg = np.mean(utilization_pct[i-window+1:i+1])
                        moving_avg_util.append(avg)
                    self.axes[0, 1].plot(episodes[window-1:], moving_avg_util, 'darkgreen', linewidth=2, label=f'이동평균({window})')
                    self.axes[0, 1].legend()
                
                # 목표선 추가
                self.axes[0, 1].axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='목표(80%)')
                if not any('목표' in str(h.get_label()) for h in self.axes[0, 1].get_children() if hasattr(h, 'get_label')):
                    self.axes[0, 1].legend()
            
            # 3. 평가 성능 (중단 왼쪽)
            if self.eval_rewards:
                # 평가 보상
                self.axes[1, 0].plot(self.eval_timesteps, self.eval_rewards, 'g-o', linewidth=2, markersize=4, label='평가 보상')
                self.axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
                
                # 평가 활용률 (있는 경우)
                if self.eval_utilization_rates:
                    eval_util_pct = [u * 100 for u in self.eval_utilization_rates]
                    ax2 = self.axes[1, 0].twinx()
                    ax2.plot(self.eval_timesteps, eval_util_pct, 'purple', marker='s', linewidth=2, markersize=3, label='평가 활용률(%)')
                    ax2.set_ylabel('활용률 (%)', color='purple')
                    ax2.set_ylim(0, 100)
                
                self.axes[1, 0].legend(loc='upper left')
            
            # 4. 성공률 (중단 오른쪽)
            if self.success_rates:
                success_percentages = [rate * 100 for rate in self.success_rates]
                self.axes[1, 1].plot(self.eval_timesteps, success_percentages, 'orange', linewidth=2, marker='s', markersize=4)
                self.axes[1, 1].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='목표(80%)')
                self.axes[1, 1].legend()
            
            # 5. 학습 안정성 지표 (하단 왼쪽)
            if len(self.reward_stability) > 0:
                stability_x = list(range(len(self.reward_stability)))
                
                # 보상 안정성 (표준편차)
                self.axes[2, 0].plot(stability_x, self.reward_stability, 'red', linewidth=2, label='보상 안정성', alpha=0.7)
                
                # 활용률 안정성
                if len(self.utilization_stability) > 0:
                    self.axes[2, 0].plot(stability_x, self.utilization_stability, 'blue', linewidth=2, label='활용률 안정성', alpha=0.7)
                
                # 학습 smoothness
                if len(self.learning_smoothness) > 0:
                    # 0~1 범위를 표준편차 범위에 맞게 스케일링
                    max_std = max(max(self.reward_stability), max(self.utilization_stability) if self.utilization_stability else 0)
                    scaled_smoothness = [s * max_std for s in self.learning_smoothness]
                    self.axes[2, 0].plot(stability_x, scaled_smoothness, 'green', linewidth=2, label='학습 smoothness', alpha=0.7)
                
                self.axes[2, 0].legend()
            
            # 6. 에피소드 길이 & 최대 활용률 (하단 오른쪽)
            if self.episode_lengths:
                episodes = list(range(1, len(self.episode_lengths) + 1))
                
                # 에피소드 길이
                self.axes[2, 1].plot(episodes, self.episode_lengths, 'purple', alpha=0.4, linewidth=0.5, label='에피소드 길이')
                
                # 에피소드 길이 이동평균
                if len(self.episode_lengths) >= 20:
                    window = min(20, len(self.episode_lengths) // 4)
                    moving_avg_lengths = []
                    for i in range(window-1, len(self.episode_lengths)):
                        avg = np.mean(self.episode_lengths[i-window+1:i+1])
                        moving_avg_lengths.append(avg)
                    self.axes[2, 1].plot(episodes[window-1:], moving_avg_lengths, 'darkred', linewidth=2, label=f'길이 이동평균({window})')
                
                # 최대 활용률 (있는 경우)
                if len(self.max_utilization_rates) > 0:
                    # 두 번째 y축 사용
                    ax3 = self.axes[2, 1].twinx()
                    max_util_pct = [u * 100 for u in self.max_utilization_rates[-len(episodes):]]  # 에피소드 수에 맞춤
                    ax3.plot(episodes[-len(max_util_pct):], max_util_pct, 'orange', linewidth=2, marker='*', markersize=3, label='최대 활용률(%)')
                    ax3.set_ylabel('최대 활용률 (%)', color='orange')
                    ax3.set_ylim(0, 100)
                
                self.axes[2, 1].legend(loc='upper left')
            
            # 플롯 업데이트
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            plt.draw()
            plt.pause(0.01)
            
            # 플롯 저장
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.fig.savefig(f'results/comprehensive_training_progress_{timestamp}.png', dpi=150, bbox_inches='tight')
            
        except Exception as e:
            print(f"플롯 업데이트 중 오류: {e}")
            if self.verbose > 1:
                import traceback
                traceback.print_exc()
    
    def _on_training_end(self) -> None:
        """학습 종료 시 호출"""
        total_time = time.time() - self.start_time
        print(f"\n=== 학습 완료! 총 소요 시간: {total_time:.1f}초 ===")
        
        # 최종 플롯 저장
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.fig.savefig(f'results/final_training_progress_{timestamp}.png', dpi=300, bbox_inches='tight')
        
        # 학습 통계 저장
        self._save_training_stats(timestamp)
        
        plt.ioff()  # 인터랙티브 모드 비활성화
        plt.close(self.fig)
    
    def _save_training_stats(self, timestamp):
        """학습 통계 저장 (활용률 및 안정성 지표 포함)"""
        stats = {
            # 기본 통계
            'total_episodes': len(self.episode_rewards),
            'total_timesteps': self.num_timesteps,
            'final_eval_reward': self.eval_rewards[-1] if self.eval_rewards else 0,
            'final_success_rate': self.success_rates[-1] if self.success_rates else 0,
            'best_eval_reward': max(self.eval_rewards) if self.eval_rewards else 0,
            'best_success_rate': max(self.success_rates) if self.success_rates else 0,
            
            # 활용률 통계
            'final_utilization_rate': self.utilization_rates[-1] if self.utilization_rates else 0,
            'best_utilization_rate': max(self.utilization_rates) if self.utilization_rates else 0,
            'average_utilization_rate': np.mean(self.utilization_rates) if self.utilization_rates else 0,
            'final_eval_utilization': self.eval_utilization_rates[-1] if self.eval_utilization_rates else 0,
            'best_eval_utilization': max(self.eval_utilization_rates) if self.eval_utilization_rates else 0,
            
            # 안정성 통계
            'final_reward_stability': self.reward_stability[-1] if self.reward_stability else 0,
            'final_utilization_stability': self.utilization_stability[-1] if self.utilization_stability else 0,
            'final_learning_smoothness': self.learning_smoothness[-1] if self.learning_smoothness else 0,
            'average_reward_stability': np.mean(self.reward_stability) if self.reward_stability else 0,
            'average_utilization_stability': np.mean(self.utilization_stability) if self.utilization_stability else 0,
            'average_learning_smoothness': np.mean(self.learning_smoothness) if self.learning_smoothness else 0,
            
            # 원시 데이터 (분석용)
            'episode_rewards': self.episode_rewards,
            'utilization_rates': self.utilization_rates,
            'eval_rewards': self.eval_rewards,
            'eval_utilization_rates': self.eval_utilization_rates,
            'eval_timesteps': self.eval_timesteps,
            'success_rates': self.success_rates,
            'episode_lengths': self.episode_lengths,
            'reward_stability': self.reward_stability,
            'utilization_stability': self.utilization_stability,
            'learning_smoothness': self.learning_smoothness,
            'max_utilization_rates': self.max_utilization_rates,
        }
        
        # 통계를 numpy 파일로 저장
        np.save(f'results/comprehensive_training_stats_{timestamp}.npy', stats)
        print(f"종합 학습 통계 저장 완료: comprehensive_training_stats_{timestamp}.npy")
        
        # 요약 텍스트 파일도 저장
        summary_path = f'results/comprehensive_summary_{timestamp}.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== 종합 학습 성과 요약 ===\n\n")
            
            f.write("📈 기본 성과 지표:\n")
            f.write(f"  • 총 에피소드: {stats['total_episodes']:,}\n")
            f.write(f"  • 총 학습 스텝: {stats['total_timesteps']:,}\n")
            f.write(f"  • 최종 평가 보상: {stats['final_eval_reward']:.3f}\n")
            f.write(f"  • 최고 평가 보상: {stats['best_eval_reward']:.3f}\n")
            f.write(f"  • 최종 성공률: {stats['final_success_rate']:.1%}\n")
            f.write(f"  • 최고 성공률: {stats['best_success_rate']:.1%}\n\n")
            
            f.write("🎯 활용률 성과:\n")
            f.write(f"  • 최종 활용률: {stats['final_utilization_rate']:.1%}\n")
            f.write(f"  • 최고 활용률: {stats['best_utilization_rate']:.1%}\n")
            f.write(f"  • 평균 활용률: {stats['average_utilization_rate']:.1%}\n")
            f.write(f"  • 최종 평가 활용률: {stats['final_eval_utilization']:.1%}\n")
            f.write(f"  • 최고 평가 활용률: {stats['best_eval_utilization']:.1%}\n\n")
            
            f.write("⚖️ 학습 안정성:\n")
            f.write(f"  • 최종 보상 안정성: {stats['final_reward_stability']:.3f}\n")
            f.write(f"  • 최종 활용률 안정성: {stats['final_utilization_stability']:.3f}\n")
            f.write(f"  • 최종 학습 smoothness: {stats['final_learning_smoothness']:.3f}\n")
            f.write(f"  • 평균 보상 안정성: {stats['average_reward_stability']:.3f}\n")
            f.write(f"  • 평균 활용률 안정성: {stats['average_utilization_stability']:.3f}\n")
            f.write(f"  • 평균 학습 smoothness: {stats['average_learning_smoothness']:.3f}\n\n")
            
            # 성과 등급 평가
            f.write("🏆 성과 등급:\n")
            if stats['best_utilization_rate'] >= 0.8:
                f.write("  • 활용률: 🥇 우수 (80% 이상)\n")
            elif stats['best_utilization_rate'] >= 0.6:
                f.write("  • 활용률: 🥈 양호 (60-80%)\n")
            else:
                f.write("  • 활용률: 🥉 개선 필요 (60% 미만)\n")
                
            if stats['best_success_rate'] >= 0.8:
                f.write("  • 성공률: 🥇 우수 (80% 이상)\n")
            elif stats['best_success_rate'] >= 0.5:
                f.write("  • 성공률: 🥈 양호 (50-80%)\n")
            else:
                f.write("  • 성공률: 🥉 개선 필요 (50% 미만)\n")
                
            if stats['average_learning_smoothness'] >= 0.7:
                f.write("  • 학습 안정성: 🥇 매우 안정적\n")
            elif stats['average_learning_smoothness'] >= 0.5:
                f.write("  • 학습 안정성: 🥈 안정적\n")
            else:
                f.write("  • 학습 안정성: 🥉 불안정\n")
        
        print(f"학습 성과 요약 저장 완료: {summary_path}")


def create_live_dashboard(stats_file):
    """
    저장된 학습 통계를 사용하여 실시간 대시보드 생성
    """
    try:
        stats = np.load(stats_file, allow_pickle=True).item()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('학습 성과 대시보드', fontsize=16)
        
        # 1. 에피소드 보상 추이
        if stats['episode_rewards']:
            episodes = list(range(1, len(stats['episode_rewards']) + 1))
            axes[0, 0].plot(episodes, stats['episode_rewards'], 'b-', alpha=0.3, linewidth=0.5)
            
            # 이동 평균
            window = min(50, len(stats['episode_rewards']) // 10)
            if len(stats['episode_rewards']) >= window:
                moving_avg = []
                for i in range(window-1, len(stats['episode_rewards'])):
                    avg = np.mean(stats['episode_rewards'][i-window+1:i+1])
                    moving_avg.append(avg)
                axes[0, 0].plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'이동평균({window})')
                axes[0, 0].legend()
            
            axes[0, 0].set_title('에피소드별 보상')
            axes[0, 0].set_xlabel('에피소드')
            axes[0, 0].set_ylabel('보상')
            axes[0, 0].grid(True)
        
        # 2. 평가 성능
        if stats['eval_rewards']:
            axes[0, 1].plot(stats['eval_timesteps'], stats['eval_rewards'], 'g-o', linewidth=2, markersize=6)
            axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('평가 성능')
            axes[0, 1].set_xlabel('학습 스텝')
            axes[0, 1].set_ylabel('평균 보상')
            axes[0, 1].grid(True)
        
        # 3. 성공률
        if stats['success_rates']:
            success_percentages = [rate * 100 for rate in stats['success_rates']]
            axes[1, 0].plot(stats['eval_timesteps'], success_percentages, 'orange', linewidth=2, marker='s', markersize=6)
            axes[1, 0].set_ylim(0, 100)
            axes[1, 0].set_title('성공률')
            axes[1, 0].set_xlabel('학습 스텝')
            axes[1, 0].set_ylabel('성공률 (%)')
            axes[1, 0].grid(True)
        
        # 4. 학습 통계 요약
        axes[1, 1].axis('off')
        summary_text = f"""
학습 통계 요약:
• 총 에피소드: {stats['total_episodes']:,}
• 총 학습 스텝: {stats['total_timesteps']:,}
• 최종 평가 보상: {stats['final_eval_reward']:.2f}
• 최종 성공률: {stats['final_success_rate']:.1%}
• 최고 평가 보상: {stats['best_eval_reward']:.2f}
• 최고 성공률: {stats['best_success_rate']:.1%}
        """
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"대시보드 생성 중 오류: {e}")
        return None


def analyze_training_performance(stats_file):
    """
    학습 성과 분석 및 개선 제안
    """
    try:
        stats = np.load(stats_file, allow_pickle=True).item()
        
        print("\n=== 학습 성과 분석 ===")
        
        # 기본 통계
        print(f"총 에피소드: {stats['total_episodes']:,}")
        print(f"총 학습 스텝: {stats['total_timesteps']:,}")
        print(f"최종 평가 보상: {stats['final_eval_reward']:.3f}")
        print(f"최종 성공률: {stats['final_success_rate']:.1%}")
        print(f"최고 평가 보상: {stats['best_eval_reward']:.3f}")
        print(f"최고 성공률: {stats['best_success_rate']:.1%}")
        
        # 학습 안정성 분석
        if len(stats['eval_rewards']) > 1:
            eval_rewards = np.array(stats['eval_rewards'])
            reward_trend = np.polyfit(range(len(eval_rewards)), eval_rewards, 1)[0]
            
            print(f"\n=== 학습 안정성 분석 ===")
            print(f"평가 보상 추세: {reward_trend:.4f} (양수면 개선, 음수면 악화)")
            
            # 변동성 분석
            if len(eval_rewards) > 2:
                reward_std = np.std(eval_rewards)
                reward_mean = np.mean(eval_rewards)
                cv = reward_std / abs(reward_mean) if reward_mean != 0 else float('inf')
                print(f"보상 변동성 (CV): {cv:.3f} (낮을수록 안정적)")
        
        # 성공률 분석
        if stats['success_rates']:
            success_rates = np.array(stats['success_rates'])
            final_success_rate = success_rates[-1]
            
            print(f"\n=== 성공률 분석 ===")
            if final_success_rate > 0.8:
                print("✅ 성공률이 매우 높습니다 (80% 이상)")
            elif final_success_rate > 0.5:
                print("⚠️ 성공률이 보통입니다 (50-80%)")
            else:
                print("❌ 성공률이 낮습니다 (50% 미만)")
        
        # 개선 제안
        print(f"\n=== 개선 제안 ===")
        if stats['final_eval_reward'] < 0:
            print("• 보상이 음수입니다. 보상 함수 조정을 고려해보세요.")
        if stats['final_success_rate'] < 0.5:
            print("• 성공률이 낮습니다. 학습률 조정이나 더 긴 학습을 고려해보세요.")
        if len(stats['eval_rewards']) > 2:
            recent_rewards = stats['eval_rewards'][-3:]
            if all(r1 >= r2 for r1, r2 in zip(recent_rewards[:-1], recent_rewards[1:])):
                print("• 최근 성능이 감소하고 있습니다. 과적합 가능성을 확인해보세요.")
        
        return stats
        
    except Exception as e:
        print(f"성과 분석 중 오류: {e}")
        return None


def make_env(
    container_size,
    num_boxes=64,
    num_visible_boxes=1,
    seed=0,
    render_mode=None,
    random_boxes=False,
    only_terminal_reward=False,
):
    """
    환경 생성 함수 (기존 코드 유지)
    
    Parameters
    ----------
    container_size: size of the container
    num_boxes: number of boxes to be packed
    num_visible_boxes: number of boxes visible to the agent
    seed: seed for RNG
    render_mode: render mode for the environment
    random_boxes: whether to use random boxes or not
    only_terminal_reward: whether to use only terminal reward or not
    """
    env = gym.make(
        "PackingEnv-v0",
        container_size=container_size,
        box_sizes=boxes_generator(container_size, num_boxes, seed),
        num_visible_boxes=num_visible_boxes,
        render_mode=render_mode,
        random_boxes=random_boxes,
        only_terminal_reward=only_terminal_reward,
    )
    return env


def train_and_evaluate(
    container_size=[10, 10, 10],
    num_boxes=64,
    num_visible_boxes=3,
    total_timesteps=100000,
    eval_freq=10000,
    seed=42,
    force_cpu=False,
    save_gif=True,
):
    """
    Maskable PPO 학습 및 평가 함수
    
    Args:
        container_size: 컨테이너 크기
        num_boxes: 박스 개수
        num_visible_boxes: 가시 박스 개수
        total_timesteps: 총 학습 스텝 수
        eval_freq: 평가 주기
        seed: 랜덤 시드
        force_cpu: CPU 강제 사용 여부
        save_gif: GIF 저장 여부
    """
    
    print("=== Maskable PPO 3D Bin Packing 학습 시작 ===")
    log_system_info()
    
    # 디바이스 설정
    device_config = setup_training_device(verbose=True)
    device = get_device(force_cpu=force_cpu)
    
    # 타임스탬프
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 디렉토리 생성
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("gifs", exist_ok=True)
    
    # 학습용 환경 생성
    env = make_env(
        container_size=container_size,
        num_boxes=num_boxes,
        num_visible_boxes=num_visible_boxes,
        seed=seed,
        render_mode=None,
        random_boxes=False,
        only_terminal_reward=False,
    )
    
    # 환경 체크
    print("환경 유효성 검사 중...")
    check_env(env, warn=True)
    
    # 평가용 환경 생성
    eval_env = make_env(
        container_size=container_size,
        num_boxes=num_boxes,
        num_visible_boxes=num_visible_boxes,
        seed=seed+1,
        render_mode="human" if save_gif else None,
        random_boxes=False,
        only_terminal_reward=False,
    )
    
    # 모니터링 설정
    env = Monitor(env, f"logs/training_monitor_{timestamp}.csv")
    eval_env = Monitor(eval_env, f"logs/eval_monitor_{timestamp}.csv")
    
    print(f"환경 설정 완료:")
    print(f"  - 컨테이너 크기: {container_size}")
    print(f"  - 박스 개수: {num_boxes}")
    print(f"  - 가시 박스 개수: {num_visible_boxes}")
    print(f"  - 액션 스페이스: {env.action_space}")
    print(f"  - 관찰 스페이스: {env.observation_space}")
    
    # 콜백 설정
    callbacks = []
    
    # 실시간 모니터링 콜백 (새로 추가)
    monitor_callback = RealTimeMonitorCallback(
        eval_env=eval_env,
        eval_freq=max(eval_freq // 2, 2000),  # 평가 주기를 더 짧게 설정
        n_eval_episodes=5,
        verbose=1,
        update_freq=max(eval_freq // 10, 1000)  # 빠른 그래프 업데이트 주기
    )
    callbacks.append(monitor_callback)
    
    # 평가 콜백 (기존 유지)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best_model",
        log_path="logs/eval_logs",
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    
    # 체크포인트 콜백 (기존 유지)
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path="models/checkpoints",
        name_prefix=f"rl_model_{timestamp}",
    )
    callbacks.append(checkpoint_callback)
    
    # 모델 생성 (기존 정책 유지)
    print("\n=== 모델 생성 중 ===")
    model = MaskablePPO(
        "MultiInputPolicy",  # 기존 코드의 정책 유지
        env,
        learning_rate=device_config["learning_rate"],
        n_steps=device_config["n_steps"],
        batch_size=device_config["batch_size"],
        n_epochs=device_config["n_epochs"],
        verbose=1,
        tensorboard_log="logs/tensorboard",
        device=str(device),
        seed=seed,
    )
    
    print(f"모델 파라미터:")
    print(f"  - 정책: MultiInputPolicy")
    print(f"  - 학습률: {device_config['learning_rate']}")
    print(f"  - 배치 크기: {device_config['batch_size']}")
    print(f"  - 스텝 수: {device_config['n_steps']}")
    print(f"  - 에포크 수: {device_config['n_epochs']}")
    print(f"  - 디바이스: {device}")
    
    # 학습 시작
    print(f"\n=== 학습 시작 (총 {total_timesteps:,} 스텝) ===")
    print(f"실시간 모니터링 활성화 - 매 {max(eval_freq // 2, 2000):,} 스텝마다 평가 및 플롯 업데이트")
    print(f"TensorBoard 로그: tensorboard --logdir=logs/tensorboard")
    print(f"시작 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name=f"maskable_ppo_{timestamp}",  # TensorBoard 로그 이름 지정
        )
        
        training_time = time.time() - start_time
        print(f"\n학습 완료! 소요 시간: {training_time:.2f}초")
        
        # 모델 저장
        model_path = f"models/ppo_mask_{timestamp}"
        model.save(model_path)
        print(f"모델 저장 완료: {model_path}")
        
        # 최종 평가
        print("\n=== 최종 모델 평가 ===")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        print(f"평균 보상: {mean_reward:.4f} ± {std_reward:.4f}")
        
        # GIF 생성 (기존 코드 스타일 유지)
        if save_gif:
            print("\n=== GIF 생성 중 ===")
            create_demonstration_gif(model, eval_env, timestamp)
        
        # 결과 저장
        results = {
            "timestamp": timestamp,
            "total_timesteps": total_timesteps,
            "training_time": training_time,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "container_size": container_size,
            "num_boxes": num_boxes,
            "device": str(device),
            "model_path": model_path,
        }
        
        # 결과를 텍스트 파일로 저장
        results_path = f"results/training_results_{timestamp}.txt"
        with open(results_path, "w") as f:
            f.write("=== Maskable PPO 3D Bin Packing 학습 결과 ===\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
        print(f"결과 저장 완료: {results_path}")
        
        # 학습 통계 파일 확인 및 성과 분석
        stats_file = f"results/training_stats_{timestamp}.npy"
        if os.path.exists(stats_file):
            print("\n=== 자동 성과 분석 시작 ===")
            analyze_training_performance(stats_file)
            
            # 최종 대시보드 생성
            dashboard_fig = create_live_dashboard(stats_file)
            if dashboard_fig:
                dashboard_path = f"results/final_dashboard_{timestamp}.png"
                dashboard_fig.savefig(dashboard_path, dpi=300, bbox_inches='tight')
                print(f"최종 대시보드 저장: {dashboard_path}")
                plt.close(dashboard_fig)
        
        return model, results
        
    except KeyboardInterrupt:
        print("\n학습이 중단되었습니다.")
        model.save(f"models/interrupted_model_{timestamp}")
        return model, None
    
    except Exception as e:
        print(f"\n학습 중 오류 발생: {e}")
        model.save(f"models/error_model_{timestamp}")
        raise e


def create_demonstration_gif(model, env, timestamp):
    """
    학습된 모델의 시연 GIF 생성 (기존 코드 스타일 유지)
    """
    try:
        # Gymnasium API: env.reset()는 (obs, info) 튜플을 반환
        obs, info = env.reset()
        done = False
        truncated = False
        figs = []
        step_count = 0
        max_steps = 100  # 무한 루프 방지
        
        print("시연 시작...")
        
        while not (done or truncated) and step_count < max_steps:
            # 액션 마스크 가져오기
            action_masks = get_action_masks(env)
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            # Gymnasium API: env.step()는 (obs, reward, terminated, truncated, info)를 반환
            obs, rewards, done, truncated, info = env.step(action)
            
            # 렌더링
            fig = env.render()
            if fig is not None:
                fig_png = fig.to_image(format="png")
                buf = io.BytesIO(fig_png)
                img = Image.open(buf)
                figs.append(img)
            
            step_count += 1
        
        print(f"시연 완료 (총 {step_count} 스텝)")
        
        # GIF 저장 (기존 코드 스타일 유지)
        if figs:
            gif_filename = f'trained_maskable_ppo_{timestamp}.gif'
            gif_path = f'gifs/{gif_filename}'
            
            figs[0].save(gif_path, format='GIF', 
                        append_images=figs[1:],
                        save_all=True, 
                        duration=500, 
                        loop=0)
            print(f"GIF 저장 완료: {gif_filename}")
        else:
            print("렌더링된 프레임이 없습니다.")
            
    except Exception as e:
        print(f"GIF 생성 중 오류 발생: {e}")


def evaluate_model(model_path, num_episodes=5):
    """
    저장된 모델 평가 함수 (kamp_auto_run.sh에서 호출)
    """
    try:
        print(f"모델 평가 시작: {model_path}")
        
        # 모델 로드
        model = MaskablePPO.load(model_path)
        
        # 평가용 환경 생성
        eval_env = make_env(
            container_size=[10, 10, 10],
            num_boxes=64,
            num_visible_boxes=3,
            seed=42,
            render_mode=None,
            random_boxes=False,
            only_terminal_reward=False,
        )
        
        # 평가 실행
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=num_episodes, deterministic=True
        )
        
        print(f"평가 완료 - 평균 보상: {mean_reward:.4f} ± {std_reward:.4f}")
        
        # 환경 정리
        eval_env.close()
        
        return mean_reward, std_reward
        
    except Exception as e:
        print(f"모델 평가 중 오류: {e}")
        return None, None


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Maskable PPO 3D Bin Packing")
    parser.add_argument("--timesteps", type=int, default=100000, help="총 학습 스텝 수")
    parser.add_argument("--container-size", nargs=3, type=int, default=[10, 10, 10], help="컨테이너 크기")
    parser.add_argument("--num-boxes", type=int, default=64, help="박스 개수")
    parser.add_argument("--visible-boxes", type=int, default=3, help="가시 박스 개수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--force-cpu", action="store_true", help="CPU 강제 사용")
    parser.add_argument("--no-gif", action="store_true", help="GIF 생성 비활성화")
    parser.add_argument("--eval-freq", type=int, default=10000, help="평가 주기")
    parser.add_argument("--analyze-only", type=str, help="학습 통계 파일 분석만 수행 (파일 경로 지정)")
    parser.add_argument("--dashboard-only", type=str, help="대시보드만 생성 (학습 통계 파일 경로 지정)")
    
    args = parser.parse_args()
    
    # 분석 전용 모드
    if args.analyze_only:
        if os.path.exists(args.analyze_only):
            print(f"학습 통계 파일 분석: {args.analyze_only}")
            analyze_training_performance(args.analyze_only)
        else:
            print(f"파일을 찾을 수 없습니다: {args.analyze_only}")
        return
    
    # 대시보드 전용 모드
    if args.dashboard_only:
        if os.path.exists(args.dashboard_only):
            print(f"대시보드 생성: {args.dashboard_only}")
            dashboard_fig = create_live_dashboard(args.dashboard_only)
            if dashboard_fig:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dashboard_path = f"results/dashboard_{timestamp}.png"
                dashboard_fig.savefig(dashboard_path, dpi=300, bbox_inches='tight')
                print(f"대시보드 저장: {dashboard_path}")
                plt.show()  # 대시보드 표시
                plt.close(dashboard_fig)
        else:
            print(f"파일을 찾을 수 없습니다: {args.dashboard_only}")
        return
    
    # 기존 코드 스타일로 간단한 테스트도 실행 가능
    if args.timesteps <= 100:
        print("=== 간단 테스트 모드 ===")
        # 기존 train.py의 간단한 테스트 코드 스타일
        container_size = args.container_size
        box_sizes = [[3, 3, 3], [3, 2, 3], [3, 4, 2], [3, 2, 4], [3, 2, 3]]
        
        env = gym.make(
            "PackingEnv-v0",
            container_size=container_size,
            box_sizes=box_sizes,
            num_visible_boxes=1,
            render_mode="human",
            random_boxes=False,
            only_terminal_reward=False,
        )
        
        model = MaskablePPO("MultiInputPolicy", env, verbose=1)
        print("간단 학습 시작")
        model.learn(total_timesteps=args.timesteps)
        print("간단 학습 완료")
        model.save("models/ppo_mask_simple")
        
    else:
        # 전체 학습 실행
        train_and_evaluate(
            container_size=args.container_size,
            num_boxes=args.num_boxes,
            num_visible_boxes=args.visible_boxes,
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            seed=args.seed,
            force_cpu=args.force_cpu,
            save_gif=not args.no_gif,
        )


if __name__ == "__main__":
    main() 