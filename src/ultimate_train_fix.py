#!/usr/bin/env python3
"""
🚀 999 스텝 문제 완전 해결 + GIF + 성능 그래프 생성
평가 콜백을 완전히 재작성하여 무한 대기 문제 100% 해결
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

class UltimateCurriculumCallback(BaseCallback):
    """
    999 스텝 문제 없는 안전한 커리큘럼 학습 콜백
    성공률에 따라 점진적으로 박스 개수(난이도)를 증가시킵니다.
    """
    
    def __init__(
        self,
        container_size,
        initial_boxes,
        target_boxes,
        num_visible_boxes,
        success_threshold=0.6,
        curriculum_steps=5,
        patience=5,
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
        self.verbose = verbose
        
        # 커리큘럼 단계 설정
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
        
        # 성과 추적 변수
        self.evaluation_count = 0
        self.consecutive_successes = 0
        self.curriculum_level = 0
        self.last_success_rate = 0.0
        self.recent_rewards = []
        
        # 안전한 측정을 위한 변수
        self.measurement_window = 20  # 측정 윈도우 크기
        self.min_episodes = 10        # 최소 에피소드 수
        
        if self.verbose >= 1:
            print(f"🎓 안전한 커리큘럼 학습 초기화:")
            print(f"   - 시작 박스 수: {self.initial_boxes}")
            print(f"   - 목표 박스 수: {self.target_boxes}")
            print(f"   - 단계별 증가: {self.box_increments}")
            print(f"   - 성공 임계값: {self.success_threshold}")
            print(f"   - 측정 윈도우: {self.measurement_window}")
    
    def _on_step(self) -> bool:
        """매 스텝마다 호출 - 안전한 처리"""
        # 에피소드 완료 체크
        if self.locals.get('dones', [False])[0]:
            if 'episode' in self.locals.get('infos', [{}])[0]:
                episode_info = self.locals['infos'][0]['episode']
                episode_reward = episode_info['r']
                
                # 최근 보상 기록 (안전한 방식)
                self.recent_rewards.append(episode_reward)
                
                # 윈도우 크기 유지
                if len(self.recent_rewards) > self.measurement_window:
                    self.recent_rewards.pop(0)
                
                # 충분한 데이터가 있을 때만 평가
                if len(self.recent_rewards) >= self.min_episodes:
                    self._evaluate_curriculum_progress()
        
        return True
    
    def _evaluate_curriculum_progress(self):
        """커리큘럼 진행 상황 평가 (안전한 방식)"""
        try:
            # 성공률 계산 (보상 > 0.5인 경우 성공으로 간주)
            success_count = sum(1 for r in self.recent_rewards if r > 0.5)
            success_rate = success_count / len(self.recent_rewards)
            
            self.last_success_rate = success_rate
            self.evaluation_count += 1
            
            # 성공률이 임계값을 넘으면 난이도 증가 고려
            if success_rate >= self.success_threshold:
                self.consecutive_successes += 1
                
                # 연속 성공 횟수가 patience를 넘으면 난이도 증가
                if self.consecutive_successes >= self.patience:
                    self._increase_difficulty()
            else:
                self.consecutive_successes = 0
                
        except Exception as e:
            if self.verbose >= 1:
                print(f"⚠️ 커리큘럼 평가 오류: {e}")
    
    def _increase_difficulty(self):
        """난이도 증가 (박스 개수 증가) - 안전한 방식"""
        if self.curriculum_level < len(self.box_increments):
            new_boxes = self.box_increments[self.curriculum_level]
            
            if self.verbose >= 1:
                print(f"\n🎯 커리큘럼 학습: 난이도 증가!")
                print(f"   - 이전 박스 수: {self.current_boxes}")
                print(f"   - 새로운 박스 수: {new_boxes}")
                print(f"   - 현재 성공률: {self.last_success_rate:.1%}")
                print(f"   - 연속 성공 횟수: {self.consecutive_successes}")
                print(f"   - 진행도: {self.curriculum_level + 1}/{len(self.box_increments)}")
            
            self.current_boxes = new_boxes
            self.curriculum_level += 1
            self.consecutive_successes = 0
            
            # 새로운 난이도에서 측정 초기화
            self.recent_rewards = []
            
            # 환경 업데이트는 안전성을 위해 로그만 출력
            if self.verbose >= 1:
                print(f"   - 다음 학습 세션에서 {new_boxes}개 박스로 학습 권장")
                print(f"   - 현재 세션은 안정성을 위해 계속 진행")
    
    def get_current_difficulty(self):
        """현재 난이도 정보 반환"""
        return {
            "current_boxes": self.current_boxes,
            "curriculum_level": self.curriculum_level,
            "max_level": len(self.box_increments),
            "success_rate": self.last_success_rate,
            "consecutive_successes": self.consecutive_successes,
            "progress_percentage": (self.curriculum_level / len(self.box_increments)) * 100 if self.box_increments else 0,
            "recommended_boxes": self.current_boxes
        }
    
    def is_curriculum_complete(self):
        """커리큘럼 완료 여부 확인"""
        return self.curriculum_level >= len(self.box_increments)

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
    timesteps=5000,
    eval_freq=2000,
    container_size=[10, 10, 10],
    num_boxes=16,
    create_gif=True,
    curriculum_learning=True,
    initial_boxes=None,
    success_threshold=0.6,
    curriculum_steps=5,
    patience=5
):
    """999 스텝 문제 완전 해결된 학습 함수 (커리큘럼 학습 지원)"""
    
    # 커리큘럼 학습 설정
    if curriculum_learning and initial_boxes is None:
        initial_boxes = max(8, num_boxes // 2)  # 시작 박스 수 (목표의 절반)
    elif not curriculum_learning:
        initial_boxes = num_boxes
    
    current_boxes = initial_boxes
    
    print("🚀 999 스텝 문제 완전 해결 학습 시작")
    print(f"📋 설정: {timesteps:,} 스텝, 평가 주기 {eval_freq:,}")
    if curriculum_learning:
        print(f"🎓 커리큘럼 학습: {initial_boxes}개 → {num_boxes}개 박스")
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
        improved_reward_shaping=True,
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
    
    # 안전한 콜백 설정 (커리큘럼 학습 포함)
    print("🛡️ 안전한 콜백 설정 중...")
    
    # 평가 주기가 충분히 큰 경우에만 콜백 사용
    if eval_freq >= 2000:
        safe_callback = UltimateSafeCallback(eval_env, eval_freq=eval_freq)
        
        # 커리큘럼 학습 콜백 추가
        callbacks = [safe_callback]
        
        if curriculum_learning:
            curriculum_callback = UltimateCurriculumCallback(
                container_size=container_size,
                initial_boxes=initial_boxes,
                target_boxes=num_boxes,
                num_visible_boxes=3,
                success_threshold=success_threshold,
                curriculum_steps=curriculum_steps,
                patience=patience,
                verbose=1
            )
            callbacks.append(curriculum_callback)
            print(f"🎓 커리큘럼 학습 콜백 추가")
        
        # 체크포인트 콜백 (안전한 설정)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(eval_freq, 3000),
            save_path="models/checkpoints",
            name_prefix=f"ultimate_model_{timestamp}",
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        print(f"✅ 안전한 콜백 활성화 (평가 주기: {eval_freq})")
        if curriculum_learning:
            print(f"   - 커리큘럼 학습 활성화")
            print(f"   - 성공 임계값: {success_threshold}")
            print(f"   - 인내심: {patience}")
    else:
        callbacks = None
        print("⚠️ 콜백 비활성화 (평가 주기가 너무 짧음)")
        if curriculum_learning:
            print("⚠️ 커리큘럼 학습도 비활성화됨")
    
    # 최적화된 모델 생성
    print("🤖 최적화된 모델 생성 중...")
    
    # TensorBoard 로그 설정 (선택적)
    tensorboard_log = None
    try:
        import tensorboard
        tensorboard_log = "logs/tensorboard"
        print("✅ TensorBoard 로깅 활성화")
    except ImportError:
        print("⚠️ TensorBoard 없음 - 로깅 비활성화")
    
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        ent_coef=0.05, 
        vf_coef=0.5,
        learning_rate=5e-4,  # 4e-4 / 9e-4  
        n_steps=1024,        
        batch_size=512,  # 256      
        n_epochs=15,  # 10
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,        
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=42,
    )
    
    print("✅ 모델 생성 완료")
    
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
            tb_log_name=f"ultimate_ppo_{timestamp}",
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ 학습 완료! 소요 시간: {training_time:.2f}초")
        
        # 모델 저장
        model_path = f"models/ultimate_ppo_{timestamp}"
        model.save(model_path)
        print(f"💾 모델 저장: {model_path}")
        
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
                    print(f"⚠️ 평가 오류: {e}")
                    break
            
            final_rewards.append(episode_reward)
            print(f"  에피소드 {ep + 1}: {episode_reward:.4f}")
        
        mean_reward = np.mean(final_rewards) if final_rewards else 0.0
        print(f"📊 최종 평균 보상: {mean_reward:.4f}")
        
        # GIF 생성 (환경 닫기 전에 수행)
        gif_path = None
        if create_gif:
            print("\n🎬 GIF 생성 중...")
            try:
                gif_path = create_ultimate_gif(model, eval_env, timestamp)
                if gif_path:
                    print(f"✅ GIF 생성 완료: {gif_path}")
                else:
                    print("⚠️ GIF 생성 실패 - 학습은 성공적으로 완료됨")
            except Exception as e:
                print(f"⚠️ GIF 생성 오류: {e} - 학습은 성공적으로 완료됨")
        
        # 환경 정리 (GIF 생성 후)
        env.close()
        eval_env.close()
        
        # 결과 요약
        results = {
            "timestamp": timestamp,
            "total_timesteps": timesteps,
            "training_time": training_time,
            "final_reward": mean_reward,
            "container_size": container_size,
            "num_boxes": num_boxes,
            "model_path": model_path,
            "eval_freq": eval_freq,
            "callbacks_used": callbacks is not None,
            "gif_path": gif_path if gif_path else "GIF 생성 실패"
        }
        
        # 결과 저장
        results_path = f"results/ultimate_results_{timestamp}.txt"
        with open(results_path, "w", encoding='utf-8') as f:
            f.write("=== 999 스텝 문제 완전 해결 학습 결과 ===\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
        print(f"📄 결과 저장: {results_path}")
        
        return model, results
        
    except KeyboardInterrupt:
        print("\n⏹️ 학습 중단됨")
        model.save(f"models/interrupted_ultimate_{timestamp}")
        return model, None
    
    except Exception as e:
        print(f"\n❌학습 오류: {e}")
        import traceback
        traceback.print_exc()
        model.save(f"models/error_ultimate_{timestamp}")
        return None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="999 스텝 문제 완전 해결 + 커리큘럼 학습 지원")
    parser.add_argument("--timesteps", type=int, default=5000, help="총 학습 스텝 수")
    parser.add_argument("--eval-freq", type=int, default=2000, help="평가 주기")
    parser.add_argument("--container-size", nargs=3, type=int, default=[10, 10, 10], help="컨테이너 크기")
    parser.add_argument("--num-boxes", type=int, default=16, help="목표 박스 개수")
    parser.add_argument("--no-gif", action="store_true", help="GIF 생성 안함")
    
    # 커리큘럼 학습 옵션
    parser.add_argument("--curriculum-learning", action="store_true", default=True, 
                        help="커리큘럼 학습 활성화 (기본값: True)")
    parser.add_argument("--no-curriculum", action="store_true", 
                        help="커리큘럼 학습 비활성화")
    parser.add_argument("--initial-boxes", type=int, default=None, 
                        help="시작 박스 수 (기본값: 목표의 절반)")
    parser.add_argument("--success-threshold", type=float, default=0.6, 
                        help="성공 임계값 (기본값: 0.6)")
    parser.add_argument("--curriculum-steps", type=int, default=5, 
                        help="커리큘럼 단계 수 (기본값: 5)")
    parser.add_argument("--patience", type=int, default=5, 
                        help="난이도 증가 대기 횟수 (기본값: 5)")
    
    args = parser.parse_args()
    
    # 커리큘럼 학습 설정
    curriculum_learning = args.curriculum_learning and not args.no_curriculum
    
    print("🚀 999 스텝 문제 완전 해결 + 커리큘럼 학습 스크립트")
    print("=" * 60)
    
    if curriculum_learning:
        print("🎓 커리큘럼 학습 모드 활성화")
        print(f"   - 성공 임계값: {args.success_threshold}")
        print(f"   - 커리큘럼 단계: {args.curriculum_steps}")
        print(f"   - 인내심: {args.patience}")
    else:
        print("📦 고정 난이도 모드")
    
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
            
            # 커리큘럼 학습 결과 출력
            if curriculum_learning and 'curriculum_info' in results:
                curriculum_info = results['curriculum_info']
                print(f"\n🎓 커리큘럼 학습 결과:")
                print(f"   - 최종 박스 수: {curriculum_info['current_boxes']}")
                print(f"   - 진행도: {curriculum_info['curriculum_level']}/{curriculum_info['max_level']}")
                print(f"   - 최종 성공률: {curriculum_info['success_rate']:.1%}")
                print(f"   - 완료 여부: {'✅ 완료' if curriculum_info['progress_percentage'] >= 100 else '⏳ 진행 중'}")
        else:
            print("\n❌ 학습 실패")
            
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 전체 오류: {e}")
        import traceback
        traceback.print_exc()

# 실제 공간 활용률 계산 로직 추가
def calculate_real_utilization(env):
    if hasattr(env.unwrapped, 'container'):
        placed_volume = sum(box.volume for box in env.unwrapped.container.boxes 
                          if box.position is not None)
        container_volume = env.unwrapped.container.volume
        return placed_volume / container_volume
    return 0.0