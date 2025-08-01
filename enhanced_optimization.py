#!/usr/bin/env python3
"""
Enhanced Optimization Script for 3D Bin Packing RL
Phase 4: 정밀 최적화를 통한 목표 18.57점 달성

Based on Phase 3 results: Current best 16.116 -> Target 18.57 (+13.2% needed)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.envs import PackingEnv, ImprovedRewardWrapper, ActionMaskWrapper
    from src.agents import create_ppo_agent
    from src.device_utils import get_device_info, optimize_for_device
except ImportError as e:
    print(f"❌ Import 오류: {e}")
    print("src 폴더와 필요한 모듈들이 있는지 확인하세요.")
    sys.exit(1)

class EnhancedOptimizer:
    """Phase 4: 정밀 최적화 클래스"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Phase 3 최고 성능 기준점
        self.phase3_best = {
            'score': 16.116,
            'params': {
                'learning_rate': 0.00015,
                'n_steps': 512,
                'batch_size': 128,
                'n_epochs': 4,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'gae_lambda': 0.95,
                'net_arch': [256, 128, 64]
            }
        }
        
        self.target_score = 18.57
        self.improvement_needed = (self.target_score - self.phase3_best['score']) / self.phase3_best['score']
        
        print(f"🎯 Phase 4 Enhanced Optimization 시작")
        print(f"📊 기준점: {self.phase3_best['score']:.3f}점")
        print(f"🏆 목표: {self.target_score}점 ({self.improvement_needed:.1%} 개선 필요)")
        
    def create_enhanced_environment(self, num_boxes: int = 12, container_size: List[int] = [10, 10, 10], 
                                  enhanced_reward: bool = True) -> gym.Env:
        """향상된 환경 생성"""
        try:
            print(f"생성된 박스 개수: {num_boxes}")
            print(f"컨테이너 크기: {container_size}")
            
            env = gym.make('PackingEnv-v0', 
                          container_size=container_size,
                          item_set_size=num_boxes,
                          data_name="small_set",
                          reward_type="C+P+S-add-min",
                          reward_params={})
            print("환경 생성 성공: PackingEnv-v0")
            
            if enhanced_reward:
                env = EnhancedRewardWrapper(env)
                print("강화된 보상 래퍼 적용됨")
            else:
                env = ImprovedRewardWrapper(env)
                print("개선된 보상 래퍼 적용됨")
                
            env = ActionMaskWrapper(env)
            print("액션 마스킹 래퍼 적용됨")
            
            return env
            
        except Exception as e:
            print(f"❌ 환경 생성 실패: {e}")
            raise e
    
    def get_enhanced_parameter_sets(self) -> Dict[str, Dict]:
        """강화된 파라미터 세트들"""
        
        # 1. 학습 안정성 강화 세트
        stability_sets = {
            'stability_conservative': {
                'learning_rate': 1.2e-04,  # 더 보수적
                'n_steps': 1024,           # 더 많은 경험
                'batch_size': 64,          # 안정적 업데이트
                'n_epochs': 6,             # 더 깊은 학습
                'clip_range': 0.15,        # 보수적 업데이트
                'ent_coef': 0.005,         # 탐색 감소
                'vf_coef': 0.5,
                'gae_lambda': 0.98,        # 장기 보상 중시
                'net_arch': [256, 128, 64]
            },
            'stability_balanced': {
                'learning_rate': 1.3e-04,
                'n_steps': 768,
                'batch_size': 96,
                'n_epochs': 5,
                'clip_range': 0.18,
                'ent_coef': 0.008,
                'vf_coef': 0.5,
                'gae_lambda': 0.96,
                'net_arch': [256, 128, 64]
            }
        }
        
        # 2. 고급 네트워크 아키텍처 세트
        architecture_sets = {
            'arch_wide': {
                'learning_rate': 1.5e-04,
                'n_steps': 512,
                'batch_size': 128,
                'n_epochs': 4,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'gae_lambda': 0.95,
                'net_arch': [512, 256, 128]  # 더 큰 첫 레이어
            },
            'arch_deep': {
                'learning_rate': 1.4e-04,
                'n_steps': 512,
                'batch_size': 128,
                'n_epochs': 4,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'gae_lambda': 0.95,
                'net_arch': [256, 256, 128, 64]  # 추가 레이어
            },
            'arch_balanced': {
                'learning_rate': 1.5e-04,
                'n_steps': 512,
                'batch_size': 128,
                'n_epochs': 4,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'gae_lambda': 0.95,
                'net_arch': [384, 192, 96]  # 균형 잡힌 감소
            },
            'arch_reinforced': {
                'learning_rate': 1.5e-04,
                'n_steps': 512,
                'batch_size': 128,
                'n_epochs': 4,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'gae_lambda': 0.95,
                'net_arch': [256, 128, 128, 64]  # 중간 레이어 강화
            }
        }
        
        # 3. 최적화된 하이퍼파라미터 세트
        optimized_sets = {
            'opt_precision': {
                'learning_rate': 1.1e-04,  # 정밀 학습
                'n_steps': 1536,           # 많은 경험
                'batch_size': 192,         # 큰 배치
                'n_epochs': 8,             # 깊은 학습
                'clip_range': 0.12,        # 매우 보수적
                'ent_coef': 0.003,         # 최소 탐색
                'vf_coef': 0.6,            # 가치 함수 중시
                'gae_lambda': 0.99,        # 최대 장기 보상
                'net_arch': [256, 128, 64]
            },
            'opt_aggressive': {
                'learning_rate': 1.8e-04,  # 적극적 학습
                'n_steps': 256,            # 빠른 업데이트
                'batch_size': 64,          # 작은 배치
                'n_epochs': 3,             # 빠른 에포크
                'clip_range': 0.25,        # 큰 업데이트
                'ent_coef': 0.02,          # 많은 탐색
                'vf_coef': 0.4,
                'gae_lambda': 0.92,
                'net_arch': [256, 128, 64]
            }
        }
        
        # 모든 세트 결합
        all_sets = {}
        all_sets.update(stability_sets)
        all_sets.update(architecture_sets)
        all_sets.update(optimized_sets)
        
        return all_sets
    
    def train_and_evaluate(self, params: Dict, name: str, timesteps: int = 35000, 
                          eval_episodes: int = 25, enhanced_reward: bool = True) -> Dict[str, Any]:
        """모델 훈련 및 평가"""
        print(f"\n🔧 {name} 최적화 중...")
        
        # 환경 생성
        env = self.create_enhanced_environment(enhanced_reward=enhanced_reward)
        env.seed(42)
        print(f"✅ 환경 생성 성공: 컨테이너{env.unwrapped.container_size}, 박스{env.unwrapped.item_set_size}개")
        
        # 모델 생성
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecMonitor(vec_env)
        
        model = PPO(
            'MlpPolicy',
            vec_env,
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            clip_range=params['clip_range'],
            ent_coef=params['ent_coef'],
            vf_coef=params['vf_coef'],
            gae_lambda=params['gae_lambda'],
            policy_kwargs={'net_arch': params['net_arch']},
            verbose=0,
            device='auto'
        )
        
        # 훈련
        print(f"🎓 {name} 학습 시작: {timesteps:,} 스텝 (LR: {params['learning_rate']:.2e}, Net: {params['net_arch']})")
        start_time = time.time()
        
        model.learn(total_timesteps=timesteps)
        
        training_time = time.time() - start_time
        print(f"⏱️ {name} 학습 완료: {training_time:.1f}초")
        
        # 평가
        print(f"🔍 {name} 평가 시작 ({eval_episodes} 에피소드, 최대 50 스텝)")
        
        eval_env = self.create_enhanced_environment(enhanced_reward=enhanced_reward)
        
        rewards = []
        utilizations = []
        placements = []
        
        for i in range(eval_episodes):
            eval_env.seed(100 + i * 5)
            print(f"✅ 환경 생성 성공: 컨테이너{eval_env.unwrapped.container_size}, 박스{eval_env.unwrapped.item_set_size}개")
            
            obs, _ = eval_env.reset()
            episode_reward = 0
            step_count = 0
            max_steps = 50
            
            while step_count < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                step_count += 1
                
                if terminated or truncated:
                    break
            
            # 통계 수집
            final_utilization = info.get('utilization', 0.0)
            placement_count = info.get('num_items', 0)
            
            rewards.append(episode_reward)
            utilizations.append(final_utilization)
            placements.append(placement_count)
            
            # 주요 에피소드만 출력
            if i < 6 or i in [10, 15, 20] or i == eval_episodes - 1:
                print(f"   에피소드 {i+1}: 보상={episode_reward:.3f}, 활용률={final_utilization:.1%}, 박스={placement_count}개")
        
        # 결과 계산
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_utilization = np.mean(utilizations)
        std_utilization = np.std(utilizations)
        mean_placement = np.mean(placements)
        max_placement = np.max(placements)
        
        # 성공률 (5개 이상 배치)
        success_count = sum(1 for p in placements if p >= 5)
        success_rate = success_count / eval_episodes
        
        # 종합 점수 계산 (Phase 3와 동일한 공식)
        combined_score = (mean_reward * 2.0) + (mean_utilization * 100 * 0.3) + (success_rate * 20)
        
        # 결과 출력
        print(f"📊 {name} 최종 결과:")
        print(f"   평균 보상: {mean_reward:.3f} ± {std_reward:.3f}")
        print(f"   평균 활용률: {mean_utilization:.1%} ± {std_utilization:.1%}")
        print(f"   평균 배치: {mean_placement:.1f}개 (최대: {max_placement}개)")
        print(f"   성공률: {success_rate:.1%}")
        print(f"   종합 점수: {combined_score:.3f}")
        
        # 환경 정리
        eval_env.close()
        vec_env.close()
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_utilization': mean_utilization,
            'std_utilization': std_utilization,
            'mean_placement': mean_placement,
            'max_placement': max_placement,
            'success_rate': success_rate,
            'combined_score': combined_score,
            'episodes': eval_episodes,
            'training_time': training_time,
            'params': params
        }
    
    def run_phase4_optimization(self, focus: str = 'all', timesteps: int = 35000) -> Dict:
        """Phase 4 최적화 실행"""
        print(f"\n{'='*60}")
        print(f"🚀 Phase 4 Enhanced Optimization 시작")
        print(f"🎯 포커스: {focus}")
        print(f"⏱️ 학습 스텝: {timesteps:,}")
        print(f"{'='*60}")
        
        all_params = self.get_enhanced_parameter_sets()
        results = {}
        best_score = 0
        best_config = None
        
        # 포커스에 따른 필터링
        if focus == 'stability':
            params_to_test = {k: v for k, v in all_params.items() if k.startswith('stability')}
        elif focus == 'architecture':
            params_to_test = {k: v for k, v in all_params.items() if k.startswith('arch')}
        elif focus == 'optimization':
            params_to_test = {k: v for k, v in all_params.items() if k.startswith('opt')}
        else:
            params_to_test = all_params
        
        print(f"📋 테스트할 설정: {len(params_to_test)}개")
        
        total_start_time = time.time()
        
        for i, (name, params) in enumerate(params_to_test.items(), 1):
            print(f"\n[{i}/{len(params_to_test)}] {name} 테스트 중...")
            
            try:
                result = self.train_and_evaluate(
                    params, name, timesteps=timesteps, 
                    enhanced_reward=True  # 강화된 보상 사용
                )
                results[name] = result
                
                # 최고 성능 업데이트
                if result['combined_score'] > best_score:
                    best_score = result['combined_score']
                    best_config = name
                    print(f"🏆 새로운 최고 성능: {best_score:.3f}점")
                
            except Exception as e:
                print(f"❌ {name} 실행 중 오류: {e}")
                continue
        
        total_time = time.time() - total_start_time
        
        # 결과 정리 및 출력
        if results:
            print(f"\n{'='*60}")
            print(f"🏆 Phase 4 최적화 결과")
            print(f"{'='*60}")
            
            # 점수 순으로 정렬
            sorted_results = sorted(results.items(), key=lambda x: x[1]['combined_score'], reverse=True)
            
            print("순위  설정명                    점수      개선율   활용률   성공률")
            print("-" * 70)
            
            for rank, (name, result) in enumerate(sorted_results[:10], 1):
                improvement = (result['combined_score'] - self.phase3_best['score']) / self.phase3_best['score'] * 100
                print(f"{rank:2d}    {name:<22} {result['combined_score']:6.2f}   {improvement:+5.1f}%   "
                      f"{result['mean_utilization']:5.1%}   {result['success_rate']:5.1%}")
            
            best_result = sorted_results[0][1]
            target_achievement = best_score / self.target_score * 100
            
            print(f"\n🏆 최고 성능: {best_score:.3f}점 ({best_config})")
            print(f"📈 목표 달성도: {target_achievement:.1f}% (목표 {self.target_score} 대비)")
            
            if best_score >= self.target_score:
                print(f"🎉 목표 달성 성공!")
            else:
                remaining = self.target_score - best_score
                print(f"📊 목표까지 {remaining:.3f}점 부족")
            
            # 결과 저장
            output_data = {
                'timestamp': self.timestamp,
                'phase': 'phase4_enhanced_optimization',
                'focus': focus,
                'timesteps': timesteps,
                'target_score': self.target_score,
                'phase3_baseline': self.phase3_best['score'],
                'best_score': best_score,
                'best_config': best_config,
                'target_achievement': target_achievement,
                'total_time_minutes': total_time / 60,
                'results': results
            }
            
            output_file = os.path.join(self.results_dir, f'phase4_enhanced_{focus}_{self.timestamp}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Phase 4 결과: {output_file}")
            print(f"⏱️ 총 소요 시간: {total_time/60:.1f}분")
            
            return output_data
        
        else:
            print("❌ 유효한 결과가 없습니다.")
            return {}
    
    def create_performance_analysis(self, results_file: str):
        """성능 분석 차트 생성"""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data.get('results'):
                print("분석할 결과가 없습니다.")
                return
            
            # 데이터 준비
            configs = []
            scores = []
            utilizations = []
            success_rates = []
            
            for name, result in data['results'].items():
                configs.append(name)
                scores.append(result['combined_score'])
                utilizations.append(result['mean_utilization'] * 100)
                success_rates.append(result['success_rate'] * 100)
            
            # 차트 생성
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Phase 4 Enhanced Optimization Analysis\n'
                        f'Best: {data["best_score"]:.2f} (Target: {data["target_score"]})', 
                        fontsize=16, fontweight='bold')
            
            # 1. 종합 점수
            axes[0,0].bar(range(len(configs)), scores, color='skyblue', alpha=0.7)
            axes[0,0].axhline(y=data['target_score'], color='red', linestyle='--', label=f'Target: {data["target_score"]}')
            axes[0,0].axhline(y=data['phase3_baseline'], color='orange', linestyle='--', label=f'Phase3: {data["phase3_baseline"]:.2f}')
            axes[0,0].set_title('Combined Scores')
            axes[0,0].set_ylabel('Score')
            axes[0,0].legend()
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # 2. 활용률
            axes[0,1].bar(range(len(configs)), utilizations, color='lightgreen', alpha=0.7)
            axes[0,1].set_title('Space Utilization (%)')
            axes[0,1].set_ylabel('Utilization %')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # 3. 성공률
            axes[1,0].bar(range(len(configs)), success_rates, color='lightcoral', alpha=0.7)
            axes[1,0].set_title('Success Rate (%)')
            axes[1,0].set_ylabel('Success Rate %')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # 4. 상관관계
            axes[1,1].scatter(utilizations, scores, alpha=0.7, s=100)
            axes[1,1].set_xlabel('Utilization %')
            axes[1,1].set_ylabel('Combined Score')
            axes[1,1].set_title('Utilization vs Score')
            
            # x축 라벨 설정
            for ax in axes.flat:
                if hasattr(ax, 'set_xticks'):
                    ax.set_xticks(range(len(configs)))
                    ax.set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in configs], 
                                     rotation=45, ha='right')
            
            plt.tight_layout()
            
            # 저장
            chart_file = results_file.replace('.json', '_analysis.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"📊 분석 차트 저장: {chart_file}")
            
            plt.close()
            
        except Exception as e:
            print(f"❌ 차트 생성 오류: {e}")


class EnhancedRewardWrapper(gym.RewardWrapper):
    """강화된 보상 래퍼 - 활용률과 배치 효율성 극대화"""
    
    def __init__(self, env):
        super().__init__(env)
        self.previous_utilization = 0.0
        self.consecutive_placements = 0
        
    def reward(self, reward):
        # 기본 정보 가져오기
        info = getattr(self.env, '_last_info', {})
        current_utilization = info.get('utilization', 0.0)
        placement_count = info.get('num_items', 0)
        
        # 1. 기본 보상
        enhanced_reward = reward
        
        # 2. 활용률 개선 보상 (제곱근 -> 1.5승으로 강화)
        if current_utilization > 0:
            util_bonus = (current_utilization ** 1.5) * 3.0
            enhanced_reward += util_bonus
        
        # 3. 활용률 증가 보상
        if current_utilization > self.previous_utilization:
            improvement_bonus = (current_utilization - self.previous_utilization) * 5.0
            enhanced_reward += improvement_bonus
        
        # 4. 연속 배치 보너스
        if placement_count > 0:
            self.consecutive_placements += 1
            consecutive_bonus = min(self.consecutive_placements * 0.1, 1.0)
            enhanced_reward += consecutive_bonus
        else:
            self.consecutive_placements = 0
        
        # 5. 임계값 돌파 보너스
        if current_utilization > 0.25:
            threshold_bonus = 2.0
            enhanced_reward += threshold_bonus
        elif current_utilization > 0.20:
            threshold_bonus = 1.0
            enhanced_reward += threshold_bonus
        
        # 6. 높은 배치 수 보너스
        if placement_count >= 5:
            placement_bonus = (placement_count - 4) * 0.5
            enhanced_reward += placement_bonus
        
        # 7. 공간 효율성 보너스 (박스당 활용률)
        if placement_count > 0:
            efficiency = current_utilization / placement_count
            efficiency_bonus = efficiency * 2.0
            enhanced_reward += efficiency_bonus
        
        self.previous_utilization = current_utilization
        
        return enhanced_reward


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Optimization for 3D Bin Packing')
    parser.add_argument('--focus', choices=['all', 'stability', 'architecture', 'optimization'], 
                       default='all', help='Optimization focus area')
    parser.add_argument('--timesteps', type=int, default=35000, help='Training timesteps')
    parser.add_argument('--analyze', type=str, help='Analyze results from JSON file')
    
    args = parser.parse_args()
    
    optimizer = EnhancedOptimizer()
    
    if args.analyze:
        # 결과 분석 모드
        optimizer.create_performance_analysis(args.analyze)
    else:
        # 최적화 실행 모드
        result = optimizer.run_phase4_optimization(focus=args.focus, timesteps=args.timesteps)
        
        if result and result.get('results'):
            # 자동으로 분석 차트 생성
            output_file = os.path.join(optimizer.results_dir, 
                                     f'phase4_enhanced_{args.focus}_{optimizer.timestamp}.json')
            optimizer.create_performance_analysis(output_file)
            
            # 다음 단계 권장사항
            best_score = result['best_score']
            target_score = result['target_score']
            
            if best_score >= target_score:
                print(f"\n🎉 축하합니다! 목표 {target_score}점을 달성했습니다!")
                print(f"🏆 최종 성능: {best_score:.3f}점")
            else:
                remaining = target_score - best_score
                print(f"\n📊 추가 개선 권장사항:")
                print(f"   목표까지 {remaining:.3f}점 부족")
                
                if remaining > 1.0:
                    print(f"   ➡️ 학습 시간을 50,000스텝으로 증가 시도")
                    print(f"   ➡️ 앙상블 모델링 시도")
                else:
                    print(f"   ➡️ 미세 조정으로 달성 가능")
                    print(f"   ➡️ 보상 함수 추가 최적화 권장")


if __name__ == "__main__":
    main() 