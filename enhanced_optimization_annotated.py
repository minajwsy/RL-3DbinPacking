#!/usr/bin/env python3
"""
3D Bin Packing 강화학습 최적화(Phase 4) 주석 확장판

개요
- 본 스크립트는 Maskable PPO를 활용해 3D 빈패킹 환경(`src/packing_env.py`)에서
  다양한 하이퍼파라미터 구성의 성능을 비교·평가한다.
- 관찰(Observation)은 컨테이너 상단 높이맵과 가시 박스의 크기 벡터이며,
  행동(Action)은 [가시 박스 선택 × XY 좌표]의 단일 정수 index로 디스크리트화된다.
- 논문 맥락: 첨부 문헌(Transformer 기반 DRL)과 달리 본 코드는 MLP 기반 MaskablePPO를 사용하지만,
  공통적으로 높이맵 기반 상태표현, 불가능행동 마스킹, 보상 쉐이핑을 통해 탐색 공간을 줄이고 안정적 학습을 지향한다.

사용 방법(예)
- 모든 세트 실험:  python enhanced_optimization_annotated.py --focus all --timesteps 35000
- 안정성 중심:     python enhanced_optimization_annotated.py --focus stability
- 결과 분석만:     python enhanced_optimization_annotated.py --analyze results/phase4_enhanced_*.json

출력
- `results/phase4_enhanced_{focus}_{timestamp}.json` 종합 결과와 `_analysis.png` 차트를 자동 저장
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings

# 환경 설정: 서버/헤드리스 환경에서도 안전하게 실행
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings("ignore")

# src 폴더를 import 경로에 추가
sys.path.append('src')

import gymnasium as gym
from gymnasium.envs.registration import register
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import matplotlib.pyplot as plt

# 프로젝트 핵심 모듈 로드
try:
    from packing_env import PackingEnv                    # 상태·행동 정의된 Gym 환경
    from train_maskable_ppo import ImprovedRewardWrapper  # 개선형 보상 래퍼
    from utils import boxes_generator                     # 문제 인스턴스(박스 시퀀스) 생성기
    print("✅ 모든 모듈 import 성공")
except ImportError as e:
    print(f"❌ Import 오류: {e}")
    print("src 폴더와 필요한 모듈들이 있는지 확인하세요.")
    sys.exit(1)


def get_env_info(env):
    """래퍼를 벗겨 실제 `PackingEnv`에 접근해 컨테이너 크기/박스 개수 요약.

    반환값
    - (container_size, box_count)
    """
    try:
        unwrapped_env = env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env

        if hasattr(unwrapped_env, 'container') and hasattr(unwrapped_env.container, 'size'):
            container_size = unwrapped_env.container.size
        else:
            container_size = [10, 10, 10]

        if hasattr(unwrapped_env, 'initial_boxes'):
            box_count = len(unwrapped_env.initial_boxes)
        elif hasattr(unwrapped_env, 'num_initial_boxes'):
            box_count = unwrapped_env.num_initial_boxes
        else:
            box_count = 12

        return container_size, box_count
    except Exception as e:
        print(f"⚠️ 환경 정보 가져오기 실패: {e}")
        return [10, 10, 10], 12


def calculate_utilization_and_items(env):
    """최종 시점 컨테이너 활용률과 배치된 박스 개수 산출.

    - 활용률 = (배치된 박스 총부피) / (컨테이너 부피)
    - 배치 판정: position이 [-1,-1,-1]이 아닌 박스
    """
    try:
        unwrapped_env = env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env

        if hasattr(unwrapped_env, 'container'):
            placed_volume = 0
            placed_count = 0
            for box in unwrapped_env.container.boxes:
                if hasattr(box, 'position') and box.position is not None:
                    if not (box.position[0] == -1 and box.position[1] == -1 and box.position[2] == -1):
                        placed_volume += box.volume
                        placed_count += 1

            container_volume = unwrapped_env.container.volume
            utilization = placed_volume / container_volume if container_volume > 0 else 0.0
            return utilization, placed_count
        else:
            return 0.0, 0
    except Exception as e:
        print(f"⚠️ 활용률 계산 실패: {e}")
        return 0.0, 0


class EnhancedOptimizer:
    """Phase 4 정밀 최적화 오케스트레이터.

    - 여러 하이퍼파라미터 세트를 순회 학습/평가하여 종합 점수 기준으로 비교
    - 점수 = 0.3×평균 보상 + 0.7×(평균 활용률[%])
    """

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        # Phase 3 최고 성능 기준점 (비교·개선율 계산에 사용)
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

    def create_enhanced_environment(self, num_boxes: int = 12, container_size: Optional[List[int]] = None,
                                    enhanced_reward: bool = True, seed: int = 42) -> gym.Env:
        """실험용 환경 생성.

        - `utils.boxes_generator`로 고정 컨테이너에 상응하는 박스 시퀀스 생성
        - `ImprovedRewardWrapper` 또는 본 파일의 `EnhancedRewardWrapper` 적용
        - `ActionMasker`로 불가능 행동을 마스킹하여 샘플 효율 개선
        """
        try:
            if container_size is None:
                container_size = [10, 10, 10]
            print(f"생성된 박스 개수: {num_boxes}")
            print(f"컨테이너 크기: {container_size}")

            # Gym 환경 등록 (중복 등록 방지)
            if 'PackingEnv-v0' not in gym.envs.registry:
                register(id='PackingEnv-v0', entry_point='packing_env:PackingEnv')

            # 박스 시퀀스 생성
            box_sizes = boxes_generator(container_size, num_boxes, seed)

            # 환경 생성
            env = gym.make(
                "PackingEnv-v0",
                container_size=container_size,
                box_sizes=box_sizes,
                num_visible_boxes=min(3, num_boxes),
                render_mode=None,
                random_boxes=False,
                only_terminal_reward=False,
            )
            print("환경 생성 성공: PackingEnv-v0")

            # 보상 래퍼 선택 적용
            if enhanced_reward:
                env = EnhancedRewardWrapper(env)
                print("강화된 보상 래퍼 적용됨")
            else:
                env = ImprovedRewardWrapper(env)
                print("개선된 보상 래퍼 적용됨")

            # 액션 마스킹 (불가능 좌표/박스 선택 제거)
            def get_action_masks(env):
                try:
                    unwrapped_env = env
                    while hasattr(unwrapped_env, 'env'):
                        unwrapped_env = unwrapped_env.env

                    if hasattr(unwrapped_env, 'action_masks'):
                        masks = unwrapped_env.action_masks()
                        if isinstance(masks, list):
                            return np.array(masks, dtype=bool)
                        return masks
                    return np.ones(env.action_space.n, dtype=bool)
                except Exception as e:
                    print(f"⚠️ 액션 마스크 생성 실패: {e}")
                    return np.ones(env.action_space.n, dtype=bool)

            env = ActionMasker(env, get_action_masks)
            print("액션 마스킹 래퍼 적용됨")

            print(f"시드 설정 완료: {seed}")
            return env

        except Exception as e:
            print(f"❌ 환경 생성 실패: {e}")
            raise e

    def get_enhanced_parameter_sets(self) -> Dict[str, Dict]:
        """탐색할 하이퍼파라미터 세트 사전.

        - 안정성 강화, 아키텍처 변화, 적극/보수 최적화 등 카테고리로 구성
        - 각 항목은 MaskablePPO의 주요 인자와 `policy_kwargs.net_arch`를 포함
        """
        # 1) 학습 안정성 강화 세트
        stability_sets = {
            'stability_conservative': {
                'learning_rate': 1.2e-04,
                'n_steps': 1024,
                'batch_size': 64,
                'n_epochs': 6,
                'clip_range': 0.15,
                'ent_coef': 0.005,
                'vf_coef': 0.5,
                'gae_lambda': 0.98,
                'net_arch': [dict(pi=[256, 128, 64], vf=[256, 128, 64])]
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
                'net_arch': [dict(pi=[256, 128, 64], vf=[256, 128, 64])]
            }
        }

        # 2) 네트워크 아키텍처 변화 세트 (폭/깊이/균형)
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
                'net_arch': [dict(pi=[512, 256, 128], vf=[512, 256, 128])]
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
                'net_arch': [dict(pi=[256, 256, 128, 64], vf=[256, 256, 128, 64])]
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
                'net_arch': [dict(pi=[384, 192, 96], vf=[384, 192, 96])]
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
                'net_arch': [dict(pi=[256, 128, 128, 64], vf=[256, 128, 128, 64])]
            }
        }

        # 3) 하이퍼파라미터 극단/정밀 세트
        optimized_sets = {
            'opt_precision': {
                'learning_rate': 1.1e-04,
                'n_steps': 1536,
                'batch_size': 192,
                'n_epochs': 8,
                'clip_range': 0.12,
                'ent_coef': 0.003,
                'vf_coef': 0.6,
                'gae_lambda': 0.99,
                'net_arch': [dict(pi=[256, 128, 64], vf=[256, 128, 64])]
            },
            'opt_aggressive': {
                'learning_rate': 1.8e-04,
                'n_steps': 256,
                'batch_size': 64,
                'n_epochs': 3,
                'clip_range': 0.25,
                'ent_coef': 0.02,
                'vf_coef': 0.4,
                'gae_lambda': 0.92,
                'net_arch': [dict(pi=[256, 128, 64], vf=[256, 128, 64])]
            }
        }

        all_sets = {}
        all_sets.update(stability_sets)
        all_sets.update(architecture_sets)
        all_sets.update(optimized_sets)
        return all_sets

    def train_and_evaluate(self, params: Dict, name: str, timesteps: int = 35000,
                           eval_episodes: int = 25, enhanced_reward: bool = True) -> Dict[str, Any]:
        """단일 구성 학습 및 평가 루틴.

        - MaskablePPO로 timesteps 만큼 학습 후, 여러 시드로 짧은 에피소드 평가
        - 논문 관점: 불가능 행동 마스킹으로 탐색 공간을 축소해 샘플 효율과 안정성을 향상
        """
        print(f"\n🔧 {name} 최적화 중...")

        env = self.create_enhanced_environment(enhanced_reward=enhanced_reward, seed=42)
        container_size, box_count = get_env_info(env)
        print(f"✅ 환경 생성 성공: 컨테이너{container_size}, 박스{box_count}개")

        model = MaskablePPO(
            'MultiInputPolicy',
            env,
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

        print(f"🎓 {name} 학습 시작: {timesteps:,} 스텝 (LR: {params['learning_rate']:.2e}, Net: {params['net_arch']})")
        start_time = time.time()
        model.learn(total_timesteps=timesteps)
        training_time = time.time() - start_time
        print(f"⏱️ {name} 학습 완료: {training_time:.1f}초")

        # 평가: 동일 환경 설정에서 시드만 다르게 반복
        print(f"🔍 {name} 평가 시작 ({eval_episodes} 에피소드, 최대 25 스텝)")
        rewards, utilizations, placements = [], [], []
        for i in range(eval_episodes):
            eval_env = self.create_enhanced_environment(enhanced_reward=enhanced_reward, seed=100 + i * 5)
            container_size, box_count = get_env_info(eval_env)
            obs = eval_env.reset(seed=100 + i * 5)
            if isinstance(obs, tuple):
                obs = obs[0]

            episode_reward = 0
            step_count = 0
            max_steps = 25
            while step_count < max_steps:
                try:
                    # 평가 시에도 확률적 행동으로 다양성 확보
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    step_count += 1
                    if terminated or truncated:
                        break
                except Exception as e:
                    print(f"⚠️ 평가 중 오류: {e}")
                    break

            final_utilization, placement_count = calculate_utilization_and_items(eval_env)
            rewards.append(episode_reward)
            utilizations.append(final_utilization)
            placements.append(placement_count)

            if i < 6 or i in [10, 15, 20] or i == eval_episodes - 1:
                print(f"   에피소드 {i+1}: 보상={episode_reward:.3f}, 활용률={final_utilization:.1%}, 박스={placement_count}개")
            eval_env.close()

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_utilization = np.mean(utilizations)
        std_utilization = np.std(utilizations)
        mean_placement = np.mean(placements)
        max_placement = np.max(placements)
        success_count = sum(1 for p in placements if p >= 5)
        success_rate = success_count / eval_episodes
        combined_score = mean_reward * 0.3 + mean_utilization * 100 * 0.7

        print(f"📊 {name} 최종 결과:")
        print(f"   평균 보상: {mean_reward:.3f} ± {std_reward:.3f}")
        print(f"   평균 활용률: {mean_utilization:.1%} ± {std_utilization:.1%}")
        print(f"   평균 배치: {mean_placement:.1f}개 (최대: {max_placement}개)")
        print(f"   성공률: {success_rate:.1%}")
        print(f"   종합 점수: {combined_score:.3f}")

        env.close()
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
        """Phase 4 전체 실험 실행 및 결과 집계/저장."""
        print(f"\n{'='*60}")
        print(f"🚀 Phase 4 Enhanced Optimization 시작")
        print(f"🎯 포커스: {focus}")
        print(f"⏱️ 학습 스텝: {timesteps:,}")
        print(f"{'='*60}")

        all_params = self.get_enhanced_parameter_sets()
        results: Dict[str, Any] = {}
        best_score = 0
        best_config = None

        # 카테고리 필터링
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
                    params, name, timesteps=timesteps, enhanced_reward=True
                )
                results[name] = result
                if result['combined_score'] > best_score:
                    best_score = result['combined_score']
                    best_config = name
                    print(f"🏆 새로운 최고 성능: {best_score:.3f}점")
            except Exception as e:
                print(f"❌ {name} 실행 중 오류: {e}")
                import traceback
                traceback.print_exc()
                continue

        total_time = time.time() - total_start_time

        if results:
            print(f"\n{'='*60}")
            print(f"🏆 Phase 4 최적화 결과")
            print(f"{'='*60}")

            sorted_results = sorted(results.items(), key=lambda x: x[1]['combined_score'], reverse=True)
            print("순위  설정명                    점수      개선율   활용률   성공률")
            print("-" * 70)
            for rank, (name, result) in enumerate(sorted_results[:10], 1):
                improvement = (result['combined_score'] - self.phase3_best['score']) / self.phase3_best['score'] * 100
                print(f"{rank:2d}    {name:<22} {result['combined_score']:6.2f}   {improvement:+5.1f}%   "
                      f"{result['mean_utilization']:5.1%}   {result['success_rate']:5.1%}")

            # 상위 1개 결과는 출력용으로만 사용 (미사용 변수 경고 방지 위해 직접 사용 생략)
            target_achievement = best_score / self.target_score * 100
            print(f"\n🏆 최고 성능: {best_score:.3f}점 ({best_config})")
            print(f"📈 목표 달성도: {target_achievement:.1f}% (목표 {self.target_score} 대비)")
            if best_score >= self.target_score:
                print(f"🎉 목표 달성 성공!")
            else:
                remaining = self.target_score - best_score
                print(f"📊 목표까지 {remaining:.3f}점 부족")

            # JSON 직렬화를 위한 numpy -> 기본타입 변환기
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            converted_results = convert_numpy_types(results)
            output_data = {
                'timestamp': self.timestamp,
                'phase': 'phase4_enhanced_optimization',
                'focus': focus,
                'timesteps': int(timesteps),
                'target_score': float(self.target_score),
                'phase3_baseline': float(self.phase3_best['score']),
                'best_score': float(best_score),
                'best_config': best_config,
                'target_achievement': float(target_achievement),
                'total_time_minutes': float(total_time / 60),
                'results': converted_results
            }

            output_file = os.path.join(self.results_dir, f'phase4_enhanced_{focus}_{self.timestamp}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Phase 4 결과: {output_file}")
            print(f"⏱️ 총 소요 시간: {total_time/60:.1f}분")
            return output_data
        else:
            print("❌ 유효한 결과가 없다.")
            return {}

    def create_performance_analysis(self, results_file: str):
        """결과 JSON을 바탕으로 종합 성능 차트를 생성 및 저장함."""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not data.get('results'):
                print("분석할 결과가 없다.")
                return

            configs, scores, utilizations, success_rates = [], [], [], []
            for name, result in data['results'].items():
                configs.append(name)
                scores.append(result['combined_score'])
                utilizations.append(result['mean_utilization'] * 100)
                success_rates.append(result['success_rate'] * 100)

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Phase 4 Enhanced Optimization Analysis\n'
                         f'Best: {data["best_score"]:.2f} (Target: {data["target_score"]})',
                         fontsize=16, fontweight='bold')

            # 1) 종합 점수 바차트
            axes[0,0].bar(range(len(configs)), scores, color='skyblue', alpha=0.7)
            axes[0,0].axhline(y=data['target_score'], color='red', linestyle='--', label=f'Target: {data["target_score"]}')
            axes[0,0].axhline(y=data['phase3_baseline'], color='orange', linestyle='--', label=f'Phase3: {data["phase3_baseline"]:.2f}')
            axes[0,0].set_title('Combined Scores')
            axes[0,0].set_ylabel('Score')
            axes[0,0].legend()
            axes[0,0].tick_params(axis='x', rotation=45)

            # 2) 활용률
            axes[0,1].bar(range(len(configs)), utilizations, color='lightgreen', alpha=0.7)
            axes[0,1].set_title('Space Utilization (%)')
            axes[0,1].set_ylabel('Utilization %')
            axes[0,1].tick_params(axis='x', rotation=45)

            # 3) 성공률
            axes[1,0].bar(range(len(configs)), success_rates, color='lightcoral', alpha=0.7)
            axes[1,0].set_title('Success Rate (%)')
            axes[1,0].set_ylabel('Success Rate %')
            axes[1,0].tick_params(axis='x', rotation=45)

            # 4) 상관관계 산점도 (활용률 vs 점수)
            axes[1,1].scatter(utilizations, scores, alpha=0.7, s=100)
            axes[1,1].set_xlabel('Utilization %')
            axes[1,1].set_ylabel('Combined Score')
            axes[1,1].set_title('Utilization vs Score')

            for ax in axes.flat:
                if hasattr(ax, 'set_xticks'):
                    ax.set_xticks(range(len(configs)))
                    ax.set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in configs], rotation=45, ha='right')

            plt.tight_layout()
            chart_file = results_file.replace('.json', '_analysis.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"📊 분석 차트 저장: {chart_file}")
            plt.close()
        except Exception as e:
            print(f"❌ 차트 생성 오류: {e}")


class EnhancedRewardWrapper(gym.RewardWrapper):
    """강화된 보상 래퍼(간단형).

    - 컨테이너 활용률, 연속 배치, 임계값 돌파 등의 보너스를 추가해 학습을 가속
    - 논문적 관점에서의 Reward Shaping: 탐색을 유도하되 과도한 편향 방지
    """

    def __init__(self, env):
        super().__init__(env)
        self.previous_utilization = 0.0
        self.consecutive_placements = 0

    def reset(self, **kwargs):
        self.previous_utilization = 0.0
        self.consecutive_placements = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        enhanced_reward = self.reward(reward)
        return obs, enhanced_reward, terminated, truncated, info

    def reward(self, reward):
        current_utilization, placement_count = calculate_utilization_and_items(self.env)

        enhanced_reward = reward
        if current_utilization > 0:
            enhanced_reward += (current_utilization ** 1.5) * 3.0
        if current_utilization > self.previous_utilization:
            enhanced_reward += (current_utilization - self.previous_utilization) * 5.0
        if placement_count > 0:
            self.consecutive_placements += 1
            enhanced_reward += min(self.consecutive_placements * 0.1, 1.0)
        else:
            self.consecutive_placements = 0
        if current_utilization > 0.25:
            enhanced_reward += 2.0
        elif current_utilization > 0.20:
            enhanced_reward += 1.0
        if placement_count >= 5:
            enhanced_reward += (placement_count - 4) * 0.5
        if placement_count > 0:
            efficiency = current_utilization / placement_count
            enhanced_reward += efficiency * 2.0

        self.previous_utilization = current_utilization
        return enhanced_reward


def main():
    """CLI 엔트리포인트: 실험 실행 또는 분석 모드."""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Optimization for 3D Bin Packing')
    parser.add_argument('--focus', choices=['all', 'stability', 'architecture', 'optimization'],
                        default='all', help='Optimization focus area')
    parser.add_argument('--timesteps', type=int, default=35000, help='Training timesteps')
    parser.add_argument('--analyze', type=str, help='Analyze results from JSON file')

    args = parser.parse_args()
    optimizer = EnhancedOptimizer()

    if args.analyze:
        optimizer.create_performance_analysis(args.analyze)
    else:
        result = optimizer.run_phase4_optimization(focus=args.focus, timesteps=args.timesteps)
        if result and result.get('results'):
            output_file = os.path.join(optimizer.results_dir,
                                       f'phase4_enhanced_{args.focus}_{optimizer.timestamp}.json')
            optimizer.create_performance_analysis(output_file)

            best_score = result['best_score']
            target_score = result['target_score']
            if best_score >= target_score:
                print(f"\n🎉 축하! 목표 {target_score}점 달성!")
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