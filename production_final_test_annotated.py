#!/usr/bin/env python3
"""
프로덕션 최적 설정 최종 검증 스크립트 (주석 확장판)

개요
- Phase 4에서 도출된 최적 하이퍼파라미터(`PRODUCTION_OPTIMAL`)로 학습/평가를 수행하고,
  목표 점수(20.591) 대비 재현성과 안정성을 검증합니다.
- 평가 후, train_15_boxes.gif와 동일한 사양(1200x1200, 11프레임, 지연시간/루프 설정)의 데모 GIF를 생성합니다.

핵심 흐름
1) 환경 생성(create_production_env)
2) 모델 학습(train_production_model)
3) 성능 평가(evaluate_production_model)
4) 결과 저장(JSON) 및 데모 GIF 생성

사용 예시
- 완전 테스트: python production_final_test_annotated.py --timesteps 50000 --episodes 50
- 빠른 테스트: python production_final_test_annotated.py --quick

출력물
- 모델:   models/production_optimal_{timestamp}
- 결과:   results/production_final_{timestamp}.json
- 데모GIF: gifs/production_final_demo.gif (train_15_boxes.gif 포맷 준수)
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
import warnings
import io

# 환경 설정: 서버/헤드리스 환경에서 안전하게 동작
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings("ignore")
sys.path.append('src')  # src 모듈(환경/유틸) 임포트를 위해 경로 추가

# Phase 4에서 검증된 최적 하이퍼파라미터 설정
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
    """프로덕션 환경 생성

    - `train_maskable_ppo.make_env`를 통해 Gym 환경(`PackingEnv-v0`) 인스턴스 생성
    - 개선형 보상(`improved_reward_shaping=True`)과 불가능 행동 마스킹(ActionMasker) 적용
    - seed를 통해 재현 가능성 확보
    """
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
    """프로덕션 최적 설정으로 모델 학습

    반환: (model, duration_seconds)
    """
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
    """강화된 프로덕션 평가 (더 많은 에피소드)

    - 다양한 시드로 에피소드를 반복 실행하여 평균 보상/활용률/성공률을 측정
    - 성공 기준: 활용률 25% 이상 또는 박스 50% 이상 배치
    """
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

            except Exception:
                break

        # 성과 계산 (보상과 별도로 안전하게 활용률/배치 수 재계산)
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
        except Exception:
            pass

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


def _render_env_frame_3d(env, step_num=0, fig_size_px=(1200, 1200)):
    """현재 환경 상태를 1200x1200 PNG 이미지로 렌더링 후 PIL Image 반환

    - matplotlib 3D로 컨테이너 경계/배치 박스를 그립니다.
    - 이미지 크기는 train_15_boxes.gif와 동일(1200x1200)하게 고정합니다.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from PIL import Image

        # 컨테이너 크기
        container_size = getattr(env.unwrapped, 'container').size

        # 1200x1200 보장: inches * dpi = pixels
        target_w, target_h = fig_size_px
        dpi = 100
        fig_w_in, fig_h_in = target_w / dpi, target_h / dpi

        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')

        # 컨테이너 모서리와 에지 그리기
        corners = [
            [0, 0, 0], [container_size[0], 0, 0],
            [container_size[0], container_size[1], 0], [0, container_size[1], 0],
            [0, 0, container_size[2]], [container_size[0], 0, container_size[2]],
            [container_size[0], container_size[1], container_size[2]], [0, container_size[1], container_size[2]]
        ]
        for cx, cy, cz in corners:
            ax.scatter(cx, cy, cz, color='red', s=20, alpha=0.8)

        edges = [
            ([0, container_size[0]], [0, 0], [0, 0]),
            ([container_size[0], container_size[0]], [0, container_size[1]], [0, 0]),
            ([container_size[0], 0], [container_size[1], container_size[1]], [0, 0]),
            ([0, 0], [container_size[1], 0], [0, 0]),
            ([0, container_size[0]], [0, 0], [container_size[2], container_size[2]]),
            ([container_size[0], container_size[0]], [0, container_size[1]], [container_size[2], container_size[2]]),
            ([container_size[0], 0], [container_size[1], container_size[1]], [container_size[2], container_size[2]]),
            ([0, 0], [container_size[1], 0], [container_size[2], container_size[2]]),
            ([0, 0], [0, 0], [0, container_size[2]]),
            ([container_size[0], container_size[0]], [0, 0], [0, container_size[2]]),
            ([container_size[0], container_size[0]], [container_size[1], container_size[1]], [0, container_size[2]]),
            ([0, 0], [container_size[1], container_size[1]], [0, container_size[2]])
        ]
        for ex, ey, ez in edges:
            ax.plot(ex, ey, ez, 'r-', alpha=0.35, linewidth=1)

        # 배치된 박스 그리기
        packed_boxes = getattr(env.unwrapped, 'packed_boxes', [])
        if packed_boxes:
            colors = plt.cm.Set3(np.linspace(0, 1, len(packed_boxes)))
            for idx, box in enumerate(packed_boxes):
                x, y, z = box.position
                dx, dy, dz = box.size
                vertices = [
                    [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
                    [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
                ]
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],
                    [vertices[4], vertices[5], vertices[6], vertices[7]],
                    [vertices[0], vertices[1], vertices[5], vertices[4]],
                    [vertices[2], vertices[3], vertices[7], vertices[6]],
                    [vertices[0], vertices[3], vertices[7], vertices[4]],
                    [vertices[1], vertices[2], vertices[6], vertices[5]]
                ]
                pc = Poly3DCollection(faces, facecolor=colors[idx], edgecolor='black', alpha=0.8, linewidth=0.5)
                ax.add_collection3d(pc)

        ax.set_xlabel('X (Depth)')
        ax.set_ylabel('Y (Length)')
        ax.set_zlabel('Z (Height)')
        ax.set_xlim(0, container_size[0])
        ax.set_ylim(0, container_size[1])
        ax.set_zlim(0, container_size[2])
        ax.set_title(f'3D Bin Packing - Step {step_num}\n'
                     f'Packed: {len(packed_boxes)}  Container: {container_size}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=25, azim=45)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img
    except Exception:
        # 렌더 실패 시 흰색 1200x1200 이미지 반환
        try:
            from PIL import Image
            return Image.new('RGB', (1200, 1200), color='white')
        except Exception:
            return None


def save_gif_like_train15(frames, out_path):
    """train_15_boxes.gif 형식과 동일하게 저장 (1200x1200, 11프레임, 300ms/마지막 2100ms, loop=10)

    - 전달된 프레임 수가 부족하면 마지막 프레임 반복으로 11장 맞춤, 많으면 앞 11장 사용
    - 모든 프레임은 1200x1200으로 리사이즈하여 저장
    - durations: [300, 300, ..., 2100]
    """
    from PIL import Image

    if not frames:
        return False

    # 프레임 수를 11개로 맞춤
    target_frames = 11
    if len(frames) < target_frames:
        frames = frames + [frames[-1]] * (target_frames - len(frames))
    elif len(frames) > target_frames:
        frames = frames[:target_frames]

    # 크기 1200x1200으로 강제
    resized = [f.resize((1200, 1200)) for f in frames]

    # durations: 앞 10프레임 300ms, 마지막 2100ms
    durations = [300] * (target_frames - 1) + [2100]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        resized[0].save(
            out_path,
            format='GIF',
            append_images=resized[1:],
            save_all=True,
            duration=durations,
            loop=10,
            optimize=True
        )
        return True
    except Exception:
        return False


def generate_production_demo_gif(model, container_size=[10,10,10], num_boxes=12, max_steps=80, out_name='production_final_demo.gif'):
    """여러 시도 중 최다 배치 결과를 사용해 train_15_boxes.gif 포맷으로 GIF 생성

    - 다양한 시드(seeds)로 시연을 수행하고, 배치된 박스 수가 가장 많은 시도의 프레임을 선택해 GIF로 저장합니다.
    - 모델 예측을 기본으로 하되, 정체되면 마스크 기반 유효 액션에서 랜덤 시도로 탈출을 유도합니다.
    """
    try:
        from sb3_contrib.common.maskable.utils import get_action_masks
    except Exception:
        get_action_masks = None

    seeds = [777, 778, 779, 880, 881]  # 다중 시도
    best = {"frames": None, "placed": -1}

    for seed in seeds:
        env = create_production_env(container_size, num_boxes, seed=seed)
        if env is None:
            continue

        try:
            obs, _ = env.reset(seed=seed)
            frames = []
            frames.append(_render_env_frame_3d(env, step_num=0))  # 초기 상태 캡처

            done = False
            truncated = False
            step = 0
            last_packed = len(getattr(env.unwrapped, "packed_boxes", []))
            stagnation = 0   # 배치 정체 카운터

            while not (done or truncated) and step < max_steps:
                try:
                    # 1) 기본: 모델 예측 (결정론/비결정론 혼용으로 탐험 유도)
                    use_deterministic = (step % 3 != 0)
                    action = None
                    masks = None
                    if get_action_masks is not None:
                        masks = get_action_masks(env)
                        try:
                            action, _ = model.predict(obs, action_masks=masks, deterministic=use_deterministic)
                        except Exception:
                            action = None
                    else:
                        try:
                            action, _ = model.predict(obs, deterministic=use_deterministic)
                        except Exception:
                            action = None

                    # 2) 폴백: 유효 액션 중 하나 선택(무작위). 없으면 종료
                    if action is None and masks is not None:
                        valid_idx = np.flatnonzero(masks)
                        if valid_idx.size > 0:
                            action = int(np.random.choice(valid_idx))
                        else:
                            break

                    # 그래도 없으면 종료
                    if action is None:
                        break

                    # 스텝 실행
                    obs, reward, done, truncated, info = env.step(action)
                    step += 1

                    # 배치 이벤트 기반 캡처 (배치된 박스 수 증가 시에만 프레임 추가)
                    current_packed = len(getattr(env.unwrapped, "packed_boxes", []))
                    if current_packed > last_packed:
                        frames.append(_render_env_frame_3d(env, step_num=step))
                        last_packed = current_packed
                        stagnation = 0
                    else:
                        stagnation += 1

                    # 정체 해소: 일정 스텝 배치 실패 시 강제 탐험(마스크에서 랜덤 샘플)
                    if stagnation >= 8 and masks is not None:
                        valid_idx = np.flatnonzero(masks)
                        if valid_idx.size > 0:
                            fallback_action = int(np.random.choice(valid_idx))
                            obs, reward, done, truncated, info = env.step(fallback_action)
                            step += 1
                            current_packed2 = len(getattr(env.unwrapped, "packed_boxes", []))
                            if current_packed2 > last_packed:
                                frames.append(_render_env_frame_3d(env, step_num=step))
                                last_packed = current_packed2
                                stagnation = 0
                            else:
                                # 여전히 정체면 소폭 리셋(마이그레이션 방지)
                                stagnation = max(0, stagnation - 4)

                except Exception:
                    break

            # 최다 배치 시도 선택
            placed_count = last_packed
            if placed_count > best["placed"] and frames:
                best["placed"] = placed_count
                best["frames"] = frames

        finally:
            env.close()

    # 최종 저장
    if best["frames"]:
        out_path = os.path.join("gifs", out_name)
        ok = save_gif_like_train15(best["frames"], out_path)
        if ok:
            print(f"🎬 데모 GIF 생성 완료: {out_path} (최다 배치: {best['placed']}개)")
        else:
            print("⚠️ 데모 GIF 생성 실패")
        return ok
    else:
        print("⚠️ 유효한 프레임이 없습니다.")
        return False


def production_final_test(timesteps=50000, eval_episodes=50):
    """프로덕션 최종 테스트 실행

    1) 환경 생성 → 2) 모델 학습 → 3) 평가 → 4) 결과 저장(JSON) → 5) 데모 GIF 생성
    """
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
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\n💾 상세 결과 저장: {results_file}")

    # === 3단계: train_15_boxes.gif와 동일 형식의 데모 GIF 생성 ===
    try:
        demo_ok = generate_production_demo_gif(
            model,
            container_size=container_size,
            num_boxes=num_boxes,
            max_steps=80,  # 더 풍부한 시연을 위해 80스텝
            out_name='production_final_demo.gif'
        )
        if demo_ok:
            # 검증 출력: 프레임 수/loop/duration 확인
            try:
                from PIL import Image
                img = Image.open('gifs/production_final_demo.gif')
                print(f"🖼️ GIF 검증: size={img.size}, frames={getattr(img,'n_frames',1)}, loop={img.info.get('loop')}")
            except Exception:
                pass
    except Exception:
        print("⚠️ 데모 GIF 생성 중 예외 발생 (무시하고 진행)")

    return results['combined_score'] >= 18.57


def main():
    """메인 실행 함수

    - 완전 테스트(기본) 또는 빠른 테스트 모드(--quick) 제공
    """
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

#!/usr/bin/env python3
"""
프로덕션 최적 설정 최종 검증 스크립트(주석 확장판)

개요
- Phase 4 탐색 결과로 얻은 최적 하이퍼파라미터(`PRODUCTION_OPTIMAL`)로 학습/평가를 수행해
  재현성·안정성을 점검한다.
- 환경은 `src/train_maskable_ppo.make_env`를 통해 생성되며, 불가능 행동 마스킹과 개선형 보상
  쉐이핑을 사용한다.
- 논문 맥락: Transformer 기반 DRL과 달리 본 코드는 MLP+MaskablePPO를 사용하지만,
  상태 표현(높이맵+가시박스), 불가능행동 마스킹, 보상 설계를 통해 효율적 탐색이라는 공통 목표를 지향한다.

사용 방법(예)
- 완전 테스트: python production_final_test_annotated.py --timesteps 50000 --episodes 50
- 빠른 테스트: python production_final_test_annotated.py --quick

출력
- 모델:  models/production_optimal_{timestamp}
- 결과:  results/production_final_{timestamp}.json (종합 점수, 활용률/성공률 등)
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
import warnings

# 서버/헤드리스 환경 안전 설정
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings("ignore")
sys.path.append('src')

# Phase 4에서 확정된 프로덕션 최적 구성
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


def create_production_env(container_size=None, num_boxes=12, seed=42):
    """프로덕션 환경 팩토리.

    - `train_maskable_ppo.make_env`를 통해 Gym 환경(`PackingEnv-v0`) 생성
    - 개선형 보상(`improved_reward_shaping=True`)과 ActionMasker 적용
    """
    try:
        from train_maskable_ppo import make_env
        if container_size is None:
            container_size = [10, 10, 10]
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
        # src가 경로에 없거나 런타임 모듈 문제일 때 친절 안내
        print(f"❌ 환경 생성 실패: {str(e)}")
        return None


def train_production_model(env, timesteps=50000):
    """최적 하이퍼파라미터로 MaskablePPO 학습.

    반환값
    - (model, duration_seconds)
    """
    try:
        import torch
        import torch.nn as nn
        from sb3_contrib import MaskablePPO

        print(f"🚀 프로덕션 학습 시작: {timesteps:,} 스텝")
        print(f"📊 최적 설정: LR={PRODUCTION_OPTIMAL['learning_rate']:.2e}, "
              f"Steps={PRODUCTION_OPTIMAL['n_steps']}, "
              f"Batch={PRODUCTION_OPTIMAL['batch_size']}")

        start_time = time.time()
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
        model.learn(total_timesteps=timesteps, progress_bar=True)
        duration = time.time() - start_time
        print(f"⏱️ 학습 완료: {duration/60:.1f}분")
        return model, duration
    except Exception as e:
        print(f"❌ 학습 실패: {str(e)}")
        return None, 0


def evaluate_production_model(model, container_size=None, num_boxes=12, n_episodes=50):
    """강화된 프로덕션 평가 루틴.

    - 다양한 시드로 다수 에피소드를 실행하여 보상·활용률·성공률을 측정
    - 성공 기준: 활용률 25% 이상 또는 박스 50% 이상 배치
    """
    print(f"🔍 프로덕션 평가 시작: {n_episodes} 에피소드")

    all_rewards, all_utilizations, placement_counts = [], [], []
    success_count = 0

    for ep in range(n_episodes):
        seed = 200 + ep * 3
        eval_env = create_production_env(container_size, num_boxes, seed)
        if eval_env is None:
            continue

        obs, _ = eval_env.reset(seed=seed)
        episode_reward = 0.0

        for _ in range(50):  # 최대 50스텝
            try:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            except Exception:
                break

        # 성과 계산(보상과 활용률은 환경에 따라 상이할 수 있어 안전하게 재계산)
        utilization = 0.0
        placed_boxes = 0
        try:
            if hasattr(eval_env.unwrapped, 'container'):
                placed_volume = sum(
                    box.volume for box in eval_env.unwrapped.container.boxes if box.position is not None
                )
                container_volume = eval_env.unwrapped.container.volume
                utilization = placed_volume / container_volume if container_volume > 0 else 0.0
                placed_boxes = sum(
                    1 for box in eval_env.unwrapped.container.boxes if box.position is not None
                )
        except Exception:
            pass

        if utilization >= 0.25 or placed_boxes >= num_boxes * 0.5:
            success_count += 1

        all_rewards.append(episode_reward)
        all_utilizations.append(utilization)
        placement_counts.append(placed_boxes)

        if ep < 10 or ep % 10 == 0:
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
        'episodes': len(all_rewards),
        'all_rewards': all_rewards,
        'all_utilizations': all_utilizations
    }
    return results


def production_final_test(timesteps=50000, eval_episodes=50):
    """엔드투엔드 프로덕션 검증: 학습→저장→평가→요약 저장."""
    print("🏆 프로덕션 최적 설정 최종 검증 시작")
    print(f"📊 목표: 20.591점 재현 및 안정성 검증")
    print("="*60)

    container_size = [10, 10, 10]
    num_boxes = 12

    env = create_production_env(container_size, num_boxes, 42)
    if env is None:
        return False

    print(f"\n🎓 1단계: 프로덕션 모델 학습 ({timesteps:,} 스텝)")
    model, train_time = train_production_model(env, timesteps)
    if model is None:
        env.close()
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/production_optimal_{timestamp}"
    os.makedirs('models', exist_ok=True)
    model.save(model_path)
    print(f"💾 모델 저장: {model_path}")

    print(f"\n📊 2단계: 강화된 평가 ({eval_episodes} 에피소드)")
    results = evaluate_production_model(model, container_size, num_boxes, eval_episodes)
    env.close()
    if results is None:
        return False

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

    if results['combined_score'] >= 20.0:
        print(f"🎉 우수! 목표 성능 달성 또는 근접")
    elif results['combined_score'] >= 18.57:
        print(f"✅ 성공! Phase 3 목표 달성")
    else:
        print(f"📈 개선 필요: 추가 튜닝 권장")

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
    os.makedirs('results', exist_ok=True)
    results_file = f"results/production_final_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\n💾 상세 결과 저장: {results_file}")

    return results['combined_score'] >= 18.57


def main():
    """CLI 엔트리포인트: 완전 테스트/빠른 테스트 모드 지원."""
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
    print("🎉 프로덕션 최종 테스트 성공!" if success else "📈 성능 개선이 필요하다.")


if __name__ == "__main__":
    main()