"""
Maskable PPO를 사용한 3D bin packing 학습 스크립트
GPU/CPU 자동 선택 기능 포함
"""
import os
import sys
import time
import warnings
import datetime
from typing import Optional

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from src.utils import boxes_generator
from src.device_utils import setup_training_device, log_system_info, get_device

from plotly_gif import GIF
import io
from PIL import Image

# 경고 메시지 억제
warnings.filterwarnings("ignore")


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
    
    # 평가 콜백
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
    
    # 체크포인트 콜백
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
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
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
        obs = env.reset()
        done = False
        figs = []
        step_count = 0
        max_steps = 100  # 무한 루프 방지
        
        print("시연 시작...")
        
        while not done and step_count < max_steps:
            # 액션 마스크 가져오기
            action_masks = get_action_masks(env)
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, rewards, done, info = env.step(action)
            
            # 렌더링
            fig = env.render(mode="human")
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
    
    args = parser.parse_args()
    
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