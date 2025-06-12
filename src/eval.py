"""
Maskable PPO 3D Bin Packing 모델 평가 스크립트
"""
import argparse
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd

import gym
from sb3_contrib.ppo_mask import MaskablePPO

from src.utils import boxes_generator
from src.device_utils import get_device, print_device_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    num_episodes: int = 100,
    container_size=[10, 10, 10],
    num_boxes=64,
    render=False,
    save_results=True,
):
    """
    훈련된 모델을 평가합니다.
    
    Parameters:
    -----------
    model_path: str
        모델 파일 경로
    num_episodes: int
        평가 에피소드 수
    container_size: List[int]
        컨테이너 크기
    num_boxes: int
        박스 개수
    render: bool
        렌더링 여부
    save_results: bool
        결과 저장 여부
    """
    logger.info(f"모델 평가 시작: {model_path}")
    
    # 모델 로드
    try:
        model = MaskablePPO.load(model_path)
        logger.info("모델 로드 성공")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return None
    
    # 환경 생성
    env = gym.make(
        "PackingEnv-v0",
        container_size=container_size,
        box_sizes=boxes_generator(container_size, num_boxes),
        num_visible_boxes=1,
        render_mode="human" if render else None,
        random_boxes=True,
        only_terminal_reward=True,
    )
    
    # 평가 실행
    rewards = []
    utilizations = []
    packed_items = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_packed = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            if reward > 0:  # 성공적으로 패킹된 경우
                episode_packed += 1
        
        rewards.append(episode_reward)
        utilizations.append(episode_reward)  # 최종 보상이 활용도
        packed_items.append(episode_packed)
        
        if (episode + 1) % 10 == 0:
            logger.info(f"에피소드 {episode + 1}/{num_episodes} 완료")
    
    env.close()
    
    # 결과 분석
    results = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_utilization': np.mean(utilizations),
        'std_utilization': np.std(utilizations),
        'mean_packed_items': np.mean(packed_items),
        'std_packed_items': np.std(packed_items),
        'max_reward': np.max(rewards),
        'min_reward': np.min(rewards),
    }
    
    # 결과 출력
    logger.info("=== 평가 결과 ===")
    logger.info(f"평균 보상: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    logger.info(f"평균 활용도: {results['mean_utilization']:.4f} ± {results['std_utilization']:.4f}")
    logger.info(f"평균 패킹 아이템: {results['mean_packed_items']:.2f} ± {results['std_packed_items']:.2f}")
    logger.info(f"최고 보상: {results['max_reward']:.4f}")
    logger.info(f"최저 보상: {results['min_reward']:.4f}")
    
    # 결과 저장
    if save_results:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # CSV 저장
        df = pd.DataFrame({
            'episode': range(num_episodes),
            'reward': rewards,
            'utilization': utilizations,
            'packed_items': packed_items,
        })
        
        csv_path = results_dir / f"evaluation_results_{Path(model_path).stem}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"결과 저장: {csv_path}")
        
        # 요약 결과 저장
        summary_path = results_dir / f"evaluation_summary_{Path(model_path).stem}.txt"
        with open(summary_path, 'w') as f:
            f.write("=== Maskable PPO 3D Bin Packing 평가 결과 ===\n")
            f.write(f"모델: {model_path}\n")
            f.write(f"평가 에피소드: {num_episodes}\n")
            f.write(f"컨테이너 크기: {container_size}\n")
            f.write(f"박스 개수: {num_boxes}\n\n")
            
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")
        
        logger.info(f"요약 저장: {summary_path}")
    
    return results


def compare_models(model_paths: list, **eval_kwargs):
    """
    여러 모델을 비교 평가합니다.
    
    Parameters:
    -----------
    model_paths: List[str]
        모델 파일 경로 리스트
    **eval_kwargs: dict
        평가 함수 인자들
    """
    logger.info("모델 비교 평가 시작")
    
    results_comparison = []
    
    for model_path in model_paths:
        logger.info(f"모델 평가 중: {model_path}")
        results = evaluate_model(model_path, **eval_kwargs)
        
        if results:
            results['model_path'] = model_path
            results['model_name'] = Path(model_path).stem
            results_comparison.append(results)
    
    # 비교 결과 출력
    if results_comparison:
        logger.info("\n=== 모델 비교 결과 ===")
        df_comparison = pd.DataFrame(results_comparison)
        df_comparison = df_comparison.sort_values('mean_reward', ascending=False)
        
        for _, row in df_comparison.iterrows():
            logger.info(f"{row['model_name']}: 평균 보상 {row['mean_reward']:.4f}, "
                       f"활용도 {row['mean_utilization']:.4f}")
        
        # 비교 결과 저장
        comparison_path = Path("results") / "model_comparison.csv"
        df_comparison.to_csv(comparison_path, index=False)
        logger.info(f"비교 결과 저장: {comparison_path}")
    
    return results_comparison


def main():
    parser = argparse.ArgumentParser(description="Maskable PPO 3D Bin Packing 모델 평가")
    parser.add_argument("--model_path", type=str, help="모델 파일 경로")
    parser.add_argument("--models_dir", type=str, default="models", help="모델 디렉토리")
    parser.add_argument("--num_episodes", type=int, default=100, help="평가 에피소드 수")
    parser.add_argument("--container_size", nargs=3, type=int, default=[10, 10, 10], 
                       help="컨테이너 크기 (예: 10 10 10)")
    parser.add_argument("--num_boxes", type=int, default=64, help="박스 개수")
    parser.add_argument("--render", action="store_true", help="렌더링 활성화")
    parser.add_argument("--compare", action="store_true", help="모든 모델 비교")
    
    args = parser.parse_args()
    
    # 디바이스 정보 출력
    print_device_info()
    
    if args.model_path:
        # 단일 모델 평가
        evaluate_model(
            model_path=args.model_path,
            num_episodes=args.num_episodes,
            container_size=args.container_size,
            num_boxes=args.num_boxes,
            render=args.render,
        )
    elif args.compare:
        # 모든 모델 비교
        models_dir = Path(args.models_dir)
        if not models_dir.exists():
            logger.error(f"모델 디렉토리가 존재하지 않습니다: {models_dir}")
            return
        
        model_files = list(models_dir.glob("*.zip"))
        if not model_files:
            logger.error(f"모델 파일을 찾을 수 없습니다: {models_dir}")
            return
        
        model_paths = [str(path) for path in model_files]
        compare_models(
            model_paths=model_paths,
            num_episodes=args.num_episodes,
            container_size=args.container_size,
            num_boxes=args.num_boxes,
            render=False,  # 비교시에는 렌더링 비활성화
        )
    else:
        # 최신 모델 자동 선택
        models_dir = Path(args.models_dir)
        if models_dir.exists():
            model_files = list(models_dir.glob("*.zip"))
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"최신 모델 선택: {latest_model}")
                
                evaluate_model(
                    model_path=str(latest_model),
                    num_episodes=args.num_episodes,
                    container_size=args.container_size,
                    num_boxes=args.num_boxes,
                    render=args.render,
                )
            else:
                logger.error("평가할 모델을 찾을 수 없습니다.")
        else:
            logger.error("모델 디렉토리가 존재하지 않습니다.")


 