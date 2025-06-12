import warnings
import datetime
import logging
import os
from pathlib import Path

import gym
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from src.utils import boxes_generator
from src.device_utils import get_device, get_sb3_device, print_device_info

from plotly_gif import GIF

import io
from PIL import Image

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_env(
    container_size,
    num_boxes,
    num_visible_boxes=1,  
    seed=0,
    render_mode=None,
    random_boxes=False,
    only_terminal_reward=False,
):
    """
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


def train_maskable_ppo(
    container_size=[10, 10, 10],
    num_boxes=64,
    num_visible_boxes=1,
    total_timesteps=100000,
    device=None,
    save_path="models",
):
    """
    Maskable PPO를 사용하여 3D bin packing 모델을 훈련합니다.
    
    Parameters:
    -----------
    container_size: List[int]
        컨테이너 크기
    num_boxes: int
        박스 개수
    num_visible_boxes: int
        에이전트가 볼 수 있는 박스 개수
    total_timesteps: int
        총 훈련 스텝 수
    device: str
        사용할 디바이스 ('cuda' 또는 'cpu')
    save_path: str
        모델 저장 경로
    """
    # 디바이스 설정
    if device is None:
        device = get_sb3_device()
    
    print_device_info()
    logger.info(f"훈련에 사용할 디바이스: {device}")
    
    # 결과 저장 디렉토리 생성
    Path(save_path).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("gifs").mkdir(parents=True, exist_ok=True)
    
    # 환경 생성
    env = make_env(
        container_size=container_size,
        num_boxes=num_boxes,
        num_visible_boxes=num_visible_boxes,
        render_mode="human",
        random_boxes=False,
        only_terminal_reward=False,
    )
    
    # 환경 검증
    check_env(env, warn=True)
    logger.info("환경 검증 완료")
    
    # 평가 환경 생성
    eval_env = make_env(
        container_size=container_size,
        num_boxes=num_boxes,
        num_visible_boxes=num_visible_boxes,
        render_mode=None,
        random_boxes=False,
        only_terminal_reward=True,
    )
    
    # 콜백 설정
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path="logs",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )
    
    # 모델 생성
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        device=device,
        verbose=1,
        tensorboard_log="logs/tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    logger.info("Maskable PPO 훈련 시작")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )
    logger.info("훈련 완료")
    
    # 모델 저장
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{save_path}/ppo_3dbp_{timestamp}"
    model.save(model_path)
    logger.info(f"모델 저장 완료: {model_path}")
    
    return model, model_path


def evaluate_and_create_gif(
    model_path: str,
    container_size=[10, 10, 10],
    box_sizes=None,
    gif_name=None,
):
    """
    훈련된 모델을 평가하고 GIF를 생성합니다.
    
    Parameters:
    -----------
    model_path: str
        모델 파일 경로
    container_size: List[int]
        컨테이너 크기
    box_sizes: List[List[int]]
        박스 크기 리스트
    gif_name: str
        GIF 파일명
    """
    if box_sizes is None:
        box_sizes = [[3, 3, 3], [3, 2, 3], [3, 4, 2], [3, 2, 4], [3, 2, 3]]
    
    if gif_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_name = f"trained_model_{timestamp}.gif"
    
    # 모델 로드
    model = MaskablePPO.load(model_path)
    logger.info(f"모델 로드 완료: {model_path}")
    
    # 환경 생성
    env = gym.make(
        "PackingEnv-v0",
        container_size=container_size,
        box_sizes=box_sizes,
        num_visible_boxes=1,
        render_mode="human",
        random_boxes=False,
        only_terminal_reward=False,
    )
    
    # 에피소드 실행 및 GIF 생성
    obs = env.reset()
    done = False
    figs = []
    
    logger.info("모델 평가 및 GIF 생성 시작")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        fig = env.render(mode="human")
        
        # 플롯을 이미지로 변환
        fig_png = fig.to_image(format="png")
        buf = io.BytesIO(fig_png)
        img = Image.open(buf)
        figs.append(img)
    
    env.close()
    
    # GIF 저장
    if figs:
        gif_path = f'gifs/{gif_name}'
        figs[0].save(gif_path, format='GIF', 
                    append_images=figs[1:],
                    save_all=True, 
                    duration=500, 
                    loop=0)
        logger.info(f"GIF 저장 완료: {gif_path}")
    
    return gif_path


def main():
    """메인 훈련 함수"""
    warnings.filterwarnings("ignore")
    
    # 기본 설정 (원본 유지)
    container_size = [10, 10, 10]
    box_sizes = [[3, 3, 3], [3, 2, 3], [3, 4, 2], [3, 2, 4], [3, 2, 3]]
    
    # 환경 변수에서 설정 읽기 (클라우드/로컬 호환)
    total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", "50000"))
    force_cpu = os.getenv("FORCE_CPU", "false").lower() == "true"
    
    device = "cpu" if force_cpu else get_sb3_device()
    
    # 모델 훈련
    model, model_path = train_maskable_ppo(
        container_size=container_size,
        num_boxes=len(box_sizes),
        total_timesteps=total_timesteps,
        device=device,
    )
    
    # 평가 및 GIF 생성
    gif_path = evaluate_and_create_gif(
        model_path=model_path,
        container_size=container_size,
        box_sizes=box_sizes,
    )
    
    logger.info("모든 작업 완료")
    return model_path, gif_path


if __name__ == "__main__":
    main()