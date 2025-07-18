import warnings

import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from utils import boxes_generator  #org: src.utils 

from plotly_gif import GIF

import io
from PIL import Image

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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    container_size = [10, 10, 10]
    box_sizes2 = [[3, 3, 3], [3, 2, 3], [3, 4, 2], [3, 2, 4], [3, 2, 3]]

    orig_env = gym.make(
        "PackingEnv-v0",
        container_size=container_size,
        box_sizes=box_sizes2,
        num_visible_boxes=1,
        render_mode="human",
        random_boxes=False,
        only_terminal_reward=False,
    )

    env = gym.make(
        "PackingEnv-v0",
        container_size=container_size,
        box_sizes=box_sizes2,
        num_visible_boxes=1,
        render_mode="human",
        random_boxes=False,
        only_terminal_reward=False,
    )

    check_env(env, warn=True)

    model = MaskablePPO("MultiInputPolicy", env, verbose=1)
    print("begin training")
    model.learn(total_timesteps=10)
    print("done training")
    model.save("ppo_mask")

    obs, info = orig_env.reset()
    done = False
    truncated = False
    gif = GIF(gif_name="trained_5boxes.gif", gif_path="../gifs")
    figs = []

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = orig_env.step(action)
        fig = env.render()
        fig_png = fig.to_image(format="png")
        buf = io.BytesIO(fig_png)
        img = Image.open(buf)
        figs.append(img)
    print("done packing")
    env.close()

## Save gif
    # figs[0].save('../gifs/train_5_boxes.gif', format='GIF',
    #                append_images=figs[1:],
    #                save_all=True,
    #                duration=300, loop=0)
    # gif.create_gif(length=5000)

# src/train.py의 97번째 줄 근처 수정
    import datetime

# 현재 시간을 포함한 고유한 파일명 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_filename = f'train_5_boxes_{timestamp}.gif'
    gif_path = f'gifs/{gif_filename}'
    
    figs[0].save(gif_path, format='GIF', save_all=True, duration=500, loop=0)
    print(f"GIF saved as: {gif_filename}")