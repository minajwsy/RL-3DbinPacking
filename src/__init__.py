"""
3D Bin Packing Environment with Maskable PPO
Original code from Luis Garcia's 3D bin packing project
Enhanced with Maskable PPO and GPU/CPU auto-detection
"""

import gym
from gym.envs.registration import register

# 원본 환경 등록 (luisgarciar 구조 유지)
register(
    id="PackingEnv-v0",
    entry_point="src.packing_env:PackingEnv",
    max_episode_steps=1000,
    reward_threshold=0.9,
)

# 추가 환경 등록 (Maskable PPO 최적화)
register(
    id="PackingEnvMaskable-v0", 
    entry_point="src.packing_env:PackingEnv",
    max_episode_steps=1000,
    reward_threshold=0.9,
    kwargs={
        'enable_action_mask': True,
        'reward_type': 'C+P+S+A',
    }
)

__version__ = "0.1.0"
__author__ = "Luis Garcia (Original), AI Development Team (Enhanced)"

# 주요 클래스 노출 (원본 구조 유지)
from src.packing_kernel import Box, Container
from src.packing_env import PackingEnv
from src.utils import boxes_generator

__all__ = [
    "Box",
    "Container", 
    "PackingEnv",
    "boxes_generator",
]
