from src import packing_kernel, packing_env, utils
from gym.envs.registration import register

register(id='PackingEnv0',
         entry_point='src.packing_env:PackingEnv0')
