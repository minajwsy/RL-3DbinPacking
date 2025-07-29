#!/usr/bin/env python3
"""
점진적 확장 테스트 - 기존 PyTorch 환경 유지
"""

import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
sys.path.append('src')

print("📈 점진적 확장 테스트")

OPTIMAL_PARAMS = {
    'learning_rate': 2.6777169756959113e-06,
    'n_steps': 64,      # 조금 더 크게
    'batch_size': 8,    # 조금 더 크게
    'n_epochs': 2,      # 2 에포크
    'clip_range': 0.17716239549317803,
    'ent_coef': 0.06742268917730829,
    'vf_coef': 0.4545305173856873,
    'gae_lambda': 0.9449228658070746
}

def gradual_test():
    try:
        import torch
        torch.set_default_tensor_type('torch.FloatTensor')
        torch.set_num_threads(2)
        
        import gymnasium as gym
        from gymnasium.envs.registration import register
        from packing_env import PackingEnv, utils
        
        if 'PackingEnv-v0' not in gym.envs.registry:
            register(id='PackingEnv-v0', entry_point='packing_env:PackingEnv')
        
        # 조금 더 큰 문제
        from utils import boxes_generator
        box_sizes = boxes_generator([6, 6, 6], 4, 42)  # 6x6x6, 4개 박스
        env = gym.make(
            "PackingEnv-v0",
            container_size=[6, 6, 6],
            box_sizes=box_sizes,
            num_visible_boxes=2,
            render_mode=None,
            random_boxes=False,
            only_terminal_reward=False,
        )
        
        from sb3_contrib import MaskablePPO
        import torch.nn as nn
        
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=OPTIMAL_PARAMS['learning_rate'],
            n_steps=OPTIMAL_PARAMS['n_steps'],
            batch_size=OPTIMAL_PARAMS['batch_size'],
            n_epochs=OPTIMAL_PARAMS['n_epochs'],
            gamma=0.99,
            gae_lambda=OPTIMAL_PARAMS['gae_lambda'],
            clip_range=OPTIMAL_PARAMS['clip_range'],
            ent_coef=OPTIMAL_PARAMS['ent_coef'],
            vf_coef=OPTIMAL_PARAMS['vf_coef'],
            max_grad_norm=0.5,
            verbose=0,
            device='cpu',
            policy_kwargs=dict(
                net_arch=[32, 32],  # 조금 더 큰 네트워크
                activation_fn=nn.ReLU,
            )
        )
        
        print("🎓 확장된 학습 시작 (200 스텝)...")
        import time
        start_time = time.time()
        
        model.learn(total_timesteps=200, progress_bar=False)
        
        training_time = time.time() - start_time
        print(f"✅ 확장된 학습 완료: {training_time:.1f}초")
        
        # 3회 평가
        rewards = []
        for i in range(3):
            obs, _ = env.reset()
            total_reward = 0
            for step in range(8):  # 8스텝씩
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            rewards.append(total_reward)
            print(f"   에피소드 {i+1}: 보상={total_reward:.3f}")
        
        avg_reward = sum(rewards) / len(rewards)
        print(f"\n🎉 확장된 테스트 성공!")
        print(f"   평균 보상: {avg_reward:.4f}")
        print(f"   학습 시간: {training_time:.1f}초")
        
        env.close()
        del model
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ 확장 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    success = gradual_test()
    if success:
        print("\n✅ 확장 테스트 성공! 더 큰 실험 가능합니다.")
    else:
        print("\n❌ 확장 테스트 실패")
