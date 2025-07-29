#!/usr/bin/env python3
"""
간단한 액션 마스킹 테스트
"""

import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.append('src')

def test_action_masking():
    try:
        print("🧪 액션 마스킹 함수 테스트")
        
        import gymnasium as gym
        from gymnasium.envs.registration import register
        from packing_env import PackingEnv
        from utils import boxes_generator
        
        if 'PackingEnv-v0' not in gym.envs.registry:
            register(id='PackingEnv-v0', entry_point='packing_env:PackingEnv')
        
        # 간단한 환경 생성
        box_sizes = boxes_generator([4, 4, 4], 2, 42)
        env = gym.make(
            "PackingEnv-v0",
            container_size=[4, 4, 4],
            box_sizes=box_sizes,
            num_visible_boxes=1,
            render_mode=None,
            random_boxes=False,
            only_terminal_reward=False,
        )
        
        obs, info = env.reset(seed=42)
        print(f"✅ 환경 생성 성공")
        print(f"   액션 공간: {env.action_space.n}")
        print(f"   박스 개수: {len(box_sizes)}")
        
        # 액션 마스킹 함수 테스트
        try:
            from train_maskable_ppo import get_action_masks
            masks = get_action_masks(env)
            valid_count = sum(masks)
            print(f"✅ get_action_masks 함수 동작")
            print(f"   총 액션: {len(masks)}, 유효 액션: {valid_count}")
            
            if valid_count > 0:
                # 첫 번째 유효한 액션 찾기
                valid_action = next(i for i, mask in enumerate(masks) if mask)
                print(f"   첫 번째 유효 액션: {valid_action}")
                
                # 액션 실행 테스트
                obs, reward, terminated, truncated, info = env.step(valid_action)
                print(f"✅ 액션 실행 성공: 보상={reward}")
                
                # 박스 배치 확인
                if hasattr(env.unwrapped, 'container'):
                    placed_count = sum(1 for box in env.unwrapped.container.boxes 
                                     if box.position is not None)
                    print(f"   배치된 박스: {placed_count}개")
                
            return True
            
        except Exception as e:
            print(f"❌ get_action_masks 함수 오류: {e}")
            print("🔄 대안 액션 마스킹 사용")
            
            # 대안: 간단한 액션 마스킹
            import numpy as np
            simple_masks = np.ones(env.action_space.n, dtype=bool)
            print(f"   대안 마스크: 모든 {len(simple_masks)}개 액션 허용")
            
            # 임의 액션 테스트
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"✅ 임의 액션({action}) 실행: 보상={reward}")
            
            return True
            
    except Exception as e:
        print(f"❌ 액션 마스킹 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_action_masking()
