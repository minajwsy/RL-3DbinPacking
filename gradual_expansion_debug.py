#!/usr/bin/env python3
"""
점진적 확장 테스트 - 디버깅 강화 버전
평가 결과 0 문제 해결
"""

import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
sys.path.append('src')

print("🔍 점진적 확장 테스트 (디버깅 강화 버전)")

OPTIMAL_PARAMS = {
    'learning_rate': 2.6777169756959113e-06,
    'n_steps': 64,
    'batch_size': 8,
    'n_epochs': 2,
    'clip_range': 0.17716239549317803,
    'ent_coef': 0.06742268917730829,
    'vf_coef': 0.4545305173856873,
    'gae_lambda': 0.9449228658070746
}

def debug_action_masks(env):
    """액션 마스크 디버깅"""
    try:
        from train_maskable_ppo import get_action_masks
        masks = get_action_masks(env)
        valid_actions = sum(masks)
        print(f"   🎯 액션 마스크: 총 {len(masks)}개 중 {valid_actions}개 유효")
        return masks
    except Exception as e:
        print(f"   ⚠️ 액션 마스크 생성 실패: {e}")
        import numpy as np
        fallback_masks = np.ones(env.action_space.n, dtype=bool)
        print(f"   🔄 대체 마스크 사용: 모든 {len(fallback_masks)}개 액션 허용")
        return fallback_masks

def debug_test():
    try:
        print("\n1️⃣ PyTorch 설정...")
        import torch
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(2)
        print(f"   ✅ PyTorch {torch.__version__} 설정 완료")
        
        print("\n2️⃣ 환경 등록...")
        import gymnasium as gym
        from gymnasium.envs.registration import register
        from packing_env import PackingEnv
        
        if 'PackingEnv-v0' not in gym.envs.registry:
            register(id='PackingEnv-v0', entry_point='packing_env:PackingEnv')
            print("   ✅ 환경 등록 완료")
        else:
            print("   ✅ 환경 이미 등록됨")
        
        print("\n3️⃣ 디버깅 환경 생성...")
        from utils import boxes_generator
        
        box_sizes = boxes_generator([6, 6, 6], 4, 42)
        print(f"   📦 생성된 박스: {box_sizes}")
        
        env = gym.make(
            "PackingEnv-v0",
            container_size=[6, 6, 6],
            box_sizes=box_sizes,
            num_visible_boxes=2,
            render_mode=None,
            random_boxes=False,
            only_terminal_reward=False,
        )
        
        # 개선된 보상 래퍼 적용
        try:
            from train_maskable_ppo import ImprovedRewardWrapper
            env = ImprovedRewardWrapper(env)
            print("   ✅ 개선된 보상 래퍼 적용")
        except:
            print("   ⚠️ 개선된 보상 래퍼 없음 - 기본 환경 사용")
        
        # 액션 마스킹 래퍼 적용
        from sb3_contrib.common.wrappers import ActionMasker
        env = ActionMasker(env, debug_action_masks)
        
        print(f"   ✅ 디버깅 환경 생성 성공")
        print(f"   컨테이너: [6, 6, 6], 박스: {len(box_sizes)}개")
        
        # 환경 초기 상태 확인
        obs, info = env.reset(seed=42)
        print(f"   🔍 초기 관찰 공간 키: {obs.keys() if isinstance(obs, dict) else 'Not dict'}")
        print(f"   🎮 액션 공간 크기: {env.action_space.n}")
        
        print("\n4️⃣ 디버깅 모델 생성...")
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
            verbose=1,  # verbose 모드로 학습 과정 확인
            device='cpu',
            policy_kwargs=dict(
                net_arch=[32, 32],
                activation_fn=nn.ReLU,
            )
        )
        print("   ✅ 디버깅 모델 생성 성공")
        
        print("\n5️⃣ 디버깅 학습 시작 (500 스텝으로 증가)...")
        import time
        start_time = time.time()
        
        model.learn(total_timesteps=500, progress_bar=True)  # 더 많은 스텝과 progress_bar
        
        training_time = time.time() - start_time
        print(f"   ✅ 디버깅 학습 완료: {training_time:.1f}초")
        
        print("\n6️⃣ 상세 디버깅 평가...")
        
        for episode in range(3):
            print(f"\n   === 에피소드 {episode + 1} 상세 분석 ===")
            
            obs, info = env.reset()
            total_reward = 0
            steps = 0
            placed_boxes = 0
            
            print(f"   초기 상태: info={info}")
            
            for step in range(15):  # 더 많은 스텝
                # 액션 마스크 확인
                action_masks = debug_action_masks(env)
                valid_actions = [i for i, mask in enumerate(action_masks) if mask]
                
                if len(valid_actions) == 0:
                    print(f"     스텝 {step}: 유효한 액션 없음 - 종료")
                    break
                
                # 모델 예측
                action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
                
                print(f"     스텝 {step}: 액션={action}, 유효={action in valid_actions}")
                
                # 스텝 실행
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # 박스 배치 확인
                if hasattr(env.unwrapped, 'container'):
                    current_placed = sum(1 for box in env.unwrapped.container.boxes 
                                       if box.position is not None)
                    if current_placed > placed_boxes:
                        placed_boxes = current_placed
                        print(f"     스텝 {step}: 박스 배치 성공! (총 {placed_boxes}개)")
                
                print(f"     스텝 {step}: 보상={reward:.3f}, 누적보상={total_reward:.3f}")
                
                if terminated or truncated:
                    print(f"     스텝 {step}: 에피소드 종료 (terminated={terminated}, truncated={truncated})")
                    break
            
            # 최종 활용률 계산
            utilization = 0.0
            if hasattr(env.unwrapped, 'container'):
                placed_volume = sum(box.volume for box in env.unwrapped.container.boxes 
                                  if box.position is not None)
                container_volume = env.unwrapped.container.volume
                utilization = placed_volume / container_volume if container_volume > 0 else 0.0
                
                print(f"   📊 최종 결과:")
                print(f"     배치된 박스: {placed_boxes}개")
                print(f"     배치된 부피: {placed_volume}")
                print(f"     컨테이너 부피: {container_volume}")
                print(f"     활용률: {utilization:.1%}")
            
            print(f"   🎯 에피소드 {episode + 1}: 보상={total_reward:.3f}, 활용률={utilization:.1%}")
        
        print(f"\n🎉 디버깅 테스트 완료!")
        print(f"   학습 시간: {training_time:.1f}초")
        
        # 정리
        env.close()
        del model
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ 디버깅 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 디버깅 환경:")
    print(f"   Python: {sys.version}")
    print(f"   작업 디렉토리: {os.getcwd()}")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    success = debug_test()
    if success:
        print("\n✅ 디버깅 테스트 완료!")
        print("💡 문제가 식별되었습니다.")
    else:
        print("\n❌ 디버깅 테스트 실패")
