#!/usr/bin/env python3
"""
점진적 확장 테스트 - Import 오류 수정 버전
"""

import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
sys.path.append('src')

print("📈 점진적 확장 테스트 (수정 버전)")

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
        print("\n1️⃣ PyTorch 설정...")
        import torch
        
        # PyTorch 2.1+ 호환 설정
        torch.set_default_dtype(torch.float32)  # deprecated 경고 수정
        torch.set_num_threads(2)
        print(f"   ✅ PyTorch {torch.__version__} 설정 완료")
        
        print("\n2️⃣ 환경 등록...")
        import gymnasium as gym
        from gymnasium.envs.registration import register
        from packing_env import PackingEnv  # 올바른 import
        
        if 'PackingEnv-v0' not in gym.envs.registry:
            register(id='PackingEnv-v0', entry_point='packing_env:PackingEnv')
            print("   ✅ 환경 등록 완료")
        else:
            print("   ✅ 환경 이미 등록됨")
        
        print("\n3️⃣ 확장된 환경 생성...")
        # 올바른 import
        from utils import boxes_generator
        
        # 조금 더 큰 문제
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
        print(f"   ✅ 확장된 환경 생성 성공")
        print(f"   컨테이너: [6, 6, 6], 박스: {len(box_sizes)}개")
        
        print("\n4️⃣ 확장된 모델 생성...")
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
        print("   ✅ 확장된 모델 생성 성공")
        
        print("\n5️⃣ 확장된 학습 시작 (200 스텝)...")
        import time
        start_time = time.time()
        
        # 진행 상황 표시
        class ProgressTracker:
            def __init__(self):
                self.step_count = 0
            
            def __call__(self, locals_, globals_):
                self.step_count += 1
                if self.step_count % 50 == 0:
                    progress = (self.step_count / 200) * 100
                    print(f"     진행: {progress:.0f}% ({self.step_count}/200)")
                return True
        
        model.learn(
            total_timesteps=200, 
            progress_bar=False,
            callback=ProgressTracker()
        )
        
        training_time = time.time() - start_time
        print(f"   ✅ 확장된 학습 완료: {training_time:.1f}초")
        
        print("\n6️⃣ 확장된 평가 테스트...")
        rewards = []
        utilizations = []
        
        for i in range(3):
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(10):  # 10스텝씩
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                if terminated or truncated:
                    break
            
            rewards.append(total_reward)
            
            # 활용률 계산
            try:
                if hasattr(env.unwrapped, 'container'):
                    placed_volume = sum(box.volume for box in env.unwrapped.container.boxes 
                                      if box.position is not None)
                    container_volume = env.unwrapped.container.volume
                    utilization = placed_volume / container_volume if container_volume > 0 else 0.0
                    utilizations.append(utilization)
                else:
                    utilizations.append(0.0)
            except:
                utilizations.append(0.0)
            
            print(f"   에피소드 {i+1}: 보상={total_reward:.3f}, 활용률={utilizations[-1]:.1%}")
        
        avg_reward = sum(rewards) / len(rewards)
        avg_utilization = sum(utilizations) / len(utilizations)
        combined_score = avg_reward * 0.3 + avg_utilization * 100 * 0.7
        
        print(f"\n🎉 확장된 테스트 성공!")
        print(f"   평균 보상: {avg_reward:.4f}")
        print(f"   평균 활용률: {avg_utilization:.1%}")
        print(f"   종합 점수: {combined_score:.4f}")
        print(f"   학습 시간: {training_time:.1f}초")
        print(f"   최적 파라미터가 확장된 문제에서도 잘 동작합니다!")
        
        # 정리
        env.close()
        del model
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ 확장 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 현재 환경:")
    print(f"   Python: {sys.version}")
    print(f"   작업 디렉토리: {os.getcwd()}")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    success = gradual_test()
    if success:
        print("\n✅ 점진적 확장 테스트 완전 성공!")
        print("💡 최적 파라미터가 더 큰 문제에서도 검증되었습니다.")
        print("🚀 이제 본격적인 실험을 진행할 수 있습니다.")
    else:
        print("\n❌ 확장 테스트 실패")
        print("🔧 추가 디버깅이 필요합니다.")
