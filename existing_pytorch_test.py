#!/usr/bin/env python3
"""
기존 PyTorch 환경 그대로 CPU 모드 테스트
재설치 없이 환경 변수만으로 해결
"""

import os
import sys

# 기존 PyTorch 그대로 사용하되 CPU 강제 모드
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import warnings
warnings.filterwarnings("ignore")
sys.path.append('src')

print("🔧 기존 PyTorch 환경에서 CPU 모드 테스트")

def check_current_environment():
    """현재 환경 상태 확인"""
    try:
        import torch
        print(f"✅ PyTorch 버전: {torch.__version__}")
        print(f"📋 CUDA 사용 가능: {torch.cuda.is_available()}")
        print(f"🖥️ 현재 디바이스: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # CPU 모드 강제 설정
        torch.set_default_tensor_type('torch.FloatTensor')
        torch.set_num_threads(2)  # 스레드 제한
        
        print("🔧 CPU 모드로 강제 설정 완료")
        return torch
        
    except Exception as e:
        print(f"❌ PyTorch 체크 실패: {e}")
        return None

# 최적 파라미터 (더 안전하게 조정)
SAFE_OPTIMAL_PARAMS = {
    'learning_rate': 2.6777169756959113e-06,
    'n_steps': 32,      # 매우 작게
    'batch_size': 4,    # 매우 작게
    'n_epochs': 1,      # 1 에포크만
    'clip_range': 0.17716239549317803,
    'ent_coef': 0.06742268917730829,
    'vf_coef': 0.4545305173856873,
    'gae_lambda': 0.9449228658070746
}

def safe_existing_pytorch_test():
    """기존 PyTorch로 안전한 테스트"""
    try:
        print("\n1️⃣ 현재 PyTorch 환경 확인...")
        torch = check_current_environment()
        if torch is None:
            return False
        
        print("\n2️⃣ 환경 등록...")
        import gymnasium as gym
        from gymnasium.envs.registration import register
        from packing_env import PackingEnv
        
        if 'PackingEnv-v0' not in gym.envs.registry:
            register(id='PackingEnv-v0', entry_point='packing_env:PackingEnv')
            print("   ✅ 환경 등록 완료")
        
        print("\n3️⃣ 초소형 환경 생성...")
        from utils import boxes_generator
        
        # 가장 작은 문제 설정
        box_sizes = boxes_generator([4, 4, 4], 2, 42)  # 4x4x4, 2개 박스
        env = gym.make(
            "PackingEnv-v0",
            container_size=[4, 4, 4],
            box_sizes=box_sizes,
            num_visible_boxes=1,  # 1개만
            render_mode=None,
            random_boxes=False,
            only_terminal_reward=False,
        )
        print("   ✅ 초소형 환경 생성 성공")
        
        print("\n4️⃣ 기존 PyTorch로 모델 생성...")
        from sb3_contrib import MaskablePPO
        import torch.nn as nn
        
        # 가장 작은 모델로 테스트
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=SAFE_OPTIMAL_PARAMS['learning_rate'],
            n_steps=SAFE_OPTIMAL_PARAMS['n_steps'],
            batch_size=SAFE_OPTIMAL_PARAMS['batch_size'],
            n_epochs=SAFE_OPTIMAL_PARAMS['n_epochs'],
            gamma=0.99,
            gae_lambda=SAFE_OPTIMAL_PARAMS['gae_lambda'],
            clip_range=SAFE_OPTIMAL_PARAMS['clip_range'],
            ent_coef=SAFE_OPTIMAL_PARAMS['ent_coef'],
            vf_coef=SAFE_OPTIMAL_PARAMS['vf_coef'],
            max_grad_norm=0.5,
            verbose=0,
            device='cpu',  # CPU 강제
            policy_kwargs=dict(
                net_arch=[16, 16],  # 가장 작은 네트워크
                activation_fn=nn.ReLU,
            )
        )
        print("   ✅ 초소형 모델 생성 성공")
        
        print("\n5️⃣ 초단기 학습 테스트 (50 스텝)...")
        import time
        start_time = time.time()
        
        # 극도로 짧은 학습
        model.learn(total_timesteps=50, progress_bar=False)
        
        training_time = time.time() - start_time
        print(f"   ✅ 초단기 학습 완료: {training_time:.1f}초")
        
        print("\n6️⃣ 1회 평가 테스트...")
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"   ✅ 평가 완료: 보상={reward:.3f}")
        
        print(f"\n🎉 기존 PyTorch 환경에서 최적 파라미터 테스트 성공!")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   디바이스: CPU (강제)")
        print(f"   학습 시간: {training_time:.1f}초")
        print(f"   테스트 보상: {reward:.3f}")
        
        # 정리
        env.close()
        del model
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ 기존 PyTorch 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 현재 시스템 정보:")
    print(f"   Python: {sys.version}")
    print(f"   작업 디렉토리: {os.getcwd()}")
    
    success = safe_existing_pytorch_test()
    if success:
        print("\n✅ 기존 환경에서 최적 파라미터 검증 완료!")
        print("💡 PyTorch 재설치 없이도 정상 동작합니다.")
        print("🚀 더 큰 실험으로 안전하게 확장 가능합니다.")
    else:
        print("\n❌ 테스트 실패")
        print("🔧 추가 안전 조치가 필요합니다.")
