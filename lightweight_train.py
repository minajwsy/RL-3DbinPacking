#!/usr/bin/env python3
"""
경량화된 3D Bin Packing 학습 스크립트 (터미널 크래시 방지)
"""
import os
import sys
import argparse
import warnings

# 경고 억제
warnings.filterwarnings("ignore")

# matplotlib 서버 모드 강제 설정
os.environ['MPLBACKEND'] = 'Agg'

# 환경 경로 설정
sys.path.append('src')

print("🚀 경량화된 3D Bin Packing 학습 시작")

def minimal_train():
    """최소한의 학습 함수"""
    try:
        # 필수 모듈만 import
        import numpy as np
        print("✅ numpy 로드 완료")
        
        # 로컬 모듈 import (특정 함수만 import)
        from packing_kernel import Container, Box, BoxCreator
        print("✅ packing_kernel 로드 완료")
        
        from train_maskable_ppo import make_env
        print("✅ make_env 로드 완료")
        
        # 환경 생성 테스트
        env = make_env(
            container_size=[10, 10, 10],
            num_boxes=8,  # 작은 문제로 시작
            num_visible_boxes=3,
            seed=42,
            render_mode=None,
            random_boxes=False,
            only_terminal_reward=False,
            improved_reward_shaping=True,
        )()
        
        print("✅ 환경 생성 완료")
        print(f"📦 컨테이너 크기: {env.unwrapped.container.size}")
        print(f"🎲 박스 개수: {len(env.unwrapped.box_creator.box_set)}")
        
        # 환경 테스트
        obs, _ = env.reset()
        action_space_size = env.action_space.n
        print(f"🎯 액션 공간 크기: {action_space_size}")
        print(f"👁️ 관찰 공간: {obs.keys()}")
        
        env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 경량화 학습 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="경량화된 3D Bin Packing 학습")
    parser.add_argument("--test", action="store_true", help="테스트 모드")
    
    try:
        args = parser.parse_args()
        
        if args.test:
            print("🧪 테스트 모드 실행")
            success = minimal_train()
            if success:
                print("✅ 테스트 성공!")
            else:
                print("❌ 테스트 실패!")
        else:
            print("💡 사용법: python lightweight_train.py --test")
            
    except Exception as e:
        print(f"❌ 메인 오류: {e}")

if __name__ == "__main__":
    main() 