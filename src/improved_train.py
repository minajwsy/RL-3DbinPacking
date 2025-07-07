#!/usr/bin/env python3
"""
개선된 3D Bin Packing 학습 스크립트
기존 5.8725 성과를 바탕으로 한 최적화 버전
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ultimate_train_fix import ultimate_train

def improved_experiment_suite():
    """개선된 실험 세트"""
    
    print("🚀 개선된 3D Bin Packing 실험 시작")
    print("=" * 50)
    
    # 실험 1: 현재 성과 기준 더 긴 학습
    print("\n📈 실험 1: 장기 학습 (50,000 스텝)")
    ultimate_train(
        timesteps=30000,  # 50000 
        eval_freq=5000,   # 8000 
        container_size=[10, 10, 10],
        num_boxes=18,
        create_gif=True
    )
    
    # 실험 2: 더 도전적인 문제
    print("\n🎯 실험 2: 도전적 문제 (22개 박스)")
    ultimate_train(
        timesteps=30000,   # 40000
        eval_freq=5000,    # 6000
        container_size=[10, 10, 10],
        num_boxes=22,
        create_gif=True
    )
    
    # 실험 3: 다른 컨테이너 형태
    print("\n📦 실험 3: 직육면체 컨테이너")
    ultimate_train(
        timesteps=30000,  # 35000
        eval_freq=5000,   # 5000
        container_size=[15, 10, 8],
        num_boxes=20,
        create_gif=True
    )
    
    print("\n✅ 모든 개선 실험 완료!")
    print("results/ 폴더에서 결과를 확인하세요.")

if __name__ == "__main__":
    improved_experiment_suite() 