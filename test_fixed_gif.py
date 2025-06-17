#!/usr/bin/env python3
"""
수정된 GIF 생성 함수 테스트 스크립트
"""

import os
import sys
from datetime import datetime

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from sb3_contrib import MaskablePPO
    from train_maskable_ppo import create_demonstration_gif
    from packing_env import PackingEnv
    print("✅ 모든 모듈 로드 완료")
except ImportError as e:
    print(f"❌ 모듈 로드 실패: {e}")
    sys.exit(1)

def main():
    print("=== 수정된 GIF 생성 테스트 ===")
    
    # 모델 찾기
    models_dir = 'models'
    model_files = []
    
    for file in os.listdir(models_dir):
        if file.endswith('.zip') and 'ppo' in file:
            model_files.append(os.path.join(models_dir, file))
    
    if not model_files:
        print("❌ 모델 파일을 찾을 수 없습니다.")
        return
    
    # 가장 최신 모델 선택
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"📁 사용할 모델: {latest_model}")
    
    try:
        # 모델 로드
        print("모델 로딩 중...")
        model = MaskablePPO.load(latest_model)
        print("✅ 모델 로딩 완료")
        
        # 환경 생성
        print("환경 생성 중...")
        box_sizes = [[3, 3, 3], [3, 2, 3], [3, 4, 2], [3, 2, 4], [3, 2, 3],
                     [2, 2, 2], [2, 3, 2], [4, 2, 2], [3, 3, 2], [2, 2, 4]]
        
        env = PackingEnv(
            container_size=[10, 10, 10],
            box_sizes=box_sizes,
            num_visible_boxes=3,
            render_mode="human",
            random_boxes=False,
            only_terminal_reward=False
        )
        print("✅ 환경 생성 완료")
        
        # GIF 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"GIF 생성 시작 (timestamp: {timestamp})...")
        
        create_demonstration_gif(model, env, timestamp)
        
        # 결과 확인
        gif_path = f'gifs/trained_maskable_ppo_{timestamp}.gif'
        if os.path.exists(gif_path):
            file_size = os.path.getsize(gif_path)
            print(f"\n🎉 성공!")
            print(f"   파일: {gif_path}")
            print(f"   크기: {file_size/1024:.1f} KB")
            
            # 프레임 수 확인
            try:
                from PIL import Image
                img = Image.open(gif_path)
                frame_count = getattr(img, 'n_frames', 1)
                print(f"   프레임 수: {frame_count}")
                if frame_count > 2:
                    print("✅ 정상적인 애니메이션 GIF가 생성되었습니다!")
                else:
                    print("⚠️ 프레임 수가 적습니다. 애니메이션이 제한적일 수 있습니다.")
            except Exception as e:
                print(f"   프레임 수 확인 실패: {e}")
                
        else:
            print("❌ GIF 파일이 생성되지 않았습니다.")
            
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 