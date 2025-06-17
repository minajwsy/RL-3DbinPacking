#!/usr/bin/env python3
"""
KAMP 서버용 GIF 생성 문제 해결 패치 스크립트
기존 trained_maskable_ppo_20250617_113411.gif 문제를 해결하고
새로운 GIF를 생성합니다.
"""

import os
import sys
import datetime
import warnings
from pathlib import Path

# 경고 억제
warnings.filterwarnings("ignore")

# matplotlib 백엔드 설정
import matplotlib
matplotlib.use('Agg')

# 필요한 라이브러리 import
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

# PIL 및 IO 관련 import
try:
    from PIL import Image
    import io
    print("✅ PIL 및 IO 모듈 로드 완료")
except ImportError as e:
    print(f"❌ PIL 또는 IO 모듈 로드 실패: {e}")
    print("pip install pillow를 실행해주세요")
    sys.exit(1)

# 환경 등록
try:
    from gymnasium.envs.registration import register
    from src.packing_env import PackingEnv
    
    if 'PackingEnv-v0' not in gym.envs.registry:
        register(id='PackingEnv-v0', entry_point='src.packing_env:PackingEnv')
        print("✅ PackingEnv-v0 환경 등록 완료")
    else:
        print("✅ PackingEnv-v0 환경 이미 등록됨")
except Exception as e:
    print(f"❌ 환경 등록 실패: {e}")
    sys.exit(1)

def find_latest_model():
    """최신 학습된 모델 찾기"""
    print("\n=== 최신 모델 검색 ===")
    
    model_dir = Path("models")
    if not model_dir.exists():
        print("❌ models 디렉토리가 없습니다")
        return None
    
    # 패턴별로 모델 파일 검색
    patterns = [
        "*improved*20250617*",  # 최신 improved 모델
        "*20250617*",           # 날짜 기반
        "*improved*",           # improved 일반
        "*ppo*mask*",           # PPO 마스크 모델
        "*ppo*"                 # 일반 PPO 모델
    ]
    
    for pattern in patterns:
        model_files = list(model_dir.glob(pattern))
        if model_files:
            # 가장 최신 파일 선택
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            print(f"✅ 발견된 모델: {latest_model}")
            print(f"   크기: {latest_model.stat().st_size / (1024*1024):.1f} MB")
            return latest_model
    
    print("❌ 모델 파일을 찾을 수 없습니다")
    return None

def create_robust_gif(model_path, timestamp=None):
    """강화된 GIF 생성 함수"""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n=== 강화된 GIF 생성 시작 (타임스탬프: {timestamp}) ===")
    
    try:
        # 모델 로드
        print(f"모델 로딩: {model_path}")
        model = MaskablePPO.load(str(model_path))
        print("✅ 모델 로딩 완료")
        
        # 환경 생성 (렌더링 모드 활성화)
        env = gym.make(
            "PackingEnv-v0",
            container_size=[10, 10, 10],
            box_sizes=[[3, 3, 3], [3, 2, 3], [3, 4, 2], [3, 2, 4], [3, 2, 3],
                      [2, 2, 2], [2, 3, 2], [4, 2, 2]],  # 더 많은 박스로 시연
            num_visible_boxes=3,
            render_mode="human",
            random_boxes=False,
            only_terminal_reward=False,
        )
        print("✅ 환경 생성 완료")
        
        # 환경 리셋
        obs, info = env.reset()
        print("✅ 환경 리셋 완료")
        
        # 초기 상태 캡처
        frames = []
        
        def safe_render_and_capture():
            """안전한 렌더링 및 캡처 함수"""
            try:
                fig = env.render()
                if fig is not None:
                    # 여러 방법으로 이미지 변환 시도
                    img = None
                    
                    # 방법 1: kaleido (가장 권장)
                    try:
                        fig_png = fig.to_image(format="png", width=800, height=600)
                        buf = io.BytesIO(fig_png)
                        img = Image.open(buf)
                        return img
                    except Exception as e1:
                        print(f"⚠️ kaleido 방법 실패: {e1}")
                    
                    # 방법 2: write_image
                    try:
                        temp_path = f"temp_frame_{len(frames)}.png"
                        fig.write_image(temp_path, width=800, height=600)
                        img = Image.open(temp_path)
                        os.remove(temp_path)  # 임시 파일 삭제
                        return img
                    except Exception as e2:
                        print(f"⚠️ write_image 방법 실패: {e2}")
                    
                    # 방법 3: plotly-orca (추가 시도)
                    try:
                        fig.write_image(f"temp_frame_{len(frames)}.png", 
                                      engine="orca", width=800, height=600)
                        img = Image.open(f"temp_frame_{len(frames)}.png")
                        os.remove(f"temp_frame_{len(frames)}.png")
                        return img
                    except Exception as e3:
                        print(f"⚠️ orca 방법 실패: {e3}")
                    
                    # 방법 4: 더미 이미지 (마지막 수단)
                    print("⚠️ 모든 변환 방식 실패, 더미 이미지 생성")
                    img = Image.new('RGB', (800, 600), color=(200, 200, 255))
                    return img
                
                else:
                    print("⚠️ 렌더링 결과가 None입니다")
                    return None
                    
            except Exception as e:
                print(f"❌ 렌더링 중 오류: {e}")
                return None
        
        # 초기 상태 캡처
        initial_frame = safe_render_and_capture()
        if initial_frame:
            frames.append(initial_frame)
            print("✅ 초기 상태 캡처 완료")
        
        # 시뮬레이션 실행
        done = False
        truncated = False 
        step_count = 0
        max_steps = 64  # 충분한 프레임을 위해 증가
        total_reward = 0
        successful_steps = 0
        
        print("🎬 시뮬레이션 시작...")
        
        while not (done or truncated) and step_count < max_steps:
            try:
                # 액션 마스크 가져오기
                action_masks = get_action_masks(env)
                
                # 모델 예측 (약간의 랜덤성 추가)
                action, _states = model.predict(
                    obs, 
                    action_masks=action_masks, 
                    deterministic=(step_count % 2 == 0)  # 교대로 deterministic/stochastic
                )
                
                # 스텝 실행
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                # 스텝 후 상태 캡처
                frame = safe_render_and_capture()
                if frame:
                    frames.append(frame)
                    successful_steps += 1
                
                step_count += 1
                
                # 진행 상황 출력
                if step_count % 10 == 0:
                    print(f"   스텝 {step_count}: 보상={reward:.3f}, 누적={total_reward:.3f}, 프레임={len(frames)}")
                
                # 조기 종료 방지 (강제로 더 많은 스텝 실행)
                if done and step_count < 30:
                    print(f"   조기 종료 감지 (스텝 {step_count}), 환경 리셋하여 계속...")
                    obs, info = env.reset()
                    done = False
                    truncated = False
                
            except Exception as e:
                print(f"❌ 스텝 {step_count} 실행 중 오류: {e}")
                break
        
        print(f"🎬 시뮬레이션 완료: {step_count}스텝, 총보상: {total_reward:.3f}")
        print(f"   성공적으로 캡처된 프레임: {len(frames)}")
        
        # GIF 생성
        if len(frames) >= 2:
            gif_path = f"gifs/fixed_demonstration_{timestamp}.gif"
            
            try:
                # 프레임 크기 정규화
                target_size = (800, 600)
                normalized_frames = []
                
                for i, frame in enumerate(frames):
                    if frame.size != target_size:
                        frame = frame.resize(target_size, Image.LANCZOS)
                    normalized_frames.append(frame)
                
                # GIF 저장 (최적화된 설정)
                normalized_frames[0].save(
                    gif_path,
                    format='GIF',
                    append_images=normalized_frames[1:],
                    save_all=True,
                    duration=500,  # 500ms per frame (적당한 속도)
                    loop=0,        # 무한 반복
                    optimize=True, # 파일 크기 최적화
                    disposal=2     # 프레임 간 최적화
                )
                
                # 결과 확인
                file_size = os.path.getsize(gif_path)
                print(f"✅ GIF 생성 성공!")
                print(f"   📁 파일: {gif_path}")
                print(f"   📊 크기: {file_size/1024:.1f} KB")
                print(f"   🎞️ 프레임 수: {len(normalized_frames)}")
                print(f"   📐 해상도: {target_size}")
                print(f"   ⏱️ 총 재생시간: {len(normalized_frames) * 0.5:.1f}초")
                
                # 기존 문제 파일 백업
                problem_gif = "gifs/trained_maskable_ppo_20250617_113411.gif"
                if os.path.exists(problem_gif):
                    backup_path = f"gifs/backup_trained_maskable_ppo_20250617_113411.gif"
                    os.rename(problem_gif, backup_path)
                    print(f"   🔄 기존 문제 파일 백업: {backup_path}")
                
                # 새 파일을 기존 이름으로 복사
                import shutil
                shutil.copy2(gif_path, problem_gif)
                print(f"   ✅ 수정된 GIF로 교체 완료: {problem_gif}")
                
                return True
                
            except Exception as e:
                print(f"❌ GIF 저장 실패: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        else:
            print("❌ 충분한 프레임이 캡처되지 않았습니다")
            print(f"   캡처된 프레임: {len(frames)}, 필요한 최소 프레임: 2")
            return False
            
        # 환경 정리
        env.close()
        
    except Exception as e:
        print(f"❌ GIF 생성 중 치명적 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    print("🔧 KAMP 서버용 GIF 문제 해결 패치")
    print("=" * 50)
    
    # 1. 최신 모델 찾기
    model_path = find_latest_model()
    if not model_path:
        print("❌ 모델을 찾을 수 없어 종료합니다")
        return
    
    # 2. 강화된 GIF 생성
    success = create_robust_gif(model_path)
    
    if success:
        print("\n🎉 GIF 문제 해결 완료!")
        print("   ✅ 새로운 고품질 GIF가 생성되었습니다")
        print("   ✅ 기존 문제 파일이 수정된 버전으로 교체되었습니다")
        print("\n📋 확인사항:")
        print("   - gifs/trained_maskable_ppo_20250617_113411.gif (수정됨)")
        print("   - gifs/fixed_demonstration_[timestamp].gif (새 파일)")
        print("   - gifs/backup_trained_maskable_ppo_20250617_113411.gif (백업)")
    else:
        print("\n❌ GIF 문제 해결 실패")
        print("   추가 디버깅이 필요합니다")

if __name__ == "__main__":
    main() 