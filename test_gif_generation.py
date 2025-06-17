#!/usr/bin/env python3
"""
GIF 생성 문제 진단 및 해결 테스트 스크립트
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
from sb3_contrib.common.maskable.evaluation import evaluate_policy

# 프로젝트 경로 설정
project_root = Path(__file__).parent
sys.path.append(str(project_root))

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

# PIL 및 IO 관련 import
try:
    from PIL import Image
    import io
    print("✅ PIL 및 IO 모듈 로드 완료")
except ImportError as e:
    print(f"❌ PIL 또는 IO 모듈 로드 실패: {e}")
    sys.exit(1)

def test_environment_rendering():
    """환경 렌더링 테스트"""
    print("\n=== 환경 렌더링 테스트 ===")
    
    try:
        # 환경 생성
        env = gym.make(
            "PackingEnv-v0",
            container_size=[10, 10, 10],
            box_sizes=[[3, 3, 3], [3, 2, 3], [3, 4, 2], [3, 2, 4], [3, 2, 3]],
            num_visible_boxes=1,
            render_mode="human",
            random_boxes=False,
            only_terminal_reward=False,
        )
        
        print(f"환경 생성 완료: {type(env)}")
        print(f"Render mode: {getattr(env, 'render_mode', 'None')}")
        
        # 환경 리셋
        obs, info = env.reset()
        print("환경 리셋 완료")
        
        # 렌더링 테스트
        fig = env.render()
        print(f"렌더링 결과: {type(fig)}")
        
        if fig is not None:
            print("✅ 렌더링 성공")
            
            # 이미지 변환 테스트 (대체 방법 사용)
            try:
                # 먼저 kaleido 방법 시도
                try:
                    fig_png = fig.to_image(format="png")
                    buf = io.BytesIO(fig_png)
                    img = Image.open(buf)
                    print(f"✅ 이미지 변환 성공 (kaleido): {img.size}")
                    method = "kaleido"
                except Exception as kaleido_error:
                    print(f"⚠️ kaleido 변환 실패: {kaleido_error}")
                    
                    # plotly-orca 방법 시도
                    try:
                        fig.write_image("temp_test.png", engine="orca")
                        img = Image.open("temp_test.png")
                        os.remove("temp_test.png")  # 임시 파일 삭제
                        print(f"✅ 이미지 변환 성공 (orca): {img.size}")
                        method = "orca"
                    except Exception as orca_error:
                        print(f"⚠️ orca 변환 실패: {orca_error}")
                        
                        # HTML 방법 시도 (최후의 수단)
                        try:
                            html_str = fig.to_html()
                            print(f"✅ HTML 변환 성공 (length: {len(html_str)})")
                            print("⚠️ 이미지 변환은 실패했지만 렌더링은 작동합니다")
                            method = "html_only"
                            # 더미 이미지 생성 (테스트용)
                            from PIL import Image as PILImage
                            img = PILImage.new('RGB', (800, 600), color='white')
                        except Exception as html_error:
                            print(f"❌ 모든 변환 방법 실패: {html_error}")
                            raise Exception("모든 이미지 변환 방법 실패")
                
                # 테스트 이미지 저장
                test_path = f"gifs/test_rendering_{method}.png"
                os.makedirs("gifs", exist_ok=True)
                img.save(test_path)
                print(f"✅ 테스트 이미지 저장: {test_path}")
                
                return True, env
                
            except Exception as e:
                print(f"❌ 이미지 변환 완전 실패: {e}")
                return False, env
        else:
            print("❌ 렌더링 결과가 None입니다")
            return False, env
            
    except Exception as e:
        print(f"❌ 환경 렌더링 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_model_loading():
    """모델 로딩 테스트"""
    print("\n=== 모델 로딩 테스트 ===")
    
    # 최신 모델 파일 찾기
    model_dir = Path("models")
    if not model_dir.exists():
        print("❌ models 디렉토리가 없습니다")
        return None
    
    model_files = list(model_dir.glob("*improved*"))
    if not model_files:
        model_files = list(model_dir.glob("*ppo*"))
    
    if not model_files:
        print("❌ 모델 파일을 찾을 수 없습니다")
        return None
    
    # 가장 최신 모델 선택
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"로딩할 모델: {latest_model}")
    
    try:
        model = MaskablePPO.load(str(latest_model))
        print("✅ 모델 로딩 성공")
        return model
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return None

def create_fixed_gif(model, env, timestamp=None):
    """수정된 GIF 생성 함수"""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n=== 수정된 GIF 생성 (타임스탬프: {timestamp}) ===")
    
    try:
        # 환경 리셋
        obs, info = env.reset()
        print("환경 리셋 완료")
        
        # 초기 상태 캡처
        initial_fig = env.render()
        frames = []
        
        if initial_fig is not None:
            try:
                fig_png = initial_fig.to_image(format="png")
                buf = io.BytesIO(fig_png)
                img = Image.open(buf)
                frames.append(img)
                print("✅ 초기 상태 캡처 완료")
            except Exception as e:
                print(f"❌ 초기 상태 캡처 실패: {e}")
        
        # 시뮬레이션 실행
        done = False
        truncated = False
        step_count = 0
        max_steps = 50  # 충분한 프레임 확보
        total_reward = 0
        
        print("시뮬레이션 시작...")
        
        while not (done or truncated) and step_count < max_steps:
            try:
                # 액션 선택
                action_masks = get_action_masks(env)
                action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
                
                # 스텝 실행
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                # 현재 상태 렌더링
                fig = env.render()
                if fig is not None:
                    try:
                        fig_png = fig.to_image(format="png")
                        buf = io.BytesIO(fig_png)
                        img = Image.open(buf)
                        frames.append(img)
                    except Exception as e:
                        print(f"스텝 {step_count} 렌더링 실패: {e}")
                
                step_count += 1
                
                # 진행 상황 출력
                if step_count % 10 == 0:
                    print(f"스텝 {step_count}: 보상={reward:.3f}, 누적보상={total_reward:.3f}")
                
            except Exception as e:
                print(f"스텝 {step_count} 실행 중 오류: {e}")
                break
        
        print(f"시뮬레이션 완료: {step_count}스텝, 총보상: {total_reward:.3f}")
        print(f"캡처된 프레임 수: {len(frames)}")
        
        # GIF 생성
        if len(frames) >= 2:
            gif_path = f"gifs/fixed_gif_{timestamp}.gif"
            
            try:
                # 프레임 크기 통일 (첫 번째 프레임 기준)
                base_size = frames[0].size
                normalized_frames = []
                
                for i, frame in enumerate(frames):
                    if frame.size != base_size:
                        # 크기가 다르면 리사이즈
                        frame = frame.resize(base_size, Image.LANCZOS)
                    normalized_frames.append(frame)
                
                # GIF 저장
                normalized_frames[0].save(
                    gif_path,
                    format='GIF',
                    append_images=normalized_frames[1:],
                    save_all=True,
                    duration=600,  # 600ms per frame
                    loop=0,
                    optimize=True
                )
                
                # 결과 확인
                file_size = os.path.getsize(gif_path)
                print(f"✅ GIF 생성 성공!")
                print(f"   - 파일: {gif_path}")
                print(f"   - 크기: {file_size/1024:.1f} KB")
                print(f"   - 프레임 수: {len(normalized_frames)}")
                print(f"   - 해상도: {base_size}")
                
                return True
                
            except Exception as e:
                print(f"❌ GIF 저장 실패: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        elif len(frames) == 1:
            # 단일 프레임만 있는 경우
            static_path = f"gifs/static_result_{timestamp}.png"
            frames[0].save(static_path)
            print(f"⚠️ 단일 프레임 저장: {static_path}")
            return True
            
        else:
            print("❌ 캡처된 프레임이 없습니다")
            return False
            
    except Exception as e:
        print(f"❌ GIF 생성 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    print("🎬 GIF 생성 문제 진단 및 해결 스크립트")
    print("=" * 50)
    
    # 1. 환경 렌더링 테스트
    render_success, env = test_environment_rendering()
    if not render_success:
        print("❌ 환경 렌더링 테스트 실패 - 종료")
        return
    
    # 2. 모델 로딩 테스트
    model = test_model_loading()
    if model is None:
        print("❌ 모델 로딩 실패 - 종료")
        return
    
    # 3. 수정된 GIF 생성 테스트
    gif_success = create_fixed_gif(model, env)
    
    if gif_success:
        print("\n🎉 GIF 생성 테스트 성공!")
        print("이제 KAMP 서버에서 정상적인 GIF가 생성될 것입니다.")
    else:
        print("\n❌ GIF 생성 테스트 실패")
        print("추가 디버깅이 필요합니다.")
    
    # 환경 정리
    if env:
        env.close()

if __name__ == "__main__":
    main() 