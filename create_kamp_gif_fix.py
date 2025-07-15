#!/usr/bin/env python3
"""
KAMP 서버용 GIF 생성 문제 해결 스크립트
plotly 대신 matplotlib을 사용하여 안정적인 GIF 생성
"""

import os
import sys
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 헤드리스 환경용 백엔드
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from datetime import datetime

# 현재 스크립트 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# 필요한 모듈 import
try:
    from packing_env import PackingEnv
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    print("✅ 모든 모듈이 성공적으로 로드되었습니다.")
except ImportError as e:
    print(f"❌ 모듈 로드 실패: {e}")
    sys.exit(1)

def visualize_packing_state_matplotlib(env, step_num=0):
    """
    matplotlib을 사용한 3D 박스 패킹 상태 시각화
    plotly 대신 matplotlib을 사용하여 헤드리스 환경에서 안정적으로 작동
    """
    try:
        # 컨테이너 크기 정보 (올바른 속성 사용)
        container_size = env.unwrapped.container.size
        
        # 3D 플롯 생성
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 컨테이너 경계 그리기
        # 컨테이너 프레임
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    ax.scatter([i*container_size[0]], [j*container_size[1]], [k*container_size[2]], 
                              color='red', s=30, alpha=0.7)
        
        # 컨테이너 경계 라인
        # X축 라인
        for j in [0, container_size[1]]:
            for k in [0, container_size[2]]:
                ax.plot([0, container_size[0]], [j, j], [k, k], 'r-', alpha=0.3)
        # Y축 라인
        for i in [0, container_size[0]]:
            for k in [0, container_size[2]]:
                ax.plot([i, i], [0, container_size[1]], [k, k], 'r-', alpha=0.3)
        # Z축 라인
        for i in [0, container_size[0]]:
            for j in [0, container_size[1]]:
                ax.plot([i, i], [j, j], [0, container_size[2]], 'r-', alpha=0.3)
        
        # 배치된 박스들 그리기
        if hasattr(env.unwrapped, 'placed_boxes') and env.unwrapped.placed_boxes:
            colors = plt.cm.Set3(np.linspace(0, 1, len(env.unwrapped.placed_boxes)))
            
            for idx, (pos, size) in enumerate(env.placed_boxes):
                x, y, z = pos
                dx, dy, dz = size
                
                # 박스의 8개 꼭짓점 계산
                vertices = [
                    [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],  # 하단면
                    [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]  # 상단면
                ]
                
                # 박스의 면들 그리기
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                # 6개 면 정의
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],  # 하단면
                    [vertices[4], vertices[5], vertices[6], vertices[7]],  # 상단면
                    [vertices[0], vertices[1], vertices[5], vertices[4]],  # 앞면
                    [vertices[2], vertices[3], vertices[7], vertices[6]],  # 뒷면
                    [vertices[0], vertices[3], vertices[7], vertices[4]],  # 왼쪽면
                    [vertices[1], vertices[2], vertices[6], vertices[5]]   # 오른쪽면
                ]
                
                # 면 컬렉션 생성
                face_collection = Poly3DCollection(faces, alpha=0.7, facecolor=colors[idx], edgecolor='black')
                ax.add_collection3d(face_collection)
        
        # 다음에 배치될 박스 정보 (있다면)
        current_box_info = ""
        if hasattr(env, 'current_box_size'):
            current_box_info = f"다음 박스: {env.current_box_size}"
        
        # 축 설정
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, container_size[0])
        ax.set_ylim(0, container_size[1])
        ax.set_zlim(0, container_size[2])
        
        # 제목 설정
        utilization = 0
        if hasattr(env, 'get_utilization'):
            try:
                utilization = env.get_utilization()
            except:
                pass
        
        ax.set_title(f'3D Bin Packing - Step {step_num}\n'
                    f'배치된 박스: {len(env.placed_boxes) if hasattr(env, "placed_boxes") else 0}\n'
                    f'컨테이너 크기: {container_size}\n'
                    f'활용률: {utilization:.1f}%\n'
                    f'{current_box_info}', 
                    fontsize=10)
        
        # 그리드 설정
        ax.grid(True, alpha=0.3)
        
        # 뷰 각도 설정 (더 보기 좋게)
        ax.view_init(elev=20, azim=45)
        
        # 여백 조정
        plt.tight_layout()
        
        # 이미지로 변환
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image = Image.open(buffer)
        
        plt.close(fig)  # 메모리 절약
        return image
        
    except Exception as e:
        print(f"matplotlib 시각화 오류: {e}")
        # 빈 이미지 반환
        blank_img = Image.new('RGB', (800, 600), color='white')
        return blank_img

def create_demonstration_gif_matplotlib(model_path, timestamp, max_steps=50):
    """
    matplotlib을 사용한 안정적인 GIF 생성 함수
    KAMP 서버 환경에서 plotly 대신 사용
    """
    print("=== matplotlib 기반 GIF 생성 시작 ===")
    
    try:
        # 환경 생성 (올바른 인터페이스 사용)
        box_sizes = [[3, 3, 3], [3, 2, 3], [3, 4, 2], [3, 2, 4], [3, 2, 3],
                     [2, 2, 2], [2, 3, 2], [4, 2, 2], [3, 3, 2], [2, 2, 4],
                     [3, 2, 2], [2, 3, 3], [4, 2, 3], [2, 4, 2], [3, 3, 4],
                     [2, 2, 3], [3, 2, 4], [2, 3, 4], [4, 3, 2], [3, 4, 3]]
        
        env = PackingEnv(
            container_size=[10, 10, 10],
            box_sizes=box_sizes,
            num_visible_boxes=3,
            render_mode='human',  # plotly 렌더링용
            random_boxes=False,
            only_terminal_reward=False
        )
        
        # 모델 로드
        print(f"모델 로드 중: {model_path}")
        model = MaskablePPO.load(model_path)
        
        # 환경 리셋
        obs, info = env.reset()
        print("환경 리셋 완료")
        
        # 초기 상태 렌더링
        initial_img = visualize_packing_state_matplotlib(env, step_num=0)
        frames = [initial_img]
        print("초기 상태 캡처 완료")
        
        # 시뮬레이션 실행
        done = False
        truncated = False
        step_count = 0
        episode_reward = 0
        
        print("에이전트 시연 중...")
        
        while not (done or truncated) and step_count < max_steps:
            try:
                # 액션 마스크 및 예측
                action_masks = get_action_masks(env)
                action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
                
                # 스텝 실행
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                # 스텝 후 상태 렌더링
                step_img = visualize_packing_state_matplotlib(env, step_num=step_count)
                frames.append(step_img)
                
                # 진행상황 출력
                if step_count % 10 == 0:
                    print(f"스텝 {step_count}: 누적 보상 = {episode_reward:.3f}")
                    
            except Exception as e:
                print(f"스텝 {step_count} 실행 중 오류: {e}")
                break
        
        print(f"시연 완료: {step_count}스텝, 총 보상: {episode_reward:.3f}")
        print(f"캡처된 프레임 수: {len(frames)}")
        
        # GIF 저장
        if len(frames) >= 2:
            gif_filename = f'matplotlib_demo_{timestamp}.gif'
            gif_path = f'gifs/{gif_filename}'
            
            # 프레임 지속시간 설정
            frame_duration = 800  # 0.8초
            
            try:
                frames[0].save(
                    gif_path,
                    format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=frame_duration,
                    loop=0,
                    optimize=True
                )
                
                # 파일 크기 확인
                file_size = os.path.getsize(gif_path)
                print(f"✅ GIF 저장 완료: {gif_filename}")
                print(f"   - 파일 크기: {file_size/1024:.1f} KB")
                print(f"   - 프레임 수: {len(frames)}")
                print(f"   - 프레임 지속시간: {frame_duration}ms")
                
                return gif_path
                
            except Exception as e:
                print(f"❌ GIF 저장 실패: {e}")
                return None
                
        else:
            print("❌ 충분한 프레임이 없습니다.")
            return None
            
    except Exception as e:
        print(f"❌ GIF 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """메인 실행 함수"""
    print("=== KAMP 서버용 GIF 생성 도구 ===")
    
    # 최신 모델 찾기
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print(f"❌ 모델 디렉토리가 없습니다: {models_dir}")
        return
    
    # 최신 모델 파일 찾기 (다양한 패턴 지원)
    model_files = []
    patterns = ['improved_ppo_mask_', 'ppo_mask_', 'best_model']
    
    for file in os.listdir(models_dir):
        if file.endswith('.zip'):
            for pattern in patterns:
                if pattern in file:
                    model_files.append(os.path.join(models_dir, file))
                    break
    
    # best_model 디렉토리도 확인
    best_model_dir = os.path.join(models_dir, 'best_model')
    if os.path.exists(best_model_dir):
        for file in os.listdir(best_model_dir):
            if file.endswith('.zip'):
                model_files.append(os.path.join(best_model_dir, file))
    
    if not model_files:
        print("❌ 학습된 모델을 찾을 수 없습니다.")
        print(f"   확인된 파일들: {os.listdir(models_dir)}")
        return
    
    # 가장 최신 모델 선택
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"📁 사용할 모델: {latest_model}")
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # GIF 생성
    gif_path = create_demonstration_gif_matplotlib(latest_model, timestamp)
    
    if gif_path:
        print(f"\n🎬 성공적으로 GIF가 생성되었습니다!")
        print(f"   파일 경로: {gif_path}")
        print(f"   파일 크기: {os.path.getsize(gif_path)/1024:.1f} KB")
    else:
        print("\n❌ GIF 생성에 실패했습니다.")

if __name__ == "__main__":
    main() 