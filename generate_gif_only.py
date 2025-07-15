#!/usr/bin/env python3
"""
기존 학습된 모델을 사용하여 GIF만 생성하는 스크립트
KAMP 서버에 최적화된 matplotlib 버전
"""

import os
import sys
import numpy as np
from datetime import datetime

# matplotlib 백엔드 설정 (헤드리스 환경용)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    from packing_env import PackingEnv
    from PIL import Image
    import io
    print("✅ 모든 필요한 모듈 로드 완료")
except ImportError as e:
    print(f"❌ 모듈 로드 실패: {e}")
    sys.exit(1)

def create_3d_visualization(env, step_num=0):
    """matplotlib을 사용한 3D 박스 패킹 시각화"""
    try:
        container_size = env.unwrapped.container.size
        
        # 3D 플롯 생성
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 컨테이너 경계 그리기
        # 8개 모서리 점
        corners = [
            [0, 0, 0], [container_size[0], 0, 0], 
            [container_size[0], container_size[1], 0], [0, container_size[1], 0],
            [0, 0, container_size[2]], [container_size[0], 0, container_size[2]], 
            [container_size[0], container_size[1], container_size[2]], [0, container_size[1], container_size[2]]
        ]
        
        for corner in corners:
            ax.scatter(corner[0], corner[1], corner[2], color='red', s=40, alpha=0.8)
        
        # 컨테이너 경계 라인
        edges = [
            # 하단면
            ([0, container_size[0]], [0, 0], [0, 0]),
            ([container_size[0], container_size[0]], [0, container_size[1]], [0, 0]),
            ([container_size[0], 0], [container_size[1], container_size[1]], [0, 0]),
            ([0, 0], [container_size[1], 0], [0, 0]),
            # 상단면
            ([0, container_size[0]], [0, 0], [container_size[2], container_size[2]]),
            ([container_size[0], container_size[0]], [0, container_size[1]], [container_size[2], container_size[2]]),
            ([container_size[0], 0], [container_size[1], container_size[1]], [container_size[2], container_size[2]]),
            ([0, 0], [container_size[1], 0], [container_size[2], container_size[2]]),
            # 수직 라인
            ([0, 0], [0, 0], [0, container_size[2]]),
            ([container_size[0], container_size[0]], [0, 0], [0, container_size[2]]),
            ([container_size[0], container_size[0]], [container_size[1], container_size[1]], [0, container_size[2]]),
            ([0, 0], [container_size[1], container_size[1]], [0, container_size[2]])
        ]
        
        for edge in edges:
            ax.plot(edge[0], edge[1], edge[2], 'r-', alpha=0.4, linewidth=1)
        
        # 배치된 박스들 그리기
        packed_count = 0
        if hasattr(env.unwrapped, 'packed_boxes') and env.unwrapped.packed_boxes:
            packed_count = len(env.unwrapped.packed_boxes)
            colors = plt.cm.Set3(np.linspace(0, 1, packed_count))
            
            for idx, box in enumerate(env.unwrapped.packed_boxes):
                x, y, z = box.position
                dx, dy, dz = box.size
                
                # 박스의 8개 꼭짓점
                vertices = [
                    [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
                    [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
                ]
                
                # 6개 면 정의
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],  # 하단면
                    [vertices[4], vertices[5], vertices[6], vertices[7]],  # 상단면
                    [vertices[0], vertices[1], vertices[5], vertices[4]],  # 앞면
                    [vertices[2], vertices[3], vertices[7], vertices[6]],  # 뒷면
                    [vertices[0], vertices[3], vertices[7], vertices[4]],  # 왼쪽면
                    [vertices[1], vertices[2], vertices[6], vertices[5]]   # 오른쪽면
                ]
                
                face_collection = Poly3DCollection(faces, alpha=0.8, 
                                                 facecolor=colors[idx], edgecolor='black', linewidth=0.5)
                ax.add_collection3d(face_collection)
        
        # 축 설정
        ax.set_xlabel('X (Depth)', fontsize=10)
        ax.set_ylabel('Y (Length)', fontsize=10)
        ax.set_zlabel('Z (Height)', fontsize=10)
        ax.set_xlim(0, container_size[0])
        ax.set_ylim(0, container_size[1])
        ax.set_zlim(0, container_size[2])
        
        # 제목 설정
        ax.set_title(f'3D Bin Packing - Step {step_num}\n'
                    f'Packed Boxes: {packed_count}\n'
                    f'Container Size: {container_size}', 
                    fontsize=12, fontweight='bold')
        
        # 그리드 및 뷰 설정
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=25, azim=45)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 이미지로 변환
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        image = Image.open(buffer)
        plt.close(fig)  # 메모리 해제
        
        return image
        
    except Exception as e:
        print(f"시각화 오류: {e}")
        # 오류 발생시 빈 이미지 반환
        blank_img = Image.new('RGB', (800, 600), color='lightgray')
        return blank_img

def generate_gif_from_model(model_path, timestamp):
    """학습된 모델을 사용하여 GIF 생성"""
    print(f"=== GIF 생성 시작 (Model: {model_path}) ===")
    
    try:
        # 모델 로드
        print("모델 로딩 중...")
        model = MaskablePPO.load(model_path)
        print("✅ 모델 로딩 완료")
        
        # 환경 생성 (학습에 사용된 것과 동일한 설정)
        print("환경 생성 중...")
        box_sizes = [
            [3, 3, 3], [3, 2, 3], [3, 4, 2], [3, 2, 4], [3, 2, 3],
            [2, 2, 2], [2, 3, 2], [4, 2, 2], [3, 3, 2], [2, 2, 4],
            [3, 2, 2], [2, 3, 3], [4, 2, 3], [2, 4, 2], [3, 3, 4],
            [2, 2, 3], [3, 2, 4], [2, 3, 4], [4, 3, 2], [3, 4, 3]
        ]
        
        env = PackingEnv(
            container_size=[10, 10, 10],
            box_sizes=box_sizes,
            num_visible_boxes=1,  # 간단한 설정
            render_mode="human",
            random_boxes=False,
            only_terminal_reward=True
        )
        print("✅ 환경 생성 완료")
        
        # 시뮬레이션 시작
        obs, info = env.reset()
        print("환경 리셋 완료")
        
        # 초기 상태 캡처
        frames = []
        initial_frame = create_3d_visualization(env, step_num=0)
        frames.append(initial_frame)
        print("초기 상태 캡처 완료")
        
        # 에피소드 실행
        done = False
        truncated = False
        step_count = 0
        max_steps = 30  # 적절한 프레임 수
        total_reward = 0
        
        print("시뮬레이션 실행 중...")
        
        while not (done or truncated) and step_count < max_steps:
            try:
                # 액션 예측
                action_masks = get_action_masks(env)
                action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
                
                # 스텝 실행
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
                
                # 상태 캡처
                frame = create_3d_visualization(env, step_num=step_count)
                frames.append(frame)
                
                # 진행상황 출력
                if step_count % 5 == 0:
                    packed_boxes = len(env.unwrapped.packed_boxes) if hasattr(env.unwrapped, 'packed_boxes') else 0
                    print(f"  Step {step_count}: 보상={reward:.3f}, 누적={total_reward:.3f}, 박스={packed_boxes}")
                    
            except Exception as e:
                print(f"Step {step_count} 실행 중 오류: {e}")
                break
        
        print(f"시뮬레이션 완료: {step_count} 스텝, 총 보상: {total_reward:.3f}")
        print(f"캡처된 프레임: {len(frames)}개")
        
        # GIF 생성
        if len(frames) >= 2:
            gif_filename = f'fixed_demonstration_{timestamp}.gif'
            gif_path = f'gifs/{gif_filename}'
            
            # 기존 문제 파일 백업
            old_gif = f'gifs/trained_maskable_ppo_20250618_052442.gif'
            if os.path.exists(old_gif):
                backup_path = f'gifs/backup_trained_maskable_ppo_20250618_052442.gif'
                if not os.path.exists(backup_path):
                    os.rename(old_gif, backup_path)
                    print(f"기존 파일 백업: {backup_path}")
            
            try:
                # GIF 저장
                frames[0].save(
                    gif_path,
                    format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=700,  # 0.7초 per frame
                    loop=0,
                    optimize=True
                )
                
                # 결과 확인
                file_size = os.path.getsize(gif_path)
                print(f"✅ 새 GIF 생성 완료: {gif_filename}")
                print(f"   파일 크기: {file_size/1024:.1f} KB")
                print(f"   프레임 수: {len(frames)}")
                print(f"   재생 시간: {len(frames) * 0.7:.1f}초")
                
                # 기존 파일명으로 복사
                import shutil
                shutil.copy2(gif_path, old_gif)
                print(f"✅ 기존 파일 교체 완료: {old_gif}")
                
                return True
                
            except Exception as e:
                print(f"❌ GIF 저장 실패: {e}")
                return False
                
        else:
            print("❌ 충분한 프레임이 없습니다.")
            return False
            
    except Exception as e:
        print(f"❌ GIF 생성 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    print("🎬 KAMP 서버용 GIF 생성 도구")
    print("=" * 50)
    
    # 최신 모델 찾기
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print("❌ models 디렉토리가 없습니다.")
        return
    
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
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # GIF 생성
    success = generate_gif_from_model(latest_model, timestamp)
    
    if success:
        print("\n🎉 GIF 생성 성공!")
        print("   새로운 고품질 GIF가 생성되었습니다.")
        print("   기존 문제 파일이 수정된 버전으로 교체되었습니다.")
    else:
        print("\n❌ GIF 생성 실패")

if __name__ == "__main__":
    main() 