# 의존성 설치완료 확인
echo "=== 의존성 설치완료 확인 중 ==="
python -c "
try:
    import narwhals
    print('✅ narwhals 설치 확인')
    print(f'   버전: {narwhals.__version__}')
except ImportError as e:
    print(f'❌ narwhals 설치 실패: {e}')

try:
    import plotly
    print('✅ plotly 설치 확인')
    print(f'   버전: {plotly.__version__}')
except ImportError as e:
    print(f'❌ plotly 설치 실패: {e}')

try:
    import plotly.express as px
    print('✅ plotly.express 임포트 성공')
except ImportError as e:
    print(f'❌ plotly.express 임포트 실패: {e}')

try:
    from _plotly_utils.basevalidators import ColorscaleValidator
    print('✅ _plotly_utils.basevalidators 임포트 성공')
except ImportError as e:
    print(f'❌ _plotly_utils.basevalidators 임포트 실패: {e}')

try:
    import plotly_gif
    print('✅ plotly_gif 설치 확인')
    print(f'   버전: {plotly_gif.__version__}')
except ImportError as e:
    print(f'❌ plotly_gif 설치 실패: {e}')
except AttributeError:
    print('✅ plotly_gif 설치 확인 (버전 정보 없음)')
"

# 학습 스크립트 테스트
echo "=== 학습 스크립트 임포트 테스트 ==="
python -c "
try:
    import sys
    sys.path.append('src')
    from packing_kernel import *
    print('✅ packing_kernel 임포트 성공')
except ImportError as e:
    print(f'❌ packing_kernel 임포트 실패: {e}')

try:
    from src.train_maskable_ppo import *
    print('✅ train_maskable_ppo 임포트 성공')
except ImportError as e:
    print(f'❌ train_maskable_ppo 임포트 실패: {e}')
"

echo "=== narwhals 에러 해결 완료 ==="
echo "이제 다음 명령어로 학습을 시작 :"
echo "python -m src/ultimate_train_fix --timesteps 20000 --eval-freq 1500 --container-size 10 10 10 --num-boxes 32 --curriculum-learning --improved-rewards"