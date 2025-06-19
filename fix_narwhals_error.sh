#!/bin/bash
# KAMP 서버용 narwhals 에러 즉시 해결 스크립트
# 사용법: ./fix_narwhals_error.sh

echo "=== KAMP 서버 narwhals 에러 해결 시작 ==="

# 가상환경 활성화 확인
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "가상환경을 활성화합니다..."
    source .venv/bin/activate
fi

echo "현재 Python 환경: $VIRTUAL_ENV"
echo "Python 버전: $(python --version)"

# narwhals 및 관련 의존성 설치
echo "=== 누락된 의존성 설치 중 ==="
uv pip install narwhals
uv pip install tenacity packaging

# plotly 재설치 (의존성 포함)
echo "=== plotly 재설치 중 ==="
uv pip uninstall plotly --yes || true
uv pip install plotly

# plotly_gif 설치 (누락된 의존성)
echo "=== plotly_gif 설치 중 ==="
uv pip install plotly_gif

# 설치 확인
echo "=== 설치 확인 중 ==="
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
echo "이제 다음 명령어로 학습을 재시작할 수 있습니다:"
echo "python -m src.train_maskable_ppo --timesteps 20000 --eval-freq 1500 --container-size 10 10 10 --num-boxes 32 --curriculum-learning --improved-rewards"