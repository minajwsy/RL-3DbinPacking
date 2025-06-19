#!/bin/bash
# Windows GitBash용 narwhals 에러 해결 스크립트
# 사용법: ./fix_narwhals_error_windows.sh

echo "=== Windows GitBash narwhals 에러 해결 시작 ==="

# 현재 디렉토리 확인
echo "현재 작업 디렉토리: $(pwd)"

# 가상환경 활성화 (Windows 경로)
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "가상환경을 활성화합니다..."
    if [ -d ".venv/Scripts" ]; then
        source .venv/Scripts/activate
        echo "✅ Windows 가상환경 활성화 완료"
    elif [ -d ".venv/bin" ]; then
        source .venv/bin/activate
        echo "✅ Linux 스타일 가상환경 활성화 완료"
    else
        echo "❌ 가상환경을 찾을 수 없습니다. .venv 디렉토리를 확인하세요."
        exit 1
    fi
else
    echo "✅ 가상환경이 이미 활성화되어 있습니다."
fi

echo "현재 Python 환경: $VIRTUAL_ENV"
echo "Python 버전: $(python --version)"

# uv 명령어 확인
if ! command -v uv &> /dev/null; then
    echo "❌ uv 명령어를 찾을 수 없습니다."
    echo "uv가 설치되어 있는지 확인하세요."
    echo "설치 방법: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "uv 버전: $(uv --version)"

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

# 학습 스크립트 테스트 (Windows 환경에서는 선택적)
echo "=== 학습 스크립트 임포트 테스트 (선택적) ==="
python -c "
import sys
import os

# Windows 경로 설정
current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

try:
    from packing_kernel import *
    print('✅ packing_kernel 임포트 성공')
except ImportError as e:
    print(f'⚠️ packing_kernel 임포트 실패 (Windows에서는 정상): {e}')

try:
    import train_maskable_ppo
    print('✅ train_maskable_ppo 임포트 성공')
except ImportError as e:
    print(f'⚠️ train_maskable_ppo 임포트 실패 (Windows에서는 정상): {e}')
" 2>/dev/null || echo "⚠️ Windows 환경에서는 일부 모듈 임포트가 제한될 수 있습니다."

echo "=== Windows GitBash narwhals 에러 해결 완료 ==="
echo ""
echo "🎯 Windows에서 해결 완료! KAMP 서버에서 다음 명령어로 학습하세요:"
echo "python -m src.train_maskable_ppo --timesteps 20000 --eval-freq 1500 --container-size 10 10 10 --num-boxes 32 --curriculum-learning --improved-rewards"
echo ""
echo "📁 이 스크립트를 KAMP 서버에 업로드하려면:"
echo "git add ."
echo "git commit -m 'narwhals 에러 해결 스크립트 추가'"
echo "git push origin main"