# KAMP 서버 narwhals 에러 해결 가이드

## 🚨 문제 상황
```
ModuleNotFoundError: No module named 'narwhals'
```

이 에러는 plotly의 최신 버전이 narwhals 의존성을 요구하는데, kamp_auto_run.sh에서 `--no-deps` 옵션으로 plotly를 설치하면서 필요한 의존성이 누락되었기 때문입니다.

## 🔧 즉시 해결 방법

### 방법 1: 자동 해결 스크립트 실행 (권장)

**KAMP 서버 (Linux):**
```bash
cd ~/RL-3DbinPacking
chmod +x fix_narwhals_error.sh
./fix_narwhals_error.sh
```

**로컬 GitBash (Windows):**
```bash
cd /c/Users/박정우/Documents/RL-3DbinPacking
chmod +x fix_narwhals_error.sh
./fix_narwhals_error.sh
```

### 방법 2: 수동 명령어 실행

**KAMP 서버 (Linux):**
```bash
# 1. 가상환경 활성화 확인
source .venv/bin/activate

# 2. 누락된 의존성 설치
uv pip install narwhals tenacity packaging

# 3. plotly 재설치 (의존성 포함)
uv pip uninstall plotly -y
uv pip install plotly

# 4. 설치 확인
python -c "import narwhals; import plotly; print('✅ 설치 완료')"
```

**로컬 GitBash (Windows):**
```bash
# 1. 가상환경 활성화 (Windows 경로)
source .venv/Scripts/activate

# 2. 누락된 의존성 설치
uv pip install narwhals tenacity packaging

# 3. plotly 재설치 (의존성 포함)
uv pip uninstall plotly -y
uv pip install plotly

# 4. 설치 확인
python -c "import narwhals; import plotly; print('✅ 설치 완료')"
```

### 방법 3: 개별 패키지 설치

**KAMP 서버 (Linux):**
```bash
# 가상환경 활성화
source .venv/bin/activate

# 필수 의존성 개별 설치
uv pip install narwhals==1.13.5
uv pip install tenacity==8.2.3
uv pip install packaging==24.0

# plotly 완전 재설치
uv pip uninstall plotly _plotly_utils -y
uv pip install plotly==5.17.0
```

**로컬 GitBash (Windows):**
```bash
# 가상환경 활성화 (Windows 경로)
source .venv/Scripts/activate

# 필수 의존성 개별 설치
uv pip install narwhals==1.13.5
uv pip install tenacity==8.2.3
uv pip install packaging==24.0

# plotly 완전 재설치
uv pip uninstall plotly _plotly_utils -y
uv pip install plotly==5.17.0
```

## 🔍 설치 확인

다음 명령어로 모든 패키지가 정상 설치되었는지 확인:

```bash
python -c "
import narwhals
import plotly
import plotly.express as px
from _plotly_utils.basevalidators import ColorscaleValidator
print('✅ 모든 패키지 정상 설치 확인')
print(f'narwhals 버전: {narwhals.__version__}')
print(f'plotly 버전: {plotly.__version__}')
"
```

## 🚀 학습 재시작

에러 해결 후 다음 명령어로 학습을 재시작:

**KAMP 서버:**
```bash
python -m src.train_maskable_ppo \
    --timesteps 20000 \
    --eval-freq 1500 \
    --container-size 10 10 10 \
    --num-boxes 32 \
    --curriculum-learning \
    --improved-rewards
```

**로컬 GitBash (테스트용):**
```bash
# Windows에서는 전체 학습보다는 간단한 테스트만 권장
python -c "
import sys
sys.path.append('src')
try:
    import narwhals, plotly
    print('✅ 의존성 해결 완료 - KAMP 서버에서 학습 가능')
except ImportError as e:
    print(f'❌ 여전히 문제가 있습니다: {e}')
"
```

## 📋 근본 원인 및 예방

### 원인
- plotly 최신 버전(5.17+)이 narwhals 의존성 요구
- `--no-deps` 옵션으로 설치 시 필수 의존성 누락

### 예방책
- `kamp_auto_run.sh` 스크립트가 수정되어 향후 자동으로 해결됨
- narwhals, tenacity, packaging을 plotly 설치 전에 미리 설치

## 🔄 수정된 kamp_auto_run.sh

향후 실행에서는 다음과 같이 개선된 설치 로직이 적용됩니다:

```bash
# plotly 관련 의존성 설치 (narwhals 포함)
echo "=== plotly 및 의존성 설치 중 ==="
uv pip install narwhals tenacity packaging
uv pip install plotly --no-deps  # 핵심 의존성은 별도 설치했으므로 --no-deps 사용
```

## 🆘 추가 문제 해결

### 만약 여전히 에러가 발생한다면:

1. **완전한 환경 재구축**:
   ```bash
   rm -rf .venv
   uv venv .venv --python 3.11
   source .venv/bin/activate
   ./kamp_auto_run.sh
   ```

2. **의존성 버전 확인**:
   ```bash
   uv pip list | grep -E "(narwhals|plotly|tenacity|packaging)"
   ```

3. **Python 경로 확인**:
   ```bash
   python -c "import sys; print('\n'.join(sys.path))"
   ```

## ✅ 성공 확인

다음 출력이 나오면 성공:
```
✅ narwhals 설치 확인
   버전: 1.13.5
✅ plotly 설치 확인
   버전: 5.17.0
✅ plotly.express 임포트 성공
✅ _plotly_utils.basevalidators 임포트 성공
✅ packing_kernel 임포트 성공
✅ train_maskable_ppo 임포트 성공
```

## 🎯 GitBash 사용자 빠른 실행 가이드

**Windows GitBash에서 즉시 실행:**
```bash
# 1. 프로젝트 디렉토리로 이동
cd /c/Users/박정우/Documents/RL-3DbinPacking

# 2. Windows용 해결 스크립트 실행
./fix_narwhals_error_windows.sh

# 3. KAMP 서버로 업로드
git add .
git commit -m "narwhals 에러 해결 완료"
git push origin main
```

**KAMP 서버에서 실행:**
```bash
# 1. 최신 코드 가져오기
git pull origin main

# 2. 해결 스크립트 실행
./fix_narwhals_error.sh

# 3. 학습 시작
python -m src.train_maskable_ppo --timesteps 20000 --eval-freq 1500 --container-size 10 10 10 --num-boxes 32 --curriculum-learning --improved-rewards
```

---

**🎯 이 가이드를 따라하면 Windows GitBash와 KAMP 서버 모두에서 narwhals 에러를 완전히 해결할 수 있습니다!** 