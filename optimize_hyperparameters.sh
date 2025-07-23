#!/bin/bash

# 하이퍼파라미터 최적화 편의 스크립트
# Usage: ./optimize_hyperparameters.sh [method] [trials] [use_wandb]

echo "🚀 PPO 하이퍼파라미터 최적화 스크립트"
echo "========================================"

# 기본값 설정
METHOD=${1:-optuna}  # optuna, wandb, both
TRIALS=${2:-30}
USE_WANDB=${3:-false}

echo "📋 설정:"
echo "   - 방법: $METHOD"
echo "   - 시행 횟수: $TRIALS"
echo "   - W&B 사용: $USE_WANDB"
echo ""

# 의존성 확인
echo "🔍 의존성 확인 중..."
python -c "import optuna; print('✅ Optuna 사용 가능')" 2>/dev/null || echo "⚠️ Optuna 없음 - pip install optuna 필요"
python -c "import wandb; print('✅ W&B 사용 가능')" 2>/dev/null || echo "⚠️ W&B 없음 - pip install wandb 필요"
echo ""

# 명령어 구성
CMD="python src/ultimate_train_fix.py --optimize --optimization-method $METHOD --n-trials $TRIALS"

if [ "$USE_WANDB" = "true" ]; then
    CMD="$CMD --use-wandb --wandb-project ppo-3d-binpacking-optimization"
fi

echo "🚀 실행 명령어:"
echo "$CMD"
echo ""

# 실행
echo "⏳ 최적화 시작..."
eval $CMD

echo ""
echo "🎉 최적화 완료!"
echo "📊 결과는 results/ 폴더를 확인하세요."

# 결과 파일 목록 표시
echo ""
echo "📂 생성된 파일들:"
ls -la results/*$(date +%Y%m%d)* 2>/dev/null || echo "   결과 파일을 찾을 수 없습니다." 