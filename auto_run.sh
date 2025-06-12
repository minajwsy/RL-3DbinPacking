#!/bin/bash
# 활용도와 안정성이 학습 진행에 따라 개선되도록 하기 위해:
# 1.활용도 계산방식 개선 2.데이터셋 생성방식 개선 3.모델 학습목표와 평가지표 연결 4.활용도 그래프에 추세선 추가 !
# 사용법: ./auto_run.sh [브랜치명]   <- 기본값은 master (활용도 최적화 모드로 실행)

# 오류 발생시엔 스크립트 중단  
set -e

# 브랜치 설정 (인자로 전달하거나 기본값 사용)
BRANCH=${1:-master}

# 환경 변수 설정
WORK_DIR="${HOME}/PJW-repository"
REPO_URL="https://github.com/minajwsy/PJW-repository.git"
LOG_DIR="${WORK_DIR}/logs"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# 로그 설정
mkdir -p "${LOG_DIR}"
exec 1> >(tee "${LOG_DIR}/auto_run_utilization_${TIMESTAMP}.log")
exec 2>&1

echo "=== 활용도 최적화 모드로 브랜치 '${BRANCH}' 에서 작업 시작: ${TIMESTAMP} ==="

# Git 저장소 확인
if [ ! -d "${WORK_DIR}/.git" ]; then
    echo "Git 저장소가 없습니다. 클론을 시작합니다."
    git clone "${REPO_URL}" "${WORK_DIR}"
fi

# 작업 디렉토리 설정
cd "${WORK_DIR}" || exit 1

# Git 상태 정리 및 동기화
echo "=== Git 저장소 동기화 중 ==="
# 원격(git서버)에서 삭제된 브랜치를 로컬(kamp서버)에서도 자동으로 제거 위해 --prune 옵션 추가
git fetch origin --prune 

# 원격(git서버)에서 삭제된 로컬(kamp서버) 브랜치 정리
echo "=== 삭제된 원격 브랜치 정리 중 ==="
for local_branch in $(git branch | grep -v "^*" | tr -d ' '); do
  if [ "${local_branch}" != "${BRANCH}" ] && ! git ls-remote --heads origin ${local_branch} | grep -q ${local_branch}; then
    echo "로컬 브랜치 '${local_branch}'가 원격에 없으므로 삭제합니다."
    git branch -D ${local_branch} || true
  fi
done

# 브랜치 체크아웃 또는 생성 (중요: 이 부분 추가)
git checkout ${BRANCH} 2>/dev/null || git checkout -b ${BRANCH} origin/${BRANCH} 2>/dev/null || git checkout -b ${BRANCH}

# 원격 브랜치가 있으면 동기화
if git ls-remote --heads origin ${BRANCH} | grep -q ${BRANCH}; then
    git reset --hard "origin/${BRANCH}"
fi

git clean -fd || true

# 실험 환경 준비
cd Online-3D-BPP-DRL
touch __init__.py experiments/__init__.py
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# GPU 상태 확인
if ! nvidia-smi > /dev/null 2>&1; then
    echo "경고: GPU를 찾을 수 없습니다"
fi

# 실험 실행
echo "=== 활용도 최적화 PPO 실험 시작: $(date '+%Y-%m-%d %H:%M:%S') ==="
cd "${WORK_DIR}/Online-3D-BPP-DRL"
export PYTHONPATH="${PWD}:${PYTHONPATH}"
if python -m src.ppo.train_ppo 2>&1 | tee results/experOut.txt; then
    echo "활용도 최적화 PPO 실험이 성공적으로 완료되었습니다."
else
    echo "실험 실행 중 오류가 발생했습니다. 오류 로그를 확인하세요."
    # 오류가 있더라도 계속 진행
fi

# 실험이 성공/실패와 관계없이 5초 대기
echo "다음 단계로 진행하기 전에 5초 대기합니다..."
sleep 5

# 통합 테스트 실행
echo "=== 통합 테스트 시작: $(date '+%Y-%m-%d %H:%M:%S') ==="
python unified_test.py 2>&1 | tee results/test_results_latest.txt

# 결과 처리 
# 상위 디렉토리로 이동하여 Git 저장소 루트에서 작업
cd "${WORK_DIR}"

# 변경사항 확인 및 커밋
echo "=== 변경사항 확인 중 ==="
git status

# 파일 존재 여부 확인 후 추가
echo "=== 결과 파일 추가 중 ==="

# experOut.txt 파일 추가
if [ -f "Online-3D-BPP-DRL/results/experOut.txt" ]; then
    git add "Online-3D-BPP-DRL/results/experOut.txt"
    echo "experOut.txt 파일이 추가되었습니다."
else
    echo "experOut.txt 파일이 존재하지 않습니다."
fi

# test_results_latest.txt 파일 추가
if [ -f "Online-3D-BPP-DRL/results/test_results_latest.txt" ]; then
    git add "Online-3D-BPP-DRL/results/test_results_latest.txt"
    echo "test_results_latest.txt 파일이 추가되었습니다."
else
    echo "test_results_latest.txt 파일이 존재하지 않습니다."
fi

# 타임스탬프가 포함된 테스트 결과 파일 추가
TEST_RESULTS_FILES=$(find "Online-3D-BPP-DRL/results" -name "test_results_*.txt" 2>/dev/null)
if [ -n "$TEST_RESULTS_FILES" ]; then
    git add $TEST_RESULTS_FILES
    echo "타임스탬프가 포함된 테스트 결과 파일이 추가되었습니다."
else
    echo "타임스탬프가 포함된 테스트 결과 파일이 존재하지 않습니다."
fi

# PNG 파일 추가
PNG_FILES=$(find "Online-3D-BPP-DRL/results" -name "*.png" 2>/dev/null)
if [ -n "$PNG_FILES" ]; then
    git add $PNG_FILES
    echo "PNG 파일이 추가되었습니다."
else
    echo "PNG 파일이 존재하지 않습니다."
fi

# 변경사항이 있는지 확인
if git diff --cached --quiet; then
    echo "=== 변경사항 없음 ==="
else
    # 커밋 및 푸시 (실험 모드에 따라 메시지 구분)
    git commit -m "활용도 최적화 실험 및 테스트 결과 업데이트: ${TIMESTAMP}"
    git push origin "${BRANCH}"
    echo "=== 결과 업로드 완료 ==="
fi

echo "=== 작업 완료: $(date '+%Y-%m-%d %H:%M:%S') ==="