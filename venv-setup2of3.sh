# GPU 확인 및 적절한 의존성 설치
echo "=== GPU 상태 확인 및 의존성 설치 중 ==="

# uv.lock 파일 삭제하여 새로 생성하도록 함
echo "=== 기존 lock 파일 정리 중 ==="
rm -f uv.lock

if nvidia-smi > /dev/null 2>&1; then
    echo "GPU가 감지되었습니다. GPU 버전 패키지를 설치합니다."
    nvidia-smi
    
    # GPU 버전 PyTorch 설치
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # 필수 패키지 설치 (narwhals 포함)
    echo "=== 필수 패키지 개별 설치 중 ==="
    # 핵심 패키지들을 개별적으로 설치
    uv pip install gymnasium numpy nptyping pillow pandas jupyter pytest
    uv pip install sb3-contrib stable-baselines3[extra] tensorboard matplotlib opencv-python tqdm rich
    
    # narwhals 및 관련 의존성 설치
    echo "=== 누락된 의존성 설치 중 ==="
    uv pip install narwhals
    uv pip install tenacity packaging
    uv pip install rich tqdm

    # plotly 재설치 (의존성 포함)
    echo "=== plotly 재설치 중 ==="
    uv pip uninstall plotly -- yes || true
    uv pip install plotly

    # plotly_gif 설치 (누락된 의존성)
    echo "=== plotly_gif 설치 중 ==="
    uv pip install plotly_gif
    
    # stable-baselines3[extra] 재설치 (rich, tqdm 포함)
    #echo "=== stable-baselines3[extra] 재설치 중 ==="
    #uv pip uninstall stable-baselines3 -- yes || true
    #uv pip install stable-baselines3[extra]

    echo "GPU 환경 설정 완료 (narwhals 및 plotly_gif 포함)"
else
    echo "GPU를 찾을 수 없습니다. CPU 버전 패키지를 설치합니다."
    
    # CPU 버전 PyTorch 설치
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # 필수 패키지 설치 (narwhals 포함)
    echo "=== 필수 패키지 개별 설치 중 ==="
    # 핵심 패키지들을 개별적으로 설치
    uv pip install gymnasium numpy nptyping pillow pandas jupyter pytest
    uv pip install sb3-contrib stable-baselines3[extra] tensorboard matplotlib opencv-python tqdm rich
    
    # narwhals 및 관련 의존성 설치
    echo "=== 누락된 의존성 설치 중 ==="
    uv pip install narwhals
    uv pip install tenacity packaging
    uv pip install rich tqdm

    # plotly 재설치 (의존성 포함)
    echo "=== plotly 재설치 중 ==="
    uv pip uninstall plotly -- yes || true
    uv pip install plotly

    # plotly_gif 설치 (누락된 의존성)
    echo "=== plotly_gif 설치 중 ==="
    uv pip install plotly_gif

    # stable-baselines3[extra] 재설치 (rich, tqdm 포함)
    #echo "=== stable-baselines3[extra] 재설치 중 ==="
    #uv pip uninstall stable-baselines3 -- yes || true
    #uv pip install stable-baselines3[extra]
    
    echo "CPU 환경 설정 완료 (narwhals 및 plotly_gif 포함)"
fi