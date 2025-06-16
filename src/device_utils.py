"""
Device utilities for automatic GPU/CPU selection
"""
import logging
import torch
import warnings
from typing import Optional, Tuple

# GPU 관련 경고 메시지 억제
warnings.filterwarnings("ignore", category=UserWarning, message=".*CUDA.*")


def check_gpu_availability() -> Tuple[bool, str]:
    """
    GPU 사용 가능 여부를 확인합니다.
    
    Returns:
        Tuple[bool, str]: (GPU 사용 가능 여부, 디바이스 정보)
    """
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            return True, f"GPU 사용 가능: {gpu_name} (총 {gpu_count}개)"
        else:
            return False, "GPU 사용 불가능, CPU 사용"
    except Exception as e:
        logging.warning(f"GPU 확인 중 오류 발생: {e}")
        return False, "GPU 확인 실패, CPU 사용"


def get_device(force_cpu: bool = False) -> torch.device:
    """
    최적의 디바이스를 자동으로 선택합니다.
    
    Args:
        force_cpu (bool): CPU 강제 사용 여부
        
    Returns:
        torch.device: 선택된 디바이스
    """
    if force_cpu:
        device = torch.device("cpu")
        print("CPU 강제 사용")
        return device
    
    gpu_available, gpu_info = check_gpu_availability()
    print(f"디바이스 정보: {gpu_info}")
    
    if gpu_available:
        device = torch.device("cuda")
        print("GPU 환경에서 실행")
    else:
        device = torch.device("cpu")
        print("CPU 환경에서 실행")
    
    return device


def setup_training_device(verbose: bool = True) -> dict:
    """
    학습 디바이스 설정 및 최적화된 하이퍼파라미터 반환 (개선된 버전)
    
    Returns:
        dict: 디바이스별 최적화된 하이퍼파라미터
    """
    device = get_device()
    
    if device.type == "cuda":
        # GPU 사용 시 더 적극적인 하이퍼파라미터
        config = {
            "learning_rate": 5e-4,  # 더 높은 학습률
            "n_steps": 2048,
            "batch_size": 512,      # 더 큰 배치 크기
            "n_epochs": 15,         # 더 많은 에포크
            "gamma": 0.995,         # 높은 감가율
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.02,       # 더 높은 엔트로피 계수 (탐험 장려)
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "device": device
        }
        if verbose:
            print(f"🚀 GPU 최적화 모드 활성화: {torch.cuda.get_device_name()}")
            try:
                print(f"CUDA 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            except:
                print("CUDA 메모리 정보를 가져올 수 없습니다")
            print("높은 성능 하이퍼파라미터 적용")
    else:
        # CPU 사용 시에도 개선된 하이퍼파라미터
        config = {
            "learning_rate": 3e-4,  # CPU에서도 더 높은 학습률
            "n_steps": 1024,
            "batch_size": 256,      # 더 큰 배치 크기
            "n_epochs": 10,         # 더 많은 에포크
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "device": device
        }
        if verbose:
            print("🖥️  CPU 최적화 모드 (개선된 하이퍼파라미터)")
    
    if verbose:
        print(f"📊 개선된 하이퍼파라미터:")
        for key, value in config.items():
            if key != 'device':
                print(f"   - {key}: {value}")
    
    return config


def log_system_info():
    """시스템 정보를 로그에 기록합니다."""
    import platform
    import sys
    
    print("=== 시스템 정보 ===")
    print(f"플랫폼: {platform.platform()}")
    print(f"Python 버전: {sys.version}")
    print(f"PyTorch 버전: {torch.__version__}")
    
    gpu_available, gpu_info = check_gpu_availability()
    print(f"GPU 정보: {gpu_info}")
    
    if gpu_available:
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"cuDNN 버전: {torch.backends.cudnn.version()}")
        print(f"GPU 메모리:")
        for i in range(torch.cuda.device_count()):
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {memory_total:.1f} GB")
    
    print("==================")


if __name__ == "__main__":
    log_system_info()
    device = get_device()
    config = setup_training_device()
    print(f"선택된 디바이스: {device}")
    print(f"학습 설정: {config}") 