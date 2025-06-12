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
    학습을 위한 디바이스 설정을 반환합니다.
    
    Args:
        verbose (bool): 상세 정보 출력 여부
        
    Returns:
        dict: 학습 설정 정보
    """
    device = get_device()
    gpu_available, device_info = check_gpu_availability()
    
    # GPU가 있을 경우 최적화된 설정
    if gpu_available:
        config = {
            "device": device,
            "n_steps": 2048,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "n_epochs": 10,
            "use_mixed_precision": True,
            "pin_memory": True,
        }
    else:
        # CPU에서는 더 작은 배치 크기 사용
        config = {
            "device": device,
            "n_steps": 512,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "n_epochs": 4,
            "use_mixed_precision": False,
            "pin_memory": False,
        }
    
    if verbose:
        print(f"학습 설정:")
        print(f"  - 디바이스: {device}")
        print(f"  - 배치 크기: {config['batch_size']}")
        print(f"  - 학습률: {config['learning_rate']}")
        print(f"  - 스텝 수: {config['n_steps']}")
    
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