"""
Device utilities for automatic GPU/CPU selection
GPU가 있을 경우 GPU를 사용하고, 없을 경우 CPU를 사용하도록 자동 선택
"""
import torch
import os
import logging

logger = logging.getLogger(__name__)


def get_device(force_cpu: bool = False) -> str:
    """
    자동으로 사용 가능한 디바이스를 선택합니다.
    
    Parameters:
    -----------
    force_cpu: bool
        True인 경우 강제로 CPU를 사용합니다.
        
    Returns:
    --------
    str: 사용할 디바이스 ('cuda' 또는 'cpu')
    """
    if force_cpu:
        logger.info("강제로 CPU 사용")
        return "cpu"
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU 사용 가능: {device_name} (총 {device_count}개)")
        return "cuda"
    else:
        logger.info("GPU 사용 불가능, CPU 사용")
        return "cpu"


def set_torch_device(device: str = None):
    """
    PyTorch 디바이스 설정
    
    Parameters:
    -----------
    device: str
        사용할 디바이스. None인 경우 자동 선택
    """
    if device is None:
        device = get_device()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' if device == 'cuda' else ''
    torch.set_default_device(device)
    
    logger.info(f"PyTorch 디바이스 설정: {device}")
    
    if device == 'cuda':
        logger.info(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def get_sb3_device(device: str = None) -> str:
    """
    Stable-Baselines3용 디바이스 문자열 반환
    
    Parameters:
    -----------
    device: str
        사용할 디바이스. None인 경우 자동 선택
        
    Returns:
    --------
    str: SB3에서 사용할 디바이스 문자열
    """
    if device is None:
        device = get_device()
    
    return device


def print_device_info():
    """현재 디바이스 정보를 출력합니다."""
    print("=== 디바이스 정보 ===")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {properties.name}")
            print(f"  메모리: {properties.total_memory / 1e9:.1f} GB")
    else:
        print("GPU 없음 - CPU 사용")
    print("==================")


if __name__ == "__main__":
    print_device_info()
    device = get_device()
    print(f"선택된 디바이스: {device}") 