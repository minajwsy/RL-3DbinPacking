#!/usr/bin/env python3
"""
원본 ultimate_train_fix.py를 클라우드 환경용으로 패치
"""

import os
import sys

def patch_ultimate_train_fix():
    """클라우드 환경 호환을 위한 패치 적용"""
    
    print("🔧 ultimate_train_fix.py 클라우드 패치 시작...")
    
    # 원본 파일 읽기
    with open('src/ultimate_train_fix.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 클라우드 환경 최적화 적용
    patches = [
        # matplotlib 백엔드 강제 설정
        ("import matplotlib.pyplot as plt", 
         "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt"),
        
        # 메모리 최적화
        ("import warnings", 
         "import warnings\nimport gc"),
        
        # GPU 메모리 제한 (필요시)
        ("warnings.filterwarnings(\"ignore\")", 
         "warnings.filterwarnings(\"ignore\")\nos.environ.setdefault('CUDA_VISIBLE_DEVICES', '')"),
    ]
    
    # 패치 적용
    patched_content = content
    for old, new in patches:
        if old in patched_content:
            patched_content = patched_content.replace(old, new, 1)
            print(f"✅ 패치 적용: {old[:30]}...")
    
    # 패치된 파일 저장
    with open('ultimate_train_fix_cloud.py', 'w', encoding='utf-8') as f:
        f.write(patched_content)
    
    print("✅ 패치 완료: ultimate_train_fix_cloud.py 생성됨")
    print("\n🚀 클라우드에서 실행:")
    print("  python ultimate_train_fix_cloud.py --optimize --optimization-method optuna --n-trials 5")

if __name__ == "__main__":
    patch_ultimate_train_fix() 