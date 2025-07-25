#!/usr/bin/env python3
"""
터미널 크래시 디버깅용 최소 테스트 스크립트
"""

import sys
import os

print("🔍 === 최소 테스트 시작 ===")
print(f"Python 버전: {sys.version}")
print(f"현재 디렉토리: {os.getcwd()}")

# 1. 기본 import 테스트
try:
    import argparse
    print("✅ argparse OK")
except Exception as e:
    print(f"❌ argparse 오류: {e}")

# 2. matplotlib 테스트 (서버 모드)
try:
    import matplotlib
    matplotlib.use('Agg')  # 서버 환경용
    import matplotlib.pyplot as plt
    print("✅ matplotlib OK")
except Exception as e:
    print(f"❌ matplotlib 오류: {e}")

# 3. numpy 테스트
try:
    import numpy as np
    print("✅ numpy OK")
except Exception as e:
    print(f"❌ numpy 오류: {e}")

# 4. 파일 존재 확인
files_to_check = [
    'src/ultimate_train_fix.py',
    'src/packing_kernel.py', 
    'src/train_maskable_ppo.py'
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"✅ {file_path} 존재 ({size} bytes)")
    else:
        print(f"❌ {file_path} 없음")

# 5. ArgumentParser 테스트
try:
    parser = argparse.ArgumentParser(description="최소 테스트")
    parser.add_argument("--test", action="store_true", help="테스트 플래그")
    args = parser.parse_args(["--help"])
    print("✅ ArgumentParser OK")
except SystemExit:
    print("✅ ArgumentParser OK (help 출력됨)")
except Exception as e:
    print(f"❌ ArgumentParser 오류: {e}")

print("🎯 === 최소 테스트 완료 ===") 