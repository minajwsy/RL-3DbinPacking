#!/usr/bin/env python3
"""
MathWorks 기반 적응적 학습률 스케줄링 분석
learning_rate = lambda progress: max(1e-5, 3e-4 * (1 - progress * 0.9))
"""

import numpy as np
import matplotlib.pyplot as plt

def adaptive_learning_rate(progress):
    """MathWorks 기반 적응적 학습률"""
    return max(1e-5, 3e-4 * (1 - progress * 0.9))

def analyze_learning_rate_schedule():
    """학습률 스케줄 분석"""
    
    print("🧮 MathWorks 기반 적응적 학습률 스케줄링 분석")
    print("=" * 70)
    
    # 진행률별 학습률 계산
    progress_points = np.linspace(0, 1, 11)  # 0%, 10%, 20%, ..., 100%
    learning_rates = [adaptive_learning_rate(p) for p in progress_points]
    
    print("📊 진행률별 학습률 변화:")
    print("-" * 50)
    for i, (progress, lr) in enumerate(zip(progress_points, learning_rates)):
        step_info = f"진행률 {progress:.1%}"
        lr_info = f"학습률 {lr:.6f}"
        if i == 0:
            status = "🚀 학습 시작"
        elif i == 5:
            status = "⚖️ 학습 중간"
        elif i == 10:
            status = "🏁 학습 완료"
        else:
            status = f"📈 진행 중"
        
        print(f"{step_info:>12} → {lr_info:>18} {status}")
    
    # 감소율 분석
    print("\n📉 학습률 감소 분석:")
    print("-" * 50)
    initial_lr = learning_rates[0]
    final_lr = learning_rates[-1]
    reduction_ratio = final_lr / initial_lr
    
    print(f"시작 학습률: {initial_lr:.6f}")
    print(f"최종 학습률: {final_lr:.6f}")
    print(f"감소 비율: {reduction_ratio:.3f} ({reduction_ratio:.1%})")
    print(f"총 감소량: {1-reduction_ratio:.3f} ({(1-reduction_ratio)*100:.1f}% 감소)")
    
    # 15,000 스텝 기준 실제 학습률 변화
    print("\n🎯 15,000 스텝 학습에서의 실제 학습률:")
    print("-" * 50)
    timesteps = [0, 1500, 3000, 7500, 12000, 15000]
    for step in timesteps:
        progress = step / 15000
        lr = adaptive_learning_rate(progress)
        print(f"스텝 {step:>6} (진행률 {progress:.1%}) → 학습률 {lr:.6f}")
    
    # 다른 스케줄링 방식과 비교
    print("\n🔍 다른 학습률 스케줄링과 비교:")
    print("-" * 50)
    
    def constant_lr(progress):
        return 3e-4
    
    def linear_decay(progress):
        return 3e-4 * (1 - progress)
    
    def exponential_decay(progress):
        return 3e-4 * (0.95 ** (progress * 10))
    
    comparison_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print(f"{'진행률':>8} {'적응적':>12} {'고정':>12} {'선형감소':>12} {'지수감소':>12}")
    print("-" * 60)
    
    for progress in comparison_points:
        adaptive = adaptive_learning_rate(progress)
        constant = constant_lr(progress)
        linear = linear_decay(progress)
        exponential = exponential_decay(progress)
        
        print(f"{progress:.1%:>8} {adaptive:.6f:>12} {constant:.6f:>12} {linear:.6f:>12} {exponential:.6f:>12}")
    
    # 시각화
    progress_detailed = np.linspace(0, 1, 100)
    adaptive_rates = [adaptive_learning_rate(p) for p in progress_detailed]
    constant_rates = [constant_lr(p) for p in progress_detailed]
    linear_rates = [linear_decay(p) for p in progress_detailed]
    exponential_rates = [exponential_decay(p) for p in progress_detailed]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(progress_detailed, adaptive_rates, 'b-', linewidth=2, label='MathWorks 적응적')
    plt.plot(progress_detailed, constant_rates, 'r--', linewidth=2, label='고정 학습률')
    plt.plot(progress_detailed, linear_rates, 'g:', linewidth=2, label='선형 감소')
    plt.plot(progress_detailed, exponential_rates, 'm-.', linewidth=2, label='지수 감소')
    
    plt.xlabel('학습 진행률')
    plt.ylabel('학습률')
    plt.title('MathWorks vs 다른 학습률 스케줄링 비교')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 15,000 스텝 기준으로 환산
    plt.subplot(2, 1, 2)
    steps_detailed = progress_detailed * 15000
    plt.plot(steps_detailed, adaptive_rates, 'b-', linewidth=2, label='MathWorks 적응적')
    plt.plot(steps_detailed, constant_rates, 'r--', linewidth=2, label='고정 학습률')
    
    plt.xlabel('학습 스텝')
    plt.ylabel('학습률')
    plt.title('15,000 스텝 학습에서의 학습률 변화')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_rate_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 그래프 저장 완료: learning_rate_analysis.png")
    
    # 장단점 분석
    print("\n🎯 MathWorks 적응적 학습률의 장단점:")
    print("=" * 70)
    
    print("✅ 장점:")
    print("  1. 초기 빠른 학습: 높은 학습률로 시작하여 빠른 개선")
    print("  2. 안정적 수렴: 학습 후반부에 낮은 학습률로 안정화")
    print("  3. 최소값 보장: 1e-5 이하로 떨어지지 않음")
    print("  4. 부드러운 감소: 급격한 변화 없이 점진적 감소")
    print("  5. 조기 종료 방지: 너무 낮은 학습률로 인한 학습 중단 방지")
    
    print("\n⚠️ 단점:")
    print("  1. 고정적: progress에만 의존, 실제 성능 변화 미반영")
    print("  2. 하이퍼파라미터: 0.9 계수와 최소값이 임의로 설정")
    print("  3. 문제 의존적: 모든 문제에 최적이 아닐 수 있음")
    
    print("\n🚀 커리큘럼 학습에서의 효과:")
    print("  1. 초기 쉬운 단계: 높은 학습률로 빠른 적응")
    print("  2. 중간 단계: 적당한 학습률로 안정적 진행")  
    print("  3. 어려운 단계: 낮은 학습률로 정밀한 조정")
    print("  4. 최종 단계: 최소 학습률로 미세 튜닝")

if __name__ == "__main__":
    analyze_learning_rate_schedule() 