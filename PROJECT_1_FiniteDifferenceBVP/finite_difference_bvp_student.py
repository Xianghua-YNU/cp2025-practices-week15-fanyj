#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：二阶常微分方程边值问题数值解法 - 完整实现

方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2
边界条件：y(0) = 0, y(5) = 3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.linalg import solve

# ============================================================================
# 方法1：有限差分法 (Finite Difference Method)
# ============================================================================

def solve_bvp_finite_difference(n):
    """
    使用有限差分法求解二阶常微分方程边值问题。
    """
    h = 5.0 / (n + 1)  # 计算步长
    x = np.linspace(0, 5, n+2)  # 包含边界点的网格
    
    # 初始化系数矩阵和右侧向量
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    for i in range(1, n+1):
        xi = x[i]
        # 处理不同位置的系数
        if i == 1:  # 左边界附近
            A[i-1, i-1] = -2/h**2 + np.exp(xi)
            A[i-1, i] = 1/h**2 + np.sin(xi)/(2*h)
            b[i-1] = xi**2 - 1/h**2 + np.sin(xi)/(2*h)*3  # 包含y0=0的影响
        elif i == n:  # 右边界附近
            A[i-1, i-2] = 1/h**2 - np.sin(xi)/(2*h)
            A[i-1, i-1] = -2/h**2 + np.exp(xi)
            b[i-1] = xi**2 - 1/h**2 - np.sin(xi)/(2*h)*3  # 包含y6=3的影响
        else:  # 中间点
            A[i-1, i-2] = 1/h**2 + np.sin(xi)/(2*h)
            A[i-1, i-1] = -2/h**2 + np.exp(xi)
            A[i-1, i] = 1/h**2 - np.sin(xi)/(2*h)
            b[i-1] = xi**2
    
    # 解线性方程组
    y_internal = solve(A, b)
    
    # 添加边界条件
    y_solution = np.concatenate(([0], y_internal, [3]))
    return x, y_solution

# ============================================================================
# 方法2：scipy.integrate.solve_bvp 方法
# ============================================================================

def ode_system_for_solve_bvp(x, y):
    """
    将二阶ODE转换为一阶系统：
    y0 = y(x)
    y1 = y'(x)
    """
    y0, y1 = y
    dy0dx = y1
    dy1dx = x**2 - np.sin(x)*y1 - np.exp(x)*y0
    return np.vstack((dy0dx, dy1dx))

def boundary_conditions_for_solve_bvp(ya, yb):
    """
    边界条件残差：y(0)=0 和 y(5)=3
    """
    return np.array([ya[0], yb[0]]) - [0, 3]

def solve_bvp_scipy(n_initial_points=11):
    """
    使用SciPy的solve_bvp方法求解边值问题
    """
    # 初始网格和猜测解
    x_initial = np.linspace(0, 5, n_initial_points)
    y_initial = np.linspace(0, 3, n_initial_points).reshape(-1, 1)
    
    # 求解BVP
    sol = solve_bvp(
        lambda x, y: ode_system_for_solve_bvp(x, y),
        boundary_conditions_for_solve_bvp,
        x_initial, 
        y_initial
    )
    
    if sol.success:
        return sol.x, sol.y[0]
    else:
        raise RuntimeError("BVP求解失败，检查初始猜测或方程实现")

# ============================================================================
# 主程序：测试和比较两种方法
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("二阶常微分方程边值问题数值解法比较")
    print("方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2")
    print("边界条件：y(0) = 0, y(5) = 3")
    print("="*60)
    
    n_points = 50  # 有限差分法的内部网格点数
    
    # 方法1：有限差分法
    try:
        print("\n1. 有限差分法求解...")
        x_fd, y_fd = solve_bvp_finite_difference(n_points)
        print(f"   网格点数：{len(x_fd)}")
        print(f"   y(0) = {y_fd[0]:.6f}, y(5) = {y_fd[-1]:.6f}")
    except NotImplementedError:
        print("   有限差分法尚未实现")
        x_fd, y_fd = None, None
    
    # 方法2：scipy.integrate.solve_bvp
    try:
        print("\n2. scipy.integrate.solve_bvp 求解...")
        x_scipy, y_scipy = solve_bvp_scipy()
        print(f"   网格点数：{len(x_scipy)}")
        print(f"   y(0) = {y_scipy[0]:.6f}, y(5) = {y_scipy[-1]:.6f}")
    except NotImplementedError:
        print("   solve_bvp 方法尚未实现")
        x_scipy, y_scipy = None, None
    
    # 绘图比较
    plt.figure(figsize=(12, 8))
    
    # 解的比较
    plt.subplot(2, 1, 1)
    if x_fd is not None and y_fd is not None:
        plt.plot(x_fd, y_fd, 'b-o', label='有限差分法', markersize=4)
    if x_scipy is not None and y_scipy is not None:
        plt.plot(x_scipy, y_scipy, 'r--', label='SciPy solve_bvp', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('数值解比较')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 解差异比较
    plt.subplot(2, 1, 2)
    if x_fd is not None and y_fd is not None and x_scipy is not None and y_scipy is not None:
        # 插值SciPy解到有限差分网格
        y_scipy_interp = np.interp(x_fd, x_scipy, y_scipy)
        difference = np.abs(y_fd - y_scipy_interp)
        
        plt.semilogy(x_fd, difference, 'g-', label='绝对误差')
        plt.xlabel('x')
        plt.ylabel('误差')
        plt.title('解的一致性比较')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 数值统计
        max_err = np.max(difference)
        mean_err = np.mean(difference)
        print(f"\n数值比较结果：")
        print(f"最大绝对误差：{max_err:.2e}")
        print(f"平均绝对误差：{mean_err:.2e}")
    else:
        plt.text(0.5, 0.5, '需要两种方法均成功运行进行比较',
                 ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("实验完成！请查看控制台输出和图形比较结果")
    print("="*60)
