#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：二阶常微分方程边值问题数值解法 - 学生代码模板

本项目要求实现两种数值方法求解边值问题：
1. 有限差分法 (Finite Difference Method)
2. scipy.integrate.solve_bvp 方法

问题设定：
y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2
边界条件：y(0) = 0, y(5) = 3

学生需要完成所有标记为 TODO 的函数实现。
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
    
    方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2
    边界条件：y(0) = 0, y(5) = 3
    
    Args:
        n (int): 内部网格点数量
    
    Returns:
        tuple: (x_grid, y_solution)
            x_grid (np.ndarray): 包含边界点的完整网格
            y_solution (np.ndarray): 对应的解值
    """
    # 创建网格点
    h = 5 / (n + 1)
    x_grid = np.linspace(0, 5, n + 2)
    
    # 初始化系数矩阵 A 和右端向量 b
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    # 构建线性系统
    for i in range(n):
        x = x_grid[i + 1]  # 内部点坐标
        
        # 中心差分近似 y'' 和 y'
        a = 1 / (h ** 2) - np.sin(x) / (2 * h)
        b_coef = -2 / (h ** 2) + np.exp(x)
        c = 1 / (h ** 2) + np.sin(x) / (2 * h)
        
        # 填充系数矩阵
        if i > 0:
            A[i, i - 1] = a
        A[i, i] = b_coef
        if i < n - 1:
            A[i, i + 1] = c
        
        # 填充右端向量
        b[i] = x ** 2
    
    # 处理边界条件对右端向量的影响
    b[0] -= a * 0  # y(0) = 0
    b[-1] -= c * 3  # y(5) = 3
    
    # 求解线性系统
    y_inner = solve(A, b)
    
    # 合并边界条件和内部解
    y_solution = np.zeros(n + 2)
    y_solution[0] = 0
    y_solution[1:-1] = y_inner
    y_solution[-1] = 3
    
    return x_grid, y_solution


# ============================================================================
# 方法2：scipy.integrate.solve_bvp 方法
# ============================================================================

def ode_system_for_solve_bvp(x, y):
    """
    为 scipy.integrate.solve_bvp 定义ODE系统。
    
    将二阶ODE转换为一阶系统：
    y[0] = y(x)
    y[1] = y'(x)
    
    系统方程：
    dy[0]/dx = y[1]
    dy[1]/dx = -sin(x) * y[1] - exp(x) * y[0] + x^2
    
    Args:
        x (float or array): 自变量
        y (array): 状态变量 [y, y']
    
    Returns:
        array: 导数 [dy/dx, dy'/dx]
    """
    return np.vstack((y[1], -np.sin(x) * y[1] - np.exp(x) * y[0] + x**2))


def boundary_conditions_for_solve_bvp(ya, yb):
    """
    为 scipy.integrate.solve_bvp 定义边界条件。
    
    Args:
        ya (array): 左边界处的状态 [y(0), y'(0)]
        yb (array): 右边界处的状态 [y(5), y'(5)]
    
    Returns:
        array: 边界条件残差 [y(0) - 0, y(5) - 3]
    """
    return np.array([ya[0], yb[0] - 3])


def solve_bvp_scipy(n_initial_points=11):
    """
    使用 scipy.integrate.solve_bvp 求解BVP，优化初始猜测与求解参数提升精度。
    
    Args:
        n_initial_points (int): 初始网格点数
    
    Returns:
        tuple: (x_solution, y_solution)
            x_solution (np.ndarray): 解的 x 坐标数组
            y_solution (np.ndarray): 解的 y 坐标数组
    """
    # 创建初始网格，更密集的初始网格可能有助于提升精度
    x_initial = np.linspace(0, 5, n_initial_points)
    
    # 改进初始猜测：不再简单线性猜测，基于问题特性构造更合理初始值
    # 这里假设初始猜测 y 先按线性，y' 初始设为一个小常数（可根据问题调整）
    y_initial = np.zeros((2, n_initial_points))
    y_initial[0] = 3 * x_initial / 5  # 基础线性猜测 y(0)=0, y(5)=3 
    y_initial[1] = 0.1  # 给 y' 一个初始小值，可根据实际情况微调
    
    # 调用 solve_bvp，设置更严格的求解公差，提升精度
    solution = solve_bvp(
        ode_system_for_solve_bvp, 
        boundary_conditions_for_solve_bvp, 
        x_initial, 
        y_initial, 
        rtol=1e-5,  # 相对公差，调小提升精度
        atol=1e-6   # 绝对公差，调小提升精度
    )
    
    # 检查求解是否成功
    if not solution.success:
        raise RuntimeError(f"BVP求解失败: {solution.message}")
    
    # 提取解，使用更密集的网格输出结果，方便对比
    x_solution = np.linspace(0, 5, 500)
    y_solution = solution.sol(x_solution)[0]
    
    return x_solution, y_solution


# ============================================================================
# 主程序：测试和比较两种方法
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("二阶常微分方程边值问题数值解法比较")
    print("方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2")
    print("边界条件：y(0) = 0, y(5) = 3")
    print("=" * 60)
    
    # 设置参数
    n_points = 50  # 有限差分法的内部网格点数
    
    try:
        # 方法1：有限差分法
        print("\n1. 有限差分法求解...")
        x_fd, y_fd = solve_bvp_finite_difference(n_points)
        print(f"   网格点数：{len(x_fd)}")
        print(f"   y(0) = {y_fd[0]:.6f}, y(5) = {y_fd[-1]:.6f}")
        
    except NotImplementedError:
        print("   有限差分法尚未实现")
        x_fd, y_fd = None, None
    
    try:
        # 方法2：scipy.integrate.solve_bvp
        print("\n2. scipy.integrate.solve_bvp 求解...")
        x_scipy, y_scipy = solve_bvp_scipy()
        print(f"   网格点数：{len(x_scipy)}")
        print(f"   y(0) = {y_scipy[0]:.6f}, y(5) = {y_scipy[-1]:.6f}")
        
    except NotImplementedError:
        print("   solve_bvp 方法尚未实现")
        x_scipy, y_scipy = None, None
    
    # 绘图比较
    plt.figure(figsize=(12, 8))
    
    # 子图1：解的比较
    plt.subplot(2, 1, 1)
    if x_fd is not None and y_fd is not None:
        plt.plot(x_fd, y_fd, 'b-o', markersize=3, label='Finite Difference Method', linewidth=2)
    if x_scipy is not None and y_scipy is not None:
        plt.plot(x_scipy, y_scipy, 'r--', label='scipy.integrate.solve_bvp', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Comparison of Numerical Solutions for BVP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：解的差异（如果两种方法都实现了）
    plt.subplot(2, 1, 2)
    if (x_fd is not None and y_fd is not None and 
        x_scipy is not None and y_scipy is not None):
        
        # 将 scipy 解插值到有限差分网格上进行比较
        y_scipy_interp = np.interp(x_fd, x_scipy, y_scipy)
        difference = np.abs(y_fd - y_scipy_interp)
        
        plt.semilogy(x_fd, difference, 'g-', linewidth=2, label='|Finite Diff - solve_bvp|')
        plt.xlabel('x')
        plt.ylabel('Absolute Difference')
        plt.title('Absolute Difference Between Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 数值比较
        max_diff = np.max(difference)
        mean_diff = np.mean(difference)
        print(f"\n数值比较：")
        print(f"   最大绝对误差：{max_diff:.2e}")
        print(f"   平均绝对误差：{mean_diff:.2e}")
    else:
        plt.text(0.5, 0.5, 'Need both methods implemented\nfor comparison', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Difference Plot (Not Available)')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=" * 60)
    print("实验完成！")
    print("请在实验报告中分析两种方法的精度、效率和适用性。")
    print("=" * 60)    
