#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题 - 学生代码模板

本项目要求实现打靶法和scipy.solve_bvp两种方法来求解二阶线性常微分方程边值问题：
u''(x) = -π(u(x)+1)/4
边界条件：u(0) = 1, u(1) = 1

学生姓名：[YOUR_NAME]
学号：[YOUR_STUDENT_ID]
完成日期：[COMPLETION_DATE]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')


def ode_system_shooting(t, y):
    """
    Define the ODE system for shooting method.
    
    Convert the second-order ODE u'' = -π(u+1)/4 into a first-order system:
    y1 = u, y2 = u'
    y1' = y2
    y2' = -π(y1+1)/4
    
    Args:
        t (float): Independent variable (time/position)
        y (array): State vector [y1, y2] where y1=u, y2=u'
    
    Returns:
        list: Derivatives [y1', y2']
    """
    return [y[1], -np.pi*(y[0]+1)/4]


def boundary_conditions_scipy(ya, yb):
    """
    Define boundary conditions for scipy.solve_bvp.
    
    Boundary conditions: u(0) = 1, u(1) = 1
    ya[0] should equal 1, yb[0] should equal 1
    
    Args:
        ya (array): Values at left boundary [u(0), u'(0)]
        yb (array): Values at right boundary [u(1), u'(1)]
    
    Returns:
        array: Boundary condition residuals
    """
    return np.array([ya[0] - 1, yb[0] - 1])


def ode_system_scipy(x, y):
    """
    Define the ODE system for scipy.solve_bvp.
    
    Note: scipy.solve_bvp uses (x, y) parameter order, different from odeint
    
    Args:
        x (float): Independent variable
        y (array): State vector [y1, y2]
    
    Returns:
        array: Derivatives as column vector
    """
    return np.vstack([y[1], -np.pi*(y[0]+1)/4])


def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, max_iterations=10, tolerance=1e-6):
    """
    Solve boundary value problem using shooting method.
    
    Algorithm:
    1. Guess initial slope m1
    2. Solve IVP with initial conditions [u(0), m1]
    3. Check if u(1) matches boundary condition
    4. If not, adjust slope using secant method and repeat
    
    Args:
        x_span (tuple): Domain (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): Number of discretization points
        max_iterations (int): Maximum iterations for shooting
        tolerance (float): Convergence tolerance
    
    Returns:
        tuple: (x_array, y_array) solution arrays
    """
    # 提取边界条件和设置域
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    
    # 设置x网格
    x_array = np.linspace(x_start, x_end, n_points)
    
    # 初始猜测两个不同的斜率
    m0 = 0.0  # 第一个斜率猜测
    m1 = 1.0  # 第二个斜率猜测
    
    # 解第一个IVP
    sol0 = solve_ivp(ode_system_shooting, x_span, [u_left, m0], t_eval=x_array, method='RK45')
    u0 = sol0.y[0][-1]  # 在x=1处的解
    
    # 迭代直到收敛或达到最大迭代次数
    for i in range(max_iterations):
        # 解第二个IVP
        sol1 = solve_ivp(ode_system_shooting, x_span, [u_left, m1], t_eval=x_array, method='RK45')
        u1 = sol1.y[0][-1]  # 在x=1处的解
        
        # 检查收敛性
        if abs(u1 - u_right) < tolerance:
            return x_array, sol1.y
        
        # 使用割线法更新斜率猜测
        m2 = m1 - (u1 - u_right) * (m1 - m0) / (u1 - u0)
        
        # 更新猜测
        m0, m1 = m1, m2
        u0 = u1
    
    # 如果未收敛，返回最后一次尝试的结果
    return x_array, sol1.y


def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    """
    Solve boundary value problem using scipy.solve_bvp.
    
    Args:
        x_span (tuple): Domain (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): Number of initial mesh points
    
    Returns:
        tuple: (x_array, y_array) solution arrays
    """
    # 设置初始网格
    x_start, x_end = x_span
    x = np.linspace(x_start, x_end, n_points)
    
    # 设置初始猜测
    y_guess = np.zeros((2, x.size))
    y_guess[0] = 1.0  # u的初始猜测为1
    
    # 调用scipy.solve_bvp
    sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x, y_guess)
    
    # 提取并返回解决方案
    x_array = np.linspace(x_start, x_end, 100)
    y_array = sol.sol(x_array)
    
    return x_array, y_array


def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    """
    Compare shooting method and scipy.solve_bvp, generate comparison plot.
    
    Args:
        x_span (tuple): Domain for the problem
        boundary_conditions (tuple): Boundary values (left, right)
        n_points (int): Number of points for plotting
    
    Returns:
        dict: Dictionary containing solutions and analysis
    """
    # 使用两种方法求解
    x_shooting, y_shooting = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)
    x_scipy, y_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points)
    
    # 创建比较图
    plt.figure(figsize=(10, 6))
    plt.plot(x_shooting, y_shooting[0], 'b-', label='Shooting Method')
    plt.plot(x_scipy, y_scipy[0], 'r--', label='scipy.solve_bvp')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Comparison of Shooting Method and scipy.solve_bvp')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 计算差异
    # 在相同的点上评估两种方法
    x_common = np.linspace(x_span[0], x_span[1], n_points)
    u_shooting = np.interp(x_common, x_shooting, y_shooting[0])
    u_scipy = np.interp(x_common, x_scipy, y_scipy[0])
    max_difference = np.max(np.abs(u_shooting - u_scipy))
    mean_difference = np.mean(np.abs(u_shooting - u_scipy))
    
    print(f"两种方法的最大差异: {max_difference:.6f}")
    print(f"两种方法的平均差异: {mean_difference:.6f}")
    
    # 返回分析结果
    return {
        'shooting': {'x': x_shooting, 'u': y_shooting[0]},
        'scipy': {'x': x_scipy, 'u': y_scipy[0]},
        'differences': {'max': max_difference, 'mean': mean_difference}
    }


# Test functions for development and debugging
def test_ode_system():
    """
    Test the ODE system implementation.
    """
    print("Testing ODE system...")
    try:
        # Test point
        t_test = 0.5
        y_test = np.array([1.0, 0.5])
        
        # Test shooting method ODE system
        dydt = ode_system_shooting(t_test, y_test)
        print(f"ODE system (shooting): dydt = {dydt}")
        
        # Test scipy ODE system
        dydt_scipy = ode_system_scipy(t_test, y_test)
        print(f"ODE system (scipy): dydt = {dydt_scipy}")
        
    except NotImplementedError:
        print("ODE system functions not yet implemented.")


def test_boundary_conditions():
    """
    Test the boundary conditions implementation.
    """
    print("Testing boundary conditions...")
    try:
        ya = np.array([1.0, 0.5])  # Left boundary
        yb = np.array([1.0, -0.3])  # Right boundary
        
        bc_residual = boundary_conditions_scipy(ya, yb)
        print(f"Boundary condition residuals: {bc_residual}")
        
    except NotImplementedError:
        print("Boundary conditions function not yet implemented.")


if __name__ == "__main__":
    print("项目2：打靶法与scipy.solve_bvp求解边值问题")
    print("=" * 50)
    
    # Run basic tests
    test_ode_system()
    test_boundary_conditions()
    
    # Try to run comparison (will fail until functions are implemented)
    try:
        print("\nTesting method comparison...")
        results = compare_methods_and_plot()
        print("Method comparison completed successfully!")
    except NotImplementedError as e:
        print(f"Method comparison not yet implemented: {e}")
    
    print("\n请实现所有标记为 TODO 的函数以完成项目。")
