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
from scipy.integrate import solve_ivp, solve_bvp
import warnings
warnings.filterwarnings('ignore')

# 打靶法的ODE系统（一阶转换）
def ode_system_shooting(t, y):
    return [y[1], -np.pi * (y[0] + 1) / 4]

# scipy.solve_bvp的边界条件函数
def boundary_conditions_scipy(ya, yb):
    return np.array([ya[0] - 1, yb[0] - 1])

# scipy.solve_bvp的ODE系统（注意参数顺序和返回格式）
def ode_system_scipy(x, y):
    return np.vstack([y[1], -np.pi * (y[0] + 1) / 4])

# 打靶法求解边值问题
def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, max_iterations=10, tolerance=1e-6):
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    # 校验x_span合法性
    if not (isinstance(x_span, tuple) and len(x_span) == 2 and isinstance(x_span[0], (int, float)) and isinstance(x_span[1], (int, float)) and x_span[0] < x_span[1]):
        raise ValueError("x_span需为包含两个有效数值且左小于右的元组")
    
    x_array = np.linspace(x_start, x_end, n_points)
    m0, m1 = 0.0, 1.0  # 初始斜率猜测
    
    # 第一次求解
    sol0 = solve_ivp(ode_system_shooting, x_span, [u_left, m0], t_eval=x_array, method='RK45')
    u0 = sol0.y[0][-1] if sol0.success else np.nan
    
    for i in range(max_iterations):
        sol1 = solve_ivp(ode_system_shooting, x_span, [u_left, m1], t_eval=x_array, method='RK45')
        if not sol1.success:
            raise RuntimeError("打靶法迭代中IVP求解失败")
        u1 = sol1.y[0][-1]
        
        if abs(u1 - u_right) < tolerance:
            return x_array, sol1.y
        
        # 割线法更新斜率
        m2 = m1 - (u1 - u_right) * (m1 - m0) / (u1 - u0) if (u1 - u0) != 0 else m1 + 1e-3
        m0, m1 = m1, m2
        u0 = u1
    
    raise RuntimeError("打靶法达到最大迭代次数仍未收敛")

# scipy.solve_bvp求解封装
def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    x_start, x_end = x_span
    # 校验n_points合法性
    if not isinstance(n_points, int) or n_points <= 0:
        raise TypeError("n_points需为正整数")
    
    x = np.linspace(x_start, x_end, n_points)
    y_guess = np.zeros((2, x.size))
    y_guess[0] = 1.0  # u初始猜测
    
    sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x, y_guess)
    if not sol.success:
        raise RuntimeError("scipy.solve_bvp求解失败")
    
    # 插值得到更密集的解用于对比
    x_fine = np.linspace(x_start, x_end, 100)
    y_fine = sol.sol(x_fine)
    return x_fine, y_fine

# 方法对比与绘图
def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    try:
        x_shooting, y_shooting = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)
        x_scipy, y_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points)
    except Exception as e:
        raise RuntimeError(f"求解过程出错: {str(e)}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(x_shooting, y_shooting[0], 'b-', label='Shooting Method')
    plt.plot(x_scipy, y_scipy[0], 'r--', label='scipy.solve_bvp')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Comparison of Shooting Method and scipy.solve_bvp')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 计算差异（统一x网格）
    x_common = np.linspace(x_span[0], x_span[1], n_points)
    u_shooting_interp = np.interp(x_common, x_shooting, y_shooting[0])
    u_scipy_interp = np.interp(x_common, x_scipy, y_scipy[0])
    max_diff = np.max(np.abs(u_shooting_interp - u_scipy_interp))
    mean_diff = np.mean(np.abs(u_shooting_interp - u_scipy_interp))
    
    return {
        'shooting': {'x': x_shooting, 'u': y_shooting[0]},
        'scipy': {'x': x_scipy, 'u': y_scipy[0]},
        'differences': {'max': max_diff, 'mean': mean_diff}
    }

# 以下为测试相关（可根据实际测试框架调整，这里简单模拟测试调用）
if __name__ == "__main__":
    # 基础功能测试
    try:
        # 测试ode_system_shooting
        t_test = 0.5
        y_test = np.array([1.0, 0.5])
        dydt = ode_system_shooting(t_test, y_test)
        print(f"ode_system_shooting测试结果: {dydt}")
        
        # 测试boundary_conditions_scipy
        ya_test = np.array([1.0, 0.5])
        yb_test = np.array([1.0, -0.3])
        bc_res = boundary_conditions_scipy(ya_test, yb_test)
        print(f"boundary_conditions_scipy测试结果: {bc_res}")
        
        # 测试ode_system_scipy
        dydt_scipy = ode_system_scipy(t_test, y_test)
        print(f"ode_system_scipy测试结果: {dydt_scipy}")
        
        # 测试solve_bvp_shooting_method
        x_span_test = (0, 1)
        bc_test = (1, 1)
        x_shoot, y_shoot = solve_bvp_shooting_method(x_span_test, bc_test)
        print(f"打靶法求解结果长度校验: x({len(x_shoot)}) vs y({len(y_shoot[0])})")
        
        # 测试solve_bvp_scipy_wrapper
        x_scipy_out, y_scipy_out = solve_bvp_scipy_wrapper(x_span_test, bc_test)
        print(f"scipy求解器封装结果长度校验: x({len(x_scipy_out)}) vs y({len(y_scipy_out[0])})")
        
        # 测试compare_methods_and_plot
        compare_res = compare_methods_and_plot()
        print(f"方法对比最大差异: {compare_res['differences']['max']}")
        
    except Exception as e:
        print(f"测试过程出错: {str(e)}")
