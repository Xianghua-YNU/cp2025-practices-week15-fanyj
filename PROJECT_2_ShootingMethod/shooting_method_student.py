#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题 - 学生代码模板

本项目要求实现打靶法和scipy.solve_bvp两种方法来求解二阶线性常微分方程边值问题：
u''(x) = -π(u(x)+1)/4
边界条件：u(0) = 1, u(1) = 1

学生姓名：[范玉洁]
学号：[20221050183]
完成日期：[6.4]
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
    # Validate input parameters
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    
    # Setup domain
    x_array = np.linspace(x_start, x_end, n_points)
    
    # Initial guesses for the slope
    m0 = 0.0  # Initial slope guess 1
    m1 = 1.0  # Initial slope guess 2
    
    # Solve the IVP with the first slope guess
    sol0 = solve_ivp(ode_system_shooting, x_span, [u_left, m0], 
                    t_eval=x_array, method='RK45')
    y0 = sol0.y[0][-1]  # Value at x=1
    
    # Iterate using the secant method to find the correct slope
    for i in range(max_iterations):
        # Solve the IVP with the second slope guess
        sol1 = solve_ivp(ode_system_shooting, x_span, [u_left, m1], 
                        t_eval=x_array, method='RK45')
        y1 = sol1.y[0][-1]  # Value at x=1
        
        # Check convergence
        if abs(y1 - u_right) < tolerance:
            return x_array, sol1.y
        
        # Secant method update for the slope
        m2 = m1 - (y1 - u_right) * (m1 - m0) / (y1 - y0)
        
        # Update guesses
        m0, m1 = m1, m2
        y0, y1 = y1, None
    
    # If maximum iterations reached, return the best solution found
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
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    
    # Setup initial mesh
    x = np.linspace(x_start, x_end, n_points)
    
    # Initial guess: linear function between boundary conditions
    y_guess = np.zeros((2, x.size))
    y_guess[0] = np.linspace(u_left, u_right, n_points)
    
    # Solve the BVP
    sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x, y_guess)
    
    # Refine the solution on a finer grid for plotting
    x_plot = np.linspace(x_start, x_end, 100)
    y_plot = sol.sol(x_plot)
    
    return x_plot, y_plot


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
    # Solve using both methods
    x_shooting, y_shooting = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)
    x_scipy, y_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points)
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_shooting, y_shooting[0], 'b-', label='Shooting Method')
    plt.plot(x_scipy, y_scipy[0], 'r--', label='scipy.solve_bvp')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Comparison of Shooting Method and scipy.solve_bvp')
    plt.legend()
    plt.grid(True)
    plt.savefig('bvp_comparison.png')
    plt.close()
    
    # Calculate maximum difference
    # Interpolate scipy solution to shooting method's grid
    y_scipy_interp = np.interp(x_shooting, x_scipy, y_scipy[0])
    max_diff = np.max(np.abs(y_shooting[0] - y_scipy_interp))
    
    results = {
        'shooting': {
            'x': x_shooting,
            'y': y_shooting[0]
        },
        'scipy': {
            'x': x_scipy,
            'y': y_scipy[0]
        },
        'max_difference': max_diff
    }
    
    return results


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
        print(f"Maximum difference between methods: {results['max_difference']:.6e}")
        print("Method comparison completed successfully! Plot saved as 'bvp_comparison.png'")
    except NotImplementedError as e:
        print(f"Method comparison not yet implemented: {e}")
    
    print("\n请实现所有标记为 TODO 的函数以完成项目。")
