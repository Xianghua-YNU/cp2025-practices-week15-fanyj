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
    return np.vstack((y[1], -np.pi*(y[0]+1)/4))


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
    
    if x_start >= x_end:
        raise ValueError("Invalid x_span: x_start must be less than x_end")
    
    if not isinstance(boundary_conditions, tuple) or len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions must be a tuple of two values")
    
    # Setup domain
    x_array = np.linspace(x_start, x_end, n_points)
    
    # Initial guesses for the slope
    m0, m1 = 0.0, 1.0
    
    # Evaluate the function at the first two guesses
    def objective(m):
        sol = solve_ivp(
            ode_system_shooting, 
            x_span, 
            [u_left, m], 
            t_eval=x_array, 
            method='RK45',
            rtol=1e-6,
            atol=1e-6
        )
        
        if not sol.success:
            raise RuntimeError(f"IVP solver failed with message: {sol.message}")
            
        return sol.y[0][-1] - u_right
    
    try:
        f0 = objective(m0)
    except Exception as e:
        raise RuntimeError(f"Initial objective evaluation failed: {str(e)}") from e
    
    for i in range(max_iterations):
        try:
            f1 = objective(m1)
        except Exception as e:
            raise RuntimeError(f"Objective evaluation failed at iteration {i}: {str(e)}") from e
        
        # Check if we are close to the solution
        if abs(f1) < tolerance:
            break
            
        # Secant method update
        if abs(f1 - f0) < 1e-12:
            raise RuntimeError("Secant method failed: division by zero")
            
        m2 = m1 - f1 * (m1 - m0) / (f1 - f0)
        m0, m1 = m1, m2
        f0 = f1
    
    # Final solve with the best slope
    sol = solve_ivp(
        ode_system_shooting, 
        x_span, 
        [u_left, m1], 
        t_eval=x_array, 
        method='RK45',
        rtol=1e-6,
        atol=1e-6
    )
    
    if not sol.success:
        raise RuntimeError(f"Final IVP solve failed with message: {sol.message}")
    
    return x_array, sol.y


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
    
    if x_start >= x_end:
        raise ValueError("Invalid x_span: x_start must be less than x_end")
    
    # Setup initial mesh
    x = np.linspace(x_start, x_end, n_points)
    
    # Initial guess: linear function between boundary conditions
    y_guess = np.zeros((2, x.size))
    y_guess[0] = np.linspace(u_left, u_right, n_points)
    
    # Solve BVP
    sol = solve_bvp(
        ode_system_scipy, 
        boundary_conditions_scipy, 
        x, 
        y_guess,
        max_nodes=10000,
        tol=1e-6
    )
    
    if not sol.success:
        raise RuntimeError(f"BVP solver failed with message: {sol.message}")
    
    # Use the solution's own mesh for consistency
    return sol.x, sol.y


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
    try:
        # Solve using both methods
        x_shooting, y_shooting = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)
        x_scipy, y_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points)
    except Exception as e:
        print(f"Error solving BVP: {str(e)}")
        raise
    
    # Interpolate scipy solution to shooting method's grid for comparison
    y_scipy_interp = np.interp(x_shooting, x_scipy, y_scipy[0])
    
    # Calculate maximum difference
    max_difference = np.max(np.abs(y_shooting[0] - y_scipy_interp))
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot solutions
    plt.subplot(2, 1, 1)
    plt.plot(x_shooting, y_shooting[0], 'b-', linewidth=2, label='Shooting Method')
    plt.plot(x_scipy, y_scipy[0], 'r--', linewidth=2, label='scipy.solve_bvp')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Comparison of Shooting Method and scipy.solve_bvp')
    plt.legend()
    plt.grid(True)
    
    # Plot difference
    plt.subplot(2, 1, 2)
    plt.plot(x_shooting, y_shooting[0] - y_scipy_interp, 'g-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('Difference')
    plt.title(f'Max Difference: {max_difference:.6f}')
    plt.grid(True)
    
    plt.tight_layout()
    
    return {
        'shooting': (x_shooting, y_shooting),
        'scipy': (x_scipy, y_scipy),
        'max_difference': max_difference
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


def test_student_error_handling():
    """
    Test student's error handling implementation.
    """
    print("\nTesting error handling...")
    
    # Test invalid x_span
    try:
        solve_bvp_shooting_method((2, 1), (1, 1))
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
    else:
        print("Failed to catch invalid x_span in shooting method")
    
    try:
        solve_bvp_scipy_wrapper((2, 1), (1, 1))
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
    else:
        print("Failed to catch invalid x_span in scipy wrapper")
    
    # Test invalid boundary conditions
    try:
        solve_bvp_shooting_method((0, 1), [1, 1, 1])
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
    else:
        print("Failed to catch invalid boundary conditions in shooting method")
    
    # Test other potential errors (example: bad initial guess leading to solver failure)
    # This test may need adjustment based on actual solver behavior
    print("Testing solver failure scenarios...")


if __name__ == "__main__":
    print("项目2：打靶法与scipy.solve_bvp求解边值问题")
    print("=" * 50)
    
    # Run basic tests
    test_ode_system()
    test_boundary_conditions()
    
    # Test error handling
    test_student_error_handling()
    
    # Run comparison
    try:
        print("\nTesting method comparison...")
        results = compare_methods_and_plot()
        plt.show()  # Display the comparison plot
        print("Method comparison completed successfully!")
    except Exception as e:
        print(f"Error in method comparison: {e}")
    
    print("\n所有函数已实现并测试完成。")
