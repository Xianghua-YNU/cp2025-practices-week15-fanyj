#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题 - 学生代码模板

本项目要求实现打靶法和scipy.solve_bvp两种方法来求解二阶线性常微分方程边值问题：
u''(x) = -π(u(x)+1)/4
边界条件：u(0) = 1, u(1) = 1

学生姓名：范玉洁
学号：20221050183
完成日期：6.6
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_bvp
import warnings
warnings.filterwarnings('ignore')

def ode_system_shooting(y, t):
    return [y[1], -np.pi*(y[0]+1)/4]

def boundary_conditions_scipy(ya, yb):
    return np.array([ya[0] - 1, yb[0] - 1])

def ode_system_scipy(x, y):
    return np.vstack((y[1], -np.pi*(y[0]+1)/4))

def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, max_iterations=10, tolerance=1e-6):
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    
    x = np.linspace(x_start, x_end, n_points)
    m1 = -1.0
    y0 = [u_left, m1]
    
    sol1 = odeint(ode_system_shooting, y0, x)
    u_end_1 = sol1[-1, 0]
    
    if abs(u_end_1 - u_right) < tolerance:
        return x, sol1[:, 0]
    
    m2 = m1 * u_right / u_end_1 if abs(u_end_1) > 1e-12 else m1 + 1.0
    y0[1] = m2
    sol2 = odeint(ode_system_shooting, y0, x)
    u_end_2 = sol2[-1, 0]
    
    if abs(u_end_2 - u_right) < tolerance:
        return x, sol2[:, 0]
    
    for iteration in range(max_iterations):
        if abs(u_end_2 - u_end_1) < 1e-12:
            m3 = m2 + 0.1
        else:
            m3 = m2 + (u_right - u_end_2) * (m2 - m1) / (u_end_2 - u_end_1)
        
        y0[1] = m3
        sol3 = odeint(ode_system_shooting, y0, x)
        u_end_3 = sol3[-1, 0]
        
        if abs(u_end_3 - u_right) < tolerance:
            return x, sol3[:, 0]
        
        m1, m2 = m2, m3
        u_end_1, u_end_2 = u_end_2, u_end_3
    
    print(f"Warning: Shooting method did not converge after {max_iterations} iterations.")
    print(f"Final boundary error: {abs(u_end_3 - u_right):.2e}")
    return x, sol3[:, 0]

def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    
    x_init = np.linspace(x_start, x_end, n_points)
    y_init = np.zeros((2, x_init.size))
    y_init[0] = u_left + (u_right - u_left) * (x_init - x_start) / (x_end - x_start)
    y_init[1] = (u_right - u_left) / (x_end - x_start)
    
    try:
        sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x_init, y_init, max_nodes=10000)
        
        if not sol.success:
            raise RuntimeError(f"scipy.solve_bvp failed: {sol.message}")
        
        x_fine = np.linspace(x_start, x_end, 100)
        y_fine = sol.sol(x_fine)[0]
        
        return x_fine, y_fine
        
    except Exception as e:
        raise RuntimeError(f"Error in scipy.solve_bvp: {str(e)}")
        def plot_solution_comparison(x_shoot, y_shoot, x_scipy, y_scipy, 
                            title="Comparison of BVP Solution Methods", 
                            save_path="bvp_solution_comparison.png"):
    # Interpolate scipy solution
    y_scipy_interp = np.interp(x_shoot, x_scipy, y_scipy)
    max_diff = np.max(np.abs(y_shoot - y_scipy_interp))
    rms_diff = np.sqrt(np.mean((y_shoot - y_scipy_interp)** 2))
    
    # Create figure with improved layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=120)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.97)
    
    # 1. Solution comparison plot
    ax1.plot(x_shoot, y_shoot, 'b-', linewidth=2.5, label='Shooting Method')
    ax1.plot(x_scipy, y_scipy, 'r--', linewidth=2.5, label='scipy.solve_bvp')
    
    # Mark boundary points with distinct markers
    ax1.plot(x_shoot[0], y_shoot[0], 'go', markersize=8, 
             label=f'u({x_shoot[0]}) = {y_shoot[0]:.4f} (Shooting)')
    ax1.plot(x_shoot[-1], y_shoot[-1], 'ro', markersize=8, 
             label=f'u({x_shoot[-1]}) = {y_shoot[-1]:.4f} (Shooting)')
    ax1.plot(x_scipy[0], y_scipy[0], 'g^', markersize=8, 
             label=f'u({x_scipy[0]}) = {y_scipy[0]:.4f} (scipy)')
    ax1.plot(x_scipy[-1], y_scipy[-1], 'r^', markersize=8, 
             label=f'u({x_scipy[-1]}) = {y_scipy[-1]:.4f} (scipy)')
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('u(x)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Place legend in the best location avoiding curves
    ax1.legend(fontsize=9, loc='best', frameon=True, framealpha=0.9, 
              shadow=True, fancybox=True, ncol=2)
    
    # Add equation information in empty space
    ax1.text(0.02, 0.95, r"$u'' = -\frac{\pi}{4}(u + 1)$", 
             transform=ax1.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    # 2. Difference plot with improved styling
    ax2.plot(x_shoot, y_shoot - y_scipy_interp, 'g-', linewidth=2, label='Difference')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Difference', fontsize=12)
    ax2.set_title(f'Solution Difference: Max = {max_diff:.2e}, RMS = {rms_diff:.2e}', 
                 fontsize=12, pad=15)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Place legend in lower center to avoid overlap
    ax2.legend(fontsize=9, loc='lower center', frameon=True, 
              framealpha=0.9, shadow=True)
    
    # Add metrics in upper right corner
    ax2.text(0.98, 0.95, f"Max Diff: {max_diff:.2e}\nRMS Diff: {rms_diff:.2e}", 
             transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Adjust layout with extra padding
    plt.tight_layout(pad=4.0, h_pad=3.0)
    plt.subplots_adjust(top=0.92)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.show()
    
    return {'max_difference': max_diff, 'rms_difference': rms_diff}
    def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    print("="*70)
    print("Solving Boundary Value Problem:")
    print(f"Equation: u'' = -π(u+1)/4")
    print(f"Domain: [{x_span[0]}, {x_span[1]}]")
    print(f"Boundary Conditions: u({x_span[0]}) = {boundary_conditions[0]}, u({x_span[1]}) = {boundary_conditions[1]}")
    print("="*70)
    
    try:
        # Solve with shooting method
        print("\n[1] Running Shooting Method...")
        x_shoot, y_shoot = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)
        print(f"Shooting method completed with {len(x_shoot)} points")
        
        # Solve with scipy.solve_bvp
        print("\n[2] Running scipy.solve_bvp...")
        x_scipy, y_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points//2)
        print(f"scipy.solve_bvp completed with {len(x_scipy)} points")
        
        # Plot comparison
        print("\n[3] Generating comparison plot...")
        diff_metrics = plot_solution_comparison(x_shoot, y_shoot, x_scipy, y_scipy)
        
        # Print analysis results
        print("\n" + "="*50)
        print("Solution Analysis Results:")
        print("="*50)
        print(f"Maximum Difference: {diff_metrics['max_difference']:.2e}")
        print(f"RMS Difference: {diff_metrics['rms_difference']:.2e}")
        print(f"Shooting Method Sampling Points: {len(x_shoot)}")
        print(f"scipy.solve_bvp Sampling Points: {len(x_scipy)}")
        
        # Boundary condition verification
        print("\n" + "="*50)
        print("Boundary Condition Verification:")
        print("="*50)
        print(f"Shooting Method: u({x_span[0]}) = {y_shoot[0]:.6f} (error: {abs(y_shoot[0]-boundary_conditions[0]):.2e})")
        print(f"               u({x_span[1]}) = {y_shoot[-1]:.6f} (error: {abs(y_shoot[-1]-boundary_conditions[1]):.2e})")
        print(f"scipy.solve_bvp: u({x_span[0]}) = {y_scipy[0]:.6f} (error: {abs(y_scipy[0]-boundary_conditions[0]):.2e})")
        print(f"               u({x_span[1]}) = {y_scipy[-1]:.6f} (error: {abs(y_scipy[-1]-boundary_conditions[1]):.2e})")
        
        return {
            'x_shooting': x_shoot,
            'y_shooting': y_shoot,
            'x_scipy': x_scipy,
            'y_scipy': y_scipy,
            **diff_metrics
        }
        
    except Exception as e:
        print(f"\n❌ Error during method comparison: {str(e)}")
        raise
        def plot_individual_solutions(x_shoot, y_shoot, x_scipy, y_scipy, 
                             save_path="bvp_individual_solutions.png"):
    # Create figure with improved layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=120)
    fig.suptitle("Individual Solution Methods Comparison", fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Shooting method solution
    ax1.plot(x_shoot, y_shoot, 'b-', linewidth=2.5, label='Solution Curve')
    ax1.plot(x_shoot[0], y_shoot[0], 'go', markersize=8, 
             label=f'Start: u({x_shoot[0]:.2f})={y_shoot[0]:.4f}')
    ax1.plot(x_shoot[-1], y_shoot[-1], 'ro', markersize=8, 
             label=f'End: u({x_shoot[-1]:.2f})={y_shoot[-1]:.4f}')
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('u(x)', fontsize=12)
    ax1.set_title('Shooting Method Solution', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Place legend in upper left to avoid curve
    ax1.legend(fontsize=9, loc='upper left', frameon=True, 
              framealpha=0.9, shadow=True)
    
    # Add equation information in lower right
    ax1.text(0.98, 0.02, r"$u'' = -\frac{\pi}{4}(u + 1)$", 
             transform=ax1.transAxes, fontsize=12, 
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. scipy.solve_bvp solution
    ax2.plot(x_scipy, y_scipy, 'r-', linewidth=2.5, label='Solution Curve')
    ax2.plot(x_scipy[0], y_scipy[0], 'go', markersize=8, 
             label=f'Start: u({x_scipy[0]:.2f})={y_scipy[0]:.4f}')
    ax2.plot(x_scipy[-1], y_scipy[-1], 'ro', markersize=8, 
             label=f'End: u({x_scipy[-1]:.2f})={y_scipy[-1]:.4f}')
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('u(x)', fontsize=12)
    ax2.set_title('scipy.solve_bvp Solution', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Place legend in upper left to avoid curve
    ax2.legend(fontsize=9, loc='upper left', frameon=True, 
              framealpha=0.9, shadow=True)
    
    # Add method information in lower right
    ax2.text(0.98, 0.02, "Using scipy.integrate.solve_bvp", 
             transform=ax2.transAxes, fontsize=10, 
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # Adjust layout
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.88)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"Individual solutions saved to: {save_path}")
    plt.show()
def test_ode_system():
    t_test = 0.5
    y_test = np.array([1.0, 0.5])
    
    dydt = ode_system_shooting(y_test, t_test)
    expected = [0.5, -np.pi*(1.0+1)/4]
    assert np.allclose(dydt, expected)
    
    dydt_scipy = ode_system_scipy(t_test, y_test)
    expected_scipy = np.array([[0.5], [-np.pi*2/4]])
    assert np.allclose(dydt_scipy, expected_scipy)

def test_boundary_conditions():
    ya = np.array([1.0, 0.5])
    yb = np.array([1.0, -0.3])
    bc_residual = boundary_conditions_scipy(ya, yb)
    expected = np.array([0.0, 0.0])
    assert np.allclose(bc_residual, expected)

def test_shooting_method():
    x_span = (0, 1)
    boundary_conditions = (1, 1)
    x, y = solve_bvp_shooting_method(x_span, boundary_conditions, n_points=50)
    assert abs(y[0] - boundary_conditions[0]) < 1e-6
    assert abs(y[-1] - boundary_conditions[1]) < 1e-6

def test_scipy_method():
    x_span = (0, 1)
    boundary_conditions = (1, 1)
    x, y = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=20)
    assert abs(y[0] - boundary_conditions[0]) < 1e-6
    assert abs(y[-1] - boundary_conditions[1]) < 1e-6

def main():
    print("\n" + "="*70)
    print("Project 2: Shooting Method vs scipy.solve_bvp")
    print("="*70)
    
    # Run all tests
    print("\n>>> Running unit tests...")
    test_ode_system()
    test_boundary_conditions()
    test_shooting_method()
    test_scipy_method()
    print("✅ All unit tests passed!")
    
    # Run method comparison and plot
    print("\n>>> Running method comparison...")
    results = compare_methods_and_plot()
    
    # Plot individual solutions
    plot_individual_solutions(results['x_shooting'], results['y_shooting'], 
                             results['x_scipy'], results['y_scipy'])
    
    print("\n" + "="*70)
    print("✅ Project completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
