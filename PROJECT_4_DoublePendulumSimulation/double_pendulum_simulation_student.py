"""
学生模板：双摆模拟
课程：计算物理
说明：请实现标记为 TODO 的函数。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation

# 物理常量
G_CONST = 9.81  # 重力加速度 (m/s²)
L_CONST = 0.4   # 摆臂长度 (m)
M_CONST = 1.0   # 摆锤质量 (kg)

def derivatives(y, t, L1, L2, m1, m2, g):
    """计算双摆状态向量的时间导数"""
    theta1, omega1, theta2, omega2 = y
    
    dtheta1_dt = omega1
    dtheta2_dt = omega2
    
    # 公共分母
    denom = 3 - np.cos(2 * theta1 - 2 * theta2)
    
    # 计算 domega1/dt
    term1 = omega1**2 * np.sin(2 * theta1 - 2 * theta2)
    term2 = 2 * omega2**2 * np.sin(theta1 - theta2)
    term3 = (g / L1) * (np.sin(theta1 - 2 * theta2) + 3 * np.sin(theta1))
    domega1_dt = - (term1 + term2 + term3) / denom
    
    # 计算 domega2/dt
    term4 = 4 * omega1**2 * np.sin(theta1 - theta2)
    term5 = omega2**2 * np.sin(2 * theta1 - 2 * theta2)
    term6 = 2 * (g / L1) * (np.sin(2 * theta1 - theta2) - np.sin(theta2))
    domega2_dt = (term4 + term5 + term6) / denom
    
    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

def solve_double_pendulum(initial_conditions, t_span, t_points, 
                          L_param=L_CONST, g_param=G_CONST):
    """使用 odeint 求解双摆运动方程"""
    # 构建初始状态向量
    y0 = [initial_conditions['theta1'], 
          initial_conditions['omega1'], 
          initial_conditions['theta2'], 
          initial_conditions['omega2']]
    
    # 生成时间序列
    t_arr = np.linspace(t_span[0], t_span[1], t_points)
    
    # 高精度求解（确保能量守恒）
    sol_arr = odeint(derivatives, y0, t_arr, 
                    args=(L_param, L_param, M_CONST, M_CONST, g_param),
                    rtol=1e-8, atol=1e-8)
    
    return t_arr, sol_arr

def calculate_energy(sol_arr, L_param=L_CONST, m_param=M_CONST, g_param=G_CONST):
    """计算双摆系统的总能量（动能+势能）"""
    theta1 = sol_arr[:, 0]
    omega1 = sol_arr[:, 1]
    theta2 = sol_arr[:, 2]
    omega2 = sol_arr[:, 3]
    
    # 势能计算
    potential = -m_param * g_param * L_param * (2 * np.cos(theta1) + np.cos(theta2))
    
    # 动能计算（包含相对运动项）
    kinetic = m_param * L_param**2 * (
        omega1**2 + 
        0.5 * omega2**2 + 
        omega1 * omega2 * np.cos(theta1 - theta2)
    )
    
    return potential + kinetic

def animate_double_pendulum(t_arr, sol_arr, L_param=L_CONST, skip_frames=10):
    """生成双摆运动动画（可选）"""
    # 提取角度数据并筛选帧
    theta1 = sol_arr[:, 0][::skip_frames]
    theta2 = sol_arr[:, 2][::skip_frames]
    t_anim = t_arr[::skip_frames]
    
    # 转换为笛卡尔坐标
    x1 = L_param * np.sin(theta1)
    y1 = -L_param * np.cos(theta1)
    x2 = x1 + L_param * np.sin(theta2)
    y2 = y1 - L_param * np.cos(theta2)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2*L_param-0.1, 2*L_param+0.1)
    ax.set_ylim(-2*L_param-0.1, 0.1)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Double Pendulum Animation')
    
    # 初始化绘图元素
    line, = ax.plot([], [], 'o-', lw=2, markersize=8, color='red')
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(i):
        # 绘制摆线和摆锤
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        line.set_data(thisx, thisy)
        
        # 更新时间显示
        time_text.set_text(f'Time = {t_anim[i]:.1f}s')
        return line, time_text
    
    # 生成动画对象
    ani = animation.FuncAnimation(
        fig, animate, frames=len(t_anim),
        interval=50, blit=True, init_func=init
    )
    
    return ani

if __name__ == '__main__':
    # 初始条件（弧度制）
    initial = {
        'theta1': np.pi/2,   # 90度
        'omega1': 0.0,
        'theta2': np.pi/4,   # 45度
        'omega2': 0.0
    }
    
    # 模拟参数
    t_span = (0, 20)    # 模拟时长（秒）
    t_points = 2000     # 时间点数
    
    # 求解运动方程
    t_arr, sol_arr = solve_double_pendulum(initial, t_span, t_points)
    
    # 计算能量
    energy = calculate_energy(sol_arr)
    
    # 绘制能量曲线
    plt.figure(figsize=(10, 5))
    plt.plot(t_arr, energy, label='Total Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title('Energy Conservation of Double Pendulum')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 生成动画（取消注释以运行）
    # anim = animate_double_pendulum(t_arr, sol_arr, skip_frames=5)
    # plt.show()
