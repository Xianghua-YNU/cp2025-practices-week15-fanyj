"""
学生模板：平方反比引力场中的运动
文件：inverse_square_law_motion_student.py
作者：[你的名字]
日期：[完成日期]

重要：函数名称、参数名称和返回值的结构必须与参考答案保持一致！
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 常量 (如果需要，学生可以自行定义或从参数传入)
# 例如：GM = 1.0 # 引力常数 * 中心天体质量

def derivatives(t, state_vector, gm_val):
    """
    计算状态向量 [x, y, vx, vy] 的导数。

    运动方程（直角坐标系）:
    dx/dt = vx
    dy/dt = vy
    dvx/dt = -GM * x / r^3
    dvy/dt = -GM * y / r^3
    其中 r = sqrt(x^2 + y^2)。

    参数:
        t (float): 当前时间 (solve_ivp 需要，但在此自治系统中不直接使用)。
        state_vector (np.ndarray): 一维数组 [x, y, vx, vy]，表示当前状态。
        gm_val (float): 引力常数 G 与中心天体质量 M 的乘积。

    返回:
        np.ndarray: 一维数组，包含导数 [dx/dt, dy/dt, dvx/dt, dvy/dt]。
    """
    # 解包状态向量
    x, y, vx, vy = state_vector
    
    # 计算距离的立方
    r_squared = x**2 + y**2
    r_cubed = r_squared ** 1.5
    
    # 避免除以零（当距离极小时）
    if r_cubed < 1e-10:
        r_cubed = 1e-10
    
    # 计算加速度
    ax = -gm_val * x / r_cubed
    ay = -gm_val * y / r_cubed
    
    return np.array([vx, vy, ax, ay])

def solve_orbit(initial_conditions, t_span, t_eval, gm_val):
    """
    使用 scipy.integrate.solve_ivp 求解轨道运动问题。

    参数:
        initial_conditions (list or np.ndarray): 初始状态 [x0, y0, vx0, vy0]。
        t_span (tuple): 积分时间区间 (t_start, t_end)。
        t_eval (np.ndarray): 需要存储解的时间点数组。
        gm_val (float): GM 值 (引力常数 * 中心天体质量)。

    返回:
        scipy.integrate.OdeSolution: solve_ivp 返回的解对象。
                                     可以通过 sol.y 访问解的数组，sol.t 访问时间点。
    """
    # 调用 solve_ivp 求解微分方程
    solution = solve_ivp(
        fun=derivatives,
        t_span=t_span,
        y0=initial_conditions,
        t_eval=t_eval,
        args=(gm_val,),
        method='DOP853',  # 高精度方法
        rtol=1e-9,        # 相对容差
        atol=1e-12        # 绝对容差
    )
    
    return solution

def calculate_energy(state_vector, gm_val, m=1.0):
    """
    计算质点的（比）机械能。
    （比）能量 E/m = 0.5 * v^2 - GM/r

    参数:
        state_vector (np.ndarray): 二维数组，每行是 [x, y, vx, vy]，或单个状态的一维数组。
        gm_val (float): GM 值。
        m (float, optional): 运动质点的质量。默认为 1.0，此时计算的是比能 (E/m)。

    返回:
        np.ndarray or float: （比）机械能。
    """
    # 处理一维或二维输入
    if state_vector.ndim == 1:
        x, y, vx, vy = state_vector
        # 计算距离
        r = np.sqrt(x**2 + y**2)
        # 避免除以零
        if r < 1e-10:
            r = 1e-10
        # 计算速度平方
        v_squared = vx**2 + vy**2
        # 计算比能
        specific_energy = 0.5 * v_squared - gm_val / r
        # 如果需要总能量，乘以质量
        return specific_energy * m if m != 1.0 else specific_energy
    else:
        x, y, vx, vy = state_vector[:, 0], state_vector[:, 1], state_vector[:, 2], state_vector[:, 3]
        # 计算距离
        r = np.sqrt(x**2 + y**2)
        # 避免除以零
        r[r < 1e-10] = 1e-10
        # 计算速度平方
        v_squared = vx**2 + vy**2
        # 计算比能
        specific_energy = 0.5 * v_squared - gm_val / r
        # 如果需要总能量，乘以质量
        return specific_energy * m if m != 1.0 else specific_energy

def calculate_angular_momentum(state_vector, m=1.0):
    """
    计算质点的（比）角动量 (z分量)。
    （比）角动量 Lz/m = x*vy - y*vx

    参数:
        state_vector (np.ndarray): 二维数组，每行是 [x, y, vx, vy]，或单个状态的一维数组。
        m (float, optional): 运动质点的质量。默认为 1.0，此时计算的是比角动量 (Lz/m)。

    返回:
        np.ndarray or float: （比）角动量。
    """
    # 处理一维或二维输入
    if state_vector.ndim == 1:
        x, y, vx, vy = state_vector
        # 计算比角动量
        specific_Lz = x * vy - y * vx
        # 如果需要总角动量，乘以质量
        return specific_Lz * m if m != 1.0 else specific_Lz
    else:
        x, y, vx, vy = state_vector[:, 0], state_vector[:, 1], state_vector[:, 2], state_vector[:, 3]
        # 计算比角动量
        specific_Lz = x * vy - y * vx
        # 如果需要总角动量，乘以质量
        return specific_Lz * m if m != 1.0 else specific_Lz

if __name__ == "__main__":
    # --- 学生可以在此区域编写测试代码或进行实验 ---
    print("平方反比引力场中的运动 - 学生模板")

    # 测试不同能量下的轨道
    GM_val = 1.0
    
    # 设置三种能量的初始条件
    # 1. 椭圆轨道 (E < 0)
    ic_ellipse = [1.0, 0.0, 0.0, 0.8]  # 初始位置(1,0)，初始速度(0,0.8)
    # 2. 抛物线轨道 (E = 0)
    ic_parabola = [1.0, 0.0, 0.0, np.sqrt(2*GM_val/1.0)]  # 初始速度恰好为逃逸速度
    # 3. 双曲线轨道 (E > 0)
    ic_hyperbola = [1.0, 0.0, 0.0, 1.5]  # 初始速度大于逃逸速度

    # 时间设置
    t_start = 0
    t_end_ellipse = 20
    t_end_parabola = 15
    t_end_hyperbola = 10
    t_eval_ellipse = np.linspace(t_start, t_end_ellipse, 500)
    t_eval_parabola = np.linspace(t_start, t_end_parabola, 300)
    t_eval_hyperbola = np.linspace(t_start, t_end_hyperbola, 200)

    try:
        # 求解三种轨道
        sol_ellipse = solve_orbit(ic_ellipse, (t_start, t_end_ellipse), t_eval_ellipse, gm_val=GM_val)
        sol_parabola = solve_orbit(ic_parabola, (t_start, t_end_parabola), t_eval_parabola, gm_val=GM_val)
        sol_hyperbola = solve_orbit(ic_hyperbola, (t_start, t_end_hyperbola), t_eval_hyperbola, gm_val=GM_val)
        
        # 计算能量和角动量
        ellipse_energy = calculate_energy(sol_ellipse.y.T, GM_val)
        parabola_energy = calculate_energy(sol_parabola.y.T, GM_val)
        hyperbola_energy = calculate_energy(sol_hyperbola.y.T, GM_val)
        
        ellipse_angular_momentum = calculate_angular_momentum(sol_ellipse.y.T)
        parabola_angular_momentum = calculate_angular_momentum(sol_parabola.y.T)
        hyperbola_angular_momentum = calculate_angular_momentum(sol_hyperbola.y.T)
        
        print(f"椭圆轨道: 初始能量 ≈ {ellipse_energy[0]:.6f}, 角动量 ≈ {ellipse_angular_momentum[0]:.6f}")
        print(f"抛物线轨道: 初始能量 ≈ {parabola_energy[0]:.6f}, 角动量 ≈ {parabola_angular_momentum[0]:.6f}")
        print(f"双曲线轨道: 初始能量 ≈ {hyperbola_energy[0]:.6f}, 角动量 ≈ {hyperbola_angular_momentum[0]:.6f}")

        # 绘制三种轨道
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(sol_ellipse.y[0], sol_ellipse.y[1], 'b-', label='椭圆轨道')
        plt.plot(0, 0, 'ko', markersize=8, label='中心天体')
        plt.title(f'椭圆轨道 (E < 0)\nE ≈ {ellipse_energy[0]:.6f}')
        plt.xlabel('x 坐标')
        plt.ylabel('y 坐标')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        plt.subplot(1, 3, 2)
        plt.plot(sol_parabola.y[0], sol_parabola.y[1], 'g-', label='抛物线轨道')
        plt.plot(0, 0, 'ko', markersize=8, label='中心天体')
        plt.title(f'抛物线轨道 (E = 0)\nE ≈ {parabola_energy[0]:.6f}')
        plt.xlabel('x 坐标')
        plt.grid(True)
        plt.axis('equal')
        
        plt.subplot(1, 3, 3)
        plt.plot(sol_hyperbola.y[0], sol_hyperbola.y[1], 'r-', label='双曲线轨道')
        plt.plot(0, 0, 'ko', markersize=8, label='中心天体')
        plt.title(f'双曲线轨道 (E > 0)\nE ≈ {hyperbola_energy[0]:.6f}')
        plt.xlabel('x 坐标')
        plt.grid(True)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()

        # 测试固定能量下不同角动量的椭圆轨道
        plt.figure(figsize=(10, 8))
        
        # 固定能量，改变角动量
        energies = []
        angular_momenta = []
        labels = []
        
        for v_init in [0.7, 0.8, 0.9, 1.0]:
            ic = [1.0, 0.0, 0.0, v_init]
            sol = solve_orbit(ic, (t_start, t_end_ellipse), t_eval_ellipse, gm_val=GM_val)
            
            # 计算能量和角动量
            energy = calculate_energy(sol.y.T, GM_val)
            angular_momentum = calculate_angular_momentum(sol.y.T)
            
            energies.append(energy[0])
            angular_momenta.append(angular_momentum[0])
            labels.append(f'v₀ = {v_init}, L ≈ {angular_momentum[0]:.4f}')
            
            plt.plot(sol.y[0], sol.y[1], label=labels[-1])
        
        plt.plot(0, 0, 'ko', markersize=8, label='中心天体')
        plt.title(f'固定能量 (E ≈ {energies[0]:.4f}) 下不同角动量的椭圆轨道')
        plt.xlabel('x 坐标')
        plt.ylabel('y 坐标')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    except Exception as e:
        print(f"运行示例时发生错误: {e}")

    print("\n请参照 '项目说明.md' 完成各项任务。")
    print("使用 'tests/test_inverse_square_law_motion.py' 文件来测试你的代码实现。")
