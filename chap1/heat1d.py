# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm

def switch_diff(accurate):
    if accurate==2:
        diff = second_accurate_central_diff
    else:
        diff = fourth_accurate_central_diff
    return diff

def space_diff_common(array):
    arr = np.zeros_like(array)
    arr[0] = array[0]
    arr[-1] = array[-1]
    return arr

def second_accurate_central_diff(array):
    arr = space_diff_common(array)
    arr[1:-1] = (array[2:] - 2*array[1:-1] + array[:-2])
    return arr

def fourth_accurate_central_diff(array):
    arr = space_diff_common(array)
    arr[2:-2] = 0.5 * (-array[4:] + 8*array[3:-1] - 14*array[2:-2] + 8*array[1:-3] - array[:-4])
    start3 = second_accurate_central_diff(array[:3])
    end3 = second_accurate_central_diff(array[-3:])
    arr[1] = start3[1]
    arr[-2] = end3[1]
    return arr

def implicit_diff_coef(alpha, accurate=2):
    if accurate == 2:
        return (-alpha)*np.array([1, -2, 1])
    elif accurate == 3:
        return (-0.5*alpha)*np.array([-1, 8, -14, 8, -1])
    else:
        raise ValueError

def set_boundary(array, b0=None, bN=None, firstStep=False):
    if firstStep:
        array = np.concatenate([np.zeros(1), array, np.zeros(1)]).flatten()

    if b0 is None:
        array[0] = array[1]
    else:
        array[0] = b0

    if bN is None:
        array[-1] = array[-2]
    else:
        array[-1] = bN
    return array

def plot(array, x=None, dt=None, methodName=None):
    if x is None:
        x = np.linspace(start=0, stop=1, num=array.shape[0])

    fig = plt.figure()
    if len(array.shape) == 1:
        ax = fig.add_subplot(111)
        ax.plot(x, array, ".")
        ax.set_xlabel("space")
        ax.set_ylabel("Temperature")
        plt.show()
    else:
        def update(i, title = None):
            if i != 0:
                plt.cla()
            plt.plot(x, array[:, i], ".", color="blue")
            plt.xlabel("space")
            plt.ylabel("Temperature")
            if dt is not None:
                plt.title("{1} t = {0:4f}".format(i*dt, title))
                # plt.title("explicit euler t = {0:4f}".format(i*dt, title))

        ani = anm.FuncAnimation(fig, update, fargs=(methodName))
        ani.save("sample.gif", writer="pillow")
        # plt.show()

def explicit_euler(array, step, alpha, b0=None, bN=None, accurate=2):
    arr = array[:, step]
    if accurate==2:
        diff = second_accurate_central_diff
    else:
        diff = fourth_accurate_central_diff
    arr = set_boundary(arr, b0, bN)
    return alpha * diff(arr) + arr

def adams_bashforth(array, step, alpha, b0=None, bN=None, accurate=2):
    def ab_main(arr):
        return alpha * diff(set_boundary(arr, b0, bN))

    diff = switch_diff(accurate)

    if step == 0:
        return explicit_euler(array, step, alpha, b0, bN, accurate)
    f0 = ab_main(array[:, step-1])
    f1 = ab_main(array[:, step])

    return array[:, step] + 1.5*f1 - 0.5*f0

def runge_kutta_4(array, step, alpha, b0=None, bN=None, accurate=2):
    def rk_main(arr):
        return alpha * diff(set_boundary(arr, b0, bN))

    diff = switch_diff(accurate)

    k1 = rk_main(array[:, step])
    k2 = rk_main(array[:, step] + 1 / 2 * k1)
    k3 = rk_main(array[:, step] + 1 / 2 * k2)
    k4 = rk_main(array[:, step] + k3)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6 + array[:, step]

def implicit_euler(array, step, alpha, b0=None, bN=None, accurate=2):
    if b0 is None or bN is None:
        raise ValueError

    size = array.shape[0]

    coef_matrix = np.zeros((size, size))
    coef_matrix[0, 0] = 1
    coef_matrix[-1, -1] = 1

    coefs = implicit_diff_coef(alpha, accurate)
    coef_num = coefs.shape[0]
    width = int((coef_num-1)/2)
    coefs[width] += 1

    for i in range(width, size - width):
        coef_matrix[i,i-width:i+width+1] = coefs

    if accurate == 3:
        coefs2 = implicit_diff_coef(alpha, 2)
        coefs2[1] += 1
        coef_matrix[1,:3] = coefs2
        coef_matrix[-2,-3:] = coefs2
    return np.linalg.solve(coef_matrix, array[:, step])


def main(bc="fixed", time_integral=explicit_euler):
    size = 100
    k = 1.0
    dx = 1.0 / size
    dt = 1.0 / (10*size)
    alpha = k * dt / (dx**2)
    initial = np.arange(size) / size

    if "fixed" in bc:
        b0 = 1
        b1 = 0
    else:
        b0 = None
        b1 = None

    initial = set_boundary(initial, b0, b1, firstStep=True)
    endtime = 0.1
    step = int(endtime / dt)
    array = np.zeros((initial.shape[0], step+1))
    array[:, 0] = initial

    # integrators = [explicit_euler, adams_bashforth, runge_kutta_4, implicit_euler]
    # divergence_accuracy = [2, 3]

    for i in range(step):
        # array[:, i+1] = explicit_euler(array, i, alpha, b0, b1, accurate=3)
        # array[:, i+1] = implicit_euler(array, i, alpha, b0, b1, accurate=3)
        # array[:, i+1] = runge_kutta_4(array, i, alpha, b0, b1, accurate=2)
        # array[:, i + 1] = adams_bashforth(array, i, alpha, b0, b1, accurate=2)
        array[:, i + 1] = time_integral(array, i, alpha, b0, b1, accurate=2)
        array[:, i+1] = set_boundary(array[:, i+1], b0, b1) # for plot
        # if i % 10 == 0:
            # plot(set_boundary(array[:, i], b0, b1))

    plot(array, dt=dt)

def sample():
    t_end = 0.1 # 最終時刻
    N = 100     # 格子数
    k = 1.0     # 微分方程式のパラメータk
    dx = 1.0 / N    # 格子間隔
    dt = 0.1*dx**2  # 時間刻み幅
    alpha = k * dt / (dx ** 2)  # 係数をひとまとめに

    total_timestep = int(t_end / dt)    # 反復計算回数

    phi = np.zeros((N+2, total_timestep)) # phi(x, t)   # 境界となる0番目とN+1番目も合わせて確保

    phi[1:N+1, 0] = np.arange(N) / N    # 初期条件としてphi(x, 0) = xを仮定

    boundary_x0 = 1    # Dirichlet境界条件の値
    boundary_xN = 0    # Dirichlet境界条件の値

    for n in range(total_timestep-1):
        # set boundary condition
        phi[0, n] = boundary_x0
        phi[N+1, n] = boundary_xN
        # time integration
        # by euler explicit method (1st order forward time stepping)
        for i in range(1, N+1):
            phi[i, n+1] = phi[i, n] + alpha * (phi[i+1, n] - 2*phi[i, n] + phi[i-1, n])

        # if n % 10:
            # plot(phi[:, n])
    plot(phi, dt=dt)

if __name__ == '__main__':
    # sample()
    # exit()
    main(time_integral=implicit_euler)