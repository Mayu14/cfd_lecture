# -- coding: utf-8 --
import numpy as np
from chap1.heat1d import plot

def input_init(size):
    arr = np.sin(np.linspace(0, 4.0*np.pi, num=size+1))
    arr = arr[:-1]
    arr[int(size/4):] = 0.0
    return arr

def first_order_upwind_diff(array):
    arr = np.zeros_like(array)
    arr[1:] = (array[1:] - array[:-1])
    arr[0] = (array[0] - array[-1])
    return arr

def first_order_downwind_diff(array):
    arr = np.zeros_like(array)
    arr[1:] = (array[:-1] - array[1:])
    arr[0] = (array[-1] - array[0])
    return arr

def second_order_upwind_diff(array):
    def diff(a,b,c):
        return -1.5*a + 2.0*b - 0.5*c
    arr = np.zeros_like(array)
    arr[0:-2] = diff(array[:-2], array[1:-1], array[2:])
    arr[-2] = diff(array[-2], array[-1], array[0])
    arr[-1] = diff(array[-1], array[0], array[1])
    return arr

def second_order_central_diff(array):
    def diff(a,c):
        return 0.5*(a - c)
    arr = np.zeros_like(array)
    arr[0] = diff(array[1], array[-1])
    arr[1:-1] = diff(array[2:], array[:-2])
    arr[-1] = diff(array[0], array[-2])
    return arr

def exact_sol():
    pass

def main():
    phi = np.zeros((N + 2, total_timestep))  # phi(x, t)   # 境界となる0番目とN+1番目も合わせて確保
    phi[1:N + 1, 0] = input_init(N)  # 初期条件としてphi(x, 0) = xを仮定

    dfdx = first_order_upwind_diff
    for n in range(total_timestep-1):
        phi[-1, n] = phi[0, n]
        phi[0, n] = phi[-1, n]

        phi[:, n+1] = alpha*dfdx(phi[:, n]) + phi[:, n]
        # for i in range(1, N+1):
            # phi[i, n+1] = alpha*(phi[i-1, n] - phi[i, n])/2 + phi[i, n]

    plot(phi, dt=dt, fname="convection_upwind.gif", useylim=True, ylim=[0,1])

if __name__ == '__main__':
    t_end = 10.0  # 最終時刻
    N = 100  # 格子数
    c = 1.0  # 微分方程式のパラメータk
    dx = 1.0 / N  # 格子間隔
    dt = 0.5 * dx  # 時間刻み幅
    alpha = c * dt / dx  # 係数をひとまとめに

    total_timestep = int(t_end / dt)  # 反復計算回数
    print(total_timestep)

    main()