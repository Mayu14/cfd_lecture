# -- coding: utf-8 --
import numpy as np
from chap1.heat1d import plot

class Heat1D(object):
    def __init__(self, scheme="central_diff"):
        self.t_end = 0.1  # 最終時刻
        self.N = 100  # 格子数
        self.k = 1.0  # 微分方程式のパラメータk
        self.dx = 1.0 / self.N  # 格子間隔
        self.dt = 0.1 * self.dx ** 2  # 時間刻み幅
        self.alpha = self.k * self.dt / (self.dx ** 2)  # 係数をひとまとめに

        self.total_timestep = int(self.t_end / self.dt)  # 反復計算回数

        self.phi = np.zeros((self.N + 2, self.total_timestep))  # phi(x, t)   # 境界となる0番目とN+1番目も合わせて確保

        self.phi[1:self.N + 1, 0] = np.arange(self.N) / self.N  # 初期条件としてphi(x, 0) = xを仮定

        self.boundary_x0 = 1  # Dirichlet境界条件の値
        self.boundary_xN = 0  # Dirichlet境界条件の値

        scheme = scheme.lower()
        if "central" in scheme:
            self.scheme = self.central_diff
        elif "upwind" in scheme:
            self.scheme = self.upwind_diff
        elif "downwind" in scheme:
            self.scheme = self.downwind_diff
        else:
            raise ValueError
        self.fname = scheme

    def diff_core(self, i, n, shift=0):
        # shift = 0: central difference
        # shift = 1: upwind difference
        # shift = -1: downwind difference
        return self.phi[i, n] + self.alpha * (self.phi[i + 1 + shift, n] - 2 * self.phi[i + shift, n] + self.phi[i - 1 + shift, n])

    def central_diff(self, n):
        for i in range(1, self.N + 1):
            self.phi[i, n + 1] = self.diff_core(i, n)
        return self.phi

    def upwind_diff(self, n):
        for i in range(1, self.N):
            self.phi[i, n + 1] = self.diff_core(i, n, shift=1)

        self.phi[self.N, n + 1] = self.diff_core(self.N, n, shift=0)
        return self.phi

    def downwind_diff(self, n):
        for i in range(2, self.N + 2):
            self.phi[i, n + 1] = self.diff_core(i, n, shift=-1)
        self.phi[1, n + 1] = self.diff_core(1, n, shift=0)
        return self.phi


    def solve(self):
        for n in range(self.total_timestep - 1):
            # set boundary condition
            self.phi[0, n] = self.boundary_x0
            self.phi[self.N + 1, n] = self.boundary_xN
            # time integration
            # by euler explicit method (1st order forward time stepping)
            self.phi = self.scheme(n)

        plot(self.phi, dt=self.dt, fname=self.fname+".gif")



if __name__ == '__main__':
    schemes = ["central_diff", "downwind_diff", "upwind_diff"]

    for scheme in schemes:
        Solver = Heat1D(scheme)
        Solver.solve()