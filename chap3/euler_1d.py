# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt

def energy2pressure(rho, rho_u, e):
    return (gmin1) * (e - 0.5 * rho_u**2 / rho)  # pressure

def pressure2energy(rho, u, p):
    return 1.0/ gmin1 * (p + 0.5*rho*u**2)   # internal energy

def pressure2enthalpy(rho, u, p):
    energy = pressure2energy(rho, u, p)
    return energy + p

def conserve2primitive(Q):
    q = np.zeros(3)
    q[0] = Q[0] # density
    q[1] = Q[1] / Q[0]  # velocity
    q[2] = energy2pressure(Q[0], Q[1], Q[2])   # pressure
    return q

def primitive2conserve(q):
    Q = np.zeros(3)
    Q[0] = q[0] # density
    Q[1] = q[0] * q[1]  # momentum
    Q[2] = pressure2energy(q[0], q[1], q[2])   # internal energy
    return Q

def primitive2flux(rho, u, p):
    flux = np.zeros(3)
    flux[0] = rho * u
    flux[1] = rho * u ** 2 + p
    e = p / (gmin1) + (rho * u ** 2) / 2.0
    flux[2] = u * (e + p)
    return flux

def primitive2soundspeed(rho, u, p):
    return np.sqrt(p / rho)

class Euler1D(object):
    def __init__(self, t_end=0.1, size=100, cfl=0.9, shockratio=0.1):
        self.t_end = t_end  # 最終時刻
        self.N = size  # 格子数
        self.dx = np.ones(size+2) / self.N  # 格子間隔
        self.dt = 0.001#cfl * self.dx[0] ** 2  # 時間刻み幅
        self.total_timestep = int(self.t_end / self.dt)  # 反復計算回数
        self.entropy_correct = 0.15

        self.nextQn = np.zeros((self.N+2, 3))
        self.lastQn = np.zeros((self.N+2, 3))

        self.Qn = np.ones((self.N+2, 3))
        # set initial value
        self.Qn[:, 1] = 0
        self.Qn[:, 2] /= gmin1 * gamma
        self.Qn[int(self.N / 2):, :] *= shockratio
        self.set_boundary()

    def set_boundary(self):
        # refrect boundary
            # left boundary
        self.Qn[0][:] = self.Qn[1][:]   # 勾配なし条件
        self.Qn[0][1] = -self.Qn[1][1]  # 運動量のみ全反射
            # right boundary
        self.Qn[-1][:] = self.Qn[-2][:]  # 勾配なし条件
        self.Qn[-1][1] = -self.Qn[-2][1]  # 運動量のみ全反射

    def calcCFL(self):
        self.con2pri()
        index = np.argmax(self.qn[:, 1])
        qn = self.qn[index]
        soundspeed = primitive2soundspeed(qn[0], qn[1], qn[2])
        cfl = (abs(qn[1]) + soundspeed) / (self.dx[index] / self.dt)
        return cfl

    def con2pri(self):
        self.qn = np.zeros((self.N + 2, 3))
        for i in range(self.N+2):
            self.qn[i] = conserve2primitive(self.Qn[i])

    def getFlux(self):
        def get_general_flux(q):
            return primitive2flux(q[0], q[1], q[2])

        def get_roe_average(qR, qL):
            sq_rho_R, sq_rho_L = np.sqrt(qR[0]), np.sqrt(qL[0])
            def roe_load_average(val_R, val_L):
                return (sq_rho_R * val_R + sq_rho_L * val_L) / (sq_rho_R + sq_rho_L)

            enthalpy_R, enthalpy_L = pressure2enthalpy(qR[0], qR[1], qR[2]), pressure2enthalpy(qL[0], qL[1], qL[2])

            # roe_average = np.zeros(3)
            u = roe_load_average(qR[1], qL[1]) # velocity
            H = roe_load_average(enthalpy_R, enthalpy_R)  # enthalpy
            c = np.sqrt(gmin1*H - 0.5*u**2)  # sound speed
            return c, u, H

        def get_right_eigen_matrix(c, u, H):
            right_eigen_matrix = np.zeros((3, 3))
            right_eigen_matrix[0, 0] = 1.0
            right_eigen_matrix[0, 1] = 1.0
            right_eigen_matrix[0, 2] = 1.0
            right_eigen_matrix[1, 0] = u - c
            right_eigen_matrix[1, 1] = u
            right_eigen_matrix[1, 2] = u + c
            right_eigen_matrix[2, 0] = H - u*c
            right_eigen_matrix[2, 1] = 0.5 * u**2
            right_eigen_matrix[2, 2] = H + u*c
            return right_eigen_matrix

        def get_left_eigen_matrix(c, u, H):
            beta2 = gmin1 / c**2
            beta1 = 0.5 * u**2 * beta2

            left_eigen_matrix = np.zeros((3, 3))
            left_eigen_matrix[0, 0] = 0.5 * (beta1 + u / c)
            left_eigen_matrix[0, 1] = -0.5 * (1.0 / c + beta2 * u)
            left_eigen_matrix[0, 2] = 0.5 * beta2
            left_eigen_matrix[1, 0] = 1.0 - beta1
            left_eigen_matrix[1, 1] = u * beta2
            left_eigen_matrix[1, 2] = -beta2
            left_eigen_matrix[2, 0] = 0.5 * (beta1 - u / c)
            left_eigen_matrix[2, 1] = 0.5 * (1.0 / c - u * beta2)
            left_eigen_matrix[2, 2] = 0.5 * beta2
            return left_eigen_matrix

        def get_diagonal_matrix(c, u, H):
            def entropy_correction(x):
                if x < self.entropy_correct:
                    return 0.5*(x**2 + self.entropy_correct**2)
                else:
                    return x

            dianogal_matrix = np.zeros((3,3))
            dianogal_matrix[0, 0] = entropy_correction(np.abs(u - c))
            dianogal_matrix[1, 1] = entropy_correction(np.abs(u))
            dianogal_matrix[2, 2] = entropy_correction(np.abs(u + c))
            return dianogal_matrix

        def get_roe_flux(qR, qL):
            deltaQ = qR - qL
            c, u, H = get_roe_average(qR, qL)
            r_mat = get_right_eigen_matrix(c, u, H)
            d_mat = get_diagonal_matrix(c, u, H)
            l_mat = get_left_eigen_matrix(c, u, H)
            return np.dot(np.dot(np.dot(r_mat, d_mat), l_mat), deltaQ)

        def calcFlux(qR, qL):
            roe_flux_diff = get_roe_flux(qR, qL)
            flux_R, fluxL = get_general_flux(qR), get_general_flux(qL)
            return 0.5 * ((flux_R + fluxL) + roe_flux_diff)

        self.con2pri()  # 基礎変数の準備
        self.normalFlux = np.zeros((self.N + 1, 3))   # 各面の法線方向流束
        for j in range(self.N + 1): # jは面の番号
            # j番の面はj,j+1のセルに挟まれているとする
            self.normalFlux[j] = calcFlux(qR=self.qn[j+1], qL=self.qn[j])

    def time_step(self):
        self.set_boundary()
        self.getFlux()
        dt = self.dt
        dx = self.dx
        flux = self.normalFlux
        for i in range(1, self.N + 1):
            # i番のセルはi-1番の面とi番の面に挟まれている
            # i番の面の外向き法線ベクトルは+1
            # i-1番の面の外向き法線ベクトルは-1
            self.nextQn[i] = self.Qn[i] + dt / dx[i]  * (flux[i] - flux[i-1])
        self.lastQn = self.Qn
        self.nextQn = self.Qn

    def plot(self):
        self.con2pri()
        rho = self.qn[:, 0]
        u = self.qn[:, 1]
        p = self.qn[:, 2]
        x = np.linspace(start=-0.5, stop=0.5, num=self.N+2)
        plt.plot(x, rho)
        plt.plot(x, u)
        plt.plot(x, p)
        plt.show()

def main():
    Solver = Euler1D()
    for j in range(10):
        for i in range(10):
            print(Solver.calcCFL())
            Solver.time_step()
        Solver.plot()

if __name__ == '__main__':
    gamma = 1.4
    gmin1 = gamma - 1.0
    main()