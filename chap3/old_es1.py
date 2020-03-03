# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

class Constant(object):
    def __init__(self):
        self.gamma = 1.4
        self.gmin1 = 0.4

def debug(AA):
    print(AA)
    sys.exit(0)

class Flux(object):
    def __init__(self, cell):
        self.cell = cell

    def roeLoadAverage(self, varR, varL, sqrtDR, sqrtDL):  # Roeの荷重平均式
        return (sqrtDR * varR + sqrtDL * varL) / (sqrtDR + sqrtDL)

    def roeAverage(self, qR, qL):  # Roe平均量の計算
        # 準備：密度の平方根，速度，エンタルピーの計算
        const = Constant()
        sqrtDensR, sqrtDensL = np.sqrt(qR[0, :]), np.sqrt(qL[0, :])
        vR, vL = qR[1, :] / qR[0, :], qL[1, :] / qL[0, :]
        entR = const.gamma * qR[2, :] / qR[0, :] - 0.5 * const.gmin1 * vR[:] ** 2
        entL = const.gamma * qL[2, :] / qL[0, :] - 0.5 * const.gmin1 * vL[:] ** 2

        roeQ = np.zeros(3 * (self.cell - 1)).reshape(3, (self.cell - 1))
        # 定義式に則ったRoe平均の計算
        roeQ[1, :] = self.roeLoadAverage(vR, vL[:], sqrtDensR, sqrtDensL)  # Roe平均流速
        roeQ[2, :] = self.roeLoadAverage(entR, entL[:], sqrtDensR, sqrtDensL)  # Roe平均エンタルピー
        roeQ[0, :] = np.sqrt(const.gmin1 * roeQ[2, :] - 0.5 * roeQ[1, :] ** 2)  # Roe平均音速
        return roeQ

    def generalFlux(self, qR, qL):  # セル中心における流束
        gFlux = np.zeros(3 * (self.cell - 1) * 2).reshape(3, (self.cell - 1), 2)  # variable, cell-1, side
        con = Constant()
        kinEneR = 0.5 * qR[1, :] ** 2 / qR[0, :]
        kinEneL = 0.5 * qL[1, :] ** 2 / qL[0, :]  # 運動エネルギー

        # セル中心の物理量から流束を計算
        gFlux[0, :, 1], gFlux[0, :, 0] = qR[1, :], qL[1, :]  # [i, 1] and [i+1, 0] make cell
        gFlux[1, :, 1] = 2.0 * kinEneR + con.gmin1 * (qR[2, :] - kinEneR)
        gFlux[1, :, 0] = 2.0 * kinEneL + con.gmin1 * (qL[2, :] - kinEneL)
        gFlux[2, :, 1] = qR[1, :] / qR[0, :] * (con.gamma * qR[2, :] - con.gmin1 * kinEneR)
        gFlux[2, :, 0] = qL[1, :] / qL[0, :] * (con.gamma * qL[2, :] - con.gmin1 * kinEneL)
        return gFlux

    def getEigenMatrix(self, roeQ, hhECeps):
        rightEigMat = np.zeros(9 * (self.cell - 1)).reshape(3, 3, (self.cell - 1))
        leftEigMat = np.zeros(9 * (self.cell - 1)).reshape(3, 3, (self.cell - 1))
        diagEigMat = np.zeros(9 * (self.cell - 1)).reshape(3, 3, (self.cell - 1))

        # Right Eigen
        rightEigMat[0, 0, :] = 1.0
        rightEigMat[0, 1, :] = 1.0
        rightEigMat[0, 2, :] = 1.0
        rightEigMat[1, 0, :] = roeQ[1, :] - roeQ[0, :]
        rightEigMat[1, 1, :] = roeQ[1, :]
        rightEigMat[1, 2, :] = roeQ[1, :] + roeQ[0, :]
        rightEigMat[2, 0, :] = roeQ[2, :] - roeQ[1, :] * roeQ[0, :]
        rightEigMat[2, 1, :] = 0.5 * roeQ[1, :] ** 2
        rightEigMat[2, 2, :] = roeQ[2, :] + roeQ[1, :] * roeQ[0, :]

        con = Constant()
        assist2 = con.gmin1 / roeQ[0, :] ** 2
        assist1 = 0.5 * roeQ[1, :] ** 2 * assist2

        # Left Eigen Matrix
        leftEigMat[0, 0, :] = 0.5 * (assist1 + roeQ[1, :] / roeQ[0, :])
        leftEigMat[0, 1, :] = - 0.5 * (1.0 / roeQ[0, :] + assist2 * roeQ[1, :])
        leftEigMat[0, 2, :] = 0.5 * assist2
        leftEigMat[1, 0, :] = 1.0 - assist1
        leftEigMat[1, 1, :] = assist2 * roeQ[1, :]
        leftEigMat[1, 2, :] = - assist2
        leftEigMat[2, 0, :] = 0.5 * (assist1 - roeQ[1, :] / roeQ[0, :])
        leftEigMat[2, 1, :] = 0.5 * (1.0 / roeQ[0, :] - assist2 * roeQ[1, :])
        leftEigMat[2, 2, :] = 0.5 * assist2

        # Absolute Eigenvalue Matrix + Harten-Hyman Entropy Correct
        diagEigMat[0, 0, :] = np.where(abs(roeQ[1, :] - roeQ[0, :]) < hhECeps,
                                       0.5 * (abs(roeQ[1, :] - roeQ[0, :]) ** 2 + hhECeps ** 2) / hhECeps,
                                       abs(roeQ[1, :] - roeQ[0, :]))
        diagEigMat[1, 1, :] = np.where(abs(roeQ[1, :]) < hhECeps, 0.5 * (abs(roeQ[1, :]) ** 2 + hhECeps ** 2) / hhECeps,
                                       abs(roeQ[1, :]))
        diagEigMat[2, 2, :] = np.where(abs(roeQ[1, :] + roeQ[0, :]) < hhECeps,
                                       0.5 * (abs(roeQ[1, :] + roeQ[0, :]) ** 2 + hhECeps ** 2) / hhECeps,
                                       abs(roeQ[1, :] + roeQ[0, :]))

        return rightEigMat, leftEigMat, diagEigMat

    def getRoeFlux(self, right, left, diag, qR, qL):  # Roe流束の計算
        return np.einsum('ijm,jkm,klm,lm->im', right, diag, left, qR - qL)

    def roeFDS(self, qR, qL):  # Roe流束 + FDSによる流束の計算
        genFlux = self.generalFlux(qR, qL)
        roeQ = self.roeAverage(qR, qL)
        hhECeps = 0.15
        rightEigMat, leftEigMat, diagEigMat = self.getEigenMatrix(roeQ, hhECeps)
        roeFlux = self.getRoeFlux(rightEigMat, leftEigMat, diagEigMat, qR, qL)
        return 0.5 * (genFlux[:, :, 0] + genFlux[:, :, 1] - roeFlux)

def explicitEuler(qi, dT, dX, flux):
    # [i, 1] and [i+1, 0] make cell
    fluxSum = flux[:, 1:] - flux[:, :(flux.shape[1] - 1)]
    qi[:, 1:(qi.shape[1] - 1)] -= dT / dX * fluxSum
    qi[:, 0] = bound(qi[:, 0], qi[:, 1])
    qi[:, qi.shape[1] - 1] = bound(qi[:, qi.shape[1] - 1], qi[:, qi.shape[1] - 2])

    return qi

def bound(qB, qB2):
    qB[0] = qB2[0]
    qB[1] = - qB2[1]
    qB[2] = qB2[2]
    return qB

def figplot(qRL, cell):
    priV = convert(qRL)
    x = np.linspace(0, 1, num=cell)
    plt.plot(x, priV[0, :], label='Density')
    plt.plot(x, priV[1, :], label='Velocity')
    plt.plot(x, priV[2, :], label='Pressure')
    plt.legend()
    plt.title("shocktube: t=0.2")
    plt.show()

def figplotCon(qRL, cell):
    x = np.linspace(0, 1, num=cell)
    plt.plot(x, qRL[0, :])
    plt.plot(x, qRL[1, :])
    plt.plot(x, qRL[2, :])
    plt.show()

def figplot2(qRL, qRL_g, cell):
    priV = convert(qRL)
    priV_g = convert(qRL_g)
    x = np.linspace(0, 1, num=cell)
    plt.plot(x, priV[0, :])
    plt.plot(x, priV[1, :])
    plt.plot(x, priV[2, :])
    plt.plot(x, priV_g[0, :])
    plt.plot(x, priV_g[1, :])
    plt.plot(x, priV_g[2, :])
    plt.show()

def convert(qRL):
    priV = np.zeros_like(qRL)
    con = Constant()
    priV[0, :] = qRL[0, :]
    priV[1, :] = qRL[1, :] / qRL[0, :]
    priV[2, :] = con.gmin1 * (qRL[2, :] - 0.5 * qRL[1, :] ** 2 / qRL[0, :])
    print(max(priV[1]))
    return priV

def main():
    cell = 500
    qRL = np.zeros(3 * cell).reshape(3, cell)
    # qRL = np.arange(3 * cell).reshape(3, cell)
    halfcell = int(cell / 2)

    dX = 1.0 / cell
    alpha = 10.0
    dT = dX / alpha
    # iterate = int(0.5/dT)*1000
    iterate = int(0.2 / dT)

    total0 = np.sum(qRL, axis=1)
    residual = np.zeros(3 * iterate).reshape(iterate, 3)


    qRL[0, :halfcell] = 1.0  # 1.0
    qRL[1, :halfcell] = 0.0
    qRL[2, :halfcell] = (1.0 / 1.4) / 0.4  # (1.0 / 1.4)/0.4
    qRL[0, halfcell:] = 0.1  # 0.1
    qRL[1, halfcell:] = 0.0
    qRL[2, halfcell:] = (0.1 / 1.4) / 0.4  # (0.1 / 1.4)/0.4

    qRL[:, halfcell:] *= 1.0
    qRL[:, 0] = bound(qRL[:, 0], qRL[:, 1])
    qRL[:, - 1] = bound(qRL[:, - 1], qRL[:, - 2])
    for iter in range(iterate):
        qR = qRL[:, 1:]
        qL = qRL[:, :(cell - 1)]

        calcFlux = Flux(cell)
        flux = calcFlux.roeFDS(qR, qL)
        qRL = explicitEuler(qRL, dT, dX, flux)

        """
        # plotInterval = iterate/10 # 1
        if iter / (cell / 50) % alpha == 0:
        # if iter > 75:
            figplot(qRL, cell)
            # figplotCon(qRL,cell)
            # figplot2(qRL, qRL_g, cell)
        #"""
        residual[iter] = np.sum(qRL, axis=1) / total0

        """
        if iter > 7:
            print(qRL[:,16], qRL[:, 17])
            print(flux)
            figplot(qRL, cell)
        """

    figplot(qRL, cell)

if __name__ == "__main__":
    main()