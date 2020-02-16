# -- coding: utf-8 --
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# global variable
size = 100
dx = dy = 1.0 / size

endtime = 1.0#1.0
dt = 0.000025
step = int(endtime / dt)
print(dt/(dx**2))

def set_boundary(field, bx0, bxN, by0, byN):
    if by0 is not None and byN is not None:
        field[:, 0] = by0
        field[:, -1] = byN
    else:
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]
    field[-1, :] = bxN
    field[0, :] = bx0

    return field

def explicit_euler(field):
    updated = np.zeros((size+2, size+2))
    for j in range(1,size+1):
        for i in range(1,size+1):
            updated[i, j] = (field[i + 1, j] - 2 * field[i, j] + field[i - 1, j]) / (dx ** 2) + (
                        field[i, j + 1] - 2 * field[i, j] + field[i, j - 1]) / (dy ** 2)

    return dt * updated + field

def plot(field):
    xy = np.linspace(0, 1, size)
    X, Y = np.meshgrid(xy, xy)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, field[1:-1,1:-1], cmap="bwr", linewidth=0)
    fig.colorbar(surf)
    ax.set_title("Temperature Surface")
    plt.show()



def main():
    # define
    fields = np.zeros((size+2, size+2, step))

    initial = np.linspace(0, 1, size + 2).reshape(-1, 1)
    fields[:, :, 0] = np.repeat(initial, size+2, 1)

    bx0 = 1
    bxN = 0
    by0 = None
    byN = None

    for i in range(step-1):
        fields[:, :, i] = set_boundary(fields[:, :, i], bx0, bxN, by0, byN)
        if i % 100 == 0:
            plot(fields[:, :, i])
        fields[:, :, i+1] = explicit_euler(fields[:, :, i])





if __name__ == '__main__':
    main()