from math import *
from ezc3d import c3d
import matplotlib.pyplot as plt
import scipy.signal as sgl
from scipy.fft import fft
import numpy as np
import pandas as pd
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def main():
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


    # Prepare arrays x, y, z
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 10)
    z = np.linspace(-2, 2, 10)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    ax.plot(x, y, z, label='parametric curve')
    ax.legend()
    print('x: ', x)
    print('y: ', y)
    print('z: ', z)


        


if __name__ == "__main__":
    main()

plt.show(block=False)
#plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') # all open plots are correctly closed after each run