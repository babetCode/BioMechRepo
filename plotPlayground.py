from math import *
from ezc3d import c3d
import matplotlib.pyplot as plt
import scipy.signal as sgl
from scipy.fft import fft
import numpy as np
import pandas as pd
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from functionBuilder import *


def main():
    # get IMU data
    mypath = adrienC3Dpath()
    df = c3d_analogs_df('A02', 'PWS', '01', mypath)
    LDistalShank = imu('LDistShank', df, 2)

    # set up 3d figure
    plt.close('all')
    my3dplot = plt.figure().add_subplot(projection='3d')
    my3dplot.set_xlabel('x')
    my3dplot.set_ylabel('y')
    my3dplot.set_zlabel('z')


    # make 3d plots
    p1 = [1.,0.,0.]
    plot3axes(my3dplot)
    p2 = plot_rotation(p1, .5*pi, [0,1,0], my3dplot, 'first') # rotate x=1 around y-axis
    print(p2)
    point3 = plot_rotation(p2, .5*pi, [1,0,0], my3dplot, 'second') # rotate x=1 around y-axis
    point4 = plot_rotation(p1, pi, [1,1,0], my3dplot, 'around')
    plot_rotation(p1, 2*pi, [1,1,1], my3dplot, 'around')
    plot_rotation(p1, .5*pi, [0,0,1], my3dplot, 'around')

    # plot_rotation(x1, pi*2, [0,0,1], my3dplot) # rotate x=1 around z-axis
    # plot_rotation(y1, pi*2, [1,0,0], my3dplot) # rotate x=1 around z-axis
    my3dplot.legend()
    
    # set up 2d figure
    plt.figure()

    # make 2d plots
    LDistalShank.plot_net_acc(150)
    LDistalShank.plot_PRY('PRY', 1)

    plt.legend()

    plt.show(block=False)
    close_plots = input('[enter] to close plots >')

if __name__ == '__main__':
    main()