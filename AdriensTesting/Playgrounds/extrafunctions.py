"""
|‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
|           Functions for biomechanical data analysis                  |
|______________________________________________________________________|

|‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
|Author: Adrien Babet | GitHub: @babetcode  |  Email: adrienbabet1@gmail.com  |
|_____________________|_____________________|_________________________________|
"""
import numpy as np
import math
from ezc3d import c3d
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

class imu:
    """
    Contains data structure and functionality for analyzing IMU
    measurements.

    Parameters
    -
    name: string
        Name for the IMU.
    df: dataframe
        dataframe containing IMU measurements. Using c3d_analogs_df() is
        reccomended.
    sensor_num: any
        Sensor nubmer will be converted to string to index data from df.

    Attributes
    -
    name: str
        As initialized.
    indices: list
        List of analog labels from df.
    start_row_index: int
        Index of the first row in df which contains data for this IMU.
    all_data: dataframe
        Data from df for this specific IMU.
    acc_data: dataframe
        Acceleration data from all_data.
    net_acc: dataframe
        Net acceleration.
    frames: int
        Number of frames in df.
    gyr_data: dataframe
        Gyroscopic data from df.

    Methods
    -
    raw_orientation()
        Attemts to determine orientation with raw data.
    test()
        pass.

    """
    def __init__(self, name, df, sensor_num):
        self.name = name
        self.indices = [row for row in df.index] # list of analog labels
        # find first row label
        self.start_row_index = self.indices.index(
            'DelsysTrignoBase 1: Sensor '+str(sensor_num)+'IM ACC Pitch')
        # get dataframe of the 6 rows
        self.all_data = df.iloc[self.start_row_index : self.start_row_index+6]
        
        # get the first 3 rows of acc data
        self.acc_data = self.all_data.iloc[0:3]
        # square of all acc data
        sqrt_acc = np.square(self.acc_data)
        # sum of P,R,Y acc squares for each frame
        net_acc_sq = sqrt_acc.apply(np.sum, axis=0, raw=True)
        self.net_acc = np.sqrt(net_acc_sq) # net acc for each frame
        # get the next three rows of gyr data
        self.gyr_data = self.all_data.iloc[3:7]
        # get number of frames (same as legth of rows)
        self.frames = len(self.gyr_data.columns)

    def __str__(self):
        return f'{self.name}'

    def raw_orientation(self):
        xyz_axes = np.array(
            [[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]] for i in range(self.frames)]
            ) # xyz_axes[frame, axis, xyz vector]
        [xaxis, yaxis, zaxis] = [xyz_axes[0][i] for i in range(3)]

        # print([i for i in self.gyr_data.iloc[:,0]]) # first column

        # set up 3d figure
        plt.close('all')
        my3dplot = plt.figure().add_subplot(projection='3d')
        my3dplot.set_xlabel('x')
        my3dplot.set_ylabel('y')
        my3dplot.set_zlabel('z')
        plot3axes(my3dplot)

        for i in range (self.frames):
            # print('loop # '+str(i))
            # get gyr date for previous frame
            gyr = np.array(self.gyr_data.iloc[:, i-1])
            initial_axes = xyz_axes[i-1,:,:] # get axes for previous frame
            # scale axes by component rotation velocity
            scaled_axes = [initial_axes[j] * gyr[j] for j in range(3)]
            total_axis = np.sum(scaled_axes, axis=0) # axis of rotation
            norm_gyr = math.sqrt(np.sum(np.square(gyr))) # rotational velocity
            angle_deg = norm_gyr/148
            angle_rad = angle_deg * math.pi/180 
            rotated_axes = np.array([rotate_quaternion(axisvector, angle_rad,
                total_axis) for axisvector in initial_axes])
            #print(rotated_axes)
            xyz_axes[i,:,:] = rotated_axes
        my3dplot.plot(xyz_axes[:,0,0], xyz_axes[:,0,1], xyz_axes[:,0,2])
        plt.show()


    def plot_net_acc(self, scale):
        plt.plot(self.net_acc*scale, label = 'net acc '+self.name)
    
    def plot_PRY(self, PRY, scale):
        if 'P' in PRY:
            plt.plot(
                self.gyr_data.iloc[0]*scale, label= 'gyr pitch '+self.name)
        if 'R' in PRY:
            plt.plot(self.gyr_data.iloc[1]*scale, label= 'gyr roll '+self.name)
        if 'Y' in PRY:
            plt.plot(self.gyr_data.iloc[2]*scale, label= 'gyr yaw '+self.name)


def plot_rotation(point, axis, angle, figure, name):
    """
    Plots the path a point takes while rotating with rotateQuaternion()
    function on the 'ax' figure. Returns rotatedQuaternion(point, angle,
    axis).

    Parameters
    -
    point (3 item iterable)
        Point as [point x, point y, point z].
    axis (3 item iterable)
        axis as [axis x, axis y, axis z]. The rotation will be around
        the line through this point and [0, 0, 0].
    angle (float)
        angle in radians.
    figure: pyplot figure
        It is reccomended to create figure using: \n
            plt.figure().add_subplot(projection='3d'). \n
        It may also be useful to use: \n
            my3dplot.set_xlabel('x') \n
            my3dplot.set_ylabel('y') \n
            my3dplot.set_zlabel('z').

    Returns
    -
    Rotated Point (list)
        [point x, point y, point z].
    """
    t = np.linspace(0.0, angle, 100)
    x = np.zeros(100)
    y = np.zeros(100)
    z = np.zeros(100)
    for i in range(100):
        p = rotate_quaternion(point, t[i], axis)
        x[i] = p[0]
        y[i] = p[1]
        z[i] = p[2]
    figure.plot(x, y, z, label=name)
    return(rotate_quaternion(point, angle, axis))



def plot3axes(figure):
    """
    Plots the x, y, and z axes on the 'ax' figure. Does not include
    plt.show()

    Parameters
    -
    figure: pyplot figure
        It is reccomended to create figure using: \n
            plt.figure().add_subplot(projection='3d'). \n
        It may also be useful to use: \n
            my3dplot.set_xlabel('x') \n
            my3dplot.set_ylabel('y') \n
            my3dplot.set_zlabel('z').
    """
    figure.plot((-1.3,1.3), (0,0), (0,0), label='x')
    figure.plot((0,0), (-1.3,1.3), (0,0), label='y')
    figure.plot((0,0), (0,0), (-1.3,1.3), label='z')
