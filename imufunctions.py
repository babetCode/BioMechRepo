"this is a drafting space for functions to be used in IMU gait analysis"

import numpy as np
from math import *
from ezc3d import c3d
import matplotlib.pyplot as plt
import pandas as pd


def quaternion_multiply(q1, q2):
    """
    Multiply  two quaternions, given as 4-item lists.

    Parameters
    -
    q1 (4 item iterable)
        First Quaternion.
    q2 (4 item iterable)
        second quaternion.

    Returns
    -
    Quaternion (list)
        [scalar, i, j, k].

    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def rotateQuaternion(point, axis, angle: float):
    """
    Rotates 'point' around the 'axis' through origin by the 'angle' (in radians).

    Parameters
    -
    point (3 item iterable)
        Point as [point x, point y, point z].
    axis (3 item iterable)
        axis as [axis x, axis y, axis z].
        The rotation will be around the line through this point and [0, 0, 0].
    angle (float)
        angle in radians.

    Returns
    -
    Rotated Point (list)
        [point x, point y, point z].
    """
    normalizedaxis = axis/np.linalg.norm(axis)
    q_inv = [cos(angle/2)]
    q = [cos(angle/2)]
    for value in normalizedaxis:
        q.append(sin(angle/2)*value)
        q_inv.append(-sin(angle/2)*value)
    pointquat = [0, point[0], point[1], point[2]]
    rotation = quaternion_multiply(quaternion_multiply(q, pointquat), q_inv)
    result = [float(i) for i in rotation[1:]]
    return(result)


def plot_rotation(point, axis, angle, figure, name):
    """
    Plots the path a point takes while rotating with rotateQuaternion() function on the 'ax' figure. Returns rotatedQuaternion(point, angle, axis).

    Parameters
    -
    point (3 item iterable)
        Point as [point x, point y, point z].
    axis (3 item iterable)
        axis as [axis x, axis y, axis z]. The rotation will be around the line through this point and [0, 0, 0].
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
        p = rotateQuaternion(point, t[i], axis)
        x[i] = p[0]
        y[i] = p[1]
        z[i] = p[2]
    figure.plot(x, y, z, label=name)
    return(rotateQuaternion(point, angle, axis))


def plot3axes(figure):
    """
    Plots the x, y, and z axes on the 'ax' figure. Does not include plt.show()

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


def adrien_c3d_path():
    """
    Gets directory path for C3D files on Adriens computers.

    Returns
    -
    mypath: str
        My file path.
    """
    abpcpath = ('c:/Users/goper/Files/vsCode/490R/VScodeIMUrepo', 'C:/Users/goper/Files/vsCode/490R/Walking_C3D_files/')
    abmacpath = ('/Users/adrienbabet/Documents/490R/IMU_gait_analysis', '/Users/adrienbabet/Documents/490R/Walking C3D files/')
    tmpcpath = ('C:/Users/tm4dd/Documents/00_MSU/01_PhD_Research/Python_code', 'C:/Users/tm4dd/Documents/00_MSU/01_PhD_Research/Walking_mechanics/Data/')
    pathfinder = dict([abpcpath, abmacpath, tmpcpath])
    for path in pathfinder.keys():
        if path in str(__file__).replace('\\', '/'):
            return(pathfinder[path])
    print('adrienC3Dpath() did not find a path')


def c3d_analogs_df(participant: str, speed: str, trial: str, path: str):
    """
    Returns pandas dataframe with IMU data.

    Parameters
    -
    participant: str
        Participant number as string.
    speed: str
        trial speed as string.
    trial: str
        trial number as string.
    path: str
        directory path as string. If you have set it up, using adrienC3Dpath() is reccomended.

    Returns
    -
    df: dataframe
        dataframe containing trial data.
    """
    filename = (
        participant+'_C3D/'+participant+'_'+speed+'_'+trial+'.c3d')
    file_path = path+filename
    myc3d = c3d(file_path)
    point_data = myc3d['data']['points']
    analog_data = myc3d['data']['analogs']
    analogs = analog_data[0, :, :]
    analog_labels = myc3d['parameters']['ANALOG']['LABELS']['value']
    df = pd.DataFrame(data=analogs, index=analog_labels)
    return df


def simplekalman(
    measurments, initial_state=0, initial_error_covariance=6, state_transition=1.03,
    measurement_model=1, process_noise=0, measurement_covariance=1
):
    """
    A simple kalman filter.

    Parameters
    -
    measurments: numpy array
        Array of nx1 column vector measurments.
    initial_state: numpy array
        predicted initial state as mx1 column vector.
    trial: str
        trial number as string.
    path: str
        directory path as string. If you have set it up, using adrienC3Dpath() is reccomended.

    Returns
    -
    df: dataframe
        dataframe containing trial data.
    """
    # set variables
    A = state_transition
    At = np.transpose(A)
    H = measurement_model
    Ht = np.transpose(H)
    Q = process_noise
    R = measurement_covariance
    x = initial_state
    P = initial_error_covariance
    
    filtered_data = []
    # kalman algorithm
    for z in measurments:
        xp = A * x                              # state prediction
        Pp = A * P * At + Q                     # covariance prediction
        K = Pp * Ht / (H * Pp * Ht + R)         # kalman gain
        x = xp + K * (z - H * xp)               # state update
        P = Pp - K * H * Pp                     # covariance update
        filtered_data.append(x)
    return filtered_data


class imu:
    """
    Contains data structure and functionality for analyzing IMU measurements.

    Parameters
    -
    name: string
        Name for the IMU.
    df: dataframe
        dataframe containing IMU measurements. Using c3d_analogs_df() is reccomended.
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
        self.start_row_index = self.indices.index('DelsysTrignoBase 1: Sensor '+str(sensor_num)+'IM ACC Pitch') # find first row label
        self.all_data = df.iloc[self.start_row_index : self.start_row_index+6] # get dataframe of the 6 rows
        
        self.acc_data = self.all_data.iloc[0:3] # get the first 3 rows of acc data
        sqrt_acc = np.square(self.acc_data) # square of all acc data
        net_acc_sq = sqrt_acc.apply(np.sum, axis=0, raw=True) # sum of P,R,Y acc squares for each frame
        self.net_acc = np.sqrt(net_acc_sq) # net acc for each frame
        self.gyr_data = self.all_data.iloc[3:7] # get the next three rows of gyr data
        self.frames = len(self.gyr_data.columns) # get number of frames (same as legth of rows)

    def __str__(self):
        return f'{self.name}'

    def raw_orientation(self):
        xyz_axes = np.array([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]] for i in range(self.frames)]) # xyz_axes[frame, axis, xyz vector]
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
            gyr = np.array(self.gyr_data.iloc[:, i-1]) # get gyr date for previous frame
            initial_axes = xyz_axes[i-1,:,:] # get axes for previous frame
            scaled_axes = [initial_axes[j] * gyr[j] for j in range(3)] # scale axes by component rotation velocity
            total_axis = np.sum(scaled_axes, axis=0) # axis of rotation
            # norm_axis = total_axis/np.linalg.norm(total_axis) # normalized axis - NOT NECESSARY AS rotateQuaternion() already does this
            norm_gyr = sqrt(np.sum(np.square(gyr))) # rotational velocity
            angle_deg = norm_gyr/148
            angle_rad = angle_deg * pi/180 
            rotated_axes = np.array([rotateQuaternion(axisvector, angle_rad, total_axis) for axisvector in initial_axes])
            #print(rotated_axes)
            xyz_axes[i,:,:] = rotated_axes
        my3dplot.plot(xyz_axes[:,0,0], xyz_axes[:,0,1], xyz_axes[:,0,2])
        plt.show()


    def plot_net_acc(self, scale):
        plt.plot(self.net_acc*scale, label = 'net acc '+self.name)
    
    def plot_PRY(self, PRY, scale):
        if 'P' in PRY:
            plt.plot(self.gyr_data.iloc[0]*scale, label= 'gyr pitch '+self.name)
        if 'R' in PRY:
            plt.plot(self.gyr_data.iloc[1]*scale, label= 'gyr roll '+self.name)
        if 'Y' in PRY:
            plt.plot(self.gyr_data.iloc[2]*scale, label= 'gyr yaw '+self.name)


def main():
    mypath = adrien_c3d_path()
    df = c3d_analogs_df('C07', 'Fast', '07', mypath)
    myIMU = imu('myFirstIMU', df, 9)
    imu_data = myIMU.all_data

if __name__ == '__main__':
    main()
