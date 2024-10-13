"""
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
Functions for biomechanical data analysis                              |
_______________________________________________________________________|
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
Author: Adrien Babet    GitHub: @babetcode    Email: adrienbabet1@gmail.com   |
______________________________________________________________________________|
"""
import numpy as np
import math
from ezc3d import c3d
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os


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


def rotate_quaternion(point, axis, angle: float):
    """
    Rotates 'point' around the 'axis' through origin by the 'angle'
    (in radians).

    Parameters
    -
    point (3 item iterable)
        Point as [point x, point y, point z].
    axis (3 item iterable)
        axis as [axis x, axis y, axis z]. The rotation will be around
        the line through this point and [0, 0, 0].
    angle (float)
        angle in radians.

    Returns
    -
    Rotated Point (list)
        [point x, point y, point z].
    """
    normalizedaxis = axis/np.linalg.norm(axis)
    q_inv = [math.cos(angle/2)]
    q = [math.cos(angle/2)]
    for value in normalizedaxis:
        q.append(math.sin(angle/2)*value)
        q_inv.append(-math.sin(angle/2)*value)
    pointquat = [0, point[0], point[1], point[2]]
    rotation = quaternion_multiply(quaternion_multiply(q, pointquat), q_inv)
    result = [float(i) for i in rotation[1:]]
    return(result)


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


def adrien_c3d_folder(machine):
    """
    Gets directory path for C3D files on Adriens computers.

    Returns
    -
    mypath: str
        My file path.
    """
    path2c3d = {'abpc': 'c:\\Users\\goper\\Files\\vsCode\\490R\\c3d_files\\',
        'abmac': '\\Users\\adrienbabet\\Documents\\vsCode\\490R\\c3d_files\\',
        'tmlaptop': 'C:\\Users\\tm4dd\\Documents\\00_MSU\\01_PhD_Research\\\
        Python_code\\'}
    if machine in path2c3d:
        return(path2c3d[machine])
    else:
        print('adrien_c3d_path() did not find a path')
        return('adrien_c3d_path() did not find a path')


def c3d_file(participant: str, speed: str, trial: str, path: str):
    filename = participant+'_C3D\\'+participant+'_'+speed+'_'+trial+'.c3d'
    return(path+filename)

def c3d_analogs_df(path: str):
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
        directory path as string. If you have set it up, using
        adrien_c3d_folder() is reccomended.

    Returns
    -
    df: dataframe
        dataframe containing trial data.
    """
    myc3d = c3d(path)
    #point_data = myc3d['data']['points']
    analog_data = myc3d['data']['analogs']
    analogs = analog_data[0, :, :]
    analog_labels = myc3d['parameters']['ANALOG']['LABELS']['value']
    df = pd.DataFrame(data=analogs, index=analog_labels)
    return df


def write_trc_file(marker_data, labels, frame_rate, units, output_file):
    """
    Writes marker data to a .trc file compatible with OpenSim.
    """
    marker_data = np.nan_to_num(marker_data, nan=0.0)

    num_markers = marker_data.shape[1]
    num_frames = marker_data.shape[0]

    with open(output_file, 'w') as f:
        # Write header
        f.write('PathFileType\t4\t(X/Y/Z)\t{}\n'.format(output_file))
        f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\
                \tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n'
                )
        f.write('{:.2f}\t{:.2f}\t{}\t{}\t{}\t{}\t1\t{}\n\
                '.format(frame_rate, frame_rate, num_frames, num_markers,
                         units, frame_rate, num_frames)
                )
        
        # Write marker labels
        f.write('Frame#\tTime\t' + '\t'.join(
            [f'{label}\t\t' for label in labels]) + '\n')
        f.write('\t\t' + '\t'.join(
            [f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(len(labels))]) + '\n')
        f.write('\n')  # Empty line separating header and data

        # Write data
        for i in range(num_frames):
            time = i / frame_rate
            frame_data = marker_data[i].reshape(-1)
            f.write(f'{i+1}\t{time:.5f}\t' + '\t'.join(
                [f'{x:.5f}' for x in frame_data]) + '\n')

def write_mot_file(analog_data, labels, frame_rate, output_file):
    """
    Writes analog data (forces, joint angles, etc.) to a .mot file
    compatible with OpenSim.
    """
    num_columns = analog_data.shape[1]
    num_frames = analog_data.shape[0]
    
    with open(output_file, 'w') as f:
        # Write header
        f.write('name {}\n'.format(output_file))
        f.write('datacolumns {}\n'.format(num_columns))
        f.write('datarows {}\n'.format(num_frames))
        f.write('range 0 {:.5f}\n'.format(num_frames / frame_rate))
        f.write('endheader\n')

        # Write column headers
        f.write('time\t' + '\t'.join(labels) + '\n')
        
        # Write data
        for i in range(num_frames):
            time = i / frame_rate
            frame_data = analog_data[i]
            f.write(f'{time:.5f}\t' + '\t'.join(
                [f'{x:.5f}' for x in frame_data]) + '\n')

def convert_c3d_to_opensim(c3d_file: str, trc_file: str, mot_file: str):
    """
    Writes data to .trc and .mot files compatible with OpenSim.

    Parameters
    -
    c3d_file: str
        File path to c3d.
    trc_file: str
        File path for the ouput .trc file.
    mot_file: str
        File path for the ouput .mot file.
    """
    myc3d = c3d(c3d_file)
    
    # Reshape to (n_frames, n_markers, 3)
    # Native shape: (4, n_markers, n_frames)
    marker_data = myc3d['data']['points'] 
    marker_data = marker_data[[0, 2, 1], :, :].transpose(2, 1, 0) 

    marker_labels = myc3d['parameters']['POINT']['LABELS']['value']

    frame_rate = myc3d['parameters']['POINT']['RATE']['value'][0]

    units = myc3d['parameters']['POINT']['UNITS']['value'][0]
    
    # Write marker data to .trc file
    write_trc_file(marker_data, marker_labels, frame_rate, units, trc_file)

    # Extract analog data (optional, for motion data if available)
    if 'ANALOG' in myc3d['parameters']:
        analog_data = np.array(myc3d['data']['analogs']).squeeze().T
        analog_labels = myc3d['parameters']['ANALOG']['LABELS']['value']
        
        # Write analog data to .mot file
        write_mot_file(analog_data, analog_labels, frame_rate, mot_file)
    else:
        print("No analog data found in the .c3d file. \
              Skipping .mot file creation.")



def simplekalman(measurments, initial_state=0, initial_error_covariance=6,
                 state_transition=1.03, measurement_model=1, process_noise=0,
                 measurement_covariance=1):
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
        directory path as string. If you have set it up, using
        adrien_c3d_folder() is reccomended.

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
            plt.plot(self.gyr_data.iloc[0]*scale, label= 'gyr pitch '+self.name)
        if 'R' in PRY:
            plt.plot(self.gyr_data.iloc[1]*scale, label= 'gyr roll '+self.name)
        if 'Y' in PRY:
            plt.plot(self.gyr_data.iloc[2]*scale, label= 'gyr yaw '+self.name)


def main():
    print('empty')

if __name__ == '__main__':
    main()
