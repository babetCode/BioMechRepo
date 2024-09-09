"this is a drafting space for functions to be used in IMU gait analysis"

import numpy as np
from math import *
from ezc3d import c3d
import matplotlib.pyplot as plt
import pandas as pd


#multiply  two quaternions, given as 4-item lists
def multiplyQuaternion(q1, q2):
    [w1, x1, y1, z1] = [value for value in q1]
    [w2, x2, y2, z2] = [value for value in q2]
    scalar = w1*w2 - x1*x2 - y1*y2 - z1*z2
    i = w1*x2 + x1*w2 + y1*z2 - z1*y2
    j = w1*y2 + y1*w2 + z1*x2 - x1*z2
    k = w1*z2 + z1*w2 + x1*y2 -y1*x2
    return([scalar, i, j, k])


# Rotates point (3 item list) by angle (radians) around axis through origin (3 item list)
def rotateQuaternion(point, angle, axis):
    normalizedAxis = axis/np.linalg.norm(axis)
    q_inv = [cos(angle/2)]
    q = [cos(angle/2)]
    q_inv = [cos(angle/2)]
    for value in normalizedAxis:
        q.append(sin(angle/2)*value)
        q_inv.append(-sin(angle/2)*value)
    pointQuat = [0, point[0], point[1], point[2]]
    rotation = multiplyQuaternion(multiplyQuaternion(q, pointQuat), q_inv)
    result = [float(i) for i in rotation[1:]]
    roundedPoint = [round(i, 3) for i in point]
    roundedAngle = round(angle, 3)
    roundedAxis = [round(i, 3) for i in axis]
    roundedResult = [round(i, 3) for i in result]
    return(result)


# Class object for IMUs
class imu:
    def __init__(self, name, df, sensor_num):
        self.name = name
        self.indices = [row for row in df.index] # list of analog labels
        self.start_row_index = self.indices.index('DelsysTrignoBase 1: Sensor '+str(sensor_num)+'IM ACC Pitch') # find first row label
        self.all_data = df.iloc[self.start_row_index : self.start_row_index+6] # get dataframe of the 6 rows
        
        self.acc_data = self.all_data.iloc[0:3] # get the first 3 rows of acc data
        self.sqrt_acc = np.square(self.acc_data) # square of all acc data
        self.net_acc_sq = self.sqrt_acc.apply(np.sum, axis=0, raw=True) # sum of P,R,Y acc squares for each frame
        self.net_acc = np.sqrt(self.net_acc_sq) # net acc for each frame

        self.gyr_data = self.all_data.iloc[3:7] # get the next three rows of gyr data

        self.frames = len(self.gyr_data.columns) # get number of frames (same as legth of rows)
        xyz_axes = np.array([[[1,0,0] for i in range(self.frames)],
        [[0,1,0] for i in range(self.frames)], [[0,0,1] for i in range(self.frames)]]) # make array for axes. [:,i,:] gets frame i, [0:,i,:] gets x-axis of frame i
        # print(xyz_axes.shape)
        # print(xyz_axes[:,0,:])
        # for i in range(1, frames):
        #     xyz_axes[:,i,:] = xyz_axes[:,i,:]+xyz_axes[:,i-1,:]
        # print(xyz_axes[0,50,:])

    def raw_orientation(self):
        xyz_axes = np.array([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]] for i in range(100)])
        [xaxis, yaxis, zaxis] = [xyz_axes[0][i] for i in range(3)]

        # print([i for i in self.gyr_data.iloc[:,0]]) # first column

        for i in range (95, xyz_axes.shape[0]):
            print('loop # '+str(i))
            gyr = np.array(self.gyr_data.iloc[:, i-1]) # get gyr date for previous frame
            initial_axes = xyz_axes[i-1,:,:] # get axes for previous frame
            scaled_axes = [initial_axes[j] * gyr[j] for j in range(3)] # scale axes by component rotation velocity
            total_axis = np.sum(scaled_axes, axis=0) # axis of rotation
            # norm_axis = total_axis/np.linalg.norm(total_axis) # normalized axis - NOT NECESSARY AS rotateQuaternion() already does this
            total_gyr = sqrt(np.sum(np.square(gyr))) # rotational velocity (in deg/sec?)
            angle = total_gyr*(pi/180)/148
            print(initial_axes)

            # TO DO: define 'angle', 
            rotated_axes = np.array([rotateQuaternion(axisvector, pi/6, zaxis) for axisvector in initial_axes[i-1,:,:]])
            # combined_axis = 
            # rotated_axes = np.array([rotateQuaternion(j, pi/6, zaxis) for j in xyz_axes[i-1,:,:]])

    def plot_net_acc(self, scale):
        plt.plot(self.net_acc*scale, label = 'net acc '+self.name)
    
    def plot_PRY(self, PRY, scale):
        if 'P' in PRY:
            plt.plot(self.gyr_data.iloc[0]*scale, label= 'gyr pitch '+self.name)
        if 'R' in PRY:
            plt.plot(self.gyr_data.iloc[1]*scale, label= 'gyr roll '+self.name)
        if 'Y' in PRY:
            plt.plot(self.gyr_data.iloc[2]*scale, label= 'gyr yaw '+self.name)


# Returns dataframe with 109 labeled indices
def c3d_analogs_df(participant, speed, trial, path):
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


def adrienC3Dpath():
    if str(__file__) == 'c:\\Users\\goper\\Files\\vsCode\\490R\\ABabet_490R-1\\functionBuilder.py':
        mypath = ('C:/Users/goper/Files/vsCode/490R/Walking_C3D_files/')
    elif str(__file__) == '/Users/adrienbabet/Documents/490R/IMU_gait_analysis/functionBuilder.py':
        mypath = '/Users/adrienbabet/Documents/490R/Walking C3D files/'
    return mypath


def get_sensor_data(sensor_placement, ACCorGYR, PitRolYaw, df):
    axis_dict = {'P': 'Pitch', 'Y': 'Yaw', 'R': 'Roll'}
    if type(sensor_placement) == str:
        OneToEleven = [i+1 for i in range(11)]
        IMU_placements = []
        for side in ['L', 'R']:
            for part in ['Thigh', 'Shank']:
                for place in ['Proximal', 'Distal']:
                    IMU_placements.append(side+place+part)
            IMU_placements.append(side+'Foot')
        IMU_placements.append('Sacrum')
        IMU_dict = dict(zip(IMU_placements, OneToEleven))
        index = 'DelsysTrignoBase 1: Sensor '+str(IMU_dict[sensor_placement])+'IM '+ACCorGYR+' '+axis_dict[PitRolYaw]
    elif type(sensor_placement) == int:
        index = 'DelsysTrignoBase 1: Sensor '+str(sensor_placement)+'IM '+ACCorGYR+' '+axis_dict[PitRolYaw]
    return df.loc[index]


def test_axis_rotation():
    xaxis = [1.,0.,0.]
    yaxis = [0.,1.,0.]
    zaxis = [0.,0.,1.]
    xyz_axes = np.array([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]] for i in range(5)])
    for i in range (1, xyz_axes.shape[0]):
        print('last axis:\n', xyz_axes[i-1], xyz_axes[i].shape)
        rotated_axes = np.array([rotateQuaternion(j, pi/6, zaxis) for j in xyz_axes[i-1,:,:]])
        print('rotated axis:\n', rotated_axes, rotated_axes.shape)
        xyz_axes[i] = rotated_axes
        print('new xyz:\n', xyz_axes[i])
        # for j in xyz_axes[i]:
        #     print(j.shape)
        # print(xyz_axes[i,:,:])
        # print([rotateQuaternion(i, pi/6, zaxis) for i in xyz_axes[:,i-1,:]])


def plot_rotation(point, angle, axis, ax, name):
    t = np.linspace(0.0, angle, 100)
    x = np.zeros(100)
    y = np.zeros(100)
    z = np.zeros(100)
    for i in range(100):
        p = rotateQuaternion(point, t[i], axis)
        x[i] = p[0]
        y[i] = p[1]
        z[i] = p[2]
    ax.plot(x, y, z, label=name)
    return(rotateQuaternion(point, angle, axis))


def plot3axes(ax):
    ax.plot((-1.3,1.3), (0,0), (0,0), label='x')
    ax.plot((0,0), (-1.3,1.3), (0,0), label='y')
    ax.plot((0,0), (0,0), (-1.3,1.3), label='z')


def main():
    mypath = adrienC3Dpath()
    df = c3d_analogs_df('C07', 'Fast', '07', mypath)
    plt.close('all')
    LDistalShank = imu('LDistShank', df, 2)

    print(np.zeros(10))

    test_axis_rotation()
    # x1 = [1.,0.,0.]
    # plot_rotation(x1, pi/2, [0,1,0]) # rotate x=1 around y-axis
    # print(p2) 
    # p3 = rotateQuaternion(p2, pi/2, [1,0,0])
    # print(p3)
    # qp = rotateQuaternion(x1, sqrt(2*(pi/2)*(pi/2)), [sqrt(2)/2,sqrt(2)/2,0])
    # print(qp)
    # plt.figure()
    # LDistalShank.plot_net_acc(150)
    # LDistalShank.plot_PRY('PRY', 1)
    # plt.legend()
    plt.show(block=False)
    close_plots = input('[enter] to close plots >')

if __name__ == '__main__':
    main()