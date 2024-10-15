"""
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
Functions for biomechanical data analysis                  
________________________________________________________________________

‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
Author: Adrien Babet | GitHub: @babetcode | Email: adrienbabet1@gmail.com  
_______________________________________________________________________________
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


def rotate_quaternion(point, axis, angle: float) -> list:
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


def adrien_c3d_folder(machine):
    """
    Gets directory path for C3D files on Adriens computers.

    Returns
    -
    mypath: str
        My file path.
    """
    path2c3d = {'abpc': 'c:/Users/goper/Files/vsCode/490R/c3d_files/',
        'abmac': '/Users/adrienbabet/Documents/vsCode/490R/c3d_files/',
        'tmlaptop': 'C:/Users/tm4dd/Documents/00_MSU/01_PhD_Research/\
        Python_code/'}
    if machine in path2c3d:
        return(path2c3d[machine])
    else:
        print('adrien_c3d_path() did not find a path')
        return('adrien_c3d_path() did not find a path')


def c3d_file(participant: str, speed: str, trial: str, path: str):
    filename = participant+'_C3D/'+participant+'_'+speed+'_'+trial+'.c3d'
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
    analog_data = myc3d['data']['analogs']
    analogs = analog_data[0, :, :]
    analog_labels = myc3d['parameters']['ANALOG']['LABELS']['value']
    df = pd.DataFrame(data=analogs, index=analog_labels)
    return df


def write_trc_file(marker_data, labels, frame_rate, units, output_file):
    """
    Writes marker data to a .trc file compatible with OpenSim.

    Parameters
    -
    marker_data: numpy array
        Marker data as retrieved from ezc3d.c3d()['data']['points'].
    lables: list
        Marker label as from
        ezc3d.c3d['parameters']['POINT']['LABELS']['value'].
    frame_rate:
        Frame rate of motion capture
    units: str
        Marker data units.
    output_file: str
        name of .trc output file.
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


def write_sto_file(analog_data, labels, frame_rate, name, output_file):
    """
    Writes marker data to a .sto file compatible with OpenSim.

    Parameters
    -
    analog_data: numpy array
        Marker data as retrieved from ezc3d.c3d()['data']['analogs']
    lables: list
        Marker labels as from
        ezc3d.c3d['parameters']['ANALOG']['LABELS']['value'].
    frame_rate:
        Frame rate of motion capture
    units: str
        Marker data units.
    output_file: str
        name of .trc output file.
    """
    num_columns = analog_data.shape[1]
    num_frames = analog_data.shape[0]
    
    with open(output_file, 'w') as f:
        # Write header for .sto file
        f.write(f'{name}\n')
        f.write('nRows={}\n'.format(num_frames))
        f.write('nColumns={}\n'.format(num_columns + 1)) # +1 for time col
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
        write_sto_file(analog_data, analog_labels, frame_rate,
                       mot_file.rsplit('\\',1)[0], mot_file)
    else:
        print("No analog data found in the .c3d file. \
              Skipping .sto file creation.")



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

def main():
    print('empty')

if __name__ == '__main__':
    main()
