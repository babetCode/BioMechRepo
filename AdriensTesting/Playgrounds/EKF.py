"""
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
An extended kalman filter for IMU orientation and position over time.
Code Status: In Progress
________________________________________________________________________

‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
Author: Adrien Babet | GitHub: @babetcode | Email: adrienbabet1@gmail.com  
_______________________________________________________________________________
"""

from adriensdir import BioMechDir
mydir = BioMechDir().add_imu_func_path()
import imufunctions as myfns
import glob
import math
import pandas as pd
import ezc3d
import numpy as np

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

def extended_kalman():
    def f(state, u):
        # Unpack the state vector
        x, y, z, vx, vy, vz, qx, qy, qz, qw = state
        # Unpack the control inputs
        ax, ay, az, gx, gy, gz = u
        
        # Update position
        new_x = x + vx * dt
        new_y = y + vy * dt
        new_z = z + vz * dt
        
        # Update velocity (considering gravity)
        new_vx = vx + ax * dt
        new_vy = vy + ay * dt
        new_vz = vz + az * dt - 9.8 * dt  # Adjust for gravity
        
        # Update orientation (considering gyroscope data)
        # We'll use a simple integration method for this example
        new_qx = qx + gx * dt
        new_qy = qy + gy * dt
        new_qz = qz + gz * dt
        new_qw = qw  # Assuming small-angle approximation
        
        # Normalize quaternion
        norm = math.sqrt(new_qx**2 + new_qy**2 + new_qz**2 + new_qw**2)
        new_qx /= norm
        new_qy /= norm
        new_qz /= norm
        new_qw /= norm
        
        return [
            new_x, new_y, new_z, new_vx, new_vy, new_vz, new_qx, new_qy, new_qz, new_qw]
    def h(state):
        x, y, z, vx, vy, vz, qx, qy, qz, qw = state
        # Assuming the IMU can directly measure the position and orientation
        return [x, y, z, qx, qy, qz, qw]
