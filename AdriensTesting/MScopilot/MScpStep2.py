import adrienscomputers
adrienscomputers.adriensdirectory()
from imufunctions import*
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

mypath = adrien_c3d_path()
df = c3d_analogs_df('C07', 'Fast', '07', mypath)

trim = df.iloc[32:].drop('DelsysTrignoBase 1: Sensor '+str(i+1)+'EMG' for i in range(11))

data = trim.iloc[np.r_[18:24, 48:54]]

# Convert acceleration data from gs to meters per second squared
left_shank_acc = data.iloc[:3, :].values * 9.81
left_shank_gyro = data.iloc[3:6, :].values
right_shank_acc = data.iloc[6:9, :].values * 9.81
right_shank_gyro = data.iloc[9:12, :].values

# Convert gyro data from degrees per second to radians per second
left_shank_gyro = np.deg2rad(left_shank_gyro)
right_shank_gyro = np.deg2rad(right_shank_gyro)

# Define the sampling frequency and time step
fs = 1000  # Hz
dt = 1 / fs  # seconds

# Initialize quaternion for orientation (assuming initial orientation is identity quaternion)
q_left = np.array([1, 0, 0, 0])
q_right = np.array([1, 0, 0, 0])

# Initialize position and velocity
pos_left = np.zeros((3, left_shank_acc.shape[1]))
vel_left = np.zeros((3, left_shank_acc.shape[1]))
pos_right = np.zeros((3, right_shank_acc.shape[1]))
vel_right = np.zeros((3, right_shank_acc.shape[1]))

# Function to normalize a quaternion
def normalize_quaternion(q):
    return q / np.linalg.norm(q)

# Function to update quaternion based on gyro data using small angle approximation
def update_quaternion(q, gyro, dt):
    omega = np.hstack(([0], gyro))
    dq = 0.5 * q_mult(q, omega) * dt
    q_new = q + dq
    return normalize_quaternion(q_new)

# Quaternion multiplication
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# Kalman filter initialization (simplified for demonstration purposes)
P_left = np.eye(4) * 0.01  # Initial covariance matrix for left shank
P_right = np.eye(4) * 0.01  # Initial covariance matrix for right shank

# Process noise covariance
Q = np.eye(4) * 0.001

# Measurement noise covariance
R_covariance = np.eye(4) * 0.01

# Function to perform Kalman filter update
def kalman_filter_update(q, P, gyro, acc, dt):
    # Predict step
    q_pred = update_quaternion(q, gyro, dt)
    F = np.eye(4) + 0.5 * dt * np.array([
        [0, -gyro[0], -gyro[1], -gyro[2]],
        [gyro[0], 0, gyro[2], -gyro[1]],
        [gyro[1], -gyro[2], 0, gyro[0]],
        [gyro[2], gyro[1], -gyro[0], 0]
    ])
    P_pred = F @ P @ F.T + Q

    # Update step
    acc_norm = acc / np.linalg.norm(acc)
    z = np.hstack(([0], acc_norm))
    H = np.eye(4)
    y = z - H @ q_pred
    S = H @ P_pred @ H.T + R_covariance
    K = P_pred @ H.T @ np.linalg.inv(S)
    q_new = q_pred + K @ y
    P_new = (np.eye(4) - K @ H) @ P_pred

    return normalize_quaternion(q_new), P_new

# Detect heel strikes and toe offs for left shank
heel_strike_threshold = 1.2  # This threshold may need to be adjusted based on the data
toe_off_threshold = 0.8      # This threshold may need to be adjusted based on the data

left_heel_strikes = np.where(np.linalg.norm(left_shank_acc, axis=0) > heel_strike_threshold)[0]
right_heel_strikes = np.where(np.linalg.norm(right_shank_acc, axis=0) > heel_strike_threshold)[0]

# Debug: Print detected heel strikes
print(f"Left Heel Strikes ({left_heel_strikes.shape}): {left_heel_strikes}")
print(f"Right Heel Strikes ({right_heel_strikes.shape}): {right_heel_strikes}")

# Apply Kalman filter to estimate orientation over time
for t in range(1, left_shank_acc.shape[1]):
    q_left, P_left = kalman_filter_update(q_left, P_left, left_shank_gyro[:, t], left_shank_acc[:, t], dt)
    q_right, P_right = kalman_filter_update(q_right, P_right, right_shank_gyro[:, t], right_shank_acc[:, t], dt)

    # Rotate acceleration to global frame using scipy's Rotation class
    acc_left_global = R.from_quat([q_left[1], q_left[2], q_left[3], q_left[0]]).apply(left_shank_acc[:, t])
    acc_right_global = R.from_quat([q_right[1], q_right[2], q_right[3], q_right[0]]).apply(right_shank_acc[:, t])

    # Integrate acceleration to get velocity and position
    vel_left[:, t] = vel_left[:, t-1] + acc_left_global * dt
    pos_left[:, t] = pos_left[:, t-1] + vel_left[:, t] * dt
    vel_right[:, t] = vel_right[:, t-1] + acc_right_global * dt
    pos_right[:, t] = pos_right[:, t-1] + vel_right[:, t] * dt

    # Zero-velocity update (ZUPT) during stance phase
    if t in left_heel_strikes:
        vel_left[:, t] = 0
    if t in right_heel_strikes:
        vel_right[:, t] = 0

# Debug: Print final positions
print(f"Final Left Position: {pos_left[:, -1]}")
print(f"Final Right Position: {pos_right[:, -1]}")

# Calculate stride length based on position data
stride_lengths_left = np.linalg.norm(pos_left[:, left_heel_strikes[1:]] - pos_left[:, left_heel_strikes[:-1]], axis=0)
stride_lengths_right = np.linalg.norm(pos_right[:, right_heel_strikes[1:]] - pos_right[:, right_heel_strikes[:-1]], axis=0)
average_stride_length = np.mean(np.hstack((stride_lengths_left, stride_lengths_right)))

# Print the results
print(f"Average Stride Length: {average_stride_length:.2f} meters")

