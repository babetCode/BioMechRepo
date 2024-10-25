"""
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
Using FilterPy library
Code Status: In Progress
________________________________________________________________________

‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
Author: Adrien Babet | GitHub: @babetcode | Email: adrienbabet1@gmail.com  
_______________________________________________________________________________
"""

import imufunctions as myfns
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.common import Q_discrete_white_noise
from scipy.spatial.transform import Rotation as R


def imu_ekf(data_set, data_rate):
    def quat_to_rotation_matrix(q):
        """Convert a quaternion to a 3x3 rotation matrix."""
        r = R.from_quat(q)
        return r.as_matrix()

    # Initialize EKF with 10 state variables
    # (x, y, z, v_x, v_y, v_z, q_w, q_x, q_y, q_z)
    # 6 measurements
    ekf = EKF(dim_x=10, dim_z=6)

    # Initial state vector
    ekf.x = np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.])

    # Covariance matrix
    ekf.P = np.eye(10) * 1.0  # Initial uncertainty
    
    def state_transition(dt, state, acc, gyro):
        """
        Predict the next position and orientation using acceleration
        and angular velocity (gyro) readings.
        
        Parameters
        -
        dt:
            Time step (1 / frequency)
        state:
            Current state vector
            (position, velocity, orientation quaternion)
        acc:
            Linear acceleration readings (in g's)
        gyro:
            Rotational velocity readings (in deg/sec)

        Returns
        -
        predicted state:
            np.hstack([pos_new, v_new, q_new])
        """
        # Extract quaternion
        q_w, q_x, q_y, q_z = state[6:10]
        q = np.array([q_w, q_x, q_y, q_z])
        q = q / np.linalg.norm(q) # Normalize to avoid drift
        
        # Update orientation (convert gyro to rad/sec first)
        omega = np.radians(gyro)
        r = R.from_quat(q)
        delta_q = r.integrate(omega * dt).as_quat()
        q_new = delta_q / np.linalg.norm(delta_q)
        
        # Compute the acceleration in the world frame
        rotation_matrix = quat_to_rotation_matrix(q_new)
        acc_world = rotation_matrix @ acc  # Rotate to world frame
        
        # Update velocity and position using the acceleration
        v_new = state[3:6] + acc_world * dt
        pos_new = state[0:3] + state[3:6] * dt + 0.5 * acc_world * dt**2
        
        # Return the updated state
        return np.hstack([pos_new, v_new, q_new])
    
    def jacobian_state_transition(state, dt, acc, gyro):
        # In practice, compute the Jacobian matrix analytically.
        # Here, we approximate it with finite differences or use
        # a simplified version.
        return np.eye(10)  # Placeholder for Jacobian
    
    ekf.F = np.eye(10)  # Will be updated dynamically
    ekf.H = np.zeros((6, 10))  # Measurement function (placeholder)
    ekf.R = np.eye(6) * 0.01  # Measurement noise (tunable)
    ekf.Q = Q_discrete_white_noise(
        dim=3, dt=data_rate, var=0.1)  # Process noise (tunable)

    # Measurement function
    def measurement_function(state):
        # We measure position and orientation from the IMU
        return state[:6]
    
    # Jacobian of the measurement function
    def jacobian_measurement_function(state):
        return np.eye(6, 10)
    
    for n, measurement in enumerate(data_set):
        acc = measurement[:3]
        gyro = measurement[3:]

        # Predict step (process model)
        ekf.F = jacobian_state_transition(ekf.x, data_rate, acc, gyro)
        ekf.predict_update(z=np.hstack([ekf.x[:3], ekf.x[6:9]]),
                        u=np.hstack([acc, gyro]),
                        H_jacobian=jacobian_measurement_function,
                        hx=measurement_function)
        # At each step, print or log position and orientation
        print(f"Time: {n * data_rate}s, Position: {ekf.x[:3]}, Quaternion: {ekf.x[6:10]}")