from adriensdir import BioMechDir
mydir = BioMechDir().add_imu_func_path()
import imufunctions as myfns
from filterpy.kalman import ExtendedKalmanFilter as EKF
import numpy as np

def orient_ekf(measurements, data_rate):
    # 1. Initialize EKF with state and covariance
    
    ekf = EKF(dim_x=7, dim_z=6)
    # Initial quaternion and angular velocity
    ekf.x = np.array([1, 0, 0, 0, 0, 0, 0])
    ekf.P = np.eye(7) * 0.01  # Initial covariance

    # Define state transition (f) and measurement functions (h) here
    
    # 2. Loop over measurements
    estimated_states = []
    for z in measurements:
        # 3. Predict step
        ekf.predict()
        
        # 4. Update step
        ekf.update(z)
        
        # 5. Store estimated state
        estimated_states.append(ekf.x.copy())

    return estimated_states
