import numpy as np

def Kalman_Filter(measurements, initial_state, initial_error_covariance, state_transition, 
                  measurement_model, process_noise, measurement_covariance):
    # Set variables
    A = state_transition      # n x n matrix
    H = measurement_model     # m x n matrix
    Q = process_noise         # n x n matrix
    R = measurement_covariance # m x m matrix
    x = initial_state         # n x 1 vector (state)
    P = initial_error_covariance # n x n matrix (error covariance)
    
    filtered_data = []
    
    for z in measurements:
        # Prediction step
        xp = A @ x                             # n x 1: Predict state (state transition)
        Pp = A @ P @ A.T + Q                   # n x n: Predict error covariance
        
        # Measurement update step
        S = H @ Pp @ H.T + R                   # m x m: Innovation covariance
        K = Pp @ H.T @ np.linalg.inv(S)        # n x m: Kalman gain
        
        z = np.reshape(z, (-1, 1))             # Ensure measurement is m x 1 column vector
        x = xp + K @ (z - H @ xp)              # n x 1: Update state estimate
        P = Pp - K @ H @ Pp                    # n x n: Update error covariance
        
        filtered_data.append(x)
    
    return np.array(filtered_data)              # Return filtered state estimates

# Example usage:
# n: state dimension, m: measurement dimension
n = 4
m = 2
measurements = [np.random.randn(m, 1) for _ in range(10)]  # Example m x 1 measurement vectors
initial_state = np.zeros((n, 1))  # n x 1 initial state vector
initial_error_covariance = np.eye(n) * 6  # n x n initial error covariance matrix
state_transition = np.eye(n)  # n x n state transition matrix
measurement_model = np.random.randn(m, n)  # m x n measurement model matrix
process_noise = np.eye(n) * 1e-5  # n x n process noise matrix
measurement_covariance = np.eye(m)  # m x m measurement noise covariance matrix

filtered_results = Kalman_Filter(measurements, initial_state, initial_error_covariance, 
                                 state_transition, measurement_model, process_noise, 
                                 measurement_covariance)
