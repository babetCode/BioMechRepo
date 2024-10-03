import numpy as np

class QuaternionKalmanFilter:
    def __init__(self, dt, process_noise, measurement_noise):
        self.dt = dt
        self.q = np.array([1, 0, 0, 0], dtype=np.float64)  # Initial quaternion
        self.P = np.eye(4) * 0.1  # Initial covariance matrix
        self.Q = np.eye(4) * process_noise  # Process noise covariance
        self.R = np.eye(3) * measurement_noise  # Measurement noise covariance

    def predict(self, omega):
        # Quaternion derivative
        omega_matrix = np.array([
            [0, -omega[0], -omega[1], -omega[2]],
            [omega[0], 0, omega[2], -omega[1]],
            [omega[1], -omega[2], 0, omega[0]],
            [omega[2], omega[1], -omega[0], 0]
        ])
        q_dot = 0.5 * omega_matrix @ self.q

        # Predict the next state
        self.q = self.q + q_dot * self.dt
        self.q /= np.linalg.norm(self.q)  # Normalize quaternion

        # Predict the next covariance
        F = np.eye(4) + 0.5 * omega_matrix * self.dt
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        # Measurement matrix
        H = np.array([
            [-2*self.q[2], 2*self.q[3], -2*self.q[0], 2*self.q[1]],
            [2*self.q[1], 2*self.q[0], 2*self.q[3], 2*self.q[2]],
            [2*self.q[0], -2*self.q[1], -2*self.q[2], 2*self.q[3]]
        ])

        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update the state estimate
        y = z - H @ self.q
        self.q = self.q + K @ y
        self.q /= np.linalg.norm(self.q)  # Normalize quaternion

        # Update the covariance estimate
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

    def get_orientation(self):
        return self.q

# Import the necessary functions and classes from AdriensFunctions.py
from imufunctions import adrienC3Dpath, c3d_analogs_df, imu

def main():
    dt = 0.01  # Time step
    process_noise = 1e-5
    measurement_noise = 1e-3

    kf = QuaternionKalmanFilter(dt, process_noise, measurement_noise)

    mypath = adrienC3Dpath()
    df = c3d_analogs_df('C07', 'Fast', '07', mypath)
    myIMU = imu('myFirstIMU', df, 5)

    all_data = myIMU.all_data.T.to_numpy()  # Transpose to get frames as rows and convert to numpy array

    for frame in all_data:
        linear_acceleration = frame[:3]  # Linear acceleration around the body pitch, roll, and yaw axes
        angular_velocity = frame[3:6]  # Angular velocity measurements around the body pitch, roll, and yaw axes

        kf.predict(angular_velocity)
        
        # Assuming we have some orientation measurements to update the filter with.
        orientation_measurement = np.array([0.99, 0.01, 0.02])  # Placeholder for actual orientation measurements
        
        kf.update(orientation_measurement)
        
        print("Estimated orientation:", kf.get_orientation())

if __name__ == '__main__':
    main()