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

# Example usage:
dt = 0.01  # Time step
process_noise = 1e-5
measurement_noise = 1e-3

kf = QuaternionKalmanFilter(dt, process_noise, measurement_noise)

# Simulated IMU data (angular velocity and orientation measurements)
imu_data = [
    {"omega": [0.01, 0.02, -0.01], "orientation": [0.99, 0.01, 0.02]},
    {"omega": [0.02, -0.01, 0.03], "orientation": [0.98, -0.02, 0.03]},
    {"omega": [-0.01, 0.03, -0.02], "orientation": [0.97, 0.03, -0.01]},
]

for data in imu_data:
    kf.predict(data["omega"])
    kf.update(data["orientation"])
    print("Estimated orientation:", kf.get_orientation())
