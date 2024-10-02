import sys
sys.path.insert(0, 'c:/Users/goper/Files/vsCode/490R/VScodeIMUrepo')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh

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
from AdriensFunctions import adrienC3Dpath, c3d_analogs_df, imu

def animate_imu_rotation(kf, all_data, animation_speed=1.0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    frameSize = 5
    ax.set_xlim([-frameSize, frameSize])
    ax.set_ylim([-frameSize, frameSize])
    ax.set_zlim([-frameSize, frameSize])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Load the 3D model and scale it down
    imu_mesh = mesh.Mesh.from_file('house.stl')
    imu_mesh.vectors /= 10.0  # Scale down the model by a factor of 10

    # Create a Poly3DCollection with lighting enabled
    imu_poly = Poly3DCollection(imu_mesh.vectors, facecolors='cyan', edgecolors='k', linewidths=0.5)
    imu_poly.set_alpha(0.7)
    
    def init():
        ax.add_collection3d(imu_poly)
        return ax,

    def update(frame):
        linear_acceleration = frame[:3]  # Linear acceleration around the body pitch, roll, and yaw axes
        angular_velocity = frame[3:6]  # Angular velocity measurements around the body pitch, roll, and yaw axes

        kf.predict(angular_velocity)
        
        # Assuming we have some orientation measurements to update the filter with.
        orientation_measurement = np.array([0.99, 0.01, 0.02])  # Placeholder for actual orientation measurements
        
        kf.update(orientation_measurement)
        
        q = kf.get_orientation()
        
        # Convert quaternion to rotation matrix
        R = np.array([
            [1 - 2*(q[2]**2 + q[3]**2), 2*(q[1]*q[2] - q[3]*q[0]), 2*(q[1]*q[3] + q[2]*q[0])],
            [2*(q[1]*q[2] + q[3]*q[0]), 1 - 2*(q[1]**2 + q[3]**2), 2*(q[2]*q[3] - q[1]*q[0])],
            [2*(q[1]*q[3] - q[2]*q[0]), 2*(q[2]*q[3] + q[1]*q[0]), 1 - 2*(q[1]**2 + q[2]**2)]
        ])
        
        # Rotate the IMU mesh
        rotated_vectors = np.dot(imu_mesh.vectors.reshape(-1, 3), R.T).reshape(-1, 3, 3)
        imu_poly.set_verts(rotated_vectors)
        return ax,

    ani = FuncAnimation(fig, update, frames=all_data, init_func=init, blit=True, interval=1000/(144*animation_speed))
    plt.show()

def main():
    dt = 1/144  # Time step for 144 Hz sampling rate
    process_noise = 1e-5
    measurement_noise = 1e-3

    kf = QuaternionKalmanFilter(dt, process_noise, measurement_noise)

    mypath = adrienC3Dpath()
    df = c3d_analogs_df('C07', 'Fast', '07', mypath)
    myIMU = imu('myFirstIMU', df, 9)

    all_data = myIMU.all_data.T.to_numpy()  # Transpose to get frames as rows and convert to numpy array

    animate_imu_rotation(kf, all_data, animation_speed=1.0)  # Adjust animation_speed as needed

if __name__ == '__main__':
    main()
