import numpy as np

# Quaternion normalization and multiplication as in the previous script
def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# Convert gyroscope readings (angular velocity) to quaternion derivative
def gyro_to_quaternion(gyro_data, dt):
    wx, wy, wz = gyro_data
    omega = np.array([0, wx, wy, wz]) * dt / 2.0
    return omega

# Detect stance phase based on accelerometer magnitude
def detect_stance_phase(accel_data, threshold=0.5):
    accel_magnitude = np.linalg.norm(accel_data)
    return np.abs(accel_magnitude - 9.81) < threshold

# Estimate IMU orientation with conditional accelerometer correction
def estimate_orientation_imu_walking(gyro_data, accel_data, dt, q, stance_threshold=0.5):
    # Update quaternion using gyroscope data
    omega = gyro_to_quaternion(gyro_data, dt)
    q = quaternion_multiply(q, omega)
    q = normalize_quaternion(q)
    
    # Detect stance phase
    is_stance = detect_stance_phase(accel_data, stance_threshold)
    
    if is_stance:
        # If in stance phase, use accelerometer data to correct orientation
        accel = accel_data / np.linalg.norm(accel_data)  # Normalize accelerometer data (gravity vector)
        gravity = np.array([0, 0, -1])  # Global gravity vector
        
        # Find correction quaternion to align accelerometer data with gravity
        v = np.cross(accel, gravity)
        s = np.sqrt((1 + np.dot(accel, gravity)) * 2)
        correction_q = np.array([s / 2, v[0] / s, v[1] / s, v[2] / s])
        
        # Apply correction quaternion to orientation
        q = quaternion_multiply(q, correction_q)
        q = normalize_quaternion(q)
    
    return q

# Example usage
if __name__ == "__main__":
    # Initial orientation (identity quaternion)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Example IMU data (simulated for illustration)
    gyro_data = np.array([0.01, 0.02, 0.015])  # Gyroscope data in rad/s
    accel_data_stance = np.array([0, 0, -9.81])  # Accelerometer during stance phase
    accel_data_swing = np.array([0.2, 1.5, -8.5])  # Accelerometer during swing phase
    dt = 0.01  # Time step in seconds
    
    # Simulating a walking cycle
    for i in range(50):
        if i % 10 < 5:  # Simulate stance phase for half of each cycle
            accel_data = accel_data_stance
        else:
            accel_data = accel_data_swing
        
        # Estimate orientation
        q = estimate_orientation_imu_walking(gyro_data, accel_data, dt, q)
        print(f"Estimated Quaternion at step {i}: {q}")
