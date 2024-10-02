from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from filterpy.kalman import KalmanFilter


p_imu = Path('/data/datasets/ego4d_data/v2/imu/0ce20f1b-104c-48aa-a957-75f772238639.csv')
df = pd.read_csv(p_imu, index_col=0).dropna()
df = df.sort_values('canonical_timestamp_ms')
del df['component_timestamp_ms']
df = df[['accl_x', 'accl_y', 'accl_z', 'gyro_x', 'gyro_y', 'gyro_z', 'canonical_timestamp_ms']]

# Initial state vector [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
x = np.zeros((12, 1))

# Initial covariance matrix
P = np.eye(12)


def predict(x, P, dt):
    # State transition matrix
    F = np.eye(12)
    F[0:6, 6:12] = np.eye(6) * dt

    # Process noise covariance
    Q = np.eye(12) * 0.1

    # Predict state and covariance
    x = F @ x
    P = F @ P @ F.T + Q

    return x, P


def update(x, P, z, H, R):
    # Kalman gain
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

    # Update state and covariance
    x = x + K @ (z - H @ x)
    P = (np.eye(12) - K @ H) @ P

    return x, P


def process_imu_data(acc, gyro, dt):
    global x, P

    # Predict
    x, P = predict(x, P, dt)

    # Update with accelerometer data
    z_acc = acc.reshape(3, 1)
    H_acc = np.zeros((3, 12))
    H_acc[0:3, 0:3] = np.eye(3)
    R_acc = np.eye(3) * 0.1
    x, P = update(x, P, z_acc, H_acc, R_acc)

    # Update with gyroscope data
    z_gyro = gyro.reshape(3, 1)
    H_gyro = np.zeros((3, 12))
    H_gyro[0:3, 9:12] = np.eye(3)
    R_gyro = np.eye(3) * 0.01
    x, P = update(x, P, z_gyro, H_gyro, R_gyro)

    return x[0:6]  # position(x, y, z) and orientation(roll, pitch, yaw)


imu_data = df.values
t0 = imu_data[0, -1]   # in ms

poses = []
for i in tqdm(list(range(1, len(imu_data)))):
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, t1 = imu_data[i]

    dt = t1 - t0
    t0 = t1

    pose = process_imu_data(imu_data[i, :3], imu_data[i, 3:6], dt / 1000)
    poses.append([t1, *pose.flatten()])  # t1 (ms), x, y, z, roll, pitch, yaw

poses = np.array(poses)


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-1, 10])
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')

# Line object to update
trajectory, = ax.plot([], [], [], lw=1)

def update(frame):
    trajectory.set_data_3d(poses[:frame, 1:4].T)

    return trajectory,


# Create animation
anim = FuncAnimation(fig, update, frames=range(0, 1000), blit=True)
print('Compounding the animation...')
anim.save('notebooks/imu.mp4', fps=200)
