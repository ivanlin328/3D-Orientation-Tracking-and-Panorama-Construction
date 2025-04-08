import numpy as np
import jax.numpy as jnp
from transforms3d.euler import quat2euler
from transforms3d.euler import mat2euler
import matplotlib.pyplot as plt
from load_data import read_data
from IMU_calibration import process_dataset

def read_vicon_data(filepath):
    """
    Read and process VICON data to match IMU timestamps
    """
    vicon_data = read_data(filepath)
    rotation_matrices = vicon_data['rots']  # shape: (3, 3, N)
    timestamps = vicon_data['ts'][0]        # shape: (N,) after removing singleton dimension
    
    euler_angles = []
    for t in range(len(timestamps)-1):  # Match the IMU integration loop
        # Get rotation matrix at time t
        # Note: rotation_matrices is (3,3,N), so we index with [:,:,t] to get the 3x3 matrix
        rot_matrix = rotation_matrices[:,:,t]
        
        # Convert rotation matrix to euler angles (roll, pitch, yaw)
        euler_angle = mat2euler(rot_matrix, 'sxyz')
        euler_angles.append(euler_angle)
    
    return np.array(euler_angles), timestamps

def quaternion_exp(w):
    """
    w = [wx,wy,wz]
    θ = ||w||  
    η = w/||w||
    """
    theta = jnp.linalg.norm(w)   
    return jnp.where(
        theta < 1e-8,
        jnp.array([1., w[0], w[1], w[2]]),
        jnp.array([jnp.cos(theta/2),
                  w[0]/theta * jnp.sin(theta/2),
                  w[1]/theta * jnp.sin(theta/2),
                  w[2]/theta * jnp.sin(theta/2)])
    )
def quaternion_multiply(q1, q2):
    
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_inverse(q):
    
    return jnp.array([q[0], -q[1], -q[2], -q[3]]) / jnp.sum(q**2)

def integrate_orientation(gyro_calibrated,timestamps):
    q=np.array([1.,0.,0.,0.])   #q0=[1,0,0,0]
    quaternions=[q]
    for t in range(len(timestamps)-1):
        dt = timestamps[t+1]-timestamps[t]
        w = gyro_calibrated[:,t]
        dq = quaternion_exp(0.5 * dt * w)  #calculate exp([0, τtωt/2])
        q = quaternion_multiply(q,dq)      #calculate qt multiply exp([0, τtωt/2])
        q=q/np.linalg.norm(q)              #normalized q
        quaternions.append(q)
        
    return np.array(quaternions)

def quaternions_to_euler(quaternions):
    """"
    transfer quaternions to euler angle
    """
    euler_angles=[]
    for q in quaternions:
        euler_angle = quat2euler([q[0], q[1], q[2], q[3]], 'sxyz')
        euler_angles.append(euler_angle)
    return np.array(euler_angles)

def plot_orientation(timestamps,imu_euler,vicon_euler):
    
    min_length = min(len(timestamps), len(vicon_euler))
    
    timestamps = timestamps[:min_length]
    imu_euler = imu_euler[:min_length]
    vicon_euler = vicon_euler[:min_length]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    # Plot Roll
    ax1.plot(timestamps, np.rad2deg(imu_euler[:, 0]), 'b-', label='IMU')
    ax1.plot(timestamps, np.rad2deg(vicon_euler[:, 0]), 'r--', label='VICON')
    ax1.set_ylabel('Roll (degrees)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Pitch
    ax2.plot(timestamps, np.rad2deg(imu_euler[:, 1]), 'b-', label='IMU')
    ax2.plot(timestamps, np.rad2deg(vicon_euler[:, 1]), 'r--', label='VICON')
    ax2.set_ylabel('Pitch (degrees)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot Yaw
    ax3.plot(timestamps, np.rad2deg(imu_euler[:, 2]), 'b-', label='IMU')
    ax3.plot(timestamps, np.rad2deg(vicon_euler[:, 2]), 'r--', label='VICON')
    ax3.set_ylabel('Yaw (degrees)')
    ax3.set_xlabel('Time (s)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    

    

filepath = '/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR1/data/imu/imuRaw3.p'
timestamps,accel_calibrated, gyro_calibrated = process_dataset(filepath) 

vicon_euler, vicon_timestamps  = read_vicon_data('/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR1/data/vicon/viconRot3.p')
   

quaternions = integrate_orientation(gyro_calibrated, timestamps)
      
imu_euler = quaternions_to_euler(quaternions)
    
plot_orientation(timestamps[:len(vicon_euler)], imu_euler[:len(vicon_euler)], vicon_euler)






    
        
        
        
        
        
        
  
        






    

