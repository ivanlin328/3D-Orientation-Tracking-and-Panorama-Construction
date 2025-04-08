import cv2
import numpy as np
import matplotlib.pyplot as plt
from load_data import *
from panorama import *
from IMU_calibration import * 
from Orientation_tracking import *
from Optimize import *
def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix
    
    Args:
        q: quaternion in format [w, x, y, z]
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q
    
    # First row
    r00 = 1 - 2*y*y - 2*z*z
    r01 = 2*x*y - 2*w*z
    r02 = 2*x*z + 2*w*y
    
    # Second row
    r10 = 2*x*y + 2*w*z
    r11 = 1 - 2*x*x - 2*z*z
    r12 = 2*y*z - 2*w*x
    
    # Third row
    r20 = 2*x*z - 2*w*y
    r21 = 2*y*z + 2*w*x
    r22 = 1 - 2*x*x - 2*y*y
    
    return np.array([[r00, r01, r02],
                    [r10, r11, r12],
                    [r20, r21, r22]])

def create_rotation_matrices_from_imu(optimized_quaternions):
    """Convert optimized quaternions to rotation matrices
    
    Args:
        optimized_quaternions: array of quaternions from IMU optimization
    Returns:
        Array of 3x3 rotation matrices
    """
    num_frames = len(optimized_quaternions)
    rotation_matrices = np.zeros((3, 3, num_frames))
    
    for i in range(num_frames):
        rotation_matrices[:, :, i] = quaternion_to_rotation_matrix(optimized_quaternions[i])
    
    return rotation_matrices

def process_panorama_no_vicon(cam_filepath, imu_filepath):
    """Process panorama creation without VICON data
    
    Args:
        cam_filepath: path to camera data file
        imu_filepath: path to IMU data file
    """
    # Load camera data
    cam_data = read_data(cam_filepath)
    images = cam_data['cam']
    cam_ts = cam_data['ts'][0]
    
    # Get image dimensions
    cam_H, cam_W = images.shape[0], images.shape[1]
    
    # Process IMU data and get optimized quaternions
    timestamps, accel_calibrated, gyro_calibrated = process_dataset(imu_filepath)
    initial_quaternions = integrate_orientation(gyro_calibrated, timestamps)
    optimized_quaternions = optimize_quaternions(
        initial_quaternions,
        gyro_calibrated,
        accel_calibrated,
        timestamps,
        learning_rate=0.007,
        num_iterations=100,
        tolerance=1e-6
    )
    
    # Convert quaternions to rotation matrices
    rotation_matrices = create_rotation_matrices_from_imu(optimized_quaternions)
    
    # Create spherical coordinates
    theta, phi = create_spherical_points(cam_H, cam_W)
    
    # Create panorama
    panorama = create_panorama(theta, phi, cam_ts, timestamps, rotation_matrices, images)
    
    # Display result
    plt.figure(figsize=(15, 10))
    plt.imshow(panorama)
    plt.title('Panorama (using IMU data)')
    plt.axis('off')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Process dataset 10
    cam_filepath10 = '/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR1/data/test_cam/cam10.p'
    imu_filepath10 = '/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR1/data/test_imu/imuRaw10.p'
    print("Processing dataset 10...")
    process_panorama_no_vicon(cam_filepath10, imu_filepath10)
    
    # Process dataset 11
    cam_filepath11 = '/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR1/data/test_cam/cam11.p'
    imu_filepath11 = '/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR1/data/test_imu/imuRaw11.p'
    print("Processing dataset 11...")
    process_panorama_no_vicon(cam_filepath11, imu_filepath11)