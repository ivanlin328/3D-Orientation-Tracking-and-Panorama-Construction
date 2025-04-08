import cv2
import numpy as np
import matplotlib.pyplot as plt
from load_data import *

def find_closest_timestamp(target_ts, reference_ts):
    """Find closest timestamp index"""
    time_differences = np.abs(reference_ts - target_ts)
    closest_idx = np.argmin(time_differences)
    return closest_idx  

def load_data(filepath1, filepath2):
    vicon_data = read_data(filepath1)
    rotation_matrices = vicon_data['rots']
    vicon_ts = vicon_data['ts'][0]
    
    cam_data = read_data(filepath2)
    images = cam_data['cam']
    cam_ts = cam_data['ts'][0]
    
    return rotation_matrices, images, vicon_ts, cam_ts

def spherical_to_cartesian(theta, phi, r=1.0):
    """Convert spherical coordinates to cartesian coordinates"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.stack([x, y, z], axis=-1)

def cartesian_to_spherical(cart_coords):
    """Convert cartesian coordinates to spherical coordinates"""
    x = cart_coords[..., 0]
    y = cart_coords[..., 1]
    z = cart_coords[..., 2]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / (r + 1e-10))  # Added small epsilon to prevent division by zero
    phi = np.arctan2(y, x)
    
    return theta, phi

def create_spherical_points(cam_H, cam_W):
    """Create spherical coordinates for each pixel in the camera image"""
    fov_H, fov_V = np.radians(60), np.radians(45)
    
    center_u, center_v = cam_H // 2 , cam_W // 2
    
    V, U = np.meshgrid(np.arange(cam_W), np.arange(cam_H))
    
    u_normalized = 2 * (U / cam_H) - 1
    v_normalized = 2 * (V / cam_W) - 1
    
    theta = (np.pi/2) + (fov_V/2) * u_normalized
    phi = (fov_H/2) * v_normalized

    
    return theta, phi
  

def create_panorama(theta, phi, cam_ts, vicon_ts, rotation_matrices, images):
    """Create panorama from images using rotation matrices"""
    # Initialize panorama canvas
    canvas_H, canvas_W = 720, 1280
    canvas_img = np.zeros((canvas_H, canvas_W, 3), dtype=np.uint8)
    
    # Convert to cartesian coordinates
    cart_coords = spherical_to_cartesian(theta, phi)
    
    # Process each image
    for t in range(len(cam_ts)):
        # Find closest vicon timestamp for current image
        closest_idx = find_closest_timestamp(cam_ts[t], vicon_ts)
        rot_mat = rotation_matrices[:, :, closest_idx]  
        
        # Rotate coordinates to world frame
        cart_coords_rot = np.einsum('ij,hwj->hwi', rot_mat.T, cart_coords)
        
        # Convert back to spherical coordinates
        theta_rot, phi_rot = cartesian_to_spherical(cart_coords_rot)
        
        # Map to panorama coordinates
        u = ((theta_rot) / np.pi * canvas_H).astype(int)
        # Map phi from [-pi, pi] to [0, canvas_W]
        v = ((phi_rot + np.pi) / (2 * np.pi) * canvas_W).astype(int)
        
        # Update panorama image
        canvas_img[u, v]  = images[..., t]
    
    return canvas_img

if __name__ == "__main__":
    # Load data
    filepath1 = "/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR1/data/vicon/viconRot1.p"
    filepath2 = "/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR1/data/cam/cam1.p"
    rotation_matrices, images, vicon_ts, cam_ts = load_data(filepath1, filepath2)
    
    # Get image dimensions
    cam_H, cam_W = images.shape[0], images.shape[1]
    
    # Create spherical coordinates for camera image
    theta, phi = create_spherical_points(cam_H, cam_W)
    
    # Create panorama
    panorama = create_panorama(theta, phi, cam_ts, vicon_ts, rotation_matrices, images)
    
    # Display result
    plt.figure(figsize=(15, 10))
    plt.imshow(panorama)
    plt.axis('off')
    plt.show()
    
    
    
    
    



    
 


    
    
   


