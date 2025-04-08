from Orientation_tracking import *
from IMU_calibration import *
from Optimize import *
import matplotlib.pyplot as plt

def evaluate_optimization_no_vicon(optimized_quaternions, timestamps):
    """Evaluate and visualize optimization results without VICON data"""
    optimized_euler = quaternions_to_euler(optimized_quaternions)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot Roll
    ax1.plot(timestamps, np.rad2deg(optimized_euler[:, 0]), 'g-', label='Optimized')
    ax1.set_ylabel('Roll (degrees)')
    ax1.set_title('Optimized Orientation')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Pitch
    ax2.plot(timestamps, np.rad2deg(optimized_euler[:, 1]), 'g-', label='Optimized')
    ax2.set_ylabel('Pitch (degrees)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot Yaw
    ax3.plot(timestamps, np.rad2deg(optimized_euler[:, 2]), 'g-', label='Optimized')
    ax3.set_ylabel('Yaw (degrees)')
    ax3.set_xlabel('Time (s)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

def process_and_plot_dataset(filepath):
    """Process and plot a single dataset"""
    # Load and process data
    timestamps, accel_calibrated, gyro_calibrated = process_dataset(filepath)
    
    # Get initial quaternions from integration
    initial_quaternions = integrate_orientation(gyro_calibrated, timestamps)
    
    # Optimize quaternions
    optimized_quaternions = optimize_quaternions(
        initial_quaternions,
        gyro_calibrated,
        accel_calibrated,
        timestamps, 
        learning_rate=0.007,
        num_iterations=100,
        tolerance=1e-6
    )
    
    # Plot results
    evaluate_optimization_no_vicon(optimized_quaternions, timestamps)

# Example usage for both datasets
if __name__ == "__main__":
    # Process dataset 10
    filepath10 = '/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR1/data/test_imu/imuRaw10.p'
    print("Processing dataset 10...")
    process_and_plot_dataset(filepath10)
    
    # Process dataset 11
    filepath11 = '/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR1/data/test_imu/imuRaw11.p'
    print("Processing dataset 11...")
    process_and_plot_dataset(filepath11)