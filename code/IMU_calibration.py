import numpy as np
from load_data import read_data

def read_imu_data(filepath):
    imu_data = read_data(filepath)  
    timestamps = imu_data[0,:]        # timestamps
    accel_data = imu_data[1:4, :]     # ax, ay, az 
    gyro_data = imu_data[4:7, :]      # wx, wy, wz
    
    return timestamps, accel_data, gyro_data

def calibrate_imu(accel_data, gyro_data, static_duration=100):
    
    # Constants from the ADC specifications
    Vref = 3300  # mV
    ADC_MAX = 1023  # 10-bit ADC
    
    # Sensitivities (from datasheet)
    ACCEL_SENSITIVITY = 300  # mV/g
    GYRO_SENSITIVITY = 3.33  # mV/degree/sec
    
    # Calculate scale factors using the formula: Vref/1023/sensitivity
    accel_scale = Vref / ADC_MAX / ACCEL_SENSITIVITY  # convert to g
    gyro_scale = (Vref / ADC_MAX / GYRO_SENSITIVITY) * (np.pi / 180)  # convert to rad/sec
    
    # Use the data during stationary period
    static_accel = accel_data[:, :static_duration]  # shape: (3, static_duration)
    static_gyro = gyro_data[:, :static_duration]    # shape: (3, static_duration)
    
    # Calculate biases
    accel_bias = np.mean(static_accel, axis=1, keepdims=True)  # shape: (3,1)
    gyro_bias = np.mean(static_gyro, axis=1, keepdims=True)    # shape: (3,1)
    
    return accel_bias, gyro_bias, accel_scale, gyro_scale

def apply_calibration(accel_data, gyro_data, accel_bias, gyro_bias, accel_scale, gyro_scale):
    
    # Apply calibration formula: value = (raw - bias) × scale_factor
    accel_calibrated = (accel_data - accel_bias) * accel_scale
    gyro_calibrated = (gyro_data - gyro_bias) * gyro_scale
    accel_calibrated[2, :] += 1.0  # Add 1g to z-axis after calibration
    
    return accel_calibrated, gyro_calibrated

def verify_calibration(accel_calibrated, gyro_calibrated, static_duration=100):

    # Check static period data
    static_accel = accel_calibrated[:, :static_duration]
    static_gyro = gyro_calibrated[:, :static_duration]
    
    # Calculate mean 
    accel_mean = np.mean(static_accel, axis=1)
    gyro_mean = np.mean(static_gyro, axis=1)
    
    #print("Calibrated accelerometer mean during static period (in g):", accel_mean)
    #print("Should be close to [0, 0, 1]")
    #print("\nCalibrated gyroscope mean during static period (in rad/sec):", gyro_mean)
    #print("Should be close to [0, 0, 0]")
    
def process_dataset(filepath): 
    """
    Process a single dataset with calibration
    """
    timestamps, accel_data, gyro_data = read_imu_data(filepath)
    accel_bias, gyro_bias, accel_scale, gyro_scale = calibrate_imu(accel_data, gyro_data)
    
    accel_calibrated, gyro_calibrated = apply_calibration(
        accel_data, gyro_data,
        accel_bias, gyro_bias,
        accel_scale, gyro_scale
    )
    
    verify_calibration(accel_calibrated, gyro_calibrated)
    return timestamps,accel_calibrated, gyro_calibrated

# Example usage
if __name__ == "__main__":
    filepath = '/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR1/data/imu/imuRaw1.p'
    timestamps,accel_calibrated, gyro_calibrated = process_dataset(filepath)

