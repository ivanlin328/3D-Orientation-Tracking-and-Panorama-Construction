
# IMU-Based Panorama and Orientation Tracking

This project provides a set of Python tools for processing IMU data, performing orientation tracking, and generating panoramic visualizations using magnetic and inertial sensor data.

## Example Results

### Orientation Tracking vs VICON Ground Truth

![optimize_1.png](result/dataset%201/optimize_1.png)

### Panorama Construction Example

![panorama_1.png](result/dataset%201/panorama_1.png)



## Features

- **Orientation Tracking**: Sensor fusion from IMU data to estimate device orientation  
  (`Orientation_tracking.py`)
- **Panorama Visualization**: Generate 360° panorama visualization based on orientation data  
  (`panorama.py`, `test_panorama.py`)
- **Magnetometer Calibration**: Calibrate IMU sensors to improve heading estimation  
  (`IMU_calibration.py`)
- **Data Loader**: Read and preprocess IMU log data  
  (`load_data.py`)
- **Optimization**: Utility functions for optimization and sensor calibration  
  (`Optimize.py`)
- **Test Scripts**: Example test cases for panorama and object visualization  
  (`test_object.py`, `test_panorama.py`)
- **Rotation Visualization**: Plot rotation trajectories  
  (`rotplot.py`)

## Installation

Clone the repository:

```bash
git clone https://github.com/ivanlin328/3D-Orientation-Tracking-and-Panorama-Construction.git
cd 3D-Orientation-Tracking-and-Panorama-Construction
````

Install dependencies:

```bash
pip install numpy matplotlib scipy pandas pyquaternion
```

## Usage

Example: Run orientation tracking on sample IMU data:

```bash
python Orientation_tracking.py
```

Example: Generate panorama visualization:

```bash
python test_panorama.py
```

Example: Perform magnetometer calibration:

```bash
python IMU_calibration.py
```

## File Structure

| File                      | Description                              |
| ------------------------- | ---------------------------------------- |
| `Orientation_tracking.py` | Main script for IMU orientation tracking |
| `panorama.py`             | Panorama visualization tool              |
| `test_panorama.py`        | Test example for panorama generation     |
| `test_object.py`          | Object visualization test                |
| `IMU_calibration.py`      | IMU calibration utility                  |
| `Optimize.py`             | Optimization helpers                     |
| `load_data.py`            | Data loader for IMU logs                 |
| `rotplot.py`              | Rotation plotting                        |

## License

This project is open-source and available under the MIT License.


