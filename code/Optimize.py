import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
import matplotlib.pyplot as plt
from functools import partial
from Orientation_tracking import *   
from IMU_calibration import *


@jit
def safe_normalize(v, eps=1e-10):
    """Safely normalize a vector"""
    norm = jnp.linalg.norm(v)
    return v / (norm + eps)

@jit
def quaternion_log(q):
    """Simple and direct quaternion log implementation"""
    q = safe_normalize(q)
    qw = q[0]
    qv = q[1:]
    
    qv_norm = jnp.linalg.norm(qv) + 1e-10
    qw = jnp.clip(qw, -1.0 + 1e-7, 1.0 - 1e-7)
    
   
    angle = 2.0 * jnp.arccos(qw)
    
    factor = jnp.where(qv_norm > 1e-7,
                      angle / qv_norm,
                      2.0)
    
    return factor * qv

@jit
def quaternion_multiply(p, q):
    """Quaternion multiplication"""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    
    r = jnp.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw
    ])
    
    return r

@jit
def quaternion_inverse(q):
    """Quaternion inverse"""
    return q * jnp.array([1., -1., -1., -1.])

@jit
def compute_motion_errors(quaternions, angular_velocities, dt):
    """Motion error computation"""
    T = len(dt)
    
    # Compute motion model f(qt, τtωt)
    def f(q, w, tau):
        """f(qt, τtωt) = qt ○ exp([0, τtωt/2])"""
        w_scaled = 0.5 * tau * w
        dq = quaternion_exp(w_scaled)
        return quaternion_multiply(q, dq)
    
    # Get pairs of quaternions
    q_t = quaternions[:T]
    q_tp1 = quaternions[1:T+1]
    
    # Compute predicted quaternions
    predicted_q = vmap(f)(q_t, angular_velocities[:, :T].T, dt)
    
    # Compute errors
    error_q = vmap(quaternion_multiply)(
        vmap(quaternion_inverse)(q_tp1),
        predicted_q
    )
    
    # Compute 2log of error quaternions
    log_errors = 2.0 * vmap(quaternion_log)(error_q)
    
    squared_errors = jnp.sum(jnp.square(log_errors), axis=1)
    return 0.5 * jnp.sum(squared_errors) 

@jit
def compute_observation_errors(quaternions, accelerations):
    """Observation error computation"""
    g = 1.0  
    
    def rotate_vector(q, v):
        """q ○ [0,v] ○ q^(-1)"""
        qv = jnp.array([0., v[0], v[1], v[2]])
        q_inv = quaternion_inverse(q)
        rotated = quaternion_multiply(quaternion_multiply(q, qv), q_inv)
        return rotated[1:]
    
    def h(q):
        """h(qt) = qt^(-1) ○ [0,0,0,g] ○ qt"""
        
        return rotate_vector(quaternion_inverse(q), jnp.array([0., 0., g]))
    
    def compute_single_error(q, measured_acc):
        predicted_acc = h(q)
        error = measured_acc - predicted_acc
        return jnp.sum(jnp.square(error))
    
    errors = vmap(compute_single_error)(quaternions, accelerations.T)
    return 0.5 * jnp.sum(errors)

@jit
def cost_fn(quaternions, angular_velocities, accelerations, dt):
    """Combined cost function following the assignment formula"""
    motion_error = compute_motion_errors(quaternions, angular_velocities, dt)
    obs_error = compute_observation_errors(quaternions, accelerations)
    
    total_cost = motion_error + obs_error
    return total_cost

@jit
def project_quaternions(quaternions):
    """Project quaternions to unit norm (equation 5 from homework)
    """
    return vmap(safe_normalize)(quaternions)

@jit
def optimization_step(quaternions, angular_velocities, accelerations, dt, i, learning_rate, prev_cost, best_quaternions, best_cost):
    
   
    grads = grad(lambda q: cost_fn(q, angular_velocities, accelerations, dt))(quaternions)
    

    quaternions = project_quaternions(quaternions - learning_rate * grads)
                                                      
   
    current_cost = cost_fn(quaternions, angular_velocities, accelerations, dt)
    
    
    best_quaternions = jnp.where(current_cost < best_cost,
                               quaternions,
                               best_quaternions)
    best_cost = jnp.minimum(current_cost, best_cost)
    
    return quaternions, current_cost, best_quaternions, best_cost

def optimize_quaternions(initial_quaternions, angular_velocities, accelerations, timestamps,
                        learning_rate=0.007,
                        num_iterations=200,
                        tolerance=1e-6):
    
    quaternions = safe_normalize(jnp.array(initial_quaternions, dtype=jnp.float32))
    dt = jnp.diff(timestamps).astype(jnp.float32)
    angular_velocities = jnp.array(angular_velocities, dtype=jnp.float32)
    accelerations = jnp.array(accelerations, dtype=jnp.float32)
    learning_rate = jnp.array(learning_rate, dtype=jnp.float32)
    
   
    best_quaternions = quaternions
    best_cost = jnp.inf
    prev_cost = jnp.inf
    costs = []
    
    print("Starting optimization...")
    
    try:
        initial_cost = cost_fn(quaternions, angular_velocities, accelerations, dt)
        print(f"Initial cost: {float(initial_cost)}")
        
        for i in range(num_iterations):
            quaternions, current_cost, best_quaternions, best_cost = optimization_step(
                quaternions, angular_velocities, accelerations, dt,
                jnp.array(i, dtype=jnp.float32), learning_rate,
                prev_cost, best_quaternions, best_cost
            )
            
            costs.append(float(current_cost))
            
            if i % 10 == 0:
                print(f"Iteration {i}, Cost: {float(current_cost)}")
            
           
            if jnp.abs(prev_cost - current_cost) < tolerance and i > 100:
                print(f"Converged at iteration {i}")
                break
                
            prev_cost = current_cost
            
    except Exception as e:
        print(f"Optimization error: {str(e)}")
        return best_quaternions
    
   
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.title('Cost Function for Dataset1')
    plt.xlabel('Iteration')
    plt.ylabel('Cost c(q1:T)')
    plt.grid(True)
    plt.show()
    
    return best_quaternions

def evaluate_optimization(optimized_quaternions, vicon_euler, timestamps):
    """Evaluate and visualize optimization results"""
    optimized_euler = quaternions_to_euler(optimized_quaternions)
    min_length = min(len(timestamps), len(vicon_euler))
    
    timestamps = timestamps[:min_length]
    optimized_euler = optimized_euler[:min_length]
    vicon_euler = vicon_euler[:min_length]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot Roll
    ax1.plot(timestamps, np.rad2deg(optimized_euler[:, 0]), 'g-', label='Optimized')
    ax1.plot(timestamps[:len(vicon_euler)], np.rad2deg(vicon_euler[:, 0]), 'r--', label='VICON')
    ax1.set_ylabel('Roll (degrees)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Pitch
    ax2.plot(timestamps, np.rad2deg(optimized_euler[:, 1]), 'g-', label='Optimized')
    ax2.plot(timestamps[:len(vicon_euler)], np.rad2deg(vicon_euler[:, 1]), 'r--', label='VICON')
    ax2.set_ylabel('Pitch (degrees)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot Yaw
    ax3.plot(timestamps, np.rad2deg(optimized_euler[:, 2]), 'g-', label='Optimized')
    ax3.plot(timestamps[:len(vicon_euler)], np.rad2deg(vicon_euler[:, 2]), 'r--', label='VICON')
    ax3.set_ylabel('Yaw (degrees)')
    ax3.set_xlabel('Time (s)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load and process data
    filepath = '/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR1/data/imu/imuRaw3.p'
    timestamps, accel_calibrated, gyro_calibrated = process_dataset(filepath)
    
    vicon_euler, vicon_timestamps = read_vicon_data("/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR1/data/vicon/viconRot3.p")
    
    # Get initial quaternions from integration
    initial_quaternions = integrate_orientation(gyro_calibrated, timestamps)                                       
    
    # Optimize quaternions
    optimized_quaternions = optimize_quaternions(
        initial_quaternions,
        gyro_calibrated,
        accel_calibrated,
        timestamps, 
        learning_rate=0.005,
        num_iterations=100,            
        tolerance=1e-6
    )
    
    # Evaluate results         
    evaluate_optimization(optimized_quaternions, vicon_euler, timestamps)

         



                              
    

    
    
        
        
        
        