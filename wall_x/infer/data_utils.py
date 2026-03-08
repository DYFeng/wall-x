import numpy as np

def euler_to_matrix_zyx_6d_nb(euler_angles):
    """Convert Euler angles (ZYX order) to 6D rotation representation"""
    if euler_angles is None:
        return None
    
    euler_angles = np.array(euler_angles)
    if euler_angles.ndim == 1:
        euler_angles = euler_angles[None, :]
    
    num_samples = euler_angles.shape[0]
    rotation_6d = np.zeros((num_samples, 6))
    
    for i in range(num_samples):
        roll, pitch, yaw = euler_angles[i]
        
        # Calculate rotation matrix
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # First row of rotation matrix
        rotation_6d[i, 0] = cos_yaw * cos_pitch
        rotation_6d[i, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
        
        # Second row of rotation matrix
        rotation_6d[i, 2] = sin_yaw * cos_pitch
        rotation_6d[i, 3] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
        
        # Third row of rotation matrix
        rotation_6d[i, 4] = -sin_pitch
        rotation_6d[i, 5] = cos_pitch * sin_roll
    
    return rotation_6d

def compose_state_and_delta_to_abs_rpy(delta, current_state):
    """Compose current state and delta to get absolute RPY angles"""
    if delta is None or current_state is None:
        return None
    
    delta = np.array(delta)
    current_state = np.array(current_state)
    
    if delta.ndim == 1:
        delta = delta[None, :]
    
    if current_state.ndim == 1:
        current_state = current_state[None, :]
    
    # Simply add delta to current state
    abs_rpy = current_state + delta
    
    # Normalize angles to [-pi, pi]
    abs_rpy = (abs_rpy + np.pi) % (2 * np.pi) - np.pi
    
    return abs_rpy

def so3_to_euler_zyx_batch_nb(so3_rotations):
    """Convert SO3 rotations to Euler angles (ZYX order)"""
    if so3_rotations is None:
        return None
    
    so3_rotations = np.array(so3_rotations)
    if so3_rotations.ndim == 1:
        so3_rotations = so3_rotations[None, :]
    
    num_samples = so3_rotations.shape[0]
    euler_angles = np.zeros((num_samples, 3))
    
    for i in range(num_samples):
        # Extract rotation matrix from 6D representation
        r11, r12, r21, r22, r31, r32 = so3_rotations[i]
        
        # Calculate Euler angles using ZYX order
        pitch = np.arcsin(-r31)
        
        if np.cos(pitch) != 0:
            roll = np.arctan2(r32, r31)
            yaw = np.arctan2(r21, r11)
        else:
            # Handle singularity
            roll = 0.0
            yaw = np.arctan2(-r12, r22)
        
        euler_angles[i] = [roll, pitch, yaw]
    
    return euler_angles
