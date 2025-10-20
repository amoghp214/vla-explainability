"""
Load and extract robot state data from LIBERO demonstration HDF5 files.

This module provides utilities to read HDF5 demo files created by record.py
and extract robot state information in a convenient numpy array format.
"""

import h5py
import numpy as np
from typing import Optional


def load_robot_state_from_demo(
    demo_file: str,
    demo_index: int = 0
) -> np.ndarray:
    """
    Load robot state trajectory from an HDF5 demo file.
    
    Extracts end-effector position, orientation (quaternion), and gripper position
    for each timestep in the demonstration.
    
    Args:
        demo_file: Path to the HDF5 demonstration file (created by record.py)
        demo_index: Index of the demo to load (default: 0 for "demo_0")
    
    Returns:
        np.ndarray: Robot state trajectory with shape (num_frames, 8) where:
            - [:, 0:3]: End-effector position (x, y, z)
            - [:, 3:7]: End-effector orientation as quaternion (x, y, z, w)
            - [:, 7]: Gripper position (scalar)
    
    Example:
        >>> states = load_robot_state_from_demo("demos/task_demo.hdf5")
        >>> print(f"Trajectory length: {states.shape[0]} steps")
        >>> print(f"EEF position at step 0: {states[0, :3]}")
        >>> print(f"EEF quaternion at step 0: {states[0, 3:7]}")
        >>> print(f"Gripper position at step 0: {states[0, 7]}")
    """
    with h5py.File(demo_file, "r") as f:
        # Navigate to the demo group
        demo_key = f"data/demo_{demo_index}"
        if demo_key not in f:
            raise ValueError(
                f"Demo '{demo_key}' not found in {demo_file}. "
                f"Available demos: {list(f['data'].keys())}"
            )
        
        obs_group = f[f"{demo_key}/obs"]
        
        # Extract robot state components
        # End-effector position (3D)
        eef_pos = obs_group["robot0_eef_pos"][()]  # Shape: (num_frames, 3)
        
        # End-effector orientation as quaternion (4D)
        eef_quat = obs_group["robot0_eef_quat"][()]  # Shape: (num_frames, 4)
        
        # Gripper position (2D joint positions, we'll take the mean or first value)
        gripper_qpos = obs_group["robot0_gripper_qpos"][()]  # Shape: (num_frames, 2)
        
        # Take the mean of the two gripper joint positions as a single gripper state
        # (alternatively could use just gripper_qpos[:, 0])
        gripper_pos = np.mean(gripper_qpos, axis=1, keepdims=True)  # Shape: (num_frames, 1)
        
        # Concatenate all components into a single array
        robot_state = np.concatenate([eef_pos, eef_quat, gripper_pos], axis=1)
        
    return robot_state


def load_demo_info(demo_file: str, demo_index: int = 0) -> dict:
    """
    Load metadata and summary information from an HDF5 demo file.
    
    Args:
        demo_file: Path to the HDF5 demonstration file
        demo_index: Index of the demo to load (default: 0)
    
    Returns:
        dict: Dictionary containing:
            - num_frames: Number of timesteps in the demo
            - success: Whether the demo was successful (if available)
            - total_reward: Sum of all rewards
            - available_keys: List of all observation keys in the demo
    """
    with h5py.File(demo_file, "r") as f:
        demo_key = f"data/demo_{demo_index}"
        if demo_key not in f:
            raise ValueError(
                f"Demo '{demo_key}' not found in {demo_file}. "
                f"Available demos: {list(f['data'].keys())}"
            )
        
        demo_group = f[demo_key]
        
        # Get basic info
        num_frames = len(demo_group["actions"][()])
        rewards = demo_group["rewards"][()]
        dones = demo_group["dones"][()]
        
        # Get observation keys
        obs_keys = list(demo_group["obs"].keys()) if "obs" in demo_group else []
        
        info = {
            "num_frames": num_frames,
            "success": bool(dones[-1]) if len(dones) > 0 else False,
            "total_reward": float(np.sum(rewards)),
            "final_reward": float(rewards[-1]) if len(rewards) > 0 else 0.0,
            "available_keys": obs_keys,
        }
    
    return info


def load_actions_from_demo(demo_file: str, demo_index: int = 0) -> np.ndarray:
    """
    Load action trajectory from an HDF5 demo file.
    
    Args:
        demo_file: Path to the HDF5 demonstration file
        demo_index: Index of the demo to load (default: 0)
    
    Returns:
        np.ndarray: Action trajectory with shape (num_frames, 7) where:
            - [:, 0:6]: Robot joint or EEF delta actions
            - [:, 6]: Gripper action
    """
    with h5py.File(demo_file, "r") as f:
        demo_key = f"data/demo_{demo_index}"
        if demo_key not in f:
            raise ValueError(
                f"Demo '{demo_key}' not found in {demo_file}. "
                f"Available demos: {list(f['data'].keys())}"
            )
        
        actions = f[f"{demo_key}/actions"][()]
    
    return actions


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python demo_loader.py <path_to_demo.hdf5>")
        sys.exit(1)
    
    demo_file = sys.argv[1]
    
    # Load and display demo information
    print("=" * 80)
    print("Demo Information")
    print("=" * 80)
    info = load_demo_info(demo_file)
    for key, value in info.items():
        if key == "available_keys":
            print(f"{key}:")
            for obs_key in value:
                print(f"  - {obs_key}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "=" * 80)
    print("Robot State Trajectory")
    print("=" * 80)
    
    # Load robot state
    robot_state = load_robot_state_from_demo(demo_file)
    print(f"Shape: {robot_state.shape}")
    print(f"Number of frames: {robot_state.shape[0]}")
    print(f"\nFirst frame:")
    print(f"  EEF position: {robot_state[0, :3]}")
    print(f"  EEF quaternion: {robot_state[0, 3:7]}")
    print(f"  Gripper position: {robot_state[0, 7]}")
    print(f"\nLast frame:")
    print(f"  EEF position: {robot_state[-1, :3]}")
    print(f"  EEF quaternion: {robot_state[-1, 3:7]}")
    print(f"  Gripper position: {robot_state[-1, 7]}")
    
    # Load actions
    actions = load_actions_from_demo(demo_file)
    print(f"\nActions shape: {actions.shape}")

