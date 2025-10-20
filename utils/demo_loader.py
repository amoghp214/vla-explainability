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


def get_num_demos(demo_file: str) -> int:
    """
    Get the number of demonstrations in an HDF5 file.
    
    Args:
        demo_file: Path to the HDF5 demonstration file
    
    Returns:
        int: Number of demos in the file
    
    Example:
        >>> num_demos = get_num_demos("demos/multi_demo.hdf5")
        >>> print(f"File contains {num_demos} demonstrations")
        File contains 5 demonstrations
    """
    with h5py.File(demo_file, "r") as f:
        demos = [key for key in f["data"].keys() if key.startswith("demo_")]
        return len(demos)


def load_all_robot_states(demo_file: str) -> list:
    """
    Load robot states from all demonstrations in an HDF5 file.
    
    Args:
        demo_file: Path to the HDF5 demonstration file
    
    Returns:
        list: List of numpy arrays, one for each demo. Each array has shape (num_frames, 8)
    
    Example:
        >>> all_states = load_all_robot_states("demos/multi_demo.hdf5")
        >>> print(f"Loaded {len(all_states)} demos")
        >>> for i, states in enumerate(all_states):
        ...     print(f"Demo {i}: {states.shape[0]} frames")
    """
    num_demos = get_num_demos(demo_file)
    all_states = []
    
    for demo_idx in range(num_demos):
        states = load_robot_state_from_demo(demo_file, demo_idx)
        all_states.append(states)
    
    return all_states


def save_robot_state_from_demo(
    demo_file: str,
    output_file: str,
    demo_index: int = 0,
    verbose: bool = True
) -> str:
    """
    Load robot state from an HDF5 demo and save it as a numpy .npy file.
    
    This is a convenience function that combines loading and saving in one step.
    The saved .npy file can be quickly loaded later using np.load().
    
    Args:
        demo_file: Path to the HDF5 demonstration file
        output_file: Path to save the numpy array (will add .npy extension if missing)
        demo_index: Index of the demo to load (default: 0)
        verbose: Whether to print confirmation message (default: True)
    
    Returns:
        str: The actual path where the file was saved (with .npy extension)
    
    Example:
        >>> save_robot_state_from_demo(
        ...     "demos/task_demo.hdf5",
        ...     "processed/task_states.npy"
        ... )
        Saved robot state (300, 8) to processed/task_states.npy
        'processed/task_states.npy'
        
        >>> # Later, load it quickly
        >>> states = np.load("processed/task_states.npy")
    """
    import os
    
    # Load the robot state array
    robot_state = load_robot_state_from_demo(demo_file, demo_index)
    
    # Ensure output file has .npy extension
    if not output_file.endswith('.npy'):
        output_file = output_file + '.npy'
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to .npy file
    np.save(output_file, robot_state)
    
    if verbose:
        print(f"Saved robot state {robot_state.shape} to {output_file}")
    
    return output_file


if __name__ == "__main__":
    # Example usage and command-line interface
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load and inspect LIBERO demonstration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Inspect a demo file
        python demo_loader.py demos/task_demo.hdf5
        
        # Inspect and save to .npy file
        python demo_loader.py demos/task_demo.hdf5 --output processed/task_states.npy
        
        # Or use the module directly
        python -m utils.demo_loader demos/task_demo.hdf5 --output processed/task_states.npy
        """
    )
    parser.add_argument("demo_file", type=str, help="Path to HDF5 demonstration file")
    parser.add_argument("--output", "-o", type=str, help="Optional: Save robot state to this .npy file path")
    parser.add_argument("--demo-index", type=int, default=0, help="Demo index to load (default: 0)")
    args = parser.parse_args()
    
    # Load and display demo information
    print("=" * 80)
    print("Demo Information")
    print("=" * 80)
    info = load_demo_info(args.demo_file, args.demo_index)
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
    robot_state = load_robot_state_from_demo(args.demo_file, args.demo_index)
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
    actions = load_actions_from_demo(args.demo_file, args.demo_index)
    print(f"\nActions shape: {actions.shape}")
    
    # Save to file if output path is provided
    if args.output:
        print("\n" + "=" * 80)
        saved_path = save_robot_state_from_demo(
            args.demo_file, 
            args.output, 
            args.demo_index,
            verbose=True
        )
        print(f"✓ Saved to: {saved_path}")

