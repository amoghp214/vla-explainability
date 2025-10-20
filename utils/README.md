# VLA Explainability Utilities

This directory contains utility functions for working with LIBERO demonstration data and VLA models.

## Demo Loader (`demo_loader.py`)

Load and extract robot state data from HDF5 demonstration files created by `record.py`.

### Main Functions

#### `load_robot_state_from_demo(demo_file, demo_index=0)`

Extracts robot state trajectory as a numpy array of shape `(num_frames, 8)`:
- **Columns 0-2**: End-effector position (x, y, z)
- **Columns 3-6**: End-effector orientation as quaternion (x, y, z, w)
- **Column 7**: Gripper position (scalar)

**Example Usage:**

```python
from utils import load_robot_state_from_demo

# Load robot state trajectory
states = load_robot_state_from_demo("demos/libero_10/moka_pots_demo.hdf5")

print(f"Trajectory has {states.shape[0]} frames")
print(f"EEF position at step 0: {states[0, :3]}")
print(f"EEF quaternion at step 0: {states[0, 3:7]}")
print(f"Gripper at step 0: {states[0, 7]}")
```

#### `load_demo_info(demo_file, demo_index=0)`

Returns metadata about the demonstration:
- `num_frames`: Number of timesteps
- `success`: Whether the demo completed successfully
- `total_reward`: Sum of all rewards
- `final_reward`: Final reward value
- `available_keys`: List of all observation keys

**Example Usage:**

```python
from utils.demo_loader import load_demo_info

info = load_demo_info("demos/libero_10/moka_pots_demo.hdf5")
print(f"Demo completed: {info['success']}")
print(f"Total reward: {info['total_reward']}")
print(f"Available observations: {info['available_keys']}")
```

#### `load_actions_from_demo(demo_file, demo_index=0)`

Extracts the action trajectory from the demo.

**Example Usage:**

```python
from utils.demo_loader import load_actions_from_demo

actions = load_actions_from_demo("demos/libero_10/moka_pots_demo.hdf5")
print(f"Action sequence shape: {actions.shape}")  # (num_frames, 7)
```

### Command-Line Usage

You can also run the demo loader directly to inspect a demo file:

```bash
cd vla-explainability
python -m utils.demo_loader demos/libero_10/moka_pots_demo.hdf5
```

This will print:
- Demo metadata (number of frames, success status, rewards)
- All available observation keys
- Robot state trajectory information
- First and last frame robot states

### Integration Example

Here's a complete example of how to use this in your analysis pipeline:

```python
import numpy as np
from utils import load_robot_state_from_demo
from utils.demo_loader import load_demo_info, load_actions_from_demo

# Load demo data
demo_path = "demos/libero_10/moka_pots_demo.hdf5"

# Get demo information
info = load_demo_info(demo_path)
print(f"Analyzing demo with {info['num_frames']} frames")

# Load robot states (num_frames x 8)
states = load_robot_state_from_demo(demo_path)
eef_positions = states[:, :3]
eef_orientations = states[:, 3:7]
gripper_positions = states[:, 7]

# Load actions (num_frames x 7)
actions = load_actions_from_demo(demo_path)

# Your analysis code here
# e.g., compute trajectory statistics, visualize, etc.
print(f"Average EEF height: {np.mean(eef_positions[:, 2]):.3f}")
print(f"Gripper open ratio: {np.mean(gripper_positions > 0):.2%}")
```

## Data Format

The HDF5 files created by `record.py` have the following structure:

```
demo.hdf5
└── data
    └── demo_0
        ├── actions          # (num_frames, 7) - robot actions
        ├── dones           # (num_frames,) - episode termination flags
        ├── rewards         # (num_frames,) - reward values
        ├── states          # (num_frames, state_dim) - full state vectors
        └── obs             # Observation group
            ├── robot0_eef_pos          # (num_frames, 3)
            ├── robot0_eef_quat         # (num_frames, 4)
            ├── robot0_gripper_qpos     # (num_frames, 2)
            ├── robot0_joint_pos        # (num_frames, 7)
            └── ... (other observation keys)
```

