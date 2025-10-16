"""
Record a LIBERO (or custom) demonstration episode with a Vision-Language-Action model.

This script:
  1. Sets up a simulator environment from a BDDL file.
  2. Runs inference with a model (e.g., OpenVLA) at each step.
  3. Executes predicted actions in the environment.
  4. Saves the resulting states and actions to an HDF5 file.

Usage:
    python record.py \
        --model openvla \
        --bddl_file /path/to/custom_scene.bddl \
        --out_file /path/to/output/demo.hdf5 \
        --prompt "move the phone to the left of the table" \
        --task_suite_name libero_spatial \
        --cache_dir /path/to/cache

Note: This implementation follows the OpenVLA LIBERO evaluation script from:
    https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py
"""

import os
import h5py
import argparse
import numpy as np
from PIL import Image
import cv2

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from libero.libero.envs import OffScreenRenderEnv


def normalize_gripper_action(action, binarize=True):
    """
    Normalize gripper action from [0,1] to [-1,+1] because LIBERO expects the latter.
    
    Args:
        action: Action array where the last dimension is the gripper action
        binarize: If True, binarize the gripper action to -1 or +1
    """
    # Normalize gripper action to [-1, +1]
    action[-1] = 2.0 * action[-1] - 1.0
    if binarize:
        action[-1] = 1.0 if action[-1] > 0 else -1.0
    return action


def invert_gripper_action(action):
    """
    Invert gripper action sign.
    OpenVLA's dataloader flips the sign to align with other datasets (0=close, 1=open),
    so we flip it back (-1=open, +1=close) before executing in LIBERO.
    """
    action[-1] = -action[-1]
    return action


def load_openvla(device="cuda:0", cache_dir=None):
    """Load OpenVLA model and processor."""
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True
    )
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    return processor, vla


def record_demo(bddl_file, out_file, model_flag="openvla", prompt=None, task_suite_name="libero_spatial", 
                device="cuda:0", cache_dir=None):
    """
    Record a single demo using a VLA model and save it to HDF5.
    
    Args:
        bddl_file: Path to BDDL file
        out_file: Path to save HDF5 demo
        model_flag: Model to use (only 'openvla' supported)
        prompt: Task instruction
        task_suite_name: LIBERO task suite name (used for action unnormalization)
        device: Device to run model on
        cache_dir: Cache directory for model weights
    """
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 256,  # OpenVLA uses 256x256, not 512x512
        "camera_widths": 256,
    }

    # Initialize environment
    print("Setting up environment...")
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    obs = env.reset()

    # Load model
    print(f"Loading model: {model_flag}...")
    if model_flag == "openvla":
        processor, vla = load_openvla(device=device, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unsupported model flag: {model_flag}")

    actions, dones, rewards, states, obs_list = [], [], [], [], []
    step = 0
    done = False

    while not done and step < 200:
        # Extract RGB observation for model inference
        img = Image.fromarray(obs["agentview_image"].astype(np.uint8))
        
        # Format prompt following OpenVLA's convention
        q = f"In: What action should the robot take to {prompt}?\nOut:"
        inputs = processor(q, img).to(device, dtype=torch.bfloat16)
        
        # CRITICAL: Use task_suite_name as unnorm_key, NOT "bridge_orig"!
        # OpenVLA uses different action statistics for each LIBERO task suite.
        action = vla.predict_action(**inputs, unnorm_key=task_suite_name, do_sample=False)
        
        # Process gripper action following OpenVLA's convention
        # 1. Normalize gripper action [0,1] -> [-1,+1]
        action = normalize_gripper_action(action, binarize=True)
        # 2. Invert gripper action sign (OpenVLA flips it during training)
        action = invert_gripper_action(action)

        # Step in environment (no manual scaling needed!)
        obs, reward, done, info = env.step(action.tolist())

        # Flatten obs into a single vector for 'states'
        flat_state = np.concatenate([
            np.ravel(obs[k]) for k in sorted(obs.keys()) if not k.endswith("image")
        ])

        # Save per-step info
        actions.append(action)
        dones.append(done)
        rewards.append(reward)
        states.append(flat_state)
        obs_list.append({k: np.array(v) for k, v in obs.items() if not k.endswith("image")})

        step += 1
        print(f"Step {step}, Action: {action}, Reward: {reward}, Done: {done}")

    env.close()

    # Convert lists to arrays
    actions = np.array(actions)
    dones = np.array(dones, dtype=bool)
    rewards = np.array(rewards)
    states = np.array(states, dtype=np.float32)

    # Save HDF5 in full LIBERO structure
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with h5py.File(out_file, "w") as f:
        dset = f.create_group("data/demo_0")
        dset.create_dataset("actions", data=actions, compression="gzip")
        dset.create_dataset("dones", data=dones, compression="gzip")
        dset.create_dataset("rewards", data=rewards, compression="gzip")
        dset.create_dataset("states", data=states, compression="gzip")

        # Save obs as a separate group
        obs_grp = dset.create_group("obs")
        for k in obs_list[0].keys():
            # Stack all steps along axis=0
            obs_stack = np.stack([step_obs[k] for step_obs in obs_list], axis=0)
            obs_grp.create_dataset(k, data=obs_stack, compression="gzip")

    print(f"Saved demo to {out_file} with {len(actions)} steps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openvla", help="Model flag (currently only 'openvla')")
    parser.add_argument("--bddl_file", type=str, required=True, help="Path to BDDL scene file")
    parser.add_argument("--out_file", type=str, required=True, help="Path to save HDF5 demo")
    parser.add_argument("--prompt", type=str, required=True, help="Task instruction for the model")
    parser.add_argument(
        "--task_suite_name", 
        type=str, 
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite name (used for action unnormalization)"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory for model weights")
    args = parser.parse_args()

    print(f"Recording demo with:")
    print(f"  Model: {args.model}")
    print(f"  Task suite: {args.task_suite_name}")
    print(f"  BDDL file: {args.bddl_file}")
    print(f"  Output file: {args.out_file}")
    print(f"  Prompt: {args.prompt}")
    
    record_demo(
        bddl_file=args.bddl_file,
        out_file=args.out_file,
        model_flag=args.model,
        prompt=args.prompt,
        task_suite_name=args.task_suite_name,
        device=args.device,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()