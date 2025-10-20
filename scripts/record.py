"""
Record OpenVLA demonstrations on LIBERO tasks.

This script runs OpenVLA inference on a LIBERO task and saves:
- HDF5 demonstration file with actions, observations, and rewards
- Supports recording multiple demos in a single HDF5 file
- Can add Gaussian noise to actions to simulate reality

Usage:
    python record.py --config ../configs/inference_config.yaml

Configuration:
    Edit your config YAML to specify:
    - Model and task suite
    - BDDL file and task prompt
    - Output path for demo HDF5 file
    - num_demos: Number of demonstrations to record (default: 1)
    - noise_std: Gaussian noise std deviation for actions (default: 0.0)
    - Optional action scaling

The script automatically:
    - Loads the appropriate finetuned OpenVLA model for your task suite
    - Preprocesses images for LIBERO
    - Handles action normalization and gripper control
    - Records multiple demos with different random seeds
    - Adds small Gaussian noise to simulate real-world variability

Note: To create videos from recorded demos, use playback.py with the same config file.
"""

import os
import h5py
import argparse
import json
import yaml
import numpy as np
from PIL import Image

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from libero.libero.envs import OffScreenRenderEnv


def preprocess_image(obs, resize_size=256, center_crop=True):
    """Preprocess image with LIBERO-specific 180-degree rotation."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # Critical: rotate 180 degrees for LIBERO
    img = Image.fromarray(img.astype(np.uint8))
    img = img.resize((resize_size, resize_size), Image.LANCZOS)
    
    if center_crop:
        crop_size = 224
        left = (resize_size - crop_size) // 2
        top = (resize_size - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))
    
    return img


def normalize_gripper_action(action, binarize=True):
    """Normalize gripper action from [0,1] to [-1,+1]."""
    action[-1] = 2.0 * action[-1] - 1.0
    if binarize:
        action[-1] = 1.0 if action[-1] > 0 else -1.0
    return action


def invert_gripper_action(action):
    """Invert gripper action sign for LIBERO."""
    action[-1] = -action[-1]
    return action


def load_openvla(task_suite_name, device, cache_dir):
    """Load OpenVLA model and processor with automatic model selection."""
    LIBERO_MODELS = {
        "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
        "libero_object": "openvla/openvla-7b-finetuned-libero-object",
        "libero_goal": "openvla/openvla-7b-finetuned-libero-goal",
        "libero_10": "openvla/openvla-7b-finetuned-libero-10",
    }
    
    model_path = LIBERO_MODELS.get(task_suite_name, "openvla/openvla-7b")
    print(f"Loading model: {model_path}")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    
    # Load norm_stats if not already present
    if not hasattr(vla, 'norm_stats'):
        import glob
        if cache_dir:
            pattern = os.path.join(cache_dir, f"models--{model_path.replace('/', '--')}", 
                                 "snapshots", "*", "dataset_statistics.json")
            matches = glob.glob(pattern)
            if matches:
                with open(matches[0], "r") as f:
                    vla.norm_stats = json.load(f)
                print(f"Loaded norm_stats from cache")
    
    return processor, vla


def add_gaussian_noise(action, noise_std):
    """Add Gaussian noise to action to simulate reality."""
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=action.shape)
        # Only add noise to the first 6 dimensions (position/orientation deltas)
        # Don't add noise to gripper action (last dimension)
        action[:6] = action[:6] + noise[:6]
    return action


def record_single_demo(env, processor, vla, config, demo_index, seed):
    """Record a single demonstration."""
    # Set random seed for this demo
    np.random.seed(seed)
    env.seed(seed)
    obs = env.reset()
    
    # Wait for objects to stabilize
    for _ in range(10):
        obs, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    
    # Set max steps based on task suite
    max_steps_dict = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_dict.get(config["task_suite_name"], 200)
    action_scale = config.get("action_scale", 1.0)
    noise_std = config.get("noise_std", 0.0)
    
    # Initialize recording
    actions, dones, rewards, states, obs_list = [], [], [], [], []
    step = 0
    done = False
    
    print(f"  Starting policy rollout (max {max_steps} steps, noise_std={noise_std})...")
    
    while not done and step < max_steps:
        # Preprocess image
        img = preprocess_image(obs, resize_size=256, center_crop=True)
        
        # Get action from model
        prompt = f"In: What action should the robot take to {config['prompt']}?\nOut:"
        inputs = processor(prompt, img).to(config.get("device", "cuda:0"), dtype=torch.bfloat16)
        action = vla.predict_action(**inputs, unnorm_key=config["task_suite_name"], do_sample=False)
        
        # Process gripper and apply scaling
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        if action_scale != 1.0:
            action[:6] = action[:6] * action_scale
        
        # Add Gaussian noise to simulate reality
        action = add_gaussian_noise(action, noise_std)
        
        # Execute action
        obs_new, reward, done, info = env.step(action.tolist())
        obs = obs_new
        
        # Save data
        flat_state = np.concatenate([
            np.ravel(obs[k]) for k in sorted(obs.keys()) if not k.endswith("image")
        ])
        actions.append(action)
        dones.append(done)
        rewards.append(reward)
        states.append(flat_state)
        obs_list.append({k: np.array(v) for k, v in obs.items() if not k.endswith("image")})
        
        step += 1
        if step % 10 == 0 or done:
            print(f"  Step {step}/{max_steps}, Reward: {reward:.2f}, Done: {done}")
    
    demo_data = {
        "actions": np.array(actions),
        "dones": np.array(dones, dtype=bool),
        "rewards": np.array(rewards),
        "states": np.array(states, dtype=np.float32),
        "obs_list": obs_list
    }
    
    print(f"  ✓ Demo {demo_index} completed with {len(actions)} steps, final reward: {reward:.2f}")
    
    return demo_data


def record_demo(config):
    """Record demonstration(s) using OpenVLA."""
    env_args = {
        "bddl_file_name": config["bddl_file"],
        "camera_heights": 256,
        "camera_widths": 256,
    }
    
    print("Initializing environment...")
    env = OffScreenRenderEnv(**env_args)
    
    print("Loading model...")
    processor, vla = load_openvla(
        config["task_suite_name"],
        config.get("device", "cuda:0"),
        config["cache_dir"]
    )
    
    # Get number of demos to record
    num_demos = config.get("num_demos", 1)
    print(f"\nRecording {num_demos} demonstration(s)...")
    
    # Record all demos
    all_demos = []
    for demo_idx in range(num_demos):
        print(f"\n{'='*60}")
        print(f"Recording demo {demo_idx + 1}/{num_demos}")
        print(f"{'='*60}")
        
        # Use different seed for each demo
        seed = demo_idx
        demo_data = record_single_demo(env, processor, vla, config, demo_idx, seed)
        all_demos.append(demo_data)
    
    env.close()
    
    # Save all demos to HDF5
    print(f"\nSaving {num_demos} demo(s) to {config['out_file']}...")
    os.makedirs(os.path.dirname(config['out_file']), exist_ok=True)
    
    with h5py.File(config['out_file'], "w") as f:
        for demo_idx, demo_data in enumerate(all_demos):
            dset = f.create_group(f"data/demo_{demo_idx}")
            dset.create_dataset("actions", data=demo_data["actions"], compression="gzip")
            dset.create_dataset("dones", data=demo_data["dones"], compression="gzip")
            dset.create_dataset("rewards", data=demo_data["rewards"], compression="gzip")
            dset.create_dataset("states", data=demo_data["states"], compression="gzip")
            
            obs_grp = dset.create_group("obs")
            for k in demo_data["obs_list"][0].keys():
                obs_stack = np.stack([step_obs[k] for step_obs in demo_data["obs_list"]], axis=0)
                obs_grp.create_dataset(k, data=obs_stack, compression="gzip")
    
    print(f"✓ Saved {num_demos} demo(s) to HDF5 file")


def main():
    parser = argparse.ArgumentParser(description="Record OpenVLA demonstrations on LIBERO tasks")
    parser.add_argument("--config", type=str, required=True, help="Path to inference_config.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("OpenVLA LIBERO Recording")
    print("=" * 80)
    print(f"Task suite: {config['task_suite_name']}")
    print(f"BDDL file: {config['bddl_file']}")
    print(f"Prompt: {config['prompt']}")
    print(f"Output: {config['out_file']}")
    print("=" * 80)
    
    record_demo(config)
    
    print("\n✓ Recording complete!")


if __name__ == "__main__":
    main()
