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
        --prompt "move the phone to the left of the table"
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

LINEAR_SCALE_FACTOR = 10
ANGULAR_SCALE_FACTOR = 10


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


def record_demo(bddl_file, out_file, model_flag="openvla", prompt=None, device="cuda:0", cache_dir=None):
    """
    Record a single demo using a VLA model and save it to HDF5.
    """
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 512,
        "camera_widths": 512,
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
        q = f"In: What action should the robot take to {prompt}?\nOut:"
        inputs = processor(q, img).to(device, dtype=torch.bfloat16)
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        scaled_action = action.copy()
        scaled_action[:3] *= LINEAR_SCALE_FACTOR
        scaled_action[3:6] *= ANGULAR_SCALE_FACTOR

        # Step in environment
        obs, reward, done, info = env.step(scaled_action)

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
        print(f"Step {step}, Action: {scaled_action}, Done: {done}")

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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory for model weights")
    args = parser.parse_args()

    print(f"Recording demo with model={args.model}, bddl_file={args.bddl_file}, out_file={args.out_file}")
    record_demo(
        bddl_file=args.bddl_file,
        out_file=args.out_file,
        model_flag=args.model,
        prompt=args.prompt,
        device=args.device,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()