"""
Record a LIBERO (or custom) demonstration episode with a Vision-Language-Action model.

This script:
  1. Sets up a simulator environment from a BDDL file.
  2. Automatically loads the LIBERO-finetuned OpenVLA model for your task suite
  3. Runs inference with the model at each step.
  4. Executes predicted actions in the environment.
  5. Saves the resulting states and actions to an HDF5 file.

The script automatically uses the correct finetuned model based on --task_suite_name:
  - libero_spatial  → openvla/openvla-7b-finetuned-libero-spatial
  - libero_object   → openvla/openvla-7b-finetuned-libero-object
  - libero_goal     → openvla/openvla-7b-finetuned-libero-goal
  - libero_10       → openvla/openvla-7b-finetuned-libero-10
  - libero_90       → openvla/openvla-7b (base model - no finetuned version available)

Usage:
    python record.py \
        --model openvla \
        --bddl_file /path/to/custom_scene.bddl \
        --out_file /path/to/output/demo.hdf5 \
        --prompt "put the frying pan on the stove" \
        --task_suite_name libero_90 \
        --cache_dir /path/to/cache

Note: This implementation follows the OpenVLA LIBERO evaluation script from:
    https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py
    
Finetuned models from: https://huggingface.co/openvla
"""

import os
import h5py
import argparse
import json
import numpy as np
from PIL import Image
import cv2

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
import torchvision.transforms as transforms

from libero.libero.envs import OffScreenRenderEnv

# Note: OpenVLA model components are loaded automatically via trust_remote_code=True
# The model repository on HuggingFace contains all necessary code files
# No need to manually import prismatic modules!


def preprocess_image(obs, resize_size=256, center_crop=True):
    """
    Preprocess image following OpenVLA's LIBERO convention.
    
    CRITICAL: Must match the preprocessing used during training!
    - Rotate 180 degrees (LIBERO-specific) - MOST IMPORTANT!
    - Resize with high-quality interpolation
    
    Args:
        obs: Observation dict from environment
        resize_size: Size to resize to (default 256 for LIBERO)
        center_crop: Whether to apply center crop (required for models trained with image aug)
    
    Returns:
        PIL Image ready for the processor
    """
    # Get the image from observations
    img = obs["agentview_image"]
    
    # CRITICAL: Rotate 180 degrees to match training preprocessing!
    # This is THE KEY FIX - without this, the model sees everything upside down!
    img = img[::-1, ::-1]
    
    # Convert to PIL Image
    img = Image.fromarray(img.astype(np.uint8))
    
    # Resize using Lanczos (high-quality) interpolation
    # PIL's LANCZOS is equivalent to lanczos3
    img = img.resize((resize_size, resize_size), Image.LANCZOS)
    
    # Apply center crop if needed (important for models trained with augmentations!)
    if center_crop:
        # Center crop to 224x224 (standard for vision models)
        crop_size = 224
        left = (resize_size - crop_size) // 2
        top = (resize_size - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        img = img.crop((left, top, right, bottom))
    
    return img


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


def load_openvla(task_suite_name="libero_spatial", device="cuda:0", cache_dir=None):
    """
    Load OpenVLA model and processor.
    Automatically selects the LIBERO-finetuned model based on task_suite_name.
    
    Note: libero_90 does not have a finetuned model available, so it uses the base model.
    """
    # Map task suite names to their finetuned model paths
    # Note: libero_90 is intentionally not included as no finetuned model exists for it
    LIBERO_MODELS = {
        "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
        "libero_object": "openvla/openvla-7b-finetuned-libero-object",
        "libero_goal": "openvla/openvla-7b-finetuned-libero-goal",
        "libero_10": "openvla/openvla-7b-finetuned-libero-10",
        # libero_90: No finetuned model available on Hugging Face as of now
    }
    
    # Get the appropriate model for this task suite
    model_path = LIBERO_MODELS.get(task_suite_name, "openvla/openvla-7b")
    
    print(f"Loading model: {model_path}")
    if model_path == "openvla/openvla-7b":
        print(f"⚠️  WARNING: No finetuned model available for '{task_suite_name}'")
        print(f"    Using base model: openvla/openvla-7b")
        print(f"    Performance will be SIGNIFICANTLY degraded!")
        if task_suite_name == "libero_90":
            print(f"    Note: libero_90 finetuned model does not exist on Hugging Face.")
            print(f"    Consider using libero_spatial or libero_10 models as alternatives.")
    
    # Load processor (trust_remote_code=True loads all necessary components from HF Hub)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Load model
    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # Move model to device
    vla = vla.to(device)
    
    # CRITICAL: Check if model already has norm_stats (loaded automatically with trust_remote_code)
    if hasattr(vla, 'norm_stats'):
        print(f"✅ Model already has norm_stats: {list(vla.norm_stats.keys())}")
    else:
        print(f"⚠️  WARNING: Model does NOT have norm_stats attribute!")
        print(f"    Attempting to load manually...")
        
        # Try to find dataset_statistics.json in HuggingFace cache
        # HF cache structure: <cache_dir>/models--<org>--<model>/snapshots/<hash>/
        import glob
        if cache_dir:
            pattern = os.path.join(cache_dir, f"models--{model_path.replace('/', '--')}", "snapshots", "*", "dataset_statistics.json")
            matches = glob.glob(pattern)
            if matches:
                dataset_statistics_path = matches[0]  # Use most recent
                print(f"Found dataset_statistics.json at: {dataset_statistics_path}")
                with open(dataset_statistics_path, "r") as f:
                    norm_stats = json.load(f)
                vla.norm_stats = norm_stats
                print(f"✅ Manually loaded norm_stats with keys: {list(norm_stats.keys())}")
            else:
                print(f"❌ Could not find dataset_statistics.json in cache!")
                print(f"    Searched: {pattern}")
        else:
            print(f"❌ No cache_dir provided, cannot load dataset statistics!")
        
    return processor, vla


def record_demo(bddl_file, out_file, model_flag="openvla", prompt=None, task_suite_name="libero_spatial", 
                device="cuda:0", cache_dir=None, debug=False, action_scale=1.0):
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
        debug: If True, saves debug images and prints verbose info
        action_scale: Scalar multiplier for actions (default 1.0). Increase if robot moves too slowly.
    """
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 256,  # OpenVLA uses 256x256, not 512x512
        "camera_widths": 256,
    }
    
    # Create debug directory if needed
    if debug:
        debug_dir = os.path.join(os.path.dirname(out_file), "debug_images")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug mode enabled. Saving images to: {debug_dir}")

    # Initialize environment
    print("Setting up environment...")
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    obs = env.reset()
    
    # DEBUG: Test if environment responds to actions
    if debug:
        print(f"\n[DEBUG] === TESTING ENVIRONMENT RESPONSIVENESS ===")
        print(f"[DEBUG] Initial EEF position: {obs.get('robot0_eef_pos', 'N/A')}")
        
        # Test with a simple action
        test_action = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # Small x movement
        print(f"[DEBUG] Testing with action: {test_action}")
        obs_test, _, _, _ = env.step(test_action)
        print(f"[DEBUG] EEF position after test action: {obs_test.get('robot0_eef_pos', 'N/A')}")
        
        # Check if position changed
        initial_eef = obs.get('robot0_eef_pos', np.array([0,0,0]))
        test_eef = obs_test.get('robot0_eef_pos', np.array([0,0,0]))
        test_change = np.linalg.norm(test_eef - initial_eef)
        print(f"[DEBUG] Test action change magnitude: {test_change:.8f}")
        
        if test_change < 1e-6:
            print(f"[DEBUG] ⚠️  CRITICAL: Environment not responding to actions!")
            print(f"[DEBUG] This suggests the action format or environment setup is wrong.")
        else:
            print(f"[DEBUG] ✅ Environment is responsive to actions.")
        
        # Reset to initial state
        obs = env.reset()
        print(f"[DEBUG] Reset EEF position: {obs.get('robot0_eef_pos', 'N/A')}")

    # Load model
    print(f"Loading model: {model_flag}...")
    if model_flag == "openvla":
        processor, vla = load_openvla(task_suite_name=task_suite_name, device=device, cache_dir=cache_dir)
        
        # DEBUG: Check available normalization statistics
        if debug:
            print(f"\n[DEBUG] === MODEL NORMALIZATION STATISTICS ===")
            print(f"[DEBUG] Available unnorm_keys: {list(vla.norm_stats.keys())}")
            print(f"[DEBUG] Requested unnorm_key: {task_suite_name}")
            if task_suite_name in vla.norm_stats:
                stats = vla.norm_stats[task_suite_name]
                print(f"[DEBUG] Action statistics for {task_suite_name}:")
                print(f"[DEBUG]   Stats structure: {stats.keys() if hasattr(stats, 'keys') else type(stats)}")
                if 'action' in stats:
                    print(f"[DEBUG]   Action mean: {stats['action']['mean']}")
                    print(f"[DEBUG]   Action std: {stats['action']['std']}")
                    print(f"[DEBUG]   Action max: {stats['action']['max']}")
                    print(f"[DEBUG]   Action min: {stats['action']['min']}")
            else:
                print(f"[DEBUG] ⚠️  WARNING: {task_suite_name} not found in norm_stats!")
                print(f"[DEBUG] This might cause incorrect action scaling.")
    else:
        raise ValueError(f"Unsupported model flag: {model_flag}")

    actions, dones, rewards, states, obs_list = [], [], [], [], []
    step = 0
    done = False
    prev_img_hash = None  # For debugging: track if image changes
    
    # IMPORTANT: Wait for objects to stabilize in simulation
    # The simulator drops objects at the start and we need to wait for them to fall
    num_steps_wait = 10
    print(f"Waiting {num_steps_wait} steps for objects to stabilize...")
    for _ in range(num_steps_wait):
        # Execute a "do nothing" action (all zeros except gripper stays open at -1)
        dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        obs, _, _, _ = env.step(dummy_action)
    print("Objects stabilized. Starting policy rollout...")
    
    # Set max steps based on task suite (from official OpenVLA evaluation script)
    max_steps_dict = {
        "libero_spatial": 220,  # longest training demo has 193 steps
        "libero_object": 280,   # longest training demo has 254 steps
        "libero_goal": 300,     # longest training demo has 270 steps
        "libero_10": 520,       # longest training demo has 505 steps
        "libero_90": 400,       # longest training demo has 373 steps
    }
    max_steps = max_steps_dict.get(task_suite_name, 200)
    print(f"Max steps for {task_suite_name}: {max_steps}")

    while not done and step < max_steps:
        # Extract RGB observation for model inference with proper preprocessing
        # IMPORTANT: center_crop=True for models trained with image augmentations!
        img = preprocess_image(obs, resize_size=256, center_crop=True)
        
        # Debug: Check if image is changing between steps
        if debug and step < 10:
            import hashlib
            img_bytes = np.array(img).tobytes()
            img_hash = hashlib.md5(img_bytes).hexdigest()[:8]
            if prev_img_hash is not None and img_hash == prev_img_hash:
                print(f"[DEBUG] ⚠️  WARNING: Image hasn't changed since last step!")
                print(f"[DEBUG]    This will cause identical actions!")
            prev_img_hash = img_hash
        
        # Debug: Save images to verify what the model sees
        if debug and step < 5:
            img.save(os.path.join(debug_dir, f"step_{step:03d}_input.png"))
            # Also save raw observation
            raw_img = Image.fromarray(obs["agentview_image"].astype(np.uint8))
            raw_img.save(os.path.join(debug_dir, f"step_{step:03d}_raw.png"))
        
        # Format prompt following OpenVLA's convention
        q = f"In: What action should the robot take to {prompt}?\nOut:"
        
        if debug and step == 0:
            print(f"\n[DEBUG] Prompt: {q}")
            print(f"[DEBUG] Image size after preprocessing: {img.size}")
        
        # DEBUG: Print comprehensive environment state BEFORE action
        if debug and step < 5:
            print(f"\n[DEBUG] === STEP {step} BEFORE ACTION ===")
            print(f"[DEBUG] Robot EEF position: {obs.get('robot0_eef_pos', 'N/A')}")
            print(f"[DEBUG] Robot EEF quaternion: {obs.get('robot0_eef_quat', 'N/A')}")
            print(f"[DEBUG] Robot joint positions: {obs.get('robot0_joint_pos', 'N/A')}")
            print(f"[DEBUG] Robot gripper qpos: {obs.get('robot0_gripper_qpos', 'N/A')}")
            print(f"[DEBUG] Available observation keys: {list(obs.keys())}")
            
            # Check if there are object positions
            object_keys = [k for k in obs.keys() if 'object' in k.lower() or 'moka' in k.lower()]
            if object_keys:
                print(f"[DEBUG] Object-related observations: {object_keys}")
                for key in object_keys[:3]:  # Show first 3 object keys
                    print(f"[DEBUG]   {key}: {obs[key]}")
        
        inputs = processor(q, img).to(device, dtype=torch.bfloat16)
        
        if debug and step == 0:
            print(f"[DEBUG] Input tensor shape: {inputs['pixel_values'].shape if 'pixel_values' in inputs else 'N/A'}")
            # Verify model has proper predict_action method and norm_stats
            print(f"[DEBUG] Model has predict_action: {hasattr(vla, 'predict_action')}")
            print(f"[DEBUG] Model has norm_stats: {hasattr(vla, 'norm_stats')}")
            if hasattr(vla, 'norm_stats') and task_suite_name in vla.norm_stats:
                action_stats = vla.norm_stats[task_suite_name].get('action', {})
                print(f"[DEBUG] Action mean (first 3): {action_stats.get('mean', [])[:3]}")
                print(f"[DEBUG] Action std (first 3): {action_stats.get('std', [])[:3]}")
                print(f"[DEBUG] Action max (first 3): {action_stats.get('max', [])[:3]}")
        
        # CRITICAL: Use task_suite_name as unnorm_key, NOT "bridge_orig"!
        # OpenVLA uses different action statistics for each LIBERO task suite.
        # IMPORTANT: Use do_sample=False for deterministic behavior (as in official eval)
        action = vla.predict_action(**inputs, unnorm_key=task_suite_name, do_sample=False)
        
        if debug and step < 5:
            print(f"[DEBUG] Raw action from model (after unnormalization): {action}")
            print(f"[DEBUG]   Position (xyz): [{action[0]:.6f}, {action[1]:.6f}, {action[2]:.6f}]")
            print(f"[DEBUG]   Rotation (rpy): [{action[3]:.6f}, {action[4]:.6f}, {action[5]:.6f}]")
            print(f"[DEBUG]   Gripper (raw): {action[6]:.6f}")
            print(f"[DEBUG]   Position magnitude: {np.linalg.norm(action[:3]):.6f}")
        
        # Process gripper action following OpenVLA's convention
        # 1. Normalize gripper action [0,1] -> [-1,+1]
        action = normalize_gripper_action(action, binarize=True)
        # 2. Invert gripper action sign (OpenVLA flips it during training)
        action = invert_gripper_action(action)

        if debug and step < 5:
            print(f"[DEBUG] Processed action (after gripper processing): {action}")
            print(f"[DEBUG]   Gripper (processed): {action[6]:.2f}")
            print(f"[DEBUG]   Action magnitude (excluding gripper): {np.linalg.norm(action[:6]):.6f}")
        
        # Apply action scaling if specified (only to position/rotation, NOT gripper)
        if action_scale != 1.0:
            action[:6] = action[:6] * action_scale
            if debug and step < 5:
                print(f"[DEBUG] Action after scaling by {action_scale}: {action}")
                print(f"[DEBUG]   Scaled action magnitude: {np.linalg.norm(action[:6]):.6f}")

        # Step in environment
        obs_new, reward, done, info = env.step(action.tolist())
        
        # DEBUG: Print environment state AFTER action
        if debug and step < 5:
            print(f"\n[DEBUG] === STEP {step} AFTER ACTION ===")
            print(f"[DEBUG] New Robot EEF position: {obs_new.get('robot0_eef_pos', 'N/A')}")
            print(f"[DEBUG] New Robot gripper qpos: {obs_new.get('robot0_gripper_qpos', 'N/A')}")
            print(f"[DEBUG] Reward: {reward}")
            print(f"[DEBUG] Done: {done}")
            print(f"[DEBUG] Info keys: {list(info.keys()) if info else 'N/A'}")
            
            # Check if EEF position actually changed
            old_eef = obs.get('robot0_eef_pos', np.array([0,0,0]))
            new_eef = obs_new.get('robot0_eef_pos', np.array([0,0,0]))
            eef_change = np.linalg.norm(new_eef - old_eef)
            print(f"[DEBUG] EEF position change magnitude: {eef_change:.8f}")
            
            if eef_change < 1e-6:
                print(f"[DEBUG] ⚠️  WARNING: EEF position barely changed! Environment may not be updating.")
        
        obs = obs_new

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
        # Print action breakdown for debugging
        if step <= 3 or step % 10 == 0:  # Print first 3 steps and every 10th step
            print(f"\nStep {step}:")
            print(f"  Position (xyz): [{action[0]:.6f}, {action[1]:.6f}, {action[2]:.6f}]")
            print(f"  Rotation (rpy): [{action[3]:.6f}, {action[4]:.6f}, {action[5]:.6f}]")
            print(f"  Gripper: {action[6]:.2f}")
            print(f"  Reward: {reward}, Done: {done}")
            if debug:
                print(f"  EEF Pos: {obs.get('robot0_eef_pos', 'N/A')}")
                print(f"  Action magnitude: {np.linalg.norm(action[:6]):.6f}")
        else:
            print(f"Step {step}, Done: {done}")

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
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (saves images, verbose logging)")
    parser.add_argument(
        "--action_scale",
        type=float,
        default=1.0,
        help="Scalar multiplier for actions (default 1.0). Use >1.0 if robot moves too slowly, <1.0 if too fast."
    )
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
        debug=args.debug,
        action_scale=args.action_scale,
    )


if __name__ == "__main__":
    main()