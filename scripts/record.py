"""
Record OpenVLA demonstrations on LIBERO tasks with temporal perturbations.

Extends record.py to support time-windowed mid-rollout perturbations:
  - move    : Teleport object to new XY position for a step window
  - color   : Change object color for a step window
  - distractor: Spawn a hidden object into the scene for a step window
  - replace : Swap one object for another for a step window

All perturbations automatically REVERT when their window closes.

Usage:
    python record.py --config ../configs/config.yaml

Temporal perturbation config (add to your inference_config.yaml):

    # Pre-declare hidden objects needed for distractor/replace perturbations.
    # These are added to the BDDL so MuJoCo loads them, but spawned off-screen.
    hidden_objects:
      - name: moka_pot_999
        type: moka_pot
      - name: milk_777
        type: milk

    temporal_perturbations:
      - type: move
        obj_name: akita_black_bowl_1
        start_step: 50
        end_step: 150
        delta_xy: [0.06, 0.0]     # explicit shift (m); omit for random
        max_move_m: 0.05          # max random shift if delta_xy omitted

      - type: color
        obj_name: akita_black_bowl_1
        start_step: 200
        end_step: 300
        color: red                # omit for random color

      - type: distractor
        distractor_obj_name: moka_pot_999
        start_step: 100
        end_step: 250
        distractor_xy: [0.1, -0.1]  # omit for random placement

      - type: replace
        obj_name: wine_bottle_1
        replacement_obj_name: milk_777
        start_step: 150
        end_step: 350

Notes:
  - distractor and replace objects MUST be listed in hidden_objects so they
    are added to the BDDL and pre-loaded into the sim off-screen.
  - Perturbation windows can overlap freely.
  - If a window is still open at episode end, it is auto-reverted via flush().
  - Perturbation events are recorded in the HDF5 output under
    data/demo_N/perturbation_events (JSON string).
"""

import os
import h5py
import json
import argparse
import yaml
import tempfile
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from libero.libero.envs import OffScreenRenderEnv

from libero.libero.utils.temporal_perturbations import (
    TemporalPerturbationSpec,
    TemporalPerturbationManager,
    add_hidden_objects_to_bddl,
    specs_from_config,
)
from libero.libero.utils.generate_perturbation_bddl import read_bddl, validate_bddl, extract_target_workspace


# ---------------------------------------------------------------------------
# Image preprocessing (unchanged from original record.py)
# ---------------------------------------------------------------------------

def preprocess_image(obs, resize_size=256, center_crop=True):
    """Preprocess image with LIBERO-specific 180-degree rotation."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]
    img = Image.fromarray(img.astype(np.uint8))
    img = img.resize((resize_size, resize_size), Image.LANCZOS)
    if center_crop:
        crop_size = 224
        left = (resize_size - crop_size) // 2
        top = (resize_size - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))
    return img


def normalize_gripper_action(action, binarize=True):
    action[-1] = 2.0 * action[-1] - 1.0
    if binarize:
        action[-1] = 1.0 if action[-1] > 0 else -1.0
    return action


def invert_gripper_action(action):
    action[-1] = -action[-1]
    return action


# ---------------------------------------------------------------------------
# Model loading (unchanged)
# ---------------------------------------------------------------------------

def load_openvla(task_suite_name, device, cache_dir):
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
        trust_remote_code=True,
    ).to(device)
    if not hasattr(vla, "norm_stats"):
        import glob
        if cache_dir:
            pattern = os.path.join(
                cache_dir,
                f"models--{model_path.replace('/', '--')}",
                "snapshots", "*", "dataset_statistics.json",
            )
            matches = glob.glob(pattern)
            if matches:
                with open(matches[0], "r") as f:
                    vla.norm_stats = json.load(f)
                print("Loaded norm_stats from cache")
    return processor, vla


def add_gaussian_noise(action, noise_std):
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=action.shape)
        action[:6] = action[:6] + noise[:6]
    return action


# ---------------------------------------------------------------------------
# BDDL preparation for temporal perturbations
# ---------------------------------------------------------------------------

def prepare_bddl_with_hidden_objects(
    base_bddl_path: str,
    hidden_objects: List[Dict[str, str]],
    target_workspace: Optional[str] = None,
) -> str:
    """
    Read the base BDDL, inject hidden objects for distractor/replace perturbations,
    write to a temp file, and return the temp file path.

    Args:
        base_bddl_path : Path to original BDDL file.
        hidden_objects : List of dicts with 'name' and 'type' keys.
        target_workspace: Workspace name (auto-detected from BDDL if None).

    Returns:
        Path to the modified temporary BDDL file.
    """
    if not hidden_objects:
        return base_bddl_path

    bddl_text = read_bddl(base_bddl_path)

    # Auto-detect workspace if not provided
    if target_workspace is None:
        target_workspace = extract_target_workspace(bddl_text)

    hidden_pairs = [(obj["name"], obj["type"]) for obj in hidden_objects]
    bddl_text = add_hidden_objects_to_bddl(bddl_text, hidden_pairs, target_workspace)

    if not validate_bddl(bddl_text):
        raise RuntimeError("Modified BDDL (with hidden objects) failed validation.")

    # Write to a temp file (env needs a file path, not a string)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".bddl", delete=False, prefix="temporal_"
    )
    tmp.write(bddl_text)
    tmp.close()
    print(f"[TEMPORAL] Wrote modified BDDL (with {len(hidden_objects)} hidden object(s)) → {tmp.name}")
    return tmp.name


# ---------------------------------------------------------------------------
# Single demo recording with temporal perturbations
# ---------------------------------------------------------------------------

def record_single_demo(
    env,
    processor,
    vla,
    config: Dict[str, Any],
    demo_index: int,
    seed: int,
    temporal_manager: Optional[TemporalPerturbationManager] = None,
) -> Dict[str, Any]:
    """
    Record a single demonstration, applying temporal perturbations if provided.

    Returns a dict with keys:
        actions, dones, rewards, states, obs_list, perturbation_events
    """
    np.random.seed(seed)
    # seed() must come BEFORE reset() — ControlEnv.reset() calls self.env.reset()
    # internally so seeding after would have no effect on initialization.
    env.seed(seed)
    obs = env.reset()

    # Initialize temporal perturbation manager AFTER reset() so the sim is
    # fully constructed, but BEFORE stabilization steps so hidden objects are
    # parked before the physics settle (avoids hidden objects being "thrown"
    # into scene by the initial velocity bleed-in from their off-screen spawn).
    if temporal_manager is not None:
        temporal_manager.reset(env)

    # Stabilization steps — run with gripper closed, no movement.
    # Hidden objects are already parked at (100,100,100) by manager.reset().
    for _ in range(10):
        obs, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

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

    actions, dones, rewards, states, obs_list = [], [], [], [], []
    # Track which steps had active perturbations for analysis
    perturbation_events: List[Dict[str, Any]] = []

    step = 0
    done = False
    print(f"  Starting policy rollout (max {max_steps} steps, noise_std={noise_std})...")

    while not done and step < max_steps:

        # ---- Apply / revert temporal perturbations ----
        if temporal_manager is not None:
            prev_active = dict(temporal_manager._active)
            temporal_manager.step(env, step)
            # Log any state changes for HDF5 metadata
            for i, spec in enumerate(temporal_manager.specs):
                was_active = prev_active.get(i, False)
                now_active = temporal_manager._active.get(i, False)
                if not was_active and now_active:
                    perturbation_events.append({
                        "step": step, "event": "start",
                        "type": spec.pert_type,
                        "obj": spec.obj_name or spec.distractor_obj_name,
                    })
                elif was_active and not now_active:
                    perturbation_events.append({
                        "step": step, "event": "revert",
                        "type": spec.pert_type,
                        "obj": spec.obj_name or spec.distractor_obj_name,
                    })

        # ---- Policy inference ----
        img = preprocess_image(obs, resize_size=256, center_crop=True)
        img.save(f"/home/hice1/apalasamudram6/scratch/vla-explainability/scripts/record_last_step.png")
        prompt = f"In: What action should the robot take to {config['prompt']}?\nOut:"
        inputs = processor(prompt, img).to(config.get("device", "cuda:0"), dtype=torch.bfloat16)
        action = vla.predict_action(**inputs, unnorm_key=config["task_suite_name"], do_sample=False)

        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        if action_scale != 1.0:
            action[:6] = action[:6] * action_scale
        action = add_gaussian_noise(action, noise_std)

        obs_new, reward, done, info = env.step(action.tolist())
        obs = obs_new

        flat_state = np.concatenate([
            np.ravel(obs[k]) for k in sorted(obs.keys()) if not k.endswith("image")
        ])
        actions.append(action)
        dones.append(done)
        rewards.append(reward)
        states.append(flat_state)
        obs_list.append({k: np.array(v) for k, v in obs.items() if not k.endswith("image")})

        print(f"  Step {step}/{max_steps}, Reward: {reward:.2f}, Done: {done}, Action: {action}")

        step += 1
        if step % 50 == 0 or done:
            active_perts = [
                temporal_manager.specs[i].pert_type
                for i, a in temporal_manager._active.items() if a
            ] if temporal_manager else []
            pert_str = f" | active perturbations: {active_perts}" if active_perts else ""
            print(f"  Step {step}/{max_steps}, Reward: {reward:.2f}, Done: {done}{pert_str}")

    # Revert any perturbations still active at episode end
    if temporal_manager is not None:
        temporal_manager.flush(env)
        temporal_manager.summary()

    print(f"  ✓ Demo {demo_index} completed: {len(actions)} steps, "
          f"final reward: {reward:.2f}, "
          f"{len(perturbation_events)} perturbation event(s)")

    return {
        "actions": np.array(actions),
        "dones": np.array(dones, dtype=bool),
        "rewards": np.array(rewards),
        "states": np.array(states, dtype=np.float32),
        "obs_list": obs_list,
        "perturbation_events": perturbation_events,
    }


# ---------------------------------------------------------------------------
# Main recording function
# ---------------------------------------------------------------------------

def record_demo(config: Dict[str, Any]):
    """Record demonstration(s) using OpenVLA, with optional temporal perturbations."""

    # ---- Prepare BDDL (inject hidden objects if needed) ----
    base_bddl = config["bddl_file"]
    hidden_objects = config.get("hidden_objects", [])
    bddl_file = prepare_bddl_with_hidden_objects(
        base_bddl,
        hidden_objects,
        target_workspace=config.get("target_workspace"),
    )

    # ---- Build temporal perturbation specs ----
    temporal_specs_config = config.get("temporal_perturbations", [])
    temporal_manager: Optional[TemporalPerturbationManager] = None
    if temporal_specs_config:
        specs = specs_from_config(temporal_specs_config)
        temporal_manager = TemporalPerturbationManager(specs)
        print(f"\n[TEMPORAL] {len(specs)} temporal perturbation spec(s) loaded:")
        for i, s in enumerate(specs):
            print(f"  [{i}] {s.pert_type} | steps {s.start_step}–{s.end_step} "
                  f"| obj={s.obj_name or s.distractor_obj_name}")
    else:
        print("\n[TEMPORAL] No temporal perturbations configured.")

    # ---- Initialize environment ----
    # OffScreenRenderEnv → ControlEnv → self.env (robosuite task env)
    # Sim access: env.sim  (ControlEnv property) → env.env.sim  (robosuite MjSim)
    # Note: ControlEnv.reset() has `finally: continue` which causes it to
    # always iterate the while loop once more after success — this is a
    # pre-existing LIBERO bug and does not affect correctness, only adds one
    # extra reset() call per demo. Our manager.reset(env) is called after
    # env.reset() returns so we always get a fully initialized sim.
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    print("\nInitializing environment...")
    env = OffScreenRenderEnv(**env_args)

    # ---- Load model ----
    print("Loading model...")
    processor, vla = load_openvla(
        config["task_suite_name"],
        config.get("device", "cuda:0"),
        config["cache_dir"],
    )

    # ---- Record all demos ----
    num_demos = config.get("num_demos", 1)
    print(f"\nRecording {num_demos} demonstration(s)...")
    all_demos = []

    for demo_idx in range(num_demos):
        print(f"\n{'=' * 60}")
        print(f"Recording demo {demo_idx + 1}/{num_demos}")
        print(f"{'=' * 60}")
        demo_data = record_single_demo(
            env, processor, vla, config,
            demo_index=demo_idx,
            seed=demo_idx,
            temporal_manager=temporal_manager,
        )
        all_demos.append(demo_data)

    env.close()

    # ---- Clean up temp BDDL ----
    if bddl_file != config["bddl_file"] and os.path.exists(bddl_file):
        os.unlink(bddl_file)
        print(f"[TEMPORAL] Cleaned up temp BDDL: {bddl_file}")

    # ---- Save HDF5 ----
    out_file = config["out_file"]
    print(f"\nSaving {num_demos} demo(s) to {out_file}...")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with h5py.File(out_file, "w") as f:
        for demo_idx, demo_data in enumerate(all_demos):
            dset = f.create_group(f"data/demo_{demo_idx}")
            dset.create_dataset("actions", data=demo_data["actions"], compression="gzip")
            dset.create_dataset("dones", data=demo_data["dones"], compression="gzip")
            dset.create_dataset("rewards", data=demo_data["rewards"], compression="gzip")
            dset.create_dataset("states", data=demo_data["states"], compression="gzip")

            obs_grp = dset.create_group("obs")
            for k in demo_data["obs_list"][0].keys():
                obs_stack = np.stack(
                    [step_obs[k] for step_obs in demo_data["obs_list"]], axis=0
                )
                obs_grp.create_dataset(k, data=obs_stack, compression="gzip")

            # Save perturbation event log as JSON string attribute
            dset.attrs["perturbation_events"] = json.dumps(
                demo_data["perturbation_events"]
            )

    print(f"✓ Saved {num_demos} demo(s) to {out_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Record OpenVLA demonstrations with temporal perturbations"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to inference_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("OpenVLA LIBERO Recording (with Temporal Perturbations)")
    print("=" * 80)
    print(f"Task suite : {config['task_suite_name']}")
    print(f"BDDL file  : {config['bddl_file']}")
    print(f"Prompt     : {config['prompt']}")
    print(f"Output     : {config['out_file']}")
    n_tp = len(config.get("temporal_perturbations", []))
    n_ho = len(config.get("hidden_objects", []))
    print(f"Temporal perturbations: {n_tp} | Hidden objects: {n_ho}")
    print("=" * 80)

    record_demo(config)
    print("\n✓ Recording complete!")


if __name__ == "__main__":
    main()