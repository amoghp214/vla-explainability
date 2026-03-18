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

Pre-episode collision Z corrections (z_overrides):
  - When the launcher detects that a perturbed object's BDDL init position
    would collide with another object or an open cavity (drawer/door), it
    writes a JSON sidecar (z_overrides_file in the config YAML).
  - After env.reset(), record.py loads this file, calls resolve_z_overrides()
    to compute accurate heights from real sim geometry, then writes the
    corrected poses before the stabilization steps run.  This means the
    objects start the episode at the correct height (stacked on top of a
    collider, or placed inside an open drawer) rather than clipping through.
"""

import os
import sys
import h5py
import json
import argparse
import glob
import yaml
import tempfile
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)

for _p in [_THIS_DIR, _REPO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_libero_env_path = os.environ.get("LIBERO_PATH")
if _libero_env_path and _libero_env_path not in sys.path:
    sys.path.insert(0, _libero_env_path)

try:
    from libero.libero.envs import OffScreenRenderEnv
except ModuleNotFoundError as _e:
    _tried = str(sys.path[:6])
    _msg = (
        f"Could not import libero. Tried sys.path: {_tried}\n"
        "Fix options:\n"
        "  1. Run from repo root:  cd <repo_root> && python scripts/playback.py --config ...\n"
        "  2. Install editable:    pip install -e <repo_root>\n"
        "  3. Set env var:         export LIBERO_PATH=<repo_root>\n"
        f"Original error: {_e}"
    )
    raise ModuleNotFoundError(_msg) from _e

from libero.libero.envs import OffScreenRenderEnv

from libero.libero.utils.temporal_perturbations import (
    TemporalPerturbationSpec,
    TemporalPerturbationManager,
    add_hidden_objects_to_bddl,
    specs_from_config,
    _write_object_pose,      # low-level pose writer used for z_override application
)
from libero.libero.utils.generate_perturbation_bddl import (
    read_bddl,
    validate_bddl,
    extract_target_workspace,
    resolve_z_overrides,     # converts sentinel/estimated entries to sim-accurate (cx,cy,z)
)


# ---------------------------------------------------------------------------
# Image preprocessing
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
# Model loading
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
    """
    if not hidden_objects:
        return base_bddl_path

    bddl_text = read_bddl(base_bddl_path)

    if target_workspace is None:
        target_workspace = extract_target_workspace(bddl_text)

    hidden_pairs = [(obj["name"], obj["type"]) for obj in hidden_objects]
    bddl_text = add_hidden_objects_to_bddl(bddl_text, hidden_pairs, target_workspace)

    if not validate_bddl(bddl_text):
        raise RuntimeError("Modified BDDL (with hidden objects) failed validation.")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".bddl", delete=False, prefix="temporal_"
    )
    tmp.write(bddl_text)
    tmp.close()
    print(f"[TEMPORAL] Wrote modified BDDL ({len(hidden_objects)} hidden object(s)) → {tmp.name}")
    return tmp.name


# ---------------------------------------------------------------------------
# Z-override application — resolve and apply after env.reset()
# ---------------------------------------------------------------------------

def apply_z_overrides(env, z_overrides_file: Optional[str]) -> None:
    """
    Load the z_overrides JSON sidecar produced by apply_perturbations(), resolve
    any sentinel / estimated entries against the live sim geometry, then write
    the corrected 7-DOF poses for each affected object.

    This must be called AFTER env.reset() (so the sim is fully initialised and
    sim.forward() has been called) but BEFORE the stabilization steps (so
    physics sees the objects at the correct height from the start).

    Args:
        env             : OffScreenRenderEnv / ControlEnv instance.
        z_overrides_file: Path to JSON sidecar, or None.  When None this
                          function is a no-op (no collisions were detected
                          during BDDL generation).

    The JSON sidecar contains a dict {obj_name: [element, ...]}.  Each value
    is either:
      • ["__cavity__", region_name, cx, cy]  — sentinel; Z resolved from sim
      • [cx, cy, estimated_z]               — plain stack; Z refined from sim

    After resolve_z_overrides() both forms become (cx, cy, z_final) and the
    object is teleported to that position using _write_object_pose().
    """
    if not z_overrides_file:
        return

    if not os.path.exists(z_overrides_file):
        print(f"[Z_OVERRIDE] WARN: z_overrides_file not found: {z_overrides_file}. Skipping.")
        return

    with open(z_overrides_file, "r") as f:
        raw = json.load(f)

    if not raw:
        return

    # JSON stores lists; convert back to tuples to match the internal contract.
    z_overrides = {k: tuple(v) for k, v in raw.items()}

    # Obtain the MuJoCo sim — same path used by TemporalPerturbationManager.
    sim = None
    for attr_path in (("sim",), ("env", "sim")):
        obj = env
        try:
            for attr in attr_path:
                obj = getattr(obj, attr)
            if obj is not None:
                sim = obj
                break
        except AttributeError:
            pass

    if sim is None:
        print("[Z_OVERRIDE] WARN: Could not access sim from env. Cannot apply z_overrides.")
        return

    # resolve_z_overrides() replaces sentinel/estimated entries with
    # sim-accurate (cx, cy, z) tuples using real geom positions.
    resolved = resolve_z_overrides(sim, z_overrides)

    if not resolved:
        return

    print(f"[Z_OVERRIDE] Applying {len(resolved)} collision Z correction(s)...")
    for obj_name, (cx, cy, z) in resolved.items():
        # Identity quaternion: w=1, x=0, y=0, z=0
        pose = np.array([cx, cy, z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        ok = _write_object_pose(sim, obj_name, pose)
        if ok:
            print(f"  [Z_OVERRIDE] '{obj_name}' → ({cx:.4f}, {cy:.4f}, z={z:.4f})")
        else:
            print(f"  [Z_OVERRIDE] WARN: Failed to set pose for '{obj_name}'")


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
        actions, dones, rewards, states, obs_list, perturbation_events, frames
    """
    np.random.seed(seed)
    env.seed(seed)
    obs = env.reset()

    # ---- Apply pre-episode collision Z corrections ----
    # Must happen after env.reset() (sim is live) but before stabilization
    # steps so physics sees objects at the correct height from frame 0.
    # This corrects objects that were detected to collide at BDDL-generation
    # time: they are stacked on top of colliders or placed inside open cavities.
    apply_z_overrides(env, config.get("z_overrides_file"))

    # ---- Initialize temporal perturbation manager ----
    # After reset() so the sim is constructed; hidden objects are parked here
    # (before stabilization so they don't bleed into the physics from their
    # off-screen spawn position).
    if temporal_manager is not None:
        temporal_manager.reset(env)

    # ---- Stabilization steps ----
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
    frames = []
    perturbation_events: List[Dict[str, Any]] = []

    step = 0
    done = False
    print(f"  Starting policy rollout (max {max_steps} steps, noise_std={noise_std})...")

    while not done and step < max_steps:

        # ---- Apply / revert temporal perturbations ----
        if temporal_manager is not None:
            prev_active = dict(temporal_manager._active)
            temporal_manager.step(env, step)
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
        frames.append(np.array(img))
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
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Save demo videos
# ---------------------------------------------------------------------------

def save_demo_video(frames: List[np.ndarray], output_path: str, fps: int = 20) -> None:
    """Save a list of (H, W, 3) uint8 frames as an MP4 video."""
    if not frames:
        print(f"  ⚠ No frames to save, skipping video.")
        return
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()
    print(f"  ✓ Saved video: {output_path} ({len(frames)} frames)")


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
        # Use different seed for each demo - ensure control and unperturbed have different seeds
        seed = demo_idx + (1000 if "unperturbed" in config["bddl_file"] else 0)
        print(f"{'=' * 60}")
        demo_data = record_single_demo(
            env, processor, vla, config,
            demo_index=demo_idx,
            seed=seed,
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

            dset.attrs["perturbation_events"] = json.dumps(demo_data["perturbation_events"])

    print(f"✓ Saved {num_demos} demo(s) to {out_file}")

    # ---- Save videos ----
    record_path = config["record_path"]
    video_dir = os.path.dirname(record_path)
    video_base = "video"

    print(f"\nSaving {num_demos} video(s)...")
    for demo_idx, demo_data in enumerate(all_demos):
        if num_demos == 1:
            video_path = os.path.join(video_dir, f"{video_base}.mp4")
        else:
            video_path = os.path.join(video_dir, f"{video_base}_demo_{demo_idx}.mp4")
        save_demo_video(demo_data["frames"], video_path, fps=20)

    print("\n✓ Recording and video export complete!")


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
    n_zo = 1 if config.get("z_overrides_file") else 0
    print(f"Temporal perturbations: {n_tp} | Hidden objects: {n_ho} | Z overrides: {'yes' if n_zo else 'no'}")
    print("=" * 80)

    record_demo(config)
    print("\n✓ Recording complete!")


if __name__ == "__main__":
    main()