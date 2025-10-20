"""
Render a LIBERO (or custom) demonstration file into a video.

This script replays existing demo episodes using the OffScreenRenderEnv
and saves the rendered frames to video files (MP4 or AVI).

Supports rendering multiple demos from a single HDF5 file.
When multiple demos are present, creates separate videos for each:
  - demo.mp4 -> demo_demo_0.mp4, demo_demo_1.mp4, etc.

Usage Option 1 (YAML config):
    python playback.py --config ../configs/inference_config.yaml

Usage Option 2 (Direct arguments):
    python playback.py \
        --demo_file /path/to/demo_file.hdf5 \
        --bddl_file /path/to/scene.bddl \
        --out_video /path/to/output/demo.mp4

YAML Configuration:
    When using --config, the YAML file should contain:
    - out_file: Path to the HDF5 demo file
    - bddl_file: Path to the BDDL scene file
    - record_path: Output video filepath (optional, defaults to demo.mp4)
    - num_demos: Number of demos to render (optional, auto-detects all if not specified)
"""

import os
import cv2
import h5py
import argparse
import yaml
import numpy as np

from libero.libero.envs import OffScreenRenderEnv


def render_single_demo(demo_file, bddl_file, demo_index, out_video):
    """
    Render a single demo (HDF5 + BDDL) into a video.
    """
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }

    # Initialize environment
    env = OffScreenRenderEnv(**env_args)
    env.seed(demo_index)
    frames = []

    # Load HDF5 demo
    with h5py.File(demo_file, "r") as f:
        demo_key = f"data/demo_{demo_index}"
        if demo_key not in f:
            raise ValueError(f"Demo '{demo_key}' not found in {demo_file}")
        
        actions = f[demo_key]["actions"][()]
        init_state = f[demo_key]["states"][0]

    # Set environment to initial state
    env.set_init_state(init_state)
    obs = env.reset()

    # Render first frame
    frames.append((np.clip(obs["agentview_image"], 0, 255)).astype("uint8"))

    # Step through actions
    for i, action in enumerate(actions):
        if (i + 1) % 10 == 0:
            print(f"  Frame {i + 1}/{len(actions)}")
        obs, reward, done, info = env.step(action)
        frame = (np.clip(obs["agentview_image"], 0, 255)).astype("uint8")
        frames.append(frame)
        if done:
            print(f"  Demo finished at step {i}")
            break

    env.close()

    # Save to video (path used directly)
    os.makedirs(os.path.dirname(out_video), exist_ok=True)
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") if out_video.endswith(".mp4") else cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(out_video, fourcc, 20, (w, h))

    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"  ✓ Saved video: {out_video} ({len(frames)} frames)")
    return len(frames)


def render_demo(demo_file, bddl_file, out_video="demo.mp4", num_demos=None):
    """
    Render demo(s) from HDF5 file into video(s).
    
    If num_demos is None, it will detect and render all demos in the file.
    If num_demos is specified, it will render that many demos.
    """
    # Detect available demos
    with h5py.File(demo_file, "r") as f:
        available_demos = [key for key in f["data"].keys() if key.startswith("demo_")]
        available_demos.sort()
    
    total_demos = len(available_demos)
    
    # Determine how many demos to render
    if num_demos is None:
        demos_to_render = total_demos
    else:
        demos_to_render = min(num_demos, total_demos)
    
    print(f"Found {total_demos} demo(s) in file, rendering {demos_to_render}...")
    
    if demos_to_render == 1:
        # Single demo - use provided output path directly
        print(f"\nRendering demo 0...")
        render_single_demo(demo_file, bddl_file, 0, out_video)
    else:
        # Multiple demos - create separate videos with indexed names
        base_path = out_video.rsplit(".", 1)[0]  # Remove extension
        extension = out_video.rsplit(".", 1)[1] if "." in out_video else "mp4"
        
        for demo_idx in range(demos_to_render):
            print(f"\n{'='*60}")
            print(f"Rendering demo {demo_idx + 1}/{demos_to_render}")
            print(f"{'='*60}")
            
            # Create indexed output path
            indexed_video = f"{base_path}_{demo_idx}.{extension}"
            render_single_demo(demo_file, bddl_file, demo_idx, indexed_video)
    
    print(f"\n✓ Rendered {demos_to_render} demo(s)")


def main():
    parser = argparse.ArgumentParser(description="Render LIBERO demonstrations as videos")
    parser.add_argument("--config", type=str, help="Path to YAML config file (same format as record.py)")
    parser.add_argument("--demo_file", type=str, help="Path to HDF5 demo file (if not using --config)")
    parser.add_argument("--bddl_file", type=str, help="Path to BDDL scene file (if not using --config)")
    parser.add_argument("--out_video", type=str, default="demo.mp4", help="Output video path (if not using --config)")
    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        demo_file = config.get("out_file")
        bddl_file = config.get("bddl_file")
        out_video = config.get("record_path", "demo.mp4")
        num_demos = config.get("num_demos", None)  # Auto-detect if not specified
        
        if not demo_file:
            raise ValueError("Config file must contain 'out_file' field for demo HDF5 path")
        if not bddl_file:
            raise ValueError("Config file must contain 'bddl_file' field")
    else:
        # Use direct command-line arguments
        if not args.demo_file or not args.bddl_file:
            parser.error("Either --config or both --demo_file and --bddl_file must be provided")
        
        demo_file = args.demo_file
        bddl_file = args.bddl_file
        out_video = args.out_video
        num_demos = None  # Auto-detect all demos

    print("=" * 80)
    print("LIBERO Demo Playback")
    print("=" * 80)
    print(f"BDDL file: {bddl_file}")
    print(f"Demo file: {demo_file}")
    print(f"Output video: {out_video}")
    print("=" * 80)

    render_demo(demo_file, bddl_file, out_video, num_demos)
    
    print("\n✓ Playback complete!")


if __name__ == "__main__":
    main()
