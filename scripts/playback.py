"""
Render a LIBERO (or custom) demonstration file into a video.

This script replays an existing demo episode using the OffScreenRenderEnv
and saves the rendered frames to a video file (MP4 or AVI).

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
"""

import os
import cv2
import h5py
import argparse
import yaml
import numpy as np

from libero.libero.envs import OffScreenRenderEnv


def render_demo(demo_file, bddl_file, out_video="demo.mp4"):
    """
    Render a demo (HDF5 + BDDL) into a video.
    """
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }

    # Initialize environment
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    frames = []

    # Load HDF5 demo
    with h5py.File(demo_file, "r") as f:
        actions = f["data/demo_0/actions"][()]
        init_state = f["data/demo_0/states"][0]

    # Set environment to initial state
    env.set_init_state(init_state)
    obs = env.reset()

    # Render first frame
    frames.append((np.clip(obs["agentview_image"], 0, 255)).astype("uint8"))

    # Step through actions
    for i, action in enumerate(actions):
        print(f"Action: {action}")
        obs, reward, done, info = env.step(action)
        frame = (np.clip(obs["agentview_image"], 0, 255)).astype("uint8")
        frames.append(frame)
        if done:
            print(f"Demo finished at step {i}")
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

    print(f"Saved demo video: {out_video} ({len(frames)} frames)")


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

    print("=" * 80)
    print("LIBERO Demo Playback")
    print("=" * 80)
    print(f"BDDL file: {bddl_file}")
    print(f"Demo file: {demo_file}")
    print(f"Output video: {out_video}")
    print("=" * 80)

    render_demo(demo_file, bddl_file, out_video)
    
    print("\n✓ Playback complete!")


if __name__ == "__main__":
    main()
