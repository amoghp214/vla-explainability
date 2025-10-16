"""
Render a LIBERO (or custom) demonstration file into a video.

This script replays an existing demo episode using the OffScreenRenderEnv
and saves the rendered frames to a video file (MP4 or AVI).

Unlike the original LIBERO repo utilities, this script does not rely on
benchmark_name or task_id. Instead, you directly pass the demo HDF5 file
and its associated BDDL scene file.

Usage:
    python playback.py \
        --demo_file /path/to/demo_file.hdf5 \
        --bddl_file /path/to/scene.bddl \
        --out_video /path/to/output/demo.mp4

Arguments:
    --demo_file    Path to the HDF5 demo file
    --bddl_file    Path to the BDDL scene file describing the environment
    --out_video    Full output video filepath (.mp4 or .avi). Default: demo.mp4
"""

import os
import cv2
import h5py
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_file", type=str, required=True)
    parser.add_argument("--bddl_file", type=str, required=True)
    parser.add_argument("--out_video", type=str, default="demo.mp4")
    args = parser.parse_args()

    print(f"Using BDDL file: {args.bddl_file}")
    print(f"Using demo file: {args.demo_file}")
    print(f"Output video path: {args.out_video}")

    render_demo(args.demo_file, args.bddl_file, args.out_video)


if __name__ == "__main__":
    main()
