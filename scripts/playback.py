"""
Render a LIBERO demonstration file into a video, with temporal perturbations replayed.

Replays existing demo episodes using OffScreenRenderEnv and saves rendered
frames to video (MP4 or AVI). When the original recording used temporal
perturbations, this script replays them at the exact same step windows so the
video matches what the robot actually experienced.

Supports rendering multiple demos from a single HDF5 file.
When multiple demos are present, creates separate videos for each:
  - demo.mp4 -> demo_0.mp4, demo_1.mp4, etc.

Usage Option 1 (YAML config — recommended, replays perturbations exactly):
    python playback.py --config ../configs/inference_config.yaml

Usage Option 2 (Direct arguments, no perturbations):
    python playback.py \\
        --demo_file /path/to/demo.hdf5 \\
        --bddl_file /path/to/scene.bddl \\
        --out_video /path/to/output/demo.mp4

How temporal perturbations are replayed
----------------------------------------
During recording, each demo's perturbation events are saved as a JSON string
in the HDF5 attribute  data/demo_N.attrs["perturbation_events"].

During playback the full perturbation specs are read from the YAML config
(same file used for recording), so every parameter — delta_xy, color, distractor
position, etc. — is reproduced exactly. The TemporalPerturbationManager runs
identically to how it ran during recording:

    manager.reset(env)          after env.reset()
    manager.step(env, step)     before each env.step()  <- applies / reverts
    manager.flush(env)          after episode ends

If no config is provided (direct CLI mode), perturbations are skipped and a
warning is printed. The video will still render correctly from the saved actions
but won't show the perturbation effects.

YAML Configuration:
    When using --config, the YAML file should contain:
      out_file      : Path to the HDF5 demo file produced by record.py
      bddl_file     : Path to the base BDDL scene file
      record_path   : Output video filepath (optional, defaults to demo.mp4)
      num_demos     : Number of demos to render (optional, auto-detects all)
      hidden_objects          : (same as record.py, needed for distractor/replace)
      temporal_perturbations  : (same as record.py)
"""

import os
import sys
import json
import cv2
import h5py
import argparse
import yaml
import tempfile
import numpy as np
from typing import Dict, Any, List, Optional

# ---------------------------------------------------------------------------
# Path setup — mirrors record.py so imports are robust regardless of cwd.
# See record.py for full explanation.
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
    TemporalPerturbationManager,
    add_hidden_objects_to_bddl,
    specs_from_config,
)
from libero.libero.utils.generate_perturbation_bddl import read_bddl, validate_bddl


# ---------------------------------------------------------------------------
# BDDL preparation (shared logic with record.py)
# ---------------------------------------------------------------------------

def prepare_bddl_with_hidden_objects(
    base_bddl_path: str,
    hidden_objects: List[Dict[str, str]],
    target_workspace: Optional[str] = None,
) -> str:
    """
    Inject hidden objects into a copy of the BDDL (same as record.py).
    Returns the path to use for the environment — either the original if no
    hidden objects are needed, or a temp file if objects were injected.
    """
    if not hidden_objects:
        return base_bddl_path

    bddl_text = read_bddl(base_bddl_path)

    if target_workspace is None:
        from perturbations import extract_target_workspace
        target_workspace = extract_target_workspace(bddl_text)

    hidden_pairs = [(obj["name"], obj["type"]) for obj in hidden_objects]
    bddl_text = add_hidden_objects_to_bddl(bddl_text, hidden_pairs, target_workspace)

    if not validate_bddl(bddl_text):
        raise RuntimeError("Modified BDDL (with hidden objects) failed validation.")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".bddl", delete=False, prefix="playback_temporal_"
    )
    tmp.write(bddl_text)
    tmp.close()
    print(f"[PLAYBACK] Wrote modified BDDL ({len(hidden_objects)} hidden object(s)) → {tmp.name}")
    return tmp.name


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def load_perturbation_events(demo_file: str, demo_index: int) -> List[Dict[str, Any]]:
    """
    Read the perturbation_events JSON attribute saved by record.py for a demo.
    Returns empty list if the attribute is absent (demo recorded without perturbations).
    """
    with h5py.File(demo_file, "r") as f:
        demo_key = f"data/demo_{demo_index}"
        if demo_key not in f:
            return []
        raw = f[demo_key].attrs.get("perturbation_events", "[]")
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []


def count_demos(demo_file: str) -> int:
    with h5py.File(demo_file, "r") as f:
        keys = [k for k in f["data"].keys() if k.startswith("demo_")]
    return len(keys)


# ---------------------------------------------------------------------------
# Single demo rendering
# ---------------------------------------------------------------------------

def render_single_demo(
    demo_file: str,
    bddl_file: str,
    demo_index: int,
    out_video: str,
    temporal_manager: Optional[TemporalPerturbationManager] = None,
    perturbation_events: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """
    Render a single demo to video, replaying temporal perturbations if provided.

    Args:
        demo_file         : Path to HDF5 file produced by record.py.
        bddl_file         : Path to BDDL file (may be a temp file with hidden objects).
        demo_index        : Which demo inside the HDF5 to render.
        out_video         : Output video file path (.mp4 or .avi).
        temporal_manager  : Configured TemporalPerturbationManager (or None).
        perturbation_events: Event log loaded from HDF5 attrs (used for display only).

    Returns:
        Number of frames rendered.
    """
    # ---- Load saved actions and initial state ----
    with h5py.File(demo_file, "r") as f:
        demo_key = f"data/demo_{demo_index}"
        if demo_key not in f:
            raise ValueError(f"Demo '{demo_key}' not found in {demo_file}")
        actions = f[demo_key]["actions"][()]
        init_state = f[demo_key]["states"][0]

    # ---- Build event lookup for display annotations ----
    # Maps step_idx -> list of event strings shown as overlay text
    event_at_step: Dict[int, List[str]] = {}
    if perturbation_events:
        for ev in perturbation_events:
            s = ev["step"]
            label = f"{ev['event'].upper()} {ev['type']} {ev['obj'] or ''}"
            event_at_step.setdefault(s, []).append(label)

    # ---- Initialize environment ----
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(demo_index)  # NOTE: is this correct?

    # Reset to recorded initial state.
    # ControlEnv.set_init_state() sets the MuJoCo state from the flattened
    # state vector and regenerates observations — this reproduces the exact
    # starting configuration from the recording.
    obs = env.reset()
    env.set_init_state(init_state)

    # Park hidden objects and arm the perturbation manager AFTER set_init_state
    # so the manager's sim reference is valid and object poses are stable.
    if temporal_manager is not None:
        temporal_manager.reset(env)

    frames = []

    def _grab_frame(obs_dict: dict) -> np.ndarray:
        img = obs_dict["agentview_image"]
        return np.clip(img, 0, 255).astype(np.uint8)

    # Capture the initial frame before any actions
    frames.append(_grab_frame(obs))

    # ---- Step through recorded actions ----
    for step, action in enumerate(actions):
        # Apply / revert perturbations at this step boundary
        if temporal_manager is not None:
            temporal_manager.step(env, step)

        obs, reward, done, info = env.step(action)
        frame = _grab_frame(obs)

        # Annotate frame with perturbation event text if one fired this step
        if step in event_at_step:
            for i, label in enumerate(event_at_step[step]):
                cv2.putText(
                    frame, label,
                    (8, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 80),   # yellow text
                    1, cv2.LINE_AA,
                )

        frames.append(frame)

        if (step + 1) % 50 == 0 or done:
            active = []
            if temporal_manager is not None:
                active = [
                    temporal_manager.specs[i].pert_type
                    for i, a in temporal_manager._active.items() if a
                ]
            pert_str = f" | active: {active}" if active else ""
            print(f"  Frame {step + 1}/{len(actions)}, reward={reward:.2f}{pert_str}")

        if done:
            print(f"  Episode finished at step {step}")
            break

    # Flush any still-active perturbations
    if temporal_manager is not None:
        temporal_manager.flush(env)
        temporal_manager.summary()

    env.close()

    # ---- Write video ----
    os.makedirs(os.path.dirname(os.path.abspath(out_video)), exist_ok=True)
    h, w, _ = frames[0].shape
    fourcc = (cv2.VideoWriter_fourcc(*"mp4v") if out_video.endswith(".mp4")
              else cv2.VideoWriter_fourcc(*"XVID"))
    print("fourcc:", fourcc)
    print("video path:", os.path.dirname(os.path.abspath(out_video)))
    writer = cv2.VideoWriter(out_video, fourcc, 20, (w, h))

    for frame in frames:
        # OpenCV expects BGR; camera obs is RGB
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"  ✓ Saved: {out_video} ({len(frames)} frames)")
    return len(frames)


# ---------------------------------------------------------------------------
# Multi-demo entry point
# ---------------------------------------------------------------------------

def render_demo(
    demo_file: str,
    bddl_file: str,
    out_video: str = "demo.mp4",
    num_demos: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render demo(s) from an HDF5 file into video(s).

    If config is provided, temporal perturbations are reconstructed from it
    and replayed exactly as during recording.  If config is None, perturbations
    are skipped (warning printed).

    Args:
        demo_file  : Path to HDF5 produced by record.py.
        bddl_file  : Path to the BASE bddl file (hidden objects injected internally).
        out_video  : Output path.  Multiple demos → indexed names (demo_0.mp4, …).
        num_demos  : How many demos to render. None = all.
        config     : Full parsed YAML config dict (same file used for recording).
    """
    # ---- Prepare BDDL with hidden objects ----
    hidden_objects = (config or {}).get("hidden_objects", [])
    target_workspace = (config or {}).get("target_workspace")
    effective_bddl = prepare_bddl_with_hidden_objects(
        bddl_file, hidden_objects, target_workspace
    )
    temp_bddl_created = (effective_bddl != bddl_file)

    # ---- Build temporal perturbation manager (once, reset per demo) ----
    temporal_specs_config = (config or {}).get("temporal_perturbations", [])
    temporal_manager: Optional[TemporalPerturbationManager] = None

    if temporal_specs_config:
        specs = specs_from_config(temporal_specs_config)
        temporal_manager = TemporalPerturbationManager(specs)
        print(f"\n[PLAYBACK] {len(specs)} temporal perturbation spec(s) will be replayed:")
        for i, s in enumerate(specs):
            print(f"  [{i}] {s.pert_type} | steps {s.start_step}–{s.end_step} "
                  f"| obj={s.obj_name or s.distractor_obj_name}")
    else:
        if config is not None:
            print("\n[PLAYBACK] No temporal_perturbations in config — rendering without perturbations.")
        else:
            print("\n[PLAYBACK] No config provided — rendering without perturbations.")
            print("           To replay perturbations, pass --config (same YAML used for recording).")

    # ---- Determine demos to render ----
    total = count_demos(demo_file)
    n = min(num_demos, total) if num_demos is not None else total
    print(f"\n[PLAYBACK] Found {total} demo(s), rendering {n}...\n")

    base, ext = (out_video.rsplit(".", 1) if "." in out_video else (out_video, "mp4"))

    try:
        for demo_idx in range(n):
            print(f"{'=' * 60}")
            print(f"Rendering demo {demo_idx + 1}/{n}  (demo_{demo_idx})")
            print(f"{'=' * 60}")

            video_path = out_video if n == 1 else f"{base}_{demo_idx}.{ext}"
            events = load_perturbation_events(demo_file, demo_idx)

            if events:
                print(f"  Perturbation events logged in HDF5: {len(events)}")
                for ev in events:
                    print(f"    step {ev['step']:>4}: {ev['event']:6} {ev['type']:10} "
                          f"obj={ev['obj']}")

            render_single_demo(
                demo_file=demo_file,
                bddl_file=effective_bddl,
                demo_index=demo_idx,
                out_video=video_path,
                temporal_manager=temporal_manager,
                perturbation_events=events,
            )
    finally:
        # Always clean up the temp BDDL file
        if temp_bddl_created and os.path.exists(effective_bddl):
            os.unlink(effective_bddl)
            print(f"\n[PLAYBACK] Cleaned up temp BDDL: {effective_bddl}")

    print(f"\n✓ Rendered {n} demo(s)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render LIBERO demonstrations as videos (with temporal perturbation replay)"
    )
    parser.add_argument(
        "--config", type=str,
        help="Path to YAML config (same file used with record.py). "
             "Required to replay temporal perturbations."
    )
    parser.add_argument("--demo_file", type=str,
                        help="Path to HDF5 demo file (if not using --config)")
    parser.add_argument("--bddl_file", type=str,
                        help="Path to BDDL scene file (if not using --config)")
    parser.add_argument("--out_video", type=str, default="demo.mp4",
                        help="Output video path (if not using --config)")
    args = parser.parse_args()

    config: Optional[Dict[str, Any]] = None

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        demo_file  = config.get("out_file")
        bddl_file  = config.get("bddl_file")
        out_video  = config.get("record_path", "demo.mp4")
        num_demos  = config.get("num_demos", None)

        if not demo_file:
            raise ValueError("Config must contain 'out_file' (path to HDF5 demo file)")
        if not bddl_file:
            raise ValueError("Config must contain 'bddl_file' (path to BDDL scene file)")
    else:
        if not args.demo_file or not args.bddl_file:
            parser.error("Provide --config OR both --demo_file and --bddl_file")
        demo_file = args.demo_file
        bddl_file = args.bddl_file
        out_video = args.out_video
        num_demos = None

    print("=" * 80)
    print("LIBERO Demo Playback (with Temporal Perturbation Replay)")
    print("=" * 80)
    print(f"BDDL file  : {bddl_file}")
    print(f"Demo file  : {demo_file}")
    print(f"Output     : {out_video}")
    n_tp = len((config or {}).get("temporal_perturbations", []))
    n_ho = len((config or {}).get("hidden_objects", []))
    print(f"Temporal perturbations: {n_tp} | Hidden objects: {n_ho}")
    print("=" * 80)

    render_demo(
        demo_file=demo_file,
        bddl_file=bddl_file,
        out_video=out_video,
        num_demos=num_demos,
        config=config,
    )

    print("\n✓ Playback complete!")


if __name__ == "__main__":
    main()