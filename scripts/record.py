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
from typing import List, Dict, Any, Optional, Tuple

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
    _read_object_pose,
)
from libero.libero.utils.generate_perturbation_bddl import (
    read_bddl,
    validate_bddl,
    extract_target_workspace,
    resolve_z_overrides,     # converts sentinel/estimated entries to sim-accurate (cx,cy,z)
)

from pipeline.random_design import chunk_to_start_end_step, get_max_rollout_frames


# ---------------------------------------------------------------------------
# Top-down (birdview) export for unperturbed heatmap alignment
# ---------------------------------------------------------------------------

def _chunk_start_step_to_index(num_chunks: int, max_frames: int) -> Dict[int, int]:
    """Map rollout step at chunk start -> chunk index (0 .. num_chunks-1)."""
    out: Dict[int, int] = {}
    for c in range(num_chunks):
        start_step, _ = chunk_to_start_end_step(c, num_chunks, max_frames)
        out[int(start_step)] = c
    return out


def _configure_narrow_topdown_camera(
    env,
    fovy_deg: float = 3.0,
    cam_z: float = 22.0,
    cam_x: float = -0.2,
    cam_y: float = 0.0,
) -> bool:
    """
    Telephoto top-down: very small FOV (near-parallel rays), high camera, and XY over the
    table center so the visible footprint is ~table-sized rather than table+full robot+room.
    Override via config top_down_camera: fovy_deg, camera_z, camera_x, camera_y.
    """
    sim = env.sim
    model = sim.model
    cid = -1
    cam_fn = getattr(model, "camera_name2id", None)
    if callable(cam_fn):
        try:
            cid = cam_fn("birdview")
        except ValueError as e:
            print(f"[TOP-DOWN] birdview camera: {e}")
            return False
    if cid < 0:
        try:
            import mujoco
            raw = getattr(model, "_model", model)
            cid = mujoco.mj_name2id(raw, int(mujoco.mjtObj.mjOBJ_CAMERA), "birdview")
        except Exception as e:
            print(f"[TOP-DOWN] Could not resolve birdview camera id: {e}")
            return False
    if cid < 0:
        print("[TOP-DOWN] birdview camera not found in MuJoCo model.")
        return False
    model.cam_fovy[cid] = float(fovy_deg)
    model.cam_pos[cid][0] = float(cam_x)
    model.cam_pos[cid][1] = float(cam_y)
    model.cam_pos[cid][2] = float(cam_z)
    sim.forward()
    return True


def _apply_top_down_camera_from_config(env, config: Dict[str, Any]) -> bool:
    """Apply top_down_camera YAML to birdview; must run after env.reset() (hard_reset wipes edits)."""
    td = config.get("top_down_camera") or {}
    fovy = float(td.get("fovy_deg", 3.0))
    cam_z = float(td.get("camera_z", 22.0))
    cam_x = float(td.get("camera_x", -0.2))
    cam_y = float(td.get("camera_y", 0.0))
    return _configure_narrow_topdown_camera(
        env, fovy_deg=fovy, cam_z=cam_z, cam_x=cam_x, cam_y=cam_y
    )


def _fresh_observations(env):
    """Re-render observations from the current sim state (e.g. after extra env.step calls)."""
    inner = env.env
    inner.sim.forward()
    inner._update_observables(force=True)
    return inner._get_observations()


def _offscreen_render_context(sim):
    """Robosuite MjSim stores the GL offscreen context here (see robosuite MujocoEnv._reset_internal)."""
    return getattr(sim, "_render_context_offscreen", None) or getattr(
        sim, "render_context_offscreen", None
    )


def _render_birdview_highres(env, width: int, height: int) -> Optional[np.ndarray]:
    """
    Render birdview at arbitrary resolution using the same offscreen pipeline as robosuite
    (does not change policy camera_obs size, which stays 256×256).
    """
    inner = env.env
    sim = inner.sim
    ctx = _offscreen_render_context(sim)
    if ctx is None:
        return None
    try:
        cid = sim.model.camera_name2id("birdview")
    except ValueError:
        return None
    try:
        sim.forward()
        ctx.render(int(width), int(height), camera_id=cid)
        rgb = ctx.read_pixels(int(width), int(height), depth=False)
    except Exception:
        return None
    rgb = np.asarray(rgb)
    # Match LIBERO / observable convention (same as low-res birdview_image save path).
    return rgb[::-1, ::-1]


def _estimate_table_surface_z(sim) -> float:
    """Approximate world z of tabletop from geoms named like *table* (m)."""
    sim.forward()
    z_best = 0.78
    model, data = sim.model, sim.data
    for gid in range(model.ngeom):
        try:
            nm = model.geom_id2name(gid)
        except Exception:
            continue
        if not nm or "table" not in nm.lower():
            continue
        try:
            hz = float(np.asarray(model.geom_size[gid])[2])
        except Exception:
            hz = float(np.max(np.asarray(model.geom_size[gid])))
        ztop = float(data.geom_xpos[gid, 2]) + hz
        if ztop > z_best:
            z_best = ztop
    return z_best


def _birdview_world_xy_extent_m(
    env,
    cam_name: str,
    w: int,
    h: int,
    z_table: float,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Map image edges to world XY on plane z=z_table (meters) using pinhole + cam pose.
    Returns (x_left, x_right, y_bottom, y_top) for matplotlib imshow(..., origin='upper').
    """
    inner = env.env
    sim = inner.sim
    try:
        cid = sim.model.camera_name2id(cam_name)
    except ValueError:
        return None
    sim.forward()
    C = np.asarray(sim.data.cam_xpos[cid], dtype=np.float64).reshape(3)
    R = np.asarray(sim.data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
    fovy_deg = float(np.asarray(sim.model.cam_fovy[cid]))
    fovy_rad = np.deg2rad(fovy_deg)
    tan_half_y = float(np.tan(fovy_rad * 0.5))
    tan_half_x = tan_half_y * (float(w) / float(h))

    def _world_xy_pixel(u: float, v: float) -> Optional[Tuple[float, float]]:
        # Pixel center (u,v); OpenGL-style NDC: sx left=-1, sy top=+1
        sx = (u + 0.5) / float(w) * 2.0 - 1.0
        sy = 1.0 - (v + 0.5) / float(h) * 2.0
        d_cam = np.array([sx * tan_half_x, sy * tan_half_y, -1.0], dtype=np.float64)
        d_world = R @ d_cam
        if abs(d_world[2]) < 1e-12:
            return None
        t = (z_table - C[2]) / d_world[2]
        if t <= 0.0:
            return None
        p = C + t * d_world
        return float(p[0]), float(p[1])

    def _mean_axis(samples: List[Optional[Tuple[float, float]]], idx: int) -> Optional[float]:
        vals = [s[idx] for s in samples if s is not None]
        return float(np.mean(vals)) if vals else None

    # Top / bottom image rows → world y (for origin='upper', extent top = row 0).
    y_top = _mean_axis([_world_xy_pixel(u, 0.5) for u in np.linspace(0.5, w - 0.5, 7)], 1)
    y_bot = _mean_axis([_world_xy_pixel(u, h - 0.5) for u in np.linspace(0.5, w - 0.5, 7)], 1)
    x_left = _mean_axis([_world_xy_pixel(0.5, v) for v in np.linspace(0.5, h - 0.5, 7)], 0)
    x_right = _mean_axis([_world_xy_pixel(w - 0.5, v) for v in np.linspace(0.5, h - 0.5, 7)], 0)
    if y_top is None or y_bot is None or x_left is None or x_right is None:
        return None
    xl, xr = (x_left, x_right) if x_left < x_right else (x_right, x_left)
    yb, yt = (y_bot, y_top) if y_bot < y_top else (y_top, y_bot)
    return xl, xr, yb, yt


def _save_birdview_png(
    env,
    path: str,
    export_width: Optional[int] = None,
    export_height: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save birdview RGB. Uses offscreen render at export_width×export_height when larger than
    policy resolution (256); otherwise uses birdview_image from observations.

    If config.top_down_camera.plot_distance_axes is true (default), also saves with matplotlib
    axes labeled in meters (world XY on the table plane z).
    """
    w = int(export_width) if export_width else 256
    h = int(export_height) if export_height else 256
    w = max(64, min(w, 4096))
    h = max(64, min(h, 4096))

    img = None
    if w > 256 or h > 256:
        img = _render_birdview_highres(env, w, h)
        if img is None:
            print(
                "[TOP-DOWN] WARN: high-res offscreen render unavailable; "
                "falling back to 256×256 birdview_image."
            )

    if img is None:
        obs = _fresh_observations(env)
        if "birdview_image" not in obs:
            print(f"[TOP-DOWN] WARN: no birdview_image in obs, skip {path}")
            return
        img = np.asarray(obs["birdview_image"])[::-1, ::-1]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    td = (config or {}).get("top_down_camera") or {}
    plot_axes = bool(td.get("plot_distance_axes", True))
    inner = env.env
    sim = inner.sim

    if plot_axes:
        z_table = td.get("table_plane_z")
        if z_table is not None:
            z_plane = float(z_table)
        else:
            z_plane = _estimate_table_surface_z(sim)
        extent = _birdview_world_xy_extent_m(env, "birdview", w, h, z_plane)
        if extent is not None:
            x0, x1, yb, yt = extent
            if abs(x1 - x0) >= 1e-5 and abs(yt - yb) >= 1e-5:
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    dpi = int(td.get("figure_dpi", 120))
                    dpi = max(72, min(dpi, 300))
                    fig_w = max(4.0, w / float(dpi))
                    fig_h = max(4.0, h / float(dpi))
                    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
                    ax.imshow(
                        img.astype(np.uint8),
                        extent=(x0, x1, yb, yt),
                        origin="upper",
                        interpolation="nearest",
                    )
                    ax.set_xlabel("x (m)")
                    ax.set_ylabel("y (m)")
                    ax.set_aspect("equal")
                    ax.set_title(f"Birdview (table plane z ≈ {z_plane:.2f} m)")
                    fig.savefig(path, bbox_inches="tight", pad_inches=0.15, dpi=dpi)
                    plt.close(fig)
                    print(
                        f"[TOP-DOWN] Saved {path} ({img.shape[1]}x{img.shape[0]} with axes, "
                        f"x=[{x0:.3f},{x1:.3f}] y=[{yb:.3f},{yt:.3f}] m)"
                    )
                    return
                except Exception as ex:
                    print(f"[TOP-DOWN] WARN: matplotlib axes save failed ({ex}); saving raw PNG.")

    Image.fromarray(img.astype(np.uint8)).save(path)
    print(f"[TOP-DOWN] Saved {path} ({img.shape[1]}×{img.shape[0]})")


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
    top_down_export: Optional[Dict[str, Any]] = None,
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

    # Birdview FOV/position live on mjModel; hard_reset reloads XML on reset(), so re-apply
    # after reset+stabilize or top-down exports ignore YAML (defaults from scene XML).
    if top_down_export is not None:
        td = config.get("top_down_camera") or {}
        if _apply_top_down_camera_from_config(env, config):
            print(
                f"[TOP-DOWN] birdview applied post-reset: fovy={float(td.get('fovy_deg', 3.0))}°, "
                f"pos=({float(td.get('camera_x', -0.2))}, {float(td.get('camera_y', 0.0))}, "
                f"{float(td.get('camera_z', 22.0))})"
            )

    max_steps = get_max_rollout_frames(config)
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

            # If perturbation was just applied with no collision, run 10 stabilization
            # steps and update applied_poses to the settled position so that the
            # robot-movement check at revert time is against the stabilized pose,
            # not the immediately-written pose.
            for i, spec in enumerate(temporal_manager.specs):
                was_active = prev_active.get(i, False)
                now_active = temporal_manager._active.get(i, False)
                if not was_active and now_active:
                    if not temporal_manager.perturbation_collision:
                        # Run stabilization steps without recording them
                        for _ in range(10):
                            env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
                        # Update applied_poses to post-stabilization position
                        snap = temporal_manager._snapshots.get(i)
                        if snap is not None:
                            sim = temporal_manager._sim
                            for obj_name in snap.applied_poses:
                                settled_pose = _read_object_pose(sim, obj_name)
                                if settled_pose is not None:
                                    snap.applied_poses[obj_name] = settled_pose
                                    print(f"[TEMPORAL] Updated applied_pose for '{obj_name}' "
                                          f"after stabilization → {settled_pose[:3]}")

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

        if top_down_export:
            stc = top_down_export.get("step_to_chunk") or {}
            if step in stc:
                chunk_idx = stc[step]
                out_name = os.path.join(
                    top_down_export["dir"],
                    f"demo_{demo_index:03d}_chunk_{chunk_idx:03d}.png",
                )
                _save_birdview_png(
                    env,
                    out_name,
                    export_width=top_down_export.get("birdview_width"),
                    export_height=top_down_export.get("birdview_height"),
                    config=config,
                )

        # ---- Policy inference ----
        img = preprocess_image(obs, resize_size=256, center_crop=True)
        frames.append(np.array(img))
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

    pert_id = config.get("perturbation_id", "")
    run_dir_s = config.get("run_dir")
    if not run_dir_s and config.get("out_file"):
        run_dir_s = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(config["out_file"])), "..")
        )
    temporal_cfg = config.get("temporal_perturbation") or {}
    num_chunks_td = int(temporal_cfg.get("num_chunks", 1))
    if num_chunks_td < 1:
        num_chunks_td = 1
    max_frames_td = get_max_rollout_frames(config)

    top_down_export: Optional[Dict[str, Any]] = None
    if pert_id == "unperturbed" and run_dir_s:
        top_down_dir = os.path.join(run_dir_s, "top-down-frames")
        os.makedirs(top_down_dir, exist_ok=True)
        td_pre = config.get("top_down_camera") or {}
        bvw = int(td_pre.get("birdview_width", td_pre.get("width", 1024)))
        bvh = int(td_pre.get("birdview_height", td_pre.get("height", 1024)))
        bvw = max(64, min(bvw, 4096))
        bvh = max(64, min(bvh, 4096))
        top_down_export = {
            "dir": top_down_dir,
            "step_to_chunk": _chunk_start_step_to_index(num_chunks_td, max_frames_td),
            "birdview_width": bvw,
            "birdview_height": bvh,
        }
        print(
            f"\n[TOP-DOWN] Unperturbed run: saving birdview at chunk starts "
            f"({num_chunks_td} chunk(s), max_frames={max_frames_td}, PNG {bvw}×{bvh}) → {top_down_dir}"
        )

    # ---- Initialize environment ----
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    if top_down_export is not None:
        env_args["camera_names"] = ["agentview", "birdview"]
    print("\nInitializing environment...")
    env = OffScreenRenderEnv(**env_args)
    # Note: do not configure birdview here — record_single_demo's env.reset() (hard_reset)
    # reloads mjModel from XML and would discard FOV/position before any PNG is saved.

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
            top_down_export=top_down_export,
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

    # ---- Save videos (paths match create_record_config: results/videos/<perturbation_id>.mp4) ----
    record_path = config.get("record_path")
    if record_path:
        record_path = os.path.normpath(os.path.abspath(record_path))
        video_dir = os.path.dirname(record_path)
        os.makedirs(video_dir, exist_ok=True)
        stem, ext = os.path.splitext(os.path.basename(record_path))
        if not ext:
            ext = ".mp4"

        print(f"\nSaving {num_demos} video(s)...")
        for demo_idx, demo_data in enumerate(all_demos):
            if num_demos == 1:
                video_path = record_path
            else:
                video_path = os.path.join(video_dir, f"{stem}_demo_{demo_idx}{ext}")
            save_demo_video(demo_data["frames"], video_path, fps=20)
    else:
        print("\n⚠ No record_path in config; skipping video export.")

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