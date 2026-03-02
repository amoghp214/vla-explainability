"""
Temporal (mid-rollout) perturbation engine for LIBERO/MuJoCo environments.

Perturbations are defined as time-windowed specs with [start_step, end_step].
At start_step, the perturbation is applied directly to the live MuJoCo sim.
At end_step, the original state is restored (revert=True by default).

Supported perturbation types (all applied via sim state manipulation):
  - move      : Teleport object to a new XY position in the table plane
  - color     : Change object geom RGBA color(s)
  - distractor: Teleport a pre-loaded "hidden" object into the scene
  - replace   : Hide original object off-screen, teleport replacement in

Architecture notes:
  - MuJoCo does NOT support adding new bodies mid-episode, so distractors and
    replacement objects must be declared in the BDDL and pre-spawned far
    off-screen (via a special helper BDDL writer), then teleported in/out.
  - Object positions are set via free-joint qpos (7-DOF: xyz + quaternion).
  - Colors are set via sim.model.geom_rgba (Nx4 float32 array).
  - All original state is snapshotted just before a perturbation window opens
    and fully restored when the window closes — UNLESS the robot gripper moved
    the object during the perturbation window, in which case the position revert
    is skipped so the object remains where the robot left it.

Robot-movement detection:
  - At start_step, alongside the original pose, we also record the
    perturbation-applied pose (i.e. the pose we explicitly wrote to the sim).
  - At end_step + 1, we read the object's current pose and compare it to the
    perturbation-applied pose using a configurable 3-D (XYZ) distance threshold
    (default: ROBOT_MOVE_THRESHOLD_M = 0.0 m, i.e. any movement counts).
  - If the current pose differs from the perturbation-applied pose by more than
    the threshold, we infer that the robot gripper moved the object during the
    window and skip restoring that object's position (while still reverting
    colors as normal).  This applies to ALL object types including distractors.

Usage:
    See TemporalPerturbationConfig and TemporalPerturbationManager below.
    Example at bottom of file.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Off-screen "parking" position for hidden objects
# ---------------------------------------------------------------------------
_OFFSCREEN_XYZ = np.array([100.0, 100.0, 100.0], dtype=np.float64)
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

# ---------------------------------------------------------------------------
# Robot-movement detection threshold (metres, full 3-D XYZ distance).
# If the object's position has shifted more than this from the perturbation-
# applied pose by the time the window closes, we conclude that the robot
# gripper moved it and we do NOT restore its original position.
# Default is 0.0 m, meaning ANY positional change counts as robot movement.
# ---------------------------------------------------------------------------
ROBOT_MOVE_THRESHOLD_M: float = 0.0 # NOTE: may want to have some slack

# Default color palette for color perturbations
COLOR_PALETTE: Dict[str, np.ndarray] = {
    "red":     np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
    "blue":    np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    "green":   np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    "yellow":  np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
    "purple":  np.array([0.5, 0.0, 0.5, 1.0], dtype=np.float32),
    "orange":  np.array([1.0, 0.5, 0.0, 1.0], dtype=np.float32),
    "cyan":    np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
    "magenta": np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32),
    "white":   np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    "black":   np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
}


# ---------------------------------------------------------------------------
# Low-level MuJoCo helpers
# ---------------------------------------------------------------------------

def _get_body_id(sim, body_name: str) -> int:
    """Return MuJoCo body id for a named body, or -1 if not found."""
    try:
        return sim.model.body_name2id(body_name)
    except Exception:
        return -1


def _get_joint_id(sim, joint_name: str) -> int:
    """Return MuJoCo joint id for a named joint, or -1 if not found."""
    try:
        return sim.model.joint_name2id(joint_name)
    except Exception:
        return -1


def _get_geom_ids_for_body(sim, body_id: int) -> List[int]:
    """Return all geom ids attached to a given body."""
    geom_ids = []
    for geom_id in range(sim.model.ngeom):
        if sim.model.geom_bodyid[geom_id] == body_id:
            geom_ids.append(geom_id)
    return geom_ids


def _find_object_body_name(sim, obj_name: str) -> Optional[str]:
    """
    Try to find the MuJoCo body name corresponding to a LIBERO object name.
    LIBERO typically names bodies as '{obj_name}_main' or just '{obj_name}'.
    Returns the body name string, or None if not found.
    """
    candidates = [
        obj_name,
        f"{obj_name}_main",
        f"{obj_name}_base",
        f"{obj_name}_body",
    ]
    for name in candidates:
        if _get_body_id(sim, name) >= 0:
            return name
    # Fuzzy: scan all body names for obj_name as substring
    for i in range(sim.model.nbody):
        bname = sim.model.body_id2name(i)
        if obj_name in bname:
            return bname
    return None


def _get_free_joint_qpos_addr(sim, obj_name: str) -> Optional[Tuple[int, str]]:
    """
    Return (qpos_addr, joint_name) for the free joint of a LIBERO object.
    LIBERO objects with free joints are named '{obj_name}_joint0' or similar.
    Returns None if no free joint found.
    """
    free_joint_candidates = [
        f"{obj_name}_joint0",
        f"{obj_name}_freejoint",
        f"{obj_name}:joint",
        obj_name,
    ]
    for jname in free_joint_candidates:
        jid = _get_joint_id(sim, jname)
        if jid >= 0:
            jtype = sim.model.jnt_type[jid]
            # MuJoCo joint type 0 = free joint
            if jtype == 0:
                addr = sim.model.jnt_qposadr[jid]
                return addr, jname
    # Broader search: scan all joints
    for jid in range(sim.model.njnt):
        jname = sim.model.joint_id2name(jid)
        if obj_name in jname and sim.model.jnt_type[jid] == 0:
            addr = sim.model.jnt_qposadr[jid]
            return addr, jname
    return None


def _read_object_pose(sim, obj_name: str) -> Optional[np.ndarray]:
    """Read the current 7-DOF pose (xyz + quat) of a free-jointed object."""
    result = _get_free_joint_qpos_addr(sim, obj_name)
    if result is None:
        return None
    addr, _ = result
    return sim.data.qpos[addr: addr + 7].copy()


def _write_object_pose(sim, obj_name: str, pose: np.ndarray) -> bool:
    """
    Write a 7-DOF pose (xyz + quat wxyz) to the free joint of an object.
    Calls sim.forward() to propagate kinematics.
    Returns True on success.
    """
    result = _get_free_joint_qpos_addr(sim, obj_name)
    if result is None:
        print(f"[TEMPORAL] WARN: No free joint found for '{obj_name}'. Cannot set pose.")
        return False
    addr, jname = result
    sim.data.qpos[addr: addr + 7] = pose
    # Zero out velocities for this joint to avoid ghost forces
    vel_addr = sim.model.jnt_dofadr[sim.model.joint_name2id(jname)]
    sim.data.qvel[vel_addr: vel_addr + 6] = 0.0
    sim.forward()
    return True


def _read_object_colors(sim, obj_name: str) -> Optional[Dict[int, np.ndarray]]:
    """Read RGBA of all geoms belonging to an object body. Returns {geom_id: rgba}."""
    body_name = _find_object_body_name(sim, obj_name)
    if body_name is None:
        return None
    body_id = _get_body_id(sim, body_name)
    geom_ids = _get_geom_ids_for_body(sim, body_id)
    if not geom_ids:
        return None
    return {gid: sim.model.geom_rgba[gid].copy() for gid in geom_ids}


def _write_object_colors(sim, obj_name: str, rgba: np.ndarray) -> bool:
    """Set all geoms of an object body to the given RGBA color."""
    body_name = _find_object_body_name(sim, obj_name)
    if body_name is None:
        print(f"[TEMPORAL] WARN: Body not found for '{obj_name}'. Cannot set color.")
        return False
    body_id = _get_body_id(sim, body_name)
    geom_ids = _get_geom_ids_for_body(sim, body_id)
    if not geom_ids:
        print(f"[TEMPORAL] WARN: No geoms found for '{obj_name}'. Cannot set color.")
        return False
    for gid in geom_ids:
        sim.model.geom_rgba[gid] = rgba
    return True


def _park_object(sim, obj_name: str) -> bool:
    """Teleport an object to the off-screen parking position."""
    pose = np.concatenate([_OFFSCREEN_XYZ, _IDENTITY_QUAT])
    return _write_object_pose(sim, obj_name, pose)


def _place_object_at_xy(sim, obj_name: str, x: float, y: float,
                         z_override: Optional[float] = None) -> bool:
    """
    Teleport an object to table-plane coordinates (x, y).
    Preserves current z unless z_override is provided.
    Preserves current quaternion orientation.
    """
    current_pose = _read_object_pose(sim, obj_name)
    if current_pose is None:
        # Object was parked — use a default z slightly above table
        z = z_override if z_override is not None else 0.02
        quat = _IDENTITY_QUAT
    else:
        z = z_override if z_override is not None else current_pose[2]
        quat = current_pose[3:7]
    new_pose = np.array([x, y, z, *quat], dtype=np.float64)
    return _write_object_pose(sim, obj_name, new_pose)


# ---------------------------------------------------------------------------
# Robot-movement detection helper
# ---------------------------------------------------------------------------

def _robot_moved_object(
    current_pose: Optional[np.ndarray],
    applied_pose: Optional[np.ndarray],
    threshold_m: float = ROBOT_MOVE_THRESHOLD_M,
) -> bool:
    """
    Return True if the object's current 3-D position differs from the pose that
    was written by the perturbation engine (applied_pose) by more than
    threshold_m.  This indicates that the robot gripper displaced the object
    during the perturbation window, so we should NOT reset it.

    We compare full XYZ distance because any positional change — including
    being lifted vertically by the gripper — counts as the robot moving
    the object.

    Args:
        current_pose  : 7-DOF pose read from sim just before revert (xyz+quat).
        applied_pose  : 7-DOF pose that the perturbation engine wrote at start_step.
        threshold_m   : 3-D distance threshold in metres (default: ROBOT_MOVE_THRESHOLD_M).

    Returns:
        True  → robot moved the object; skip position revert.
        False → object is still where the engine placed it; safe to revert.
    """
    if current_pose is None or applied_pose is None:
        # Can't tell — conservatively do NOT revert.
        return True

    xyz_delta = np.linalg.norm(current_pose[:3] - applied_pose[:3])
    return xyz_delta > threshold_m


# ---------------------------------------------------------------------------
# Perturbation spec dataclass
# ---------------------------------------------------------------------------

@dataclass
class TemporalPerturbationSpec:
    """
    Defines a single time-windowed perturbation.

    Fields:
        pert_type   : One of "move", "color", "distractor", "replace"
        start_step  : Episode step at which to apply the perturbation (inclusive)
        end_step    : Episode step at which to revert the perturbation (inclusive)
        obj_name    : Primary object to perturb (not needed for distractor)
        robot_move_threshold_m : 3-D XYZ distance (metres) used to decide whether
                      the robot moved the object during the window.  If the
                      object has shifted more than this from the perturbation-
                      applied pose (in full XYZ), the position revert is skipped.
                      Default is 0.0 m — any positional change counts as robot
                      movement.  Applies to all perturbation types including
                      distractors.
        # move-specific
        delta_xy    : (dx, dy) shift in meters; if None, sampled from max_move_m
        max_move_m  : Max random shift magnitude when delta_xy is None
        # color-specific
        color       : Color name from COLOR_PALETTE, or None to pick randomly
        # distractor-specific
        distractor_obj_name : Name of pre-loaded distractor object in sim
        distractor_xy       : (x, y) where distractor appears; random if None
        # replace-specific
        replacement_obj_name: Name of pre-loaded replacement object in sim
    """
    pert_type: str  # "move" | "color" | "distractor" | "replace"
    start_step: int
    end_step: int

    # Shared
    obj_name: Optional[str] = None

    # Robot-movement detection threshold (per-spec override)
    robot_move_threshold_m: float = ROBOT_MOVE_THRESHOLD_M

    # move
    delta_xy: Optional[Tuple[float, float]] = None
    max_move_m: float = 0.05

    # color
    color: Optional[str] = None

    # distractor
    distractor_obj_name: Optional[str] = None
    distractor_xy: Optional[Tuple[float, float]] = None

    # replace
    replacement_obj_name: Optional[str] = None

    def validate(self):
        assert self.end_step >= self.start_step, \
            f"end_step ({self.end_step}) must be >= start_step ({self.start_step})"
        assert self.pert_type in ("move", "color", "distractor", "replace"), \
            f"Unknown pert_type: {self.pert_type}"
        if self.pert_type in ("move", "color", "replace"):
            assert self.obj_name, f"obj_name required for pert_type='{self.pert_type}'"
        if self.pert_type == "distractor":
            assert self.distractor_obj_name, "distractor_obj_name required for pert_type='distractor'"
        if self.pert_type == "replace":
            assert self.replacement_obj_name, "replacement_obj_name required for pert_type='replace'"
        if self.color is not None:
            assert self.color in COLOR_PALETTE, \
                f"Unknown color '{self.color}'. Valid: {list(COLOR_PALETTE.keys())}"


# ---------------------------------------------------------------------------
# Snapshot: captures all state needed to fully revert a perturbation
# ---------------------------------------------------------------------------

@dataclass
class _PerturbationSnapshot:
    """
    Internal: stores original state for a single perturbation so it can be
    reverted, plus the pose(s) that the engine itself wrote so we can detect
    whether the robot moved any object during the window.
    """
    spec: TemporalPerturbationSpec

    # Keyed by object name → original 7-DOF pose (recorded at start_step,
    # BEFORE the perturbation is applied).
    original_poses: Dict[str, np.ndarray] = field(default_factory=dict)

    # Keyed by object name → 7-DOF pose WRITTEN by the perturbation engine at
    # start_step.  Used at end_step to check whether the robot displaced the
    # object relative to where we placed it.
    applied_poses: Dict[str, np.ndarray] = field(default_factory=dict)

    # Keyed by object name → {geom_id: original_rgba}
    original_colors: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class TemporalPerturbationManager:
    """
    Manages time-windowed perturbations for a single episode rollout.

    Usage:
        manager = TemporalPerturbationManager(specs)
        manager.reset(env)          # Call once after env.reset()

        for step in range(max_steps):
            manager.step(env, step) # Call once per step, BEFORE env.step()
            obs, reward, done, _ = env.step(action)

        manager.summary()           # Print a log of what was applied

    Robot-movement detection
    ------------------------
    For "move", "replace", and "distractor" perturbation types, the engine
    records the exact pose it writes at start_step (the "applied pose").  At
    end_step + 1, before reverting, it reads the object's current pose and
    computes the full XYZ Euclidean distance from the applied pose.  If that
    distance exceeds spec.robot_move_threshold_m the object is considered to
    have been picked up / repositioned by the robot during the perturbation
    window, and its position is NOT reverted.  The default threshold is 0.0 m,
    meaning any positional change — including vertical lifting — prevents the
    revert.  Color reverts are unaffected by this check.
    """

    def __init__(self, specs: List[TemporalPerturbationSpec]):
        for spec in specs:
            spec.validate()
        self.specs = specs
        self._snapshots: Dict[int, _PerturbationSnapshot] = {}  # spec_idx → snapshot
        self._active: Dict[int, bool] = {}                       # spec_idx → is_active
        self._log: List[str] = []
        self._sim = None  # cached sim reference

    def reset(self, env):
        """
        Call this after env.reset() and before the first step.
        Parks all distractor/replacement objects off-screen so they are
        invisible until their perturbation window opens.
        """
        self._sim = self._get_sim(env)
        self._snapshots = {}
        self._active = {i: False for i in range(len(self.specs))}
        self._log = []

        if self._sim is None:
            print("[TEMPORAL] WARN: Could not access sim from env. "
                  "Perturbations will be skipped.")
            return

        # Park distractors and replacement objects off-screen
        for i, spec in enumerate(self.specs):
            if spec.pert_type == "distractor" and spec.distractor_obj_name:
                ok = _park_object(self._sim, spec.distractor_obj_name)
                msg = f"[TEMPORAL] Parked distractor '{spec.distractor_obj_name}'"
                if not ok:
                    msg += " [FAILED - object not found in sim]"
                print(msg)
            if spec.pert_type == "replace" and spec.replacement_obj_name:
                ok = _park_object(self._sim, spec.replacement_obj_name)
                msg = f"[TEMPORAL] Parked replacement '{spec.replacement_obj_name}'"
                if not ok:
                    msg += " [FAILED - object not found in sim]"
                print(msg)

    def step(self, env, step_idx: int):
        """
        Call once per step (before env.step()) with the current step index.
        Applies perturbations whose window opens at this step and
        reverts perturbations whose window closes at this step.
        """
        sim = self._sim
        if sim is None:
            return

        for i, spec in enumerate(self.specs):
            currently_active = self._active[i]

            # ---- Window opens ----
            if step_idx == spec.start_step and not currently_active:
                snapshot = self._apply(sim, i, spec)
                if snapshot is not None:
                    self._snapshots[i] = snapshot
                    self._active[i] = True

            # ---- Window closes ----
            elif step_idx == spec.end_step + 1 and currently_active:
                self._revert(sim, i)
                self._active[i] = False

    def flush(self, env):
        """
        Call at episode end to revert any still-active perturbations.
        Safe to call even if no perturbations are active.
        """
        sim = self._sim
        if sim is None:
            return
        for i, active in list(self._active.items()):
            if active:
                self._revert(sim, i)
                self._active[i] = False

    def summary(self):
        """Print a summary of all perturbation events that occurred."""
        print("\n" + "=" * 60)
        print("[TEMPORAL] Perturbation Summary")
        print("=" * 60)
        if not self._log:
            print("  (no perturbations were applied)")
        for msg in self._log:
            print(f"  {msg}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Internal: apply
    # ------------------------------------------------------------------

    def _apply(self, sim, spec_idx: int, spec: TemporalPerturbationSpec
               ) -> Optional[_PerturbationSnapshot]:
        snapshot = _PerturbationSnapshot(spec=spec)

        if spec.pert_type == "move":
            return self._apply_move(sim, spec, snapshot)
        elif spec.pert_type == "color":
            return self._apply_color(sim, spec, snapshot)
        elif spec.pert_type == "distractor":
            return self._apply_distractor(sim, spec, snapshot)
        elif spec.pert_type == "replace":
            return self._apply_replace(sim, spec, snapshot)
        return None

    def _apply_move(self, sim, spec: TemporalPerturbationSpec,
                    snapshot: _PerturbationSnapshot) -> Optional[_PerturbationSnapshot]:
        # Snapshot original pose (before perturbation)
        original_pose = _read_object_pose(sim, spec.obj_name)
        if original_pose is None:
            print(f"[TEMPORAL] WARN: Cannot read pose for '{spec.obj_name}'. Skipping move.")
            return None
        snapshot.original_poses[spec.obj_name] = original_pose

        # Compute new position
        if spec.delta_xy is not None:
            dx, dy = spec.delta_xy
        else:
            dx = np.random.uniform(-spec.max_move_m, spec.max_move_m)
            dy = np.random.uniform(-spec.max_move_m, spec.max_move_m)

        new_x = original_pose[0] + dx
        new_y = original_pose[1] + dy

        ok = _place_object_at_xy(sim, spec.obj_name, new_x, new_y)
        if ok:
            # Record the pose the engine just wrote so we can detect robot movement later
            applied_pose = _read_object_pose(sim, spec.obj_name)
            snapshot.applied_poses[spec.obj_name] = applied_pose

            msg = (f"step {spec.start_step}: MOVE '{spec.obj_name}' "
                   f"by ({dx:+.4f}, {dy:+.4f}) m → "
                   f"({new_x:.4f}, {new_y:.4f}) "
                   f"[reverts at step {spec.end_step + 1} unless robot moves object]")
            print(f"[TEMPORAL] {msg}")
            self._log.append(msg)
            return snapshot
        return None

    def _apply_color(self, sim, spec: TemporalPerturbationSpec,
                     snapshot: _PerturbationSnapshot) -> Optional[_PerturbationSnapshot]:
        # Snapshot original colors
        original_colors = _read_object_colors(sim, spec.obj_name)
        if original_colors is None:
            print(f"[TEMPORAL] WARN: Cannot read colors for '{spec.obj_name}'. Skipping color.")
            return None
        snapshot.original_colors[spec.obj_name] = original_colors

        # Pick color
        color_name = spec.color if spec.color else np.random.choice(list(COLOR_PALETTE.keys()))
        rgba = COLOR_PALETTE[color_name]

        ok = _write_object_colors(sim, spec.obj_name, rgba)
        if ok:
            # Color perturbations have no position component; no applied_pose needed.
            msg = (f"step {spec.start_step}: COLOR '{spec.obj_name}' → {color_name} "
                   f"{tuple(rgba)} [reverts at step {spec.end_step + 1}]")
            print(f"[TEMPORAL] {msg}")
            self._log.append(msg)
            return snapshot
        return None

    def _apply_distractor(self, sim, spec: TemporalPerturbationSpec,
                           snapshot: _PerturbationSnapshot) -> Optional[_PerturbationSnapshot]:
        dname = spec.distractor_obj_name
        # Snapshot: record that it was parked (pose = off-screen)
        parked_pose = np.concatenate([_OFFSCREEN_XYZ, _IDENTITY_QUAT])
        snapshot.original_poses[dname] = parked_pose

        # Determine target position
        if spec.distractor_xy is not None:
            tx, ty = spec.distractor_xy
        else:
            tx = np.random.uniform(-0.2, 0.2)
            ty = np.random.uniform(-0.2, 0.2)

        ok = _place_object_at_xy(sim, dname, tx, ty, z_override=0.02)
        if ok:
            # Record the pose the engine wrote for the distractor
            applied_pose = _read_object_pose(sim, dname)
            snapshot.applied_poses[dname] = applied_pose

            msg = (f"step {spec.start_step}: DISTRACTOR '{dname}' "
                   f"appears at ({tx:.4f}, {ty:.4f}) "
                   f"[reverts at step {spec.end_step + 1} unless robot moves object]")
            print(f"[TEMPORAL] {msg}")
            self._log.append(msg)
            return snapshot
        return None

    def _apply_replace(self, sim, spec: TemporalPerturbationSpec,
                        snapshot: _PerturbationSnapshot) -> Optional[_PerturbationSnapshot]:
        # Snapshot original pose of target object (before perturbation)
        original_pose = _read_object_pose(sim, spec.obj_name)
        if original_pose is None:
            print(f"[TEMPORAL] WARN: Cannot read pose for '{spec.obj_name}'. Skipping replace.")
            return None
        snapshot.original_poses[spec.obj_name] = original_pose
        # Replacement was pre-parked
        snapshot.original_poses[spec.replacement_obj_name] = np.concatenate(
            [_OFFSCREEN_XYZ, _IDENTITY_QUAT])

        # Park original, bring replacement to same position
        _park_object(sim, spec.obj_name)
        ok = _place_object_at_xy(sim, spec.replacement_obj_name,
                                  original_pose[0], original_pose[1],
                                  z_override=original_pose[2])
        if ok:
            # Record the pose written for the replacement object
            applied_pose = _read_object_pose(sim, spec.replacement_obj_name)
            snapshot.applied_poses[spec.replacement_obj_name] = applied_pose

            msg = (f"step {spec.start_step}: REPLACE '{spec.obj_name}' "
                   f"with '{spec.replacement_obj_name}' "
                   f"[reverts at step {spec.end_step + 1} unless robot moves replacement]")
            print(f"[TEMPORAL] {msg}")
            self._log.append(msg)
            return snapshot
        return None

    # ------------------------------------------------------------------
    # Internal: revert
    # ------------------------------------------------------------------

    def _revert(self, sim, spec_idx: int):
        """
        Revert a perturbation at the end of its window.

        Position revert logic
        ---------------------
        For each object whose pose we would restore, we first check whether the
        robot moved it during the window.  We do this by comparing the object's
        *current* 3-D position to the position the engine wrote at start_step
        (the "applied pose") using a full XYZ Euclidean distance.  If the
        displacement exceeds spec.robot_move_threshold_m we skip restoring that
        object's position so the robot's work is preserved.

        This check applies uniformly to ALL object types, including distractors.
        If the robot picked up and repositioned a distractor during the window,
        it will remain where the robot left it.

        Color reverts are always applied regardless of robot movement, since a
        color change is a visual property independent of where the robot placed
        the object.
        """
        snapshot = self._snapshots.get(spec_idx)
        if snapshot is None:
            return
        spec = snapshot.spec

        # ---- Restore poses (with robot-movement check) ----
        for obj_name, original_pose in snapshot.original_poses.items():
            applied_pose = snapshot.applied_poses.get(obj_name)
            current_pose = _read_object_pose(sim, obj_name)

            # Determine whether to skip position revert.
            # Robot-movement detection applies to ALL object types uniformly,
            # including distractors.
            if _robot_moved_object(
                current_pose, applied_pose, spec.robot_move_threshold_m
            ):
                # Robot moved this object during the window — leave it where it is.
                if current_pose is not None and applied_pose is not None:
                    xyz_shift = np.linalg.norm(current_pose[:3] - applied_pose[:3])
                else:
                    xyz_shift = float("nan")
                msg = (f"step {spec.end_step + 1}: SKIP pose revert of '{obj_name}' "
                       f"(robot moved it {xyz_shift:.4f} m during window "
                       f"> threshold {spec.robot_move_threshold_m:.4f} m)")
                print(f"[TEMPORAL] {msg}")
                self._log.append(msg)
            else:
                # Robot did not move the object — restore original pose.
                _write_object_pose(sim, obj_name, original_pose)
                print(f"[TEMPORAL] step {spec.end_step + 1}: REVERT pose of '{obj_name}' "
                      f"(robot did not move object)")

        # ---- Restore colors (always, independent of robot movement) ----
        for obj_name, geom_colors in snapshot.original_colors.items():
            body_name = _find_object_body_name(sim, obj_name)
            if body_name:
                for gid, rgba in geom_colors.items():
                    sim.model.geom_rgba[gid] = rgba
            print(f"[TEMPORAL] step {spec.end_step + 1}: REVERT color of '{obj_name}'")

        msg = (f"step {spec.end_step + 1}: REVERTED {spec.pert_type} "
               f"on '{spec.obj_name or spec.distractor_obj_name}'")
        self._log.append(msg)

    # ------------------------------------------------------------------
    # Internal: sim access
    # ------------------------------------------------------------------

    @staticmethod
    def _get_sim(env):
        """
        Get the MuJoCo sim from a LIBERO OffScreenRenderEnv / ControlEnv.

        ControlEnv exposes a `sim` property that returns `self.env.sim`, where
        `self.env` is the underlying robosuite task environment.  So `env.sim`
        is always the correct path — no need to dig further.

        We still fall back to `env.env.sim` for safety (e.g. if someone passes
        the raw robosuite env instead of the ControlEnv wrapper).
        """
        # Primary path: ControlEnv.sim property → self.env.sim
        try:
            sim = env.sim
            if sim is not None:
                return sim
        except AttributeError:
            pass

        # Fallback: raw robosuite env passed directly
        try:
            sim = env.env.sim
            if sim is not None:
                return sim
        except AttributeError:
            pass

        print("[TEMPORAL] WARN: Could not locate 'sim' on env. "
              "Expected a ControlEnv (OffScreenRenderEnv) instance.")
        return None


# ---------------------------------------------------------------------------
# BDDL helper: pre-declare hidden objects for distractor/replace
# ---------------------------------------------------------------------------

def add_hidden_objects_to_bddl(bddl_text: str, hidden_objects: List[Tuple[str, str]],
                                 target_workspace: str = "kitchen_table") -> str:
    """
    Add objects to a BDDL file that will be pre-spawned off-screen, ready for
    temporal distractor or replace perturbations.

    Args:
        bddl_text: Original BDDL content
        hidden_objects: List of (obj_instance_name, obj_type_name) tuples
                        e.g. [("moka_pot_999", "moka_pot")]
        target_workspace: Workspace name for the init region target

    Returns:
        Modified BDDL text with hidden objects declared and initialized off-screen.

    Note:
        The objects are placed far off-screen (100, 100) in the :regions block
        so they exist in the sim but are invisible at episode start.
        TemporalPerturbationManager.reset() also parks them on env.reset().
    """
    OFFSCREEN_RANGES = "99.9 99.9 100.1 100.1"

    for obj_name, obj_type in hidden_objects:
        region_name = f"{obj_name}_hidden_region"

        # --- Add to :objects ---
        obj_section = re.search(r"(\(:objects\s*\n)((?:.*\n)*?)(\s*\))", bddl_text)
        if obj_section:
            obj_content = obj_section.group(2)
            last_line = obj_content.rstrip().split('\n')[-1] if obj_content.strip() else ""
            indent = re.match(r'^(\s*)', last_line).group(1) if last_line else "    "
            new_obj_content = obj_content + f"{indent}{obj_name} - {obj_type}\n"
            bddl_text = (bddl_text[:obj_section.start()] +
                         obj_section.group(1) + new_obj_content +
                         obj_section.group(3) + bddl_text[obj_section.end():])

        # --- Add region definition ---
        region_def = f"""      ({region_name}
          (:target {target_workspace})
          (:ranges (
              ({OFFSCREEN_RANGES})
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )"""

        regions_start = bddl_text.find("(:regions")
        if regions_start == -1:
            print(f"[HIDDEN] WARN: Could not find :regions block. Skipping {obj_name}.")
            continue
        depth = 0
        regions_end = -1
        for i in range(regions_start, len(bddl_text)):
            if bddl_text[i] == "(":
                depth += 1
            elif bddl_text[i] == ")":
                depth -= 1
                if depth == 0:
                    regions_end = i
                    break
        bddl_text = (bddl_text[:regions_end] +
                     "\n" + region_def + "\n" +
                     bddl_text[regions_end:])

        # --- Add to :init ---
        init_section = re.search(r"(\(:init\s*\n)((?:.*\n)*?)(\s*\))", bddl_text)
        if init_section:
            init_content = init_section.group(2)
            last_line = init_content.rstrip().split('\n')[-1] if init_content.strip() else ""
            indent = re.match(r'^(\s*)', last_line).group(1) if last_line else "    "
            new_init_content = (init_content +
                                f"{indent}(On {obj_name} "
                                f"{target_workspace}_{region_name})\n")
            bddl_text = (bddl_text[:init_section.start()] +
                         init_section.group(1) + new_init_content +
                         init_section.group(3) + bddl_text[init_section.end():])

        print(f"[HIDDEN] Added hidden object '{obj_name}' ({obj_type}) to BDDL at off-screen position")

    return bddl_text


# ---------------------------------------------------------------------------
# Config-driven factory: build specs from YAML config dict
# ---------------------------------------------------------------------------

def specs_from_config(temporal_config: Dict[str, Any]) -> List[TemporalPerturbationSpec]:
    """
    Build a list of TemporalPerturbationSpec from a config dict.

    Expected YAML structure (under 'temporal_perturbations' key):
        temporal_perturbations:
          - type: move
            obj_name: akita_black_bowl_1
            start_step: 50
            end_step: 150
            delta_xy: [0.05, 0.0]         # optional; random if omitted
            max_move_m: 0.05              # used only when delta_xy is omitted
            robot_move_threshold_m: 0.01  # optional; default 0.0 m (any movement counts)

          - type: color
            obj_name: wine_bottle_1
            start_step: 80
            end_step: 200
            color: red                    # optional; random if omitted

          - type: distractor
            distractor_obj_name: moka_pot_999
            start_step: 100
            end_step: 300
            distractor_xy: [0.1, -0.1]   # optional; random if omitted

          - type: replace
            obj_name: wine_bottle_1
            replacement_obj_name: milk_777
            start_step: 120
            end_step: 250
            robot_move_threshold_m: 0.02  # optional; default 0.0 m (any movement counts)
    """
    specs = []
    for entry in temporal_config:
        ptype = entry["type"]
        start = entry["start_step"]
        end = entry["end_step"]
        threshold = entry.get("robot_move_threshold_m", ROBOT_MOVE_THRESHOLD_M)

        if ptype == "move":
            delta_xy = tuple(entry["delta_xy"]) if "delta_xy" in entry else None
            specs.append(TemporalPerturbationSpec(
                pert_type="move",
                start_step=start,
                end_step=end,
                obj_name=entry["obj_name"],
                delta_xy=delta_xy,
                max_move_m=entry.get("max_move_m", 0.05),
                robot_move_threshold_m=threshold,
            ))

        elif ptype == "color":
            specs.append(TemporalPerturbationSpec(
                pert_type="color",
                start_step=start,
                end_step=end,
                obj_name=entry["obj_name"],
                color=entry.get("color"),
                # No threshold needed for color-only perturbations
            ))

        elif ptype == "distractor":
            dxy = tuple(entry["distractor_xy"]) if "distractor_xy" in entry else None
            specs.append(TemporalPerturbationSpec(
                pert_type="distractor",
                start_step=start,
                end_step=end,
                distractor_obj_name=entry["distractor_obj_name"],
                distractor_xy=dxy,
                robot_move_threshold_m=threshold,
            ))

        elif ptype == "replace":
            specs.append(TemporalPerturbationSpec(
                pert_type="replace",
                start_step=start,
                end_step=end,
                obj_name=entry["obj_name"],
                replacement_obj_name=entry["replacement_obj_name"],
                robot_move_threshold_m=threshold,
            ))

        else:
            raise ValueError(f"Unknown temporal perturbation type: '{ptype}'")

    return specs