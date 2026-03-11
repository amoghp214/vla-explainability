import os
import math
import random
import datetime
import re
from collections import Counter
from typing import Dict, Optional, List, Tuple, Union

# Small vertical gap (m) between stacked objects to avoid clipping.
Z_STACK_GAP_M: float = 0.002  # 2 mm

# Conservative approximate heights (m) for common LIBERO objects.
# Used to estimate stacking Z at BDDL-generation time when no sim is available.
_APPROX_OBJECT_HALF_HEIGHT_M: Dict[str, float] = {
    "akita_black_bowl":    0.04,
    "white_yellow_mug":    0.06,
    "wine_bottle":         0.15,
    "plate":               0.015,
    "alphabet_soup":       0.07,
    "cream_cheese":        0.04,
    "tomato_sauce":        0.08,
    "ketchup":             0.10,
    "butter":              0.03,
    "milk":                0.10,
    "chocolate_pudding":   0.05,
    "orange_juice":        0.09,
    "bbq_sauce":           0.10,
    "salad_dressing":      0.09,
    "black_book":          0.015,
    "moka_pot":            0.09,
    "chefmate_8_frypan":   0.04,
    # default for unknown types
    "_default":            0.05,
}

# Sub-region keywords that indicate an articulated cavity (drawer/door interior)
_CAVITY_REGION_KEYWORDS = ("bottom_region", "middle_region", "top_region", "drawer")

# Sentinel used in z_overrides to mark an entry that needs sim-based Z resolution.
# Format stored in z_overrides for cavity entries:
#   z_overrides[obj_name] = (_CAVITY_SENTINEL, region_name, cavity_cx, cavity_cy)
# resolve_z_overrides(sim, z_overrides) replaces these with final (cx, cy, z) tuples.
_CAVITY_SENTINEL = "__cavity__"

# --------------------------
# Utilities
# --------------------------

def read_bddl(path):
    with open(path, "r") as f:
        return f.read()

def save_bddl(content, base_name="perturbed_scene", folder="perturbed_bddl"):
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.bddl"
    path = os.path.join(folder, filename)
    with open(path, "w") as f:
        f.write(content)
    print(f"[INFO] Saved perturbed BDDL → {path}")
    return path

# --------------------------
# BDDL Structure Extraction
# --------------------------

def extract_declared_objects(bddl_text):
    """Extract all objects declared in :objects section.
    Supports both "inst - category" and "inst1 inst2 ... - category" per line.
    """
    obj_pattern = r":objects\s*\n((?:.*\n)*?)\s*\)"
    match = re.search(obj_pattern, bddl_text, re.MULTILINE)
    if not match:
        return set()

    objects = set()
    content = match.group(1)
    for line in content.strip().split("\n"):
        # Line format: "inst - category" or "inst1 inst2 inst3 - category"
        if " - " not in line:
            continue
        left, _category = line.strip().rsplit(" - ", 1)
        for part in left.split():
            if part and part != "-":
                objects.add(part)
    return objects

def extract_fixture_objects(bddl_text):
    """Extract all fixtures declared in :fixtures section."""
    fixture_pattern = r":fixtures\s*\n((?:.*\n)*?)\s*\)"
    match = re.search(fixture_pattern, bddl_text, re.MULTILINE)
    if not match:
        return set()
    fixtures = set()
    for line in match.group(1).strip().split('\n'):
        m = re.match(r'\s*(\w+)\s*-\s*(\w+)', line)
        if m:
            fixtures.add(m.group(1))
    return fixtures

# --------------------------
# Region parser
# --------------------------

def find_region_blocks(bddl_text):
    """
    Return dict {region_name: (start_idx, end_idx)} for all *_init_region blocks.
    """
    region_blocks = {}
    regions_start = bddl_text.find("(:regions")
    if regions_start == -1:
        return region_blocks

    depth, regions_end = 0, -1
    for i in range(regions_start, len(bddl_text)):
        if bddl_text[i] == "(":  depth += 1
        elif bddl_text[i] == ")":
            depth -= 1
            if depth == 0:
                regions_end = i + 1
                break
    if regions_end == -1:
        return region_blocks

    for m in re.compile(r"\((\w+_init_region)\b").finditer(bddl_text[regions_start:regions_end]):
        actual_start = regions_start + m.start()
        region_name = m.group(1)

        check_depth = 0
        for i in range(regions_start, actual_start):
            if bddl_text[i] == "(":  check_depth += 1
            elif bddl_text[i] == ")": check_depth -= 1
        if check_depth != 1:
            continue

        depth = 0
        for i in range(actual_start, regions_end):
            if bddl_text[i] == "(":  depth += 1
            elif bddl_text[i] == ")":
                depth -= 1
                if depth == 0:
                    region_blocks[region_name] = (actual_start, i + 1)
                    break
    return region_blocks

def parse_object_region_map(bddl_text, region_blocks):
    """Returns dict: {object_name: region_name}"""
    pattern = re.compile(r"\(On\s+(\w+)\s+([\w_]+_init_region)\)")
    obj_region_map = {}
    available_regions = set(region_blocks.keys())
    for match in pattern.finditer(bddl_text):
        obj_name = match.group(1)
        full_region_name = match.group(2)
        if full_region_name in available_regions:
            obj_region_map[obj_name] = full_region_name
        else:
            for r in available_regions:
                if full_region_name.endswith(r):
                    obj_region_map[obj_name] = r
                    break
            else:
                obj_region_map[obj_name] = full_region_name
    return obj_region_map

# --------------------------
# Workspace Detection
# --------------------------

def extract_target_workspace(bddl_text):
    matches = re.compile(r"\(:target\s+(\w+)\)").findall(bddl_text)
    if matches:
        return Counter(matches).most_common(1)[0][0]
    for kw, ws in (("kitchen", "kitchen_table"), ("living_room", "living_room_table"),
                   ("study", "study_table")):
        if kw in bddl_text.lower():
            return ws
    return "kitchen_table"

# --------------------------
# Attribute support check
# --------------------------

def supports_attributes(region_name, bddl_text):
    fixtures = extract_fixture_objects(bddl_text)
    for fixture in fixtures:
        if fixture in region_name:
            return False
    return True

# --------------------------
# Range utilities
# --------------------------

RANGES_PATTERN = re.compile(
    r":ranges\s*\(\s*\(\s*([-+]?[0-9]*\.?[0-9]+\s+[-+]?[0-9]*\.?[0-9]+\s+"
    r"[-+]?[0-9]*\.?[0-9]+\s+[-+]?[0-9]*\.?[0-9]+)\s*\)",
    re.DOTALL,
)

def _collapse_ranges_to_center(coords):
    x_min, y_min, x_max, y_max = coords
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    return [cx, cy, cx, cy]

def _center_with_fixed_extent(cx, cy, extent):
    half = extent / 2
    return [cx - half, cy - half, cx + half, cy + half]

def fix_init_ranges(bddl_text, init_object_range_m=0.0, max_init_range_m=0.001):
    region_blocks = find_region_blocks(bddl_text)
    result = bddl_text
    for region_name, (start, end) in region_blocks.items():
        block = result[start:end]
        match = RANGES_PATTERN.search(block)
        if match:
            coords = list(map(float, match.group(1).split()))
            if init_object_range_m <= 0:
                collapsed = _collapse_ranges_to_center(coords)
                new_coords = _center_with_fixed_extent(collapsed[0], collapsed[1], max_init_range_m)
            else:
                new_coords = _collapse_ranges_to_center(coords)
                half = init_object_range_m / 2
                new_coords = [new_coords[0]-half, new_coords[1]-half,
                               new_coords[0]+half, new_coords[1]+half]
            new_range = " ".join(f"{x:.6g}" for x in new_coords)
            block = block[:match.start(1)] + new_range + block[match.end(1):]
            result = result[:start] + block + result[end:]
    return result

# --------------------------
# BDDL Collision Detection & Stacking
#
# BDDL :ranges are purely XY (table plane).  There is no Z in the region spec.
# When a perturbed object's intended XY centre overlaps another region:
#
#   - OPEN DRAWER / CAVITY: detected by parsing region names that match
#     known articulated-fixture sub-region patterns (e.g. "bottom_region",
#     "middle_region" on a cabinet).  If the perturbed object's XY overlaps
#     such a region AND that region is declared as Open in :init, the object
#     is placed at the cavity's XY centre.  The z_overrides side-channel dict
#     records the required Z so the env loader can call
#     _write_object_pose(sim, obj, [cx, cy, cavity_z, ...]) after env.reset().
#
#   - PLAIN OBJECT STACK: if the overlapping region belongs to a movable
#     object (not a fixture), the z_overrides dict records a stacking Z:
#         z = other_approx_z + other_half_h + perturbed_half_h + Z_STACK_GAP_M
#     where the heights are conservative estimates based on object type names
#     (a lookup table _APPROX_OBJECT_HEIGHT_M).
#
# The BDDL XY is left at the intended position in both cases (we do NOT nudge
# to a different XY spot).  The z_overrides dict is the contract between this
# module and the env loader.
# --------------------------


def _object_half_height_from_name(obj_name: str) -> float:
    """Estimate half-height (m) from the object instance name."""
    for key, h in _APPROX_OBJECT_HALF_HEIGHT_M.items():
        if key == "_default":
            continue
        if key in obj_name:
            return h
    return _APPROX_OBJECT_HALF_HEIGHT_M["_default"]


def _region_center(block: str) -> Optional[Tuple[float, float]]:
    """Return (cx, cy) from a region's :ranges block, or None."""
    m = RANGES_PATTERN.search(block)
    if not m:
        return None
    coords = list(map(float, m.group(1).split()))
    return (coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2


def _region_radius(block: str) -> float:
    """Approximate XY radius from a region's :ranges half-extents."""
    m = RANGES_PATTERN.search(block)
    if not m:
        return 0.05
    coords = list(map(float, m.group(1).split()))
    hx = (coords[2] - coords[0]) / 2
    hy = (coords[3] - coords[1]) / 2
    return max(math.sqrt(hx**2 + hy**2), 0.05)


def _is_open_in_init(bddl_text: str, region_ref: str) -> bool:
    """
    Return True if there is an (Open <region_ref>) statement in :init.
    This indicates an articulated cavity (drawer/door) that is open at
    episode start and can receive objects.
    """
    pattern = re.compile(rf"\(Open\s+{re.escape(region_ref)}\)")
    return bool(pattern.search(bddl_text))


def _check_and_resolve_bddl_collision(
    bddl_text: str,
    region_blocks: dict,
    obj_region_map: dict,
    perturbed_obj: str,
    perturbed_region: str,
    target_cx: float,
    target_cy: float,
    z_overrides: dict,
    target_workspace: str,
):
    """
    Check whether placing *perturbed_obj*'s region at (target_cx, target_cy)
    would collide with any other region.  If it does, record the required
    information in *z_overrides* so the env loader can stack/place correctly
    after env.reset() when the sim is available.

    Resolution priority (mirrors the live-sim engine):

    1. OPEN CAVITY: if the overlapping region is a known articulated cavity
       that is declared Open in :init, record a sentinel in z_overrides.
       The Z is NOT hardcoded here — it is computed from real sim geometry
       by resolve_z_overrides(sim, z_overrides) after env.reset().
       → z_overrides[perturbed_obj] = (_CAVITY_SENTINEL, region_name, cavity_cx, cavity_cy)
       The XY is set to the cavity's BDDL centre so the object spawns inside.

    2. PLAIN OBJECT STACK: record the stacking Z estimated from object names.
       → z_overrides[perturbed_obj] = (target_cx, target_cy, stack_z)
       These estimates are replaced with sim-accurate values by
       resolve_z_overrides() when available.

    3. No collision → no z_override recorded.

    Returns:
        (final_cx, final_cy) — possibly updated XY for writing to the BDDL.
    """
    perturbed_radius = _region_radius(
        bddl_text[region_blocks[perturbed_region][0]: region_blocks[perturbed_region][1]]
        if perturbed_region in region_blocks else "")
    perturbed_half_h = _object_half_height_from_name(perturbed_obj)

    # Reverse-map: region_name → object_name
    region_to_obj = {v: k for k, v in obj_region_map.items()}

    for rname, (rstart, rend) in region_blocks.items():
        if rname == perturbed_region:
            continue
        other_block = bddl_text[rstart:rend]
        other_center = _region_center(other_block)
        if other_center is None:
            continue
        other_radius = _region_radius(other_block)
        dist = math.sqrt((target_cx - other_center[0])**2 + (target_cy - other_center[1])**2)
        if dist >= perturbed_radius + other_radius:
            continue  # no overlap

        # ----- Collision detected -----

        # 1. Open articulated cavity — defer Z to sim at load time
        is_cavity = any(kw in rname for kw in _CAVITY_REGION_KEYWORDS)
        full_ref = f"{target_workspace}_{rname}"
        if is_cavity and _is_open_in_init(bddl_text, full_ref):
            cavity_cx, cavity_cy = other_center
            # Store sentinel: Z will be resolved from sim in resolve_z_overrides()
            z_overrides[perturbed_obj] = (_CAVITY_SENTINEL, rname, cavity_cx, cavity_cy)
            print(f"[COLLISION] '{perturbed_obj}' overlaps open cavity '{rname}'. "
                  f"XY set to cavity centre ({cavity_cx:.4f},{cavity_cy:.4f}). "
                  f"Z will be resolved from sim after env.reset().")
            return cavity_cx, cavity_cy  # move BDDL XY into the cavity

        # 2. Plain object stacking — estimate Z from object names
        other_obj = region_to_obj.get(rname)
        other_half_h = (_object_half_height_from_name(other_obj)
                        if other_obj else _APPROX_OBJECT_HALF_HEIGHT_M["_default"])
        other_approx_z = other_half_h  # approximate: object centre ≈ half-height above table
        stack_z = other_approx_z + other_half_h + perturbed_half_h + Z_STACK_GAP_M
        z_overrides[perturbed_obj] = (target_cx, target_cy, stack_z)
        print(f"[COLLISION] '{perturbed_obj}' overlaps '{other_obj or rname}'. "
              f"Estimated stack z={stack_z:.4f} m (will be refined by sim if available).")
        return target_cx, target_cy

    # No collision
    return target_cx, target_cy


# ---------------------------------------------------------------------------
# Sim-based Z resolution — call this after env.reset() with the live sim
# ---------------------------------------------------------------------------

def resolve_z_overrides(sim, z_overrides: dict) -> dict:
    """
    Replace estimated / sentinel Z values in *z_overrides* with accurate
    values read from the live MuJoCo sim.  Call this AFTER env.reset().

    For cavity sentinels
    --------------------
    Scans all hinge/slide joints in the sim.  For each joint that is open
    (|qpos| > OPEN_JOINT_THRESHOLD_RAD), computes the world-frame centre of
    all geoms on that body using sim.data.geom_xpos (set by sim.forward()).
    The cavity Z is taken as the mean world-Z of those geom centres — i.e. the
    actual interior centre of the open drawer/door as the sim sees it, with
    no hardcoded constants.

    For plain-stack estimates
    -------------------------
    Reads the collider object's actual pose from qpos and its geom sizes to
    compute an accurate stacking Z, replacing the name-based estimate.

    Args:
        sim         : MuJoCo sim (mujoco_py or dm_control style).
                      Must expose .model and .data as standard MuJoCo bindings.
        z_overrides : dict as returned by apply_perturbations():
                        {obj_name: (_CAVITY_SENTINEL, region_name, cx, cy)}
                        or {obj_name: (cx, cy, estimated_z)}

    Returns:
        Resolved dict {obj_name: (cx, cy, z)} with all Z values computed from
        actual sim geometry.  Entries that cannot be resolved (body not found)
        are left with their estimated Z and a warning is printed.
    """
    # Import helpers from temporal engine if available; otherwise use local fallbacks.
    try:
        from temporal_perturbation_engine import (
            _find_open_cavities,
            _object_z_half_height,
            _read_object_pose,
            _object_xy_radius,
            OPEN_JOINT_THRESHOLD_RAD,
            Z_STACK_GAP_M as _TEMPORAL_Z_GAP,
        )
        _temporal_available = True
    except ImportError:
        _temporal_available = False

    resolved = {}

    # Pre-compute open cavities once
    open_cavities = _find_open_cavities(sim) if _temporal_available else []

    for obj_name, entry in z_overrides.items():

        # ---- Cavity sentinel ----
        if isinstance(entry, tuple) and len(entry) == 4 and entry[0] == _CAVITY_SENTINEL:
            _, region_name, cx, cy = entry

            matched_cavity = None
            if _temporal_available:
                # Find the open cavity whose body name best matches the region name
                for cavity in open_cavities:
                    # The cavity body name is usually a substring of the region name
                    # e.g. region "white_cabinet_1_bottom_region" ↔ body "white_cabinet_1_bottom"
                    if any(part in cavity.body_name for part in region_name.split("_") if len(part) > 3):
                        matched_cavity = cavity
                        break
                    # Fallback: check if the cavity XY centre is close to the region BDDL centre
                    if (abs(cavity.centre_xyz[0] - cx) < 0.15
                            and abs(cavity.centre_xyz[1] - cy) < 0.15):
                        matched_cavity = cavity
                        break

            if matched_cavity is not None:
                # Use the sim-computed world-frame Z centre of the cavity interior
                z = float(matched_cavity.centre_xyz[2])
                resolved[obj_name] = (cx, cy, z)
                print(f"[RESOLVE] '{obj_name}' cavity Z from sim: "
                      f"body='{matched_cavity.body_name}' → z={z:.4f} m")
            else:
                # Could not match — keep cx/cy but warn; Z stays as estimated
                # (entry[3] doesn't exist for sentinel, so use a safe default)
                fallback_z = 0.08
                resolved[obj_name] = (cx, cy, fallback_z)
                print(f"[RESOLVE] WARN: '{obj_name}' cavity body not found in sim for "
                      f"region '{region_name}'. Using fallback z={fallback_z:.4f} m.")

        # ---- Plain stack estimate ----
        elif isinstance(entry, tuple) and len(entry) == 3:
            cx, cy, est_z = entry
            if not _temporal_available:
                resolved[obj_name] = (cx, cy, est_z)
                continue

            # Try to find the object that the perturbed obj is stacked on top of.
            # We scan scene objects whose XY is close to (cx, cy).
            best_top_z = None
            for jid in range(sim.model.njnt):
                if sim.model.jnt_type[jid] != 0:
                    continue
                jname = sim.model.joint_id2name(jid)
                candidate = jname
                for sfx in ("_joint0", "_freejoint", ":joint"):
                    if candidate.endswith(sfx):
                        candidate = candidate[:-len(sfx)]
                        break
                if candidate == obj_name:
                    continue
                addr = sim.model.jnt_qposadr[jid]
                pose = sim.data.qpos[addr: addr + 3]
                if max(abs(pose[0]), abs(pose[1]), abs(pose[2])) > 50.0:
                    continue  # parked
                xy_dist = math.sqrt((pose[0] - cx)**2 + (pose[1] - cy)**2)
                if xy_dist > 0.15:
                    continue
                try:
                    other_z_half = _object_z_half_height(sim, candidate)
                    top_z = float(pose[2]) + other_z_half
                    if best_top_z is None or top_z > best_top_z:
                        best_top_z = top_z
                except Exception:
                    pass

            if best_top_z is not None:
                # Compute perturbed object's half-height from sim
                try:
                    perturbed_z_half = _object_z_half_height(sim, obj_name)
                except Exception:
                    perturbed_z_half = _APPROX_OBJECT_HALF_HEIGHT_M["_default"]
                z = best_top_z + perturbed_z_half + _TEMPORAL_Z_GAP
                print(f"[RESOLVE] '{obj_name}' stack Z from sim: z={z:.4f} m "
                      f"(was estimated {est_z:.4f} m)")
            else:
                z = est_z  # keep estimate if no collider found in sim
                print(f"[RESOLVE] '{obj_name}' kept estimated stack z={z:.4f} m "
                      f"(no matching collider found in sim)")
            resolved[obj_name] = (cx, cy, z)

        else:
            print(f"[RESOLVE] WARN: Unrecognised z_overrides entry for '{obj_name}': {entry}")

    return resolved


# --------------------------
# Generate move spec
# --------------------------

def generate_move_spec_dict(bddl_text, object_names, max_move_m=0.05):
    region_blocks = find_region_blocks(bddl_text)
    obj_region_map = parse_object_region_map(bddl_text, region_blocks)
    move_spec = {}
    for obj_name in object_names:
        region_name = obj_region_map.get(obj_name)
        if not region_name or region_name not in region_blocks:
            print(f"[WARN] generate_move_spec_dict: region not found for {obj_name}, skipping")
            continue
        start, end = region_blocks[region_name]
        block = bddl_text[start:end]
        match = RANGES_PATTERN.search(block)
        if not match:
            continue
        coords = list(map(float, match.group(1).split()))
        cx = (coords[0] + coords[2]) / 2
        cy = (coords[1] + coords[3]) / 2
        new_x = round(cx + random.uniform(-max_move_m, max_move_m), 4)
        new_y = round(cy + random.uniform(-max_move_m, max_move_m), 4)
        move_spec[obj_name] = [new_x, new_y]
    return {"move": move_spec} if move_spec else {}


# --------------------------
# Perturbation functions
# --------------------------

def move_object(bddl_text, obj_name, obj_region_map, region_blocks,
                init_object_range_m=0.0, max_move_m=0.05, max_init_range_m=0.001,
                center_override=None, z_overrides=None, target_workspace="kitchen_table"):
    """
    Move object's init region centre to a new XY position.

    Collision is resolved by stacking or placing in an open cavity (see
    _check_and_resolve_bddl_collision).  The XY in the BDDL is written to the
    resolved position.  If a Z adjustment is needed (stacking / cavity), the
    required (cx, cy, z) is recorded in *z_overrides* for the env loader.

    Args:
        z_overrides : dict that will be populated with
                      {obj_name: (cx, cy, z)} entries when a collision requires
                      a Z adjustment.  Pass an empty dict {} and inspect it
                      after apply_perturbations() to get all required overrides.
    """
    if z_overrides is None:
        z_overrides = {}

    region_name = obj_region_map.get(obj_name)
    if not region_name or region_name not in region_blocks:
        print(f"[WARN] Region not found for {obj_name} (looking for '{region_name}')")
        return bddl_text

    start, end = region_blocks[region_name]
    block = bddl_text[start:end]
    match = RANGES_PATTERN.search(block)
    if not match:
        return bddl_text

    coords = list(map(float, match.group(1).split()))
    orig_cx = (coords[0] + coords[2]) / 2
    orig_cy = (coords[1] + coords[3]) / 2

    if center_override is not None:
        desired_cx, desired_cy = center_override[0], center_override[1]
        print(f"[MOVE] {obj_name} centre set to ({desired_cx:.4f},{desired_cy:.4f}) from spec")
    else:
        dx = round(random.uniform(-max_move_m, max_move_m), 4)
        dy = round(random.uniform(-max_move_m, max_move_m), 4)
        desired_cx = orig_cx + dx
        desired_cy = orig_cy + dy
        print(f"[MOVE] {obj_name} Δ=({dx:+.4f},{dy:+.4f}) → "
              f"({desired_cx:.4f},{desired_cy:.4f}), init_range={init_object_range_m}m")

    # Collision resolution (updates desired_cx/cy if cavity placement chosen)
    final_cx, final_cy = _check_and_resolve_bddl_collision(
        bddl_text, region_blocks, obj_region_map,
        obj_name, region_name, desired_cx, desired_cy,
        z_overrides, target_workspace,
    )

    if init_object_range_m <= 0:
        new_coords = _center_with_fixed_extent(final_cx, final_cy, max_init_range_m)
    else:
        half = init_object_range_m / 2
        new_coords = [final_cx-half, final_cy-half, final_cx+half, final_cy+half]

    new_range = " ".join(f"{x:.6g}" for x in new_coords)
    block = block[:match.start(1)] + new_range + block[match.end(1):]
    bddl_text = bddl_text[:start] + block + bddl_text[end:]
    return bddl_text


def reorient_object(bddl_text, obj_name, obj_region_map, region_blocks):
    region_name = obj_region_map.get(obj_name)
    if not region_name or region_name not in region_blocks:
        print(f"[WARN] Region not found for {obj_name} (looking for '{region_name}')")
        return bddl_text

    start, end = region_blocks[region_name]
    block = bddl_text[start:end]
    match = re.search(
        r":yaw_rotation\s*\(\s*\(\s*([-+]?[0-9]*\.?[0-9]+(?:\s+[-+]?[0-9]*\.?[0-9]+)?)\s*\)",
        block, re.DOTALL)
    if match:
        vals = list(map(float, match.group(1).split()))
        rotation_type = random.choice(["clockwise", "anticlockwise"])
        angle = round(random.uniform(5, 30), 2)
        delta = angle if rotation_type == "clockwise" else -angle
        vals = [v + delta for v in vals]
        new_yaw = " ".join(f"{x:.2f}" for x in vals)
        block = block[:match.start(1)] + new_yaw + block[match.end(1):]
        print(f"[REORIENT] {obj_name} rotated {rotation_type} by {angle}°")
        bddl_text = bddl_text[:start] + block + bddl_text[end:]
    return bddl_text


def change_color(bddl_text, obj_name, obj_region_map, region_blocks):
    region_name = obj_region_map.get(obj_name)
    if not region_name or region_name not in region_blocks:
        print(f"[WARN] Region not found for {obj_name} (looking for '{region_name}')")
        return bddl_text
    if not supports_attributes(region_name, bddl_text):
        print(f"[SKIP] {obj_name} doesn't support color attributes (fixture)")
        return bddl_text

    start, end = region_blocks[region_name]
    block = bddl_text[start:end]

    rgba_match = re.search(r'\s*\(:rgba\s*\([^)]+\)\s*\)\s*\n?', block)
    if rgba_match:
        block = block[:rgba_match.start()] + block[rgba_match.end():]

    colors = {
        "red": [1.0, 0.0, 0.0, 1.0], "blue": [0.0, 0.0, 1.0, 1.0],
        "green": [0.0, 1.0, 0.0, 1.0], "yellow": [1.0, 1.0, 0.0, 1.0],
        "purple": [0.5, 0.0, 0.5, 1.0], "orange": [1.0, 0.5, 0.0, 1.0],
        "white": [1.0, 1.0, 1.0, 1.0], "black": [0.0, 0.0, 0.0, 1.0],
        "cyan": [0.0, 1.0, 1.0, 1.0], "magenta": [1.0, 0.0, 1.0, 1.0],
    }
    color_name = random.choice(list(colors.keys()))
    rgba_values = colors[color_name]

    last_close = block.rfind(')')
    depth, second_last_close = 0, -1
    for i in range(last_close - 1, -1, -1):
        if block[i] == ')':
            if depth == 0:
                second_last_close = i
                break
            depth += 1
        elif block[i] == '(':
            depth -= 1

    if second_last_close > 0:
        insert_pos = second_last_close + 1
        while insert_pos < len(block) and block[insert_pos] in ' \t':
            insert_pos += 1
        if insert_pos < len(block) and block[insert_pos] == '\n':
            insert_pos += 1
        prev_line_start = block.rfind('\n', 0, second_last_close)
        if prev_line_start >= 0:
            indent_m = re.match(r'^(\s*)', block[prev_line_start+1:])
            indent = indent_m.group(1) if indent_m else '          '
        else:
            indent = '          '
        rgba_str = " ".join(str(v) for v in rgba_values)
        rgba_line = f"{indent}(:rgba ({rgba_str}))\n"
        block = block[:insert_pos] + rgba_line + block[insert_pos:]

    bddl_text = bddl_text[:start] + block + bddl_text[end:]
    print(f"[COLOR] {obj_name} → {color_name} (RGBA: {rgba_values})")
    return bddl_text


def replace_object(bddl_text, obj_name, target_workspace=None):
    valid_objects = [
        "akita_black_bowl", "white_yellow_mug", "wine_bottle", "plate",
        "alphabet_soup", "cream_cheese", "tomato_sauce", "ketchup", "butter",
        "milk", "chocolate_pudding", "orange_juice", "bbq_sauce",
        "salad_dressing", "black_book", "moka_pot", "chefmate_8_frypan",
    ]
    if target_workspace is None:
        target_workspace = extract_target_workspace(bddl_text)

    new_obj_type = random.choice(valid_objects)
    new_obj = f"{new_obj_type}_{random.randint(1, 999)}"

    old_type_match = re.search(rf"{obj_name}\s*-\s*(\w+)", bddl_text)
    if not old_type_match:
        print(f"[WARN] Could not find object type for {obj_name}")
        return bddl_text
    old_obj_type = old_type_match.group(1)

    old_region = f"{obj_name}_init_region"
    new_region = f"{new_obj}_init_region"

    bddl_text = re.sub(rf"\({old_region}\b", f"({new_region}", bddl_text)
    bddl_text = re.sub(rf"(\s*){obj_name}(\s*-\s*){old_obj_type}\b",
                       rf"\1{new_obj}\2{new_obj_type}", bddl_text)
    bddl_text = re.sub(rf"\(On\s+{obj_name}\b", f"(On {new_obj}", bddl_text)
    bddl_text = re.sub(rf"\b{target_workspace}_{old_region}\b",
                       f"{target_workspace}_{new_region}", bddl_text)
    bddl_text = re.sub(rf"\(In\s+{obj_name}\b", f"(In {new_obj}", bddl_text)
    bddl_text = re.sub(rf"(\s+){obj_name}(\s*\n)", rf"\1{new_obj}\2", bddl_text)

    print(f"[REPLACE] {obj_name} → {new_obj}")
    return bddl_text

def add_distractor(bddl_text, target_workspace=None, position=None, object_type=None, z_overrides=None):
    """Add a distractor object to the scene.
    If position=(x, y) is provided, place it in a small region around that point (table-plane coords).
    Otherwise sample random position in [-0.2, 0.2] for both axes.
    If object_type is provided (str or list), use that type (or random choice from list); otherwise random from valid_objects.
    Collision is resolved by stacking/cavity placement (same logic as
    move_object).  Any required Z override is recorded in *z_overrides*.
    """
    if z_overrides is None:
        z_overrides = {}
    # Valid LIBERO object categories (expanded for all scene types)
    valid_objects = [
        "akita_black_bowl", "white_yellow_mug", "wine_bottle", "plate",
        "alphabet_soup", "cream_cheese", "tomato_sauce", "ketchup", "butter",
        "milk", "chocolate_pudding", "orange_juice", "bbq_sauce",
        "salad_dressing", "black_book", "moka_pot", "chefmate_8_frypan",
    ]

    # Auto-detect workspace if not provided
    if target_workspace is None:
        target_workspace = extract_target_workspace(bddl_text)

    if object_type is not None:
        if isinstance(object_type, (list, tuple)) and len(object_type) > 0:
            obj_type = random.choice(object_type)
        else:
            obj_type = object_type if isinstance(object_type, str) else random.choice(valid_objects)
        if obj_type not in valid_objects:
            raise ValueError(f"Distractor object_type '{obj_type}' not in valid LIBERO objects: {valid_objects}")
    else:
        obj_type = random.choice(valid_objects)
    new_obj = f"{obj_type}_{random.randint(100,999)}"
    region_name = f"{new_obj}_init_region"

    # Generate ranges: (x_min, y_min, x_max, y_max) — table-plane x, z
    if position is not None:
        cx, cy = position[0], position[1]
    else:
        cx = round(random.uniform(-0.2, 0.2), 4)
        cy = round(random.uniform(-0.2, 0.2), 4)
    
    # We need current region_blocks to check collisions
    region_blocks = find_region_blocks(bddl_text)
    obj_region_map = parse_object_region_map(bddl_text, region_blocks)

    # Collision resolution (the new region doesn't exist yet, so we check
    # against existing regions only)
    final_cx, final_cy = _check_and_resolve_bddl_collision(
        bddl_text, region_blocks, obj_region_map,
        new_obj, region_name, cx, cy,
        z_overrides, target_workspace,
    )

    half = 0.0005  # tiny extent so MuJoCo gets a valid positive size
    x_min = round(final_cx - half, 4)
    x_max = round(final_cx + half, 4)
    y_min = round(final_cy - half, 4)
    y_max = round(final_cy + half, 4)
    ranges_str = f"{x_min} {y_min} {x_max} {y_max}"

    region_def = (
        f"      ({region_name}\n"
        f"          (:target {target_workspace})\n"
        f"          (:ranges (\n"
        f"              ({ranges_str})\n"
        f"            )\n"
        f"          )\n"
        f"          (:yaw_rotation (\n"
        f"              (0.0 0.0)\n"
        f"            )\n"
        f"          )\n"
        f"      )"
    )

    regions_start = bddl_text.find("(:regions")
    if regions_start == -1:
        print("[WARN] Could not find :regions block")
        return bddl_text

    depth, regions_end = 0, -1
    for i in range(regions_start, len(bddl_text)):
        if bddl_text[i] == "(":  depth += 1
        elif bddl_text[i] == ")":
            depth -= 1
            if depth == 0:
                regions_end = i
                break
    if regions_end == -1:
        print("[WARN] Could not find end of :regions block")
        return bddl_text

    line_before = bddl_text[:regions_end].rfind('\n')
    indent_m = re.match(r'^(\s*)', bddl_text[line_before+1:regions_end])
    base_indent = indent_m.group(1) if indent_m else "    "
    bddl_text = (bddl_text[:regions_end] + "\n" + region_def + "\n"
                 + base_indent + bddl_text[regions_end:])

    # Add to :objects
    # BDDL parser assigns each category one list; a second "inst - category" line overwrites the first.
    # If obj_type already exists: add new_obj to that line (e.g. "wine_bottle_1 - wine_bottle" -> "wine_bottle_1 wine_bottle_456 - wine_bottle").
    # If obj_type is not in the scene: append a new line "new_obj - obj_type".
    obj_pattern = r"(\(:objects\s*\n)((?:.*\n)*?)(\s*\))"
    obj_match = re.search(obj_pattern, bddl_text)
    if obj_match:
        obj_content = obj_match.group(2)
        old_line = f" - {obj_type}\n"
        if old_line in obj_content:
            obj_content = obj_content.replace(old_line, f" {new_obj}{old_line}", 1)
        else:
            last_line = obj_content.rstrip().split("\n")[-1] if obj_content.strip() else ""
            indent = re.match(r"^(\s*)", last_line).group(1) if last_line else "    "
            obj_content = obj_content + f"{indent}{new_obj} - {obj_type}\n"
        bddl_text = bddl_text[:obj_match.start()] + obj_match.group(1) + obj_content + obj_match.group(3) + bddl_text[obj_match.end():]

    # Add to :init
    init_m = re.search(r"(\(:init\s*\n)((?:.*\n)*?)(\s*\))", bddl_text)
    if init_m:
        content = init_m.group(2)
        last_line = content.rstrip().split('\n')[-1] if content.strip() else ""
        indent = re.match(r'^(\s*)', last_line).group(1) if last_line else "    "
        new_content = (content
                       + f"{indent}(On {new_obj} {target_workspace}_{region_name})\n")
        bddl_text = (bddl_text[:init_m.start()] + init_m.group(1) + new_content
                     + init_m.group(3) + bddl_text[init_m.end():])

    print(f"[DISTRACTOR] Added '{new_obj}' at ({final_cx:.4f},{final_cy:.4f}) on {target_workspace}")
    return bddl_text


# --------------------------
# Apply perturbations
# --------------------------

def apply_perturbations_kitchen(bddl_text, perturbations, init_object_range_m=0.0,
                                 max_move_m=0.05, max_init_range_m=0.001,
                                 perturbation_spec_dict=None):
    """Deprecated alias — use apply_perturbations() directly."""
    return apply_perturbations(bddl_text, perturbations, init_object_range_m,
                               max_move_m, max_init_range_m, perturbation_spec_dict)


def apply_perturbations(bddl_text, perturbations, init_object_range_m=0.0,
                         max_move_m=0.05, max_init_range_m=0.001,
                         perturbation_spec_dict=None):
    """
    Apply perturbations to any LIBERO scene type.

    Args:
        bddl_text: BDDL file content as string
        perturbations: Dictionary of perturbations to apply
            - "move": list of object names to move
            - "reorient": list of object names to reorient
            - "color": list of object names to change color
            - "replace": list of object names to replace
            - "distractor": list of None values (count determines number of distractors)
        init_object_range_m: Size of init region (m). 0 = use max_init_range_m; >0 = box.
        max_move_m: Max distance (m) from unperturbed center (used when perturbation_spec_dict not provided for move).
        max_init_range_m: Max extent (m) when init_object_range_m <= 0. Set in config YAML.
        perturbation_spec_dict: Optional spec dict, e.g. {"move": {obj_name: [x, y], ...}}. When provided for move, uses these centers instead of random. 
        For distractor, can be {"distractor": [[x1, y1], [x2, y2], ...]} to place each distractor at the given (x, y) table-plane position.

    Returns
    -------
    (perturbed_bddl_text, z_overrides)

    z_overrides : dict
        Populated for every object where a collision was detected and a Z
        adjustment is required (stacking on top of another object, or placement
        inside an open cavity / drawer).

        Entries take one of two forms:
          • Cavity:  (CAVITY_SENTINEL, region_name, cx, cy)
                     Z is NOT known yet — it will be computed from real sim
                     geometry by resolve_z_overrides(sim, z_overrides).
          • Stack:   (cx, cy, estimated_z)
                     Z is a name-based estimate that resolve_z_overrides()
                     will refine using actual geom sizes from the sim.

        Recommended usage after env.reset():
            z_overrides = resolve_z_overrides(env.sim, z_overrides)
            for obj_name, (cx, cy, z) in z_overrides.items():
                _write_object_pose(sim, obj_name, [cx, cy, z, 1, 0, 0, 0])

        If no collisions occur this dict is empty and resolve_z_overrides()
        is a no-op.
    """
    region_blocks = find_region_blocks(bddl_text)
    obj_region_map = parse_object_region_map(bddl_text, region_blocks)
    target_workspace = extract_target_workspace(bddl_text)

    if perturbation_spec_dict and "move" in perturbation_spec_dict:
        move_spec = perturbation_spec_dict["move"]
        for obj_name in perturbations.get("move", []):
            if obj_name not in move_spec:
                raise ValueError(
                    f"perturbation_spec_dict['move'] missing centre for '{obj_name}'")

    print(f"[DEBUG] Workspace: {target_workspace}")
    print(f"[DEBUG] Object-Region map: {obj_region_map}")
    print(f"[DEBUG] Regions: {list(region_blocks.keys())}")
    print(f"[DEBUG] init_object_range_m={init_object_range_m}, "
          f"max_move_m={max_move_m}, max_init_range_m={max_init_range_m}")

    z_overrides: Dict[str, Tuple[float, float, float]] = {}

    for key, obj_list in perturbations.items():
        for obj_name in obj_list:
            if key == "move":
                center_override = None
                if (perturbation_spec_dict and "move" in perturbation_spec_dict
                        and obj_name in perturbation_spec_dict["move"]):
                    center_override = perturbation_spec_dict["move"][obj_name]
                bddl_text = move_object(
                    bddl_text, obj_name, obj_region_map, region_blocks,
                    init_object_range_m, max_move_m, max_init_range_m,
                    center_override=center_override,
                    z_overrides=z_overrides,
                    target_workspace=target_workspace,
                )
            elif key == "reorient":
                bddl_text = reorient_object(bddl_text, obj_name, obj_region_map, region_blocks)
            elif key == "color":
                bddl_text = change_color(bddl_text, obj_name, obj_region_map, region_blocks)
            elif key == "replace":
                bddl_text = replace_object(bddl_text, obj_name, target_workspace)

            # Refresh after any structural change
            region_blocks = find_region_blocks(bddl_text)
            obj_region_map = parse_object_region_map(bddl_text, region_blocks)

    if "distractor" in perturbations:
        distractor_positions = None
        distractor_object_type = None
        if perturbation_spec_dict and "distractor" in perturbation_spec_dict:
            distractor_positions = perturbation_spec_dict["distractor"]
        if perturbation_spec_dict:
            distractor_object_type = perturbation_spec_dict.get("distractor_object_type") or perturbation_spec_dict.get("distractor_object_types")
        for i, _ in enumerate(perturbations["distractor"]):
            pos = None
            if distractor_positions is not None and i < len(distractor_positions):
                pos = tuple(distractor_positions[i])
            bddl_text = add_distractor(bddl_text, target_workspace, position=pos, object_type=distractor_object_type, z_overrides=z_overrides)

    # Ensure every region in the scene has minimal extent (all objects, not just perturbed ones)
    bddl_text = fix_init_ranges(bddl_text, init_object_range_m=init_object_range_m, max_init_range_m=max_init_range_m)
    return bddl_text, z_overrides

# --------------------------
# Validation
# --------------------------

def validate_bddl(bddl_text):
    errors = []
    stack = []
    for i, ch in enumerate(bddl_text):
        if ch == "(":   stack.append(i)
        elif ch == ")":
            if not stack:
                errors.append(f"Extra ')' at index {i}")
                return False
            stack.pop()
    if stack:
        errors.append(f"{len(stack)} unmatched '('")
        return False

    declared_objects  = extract_declared_objects(bddl_text)
    declared_fixtures = extract_fixture_objects(bddl_text)
    all_declared = declared_objects | declared_fixtures

    for match in re.finditer(r"\(On\s+(\w+)\s+", bddl_text):
        obj_name = match.group(1)
        if obj_name not in all_declared:
            errors.append(f"Object '{obj_name}' used in :init but not declared")

    region_blocks = find_region_blocks(bddl_text)
    available_regions = set(region_blocks.keys())
    
    # Pattern: (On object_name target_region_name) or (On obj1 obj2) for object-on-object
    # The second arg can be a region (e.g. kitchen_table_akita_black_bowl_init_region) or
    # a declared object (e.g. plate_1 in (And (On akita_black_bowl_1 plate_1))).
    init_region_pattern = r"\(On\s+\w+\s+([\w_]+)\)"
    for match in re.finditer(init_region_pattern, bddl_text):
        full_region_ref = match.group(1)
        # Direct region match
        if full_region_ref in available_regions:
            continue
        # Composite region (e.g. kitchen_table_akita_black_bowl_init_region)
        found = False
        for region in available_regions:
            if full_region_ref.endswith(region):
                found = True
                break
        if found:
            continue
        # Object-on-object: second arg is a declared object/fixture (e.g. plate_1 in (On X plate_1))
        if full_region_ref in all_declared:
            continue
        errors.append(f"Region reference '{full_region_ref}' in :init doesn't match any defined region")
    
    if errors:
        print("[VALIDATION ERRORS]")
        for e in errors:
            print(f"  - {e}")
        return False
    print("[VALID] BDDL structure is correct")
    return True


# --------------------------
# Example usage
# --------------------------

if __name__ == "__main__":
    input_file = ("KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_"
                  "the_cabinet_and_close_it.bddl")
    bddl_text = read_bddl(input_file)

    perturbations = {
        "move": ["akita_black_bowl_1", "wine_bottle_1"],
        "reorient": ["wine_bottle_1"],
        "color": ["wine_bottle_1"],
        "replace": ["wine_bottle_1"],
        "distractor": [1],
    }

    perturbed_bddl, z_overrides = apply_perturbations(bddl_text, perturbations)

    if z_overrides:
        print("\n[Z OVERRIDES] Apply these after env.reset():")
        for obj, (cx, cy, z) in z_overrides.items():
            print(f"  {obj}: place at ({cx:.4f}, {cy:.4f}, z={z:.4f})")

    if validate_bddl(perturbed_bddl):
        save_bddl(perturbed_bddl, base_name="LIBERO_Kitchen_Tabletop_Manipulation_perturbed")
    else:
        print("[ERROR] Generated BDDL failed validation. Not saving.")