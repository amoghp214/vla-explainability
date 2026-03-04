"""
Perturbation logic: params -> perturbation_spec_dict, apply to BDDL.

Uses libero.utils.generate_perturbation_bddl for read_bddl, fix_init_ranges,
apply_perturbations, validate_bddl, and region parsing. Provides
params_to_move_spec_dict for BO (continuous params -> move spec) and
apply_single_perturbation for a single spec.
"""

import copy
import re
from pathlib import Path
from typing import Dict, List, Any

from .run_dir import PROJECT_ROOT

# Load libero perturbation utilities
_perturbation_utils_path = PROJECT_ROOT / "libero" / "libero" / "utils" / "generate_perturbation_bddl.py"
if not _perturbation_utils_path.exists():
    raise FileNotFoundError(f"Perturbation utilities not found at {_perturbation_utils_path}")

import importlib.util
_spec = importlib.util.spec_from_file_location("generate_perturbation_bddl", _perturbation_utils_path)
_pert_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pert_utils)

read_bddl = _pert_utils.read_bddl
fix_init_ranges = getattr(_pert_utils, "fix_init_ranges", lambda t, **kw: t)
apply_perturbations = getattr(_pert_utils, "apply_perturbations", _pert_utils.apply_perturbations_kitchen)
validate_bddl = _pert_utils.validate_bddl
find_region_blocks = _pert_utils.find_region_blocks
parse_object_region_map = _pert_utils.parse_object_region_map
generate_move_spec_dict = getattr(_pert_utils, "generate_move_spec_dict", lambda *a, **k: {})
RANGES_PATTERN = getattr(_pert_utils, "RANGES_PATTERN", re.compile(
    r":ranges\s*\(\s*\(\s*([-+]?[0-9]*\.?[0-9]+\s+[-+]?[0-9]*\.?[0-9]+\s+[-+]?[0-9]*\.?[0-9]+\s+[-+]?[0-9]*\.?[0-9]+)\s*\)",
    re.DOTALL,
))


def _get_object_centers_from_bddl(bddl_text: str, object_names: List[str]) -> Dict[str, tuple]:
    """
    Get (cx, cz) for each object from base BDDL region ranges.
    Returns dict {object_name: (cx, cz)}.
    """
    region_blocks = find_region_blocks(bddl_text)
    obj_region_map = parse_object_region_map(bddl_text, region_blocks)
    centers = {}
    for obj_name in object_names:
        region_name = obj_region_map.get(obj_name)
        if not region_name or region_name not in region_blocks:
            continue
        start, end = region_blocks[region_name]
        block = bddl_text[start:end]
        match = RANGES_PATTERN.search(block)
        if not match:
            continue
        coords = list(map(float, match.group(1).split()))
        cx = (coords[0] + coords[2]) / 2
        cz = (coords[1] + coords[3]) / 2
        centers[obj_name] = (cx, cz)
    return centers


def get_object_centers_from_bddl(bddl_text: str, object_names: List[str]) -> Dict[str, tuple]:
    """
    Public wrapper for _get_object_centers_from_bddl.
    Get (cx, cz) for each object from base BDDL region ranges.
    Returns dict {object_name: (cx, cz)}.
    """
    return _get_object_centers_from_bddl(bddl_text, object_names)


def params_to_move_spec_dict(
    base_bddl_text: str,
    object_names: List[str],
    params: Dict[str, float],
) -> Dict[str, Any]:
    """
    Map continuous BO params to perturbation_spec_dict for move type.

    Option A (recommended): params have 'dx', 'dz' -> center = (cx + dx, cz + dz).
    Option B: params have 'x', 'z' -> use (x, z) directly as new center.

    Args:
        base_bddl_text: BDDL file content (with init region ranges).
        object_names: List of object names to move (e.g. ["akita_black_bowl_1"]).
        params: Dict with either ('dx', 'dz') or ('x', 'z'). For multiple objects
                with dx/dz, same delta is applied to all; for x/z, one object only or pass per-object later.

    Returns:
        perturbation_spec_dict: {"move": {obj_name: [x, z], ...}} in table-plane coords.
    """
    if "x" in params and "z" in params:
        # Option B: absolute coords (single object typically)
        x, z = params["x"], params["z"]
        if len(object_names) != 1:
            raise ValueError("params_to_move_spec_dict: 'x'/'z' mode requires exactly one object")
        return {"move": {object_names[0]: [x, z]}}

    # Option A: dx, dz from unperturbed center
    dx = params.get("dx", 0.0)
    dz = params.get("dz", 0.0)
    centers = _get_object_centers_from_bddl(base_bddl_text, object_names)
    move_spec = {}
    for obj_name in object_names:
        if obj_name not in centers:
            continue
        cx, cz = centers[obj_name]
        move_spec[obj_name] = [round(cx + dx, 4), round(cz + dz, 4)]
    return {"move": move_spec} if move_spec else {}


def apply_single_perturbation(
    base_bddl_text: str,
    perturbation_spec_dict: Dict[str, Any],
    perturbations: Dict[str, List],
    init_object_range_m: float = 0.0,
    max_init_range_m: float = 0.001,
    max_move_m: float = 0.05,
) -> str:
    """
    Apply a single perturbation spec to base BDDL and return perturbed BDDL text.

    Args:
        base_bddl_text: Base BDDL content (already fix_init_ranges applied if desired).
        perturbation_spec_dict: e.g. {"move": {obj_name: [x, z]}}.
        perturbations: e.g. {"move": ["akita_black_bowl_1"]} (list of objects to move).
        init_object_range_m: Init region size (0 = use max_init_range_m).
        max_init_range_m: Max extent when init_object_range_m <= 0.
        max_move_m: Fallback when spec not provided (unused if spec has move).

    Returns:
        Perturbed BDDL text. Raises if validation fails (caller can catch).
    """
    perturbed = apply_perturbations(
        copy.deepcopy(base_bddl_text),
        perturbations,
        init_object_range_m=init_object_range_m,
        max_move_m=max_move_m,
        max_init_range_m=max_init_range_m,
        perturbation_spec_dict=perturbation_spec_dict,
    )
    if not validate_bddl(perturbed):
        raise ValueError("apply_single_perturbation: validate_bddl failed")
    return perturbed
