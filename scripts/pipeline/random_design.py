"""
Random-design pipeline: generate random (x, z) translation perturbations,
dispatch jobs, run VLA metric as black box, produce heatmap of metric vs x, z.

Uses pipeline modules for BDDL/config generation and SLURM, and run_analysis
for the scalar metric. Heatmap uses BoTorch GP (same pattern as botorch_random_demo).
"""

import json
import copy
from pathlib import Path
from typing import Dict, List, Any, Callable, Tuple, Optional

import numpy as np
import torch

from .run_dir import PROJECT_ROOT
from .perturbation import (
    read_bddl,
    fix_init_ranges,
    params_to_move_spec_dict,
    apply_single_perturbation,
)
from .configs import write_record_config


def generate_random_design_perturbations(
    config: Dict[str, Any],
    bddl_dir: Path,
    config_dir: Path,
    results_dir: Path,
    create_record_config_fn: Callable[[str, str, str], Dict],
    n_design: int,
    bounds_x: Tuple[float, float],
    bounds_z: Tuple[float, float],
    object_names: List[str],
    seed: int,
    include_control: bool = True,
    uniform: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate unperturbed + optional control + n_design random (x, z) move perturbations.

    Uses params_to_move_spec_dict with {"x": x, "z": z}, so object_names must have
    exactly one object (table-plane coords x, z).

    Returns:
        perturbation_info: List of dicts (id, bddl_file, config_file, prompt, ...).
        design_points: List of dicts (id, x, z) for each random design point (rd_0, rd_1, ...).
    """
    if len(object_names) != 1:
        raise ValueError("Random design with x/z bounds requires exactly one object in object_names")

    np.random.seed(seed)
    perturbation_info = []
    design_points = []

    base_bddl = Path(config["base_bddl_file"])
    if not base_bddl.is_absolute():
        base_bddl = PROJECT_ROOT / base_bddl
    if not base_bddl.exists():
        raise FileNotFoundError(f"Base BDDL file not found: {base_bddl}")

    base_bddl_text = read_bddl(str(base_bddl))
    pert_config = config.get("perturbations", {}).get("bddl_spatial", {})
    init_object_range_m = config.get("init_object_range_m", pert_config.get("init_object_range_m", 0.0))
    max_init_range_m = pert_config.get("max_init_range_m", 0.001)
    max_move_m = pert_config.get("max_move_m", 0.05)
    base_bddl_text = fix_init_ranges(
        base_bddl_text,
        init_object_range_m=init_object_range_m,
        max_init_range_m=max_init_range_m,
    )

    base_prompt = config["base_prompt"]

    # Unperturbed
    unperturbed_bddl_path = bddl_dir / "unperturbed.bddl"
    with open(unperturbed_bddl_path, "w") as f:
        f.write(base_bddl_text)
    unperturbed_config = create_record_config_fn(
        perturbation_id="unperturbed",
        bddl_file=str(unperturbed_bddl_path),
        prompt=base_prompt,
    )
    unperturbed_config_path = config_dir / "unperturbed.yaml"
    write_record_config(unperturbed_config, unperturbed_config_path)
    perturbation_info.append({
        "id": "unperturbed",
        "bddl_file": str(unperturbed_bddl_path),
        "config_file": str(unperturbed_config_path),
        "prompt": base_prompt,
        "type": "baseline",
        "description": "Baseline unperturbed task",
    })

    # Control (same BDDL as unperturbed, for run_analysis)
    if include_control:
        control_bddl_path = bddl_dir / "control.bddl"
        with open(control_bddl_path, "w") as f:
            f.write(base_bddl_text)
        control_config = create_record_config_fn(
            perturbation_id="control",
            bddl_file=str(control_bddl_path),
            prompt=base_prompt,
        )
        control_config_path = config_dir / "control.yaml"
        write_record_config(control_config, control_config_path)
        perturbation_info.append({
            "id": "control",
            "bddl_file": str(control_bddl_path),
            "config_file": str(control_config_path),
            "prompt": base_prompt,
            "type": "control",
            "description": "Control (no BDDL change)",
        })

    # Random design points: (x, z) in bounds
    x_low, x_high = bounds_x
    z_low, z_high = bounds_z
    perturbations = {"move": list(object_names)}

    if (uniform):
        xz_ratio = (x_high - x_low) / (z_high - z_low)
        num_points_z_axis = int(np.floor(np.sqrt(n_design / xz_ratio)))
        num_points_x_axis = int(np.floor(xz_ratio * num_points_z_axis))

        # num_points_per_axis = int(np.floor(np.sqrt(n_design)))
        uniform_x_values = np.linspace(x_low, x_high, num_points_x_axis)
        uniform_z_values = np.linspace(z_low, z_high, num_points_z_axis)
        uniform_xz_pairs = [(x, z) for x in uniform_x_values for z in uniform_z_values]
        for _ in range(len(uniform_xz_pairs), n_design):
            uniform_xz_pairs.append((np.random.uniform(x_low, x_high), np.random.uniform(z_low, z_high)))
    
    for i in range(n_design):
        if (uniform):
            x, z = uniform_xz_pairs[i]
        else:
            x = float(np.random.uniform(x_low, x_high))
            z = float(np.random.uniform(z_low, z_high))
        pert_id = f"rd_{i}"
        spec_dict = params_to_move_spec_dict(base_bddl_text, object_names, {"x": x, "z": z})
        try:
            perturbed_bddl = apply_single_perturbation(
                copy.deepcopy(base_bddl_text),
                spec_dict,
                perturbations,
                init_object_range_m=init_object_range_m,
                max_init_range_m=max_init_range_m,
                max_move_m=max_move_m,
            )
        except Exception as e:
            print(f"[WARN] Random design point {pert_id} (x={x:.4f}, z={z:.4f}) failed: {e}")
            continue
        pert_bddl_path = bddl_dir / f"{pert_id}.bddl"
        with open(pert_bddl_path, "w") as f:
            f.write(perturbed_bddl)
        record_config = create_record_config_fn(
            perturbation_id=pert_id,
            bddl_file=str(pert_bddl_path),
            prompt=base_prompt,
        )
        config_path = config_dir / f"{pert_id}.yaml"
        write_record_config(record_config, config_path)
        perturbation_info.append({
            "id": pert_id,
            "bddl_file": str(pert_bddl_path),
            "config_file": str(config_path),
            "prompt": base_prompt,
            "type": "random_design_move",
            "description": f"Move {object_names[0]} to (x={x:.4f}, z={z:.4f})",
            "x": x,
            "z": z,
        })
        design_points.append({"id": pert_id, "x": x, "z": z})

    return perturbation_info, design_points


def run_heatmap(
    run_dir: Path,
    design_points: List[Dict[str, Any]],
    analysis_results_path: Optional[Path] = None,
    bounds_x: Optional[Tuple[float, float]] = None,
    bounds_z: Optional[Tuple[float, float]] = None,
    model_name: str = "SingleTaskGP",
    step: float = 0.02,
    project_root: Optional[Path] = None,
) -> Path:
    """
    Load analysis results and design points, fit BoTorch GP, plot heatmap of metric vs x, z.
    Saves heatmap to run_dir / "heatmap_metric_vs_translation.png" and returns that path.
    """
    project_root = project_root or PROJECT_ROOT
    analysis_results_path = analysis_results_path or (run_dir / "analysis_results.json")
    if not analysis_results_path.exists():
        raise FileNotFoundError(f"Analysis results not found: {analysis_results_path}")

    with open(analysis_results_path, "r") as f:
        results = json.load(f)

    # Build (X, Y) from design_points and results; skip points with errors
    ids_to_xy = {p["id"]: (p["x"], p["z"]) for p in design_points}
    X_list = []
    Y_list = []
    for pid, (x, z) in ids_to_xy.items():
        r = results.get(pid)
        if r is None or "error" in r or "metric" not in r:
            continue
        X_list.append([x, z])
        Y_list.append(r["metric"])
    
    # Get results from control perturbation
    assert "control" in results, "Control results not found in analysis results"
    assert "metric" in results["control"], "Control metric not found in analysis results"
    X_list.append([0.0, 0.0])  # Control is at (0, 0) translation
    Y_list.append(results["control"]["metric"])

    if len(X_list) < 2:
        raise ValueError(f"Need at least 2 valid design points for heatmap; got {len(X_list)}")

    train_X_np = np.array(X_list, dtype=np.float64)
    train_Y_np = np.array(Y_list, dtype=np.float64).reshape(-1, 1)
    train_X = torch.from_numpy(train_X_np).double()
    train_Y = torch.from_numpy(train_Y_np).double()

    if bounds_x is None:
        bounds_x = (float(train_X_np[:, 0].min()), float(train_X_np[:, 0].max()))
    if bounds_z is None:
        bounds_z = (float(train_X_np[:, 1].min()), float(train_X_np[:, 1].max()))
    bounds_dict = {"x": bounds_x, "z": bounds_z}

    # Normalize to [0, 1] for GP (BoTorch convention in botorch_random_demo)
    x_min, x_max = bounds_x[0], bounds_x[1]
    z_min, z_max = bounds_z[0], bounds_z[1]
    train_X_norm = train_X.clone()
    train_X_norm[:, 0] = (train_X[:, 0] - x_min) / (x_max - x_min) if x_max > x_min else train_X[:, 0]
    train_X_norm[:, 1] = (train_X[:, 1] - z_min) / (z_max - z_min) if z_max > z_min else train_X[:, 1]
    d = 2

    # Import BoTorch helpers from botorch_random_demo (avoids duplicating model/plot code)
    sys_path = list(__import__("sys").path)
    if str(project_root) not in sys_path:
        __import__("sys").path.insert(0, str(project_root))
    from explainability.botorch_random_demo import (
        get_bounds_tensor,
        build_and_fit_model,
        plot_heatmap as plot_heatmap_fn,
    )

    model = build_and_fit_model(model_name, train_X_norm, train_Y, d)
    out_path = run_dir / "heatmap_metric_vs_translation.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_heatmap_fn(
        bounds_dict,
        model,
        train_X_norm,
        "VLA metric vs (x, z) translation",
        step=step,
        cmap="RdBu_r",
        ax=ax,
    )
    ax.set_xlabel("x translation (m)")
    ax.set_ylabel("z translation (m)")
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved heatmap to {out_path}")
    return out_path
