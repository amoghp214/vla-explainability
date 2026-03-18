"""
Random-design pipeline: single generator for BDDL + YAML from random sampling.

This module is the single place that generates all perturbation BDDL and record YAML
files from random (or uniform) sampling of perturbation coordinates. It produces
unperturbed.bddl, control.bddl, rd_0.bddl .. rd_{n-1}.bddl and corresponding YAMLs;
each rd_i.yaml includes temporal_perturbations with the sampled (x_i, y_i) and
start_step/end_step from config.

The temporal engine (libero.utils.temporal_perturbations / record.py) does not generate
perturbations: it only reads the generated YAMLs and applies/reverts the perturbation
at the configured start_step and end_step during the rollout.
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
    get_object_centers_from_bddl,
)
from .configs import write_record_config


def _temporal_spec_from_config(config: Dict[str, Any], design_type: str) -> Dict[str, Any]:
    """Build temporal spec template: start_step, end_step from config; type, obj_name/distractor_obj_name. No max_move_m — perturbation comes from bounds (delta_xy/distractor_xy set per design point)."""
    temporal = config.get("temporal_perturbations") or []
    for spec in temporal:
        if spec.get("type") == design_type:
            out = copy.deepcopy(spec)
            if "perturbation_start_step" in config:
                out["start_step"] = int(config["perturbation_start_step"])
            if "perturbation_stop_step" in config:
                out["end_step"] = int(config["perturbation_stop_step"])
            return out
    start = int(config.get("perturbation_start_step", config.get("start_step", 0)))
    end = int(config.get("perturbation_stop_step", config.get("end_step", 99999)))
    rd = config.get("random_design", {})
    if design_type == "move":
        obj = (rd.get("object_names") or config.get("object_names") or ["akita_black_bowl_1"])[0]
        return {"type": "move", "obj_name": obj, "start_step": start, "end_step": end}
    hidden = config.get("hidden_objects") or []
    if not hidden:
        raise ValueError("Random design type 'distract' requires hidden_objects in config.")
    return {"type": "distractor", "distractor_obj_name": hidden[0]["name"], "start_step": start, "end_step": end}


def generate_random_design_perturbations(
    config: Dict[str, Any],
    bddl_dir: Path,
    config_dir: Path,
    results_dir: Path,
    create_record_config_fn: Callable[..., Dict],
    n_design: int,
    bounds_x: Tuple[float, float],
    bounds_y: Tuple[float, float],
    object_names: Optional[List[str]] = None,
    seed: int = 1,
    include_control: bool = True,
    uniform: bool = False,
    design_type: str = "move",
    distractor_count: int = 1,
    distractor_object_type: Optional[str] = None,
    distractor_object_types: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate unperturbed + control + n_design design points. Produces n+2 BDDL files:
    unperturbed.bddl, control.bddl, rd_0.bddl .. rd_{n-1}.bddl. Each rd_i has its own BDDL
    (object or distractor at (x_i, y_i)) and rd_i.yaml with temporal_perturbations at
    start_step/end_step so the same perturbation is applied during recording.
    """
    if design_type not in ("move", "distract"):
        raise ValueError(f"design_type must be 'move' or 'distract', got {design_type!r}")
    if design_type == "move" and (not object_names or len(object_names) != 1):
        raise ValueError("Random design type 'move' requires exactly one object in object_names")

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
    init_range_m = config.get("init_range_m", pert_config.get("init_range_m", 0.001))
    # Perturbation magnitude/coordinates come from bounds only; derive fallback for apply_single_perturbation (only used when spec is incomplete).
    x_low, x_high = bounds_x
    y_low, y_high = bounds_y
    fallback_max_move = max(abs(x_high - x_low), abs(y_high - y_low)) * 0.5 if (bounds_x and bounds_y) else 0.05
    base_bddl_text = fix_init_ranges(
        base_bddl_text,
        init_range_m=init_range_m,
    )

    base_prompt = config["base_prompt"]
    unperturbed_bddl_path = bddl_dir / "unperturbed.bddl"
    with open(unperturbed_bddl_path, "w") as f:
        f.write(base_bddl_text)

    unperturbed_config = create_record_config_fn(
        perturbation_id="unperturbed",
        bddl_file=str(unperturbed_bddl_path),
        prompt=base_prompt,
        temporal_perturbations_override=[],
    )
    write_record_config(unperturbed_config, config_dir / "unperturbed.yaml")
    perturbation_info.append({
        "id": "unperturbed",
        "bddl_file": str(unperturbed_bddl_path),
        "config_file": str(config_dir / "unperturbed.yaml"),
        "prompt": base_prompt,
        "type": "baseline",
        "description": "Baseline unperturbed task",
    })

    if include_control:
        control_bddl_path = bddl_dir / "control.bddl"
        with open(control_bddl_path, "w") as f:
            f.write(base_bddl_text)
        control_config = create_record_config_fn(
            perturbation_id="control",
            bddl_file=str(control_bddl_path),
            prompt=base_prompt,
            temporal_perturbations_override=[],
        )
        write_record_config(control_config, config_dir / "control.yaml")
        perturbation_info.append({
            "id": "control",
            "bddl_file": str(control_bddl_path),
            "config_file": str(config_dir / "control.yaml"),
            "prompt": base_prompt,
            "type": "control",
            "description": "Control (no perturbation)",
        })

    temporal_template = _temporal_spec_from_config(config, design_type)
    x_low, x_high = bounds_x
    y_low, y_high = bounds_y

    if design_type == "move":
        centers = get_object_centers_from_bddl(base_bddl_text, object_names)
        if object_names[0] not in centers:
            raise ValueError(f"Object {object_names[0]} not found in base BDDL.")
        cx, cy = centers[object_names[0]]
        perturbations = {"move": list(object_names)}
    else:
        perturbations = {"distractor": [None] * distractor_count}

    if uniform:
        xy_ratio = (x_high - x_low) / (y_high - y_low) if (y_high - y_low) != 0 else 1.0
        num_y = max(1, int(np.floor(np.sqrt(n_design / xy_ratio))))
        num_x = max(1, int(np.floor(xy_ratio * num_y)))
        uniform_x = np.linspace(x_low, x_high, num_x)
        uniform_y = np.linspace(y_low, y_high, num_y)
        uniform_xy_pairs = [(float(x), float(y)) for x in uniform_x for y in uniform_y]
        while len(uniform_xy_pairs) < n_design:
            uniform_xy_pairs.append((float(np.random.uniform(x_low, x_high)), float(np.random.uniform(y_low, y_high))))
        uniform_xy_pairs = uniform_xy_pairs[:n_design]
    else:
        uniform_xy_pairs = None

    for i in range(n_design):
        if uniform and uniform_xy_pairs is not None:
            x, y = uniform_xy_pairs[i]
        else:
            x = float(np.random.uniform(x_low, x_high))
            y = float(np.random.uniform(y_low, y_high))
        pert_id = f"rd_{i}"

        if design_type == "move":
            x_abs, y_abs = cx + x, cy + y
            spec_dict = params_to_move_spec_dict(base_bddl_text, object_names, {"x": x_abs, "y": y_abs})
            design_x, design_y = x, y
        else:
            spec_dict = {"distractor": [[float(x), float(y)]] * distractor_count}
            if distractor_object_type is not None:
                spec_dict["distractor_object_type"] = distractor_object_type
            elif distractor_object_types:
                spec_dict["distractor_object_type"] = np.random.choice(distractor_object_types).item()
            design_x, design_y = x, y

        try:
            perturbed_bddl = apply_single_perturbation(
                copy.deepcopy(base_bddl_text),
                spec_dict,
                perturbations,
                init_range_m=init_range_m,
                max_move_m=fallback_max_move,
            )
        except Exception as e:
            print(f"[WARN] Random design point {pert_id} (x={x:.4f}, y={y:.4f}) failed: {e}")
            continue

        pert_bddl_path = bddl_dir / f"{pert_id}.bddl"
        with open(pert_bddl_path, "w") as f:
            f.write(perturbed_bddl)

        spec = copy.deepcopy(temporal_template)
        spec["start_step"] = int(spec.get("start_step", 0))
        spec["end_step"] = int(spec.get("end_step", 99999))
        if design_type == "move":
            spec["delta_xy"] = [round(x, 4), round(y, 4)]
        else:
            spec["distractor_xy"] = [round(x, 4), round(y, 4)]

        record_config = create_record_config_fn(
            perturbation_id=pert_id,
            bddl_file=str(pert_bddl_path),
            prompt=base_prompt,
            temporal_perturbations_override=[spec],
        )
        config_path = config_dir / f"{pert_id}.yaml"
        write_record_config(record_config, config_path)
        perturbation_info.append({
            "id": pert_id,
            "bddl_file": str(pert_bddl_path),
            "config_file": str(config_path),
            "prompt": base_prompt,
            "type": "random_design_move" if design_type == "move" else "random_design_distract",
            "description": f"rd_{i} (x={design_x:.4f}, y={design_y:.4f})",
            "x": design_x,
            "y": design_y,
        })
        design_points.append({"id": pert_id, "x": design_x, "y": design_y})

    return perturbation_info, design_points


def run_heatmap(
    run_dir: Path,
    design_points: List[Dict[str, Any]],
    analysis_results_path: Optional[Path] = None,
    bounds_x: Optional[Tuple[float, float]] = None,
    bounds_y: Optional[Tuple[float, float]] = None,
    model_name: str = "SingleTaskGP",
    step: float = 0.02,
    project_root: Optional[Path] = None,
    origin_x: Optional[float] = None,
    origin_y: Optional[float] = None,
    design_type: str = "move",
) -> Path:
    """
    Load analysis results and design points, fit BoTorch GP, plot heatmap of metric vs x, y.
    If origin_x and origin_y are provided (move mode), (x, y) are treated as deltas and
    the heatmap plots absolute positions (original + delta). For distract mode or when origin
    is not set, (x, y) are plotted as-is. design_type "distract" uses "position" axis labels.
    Saves heatmap to run_dir / "heatmap_metric_vs_translation.png" and returns that path.
    """
    project_root = project_root or PROJECT_ROOT
    analysis_results_path = analysis_results_path or (run_dir / "analysis_results.json")
    if not analysis_results_path.exists():
        raise FileNotFoundError(f"Analysis results not found: {analysis_results_path}")

    with open(analysis_results_path, "r") as f:
        results = json.load(f)

    # Build (X, Y) from design_points and results; skip points with errors
    ids_to_xy = {p["id"]: (p["x"], p["y"]) for p in design_points}
    X_list = []
    Y_list = []
    for pid, (x, y) in ids_to_xy.items():
        r = results.get(pid)
        if r is None or "error" in r or "metric" not in r:
            continue
        # Plot absolute position (original + delta) when origin is provided
        if origin_x is not None and origin_y is not None:
            x_plot = origin_x + x
            y_plot = origin_y + y
        else:
            x_plot, y_plot = x, y
        X_list.append([x_plot, y_plot])
        Y_list.append(r["metric"])

    # Get results from control perturbation (delta 0,0 -> absolute = origin when origin set)
    assert "control" in results, "Control results not found in analysis results"
    assert "metric" in results["control"], "Control metric not found in analysis results"
    if origin_x is not None and origin_y is not None:
        X_list.append([origin_x + 0.0, origin_y + 0.0])
    else:
        X_list.append([0.0, 0.0])
    Y_list.append(results["control"]["metric"])

    if len(X_list) < 2:
        raise ValueError(f"Need at least 2 valid design points for heatmap; got {len(X_list)}")

    train_X_np = np.array(X_list, dtype=np.float64)
    train_Y_np = np.array(Y_list, dtype=np.float64).reshape(-1, 1)
    train_X = torch.from_numpy(train_X_np).double()
    train_Y = torch.from_numpy(train_Y_np).double()

    bounds_x_passed = bounds_x is not None
    bounds_y_passed = bounds_y is not None
    if bounds_x is None:
        bounds_x = (float(train_X_np[:, 0].min()), float(train_X_np[:, 0].max()))
    if bounds_y is None:
        bounds_y = (float(train_X_np[:, 1].min()), float(train_X_np[:, 1].max()))
    # When using absolute positions, convert passed-in (delta) bounds to absolute; data-derived bounds are already absolute
    if origin_x is not None and origin_y is not None and bounds_x_passed and bounds_y_passed:
        bounds_x = (origin_x + bounds_x[0], origin_x + bounds_x[1])
        bounds_y = (origin_y + bounds_y[0], origin_y + bounds_y[1])
    bounds_dict = {"x": bounds_x, "y": bounds_y}

    # Normalize to [0, 1] for GP (BoTorch convention in botorch_random_demo)
    x_min, x_max = bounds_x[0], bounds_x[1]
    y_min, y_max = bounds_y[0], bounds_y[1]
    train_X_norm = train_X.clone()
    train_X_norm[:, 0] = (train_X[:, 0] - x_min) / (x_max - x_min) if x_max > x_min else train_X[:, 0]
    train_X_norm[:, 1] = (train_X[:, 1] - y_min) / (y_max - y_min) if y_max > y_min else train_X[:, 1]
    d = 2

    # Import BoTorch helpers from botorch_random_demo (avoids duplicating model/plot code)
    sys_path = list(__import__("sys").path)
    if str(project_root) not in sys_path:
        __import__("sys").path.insert(0, str(project_root))
    from explainability.botorch_random_demo import (
        get_bounds_tensor,
        build_and_fit_model,
        plot_heatmap as plot_heatmap_fn,
        calculate_rms_error,
    )

    model = build_and_fit_model(model_name, train_X_norm, train_Y, d, kernel_name="matern0.5")
    rmse = calculate_rms_error(model, train_X_norm, train_Y)
    print(f"[INFO] Fitted {model_name} with RMSE on training data: {rmse:.4f}")

    out_path = run_dir / "heatmap_metric_vs_translation.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    title = "VLA metric vs (x, y) translation - RMSE: {:.4f}".format(rmse)
    if origin_x is not None and origin_y is not None:
        title = "VLA metric vs (x, y) absolute position - RMSE: {:.4f}".format(rmse)
    elif design_type == "distract":
        title = "VLA metric vs distractor position (x, y) - RMSE: {:.4f}".format(rmse)
    plot_heatmap_fn(
        bounds_dict,
        model,
        train_X_norm,
        train_Y,
        title,
        step=step,
        cmap="RdBu_r",
        ax=ax,
    )
    if origin_x is not None and origin_y is not None:
        ax.set_xlabel("x position (m)")
        ax.set_ylabel("y position (m)")
    elif design_type == "distract":
        ax.set_xlabel("x position (m)")
        ax.set_ylabel("y position (m)")
    else:
        ax.set_xlabel("x translation (m)")
        ax.set_ylabel("y translation (m)")
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved heatmap to {out_path}")
    return out_path
