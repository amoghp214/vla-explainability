"""
Build record config dict and write YAML for a given perturbation.

Same content as the previous launcher _create_record_config: model, task_suite_name,
device, cache_dir, bddl_file, prompt, out_file, record_path, action_scale, num_demos, noise_std.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List


def create_record_config(
    perturbation_id: str,
    bddl_file: str,
    prompt: str,
    config: Dict[str, Any],
    results_dir: Path,
    config_dir: Path,
    temporal_perturbations_override: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Create a record config dict for a perturbation (for record.py).

    Args:
        perturbation_id: e.g. "unperturbed", "perturbed_0", "bo_iter0_0".
        bddl_file: Path to BDDL file (can be relative to run dir).
        prompt: Task prompt string.
        config: Main YAML config (model, device, cache_dir, etc.).
        results_dir: Run's results directory (for out_file and videos).
        config_dir: Run's config directory (used to resolve relative bddl_file if needed).
        temporal_perturbations_override: If set, use this instead of config["temporal_perturbations"]
            (used by random_design to inject per-point temporal specs).

    Returns:
        Config dict suitable for record.py and yaml.dump.
    """
    bddl_path = Path(bddl_file)
    if not bddl_path.is_absolute():
        bddl_path = config_dir / bddl_path.name

    out_file = results_dir / f"{perturbation_id}.hdf5"
    videos_dir = results_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    record_config = {
        "model": config["model"],
        "task_suite_name": config["task_suite_name"],
        "device": config["device"],
        "cache_dir": config["cache_dir"],
        "bddl_file": str(bddl_path),
        "prompt": prompt,
        "out_file": str(out_file),
        "record_path": str(videos_dir / f"{perturbation_id}.mp4"),
        "action_scale": config.get("action_scale", 1.0),
        "num_demos": config.get("num_demos", 1),
        "noise_std": config.get("noise_std", 0.0),
        "perturbation_id": perturbation_id,
        "run_dir": str(results_dir.resolve().parent),
    }
    # Always forward temporal (mid-rollout) perturbation config so all launcher paths use it.
    # Use first frame (start_step: 0) for full-rollout; use large end_step (e.g. 99999) for "until episode end".
    record_config["temporal_perturbations"] = (
        temporal_perturbations_override
        if temporal_perturbations_override is not None
        else config.get("temporal_perturbations", [])
    )
    record_config["hidden_objects"] = config.get("hidden_objects", [])
    if config.get("target_workspace") is not None:
        record_config["target_workspace"] = config["target_workspace"]
    # Chunk boundaries for top-down frames / heatmap (num_chunks, max_rollout_frames)
    if config.get("temporal_perturbation") is not None:
        record_config["temporal_perturbation"] = config["temporal_perturbation"]
    return record_config


def write_record_config(
    record_config: Dict[str, Any],
    config_path: Path,
) -> None:
    """Write record config dict to a YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(record_config, f)
