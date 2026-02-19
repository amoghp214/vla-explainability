"""
Run evaluation after jobs complete: optional hdf5_to_json, then trajectory analysis.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from .run_dir import PROJECT_ROOT


def run_analysis_subprocess(
    unperturbed_file: Path,
    perturbed_files: List[str],
    control_file: Path,
    output_file: Path,
    eval_config: Dict[str, Any],
    project_root: Optional[Path] = None,
) -> None:
    """
    Run run_analysis.py as a subprocess with the given unperturbed, perturbed, and control files.
    """
    project_root = project_root or PROJECT_ROOT
    analysis_module_path = project_root / "explainability" / "run_analysis.py"
    if not analysis_module_path.exists():
        raise FileNotFoundError(f"Analysis module not found at {analysis_module_path}")

    cmd = [
        sys.executable,
        str(analysis_module_path),
        "--unperturbed",
        str(unperturbed_file),
        "--perturbed",
    ] + [str(p) for p in perturbed_files] + [
        "--controlled",
        str(control_file),
        "--output",
        str(output_file),
        "--metric-weights",
        json.dumps(eval_config["metric_weights"]),
        "--trajectory-weights",
        json.dumps(eval_config["trajectory_weights"]),
        "--project-root",
        str(project_root),
    ]
    subprocess.run(cmd, check=True)


def run_evaluation(
    perturbation_info: List[Dict[str, Any]],
    results_dir: Path,
    run_dir: Path,
    config: Dict[str, Any],
    project_root: Optional[Path] = None,
) -> None:
    """
    Run evaluation: optionally convert HDF5 to JSON, then run trajectory analysis.
    Skips if evaluation.enabled is False. Exits early if unperturbed or no perturbed files.
    """
    project_root = project_root or PROJECT_ROOT
    eval_config = config.get("evaluation", {})
    if not eval_config.get("enabled", False):
        print("[INFO] Evaluation disabled, skipping")
        return

    print("\n[INFO] Running evaluation...")
    unperturbed_file = results_dir / "unperturbed.hdf5"
    if not unperturbed_file.exists():
        print("[ERROR] Unperturbed file not found, cannot run evaluation")
        return

    perturbed_files = []
    for pert_info in perturbation_info:
        if pert_info["id"] != "unperturbed":
            pert_file = results_dir / f"{pert_info['id']}.hdf5"
            if pert_file.exists():
                perturbed_files.append(str(pert_file))

    if not perturbed_files:
        print("[WARN] No perturbed files found for evaluation")
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    if "json" in eval_config.get("output_formats", []):
        json_output = results_dir / "trajectories.json"
        cmd = [
            sys.executable,
            str(project_root / "utils" / "hdf5_to_json.py"),
            str(unperturbed_file),
            "-p",
        ] + perturbed_files + [
            "-o",
            str(json_output),
        ]
        print(f"[INFO] Converting to JSON: {json_output}")
        subprocess.run(cmd, check=True, env=env)

    control_file = results_dir / "control.hdf5"
    if not control_file.exists():
        control_file = unperturbed_file
    output_file = run_dir / "analysis_results.json"
    print("[INFO] Running trajectory analysis...")
    run_analysis_subprocess(
        unperturbed_file=unperturbed_file,
        perturbed_files=perturbed_files,
        control_file=control_file,
        output_file=output_file,
        eval_config=eval_config,
        project_root=project_root,
    )
