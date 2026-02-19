"""
SLURM job script creation, submit, status check, and batch dispatch.

Dispatch a list of (perturbation_id, config_file) in parallel with a cap
max_concurrent_jobs; block until all complete; return list of (perturbation_id, success, job_id).
"""

import os
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from .run_dir import PROJECT_ROOT


def create_job_script(
    perturbation_id: str,
    config_file: str,
    slurm_config: Dict[str, Any],
    logs_dir: Path,
    jobs_dir: Path,
    project_root: Optional[Path] = None,
) -> str:
    """
    Create a SLURM job script for record.py and return path to the script.

    Args:
        perturbation_id: e.g. "perturbed_0", "bo_iter0_0".
        config_file: Path to record config YAML.
        slurm_config: config['slurm'] (job_params, module_load, conda_env, job_name_prefix).
        logs_dir: Run's logs directory for stdout/stderr.
        jobs_dir: Run's jobs directory for .sh files.
        project_root: Project root for cd and record.py; default PROJECT_ROOT.

    Returns:
        Path to the written job script (string).
    """
    project_root = Path(project_root) if project_root else PROJECT_ROOT
    job_script = jobs_dir / f"{perturbation_id}.sh"
    job_params = slurm_config["job_params"]

    script_content = f"""#!/bin/bash
#SBATCH --job-name={slurm_config['job_name_prefix']}_{perturbation_id}
#SBATCH --account={job_params['account']}
#SBATCH --time={job_params['time']}
#SBATCH --nodes={job_params['nodes']}
#SBATCH --ntasks-per-node={job_params['ntasks_per_node']}
#SBATCH --cpus-per-task={job_params['cpus_per_task']}
"""

    if job_params.get("partition"):
        script_content += f"#SBATCH --partition={job_params['partition']}\n"
    if job_params.get("mem"):
        script_content += f"#SBATCH --mem={job_params['mem']}\n"
    if job_params.get("gpus", 0) > 0:
        gpu_type = job_params.get("gpu_type", "V100")
        num_gpus = job_params["gpus"]
        script_content += f"#SBATCH --gres=gpu:{gpu_type}:{num_gpus}\n"
    if job_params.get("constraint"):
        script_content += f"#SBATCH -C {job_params['constraint']}\n"
    blacklisted_nodes = job_params.get("blacklisted_nodes", [])
    if blacklisted_nodes:
        node_list = ",".join(str(n) for n in blacklisted_nodes) if isinstance(blacklisted_nodes, list) else str(blacklisted_nodes)
        script_content += f"#SBATCH --exclude={node_list}\n"

    script_content += f"""#SBATCH --output={logs_dir}/{perturbation_id}_%j.out
#SBATCH --error={logs_dir}/{perturbation_id}_%j.err

cd $SLURM_SUBMIT_DIR

"""
    for module in slurm_config.get("module_load", []):
        script_content += f"module load {module}\n"

    script_content += f"""
conda activate {slurm_config['conda_env']}

cd {project_root}

if ! python -c "import libero" 2>/dev/null; then
    echo "Installing libero package..."
    pip install -e . || {{ echo "ERROR: Failed to install libero package"; exit 1; }}
else
    echo "libero package already installed, skipping installation"
fi

python scripts/record.py --config {config_file}

echo "Job completed: {perturbation_id}"
"""

    with open(job_script, "w") as f:
        f.write(script_content)
    os.chmod(job_script, 0o755)
    return str(job_script)


def submit_job(job_script: str) -> Optional[str]:
    """Submit a SLURM job; return job ID or None on failure."""
    try:
        result = subprocess.run(
            ["sbatch", job_script],
            capture_output=True,
            text=True,
            check=True,
        )
        job_id = result.stdout.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to submit job {job_script}: {e.stderr}")
        return None


def check_job_status(job_id: str) -> str:
    """Return SLURM status string or 'COMPLETED' if job not in queue."""
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            check=True,
        )
        status = result.stdout.strip()
        if not status:
            return "COMPLETED"
        return status
    except subprocess.CalledProcessError:
        return "COMPLETED"


def dispatch_batch(
    perturbation_infos: List[Tuple[str, str]],
    slurm_config: Dict[str, Any],
    results_dir: Path,
    logs_dir: Path,
    jobs_dir: Path,
    project_root: Optional[Path] = None,
) -> List[Tuple[str, bool, Optional[str]]]:
    """
    Dispatch a list of (perturbation_id, config_file), respect max_concurrent_jobs,
    block until all complete. Return list of (perturbation_id, success, job_id).

    Success is True if the job completed and the corresponding .hdf5 exists in results_dir.
    """
    project_root = Path(project_root) if project_root else PROJECT_ROOT
    max_concurrent = slurm_config.get("max_concurrent_jobs", 4)
    poll_interval = slurm_config.get("poll_interval", 30)

    job_scripts = {}
    for pert_id, config_file in perturbation_infos:
        script_path = create_job_script(
            pert_id,
            config_file,
            slurm_config,
            logs_dir,
            jobs_dir,
            project_root,
        )
        job_scripts[pert_id] = script_path

    pending = list(perturbation_infos)
    running = {}  # pert_id -> job_id
    results = []  # (pert_id, success, job_id)

    while pending or running:
        # Check running jobs
        for pert_id in list(running.keys()):
            job_id = running[pert_id]
            status = check_job_status(job_id)
            if status == "COMPLETED":
                del running[pert_id]
                out_file = results_dir / f"{pert_id}.hdf5"
                success = out_file.exists()
                results.append((pert_id, success, job_id))
                if success:
                    print(f"[INFO] Job {pert_id} completed successfully")
                else:
                    print(f"[WARN] Job {pert_id} completed but output file missing")

        # Submit new jobs up to cap
        while len(running) < max_concurrent and pending:
            pert_id, config_file = pending.pop(0)
            script_path = job_scripts[pert_id]
            job_id = submit_job(script_path)
            if job_id:
                print(f"[INFO] Submitted job {pert_id} (SLURM ID: {job_id})")
                running[pert_id] = job_id
            else:
                results.append((pert_id, False, None))

        if running:
            time.sleep(poll_interval)

    # Preserve order of perturbation_infos in results (already preserved by processing order)
    return results
