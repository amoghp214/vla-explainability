"""
Run directory creation and config loading for the pipeline.

Creates scratch/timestamped or override run_dir with subdirs: bddl_files, configs,
results, logs, jobs. Loads and holds main YAML config.
"""

import os
import yaml
from pathlib import Path
from typing import Optional
from datetime import datetime


# Project root: scripts/pipeline -> scripts -> parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def create_run_dir(
    config_path: str,
    run_dir_override: Optional[str] = None,
    config: Optional[dict] = None,
) -> "RunDir":
    """
    Create run directory structure and load config.

    Args:
        config_path: Path to main.yaml (used if config not provided).
        run_dir_override: If set, use this as run_dir; else use run_base_dir + timestamp.
        config: If provided, use this instead of loading from config_path.

    Returns:
        RunDir instance with .run_dir, .bddl_dir, .config_dir, .results_dir, .logs_dir, .jobs_dir, .config.
    """
    config_path = Path(config_path)
    if config is None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    scratch = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))
    scratch_dir = Path(scratch)

    if run_dir_override is not None:
        run_dir = Path(run_dir_override)
    else:
        run_base = config.get("run_base_dir")
        if run_base is None:
            run_base = scratch_dir / "vla-explainability-runs"
        else:
            run_base = Path(run_base)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = config.get("task_suite_name", "libero")
        run_dir = run_base / f"{task_name}_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)

    bddl_dir = run_dir / "bddl_files"
    config_dir = run_dir / "configs"
    results_dir = run_dir / "results"
    logs_dir = run_dir / "logs"
    jobs_dir = run_dir / "jobs"

    for d in (bddl_dir, config_dir, results_dir, logs_dir, jobs_dir):
        d.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "main_config.yaml", "w") as f:
        yaml.dump(config, f)

    return RunDir(
        run_dir=run_dir,
        bddl_dir=bddl_dir,
        config_dir=config_dir,
        results_dir=results_dir,
        logs_dir=logs_dir,
        jobs_dir=jobs_dir,
        config=config,
    )


class RunDir:
    """Holds paths and config for a single pipeline run."""

    def __init__(
        self,
        run_dir: Path,
        bddl_dir: Path,
        config_dir: Path,
        results_dir: Path,
        logs_dir: Path,
        jobs_dir: Path,
        config: dict,
    ):
        self.run_dir = Path(run_dir)
        self.bddl_dir = Path(bddl_dir)
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.logs_dir = Path(logs_dir)
        self.jobs_dir = Path(jobs_dir)
        self.config = config
