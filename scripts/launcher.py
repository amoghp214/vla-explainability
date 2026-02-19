#!/usr/bin/env python3
"""
Launcher script for pipelined perturbed dataset generation in PACE-ICE environment.

This script:
1. Creates a run directory structure in scratch folder
2. Generates perturbation files (BDDL and config YAMLs)
3. Dispatches SLURM jobs in a queue-like fashion
4. Runs evaluation scripts after all jobs complete

Usage:
    # Full pipeline (generate + SLURM jobs + videos + evaluation):
    python scripts/launcher.py --config configs/main.yaml

    # Local tryout: only generate perturbations, then record one config by hand:
    python scripts/launcher.py --config configs/main.yaml --generate-only --run-dir ./local_run
    python scripts/record.py --config ./local_run/configs/perturbed_0.yaml
"""

import json
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(project_root))

from scripts.pipeline.run_dir import create_run_dir, PROJECT_ROOT
from scripts.pipeline.configs import create_record_config as pipeline_create_record_config
from scripts.pipeline.slurm import dispatch_batch
from scripts.pipeline.generate_perturbations import generate_perturbations_from_config, save_perturbation_manifest
from scripts.pipeline.render import render_videos
from scripts.pipeline.evaluation import run_evaluation


class Launcher:
    """Thin orchestrator for the pipeline: run_dir, generate, dispatch, render, evaluate."""

    def __init__(self, config_path: str, run_dir_override: Optional[str] = None):
        self.config_path = Path(config_path)
        rd = create_run_dir(str(self.config_path), run_dir_override=run_dir_override)
        self.run_dir = rd.run_dir
        self.bddl_dir = rd.bddl_dir
        self.config_dir = rd.config_dir
        self.results_dir = rd.results_dir
        self.logs_dir = rd.logs_dir
        self.jobs_dir = rd.jobs_dir
        self.config = rd.config
        self.perturbation_info = []

        print(f"[INFO] Run directory: {self.run_dir}")

    def _create_record_config_fn(self, perturbation_id: str, bddl_file: str, prompt: str):
        """Bound create_record_config for pipeline callback."""
        return pipeline_create_record_config(
            perturbation_id=perturbation_id,
            bddl_file=bddl_file,
            prompt=prompt,
            config=self.config,
            results_dir=self.results_dir,
            config_dir=self.config_dir,
        )

    def generate_perturbations(self) -> None:
        """Generate all perturbation files (BDDL and config YAMLs) via pipeline."""
        print("\n[INFO] Generating perturbation files...")
        self.perturbation_info = generate_perturbations_from_config(
            config=self.config,
            bddl_dir=self.bddl_dir,
            config_dir=self.config_dir,
            results_dir=self.results_dir,
            create_record_config_fn=self._create_record_config_fn,
        )
        print(f"[INFO] Generated {len(self.perturbation_info)} perturbation files")
        save_perturbation_manifest(self.perturbation_info, self.run_dir)

    def dispatch_jobs(self) -> None:
        """Dispatch SLURM jobs via pipeline batch dispatch."""
        print("\n[INFO] Dispatching SLURM jobs...")
        perturbation_infos = [(p["id"], p["config_file"]) for p in self.perturbation_info]
        results = dispatch_batch(
            perturbation_infos,
            self.config["slurm"],
            self.results_dir,
            self.logs_dir,
            self.jobs_dir,
            project_root=PROJECT_ROOT,
        )
        completed = [r[0] for r in results if r[1]]
        failed = [r[0] for r in results if not r[1]]
        print(f"\n[INFO] All jobs dispatched")
        print(f"  Completed: {len(completed)}")
        print(f"  Failed: {len(failed)}")
        with open(self.run_dir / "job_summary.json", "w") as f:
            json.dump({"completed": completed, "failed": failed, "total": len(self.perturbation_info)}, f, indent=2)

    def render_videos(self) -> None:
        """Render videos via pipeline."""
        render_videos(
            self.perturbation_info,
            self.results_dir,
            self.config_dir,
            project_root=PROJECT_ROOT,
        )

    def run_evaluation(self) -> None:
        """Run evaluation via pipeline."""
        run_evaluation(
            self.perturbation_info,
            self.results_dir,
            self.run_dir,
            self.config,
            project_root=PROJECT_ROOT,
        )

    def run(self, generate_only: bool = False) -> None:
        if generate_only:
            self.generate_perturbations()
            self._print_local_recording_instructions()
            return
        print("=" * 80)
        print("VLA Explainability Pipeline Launcher")
        print("=" * 80)
        print(f"Run directory: {self.run_dir}")
        self.generate_perturbations()
        self.dispatch_jobs()
        self.render_videos()
        self.run_evaluation()
        print("\n" + "=" * 80)
        print("Pipeline complete!")
        print(f"Results available in: {self.run_dir}")
        print("=" * 80)

    def _print_local_recording_instructions(self) -> None:
        print("\n" + "=" * 80)
        print("Generate-only complete. To record one perturbation locally:")
        print("=" * 80)
        print(f"\n1. Run record for a single config (e.g. unperturbed or perturbed_0):")
        print(f"   python scripts/record.py --config {self.config_dir / 'unperturbed.yaml'}")
        print(f"\n   Or for a specific perturbation:")
        for info in self.perturbation_info:
            if info["id"] != "unperturbed":
                print(f"   python scripts/record.py --config {info['config_file']}")
                break
        print(f"\n2. From project root, ensure device/cache_dir in the generated config")
        print(f"   (in {self.config_dir}) match your machine, or override in the YAML.")
        print(f"\n3. After recording, render video (optional):")
        print(f"   python scripts/playback.py --config <same_config.yaml>")
        print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch pipelined perturbed dataset generation")
    parser.add_argument("--config", type=str, required=True, help="Path to main.yaml configuration file")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate perturbation files (BDDL + config YAMLs). Do not submit SLURM jobs. Use with --run-dir for local tryouts.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Override run directory (e.g. ./local_run). Useful with --generate-only for local step-by-step runs.",
    )
    args = parser.parse_args()
    launcher = Launcher(args.config, run_dir_override=args.run_dir)
    launcher.run(generate_only=args.generate_only)


if __name__ == "__main__":
    main()
