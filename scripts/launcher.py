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

    # Random-design mode: random (x, z) translation perturbations -> VLA metric -> heatmap:
    python scripts/launcher.py --config configs/main.yaml --random-design --n-design 20 --bounds -0.05,0.05

    # Local tryout: only generate perturbations, then record one config by hand:
    python scripts/launcher.py --config configs/main.yaml --generate-only --run-dir ./local_run
    python scripts/record.py --config ./local_run/configs/perturbed_0.yaml
"""

import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

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
from scripts.pipeline.perturbation import read_bddl, get_object_centers_from_bddl
from scripts.pipeline.random_design import (
    generate_random_design_perturbations,
    run_heatmap,
)


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
        self.design_points: List[dict] = []  # Used in random-design mode: [{id, x, z}, ...]

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

    def generate_random_design_perturbations(
        self,
        n_design: int,
        bounds_x: Tuple[float, float],
        bounds_z: Tuple[float, float],
        object_names: List[str],
        seed: int,
        include_control: bool = True,
        uniform: bool = False,
    ) -> None:
        """Generate random (x, z) translation perturbations for heatmap mode."""
        print("\n[INFO] Generating random-design perturbations (x, z translation)...")
        self.perturbation_info, self.design_points = generate_random_design_perturbations(
            config=self.config,
            bddl_dir=self.bddl_dir,
            config_dir=self.config_dir,
            results_dir=self.results_dir,
            create_record_config_fn=self._create_record_config_fn,
            n_design=n_design,
            bounds_x=bounds_x,
            bounds_z=bounds_z,
            object_names=object_names,
            seed=seed,
            include_control=include_control,
            uniform=uniform,
        )
        print(f"[INFO] Generated unperturbed + control + {len(self.design_points)} random design points")
        save_perturbation_manifest(self.perturbation_info, self.run_dir)
        design_points_path = self.run_dir / "random_design_points.json"
        with open(design_points_path, "w") as f:
            json.dump(self.design_points, f, indent=2)
        print(f"[INFO] Saved design points to {design_points_path}")

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

    def run_heatmap(self, bounds_x: Tuple[float, float], bounds_z: Tuple[float, float]) -> None:
        """Build heatmap of VLA metric vs (x, z) translation from analysis results. Uses absolute positions (original + delta) when base BDDL and object are available."""
        origin_x, origin_z = None, None
        try:
            base_bddl = Path(self.config["base_bddl_file"])
            if not base_bddl.is_absolute():
                base_bddl = PROJECT_ROOT / base_bddl
            if base_bddl.exists():
                base_bddl_text = read_bddl(str(base_bddl))
                object_names = self.config.get("random_design", {}).get("object_names")
                if object_names and len(object_names) >= 1:
                    centers = get_object_centers_from_bddl(base_bddl_text, object_names)
                    if object_names[0] in centers:
                        cx, cz = centers[object_names[0]]
                        origin_x, origin_z = float(cx), float(cz)
        except Exception as e:
            print(f"[WARN] Could not get object origin for absolute-position heatmap: {e}")
        run_heatmap(
            run_dir=self.run_dir,
            design_points=self.design_points,
            analysis_results_path=self.run_dir / "analysis_results.json",
            bounds_x=bounds_x,
            bounds_z=bounds_z,
            model_name="SingleTaskGP",
            step=0.001,
            project_root=PROJECT_ROOT,
            origin_x=origin_x,
            origin_z=origin_z,
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
        if self.config.get("render_videos", True):
            self.render_videos()
        self.run_evaluation()
        print("\n" + "=" * 80)
        print("Pipeline complete!")
        print(f"Results available in: {self.run_dir}")
        print("=" * 80)

    def run_random_design(
        self,
        n_design: int,
        bounds_x: Tuple[float, float],
        bounds_z: Tuple[float, float],
        object_names: List[str],
        seed: int,
        generate_only: bool = False,
        uniform: bool = False,
    ) -> None:
        """Run random-design pipeline: generate (x,z) perturbations -> dispatch -> evaluate -> heatmap."""
        print("=" * 80)
        print("VLA Random-Design Pipeline (metric vs x, z translation)")
        print("=" * 80)
        print(f"Run directory: {self.run_dir}")
        print(f"n_design={n_design}, bounds_x={bounds_x}, bounds_z={bounds_z}, object={object_names}, seed={seed}")
        self.generate_random_design_perturbations(
            n_design=n_design,
            bounds_x=bounds_x,
            bounds_z=bounds_z,
            object_names=object_names,
            seed=seed,
            include_control=True,
            uniform=uniform,
        )
        if generate_only:
            self._print_random_design_instructions()
            return
        self.dispatch_jobs()
        if self.config.get("render_videos", True):
            self.render_videos()
        eval_config = self.config.get("evaluation", {})
        if not eval_config.get("enabled", False):
            print("[INFO] Evaluation was disabled in config; enabling for random-design heatmap.")
            self.config.setdefault("evaluation", {})["enabled"] = True
        self.run_evaluation()
        self.run_heatmap(bounds_x=bounds_x, bounds_z=bounds_z)
        print("\n" + "=" * 80)
        print("Random-design pipeline complete!")
        print(f"Results and heatmap in: {self.run_dir}")
        print("=" * 80)

    def _print_random_design_instructions(self) -> None:
        print("\n" + "=" * 80)
        print("Random-design generate-only complete. Next steps:")
        print("=" * 80)
        print(f"\n1. Record unperturbed, control, and rd_* configs (e.g. with SLURM or locally):")
        print(f"   python scripts/record.py --config {self.config_dir / 'unperturbed.yaml'}")
        print(f"   python scripts/record.py --config {self.config_dir / 'control.yaml'}")
        print(f"   python scripts/record.py --config {self.config_dir / 'rd_0.yaml'}")
        print(f"\n2. After all HDF5s are in {self.results_dir}, run evaluation and heatmap:")
        print(f"   (Re-run launcher without --generate-only, or run analysis + heatmap manually.)")
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
        help="Only generate perturbation files (BDDL + config YAMLs). Do not submit SLURM jobs. Use with --run-dir for local step-by-step runs.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Override run directory (e.g. ./local_run). Useful with --generate-only for local step-by-step runs.",
    )
    # Random-design mode: random (x, z) translation -> VLA metric -> heatmap
    parser.add_argument(
        "--random-design",
        action="store_true",
        help="Run random-design pipeline: generate n_design random (x,z) move perturbations, dispatch jobs, run VLA metric, produce heatmap of metric vs x, z translation.",
    )
    parser.add_argument(
        "--n-design",
        type=int,
        default=None,
        help="Number of random design points (for --random-design). Overrides config random_design.n_design if set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for design sampling (for --random-design).",
    )
    parser.add_argument(
        "--bounds",
        type=str,
        default=None,
        help="Comma-separated low,high for both x and z in meters (e.g. -0.05,0.05). If omitted, uses (-max_move_m, max_move_m) from config perturbations.bddl_spatial.",
    )
    parser.add_argument(
        "--objects",
        type=str,
        nargs="+",
        default=None,
        help="Object name(s) for move perturbation (for --random-design). Must be exactly one object. Overrides config random_design.object_names.",
    )
    args = parser.parse_args()

    if args.random_design:
        launcher = Launcher(args.config, run_dir_override=args.run_dir)
        config = launcher.config
        rd_config = config.get("random_design", {})
        n_design = args.n_design if args.n_design is not None else rd_config.get("n_design", 20)
        seed = args.seed if args.seed is not None else rd_config.get("seed", 1)
        uniform = rd_config.get("uniform", False)
        if args.bounds:
            try:
                low, high = map(float, args.bounds.split(","))
                bounds_x = bounds_z = (low, high)
            except Exception:
                max_move_m = config.get("perturbations", {}).get("bddl_spatial", {}).get("max_move_m", 0.05)
                default_bounds = (-max_move_m, max_move_m)
                bounds_x = bounds_z = default_bounds
        else:
            max_move_m = config.get("perturbations", {}).get("bddl_spatial", {}).get("max_move_m", 0.05)
            default_bounds = (-max_move_m, max_move_m)
            bx = rd_config.get("bounds_x")
            bz = rd_config.get("bounds_z")
            if bx is not None or bz is not None:
                bounds_x = tuple(bx) if isinstance(bx, (list, tuple)) else default_bounds
                bounds_z = tuple(bz) if isinstance(bz, (list, tuple)) else default_bounds
            else:
                bounds_x = bounds_z = default_bounds
        object_names = args.objects if args.objects is not None else rd_config.get("object_names")
        if not object_names or len(object_names) != 1:
            # Fallback from main perturbation specs
            specs = config.get("perturbations", {}).get("bddl_spatial", {}).get("perturbation_specs", [])
            for s in specs:
                if s.get("type") == "move" and s.get("objects"):
                    object_names = s["objects"][:1]
                    break
            if not object_names or len(object_names) != 1:
                raise ValueError(
                    "Random-design requires exactly one object. Set --objects or config random_design.object_names (e.g. [\"akita_black_bowl_1\"])."
                )
        launcher.run_random_design(
            n_design=n_design,
            bounds_x=bounds_x,
            bounds_z=bounds_z,
            object_names=object_names,
            seed=seed,
            generate_only=args.generate_only,
            uniform=uniform,
        )
        return

    launcher = Launcher(args.config, run_dir_override=args.run_dir)
    launcher.run(generate_only=args.generate_only)


if __name__ == "__main__":
    main()
