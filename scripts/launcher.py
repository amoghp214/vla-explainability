#!/usr/bin/env python3
"""
Launcher for the VLA explainability pipeline: random-design + temporal (only workflow).

Workflow:
  1. Generate unperturbed.bddl, control.bddl, rd_0.bddl .. rd_{n-1}.bddl (n+2 BDDL files).
     Each rd_i.bddl has the object (or distractor) at the sampled (x_i, z_i) from bounds.
  2. Generate one YAML per run; each rd_i.yaml points to rd_i.bddl and sets temporal_perturbations
     at perturbation_start_step/perturbation_stop_step so the move is applied during recording.
  3. Dispatch SLURM jobs (record.py). Videos are rendered inside record.py.
  4. After jobs complete, run evaluation.
  5. Fit BO and save heatmap: axes = absolute perturbation position (m), color = VLA metric.

Usage:
    python scripts/launcher.py --config configs/vk_main.yaml
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
from scripts.pipeline.generate_perturbations import save_perturbation_manifest
from scripts.pipeline.render import render_videos
from scripts.pipeline.evaluation import run_evaluation
from scripts.pipeline.perturbation import read_bddl, get_object_centers_from_bddl
from scripts.pipeline.random_design import (
    generate_random_design_perturbations,
    run_heatmap,
)


class Launcher:
    """Orchestrator for the random-design + temporal pipeline: run_dir, generate BDDL+YAML, dispatch, render, evaluate, heatmap."""

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
        self.design_points: List[dict] = []  # [{id, x, z}, ...]
        self.random_design_type: str = "move"  # "move" or "distract"

        print(f"[INFO] Run directory: {self.run_dir}")

    def _create_record_config_fn(self, perturbation_id: str, bddl_file: str, prompt: str, **kwargs):
        """Bound create_record_config for pipeline callback. kwargs (e.g. temporal_perturbations_override) passed through."""
        return pipeline_create_record_config(
            perturbation_id=perturbation_id,
            bddl_file=bddl_file,
            prompt=prompt,
            config=self.config,
            results_dir=self.results_dir,
            config_dir=self.config_dir,
            **kwargs,
        )

    def generate_random_design_perturbations(
        self,
        n_design: int,
        bounds_x: Tuple[float, float],
        bounds_z: Tuple[float, float],
        object_names: Optional[List[str]],
        seed: int,
        include_control: bool = True,
        uniform: bool = False,
        design_type: str = "move",
        distractor_count: int = 1,
        distractor_object_type: Optional[str] = None,
        distractor_object_types: Optional[List[str]] = None,
    ) -> None:
        """Generate random-design perturbations (move or distract) for heatmap mode."""
        label = "x, z translation" if design_type == "move" else "distractor position (x, z)"
        print(f"\n[INFO] Generating random-design perturbations ({label})...")
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
            design_type=design_type,
            distractor_count=distractor_count,
            distractor_object_type=distractor_object_type,
            distractor_object_types=distractor_object_types,
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
        """Build heatmap of VLA metric vs (x, z). For move: uses absolute positions (original + delta). For distract: (x, z) are already absolute."""
        origin_x, origin_z = None, None
        if self.random_design_type == "move":
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
            design_type=self.random_design_type,
        )

    def run_random_design(
        self,
        n_design: int,
        bounds_x: Tuple[float, float],
        bounds_z: Tuple[float, float],
        object_names: Optional[List[str]],
        seed: int,
        generate_only: bool = False,
        uniform: bool = False,
        design_type: str = "move",
        distractor_count: int = 1,
        distractor_object_type: Optional[str] = None,
        distractor_object_types: Optional[List[str]] = None,
    ) -> None:
        """Run pipeline: generate (x,z) perturbations from bounds -> dispatch -> evaluate -> heatmap."""
        self.random_design_type = design_type
        title = "metric vs x, z translation" if design_type == "move" else "metric vs distractor position (x, z)"
        print("=" * 80)
        print("VLA Explainability Pipeline (random-design + temporal)")
        print("=" * 80)
        print(f"Run directory: {self.run_dir}")
        print(f"type={design_type}, n_design={n_design}, bounds_x={bounds_x}, bounds_z={bounds_z}, seed={seed}")
        if design_type == "move":
            print(f"object={object_names}")
        else:
            print(f"distractor_count={distractor_count}, distractor_object_type={distractor_object_type or distractor_object_types}")
        self.generate_random_design_perturbations(
            n_design=n_design,
            bounds_x=bounds_x,
            bounds_z=bounds_z,
            object_names=object_names,
            seed=seed,
            include_control=True,
            uniform=uniform,
            design_type=design_type,
            distractor_count=distractor_count,
            distractor_object_type=distractor_object_type,
            distractor_object_types=distractor_object_types,
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
        print("Generate-only complete. Next steps:")
        print("=" * 80)
        print(f"\n1. Record unperturbed, control, and rd_* configs (e.g. with SLURM or locally):")
        print(f"   python scripts/record.py --config {self.config_dir / 'unperturbed.yaml'}")
        print(f"   python scripts/record.py --config {self.config_dir / 'control.yaml'}")
        print(f"   python scripts/record.py --config {self.config_dir / 'rd_0.yaml'}")
        print(f"\n2. After all HDF5s are in {self.results_dir}, run evaluation and heatmap:")
        print(f"   (Re-run launcher without --generate-only, or run analysis + heatmap manually.)")
        print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch VLA explainability pipeline (random-design + temporal: generate BDDL/YAML, dispatch jobs, evaluate, heatmap)."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML (must include random_design section)")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate BDDL and record YAMLs. Do not submit SLURM jobs.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Override run directory (e.g. ./local_run).",
    )
    parser.add_argument(
        "--n-design",
        type=int,
        default=None,
        help="Number of random design points. Overrides config random_design.n_design.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for design sampling. Overrides config random_design.seed.",
    )
    parser.add_argument(
        "--bounds",
        type=str,
        default=None,
        help="Comma-separated low,high for both x and z in meters (e.g. -0.05,0.05). Overrides config random_design.bounds_x/z.",
    )
    parser.add_argument(
        "--objects",
        type=str,
        nargs="+",
        default=None,
        help="Object name(s) for type=move. Exactly one. Overrides config random_design.object_names.",
    )
    parser.add_argument(
        "--type",
        dest="design_type",
        type=str,
        choices=["move", "distract"],
        default=None,
        help="Perturbation type: move (translate object) or distract (add distractor). Overrides config random_design.type.",
    )
    args = parser.parse_args()

    launcher = Launcher(args.config, run_dir_override=args.run_dir)
    config = launcher.config
    rd_config = config.get("random_design", {})
    if not rd_config:
        raise ValueError("Config must include a 'random_design' section. See configs/vk_main.yaml.")
    design_type = args.design_type if args.design_type is not None else rd_config.get("type", "move")
    distractor_count = rd_config.get("distractor_count", 1)
    distractor_object_type = rd_config.get("distractor_object_type")
    distractor_object_types = rd_config.get("distractor_object_types")
    n_design = args.n_design if args.n_design is not None else rd_config.get("n_design", 20)
    seed = args.seed if args.seed is not None else rd_config.get("seed", 1)
    uniform = rd_config.get("uniform", False)
    if args.bounds:
        try:
            low, high = map(float, args.bounds.split(","))
            bounds_x = bounds_z = (low, high)
        except Exception:
            bounds_x = bounds_z = (-0.05, 0.05)
    else:
        bx = rd_config.get("bounds_x")
        bz = rd_config.get("bounds_z")
        if bx is not None and bz is not None:
            bounds_x = tuple(bx) if isinstance(bx, (list, tuple)) else (-0.05, 0.05)
            bounds_z = tuple(bz) if isinstance(bz, (list, tuple)) else (-0.05, 0.05)
        else:
            bounds_x = bounds_z = (-0.05, 0.05)
    object_names = None
    if design_type == "move":
        object_names = args.objects if args.objects is not None else rd_config.get("object_names")
        if not object_names or len(object_names) != 1:
            specs = config.get("perturbations", {}).get("bddl_spatial", {}).get("perturbation_specs", [])
            for s in specs:
                if s.get("type") == "move" and s.get("objects"):
                    object_names = s["objects"][:1]
                    break
            if not object_names or len(object_names) != 1:
                raise ValueError(
                    "type=move requires exactly one object. Set --objects or config random_design.object_names (e.g. [\"akita_black_bowl_1\"])."
                )
    launcher.run_random_design(
        n_design=n_design,
        bounds_x=bounds_x,
        bounds_z=bounds_z,
        object_names=object_names,
        seed=seed,
        generate_only=args.generate_only,
        uniform=uniform,
        design_type=design_type,
        distractor_count=distractor_count,
        distractor_object_type=distractor_object_type,
        distractor_object_types=distractor_object_types,
    )


if __name__ == "__main__":
    main()
