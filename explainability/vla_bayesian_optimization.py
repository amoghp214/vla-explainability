"""
VLA Bayesian Optimization pipeline.

Encapsulates: run directory setup, params -> perturbation_spec_dict -> BDDL/config,
SLURM batch dispatch, and run_analysis for the scalar metric. Uses BoTorch for
batch acquisition (suggest K points, evaluate K in parallel on SLURM, update GP with K).
"""

from __future__ import annotations

import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch

# BoTorch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition.monte_carlo import qExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
try:
    from botorch.sampling import SobolQMCNormalSampler
except ImportError:
    from botorch.sampling.normal import SobolQMCNormalSampler

# Project root and pipeline
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.run_dir import create_run_dir, PROJECT_ROOT as PIPELINE_ROOT
from scripts.pipeline.perturbation import (
    read_bddl,
    fix_init_ranges,
    params_to_move_spec_dict,
    apply_single_perturbation,
)
from scripts.pipeline.configs import create_record_config, write_record_config
from scripts.pipeline.slurm import dispatch_batch


def get_bounds_tensor(bounds_dict: Dict[str, Tuple[float, float]]) -> torch.Tensor:
    """Convert dict e.g. {'dx': (-0.05, 0.05), 'dz': (-0.05, 0.05)} to (2, d) tensor."""
    keys = sorted(bounds_dict.keys())
    lower = torch.tensor([bounds_dict[k][0] for k in keys], dtype=torch.double)
    upper = torch.tensor([bounds_dict[k][1] for k in keys], dtype=torch.double)
    return torch.stack([lower, upper])


def _load_run_analysis():
    """Load run_analysis module and return run_analysis function."""
    import explainability.run_analysis as ra
    return ra.run_analysis


class VLABayesianOptimization:
    """
    VLA BO pipeline: params (e.g. dx, dz) -> BDDL/config -> SLURM record jobs -> metric.
    Supports batch evaluation and BoTorch batch acquisition.
    """

    def __init__(
        self,
        config_path: str,
        run_dir_override: Optional[str] = None,
        pbounds: Optional[Dict[str, Tuple[float, float]]] = None,
        object_names: Optional[List[str]] = None,
        batch_size: int = 2,
        n_init: int = 5,
        n_iter: int = 10,
        metric_weights: Optional[Dict[str, float]] = None,
        trajectory_weights: Optional[List[float]] = None,
        seed: int = 1,
    ):
        self.config_path = Path(config_path)
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        bo_cfg = self.config.get("bayesian_optimization", {})
        self.pbounds = pbounds or bo_cfg.get("pbounds") or {"dx": (-0.05, 0.05), "dz": (-0.05, 0.05)}
        if isinstance(next(iter(self.pbounds.values()), None), list):
            self.pbounds = {k: tuple(v) for k, v in self.pbounds.items()}
        self.object_names = object_names or bo_cfg.get("object_names") or ["akita_black_bowl_1"]
        self.batch_size = bo_cfg.get("batch_size", batch_size)
        self.n_init = bo_cfg.get("init_points", n_init)
        self.n_iter = bo_cfg.get("n_iter", n_iter)
        eval_cfg = self.config.get("evaluation", {})
        self.metric_weights = metric_weights or eval_cfg.get("metric_weights", {"w_result": 1.0, "w_time": 1.0, "w_trajectory": 1.0})
        self.trajectory_weights = trajectory_weights or eval_cfg.get("trajectory_weights", [1.0, 1.0, 1.0, 0, 0, 0, 0, 0])
        self.seed = seed

        self._rd = create_run_dir(str(self.config_path), run_dir_override=run_dir_override)
        self.run_dir = self._rd.run_dir
        self.bddl_dir = self._rd.bddl_dir
        self.config_dir = self._rd.config_dir
        self.results_dir = self._rd.results_dir
        self.logs_dir = self._rd.logs_dir
        self.jobs_dir = self._rd.jobs_dir

        self._base_bddl_text: Optional[str] = None
        self._unperturbed_hdf5: Optional[Path] = None
        self._control_hdf5: Optional[Path] = None
        self._run_analysis_fn = None

    def _get_base_bddl_text(self) -> str:
        if self._base_bddl_text is not None:
            return self._base_bddl_text
        base_bddl = Path(self.config["base_bddl_file"])
        if not base_bddl.is_absolute():
            base_bddl = PROJECT_ROOT / base_bddl
        self._base_bddl_text = read_bddl(str(base_bddl))
        pert_config = self.config.get("perturbations", {})
        bddl_spatial = pert_config.get("bddl_spatial", {})
        init_range_m = self.config.get("init_range_m", bddl_spatial.get("init_range_m", 0.001))
        self._base_bddl_text = fix_init_ranges(
            self._base_bddl_text,
            init_range_m=init_range_m,
        )
        return self._base_bddl_text

    def ensure_setup(self) -> None:
        """Ensure unperturbed and control BDDL/configs exist and their record jobs have completed."""
        base_text = self._get_base_bddl_text()
        pert_config = self.config.get("perturbations", {})
        bddl_spatial = pert_config.get("bddl_spatial", {})
        init_range_m = self.config.get("init_range_m", bddl_spatial.get("init_range_m", 0.001))

        # Unperturbed BDDL + config
        unperturbed_bddl = self.bddl_dir / "unperturbed.bddl"
        with open(unperturbed_bddl, "w") as f:
            f.write(base_text)
        unperturbed_record = create_record_config(
            "unperturbed",
            str(unperturbed_bddl),
            self.config["base_prompt"],
            self.config,
            self.results_dir,
            self.config_dir,
        )
        write_record_config(unperturbed_record, self.config_dir / "unperturbed.yaml")

        # Control: same BDDL as unperturbed (no perturbation), different id
        control_record = create_record_config(
            "control",
            str(unperturbed_bddl),
            self.config["base_prompt"],
            self.config,
            self.results_dir,
            self.config_dir,
        )
        write_record_config(control_record, self.config_dir / "control.yaml")

        self._unperturbed_hdf5 = self.results_dir / "unperturbed.hdf5"
        self._control_hdf5 = self.results_dir / "control.hdf5"

        # Dispatch unperturbed and control if not already done
        to_run = []
        if not self._unperturbed_hdf5.exists():
            to_run.append(("unperturbed", str(self.config_dir / "unperturbed.yaml")))
        if not self._control_hdf5.exists():
            to_run.append(("control", str(self.config_dir / "control.yaml")))
        if to_run:
            print("[INFO] Running unperturbed and/or control record jobs...")
            dispatch_batch(
                to_run,
                self.config["slurm"],
                self.results_dir,
                self.logs_dir,
                self.jobs_dir,
                project_root=PIPELINE_ROOT,
            )
        return

    def evaluate(self, params: Dict[str, float], perturbation_id: str = "bo_single") -> float:
        """
        Evaluate one point: params -> BDDL + config -> one record job -> analysis -> metric.
        Returns metric or a high penalty if job failed.
        """
        base_text = self._get_base_bddl_text()
        spec_dict = params_to_move_spec_dict(base_text, self.object_names, params)
        perturbations = {"move": list(self.object_names)}
        pert_config = self.config.get("perturbations", {}).get("bddl_spatial", {})
        max_move_m = pert_config.get("max_move_m", 0.05)
        init_range_m = self.config.get("init_range_m", pert_config.get("init_range_m", 0.001))

        try:
            perturbed_bddl = apply_single_perturbation(
                base_text,
                spec_dict,
                perturbations,
                init_range_m=init_range_m,
                max_move_m=max_move_m,
            )
        except Exception as e:
            print(f"[WARN] apply_single_perturbation failed: {e}")
            return 1e6  # penalty

        bddl_path = self.bddl_dir / f"{perturbation_id}.bddl"
        with open(bddl_path, "w") as f:
            f.write(perturbed_bddl)
        record_config = create_record_config(
            perturbation_id,
            str(bddl_path),
            self.config["base_prompt"],
            self.config,
            self.results_dir,
            self.config_dir,
        )
        write_record_config(record_config, self.config_dir / f"{perturbation_id}.yaml")

        results = dispatch_batch(
            [(perturbation_id, str(self.config_dir / f"{perturbation_id}.yaml"))],
            self.config["slurm"],
            self.results_dir,
            self.logs_dir,
            self.jobs_dir,
            project_root=PIPELINE_ROOT,
        )
        if not results or not results[0][1]:
            return 1e6
        # Run analysis for this one file
        run_analysis = _load_run_analysis()
        output_file = str(self.run_dir / "analysis_bo_single.json")
        res = run_analysis(
            unperturbed_file=str(self._unperturbed_hdf5),
            perturbed_files=[str(self.results_dir / f"{perturbation_id}.hdf5")],
            controlled_file=str(self._control_hdf5),
            output_file=output_file,
            metric_weights=self.metric_weights,
            trajectory_weights=self.trajectory_weights,
            project_root=str(PROJECT_ROOT),
        )
        return float(res.get(perturbation_id, {}).get("metric", 1e6))

    def evaluate_batch(
        self,
        params_list: List[Dict[str, float]],
        iteration: int,
    ) -> List[float]:
        """
        For each params build BDDL + config (bo_iter{I}_0, ...), dispatch_batch all,
        run analysis once with all K perturbed files, return list of K metrics in order.
        """
        base_text = self._get_base_bddl_text()
        pert_config = self.config.get("perturbations", {}).get("bddl_spatial", {})
        max_move_m = pert_config.get("max_move_m", 0.05)
        init_range_m = self.config.get("init_range_m", pert_config.get("init_range_m", 0.001))
        perturbations = {"move": list(self.object_names)}

        infos = []
        for j, params in enumerate(params_list):
            pid = f"bo_iter{iteration}_{j}"
            spec_dict = params_to_move_spec_dict(base_text, self.object_names, params)
            try:
                perturbed_bddl = apply_single_perturbation(
                    base_text,
                    spec_dict,
                    perturbations,
                    init_range_m=init_range_m,
                    max_move_m=max_move_m,
                )
            except Exception as e:
                print(f"[WARN] {pid} apply failed: {e}")
                infos.append((pid, None))
                continue
            bddl_path = self.bddl_dir / f"{pid}.bddl"
            with open(bddl_path, "w") as f:
                f.write(perturbed_bddl)
            rec = create_record_config(
                pid,
                str(bddl_path),
                self.config["base_prompt"],
                self.config,
                self.results_dir,
                self.config_dir,
            )
            write_record_config(rec, self.config_dir / f"{pid}.yaml")
            infos.append((pid, str(self.config_dir / f"{pid}.yaml")))

        # Dispatch all (skip failed)
        to_dispatch = [(pid, path) for pid, path in infos if path is not None]
        if not to_dispatch:
            return [1e6] * len(params_list)
        dispatch_batch(
            to_dispatch,
            self.config["slurm"],
            self.results_dir,
            self.logs_dir,
            self.jobs_dir,
            project_root=PIPELINE_ROOT,
        )

        # Run analysis only for perturbed files that exist (successful jobs)
        perturbed_paths = [str(self.results_dir / f"{pid}.hdf5") for pid, path in infos if path is not None]
        output_file = str(self.run_dir / f"analysis_bo_iter{iteration}.json")
        res = {}
        if perturbed_paths:
            run_analysis = _load_run_analysis()
            res = run_analysis(
                unperturbed_file=str(self._unperturbed_hdf5),
                perturbed_files=perturbed_paths,
                controlled_file=str(self._control_hdf5),
                output_file=output_file,
                metric_weights=self.metric_weights,
                trajectory_weights=self.trajectory_weights,
                project_root=str(PROJECT_ROOT),
            )

        metrics = []
        for pid, path in infos:
            if path is not None and res.get(pid) and "metric" in res.get(pid, {}):
                metrics.append(float(res[pid]["metric"]))
            else:
                metrics.append(1e6)
        return metrics

    def run(self, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run BO loop: n_init random, then n_iter steps of batch_size suggestions (BoTorch qEI),
        evaluate_batch each step, minimize metric.
        Returns (train_X_norm, train_Y) in normalized and raw objective space.
        """
        self.ensure_setup()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        bounds = get_bounds_tensor(self.pbounds)
        d = bounds.shape[1]
        lower, upper = bounds[0], bounds[1]
        norm_bounds = torch.stack([torch.zeros(d), torch.ones(d)]).double()

        def unnormalize(X_norm):
            return lower + (upper - lower) * X_norm

        param_keys = sorted(self.pbounds.keys())

        # Initial random design
        train_X_norm = torch.rand(self.n_init, d, dtype=torch.double)
        initial_params = [
            {k: unnormalize(train_X_norm[i])[j].item() for j, k in enumerate(param_keys)}
            for i in range(self.n_init)
        ]
        # Store negative metric so BoTorch qEI (maximize) minimizes the actual metric
        initial_metrics = []
        for i, params in enumerate(initial_params):
            m = self.evaluate(params, perturbation_id=f"bo_init_{i}")
            initial_metrics.append(m)
        train_Y = torch.tensor([-m for m in initial_metrics], dtype=torch.double).unsqueeze(-1)

        if verbose:
            print(f"Initial design: {self.n_init} points, best metric = {min(initial_metrics):.6f}")

        for iteration in range(self.n_iter):
            gp = SingleTaskGP(
                train_X=train_X_norm,
                train_Y=train_Y,
                input_transform=Normalize(d=d, bounds=norm_bounds),
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            best_f = train_Y.max().item()  # qEI maximizes; we train on -metric
            sampler = SobolQMCNormalSampler(256, seed=self.seed + iteration)
            acq = qExpectedImprovement(model=gp, best_f=best_f, sampler=sampler)
            candidate_norm, _ = optimize_acqf(
                acq_function=acq,
                bounds=norm_bounds,
                q=self.batch_size,
                num_restarts=5,
                raw_samples=64,
            )
            candidate_params = [
                {k: unnormalize(candidate_norm[i])[j].item() for j, k in enumerate(param_keys)}
                for i in range(candidate_norm.shape[0])
            ]
            new_metrics = self.evaluate_batch(candidate_params, iteration)
            new_Y = torch.tensor([-m for m in new_metrics], dtype=torch.double).unsqueeze(-1)
            train_X_norm = torch.cat([train_X_norm, candidate_norm], dim=0)
            train_Y = torch.cat([train_Y, new_Y], dim=0)
            best_so_far = -train_Y.max().item()
            if verbose:
                print(f"  Iter {iteration + 1}: batch metric(s) = {new_metrics}, best so far = {best_so_far:.6f}")

        best_idx = train_Y.argmax().item()
        if verbose:
            best_x = unnormalize(train_X_norm[best_idx])
            print(f"Final best: metric = {-train_Y.max().item():.6f}, params = {dict(zip(param_keys, best_x.tolist()))}")
        return train_X_norm, train_Y


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VLA Bayesian Optimization (BoTorch batch)")
    parser.add_argument("--config", type=str, required=True, help="Path to main.yaml")
    parser.add_argument("--run-dir", type=str, default=None, help="Override run directory")
    parser.add_argument("--pbounds", type=str, default=None, help="JSON dict e.g. {\"dx\": [-0.05, 0.05], \"dz\": [-0.05, 0.05]}")
    parser.add_argument("--object-names", type=str, default=None, help="JSON list of object names for move")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    pbounds = json.loads(args.pbounds) if args.pbounds else None
    if pbounds and isinstance(next(iter(pbounds.values())), list):
        pbounds = {k: tuple(v) for k, v in pbounds.items()}
    object_names = json.loads(args.object_names) if args.object_names else None

    bo = VLABayesianOptimization(
        config_path=args.config,
        run_dir_override=args.run_dir,
        pbounds=pbounds,
        object_names=object_names,
        batch_size=args.batch_size,
        n_init=args.n_init,
        n_iter=args.n_iter,
        seed=args.seed,
    )
    bo.run(verbose=True)


if __name__ == "__main__":
    main()
