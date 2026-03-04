"""
BoTorch random-design demo: sample all points randomly, fit one BoTorch model per model type.

Compare different BoTorch GP models (SingleTaskGP, FixedNoiseGP, SingleTaskVariationalGP,
SaasFullyBayesianSingleTaskGP, etc.) across objective functions.
- Generates n_design points uniformly in the domain, evaluates the blackbox once.
- Fits one BoTorch model per model type on the same data.
- Plots heatmaps in a grid: rows = objective functions, columns = model types.

Which model for trajectory-difference blackbox (move perturbation -> scalar metric)?
- SingleTaskGP: Default choice; infers homoskedastic noise. Use when you have a single
  metric per perturbation and unknown noise level (e.g. one rollout per point).
- FixedNoiseGP: Use when you have (or assume) known observation noise (e.g. from
  repeated rollouts or a small constant for near-deterministic metrics).
- HeteroskedasticSingleTaskGP: Use when noise varies with input (e.g. some perturbations
  lead to more variable trajectory outcomes).
- SingleTaskVariationalGP: Scalable for large n_design (e.g. 1000+); approximate posterior.
  Prefer when fitting is slow with exact GPs.
- SaasFullyBayesianSingleTaskGP: Better uncertainty calibration and sample efficiency in
  higher-dimensional perturbation spaces; more compute (NUTS). Good when d is large and
  evaluations are expensive.

Usage:
  python explainability/botorch_random_demo.py --n-design 400 --seed 1
  python explainability/botorch_random_demo.py --objectives cos saddle --models SingleTaskGP FixedNoiseGP --n-design 500
"""

import argparse
from pathlib import Path
from typing import Callable, Dict, Tuple, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, SpectralMixtureKernel, RQKernel

try:
    from botorch.models import HeteroskedasticSingleTaskGP
    _HAS_HETEROSKEDASTIC = True
except ImportError:
    _HAS_HETEROSKEDASTIC = False

try:
    from botorch.models import SingleTaskVariationalGP
    _HAS_VARIATIONAL = True
except ImportError:
    _HAS_VARIATIONAL = False

try:
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    from botorch.fit import fit_fully_bayesian_model_nuts
    _HAS_SAAS = True
except ImportError:
    _HAS_SAAS = False


# ---------------------------------------------------------------------------
# Objective functions (blackbox)
# ---------------------------------------------------------------------------

def objective_cos(x: float, y: float) -> float:
    """Oscillatory: cos(10x) + sin(10y)."""
    return float(np.cos(10 * x) + np.sin(10 * y))


def objective_saddle(x: float, y: float) -> float:
    """Saddle: x^2 - y^2."""
    return x ** 2 - y ** 2


def objective_quadratic(x: float, y: float) -> float:
    """Bowl: (x-2)^2 + (y-3)^2 + 10."""
    return 1.0 * (x - 2) ** 2 + (y - 3) ** 2 + 10.0


def objective_rosenbrock(x: float, y: float) -> float:
    """Rosenbrock (scaled): classic test function."""
    return float(-((1 - x) ** 2 + 100 * (y - x ** 2) ** 2))


OBJECTIVES: Dict[str, Callable[[float, float], float]] = {
    "cos": objective_cos,
    "saddle": objective_saddle,
    "quadratic": objective_quadratic,
    "rosenbrock": objective_rosenbrock,
}


# ---------------------------------------------------------------------------
# BoTorch model factory
# ---------------------------------------------------------------------------

def get_bounds_tensor(bounds_dict: Dict[str, Tuple[float, float]]) -> torch.Tensor:
    keys = sorted(bounds_dict.keys())
    lower = torch.tensor([bounds_dict[k][0] for k in keys], dtype=torch.double)
    upper = torch.tensor([bounds_dict[k][1] for k in keys], dtype=torch.double)
    return torch.stack([lower, upper])


def build_and_fit_model(
    model_name: str,
    train_X_norm: torch.Tensor,
    train_Y: torch.Tensor,
    d: int,
    kernel_name: str = "",
) -> Any:
    """
    Build and fit a BoTorch model. Inputs are in [0,1]^d; uses Normalize and Standardize.
    Accepts kernel_name to manually select the gpytorch kernel used.
    Returns the fitted model (with .posterior(X)).
    """
    def _make_kernel(name: str, dim: int):
        n = (name or "").lower()
        if n in ("matern1.5", "matern15"):
            base = MaternKernel(nu=1.5, ard_num_dims=dim)
        elif n in ("matern2.5", "matern25"):
            base = MaternKernel(nu=2.5, ard_num_dims=dim)
        elif n in ("rq", "rationalquadratic"):
            base = RQKernel(ard_num_dims=dim)
        elif n in ("sm", "spectralmixture", "spectral_mixture"):
            # spectral mixture needs num_mixtures; keep small default
            return ScaleKernel(SpectralMixtureKernel(num_mixtures=4, ard_num_dims=dim))
        else:
            # default to RBF (squared exponential)
            base = RBFKernel(ard_num_dims=dim)
        return ScaleKernel(base)

    norm_bounds = torch.stack([torch.zeros(d), torch.ones(d)]).double()

    gp = None
    # Build model
    if model_name == "SingleTaskGP":
        gp = SingleTaskGP(
            train_X=train_X_norm,
            train_Y=train_Y,
            input_transform=Normalize(d=d, bounds=norm_bounds),
            outcome_transform=Standardize(m=1),
        )
    elif model_name == "FixedNoiseGP":
        # Fixed noise: use small constant variance (e.g. 1e-6 for near-deterministic)
        train_Yvar = torch.full_like(train_Y, 1e-6)
        gp = FixedNoiseGP(
            train_X=train_X_norm,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            input_transform=Normalize(d=d, bounds=norm_bounds),
            outcome_transform=Standardize(m=1),
        )
    elif model_name == "HeteroskedasticSingleTaskGP" and _HAS_HETEROSKEDASTIC:
        # Heteroskedastic: pass per-point noise (here constant for simplicity)
        train_Yvar = torch.full_like(train_Y, 1e-4)
        gp = HeteroskedasticSingleTaskGP(
            train_X=train_X_norm,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            input_transform=Normalize(d=d, bounds=norm_bounds),
            outcome_transform=Standardize(m=1),
        )

    elif model_name == "SingleTaskVariationalGP" and _HAS_VARIATIONAL:
        gp = SingleTaskVariationalGP(
            train_X=train_X_norm,
            train_Y=train_Y,
            input_transform=Normalize(d=d, bounds=norm_bounds),
            outcome_transform=Standardize(m=1),
        )

    elif model_name == "SaasFullyBayesianSingleTaskGP" and _HAS_SAAS:
        # NUTS often uses float32; use float for compatibility
        X_f = train_X_norm.float()
        Y_f = train_Y.float()
        gp = SaasFullyBayesianSingleTaskGP(
            train_X=X_f,
            train_Y=Y_f,
            outcome_transform=Standardize(m=1),
        )
        fit_fully_bayesian_model_nuts(
            gp,
            warmup_steps=64,
            num_samples=32,
            thinning=2,
        )
        return gp

    else:
        available = ["SingleTaskGP", "FixedNoiseGP"]
        if _HAS_HETEROSKEDASTIC:
            available.append("HeteroskedasticSingleTaskGP")
        if _HAS_VARIATIONAL:
            available.append("SingleTaskVariationalGP")
        if _HAS_SAAS:
            available.append("SaasFullyBayesianSingleTaskGP")
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    # assign chosen kernel (try to attach to gp or gp.model)
    kernel = _make_kernel(kernel_name, d) if kernel_name else None
    if kernel is not None:
        if hasattr(gp, "covar_module"):
            gp.covar_module = kernel
        elif hasattr(gp, "model") and hasattr(gp.model, "covar_module"):
            gp.model.covar_module = kernel
    # Fit model using the appropriate marginal/variational objective
    if model_name == "SingleTaskVariationalGP" and _HAS_VARIATIONAL:
        mll = VariationalELBO(gp.likelihood, gp.model, num_data=train_X_norm.shape[-2])
    else:
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    
    fit_gpytorch_mll(mll)

    return gp


# ---------------------------------------------------------------------------
# Data generation and fitting
# ---------------------------------------------------------------------------

def evaluate_blackbox(X: torch.Tensor, fn: Callable) -> torch.Tensor:
    """X: (n, d) in real space. Returns (n, 1)."""
    X_np = X.numpy()
    y = np.array([fn(*row) for row in X_np], dtype=np.float64)
    return torch.from_numpy(y).double().unsqueeze(-1)

def calculate_rms_error(model: Any, X: torch.Tensor, Y: torch.Tensor) -> float:
    """Calculate root mean squared error of model predictions on BO data."""
    with torch.no_grad():
        y_pred = model.posterior(X).mean.squeeze(-1)
        y_true = Y.squeeze(-1)
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
    return rmse


def run_random_design(
    objective_fn: Callable,
    bounds_dict: Dict[str, Tuple[float, float]],
    n_design: int,
    seed: int,
    model_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    """
    Sample n_design points uniformly, evaluate blackbox, fit one BoTorch model of the given type.
    Returns train_X_norm, train_Y, bounds, fitted_model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    bounds = get_bounds_tensor(bounds_dict)
    d = bounds.shape[1]
    lower, upper = bounds[0], bounds[1]

    gen = torch.Generator(device="cpu").manual_seed(seed)
    train_X_norm = torch.rand(n_design, d, dtype=torch.double, generator=gen)
    train_X = lower + (upper - lower) * train_X_norm
    train_Y = evaluate_blackbox(train_X, objective_fn)

    model = build_and_fit_model(model_name, train_X_norm, train_Y, d)
    return train_X_norm, train_Y, bounds, model


# ---------------------------------------------------------------------------
# Heatmap plotting
# ---------------------------------------------------------------------------

def plot_heatmap(
    bounds_dict: Dict[str, Tuple[float, float]],
    model: Any,
    train_X_norm: torch.Tensor,
    train_Y: torch.Tensor,
    title: str,
    step: float = 0.02,
    cmap: str = "RdBu_r",
    ax=None,
) -> np.ndarray:
    """Draw heatmap of model posterior mean. model must have .posterior(X)."""
    bounds = get_bounds_tensor(bounds_dict)
    x_min, x_max = bounds[0, 0].item(), bounds[1, 0].item()
    y_min, y_max = bounds[0, 1].item(), bounds[1, 1].item()
    lower, upper = bounds[0].numpy(), bounds[1].numpy()

    x_vals = np.arange(x_min, x_max + step * 0.5, step)
    y_vals = np.arange(y_min, y_max + step * 0.5, step)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_real = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid_norm = (grid_real - lower) / (upper - lower)
    X_grid = torch.from_numpy(grid_norm).double()
    # Some models (e.g. SaasFullyBayesianSingleTaskGP) use float; cast for compatibility
    ref = None
    if hasattr(model, "train_inputs") and model.train_inputs is not None and len(model.train_inputs) > 0:
        ref = model.train_inputs[0]
    elif hasattr(model, "model") and hasattr(model.model, "train_inputs") and model.model.train_inputs is not None:
        ref = model.model.train_inputs[0]
    if ref is not None:
        X_grid = X_grid.to(dtype=ref.dtype, device=ref.device)

    with torch.no_grad():
        posterior = model.posterior(X_grid)
        mean = posterior.mean
        # SaasFullyBayesianSingleTaskGP returns (num_mcmc_samples, n, 1); average over MCMC dim
        if mean.dim() == 3:
            mean = mean.mean(dim=0)
        mean = mean.squeeze(-1).numpy()

    Z = mean.reshape(len(y_vals), len(x_vals))

    if ax is None:
        plt.figure()
        ax = plt.gca()
    im = ax.imshow(
        Z,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=Z.min(),
        vmax=Z.max(),
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="predicted value")

    train_X_real = train_X_norm.numpy() * (upper - lower) + lower
    n_pts = train_X_real.shape[0]
    s = max(1, min(25, 5000 / n_pts))
    alpha = 0.85 if n_pts <= 80 else max(0.2, 1.2 - 0.002 * n_pts)
    # draw colored points whose facecolor encodes the true metric, with a black edge
    train_Y = train_Y.detach().cpu().numpy().squeeze()
    if train_Y.ndim > 1:
        train_Y = train_Y.reshape(-1)
    # Plot with colormap representing the true VLA metric and a black outline for each point.
    ax.scatter(
        train_X_real[:, 0],
        train_X_real[:, 1],
        c=train_Y,
        cmap=cmap,
        edgecolors="k",
        linewidths=0.4,
        s=s,
        alpha=alpha,
        vmin=Z.min(),
        vmax=Z.max(),
    )
    # ax.scatter(train_X_real[:, 0], train_X_real[:, 1], c="k", s=s, alpha=alpha)

    return Z


# ---------------------------------------------------------------------------
# Main: grid of (objectives x models)
# ---------------------------------------------------------------------------

def _available_models() -> list:
    out = ["SingleTaskGP", "FixedNoiseGP"]
    if _HAS_HETEROSKEDASTIC:
        out.append("HeteroskedasticSingleTaskGP")
    if _HAS_VARIATIONAL:
        out.append("SingleTaskVariationalGP")
    if _HAS_SAAS:
        out.append("SaasFullyBayesianSingleTaskGP")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Random-design demo: compare BoTorch models across objectives"
    )
    parser.add_argument("--n-design", type=int, default=400, help="Number of random design points")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["cos", "saddle", "quadratic"],
        choices=list(OBJECTIVES.keys()),
        help="Objective functions to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["SingleTaskGP", "FixedNoiseGP"],
        help="BoTorch model names: SingleTaskGP, FixedNoiseGP, HeteroskedasticSingleTaskGP, "
        "SingleTaskVariationalGP, SaasFullyBayesianSingleTaskGP (depending on availability)",
    )
    parser.add_argument("--bounds", type=str, default="-5,5", help="x,y bounds as low,high (e.g. -5,5)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for heatmaps")
    parser.add_argument("--step", type=float, default=0.05, help="Grid step for heatmap")
    args = parser.parse_args()

    try:
        low, high = map(float, args.bounds.split(","))
        bounds_dict = {"x": (low, high), "y": (low, high)}
    except Exception:
        bounds_dict = {"x": (-5, 5), "y": (-5, 5)}

    out_dir = Path(args.out_dir or (Path(__file__).resolve().parent / "test" / "random_demo"))
    out_dir.mkdir(parents=True, exist_ok=True)

    objectives = args.objectives
    models = args.models
    n_row, n_col = len(objectives), len(models)

    fig, axes = plt.subplots(n_row, n_col, figsize=(4 * n_col, 4 * n_row), squeeze=False)
    if n_row == 1 and n_col == 1:
        axes = axes.reshape(1, 1)

    for i, obj_name in enumerate(objectives):
        obj_fn = OBJECTIVES[obj_name]
        for j, model_name in enumerate(models):
            train_X_norm, train_Y, bounds, model = run_random_design(
                obj_fn, bounds_dict, args.n_design, args.seed + i * 100 + j, model_name
            )
            title = f"{obj_name} / {model_name}"
            plot_heatmap(
                bounds_dict, model, train_X_norm, title, step=args.step, ax=axes[i, j]
            )

    plt.tight_layout()
    out_path = out_dir / "heatmap_grid.png"
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
    print(f"Saved grid to {out_path}")

    for i, obj_name in enumerate(objectives):
        for j, model_name in enumerate(models):
            train_X_norm, train_Y, bounds, model = run_random_design(
                OBJECTIVES[obj_name], bounds_dict, args.n_design, args.seed + i * 100 + j, model_name
            )
            plt.figure(figsize=(6, 5))
            plot_heatmap(bounds_dict, model, train_X_norm, f"{obj_name} — {model_name}", step=args.step)
            out_path = out_dir / f"heatmap_{obj_name}_{model_name}.png"
            plt.savefig(str(out_path), bbox_inches="tight")
            plt.close()
    print(f"Saved individual heatmaps to {out_dir}")


if __name__ == "__main__":
    main()
