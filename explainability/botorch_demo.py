"""
BoTorch Bayesian Optimization demo (batch acquisition only).

This script uses BoTorch with batch (q-) acquisition for parallel evaluations.
Same 2D blackbox and heatmap as explainability/bayesian_optimization_demo.py.

Install BoTorch (run in your env; do not add to requirements yet):
  pip install botorch
"""

import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# BoTorch: batch acquisition only
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood


# ---------------------------------------------------------------------------
# Blackbox functions (same as bayesian_optimization_demo.py)
# ---------------------------------------------------------------------------

def black_box_function(x: float, y: float) -> float:
    return 1.0 * (x - 2) ** 2 + (y - 3) ** 2 + 10.0


def cos_black_box_function(x: float, y: float) -> float:
    return float(np.cos(10 * x) + np.sin(10 * y))


def saddle_black_box_function(x: float, y: float) -> float:
    return x ** 2 - y ** 2


# ---------------------------------------------------------------------------
# Bounds and evaluation (BoTorch uses tensors; we keep bounds in real space)
# ---------------------------------------------------------------------------

def get_bounds_tensor(bounds_dict):
    """Convert dict like {'x': (-5, 5), 'y': (-5, 5)} to 2 x d tensor."""
    keys = sorted(bounds_dict.keys())
    lower = torch.tensor([bounds_dict[k][0] for k in keys], dtype=torch.double)
    upper = torch.tensor([bounds_dict[k][1] for k in keys], dtype=torch.double)
    return torch.stack([lower, upper])  # shape (2, d)


def evaluate_blackbox(X: torch.Tensor, fn) -> torch.Tensor:
    """
    Evaluate blackbox at points X.
    X: (n, 2) tensor in real space (x, y).
    Returns: (n, 1) tensor of objective values.
    """
    X_np = X.numpy()
    y = np.array([fn(float(X_np[i, 0]), float(X_np[i, 1])) for i in range(X_np.shape[0])])
    return torch.from_numpy(y).double().unsqueeze(-1)  # (n, 1)


# ---------------------------------------------------------------------------
# BoTorch optimization loop (maximization, like the original demo)
# ---------------------------------------------------------------------------

def run_botorch_optimization(
    blackbox_fn,
    bounds_dict,
    n_init: int = 10,
    n_iter: int = 5,
    batch_size: int = 2,
    seed: int = 1,
    verbose: bool = True,
):
    """
    Run Bayesian optimization with BoTorch (maximization), batch acquisition only.

    Args:
        blackbox_fn: function(x, y) -> float to maximize.
        bounds_dict: e.g. {'x': (-5, 5), 'y': (-5, 5)}.
        n_init: number of random initial points.
        n_iter: number of BO steps (each step suggests batch_size points).
        batch_size: q in BoTorch; number of points to suggest per step (parallel eval).
        seed: random seed.
        verbose: print progress.

    Returns:
        train_X_norm, train_Y, bounds, gp (last GP fit).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    bounds = get_bounds_tensor(bounds_dict)  # (2, 2)
    d = bounds.shape[1]
    # Normalize to [0, 1]^d for the GP (BoTorch convention)
    lower, upper = bounds[0], bounds[1]
    norm_bounds = torch.stack([torch.zeros(d), torch.ones(d)]).double()

    def unnormalize(X_norm):
        """X_norm in [0,1]^d -> real space."""
        return lower + (upper - lower) * X_norm

    # Initial random design in [0,1]^d
    train_X_norm = torch.rand(n_init, d, dtype=torch.double)
    train_X = unnormalize(train_X_norm)
    train_Y = evaluate_blackbox(train_X, blackbox_fn)

    if verbose:
        print(f"Initial design: {n_init} points, best value = {train_Y.max().item():.6f}")

    for iteration in range(n_iter):
        # Fit GP on normalized inputs (recommended for stability)
        gp = SingleTaskGP(
            train_X=train_X_norm,
            train_Y=train_Y,
            input_transform=Normalize(d=d, bounds=norm_bounds),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        best_f = train_Y.max().item()
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]), seed=seed + iteration)
        acq = qLogExpectedImprovement(
            model=gp, best_f=best_f, sampler=sampler, fat=True
        )
        candidate_norm, _ = optimize_acqf(
            acq_function=acq,
            bounds=norm_bounds,
            q=batch_size,
            num_restarts=5,
            raw_samples=64,
        )

        # Evaluate the q points (here: sequential in-process; for VLA: dispatch q
        # SLURM jobs in parallel, wait for all, then collect q metrics as new_Y).
        candidate_real = unnormalize(candidate_norm)
        new_Y = evaluate_blackbox(candidate_real, blackbox_fn)

        # Append to data
        train_X_norm = torch.cat([train_X_norm, candidate_norm], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)
        train_X = unnormalize(train_X_norm)

        if verbose:
            print(f"  Iter {iteration + 1}: suggested {batch_size} point(s), "
                  f"new values = {new_Y.squeeze().tolist()}, "
                  f"best so far = {train_Y.max().item():.6f}")

    best_idx = train_Y.argmax().item()
    best_x_norm = train_X_norm[best_idx]
    best_x_real = train_X[best_idx]
    if verbose:
        print(f"Final best: value = {train_Y.max().item():.6f}, "
              f"x = ({best_x_real[0].item():.4f}, {best_x_real[1].item():.4f})")

    return train_X_norm, train_Y, bounds, gp


# ---------------------------------------------------------------------------
# Heatmap of GP posterior (similar to bayesian_optimization_demo)
# ---------------------------------------------------------------------------

def plot_botorch_heatmap(
    bounds_dict,
    gp_model,
    train_X_norm,
    step=0.02,
    cmap="RdBu_r",
    show=True,
    output_path=None,
):
    """
    Plot heatmap of GP posterior mean over the 2D domain.
    Assumes 2D and that the GP was fit in normalized [0,1]^2 space.
    """
    bounds = get_bounds_tensor(bounds_dict)
    x_min, x_max = bounds[0, 0].item(), bounds[1, 0].item()
    y_min, y_max = bounds[0, 1].item(), bounds[1, 1].item()

    x_vals = np.arange(x_min, x_max + step * 0.5, step)
    y_vals = np.arange(y_min, y_max + step * 0.5, step)

    # Build grid in real space, then normalize for the GP
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_real = np.stack([xx.ravel(), yy.ravel()], axis=1)
    lower = bounds[0].numpy()
    upper = bounds[1].numpy()
    grid_norm = (grid_real - lower) / (upper - lower)
    X_grid = torch.from_numpy(grid_norm).double()

    with torch.no_grad():
        posterior = gp_model.posterior(X_grid)
        mean = posterior.mean.squeeze(-1).numpy()

    Z = mean.reshape(len(y_vals), len(x_vals))
    print(f"Predicted value range: min = {Z.min():.4f}, max = {Z.max():.4f}")

    plt.figure()
    im = plt.imshow(
        Z,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=Z.min(),
        vmax=Z.max(),
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("BoTorch GP posterior mean (optimizer surrogate)")
    plt.colorbar(im, label="predicted value")

    # Overlay observed points
    train_X_real = (train_X_norm.numpy() * (upper - lower) + lower)
    plt.scatter(train_X_real[:, 0], train_X_real[:, 1], c="k", s=20, alpha=0.8, label="observed")
    plt.legend()

    if output_path:
        p = Path(output_path)
        if str(output_path).endswith(os.sep) or (p.exists() and p.is_dir()):
            p = p / "botorch_heatmap.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), bbox_inches="tight")

    if show:
        plt.show()

    return Z, x_vals, y_vals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("BoTorch Bayesian Optimization demo (maximize cos(10x) + sin(10y))")
    print("=" * 60)

    pbounds = {"x": (-5, 5), "y": (-5, 5)}

    train_X_norm, train_Y, bounds, gp = run_botorch_optimization(
        blackbox_fn=cos_black_box_function,
        bounds_dict=pbounds,
        n_init=20,
        n_iter=5,
        batch_size=2,
        seed=1,
        verbose=True,
    )

    out_dir = Path(__file__).resolve().parent / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_botorch_heatmap(
        pbounds,
        gp,
        train_X_norm,
        step=0.1,
        cmap="RdBu_r",
        show=False,
        output_path=out_dir / "botorch_heatmap.png",
    )
    print(f"Heatmap saved to {out_dir / 'botorch_heatmap.png'}")
