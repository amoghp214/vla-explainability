"""
BoTorch Bayesian Optimization demo (batch acquisition only).

Targets BoTorch 0.8.5: uses qExpectedImprovement from monte_carlo (no logei module).
Same 2D blackbox and heatmap as explainability/bayesian_optimization_demo.py.

Heatmap quality: acquisition-only (qEI) tends to cluster evaluations near one region, so the
GP heatmap can look very localized (one smooth bowl). For a better estimate over the full domain:
use use_random_sampling=True, or set random_frac in (0,1) to mix random points with acquisition
each iteration (e.g. random_frac=0.5 for half random, half qEI).

Install: pip install botorch
"""

import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# BoTorch 0.8.5–compatible imports (no botorch.acquisition.logei in 0.8.x)
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition.monte_carlo import qExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood

# SobolQMCNormalSampler: in 0.8.5 lives in botorch.sampling.normal; many envs re-export from botorch.sampling
try:
    from botorch.sampling import SobolQMCNormalSampler
except ImportError:
    from botorch.sampling.normal import SobolQMCNormalSampler


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

def _random_candidates(batch_size: int, d: int, seed: int, dtype=torch.double):
    """Pure random sampler for new points: uniform draw in [0, 1]^d. No BoTorch sampler required."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(batch_size, d, dtype=dtype, generator=gen)


def run_botorch_optimization(
    blackbox_fn,
    bounds_dict,
    n_init: int = 10,
    n_iter: int = 5,
    batch_size: int = 2,
    seed: int = 1,
    verbose: bool = True,
    use_random_sampling: bool = False,
    random_frac: float = 0.0,
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
        use_random_sampling: if True, suggest new points uniformly at random in [0,1]^d
            instead of using the acquisition (no StochasticSampler / optimize_acqf needed).
        random_frac: when using acquisition, fraction of each batch that is random (0--1).
            E.g. 0.5 = half random, half from qEI. Improves space-filling and heatmap quality
            (less localized); 0 = pure acquisition (can be very localized).

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
        if use_random_sampling:
            mode = "random sampling"
        elif random_frac > 0:
            mode = f"qEI + {random_frac:.0%} random per batch"
        else:
            mode = "qEI acquisition"
        print(f"Initial design: {n_init} points, best value = {train_Y.max().item():.6f} (mode: {mode})")

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

        if use_random_sampling:
            # Pure random sampler for new points: no acquisition, no BoTorch sampler
            candidate_norm = _random_candidates(batch_size, d, seed + 1000 + iteration)
        else:
            n_acq = batch_size
            n_rand = 0
            if 0 < random_frac < 1:
                n_rand = max(1, int(round(batch_size * random_frac)))
                n_acq = batch_size - n_rand
            if n_acq > 0:
                best_f = train_Y.max().item()
                sampler = SobolQMCNormalSampler(256, seed=seed + iteration)
                acq = qExpectedImprovement(model=gp, best_f=best_f, sampler=sampler)
                acq_candidates, _ = optimize_acqf(
                    acq_function=acq,
                    bounds=norm_bounds,
                    q=n_acq,
                    num_restarts=5,
                    raw_samples=64,
                )
            if n_rand > 0:
                rand_candidates = _random_candidates(n_rand, d, seed + 2000 + iteration)
                candidate_norm = (
                    torch.cat([acq_candidates, rand_candidates], dim=0)
                    if n_acq > 0
                    else rand_candidates
                )
            else:
                candidate_norm = acq_candidates

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
        
        plot_botorch_heatmap(bounds_dict, )

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

    # Overlay observed points (scale marker size so 500+ points don't overlap into a few blobs)
    train_X_real = (train_X_norm.numpy() * (upper - lower) + lower)
    n_pts = train_X_real.shape[0]
    s = max(1, min(25, 5000 / n_pts))  # smaller when many points: ~20 at n=250, ~10 at n=500, ~2 at n=2500
    alpha = 0.85 if n_pts <= 80 else max(0.2, 1.2 - 0.002 * n_pts)
    plt.scatter(train_X_real[:, 0], train_X_real[:, 1], c="k", s=s, alpha=alpha, label=f"observed (n={n_pts})")
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
    import argparse
    parser = argparse.ArgumentParser(description="BoTorch BO demo (maximize cos(10x) + sin(10y))")
    parser.add_argument("--random-sampling", type=bool, default=True, help="Use pure random sampler for new points (no acquisition)")
    parser.add_argument("--random-frac", type=float, default=0.0, help="Fraction of each batch that is random when using acquisition (0-1). Improves heatmap coverage, e.g. 0.5")
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    print("BoTorch Bayesian Optimization demo (maximize cos(10x) + sin(10y))")
    print("=" * 60)

    pbounds = {"x": (-5, 5), "y": (-5, 5)}

    train_X_norm, train_Y, bounds, gp = run_botorch_optimization(
        blackbox_fn=saddle_black_box_function,
        bounds_dict=pbounds,
        n_init=args.n_init,
        n_iter=args.n_iter,
        batch_size=args.batch_size,
        seed=1,
        verbose=True,
        use_random_sampling=args.random_sampling,
        random_frac=args.random_frac,
    )

    out_dir = Path(__file__).resolve().parent / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.random_sampling:
        out_name = "botorch_heatmap_random.png"
    elif args.random_frac > 0:
        out_name = f"botorch_heatmap_mix_{args.random_frac:.0%}.png"
    else:
        out_name = "botorch_heatmap.png"
    plot_botorch_heatmap(
        pbounds,
        gp,
        train_X_norm,
        step=0.1,
        cmap="RdBu_r",
        show=False,
        output_path=out_dir / out_name,
    )
    print(f"Heatmap saved to {out_dir / out_name}")
