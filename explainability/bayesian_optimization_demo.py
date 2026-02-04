import os
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
print("MKL_DEBUG_CPU_TYPE =", os.environ['MKL_DEBUG_CPU_TYPE'])
from bayes_opt import BayesianOptimization
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


def black_box_function(x, y):
    return 1 * (x - 2) ** 2 + (y - 3) ** 2 + 10

def cos_black_box_function(x, y):
    return (np.cos(10*x) + np.sin(10*y))

def inp_black_box_function(x, y):
    a = int(input("Enter a number: "))
    return x + y + a

def saddle_black_box_function(x, y):
    return x**2 - y**2

def plot_optimizer_heatmap(x_bounds, y_bounds, step, optimizer, cmap='RdBu_r', show=True, output_path=None):
    # Ensure optimizer has a trained GP surrogate
    if not hasattr(optimizer, "_gp") or optimizer._gp is None:
        raise ValueError("Optimizer has no trained Gaussian Process. Call optimizer.maximize(...) first.")

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    x_vals = np.arange(x_min, x_max + step * 0.5, step)
    y_vals = np.arange(y_min, y_max + step * 0.5, step)

    # 2D array to hold predictions: rows correspond to y, columns to x
    Z = np.empty((len(y_vals), len(x_vals)), dtype=float)

    # Iterate through all combinations and query the optimizer's GP
    for i, yv in enumerate(y_vals):
        for j, xv in enumerate(x_vals):
            pred = optimizer._gp.predict(np.array([[xv, yv]]))
            Z[i, j] = float(pred)

    print("Predicted value range: min =", Z.min(), ", max =", Z.max())

    # Plot heatmap with red = max, blue = min
    plt.figure()
    im = plt.imshow(
        Z,
        extent=(x_min, x_max, y_min, y_max),
        origin='lower',
        aspect='auto',
        cmap=cmap,
        vmin=Z.min(),
        vmax=Z.max(),
    )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimizer predicted values heatmap')
    cb = plt.colorbar(im)
    cb.set_label('predicted value')

    # Save to file if requested
    if output_path:
        p = Path(output_path)
        # If a directory (or path ends with a separator), use a default filename
        if str(output_path).endswith(os.sep) or (p.exists() and p.is_dir()):
            p = p / 'optimizer_heatmap.png'
        # Ensure parent directory exists
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), bbox_inches='tight')

    if show:
        plt.show()

    return Z, x_vals, y_vals




if __name__ == "__main__":
    print("Starting Bayesian Optimization...")
    pbounds = {'x': (-5, 5), 'y': (-5, 5)}
    optimizer = BayesianOptimization(
        f=cos_black_box_function,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
        allow_duplicate_points=True,  # TODO: Should this be used?
    )

    print("Maximizing...")
    has_optimized = False
    prev_Z = None
    threshold = 5e-2
    while (not has_optimized):
        optimizer.maximize(
            init_points=200,  # Random Exploration
            n_iter=5,  # Bayesian Optimization Steps - NOTE: for some reason 10 creates bad graphs
        )

        print("Final result:")
        print(optimizer.max)


        Z, _, _ = plot_optimizer_heatmap(
            x_bounds=pbounds['x'],
            y_bounds=pbounds['y'],
            step=0.1,
            optimizer=optimizer,
            cmap='RdBu_r',
            show=False,
            output_path="./explainability/test/optimizer_heatmap_all_random.png"
        )

        has_optimized = prev_Z is not None and np.mean(np.abs(Z - prev_Z)) < threshold and Z.min() != Z.max()
        prev_Z = Z