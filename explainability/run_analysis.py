"""
Analysis script for comparing unperturbed and perturbed trajectories.

This module computes VLA metrics comparing unperturbed baseline trajectories
with perturbed trajectories.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path if not already present
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.demo_loader import load_all_robot_states
from explainability.vla_metrics import calculate_vla_metric


def run_analysis(
    unperturbed_file: str,
    perturbed_files: List[str],
    output_file: str,
    metric_weights: Dict[str, float],
    trajectory_weights: List[float],
    project_root: str = None
) -> Dict[str, Any]:
    """
    Run trajectory analysis comparing unperturbed and perturbed episodes.
    
    Args:
        unperturbed_file: Path to unperturbed HDF5 file
        perturbed_files: List of paths to perturbed HDF5 files
        output_file: Path to save analysis results JSON
        metric_weights: Dictionary with w_result, w_time, w_trajectory weights
        trajectory_weights: List of 8 weights for trajectory dimensions
        project_root: Optional project root path (for imports)
    
    Returns:
        Dictionary with analysis results
    """
    # Add project root to path if specified
    if project_root and str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Load unperturbed trajectories
    print(f"[INFO] Loading unperturbed trajectories from {unperturbed_file}")
    unperturbed_trajs = load_all_robot_states(unperturbed_file)
    
    # Compute results (simplified: assume success if trajectory completes)
    # In practice, you'd check actual task completion from environment
    unperturbed_results = torch.ones(len(unperturbed_trajs))
    unperturbed_lengths = torch.tensor([len(t) for t in unperturbed_trajs]).float()
    
    results = {}
    
    # Convert trajectory weights to numpy array
    traj_weights = np.array(trajectory_weights)
    
    # Analyze each perturbed file
    for pert_file in perturbed_files:
        pert_id = Path(pert_file).stem
        print(f"[INFO] Analyzing {pert_id}...")
        
        try:
            perturbed_trajs = load_all_robot_states(pert_file)
            perturbed_results = torch.ones(len(perturbed_trajs))
            perturbed_lengths = torch.tensor([len(t) for t in perturbed_trajs]).float()
            
            # Calculate metric
            metric = calculate_vla_metric(
                unperturbed_episode_results=unperturbed_results,
                perturbed_episode_results=perturbed_results,
                unperturbed_episode_lengths=unperturbed_lengths,
                perturbed_episode_lengths=perturbed_lengths,
                unperturbed_trajectories=unperturbed_trajs,
                perturbed_trajectories=perturbed_trajs,
                w_result=metric_weights['w_result'],
                w_time=metric_weights['w_time'],
                w_trajectory=metric_weights['w_trajectory'],
                W=traj_weights
            )
            
            results[pert_id] = {
                'metric': float(metric),
                'num_demos': len(perturbed_trajs),
                'avg_length': float(torch.mean(perturbed_lengths))
            }
            print(f"  Metric: {metric:.4f}")
        except Exception as e:
            print(f"  Error analyzing {pert_id}: {e}")
            import traceback
            traceback.print_exc()
            results[pert_id] = {'error': str(e)}
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[INFO] Analysis complete. Results saved to {output_file}")
    return results


def main():
    """Command-line interface for running analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run trajectory analysis comparing unperturbed and perturbed episodes"
    )
    parser.add_argument(
        "--unperturbed",
        type=str,
        required=True,
        help="Path to unperturbed HDF5 file"
    )
    parser.add_argument(
        "--perturbed",
        type=str,
        nargs="+",
        required=True,
        help="Paths to perturbed HDF5 files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save analysis results JSON"
    )
    parser.add_argument(
        "--metric-weights",
        type=str,
        default=None,
        help="JSON string or path to JSON file with metric weights (w_result, w_time, w_trajectory)"
    )
    parser.add_argument(
        "--trajectory-weights",
        type=str,
        default=None,
        help="JSON string or path to JSON file with trajectory weights (8 values)"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (defaults to parent of script directory)"
    )
    
    args = parser.parse_args()
    
    # Parse metric weights
    if args.metric_weights:
        if Path(args.metric_weights).exists():
            with open(args.metric_weights, 'r') as f:
                metric_weights = json.load(f)
        else:
            metric_weights = json.loads(args.metric_weights)
    else:
        metric_weights = {
            'w_result': 1.0,
            'w_time': 1.0,
            'w_trajectory': 1.0
        }
    
    # Parse trajectory weights
    if args.trajectory_weights:
        if Path(args.trajectory_weights).exists():
            with open(args.trajectory_weights, 'r') as f:
                trajectory_weights = json.load(f)
        else:
            trajectory_weights = json.loads(args.trajectory_weights)
    else:
        trajectory_weights = [1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.5]
    
    # Run analysis
    run_analysis(
        unperturbed_file=args.unperturbed,
        perturbed_files=args.perturbed,
        output_file=args.output,
        metric_weights=metric_weights,
        trajectory_weights=trajectory_weights,
        project_root=args.project_root
    )


if __name__ == "__main__":
    main()

