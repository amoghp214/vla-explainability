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

# Add project root to path if not already present (resolve to absolute path)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.demo_loader import load_all_robot_states
from explainability.vla_metrics import calculate_vla_metric, get_dtw_trajectory_distance_matrix, calculate_wasserstein_1_dist
from explainability.vla_metrics import calculate_dtw_trajectory_difference
from explainability.data_visualization import visualize_trajectory_difference


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
    
    # Create visualizations directory
    output_path = Path(output_file)
    viz_dir = output_path.parent / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
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
            
            # Generate trajectory visualizations
            # Find best matching pair using Wasserstein distance
            if len(unperturbed_trajs) > 0 and len(perturbed_trajs) > 0:
                # Calculate distance matrix to find best match
                distance_matrix = get_dtw_trajectory_distance_matrix(
                    unperturbed_trajs, perturbed_trajs, traj_weights
                )
                
                # Use first trajectory pair for visualization (or best match)
                # For simplicity, use first unperturbed and first perturbed
                # In practice, you might want to use the optimal assignment
                unpert_traj = unperturbed_trajs[0]
                pert_traj = perturbed_trajs[0]
                
                # Calculate DTW with triangles for visualization
                try:
                    dtw_area, warp_path, triangles = calculate_dtw_trajectory_difference(
                        unpert_traj, pert_traj, traj_weights
                    )
                    
                    # Convert warp_path to lines for visualization
                    lines = None
                    if warp_path is not None and len(warp_path) > 0:
                        # Create lines connecting matched points in warp path
                        lines = []
                        for i in range(len(warp_path) - 1):
                            idx1, idx2 = warp_path[i]
                            next_idx1, next_idx2 = warp_path[i + 1]
                            # Create line segment connecting consecutive matched points
                            # Line from unperturbed point to next matched perturbed point
                            line = np.array([
                                unpert_traj[idx1, :3],  # Start: XYZ from unperturbed
                                pert_traj[next_idx2, :3]  # End: XYZ from perturbed
                            ])
                            lines.append(line)
                        if lines:
                            lines = np.array(lines)
                    
                    # Generate visualization
                    viz_file = viz_dir / f"{pert_id}_trajectory_diff.html"
                    visualize_trajectory_difference(
                        unpert_traj, pert_traj, 
                        triangles=triangles, 
                        lines=lines,
                        output_file=str(viz_file)
                    )
                    print(f"  Generated visualization: {viz_file}")
                except Exception as viz_e:
                    print(f"  Warning: Could not generate visualization: {viz_e}")
            
            results[pert_id] = {
                'metric': float(metric),
                'num_demos': len(perturbed_trajs),
                'avg_length': float(torch.mean(perturbed_lengths)),
                'visualization': str(viz_dir / f"{pert_id}_trajectory_diff.html")
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

