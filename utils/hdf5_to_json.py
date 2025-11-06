"""
Convert HDF5 demonstration files to JSON format.

This script loads robot states from an unperturbed HDF5 file and multiple
perturbed HDF5 files, then combines them into a single JSON file with the
structure: {key: [[[8 values] for steps] for demos]}.
"""

import argparse
import h5py
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from utils.demo_loader import load_all_robot_states


def numpy_to_list(data: np.ndarray) -> List[List[float]]:
    """
    Convert numpy array to nested list format for JSON serialization.
    
    Args:
        data: Numpy array with shape (num_frames, 8)
    
    Returns:
        List of lists, where each inner list contains 8 float values
    """
    return data.tolist()


def load_file_to_json_format(demo_file: str) -> List[List[List[float]]]:
    """
    Load all demos from an HDF5 file and convert to JSON format.
    
    The output format is:
    - Outer list: one entry per demo
    - Middle list: one entry per step in that demo
    - Inner list: 8 float values (EEF pos, quat, gripper)
    
    Args:
        demo_file: Path to the HDF5 demonstration file
    
    Returns:
        Nested list structure ready for JSON serialization
    """
    all_states = load_all_robot_states(demo_file)
    
    # Convert each numpy array to list format
    json_data = []
    for states in all_states:
        # states has shape (num_frames, 8)
        # Convert to list of lists
        demo_list = numpy_to_list(states)
        json_data.append(demo_list)
    
    return json_data


def extract_perturbation_key(file_path: str) -> str:
    """
    Extract perturbation key from file path.
    
    This function tries to extract a meaningful key from the filename.
    For example, if the file is "demo_perturbed_language_remove_the.hdf5",
    it would extract "perturbed_language_remove_the".
    
    Args:
        file_path: Path to the perturbed HDF5 file
    
    Returns:
        String key for the perturbation type
    """
    path = Path(file_path)
    filename = path.stem  # filename without extension
    
    # Try to extract perturbation info from filename
    if "perturbed" in filename.lower():
        # Extract everything after a known prefix pattern
        parts = filename.lower().split("perturbed")
        if len(parts) > 1:
            key = "perturbed" + parts[1]
            # Clean up separators
            key = key.replace("_", "_").strip("_")
        else:
            key = filename
    else:
        # Use filename as key, but prefix with "perturbed_" if not already there
        key = filename if filename.startswith("perturbed_") else f"perturbed_{filename}"
    
    # Replace any problematic characters
    key = key.replace(" ", "_").replace("-", "_")
    
    return key


def create_json_from_hdf5_files(
    unperturbed_file: str,
    perturbed_files: List[str],
    output_file: str,
    perturbation_keys: List[str] = None
) -> None:
    """
    Create a JSON file from unperturbed and perturbed HDF5 files.
    
    Args:
        unperturbed_file: Path to the unperturbed HDF5 file
        perturbed_files: List of paths to perturbed HDF5 files
        output_file: Path to save the output JSON file
        perturbation_keys: Optional list of custom keys for perturbations.
                          If None, keys will be extracted from filenames.
                          Must match the length of perturbed_files if provided.
    """
    result = {}
    
    # Load unperturbed data
    print(f"Loading unperturbed file: {unperturbed_file}")
    result["unperturbed"] = load_file_to_json_format(unperturbed_file)
    print(f"  Loaded {len(result['unperturbed'])} demo(s)")
    
    # Load perturbed data
    if perturbation_keys is None:
        perturbation_keys = [extract_perturbation_key(f) for f in perturbed_files]
    
    if len(perturbation_keys) != len(perturbed_files):
        raise ValueError(
            f"Number of perturbation keys ({len(perturbation_keys)}) "
            f"must match number of perturbed files ({len(perturbed_files)})"
        )
    
    for perturbed_file, key in zip(perturbed_files, perturbation_keys):
        print(f"Loading perturbed file: {perturbed_file}")
        result[key] = load_file_to_json_format(perturbed_file)
        print(f"  Loaded {len(result[key])} demo(s) with key '{key}'")
    
    # Save to JSON file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"\n✓ Successfully saved JSON to: {output_file}")
    print(f"  Total keys: {len(result)}")
    for key, demos in result.items():
        total_steps = sum(len(demo) for demo in demos)
        print(f"  {key}: {len(demos)} demo(s), {total_steps} total steps")


def main():
    """Command-line interface for the script."""
    parser = argparse.ArgumentParser(
        description="Convert HDF5 demonstration files to JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Basic usage with unperturbed and one perturbed file
            python utils/hdf5_to_json.py unperturbed.hdf5 -p perturbed_language.hdf5 -o output.json
            
            # Multiple perturbed files
            python utils/hdf5_to_json.py unperturbed.hdf5 \\
                -p perturbed_language_remove_the.hdf5 perturbed_language_replace.hdf5 \\
                -o output.json
            
            # Custom keys for perturbations
            python utils/hdf5_to_json.py unperturbed.hdf5 \\
                -p file1.hdf5 file2.hdf5 \\
                -k perturbed_type1 perturbed_type2 \\
                -o output.json
            """
    )
    
    parser.add_argument(
        "unperturbed_file",
        type=str,
        help="Path to the unperturbed HDF5 file"
    )
    parser.add_argument(
        "-p", "--perturbed",
        nargs="+",
        required=True,
        help="Path(s) to perturbed HDF5 file(s)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "-k", "--keys",
        nargs="+",
        default=None,
        help="Custom keys for perturbations (must match number of perturbed files)"
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not Path(args.unperturbed_file).exists():
        raise FileNotFoundError(f"Unperturbed file not found: {args.unperturbed_file}")
    
    for perturbed_file in args.perturbed:
        if not Path(perturbed_file).exists():
            raise FileNotFoundError(f"Perturbed file not found: {perturbed_file}")
    
    # Create JSON file
    create_json_from_hdf5_files(
        args.unperturbed_file,
        args.perturbed,
        args.output,
        args.keys
    )


if __name__ == "__main__":
    main()

