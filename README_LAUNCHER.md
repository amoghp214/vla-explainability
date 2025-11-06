# Pipeline Launcher Documentation

This document describes the pipelined perturbed dataset generation system for VLA explainability research in the PACE-ICE environment.

## Overview

The launcher system automates the process of:
1. Generating perturbed BDDL files and configuration YAMLs
2. Dispatching SLURM jobs to record demonstrations for each perturbation
3. Managing job queues and monitoring completion
4. Running evaluation and analysis scripts after all jobs complete

## Quick Start

### 1. Configure main.yaml

Edit `configs/main.yaml` with your specific parameters:

```yaml
# Base task configuration
base_bddl_file: ../libero/libero/bddl_files/libero_spatial/your_task.bddl
base_prompt: your task prompt here
task_suite_name: libero_spatial

# SLURM configuration
slurm:
  job_params:
    account: your_pace_account
    partition: gpu
    gpu_type: "v100"
```

### 2. Run the launcher

```bash
python scripts/launcher.py --config configs/main.yaml
```

## Directory Structure

The launcher creates a run directory in your scratch folder (or as specified in `run_base_dir`):

```
$SCRATCH/vla-explainability-runs/
└── libero_spatial_20240101_120000/
    ├── main_config.yaml          # Copy of main config
    ├── perturbation_manifest.json # List of all perturbations
    ├── job_summary.json           # Job completion summary
    ├── analysis_results.json      # Evaluation results
    │
    ├── bddl_files/                # Generated BDDL files
    │   ├── unperturbed.bddl
    │   ├── perturbed_0.bddl
    │   └── ...
    │
    ├── configs/                   # Generated record configs
    │   ├── unperturbed.yaml
    │   ├── perturbed_0.yaml
    │   └── ...
    │
    ├── results/                   # Recorded trajectories (HDF5)
    │   ├── unperturbed.hdf5
    │   ├── perturbed_0.hdf5
    │   ├── trajectories.json      # Combined JSON format
    │   └── ...
    │
    ├── logs/                      # SLURM job logs
    │   └── ...
    │
    └── jobs/                      # SLURM job scripts
        └── ...
```

## Configuration Reference

### Base Task Configuration

- `model`: Model type (default: "openvla")
- `task_suite_name`: LIBERO task suite name
- `device`: Device for inference (e.g., "cuda:0")
- `cache_dir`: HuggingFace cache directory
- `base_bddl_file`: Path to base BDDL file
- `base_prompt`: Original task prompt
- `num_demos`: Number of demonstrations per perturbation
- `noise_std`: Gaussian noise standard deviation for actions

### Perturbation Configuration

#### BDDL Spatial Perturbations

Spatial perturbations modify object positions, orientations, colors, or add distractors:

```yaml
perturbations:
  types:
    - bddl_spatial
  bddl_spatial:
    perturbation_specs:
      - type: move
        objects: ["akita_black_bowl_1"]
      - type: reorient
        objects: ["wine_bottle_1"]
      - type: color
        objects: ["akita_black_bowl_1"]
      - type: replace
        objects: ["akita_black_bowl_1"]
      - type: distractor
        count: 1
```

Available perturbation types:
- `move`: Move object to a nearby location
- `reorient`: Rotate object by random angle
- `color`: Change object color to random color
- `replace`: Replace object with different type
- `distractor`: Add new distractor object to scene

#### Language Perturbations

Language perturbations modify the task prompt:

```yaml
perturbations:
  types:
    - language
```

Generates various text perturbations:
- Character-level: keyboard typos, OCR errors, character swaps/deletions
- Word-level: word swaps, deletions, punctuation insertion
- Sentence-level: T5-based paraphrases

### SLURM Configuration

```yaml
slurm:
  max_concurrent_jobs: 4          # Max simultaneous jobs
  job_params:
    account: your_account
    partition: gpu
    time: "02:00:00"
    nodes: 1
    gpus: 1
    gpu_type: "v100"
  conda_env: vla-explainability
  poll_interval: 30              # Seconds between status checks
```

### Evaluation Configuration

```yaml
evaluation:
  enabled: true
  metric_weights:
    w_result: 1.0    # Success rate weight
    w_time: 1.0      # Episode length weight
    w_trajectory: 1.0  # Trajectory difference weight
  trajectory_weights: [1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.5]
  output_formats:
    - json
    - hdf5
  generate_visualizations: true
```

## Workflow

### 1. Perturbation Generation

The launcher:
- Reads the base BDDL file
- Applies specified perturbations
- Validates each perturbed BDDL file
- Creates corresponding config YAML files for `record.py`

### 2. Job Dispatch

The launcher:
- Creates SLURM job scripts for each perturbation
- Submits jobs to SLURM queue
- Monitors job status and manages queue
- Tracks completed/failed jobs

### 3. Evaluation

After all jobs complete:
- Converts HDF5 files to JSON (if configured)
- Computes VLA metrics comparing unperturbed vs. perturbed trajectories
- Saves analysis results to `analysis_results.json`

## Job Management

The launcher implements a queue-like system:
- Maintains a pool of running jobs (up to `max_concurrent_jobs`)
- Polls job status at regular intervals
- Submits new jobs as slots become available
- Tracks job completion and failures

## Output Files

### Trajectory Files

- `results/{perturbation_id}.hdf5`: HDF5 format with actions, states, observations
- `results/trajectories.json`: Combined JSON format for all perturbations

### Analysis Results

`analysis_results.json` contains:
```json
{
  "perturbed_0": {
    "metric": 0.1234,
    "num_demos": 5,
    "avg_length": 150.2
  },
  ...
}
```

## Troubleshooting

### Jobs Not Starting

- Check SLURM account and partition are correct
- Verify conda environment name is correct
- Check `squeue` to see if jobs are queued
- Review logs in `logs/` directory

### Perturbation Generation Fails

- Verify base BDDL file exists and is valid
- Check object names in perturbation specs match BDDL file
- Review validation errors in console output

### Evaluation Errors

- Ensure all record jobs completed successfully
- Check that output HDF5 files exist in `results/`
- Verify trajectory data format matches expected structure

## Customization

### Adding New Perturbation Types

1. Extend `_generate_bddl_spatial_perturbations()` or create new method
2. Add perturbation function to `libero/libero/utils/generate_perturbation_bddl.py`
3. Update config YAML schema

### Custom Evaluation Metrics

Modify `_run_analysis()` method to:
- Load custom metric functions
- Compute additional metrics
- Save to custom output format

## Environment Setup

The launcher expects:
- Python 3.7+
- SLURM job scheduler
- Conda environment with required packages
- Access to PACE-ICE scratch directory (`$SCRATCH`)

Required packages:
- PyYAML
- NumPy
- h5py

## Examples

### Minimal Configuration

```yaml
base_bddl_file: ../libero/libero/bddl_files/libero_spatial/task.bddl
base_prompt: pick up the bowl
task_suite_name: libero_spatial
perturbations:
  types:
    - language
slurm:
  job_params:
    account: my_account
    partition: gpu
```

### Spatial Only

```yaml
perturbations:
  types:
    - bddl_spatial
  bddl_spatial:
    perturbation_specs:
      - type: move
        objects: ["object_1"]
      - type: color
        objects: ["object_1"]
```

## Support

For issues or questions:
1. Check job logs in `logs/` directory
2. Review `job_summary.json` for completion status
3. Verify configuration in `main_config.yaml` (saved in run directory)

