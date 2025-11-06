# Pipeline Launcher System - Summary

## Overview

A complete pipelined system for generating perturbed datasets in the PACE-ICE environment has been created. The system automates the entire workflow from perturbation generation through SLURM job dispatch to final evaluation.

## Created Files

### 1. `configs/main.yaml`
- Main configuration file that serves as a superset of all required parameters
- Contains configuration for:
  - Base task setup (BDDL file, prompt, model settings)
  - Recording parameters (num_demos, noise_std, etc.)
  - Perturbation specifications (BDDL spatial and language)
  - SLURM job parameters (account, partition, GPU settings, etc.)
  - Evaluation configuration (metric weights, output formats)

### 2. `scripts/launcher.py`
- Main launcher script that orchestrates the entire pipeline
- Features:
  - Creates organized directory structure in scratch folder
  - Generates perturbation files (BDDL and YAML configs)
  - Manages SLURM job queue with concurrent job limits
  - Monitors job completion and tracks status
  - Runs evaluation scripts automatically after completion
  - Handles both BDDL spatial and language perturbations

### 3. `README_LAUNCHER.md`
- Comprehensive documentation
- Includes configuration reference, workflow description, troubleshooting

## Workflow

### Step 1: Configuration
User edits `configs/main.yaml` with:
- Base task information
- Perturbation specifications
- SLURM parameters
- Evaluation settings

### Step 2: Run Launcher
```bash
python scripts/launcher.py --config configs/main.yaml
```

### Step 3: Directory Creation
Launcher creates a timestamped run directory in `$SCRATCH/vla-explainability-runs/`:
```
libero_spatial_20240101_120000/
├── bddl_files/        # Generated BDDL files
├── configs/           # Record config YAMLs
├── results/           # Output HDF5 files
├── logs/              # SLURM job logs
└── jobs/              # SLURM job scripts
```

### Step 4: Perturbation Generation
- Reads base BDDL file
- Applies specified perturbations:
  - **BDDL Spatial**: move, reorient, color, replace, distractor
  - **Language**: character/word/sentence-level text perturbations
- Validates each perturbed BDDL
- Creates corresponding config YAML for `record.py`

### Step 5: Job Dispatch
- Creates SLURM job script for each perturbation
- Submits jobs to queue (respects `max_concurrent_jobs`)
- Monitors job status via `squeue`
- Tracks completed/failed jobs
- Queues new jobs as slots become available

### Step 6: Evaluation
After all jobs complete:
- Converts HDF5 to JSON format (if configured)
- Runs trajectory analysis using `vla_metrics`
- Computes comparison metrics between unperturbed and perturbed
- Saves results to `analysis_results.json`

## Key Features

### Queue Management
- Maintains pool of running jobs (configurable max concurrent)
- Automatic job submission as slots free up
- Status polling at regular intervals
- Comprehensive job tracking

### Perturbation Support
- **BDDL Spatial Perturbations**:
  - Object movement
  - Object reorientation
  - Color changes
  - Object replacement
  - Distractor addition
- **Language Perturbations**:
  - Character-level (typos, OCR errors)
  - Word-level (swaps, deletions)
  - Sentence-level (T5 paraphrases)

### Error Handling
- Validates BDDL files before saving
- Checks job completion by verifying output files
- Tracks failed jobs separately
- Comprehensive error messages

### Integration
- Uses existing `record.py` script
- Integrates with `hdf5_to_json.py` utility
- Uses `vla_metrics` for analysis
- Compatible with existing codebase structure

## Configuration Example

```yaml
# Minimal working example
base_bddl_file: ../libero/libero/bddl_files/libero_spatial/task.bddl
base_prompt: pick up the bowl
task_suite_name: libero_spatial

perturbations:
  types:
    - bddl_spatial
  bddl_spatial:
    perturbation_specs:
      - type: move
        objects: ["akita_black_bowl_1"]

slurm:
  job_params:
    account: your_account
    partition: gpu
    gpu_type: "v100"
```

## Usage Tips

1. **Start Small**: Test with 2-3 perturbations first
2. **Monitor Logs**: Check `logs/` directory for job output
3. **Check Queue**: Use `squeue` to see job status
4. **Verify Outputs**: Check `results/` for HDF5 files after completion
5. **Review Analysis**: Check `analysis_results.json` for metrics

## Dependencies

The launcher integrates with existing code:
- `scripts/record.py` - Recording script
- `utils/hdf5_to_json.py` - Format conversion
- `explainability/vla_metrics.py` - Metrics calculation
- `libero/libero/utils/generate_perturbation_bddl.py` - BDDL perturbations
- `explainability/perturbations/language/generate_perturbations.py` - Language perturbations

## Next Steps

1. Edit `configs/main.yaml` with your parameters
2. Run `python scripts/launcher.py --config configs/main.yaml`
3. Monitor progress via logs and job queue
4. Review results in run directory

## Notes

- The launcher assumes PACE-ICE environment with SLURM
- Scratch directory defaults to `$SCRATCH` environment variable
- All paths are handled automatically (relative paths resolved)
- Job scripts are created with proper SLURM directives
- Analysis runs automatically after all jobs complete

