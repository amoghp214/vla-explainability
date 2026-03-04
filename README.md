# VLA Explainability Pipeline

Pipeline for generating perturbed LIBERO datasets and recording OpenVLA demonstrations for VLA explainability research. Supports batch runs on HPC (e.g. PACE-ICE with SLURM) and local step-by-step runs.

---

## Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Random-design mode (metric vs translation heatmap)](#random-design-mode-metric-vs-translation-heatmap)
- [Configuration](#configuration)
- [Perturbation types and spec dicts](#perturbation-types-and-spec-dicts)
- [Local runs (no SLURM)](#local-runs-no-slurm)
- [Directory structure](#directory-structure)
- [Troubleshooting](#troubleshooting)
- [Customization](#customization)
- [Citation and license](#citation-and-license)

---

## Overview

The system:

1. **Generates perturbations** — Reads a base BDDL task, applies spatial and/or language perturbations, writes BDDL files and record configs. Init regions use a small fixed extent (`max_init_range_m`) so the env barely changes across runs and MuJoCo gets valid geom sizes.
2. **Dispatches jobs** — Submits SLURM jobs to record each perturbation with OpenVLA, manages concurrency and completion.
3. **Post-processes** — Optionally renders videos from HDF5 recordings (controlled by `render_videos` in config) and runs evaluation (trajectory comparison, metrics).

You can run the full pipeline (generate → SLURM → videos → evaluation) or only generate files and run/playback one perturbation locally. A **random-design** mode generates random (x, z) translation perturbations, runs the pipeline, and produces a heatmap of the VLA metric vs translation (see [Random-design mode](#random-design-mode-metric-vs-translation-heatmap)).

---

## Quick Start

### 1. Configure

Edit `configs/main.yaml`:

- **Base task:** `base_bddl_file`, `base_prompt`, `task_suite_name`
- **Paths:** `cache_dir` (HuggingFace), `run_base_dir` (optional; default `$SCRATCH/vla-explainability-runs`)
- **SLURM:** `slurm.job_params` (account, partition, time, gpus), `slurm.conda_env`, `slurm.module_load`
- **Perturbations:** `perturbations.types`, `perturbations.bddl_spatial.perturbation_specs` (see below)

### 2. Run full pipeline

```bash
python scripts/launcher.py --config configs/main.yaml
```

This creates a timestamped run directory, generates all BDDL/configs, submits SLURM jobs, then runs optional video rendering and evaluation when jobs finish. Set `render_videos: false` in the config to skip rendering videos from HDF5s.

### 3. Generate only (no jobs)

To only write BDDL and configs to a local folder:

```bash
python scripts/launcher.py --config configs/main.yaml --generate-only --run-dir ./local_run
```

Then record and playback manually (see [Local runs](#local-runs-no-slurm)).

---

## Random-design mode (metric vs translation heatmap)

Random-design treats **one perturbation run as one black-box evaluation**: the input is (x, z) **delta** translation (move) for a single object from its original BDDL position, and the output is the **VLA metric** from trajectory analysis (unperturbed vs perturbed). The launcher generates `n_design` random (dx, dz) deltas in the given bounds, places the object at **(original_center + dx, original_center + dz)**, dispatches record jobs, runs evaluation, fits a BoTorch GP, and saves a **heatmap of metric vs absolute (x, z) position** (i.e. original object position + delta).

### Usage

```bash
# Full pipeline: generate random points → SLURM → evaluate → heatmap
# Bounds default to (-max_move_m, max_move_m) from config (see Configuration)
python scripts/launcher.py --config configs/main.yaml --random-design --n-design 20

# Override bounds (comma-separated low,high for both x and z, in meters)
python scripts/launcher.py --config configs/main.yaml --random-design --n-design 20 --bounds -0.05,0.05

# Generate only (then record and re-run without --generate-only to get heatmap)
python scripts/launcher.py --config configs/main.yaml --random-design --generate-only --run-dir ./rd_run
```

### Options

| Option | Description |
|--------|-------------|
| `--random-design` | Enable random-design pipeline (random x, z move → metric → heatmap). |
| `--n-design` | Number of random design points (default from config or 20). |
| `--seed` | Random seed for sampling (default from config or 1). |
| `--bounds` | Optional. Comma-separated `low,high` in meters for both x and z. If omitted, uses **`(-max_move_m, max_move_m)`** from `perturbations.bddl_spatial.max_move_m` in the config. |
| `--objects` | Object name for move (exactly one). If omitted, uses first move object from `perturbation_specs` in config. |

### Config (optional)

You can set defaults in `configs/main.yaml` under `random_design`:

```yaml
random_design:
  n_design: 20
  seed: 1
  bounds_x: [-0.05, 0.05]   # optional; default from max_move_m
  bounds_z: [-0.05, 0.05]   # optional; default from max_move_m
  object_names: ["akita_black_bowl_1"]   # exactly one object
```

If `bounds_x` / `bounds_z` are omitted, they default to `(-max_move_m, max_move_m)` from `perturbations.bddl_spatial.max_move_m`.

### Outputs

In the run directory you get:

- **`random_design_points.json`** — List of `{id, x, z}` for each design point (e.g. `rd_0`, `rd_1`, ...). The **x, z values are deltas** (m) from the object’s original BDDL center.
- **`heatmap_metric_vs_translation.png`** — Heatmap of the GP-predicted VLA metric over **(x, z) absolute position** (original object center + delta). Axes are labeled “x position (m)” and “z position (m).”

Plus the usual `bddl_files/`, `configs/`, `results/`, `analysis_results.json`, etc. Perturbation IDs are `unperturbed`, `control`, and `rd_0`, `rd_1`, ... for the random design points.

---

### Base task and recording

| Key | Description |
|-----|-------------|
| `model` | Model type (e.g. `openvla`) |
| `task_suite_name` | LIBERO suite: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, etc. |
| `device` | e.g. `cuda:0` |
| `cache_dir` | HuggingFace cache for models |
| `base_bddl_file` | Path to base BDDL (relative to project root or absolute) |
| `base_prompt` | Task instruction text |
| `action_scale`, `num_demos`, `noise_std` | Recording options |

### Video rendering

| Key | Description |
|-----|-------------|
| `render_videos` | If `true` (default), render videos from HDF5 files after recording (via `playback.py`). If `false`, skip video rendering in the pipeline. |

### Perturbations

- **`perturbations.types`** — List: `bddl_spatial`, `language` (or both).
- **`perturbations.bddl_spatial`**:
  - **`max_init_range_m`** — Extent (m) used for every init region when init is “exact,” so the env is nearly deterministic and MuJoCo gets positive geom size. Default `0.001` (1 mm).
- **`max_move_m`** — Default max distance (m) for “move” perturbations (can be overridden per spec). Also used as the **default translation bounds** for [random-design mode](#random-design-mode-metric-vs-translation-heatmap): if bounds are not set, x and z are sampled in `(-max_move_m, max_move_m)`.
- **`perturbation_specs`** — List of entries; each entry produces one run (one BDDL + one record config).

### Perturbation spec entries (YAML)

Each entry in `perturbation_specs` has:

- **`type`** — One of: `move`, `reorient`, `color`, `replace`, `distractor`, `control`.
- **`objects`** — Object names (e.g. `["akita_black_bowl_1", "wine_bottle_1"]`). Omit for `distractor` and `control`.
- **`count`** — For `distractor` only (number of distractors).
- **`max_move_m`** — Optional; overrides the default for `move` only.

Examples:

```yaml
perturbation_specs:
  - type: move
    objects: ["akita_black_bowl_1", "wine_bottle_1"]
    max_move_m: 0.05
  - type: reorient
    objects: ["wine_bottle_1"]
  - type: color
    objects: ["akita_black_bowl_1"]
  - type: distractor
    count: 1
  - type: control
    objects: []
```

### SLURM

- **`slurm.max_concurrent_jobs`** — How many record jobs run at once.
- **`slurm.job_params`** — `partition`, `account`, `time`, `nodes`, `gpus`, `gpu_type`, `mem`, `constraint`, `blacklisted_nodes`, etc.
- **`slurm.conda_env`** — Conda env path or name used in job scripts.
- **`slurm.module_load`** — List of modules to load (e.g. `anaconda3/2023.03`, `cuda/12.6.1`).
- **`slurm.poll_interval`** — Seconds between job status checks.

### Evaluation

- **`evaluation.enabled`** — Whether to run trajectory analysis after jobs complete.
- **`evaluation.metric_weights`**, **`evaluation.trajectory_weights`**, **`evaluation.output_formats`** — Control metrics and outputs (e.g. JSON, hdf5).

---

## Perturbation types and spec dicts

### Types

- **move** — Object init region is moved. New center is chosen in code (see below).
- **reorient** — Object yaw is perturbed by a random angle.
- **color** — Object color is changed to a random supported color.
- **replace** — Object is replaced with another type in the workspace.
- **distractor** — Extra distractor object(s) added (count from `count`).
- **control** — No BDDL change; same as base. Used for baseline comparison.

### Spec dicts (in code, not in YAML)

Perturbation **spec dicts** are generated in code and passed into the perturbation logic. They are **not** defined in `main.yaml`.

- **Shape (for move):**  
  `{"move": {"object_name": [center_x, center_z], ...}}`  
  Coordinates are in the same table-plane frame as BDDL `:ranges` (x, z).

- **Who generates them:**  
  For each **move** entry in `perturbation_specs`, the launcher calls `generate_move_spec_dict(base_bddl_text, objects, max_move_m)`. That function:
  - Reads unperturbed center (x, z) for each object from the BDDL.
  - Samples a new center per object: `center + random.uniform(-max_move_m, max_move_m)` in x and z.
  - Returns one spec dict per perturbation file.

- **Control and other types:**  
  No spec dict is used (control leaves BDDL unchanged; reorient/color/replace/distractor use their existing random behavior).

So: **one move entry → one call to `generate_move_spec_dict` → one spec dict → one BDDL file** with deterministic (but randomly chosen) move centers. To use custom positions instead of random, you can later replace or extend the generator (e.g. in `libero/libero/utils/generate_perturbation_bddl.py`) without changing the config.

---

## Local runs (no SLURM)

To try one perturbation locally:

**1. Generate only**

```bash
python scripts/launcher.py --config configs/main.yaml --generate-only --run-dir ./local_run
```

This creates `./local_run/` with:

- `bddl_files/` — `unperturbed.bddl`, `perturbed_0.bddl`, ...
- `configs/` — `unperturbed.yaml`, `perturbed_0.yaml`, ...
- `perturbation_manifest.json`

**2. (Optional) Edit a config**  
Adjust `./local_run/configs/<id>.yaml` if needed (e.g. `device`, `cache_dir`, `num_demos`).

**3. Record one run**

```bash
python scripts/record.py --config ./local_run/configs/unperturbed.yaml
# or
python scripts/record.py --config ./local_run/configs/perturbed_0.yaml
```

Output: `./local_run/results/<id>.hdf5`.

**4. (Optional) Render video**

```bash
python scripts/playback.py --config ./local_run/configs/perturbed_0.yaml
```

---

## Directory structure

After a run (full or generate-only), the run directory looks like:

```
run_dir/
├── main_config.yaml           # Copy of main config
├── perturbation_manifest.json  # List of perturbations and descriptions
├── random_design_points.json   # (Random-design only) design points {id, x, z} (deltas in m)
├── heatmap_metric_vs_translation.png  # (Random-design only) metric vs absolute (x,z) position
├── job_summary.json            # (After jobs) completed/failed counts
├── analysis_results.json       # (After evaluation) metrics
├── bddl_files/
│   ├── unperturbed.bddl
│   ├── perturbed_0.bddl       # or rd_0.bddl, rd_1.bddl, ... in random-design
│   └── ...
├── configs/
│   ├── unperturbed.yaml
│   ├── perturbed_0.yaml
│   └── ...
├── results/
│   ├── unperturbed.hdf5
│   ├── perturbed_0.hdf5
│   ├── trajectories.json
│   └── videos/
├── logs/                       # SLURM logs (when using launcher without --generate-only)
└── jobs/                       # SLURM scripts
```

---

## Troubleshooting

### Record script exits with "Aborted"

Usually the process is killed (e.g. OOM). Run recording **inside a GPU job** with enough memory, not on a login node:

```bash
srun --gres=gpu:1 --mem=32G --time=2:00:00 --pty bash
# then: conda activate your_env; python scripts/record.py --config ...
```

Set `num_demos: 1` in the config when testing. Ensure `device` and `cache_dir` in the generated config match your machine.

### MuJoCo error: "size 0 must be positive in geom"

Init regions had zero extent. The pipeline applies `fix_init_ranges` with `max_init_range_m` so every region has a small positive extent. Regenerate BDDLs with the launcher (e.g. `--generate-only --run-dir ./local_run`) so the fixed BDDLs are used.

### Python codec / init_fs_encoding error on PACE-ICE

Often due to a broken or moved conda env. Set locale and use a clean env:

```bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

Recreate the env on the cluster (do not copy from another machine). Use the project’s `environment.yml` or `requirements-env-pip.txt` and `pip install -e .` for the libero package.

### Jobs not starting

Check SLURM `account`, `partition`, and `conda_env` in `configs/main.yaml`. Inspect `logs/` and `squeue`.

### Perturbation generation fails

Confirm `base_bddl_file` exists and object names in `perturbation_specs` match the BDDL (e.g. `akita_black_bowl_1`). Check console for validation errors.

---

## Customization

### Adding or changing perturbation logic

- **Spatial:** `libero/libero/utils/generate_perturbation_bddl.py` — `apply_perturbations`, `move_object`, `reorient_object`, `change_color`, `replace_object`, `add_distractor`, and `generate_move_spec_dict`.
- **Language:** `explainability/perturbations/language/generate_perturbations.py`.
- **Launcher:** `scripts/launcher.py` — config-driven generation; random-design in `run_random_design()` and `scripts/pipeline/random_design.py` (`generate_random_design_perturbations`, `run_heatmap`).

### Custom move positions

Replace or extend `generate_move_spec_dict` in `generate_perturbation_bddl.py` to return `{"move": {obj: [x, z], ...}}` from your own source (e.g. file, grid). The launcher and `apply_perturbations(..., perturbation_spec_dict=...)` already consume that shape.

### Evaluation metrics

Adjust `scripts/launcher.py` `_run_analysis` and the evaluation config (e.g. `evaluation.metric_weights`, `evaluation.trajectory_weights`).

---

## Citation and license

**LIBERO** (benchmark and env):

```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}
```

- **Codebase:** [MIT License](LICENSE)
- **LIBERO datasets:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)
