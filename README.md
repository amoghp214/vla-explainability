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

The pipeline has a single workflow: **random-design + temporal**.

1. **Generate** — Samples (x, z) from bounds, writes unperturbed + control + rd_0..rd_{n-1} BDDL and record YAMLs. Init regions use a small box (`init_range_m`). Each rd_i.yaml includes temporal_perturbations so the perturbation is applied at configurable steps during recording.
2. **Dispatch** — Submits SLURM jobs to record each config with OpenVLA.
3. **Post-process** — Optionally renders videos (`render_videos` in config) and runs evaluation (trajectory comparison, VLA metric).
4. **Heatmap** — Fits a BoTorch GP and saves a heatmap of metric vs (x, z).

Perturbation type: **move** (translate one object by delta; heatmap vs absolute position) or **distract** (add distractor at (x, z); heatmap vs distractor position). Config must include a `random_design` section (see `configs/vk_main.yaml`).

---

## Quick Start

### 1. Configure

Edit `configs/vk_main.yaml` (or `configs/main.yaml`):

- **Base task:** `base_bddl_file`, `base_prompt`, `task_suite_name`
- **random_design:** `type` (move | distract), `n_design`, `bounds_x`, `bounds_z`, `object_names` (for move), `seed`, `uniform`
- **Temporal:** `perturbation_start_step`, `perturbation_stop_step`
- **SLURM:** `slurm.job_params`, `slurm.conda_env`, `slurm.module_load`

### 2. Run pipeline

```bash
python scripts/launcher.py --config configs/vk_main.yaml
```

This creates a run directory, generates BDDL and record YAMLs (unperturbed, control, rd_0..rd_{n-1}), submits SLURM jobs, then runs evaluation and heatmap when jobs finish. Set `render_videos: false` to skip video rendering.

### 3. Generate only (no jobs)

```bash
python scripts/launcher.py --config configs/vk_main.yaml --generate-only --run-dir ./local_run
```

Then record configs manually (see [Local runs](#local-runs-no-slurm)).

---

## Random-design mode (metric vs translation heatmap)

The pipeline supports two perturbation types (`random_design.type` or `--type`):

- **move** — (x, z) are **deltas** from the object's original BDDL position. Samples in bounds, heatmap of metric vs absolute (x, z).
- **distract** — (x, z) are the **position** where a distractor is added. Heatmap of metric vs distractor position.

Config must include a `random_design` section (see `configs/vk_main.yaml`).

**Generator vs temporal:** A single component (`scripts/pipeline/random_design.py`) generates all BDDL and record YAML files from random sampling. The temporal engine (`libero.utils.temporal_perturbations`, used by `record.py`) does not generate perturbations—it only reads the generated config and applies/reverts each perturbation at the configured `perturbation_start_step` and `perturbation_stop_step` during the rollout.

### Usage

```bash
# Default: move mode, n_design and bounds from config
python scripts/launcher.py --config configs/vk_main.yaml

# Override design count and bounds
python scripts/launcher.py --config configs/vk_main.yaml --n-design 20 --bounds -0.05,0.05

# Distract mode
python scripts/launcher.py --config configs/vk_main.yaml --type distract --n-design 20 --bounds -0.2,0.2

# Generate only
python scripts/launcher.py --config configs/vk_main.yaml --generate-only --run-dir ./rd_run
```

### Options

| Option | Description |
|--------|-------------|
| `--type` | `move` (default) or `distract`. Overrides config `random_design.type`. |
| `--n-design` | Number of random design points. Overrides config. |
| `--seed` | Random seed. Overrides config. |
| `--bounds` | Comma-separated `low,high` in meters for both x and z. Overrides config `random_design.bounds_x/z`. |
| `--objects` | Object name(s) for type=move only (exactly one). Overrides config. |

### Config

Set `random_design` in your config (e.g. `configs/vk_main.yaml` or `configs/main.yaml`):

```yaml
random_design:
  type: move   # "move" | "distract"
  n_design: 20
  seed: 1
  bounds_x: [-0.05, 0.05]   # For move: delta range (m). For distract: position range (m).
  bounds_z: [-0.05, 0.05]
  object_names: ["akita_black_bowl_1"]   # Required for type=move only (exactly one object)
  uniform: true
  # Distract mode only:
  distractor_count: 1
  distractor_object_type: "akita_black_bowl"   # LIBERO type (e.g. plate, wine_bottle). Omit for random.
  # distractor_object_types: ["akita_black_bowl", "plate"]   # Or list: random choice per design point
```

- **type** — `move` (translate one object) or `distract` (add distractor at position). Default `move`.
- **object_names** — Required only for `type: move` (exactly one object). Ignored for `type: distract`.
- **distractor_count** — For `type: distract`: number of distractors per design point (default 1).
- **distractor_object_type** — For `type: distract`: single LIBERO object type for all distractors (e.g. `akita_black_bowl`, `plate`, `wine_bottle`). Omit to use random type.
- **distractor_object_types** — For `type: distract`: list of types; one is chosen at random per design point. Ignored if `distractor_object_type` is set.

If `bounds_x` / `bounds_z` are omitted in config and `--bounds` is not passed, they default to `(-0.05, 0.05)`. Perturbation amount (move) or coordinates (distractor) come only from these bounds.

### Outputs

In the run directory you get:

- **`random_design_points.json`** — List of `{id, x, z}` for each design point (e.g. `rd_0`, `rd_1`, ...). For **move**: x, z are **deltas** (m) from the object’s original BDDL center. For **distract**: x, z are the **absolute position** (m) of the distractor.
- **`heatmap_metric_vs_translation.png`** — Heatmap of the GP-predicted VLA metric. For **move**: over (x, z) absolute position (original + delta); axes “x position (m)” and “z position (m)”. For **distract**: over distractor (x, z) position; same axis labels.

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
  - **`init_range_m`** — Side length (m) of the init placement box for each object region; small value (e.g. 0.001) keeps the env nearly deterministic and gives MuJoCo a positive geom size. Default `0.001` (1 mm).
- **`max_move_m`** — Used only for non-random-design perturbation_specs. For random-design, bounds come from random_design.bounds_x and bounds_z (or --bounds); default (-0.05, 0.05).
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
├── random_design_points.json   # (Random-design only) design points {id, x, z}: deltas (move) or position (distract)
├── heatmap_metric_vs_translation.png  # (Random-design only) metric vs (x,z) position
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

Init regions had zero extent. The pipeline applies `fix_init_ranges` with `init_range_m` so every region has a small positive extent. Regenerate BDDLs with the launcher (e.g. `--generate-only --run-dir ./local_run`) so the fixed BDDLs are used.

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

Confirm `base_bddl_file` exists and `random_design.object_names` (for type=move) or `hidden_objects` (for type=distract) match the BDDL. Check console for validation errors.

---

## Customization

### Adding or changing perturbation logic

- **Spatial:** `libero/libero/utils/generate_perturbation_bddl.py` — `apply_perturbations`, `move_object`, `reorient_object`, `change_color`, `replace_object`, `add_distractor`, and `generate_move_spec_dict`.
- **Language:** `explainability/perturbations/language/generate_perturbations.py`.
- **Launcher:** `scripts/launcher.py` — single workflow: random-design + temporal (`run_random_design()`, `scripts/pipeline/random_design.py`).

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
